import transformers

import pathlib

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from operator import itemgetter
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List, Dict
import os
import torch

def read_tsv(fpath: pathlib.PosixPath) -> pd.DataFrame:
    return pd.read_csv(
        fpath, 
        index_col=False,
        names=['summary', 'text'],
        sep="\t",
        encoding="utf-8")

class TextAbstractSummarizationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        tokenizer,
        config,
        fpath: pathlib.PosixPath,
        mode: str = "train",
    ):
        super(TextAbstractSummarizationDataset, self).__init__()

        self.df = read_tsv(fpath)
        # self.tok = tokenizer -> don't keep
        self.df.loc[:, "id"] = [i for i in range(self.df.shape[0])]
        
        ## Mode.
        assert mode in ["train", "test"]
        self.mode = mode

        ## Apply tokenize first to speed up in training phase and make code more simply.
        tqdm.pandas(desc="Tokenizing input texts")

        # 요약을 위한 키워드
        prompt4summary = config.prompt4summary

        # 여기서 텍스트에 사전에 prompt를 입력하기
        self.df.loc[:, "text"] = self.df.loc[:, "text"] + prompt4summary
        self.df.loc[:, "text_tok"] = self.df.loc[:, "text"].progress_apply(lambda x: tokenizer.encode(x))
        self.df.loc[:, "text_tok_len"] = self.df.loc[:, "text_tok"].apply(lambda x: len(x))

        if self.mode == "train":
            tqdm.pandas(desc="Tokenizing target summaries")

            # 여기서 텍스트에 사전에 prompt를 입력하기
            self.df.loc[:, "text"] = self.df.loc[:, "text"] + prompt4summary
            self.df.loc[:, "text_tok"] = self.df.loc[:, "text"].progress_apply(lambda x: tokenizer.encode(x))
            self.df.loc[:, "text_tok_len"] = self.df.loc[:, "text_tok"].apply(lambda x: len(x))

            self.df.loc[:, "summary_tok"] = self.df.loc[:, "summary"].progress_apply(lambda x: tokenizer.encode(x))
            self.df.loc[:, "summary_tok_len"] = self.df.loc[:, "summary_tok"].apply(lambda x: len(x))

            # text와 summary 합쳐서 다시 text_tok으로 반환
            self.df.loc[:, "text_tok"] = self.df.loc[:, "text_tok"] + self.df.loc[:, "summary_tok"]
        else:
            # 여기서 텍스트에 사전에 prompt를 입력하기
            self.df.loc[:, "text"] = self.df.loc[:, "text"] + prompt4summary
            self.df.loc[:, "text_tok"] = self.df.loc[:, "text"].progress_apply(lambda x: tokenizer.encode(x))
            # 합치지 않고 반환
            self.df.loc[:, "text_tok_len"] = self.df.loc[:, "text_tok"].apply(lambda x: len(x))

        ## Sort by tokenized length with tqdm progress bar.
        ## 
        ## By sorting sequentially, starting with the longest sentence, 
        ## we can determine the maximum VRAM size the model is using for
        ## training. That is, if OOM does not occur for the maximum VRAM
        ## size at the beginning of training, it is guaranteed that OOM
        ## does not occur during training.
        self.df.sort_values(by=["text_tok_len"], axis=0, ascending=False, inplace=True)

    
    def __len__(self) -> int:
        return self.df.shape[0]


    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        instance = self.df.iloc[idx]

        return_value = {
            "id": instance["id"], ## for sorting in inference mode
            "text": instance["text_tok"],
            "length": len(instance["text_tok"]),
        }

        return return_value

class TextAbstractSummarizationCollator():

    def __init__(
        self,
        # bos_token_id: int,
        # eos_token_id: int,
        # pad_token_id: int,
        # inp_max_len: int = 1024,
        # tar_max_len: int = 256,
        # 
        tokenizer,
        config,
        ignore_index: int = -100,
        mode: str = "train",
    ):
        super(TextAbstractSummarizationCollator, self).__init__()

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.inp_max_len = config.inp_max_len
        self.tar_max_len = config.tar_max_len
        self.ignore_index = ignore_index

        ## Mode.
        assert mode in ["train", "test"]
        self.mode = mode


    def _pad(self, sentences: List[List[int]], token_id: int) -> np.ndarray:
        ## We will pad as max length per batch, not "inp_max_len(=1024, etc)".
        max_length_per_batch = max([len(i) for i in sentences])

        ## Stack as dimension 0 (batch dimension).
        ## "token_id" can be "tokenizer.pad_token_id(=3)" or "ignore_index(=-100)"
        return np.stack([i + [token_id] * (max_length_per_batch - len(i)) for i in sentences], axis=0)


    def _train_collator(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        ## Unpack.

        # For Autoregressive

        ## If input max length > 1024, you can see below error:
        ##   1) Assertion `srcIndex < srcSelectDimSize` failed
        ##   2) Device-side assert triggered
        tokenized_texts     = [s["text"][:self.inp_max_len] for s in samples]

        ## Inputs for encoder.
        # [BOS] + TEXT
        #input_ids = [ + i for i in tokenized_texts]
        input_ids = [[self.bos_token_id] + i for i in tokenized_texts]
        input_ids = self._pad(input_ids, token_id=self.pad_token_id)#.astype(np.int64)  ## numpy format
        attention_mask = (input_ids != self.pad_token_id).astype(float) ## numpy format

        ## Answer.
        # TEXT + [EOS]
        labels = [i + [self.eos_token_id] for i in tokenized_texts]
        labels = self._pad(labels, token_id=self.ignore_index)#.astype(np.int64) ## why != "padding_id" ???

        print("input_ids", input_ids)
        print("labels", labels)
        
        ## Pack as pre-defined arguments. See:
        ##   https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
        return {
            "input_ids":                torch.from_numpy(input_ids),
            "attention_mask":           torch.from_numpy(attention_mask),
            "labels":                   torch.from_numpy(labels),
        }

    def _test_collator(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        ## Unpack.
        ids              = [s["id"]                      for s in samples]
        tokenized_texts  = [s["text"][:self.inp_max_len] for s in samples]   ## no <bos> token included

        ## Inputs for encoder.
        #input_ids = [[self.bos_token_id] + i for i in tokenized_texts]
        input_ids = self._pad([self.bos_token_id] + tokenized_texts, token_id=self.pad_token_id)  ## numpy format
        attention_mask = (input_ids != self.pad_token_id).astype(float)     ## numpy format

        ## Pack as pre-defined arguments:
        ## See: https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
        return {
            "input_ids":        torch.from_numpy(input_ids),
            "attention_mask":   torch.from_numpy(attention_mask),
            "id":               ids,
        }


    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        return self._train_collator(samples) if self.mode == "train" else self._test_collator(samples)


def get_datasets(tokenizer, config, fpath: pathlib.PosixPath, mode: str = "train"):
    return TextAbstractSummarizationDataset(tokenizer, config, fpath, mode=mode)