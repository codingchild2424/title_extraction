import transformers

import datetime
import easydict
import itertools
import json
import matplotlib
import pathlib
import pprint
import re

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
        self.df.loc[:, "text_tok"] = self.df.loc[:, "text"].progress_apply(lambda x: tokenizer.encode(x))
        self.df.loc[:, "text_tok_len"] = self.df.loc[:, "text_tok"].apply(lambda x: len(x))

        if self.mode == "train":
            tqdm.pandas(desc="Tokenizing target summaries")
            self.df.loc[:, "summary_tok"] = self.df.loc[:, "summary"].progress_apply(lambda x: tokenizer.encode(x))
            self.df.loc[:, "summary_tok_len"] = self.df.loc[:, "summary_tok"].apply(lambda x: len(x))

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
        if self.mode == "train":
            return_value["summary"] = instance["summary_tok"]
        
        return return_value


def get_datasets(tokenizer, fpath: pathlib.PosixPath, mode: str = "train"):
    return TextAbstractSummarizationDataset(tokenizer, fpath, mode=mode)