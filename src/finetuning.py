
import argparse
import torch

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

from dataloaders.dataloader import get_datasets
from dataloaders.data_utils import TextAbstractSummarizationCollator

config = easydict.EasyDict({
    "raw_train_path": "./data/Training",
    "raw_valid_path": "./data/Validation",
    "clean": False,
    "data": "data", ## data path
    "seed": 42,
    "sample_submission_path": "sample_submission.tsv",
    "answer_path": "answer.tsv",
    "prediction_path": "prediction.tsv",
    "pretrained_model_name": "gogamza/kobart-base-v1",
    "train": "data/train.tsv",
    "valid": "data/valid.tsv",
    "test": "data/test.tsv",
    ## Training arguments.
    "ckpt": "ckpt", ## path
    "logs": "logs", ## path
    "per_replica_batch_size": 8,
    "gradient_accumulation_steps": 16,
    "lr": 5e-5,
    "weight_decay": 1e-2,
    "warmup_ratio": 0.2,
    "n_epochs": 10,
    "inp_max_len": 1024,
    "tar_max_len": 256,
    "model_fpath": "/workspace/home/uglee/Projects/title_extraction/src/model_records/kobart-model.pth",
    ## Inference.
    "gpu_id": 3,
    "beam_size": 5,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 3,
    "var_len": False,
})

def main(config):
    # device는 cpu, cuda 선택하도록
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        'gogamza/kobart-base-v1'
    )

    tr_ds = get_datasets(
        tokenizer=tokenizer,
        fpath=Path("/workspace/home/uglee/Projects/title_extraction/datasets/integrated_pre_datasets/train_data.tsv")
    )

    vl_ds = get_datasets(
        tokenizer=tokenizer,
        fpath=Path("/workspace/home/uglee/Projects/title_extraction/datasets/integrated_pre_datasets/valid_data.tsv")
    )

    len(tr_ds)
    len(vl_ds)

    model = transformers.BartForConditionalGeneration.from_pretrained(
        'gogamza/kobart-base-v1'
        )

    ## Path arguments.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.ckpt, nowtime)
    logging_dir = Path(config.logs, nowtime, "run")

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.per_replica_batch_size,
        per_device_eval_batch_size=config.per_replica_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.n_epochs,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        # save_steps=1000,
        fp16=True,
        dataloader_num_workers=0, # 원래 4
        disable_tqdm=False,
        load_best_model_at_end=True,
        ## As below, only Seq2SeqTrainingArguments' arguments.
        # sortish_sampler=True,
        # predict_with_generate=True,
        # generation_max_length=config.tar_max_len,   ## 256
        # generation_num_beams=config.beam_size,      ## 5
    )

    print("KoBART Fine Tuning Start")

    ## Define trainer.
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=TextAbstractSummarizationCollator(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            inp_max_len=config.inp_max_len,
            tar_max_len=config.tar_max_len,
        ),
        train_dataset=tr_ds,
        eval_dataset=vl_ds,
    )

    ## Just train.
    trainer.train()

    print("훈련 완료")

    # Save the best model
    torch.save({
        "model": trainer.model.state_dict(),
        "config": config,
        "tokenizer": tokenizer
    }, Path(config.model_fpath))

if __name__ == "__main__":

    #config = define_argparser()

    main(config)