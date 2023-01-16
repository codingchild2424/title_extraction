
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

from define_argparser import define_argparser

from dataloaders.dataloader import get_datasets
from dataloaders.dataloader import TextAbstractSummarizationCollator

'''
Thanks to 이야기연구소 주식회사 팀
https://dacon.io/competitions/official/235829/codeshare/4047
'''

def main(config):
    # device는 cpu, cuda 선택하도록
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        config.pretrained_model_name
    )

    tr_ds = get_datasets(
        tokenizer=tokenizer,
        fpath=Path(config.train_data_path)
    )

    vl_ds = get_datasets(
        tokenizer=tokenizer,
        fpath=Path(config.valid_data_path)
    )

    len(tr_ds)
    len(vl_ds)

    model = transformers.BartForConditionalGeneration.from_pretrained(
        config.pretrained_model_name
        )

    ## Path arguments.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.ckpt, nowtime)
    logging_dir = Path(config.logs, nowtime, "run")

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
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

    config = define_argparser()

    main(config)