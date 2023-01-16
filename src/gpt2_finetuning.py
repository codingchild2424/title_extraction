
import torch

import transformers

import datetime

from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List, Dict
import os

from gpt2_define_argparser import define_argparser

from dataloaders.gpt2_dataloader import get_datasets
from dataloaders.gpt2_dataloader import TextAbstractSummarizationCollator

from bart_trainer import Trainer

'''
Thanks to 이야기연구소 주식회사 팀
https://dacon.io/competitions/official/235829/codeshare/4047
'''

def main(config):
    # device는 cpu, cuda 선택하도록
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        config.pretrained_model_name # kogpt2
    )

    tr_ds = get_datasets(
        tokenizer=tokenizer,
        fpath=Path(config.train_data_path)
    )

    vl_ds = get_datasets(
        tokenizer=tokenizer,
        fpath=Path(config.valid_data_path)
    )

    print("train_data: %s" % str(len(tr_ds)))
    print("valid_data: %s" % str(len(vl_ds)))

    model = transformers.GPT2LMHeadModel.from_pretrained(
        config.pretrained_model_name # kogpt2
        )

    trainer = Trainer(config)

    trainer._train(
        model=model,
        data_collator=TextAbstractSummarizationCollator(
            tokenizer=tokenizer,
            config=config
        ),
        train_dataset=tr_ds,
        valid_dataset=vl_ds,
        config=config,
        tokenizer=tokenizer
    )


if __name__ == "__main__":

    config = define_argparser()

    main(config)