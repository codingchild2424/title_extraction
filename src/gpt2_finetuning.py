
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

from gpt2_trainer import Trainer

from torch.optim import Adam

import torch.nn as nn

from torch.utils.data import DataLoader


'''
Thanks to 이야기연구소 주식회사 팀
https://dacon.io/competitions/official/235829/codeshare/4047
'''

def main(config):
    # device는 cpu, cuda 선택하도록
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        config.pretrained_model_name, # kogpt2
        # kogpt는 사전에 토큰을 지정해주지 않으면, None 값으로 반영되어있음
        # 반드시 지정해주어야 함
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>'
    )
    # token len 51200

    #special_tokens_dict = {'additional_special_tokens': ['[C1]','[C2]','[C3]','[C4]']}
    #num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


    tr_ds = get_datasets(
        tokenizer=tokenizer,
        config=config,
        fpath=Path(config.train_data_path)
    )

    vl_ds = get_datasets(
        tokenizer=tokenizer,
        config=config,
        fpath=Path(config.valid_data_path)
    )

    print("train_data: %s" % str(len(tr_ds)))
    print("valid_data: %s" % str(len(vl_ds)))

    # 기존 모델 코드
    model = transformers.GPT2LMHeadModel.from_pretrained(
        config.pretrained_model_name # kogpt2
        )
    model.resize_token_embeddings( len(tokenizer) ) #51200 / 다른 코드에서는 50000을 쓴 경우도 있음

    model = model.to(device)

    #print("model config", model.config)

    print("tr_ds", tr_ds)

    # loader

    train_loader = DataLoader(
        tr_ds,
        batch_size=config.batch_size_per_device,
        shuffle=True,
        collate_fn=TextAbstractSummarizationCollator(
            tokenizer=tokenizer,
            config=config
        )
    )

    valid_loader = DataLoader(
        vl_ds,
        batch_size=config.batch_size_per_device,
        shuffle=False,
        collate_fn=TextAbstractSummarizationCollator(
            tokenizer=tokenizer,
            config=config
        )
    )

    optimizer = Adam(model.parameters(), config.lr)
    crit = nn.CrossEntropyLoss

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        n_epochs=config.n_epochs,
        device=device,
        crit=crit,
        max_seq_len=config.inp_max_len,
        config=config
        )

    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config
    )


if __name__ == "__main__":

    config = define_argparser()

    main(config)