
import argparse
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from datasets import load_metric
from transformers import Trainer, TrainingArguments
from define_argparser import define_argparser
from dataloaders.data_utils import TextClassificationCollator
from torch.utils.data import DataLoader

from dataloaders.get_datasets import get_datasets

from trainer import KoBARTTrainer

from rouge import Rouge

from dotenv import load_dotenv
import os

def main(config):
    # device는 cpu, cuda 선택하도록
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # 모델 경로
    model_path = './model_records/'
    tokenizer_path = './tokenizer_records/'

    # model
    model = BartForConditionalGeneration.from_pretrained(
        model_path
    ).to(device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path
    )
    
    # Inference
    raw_infer_data = "일본어를 배우는 교회에 나는 집에서 일본어에 간다."
    raw_input_ids = tokenizer.encode(raw_infer_data)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id] #[tokenizer.eos_token_id]

    summary_ids = model.generate(
        torch.tensor([input_ids]).to(device),
        num_beams=4,
        max_length=20,
        eos_token_id=1
    ).to(device)

    generated = tokenizer.decode(
        summary_ids.squeeze().tolist(),
        skip_special_tokens=True
    )

    print("generated", generated)

if __name__ == "__main__":

    config = define_argparser()

    main(config)