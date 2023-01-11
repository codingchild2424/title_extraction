
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

    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    #num_added_toks = tokenizer.add_tokens(['[EOT]'], special_tokens=True)

    # Get datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(
        config.train_fn, 
        config.valid_ratio, 
        config.test_ratio
        )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=total_batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(
            tokenizer, 
            config.max_length
            )
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=total_batch_size,
        shuffle=False,
        collate_fn=TextClassificationCollator(
            tokenizer, 
            config.max_length
            )
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=total_batch_size,
        shuffle=False,
        collate_fn=TextClassificationCollator(
            tokenizer, 
            config.max_length
            )
    )

    model = BartForConditionalGeneration.from_pretrained(
        'gogamza/kobart-base-v1'
    ).to(device)

    learning_rate = 3e-5
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("KoBART Fine Tuning Start")

    # Trainer

    trainer = KoBARTTrainer(
        model=model,
        optimizer=optimizer,
        n_epochs=config.n_epochs,
        device=device,
        crit=criterion,
        max_seq_len=config.max_length,
        warmup_ratio=config.warmup_ratio
    )

    trainer.train(
        train_loader, 
        valid_loader, 
        test_loader, 
        config
        )

    print("훈련 완료")

    # 모델 경로
    model_path = './model_records/' #+ config.model_fn + ".pth"
    tokenizer_path = './tokenizer_records/' #+ config.model_fn + "_tokenizer.pth"

    # local에 모델 저장
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    # Inference
    raw_infer_data = "요즘엔 소중하다라는 표현을 잘 쓰지 않는다.#@문장구분# '소중하다'보다는 '필요하다'가 더 많이 쓰인다.#@문장구분# 나도 그렇다.#@문장구분# 이것이 나에게 소중한 것이다.#@문장구분# 이것이 나에게 필요한 것이다.#@문장구분# 두 표현 다 의미는 다르지만 후자가 더 익숙하게 느껴진다.#@문장구분# 필요한 것과 소중한 것은 같은 것일까?#@문장구분# 소중한 것은 이미 내가 가지고 있는 것을 의미한다.#@문장구분# 반면에 필요한 것은 내가 현재 가지고 있지 않지만 내가 가지길 원하는 것이다.#@문장구분# 사전적 의미로 비교하면 소중하다는 '중요한 의미나 가치를 가진 상태에 있다'이고 필요하다는 '반드시 요구되는 바가 있다'이다.#@문장구분# 따라서 소중하다는 우리가 가지고 있는 것이라 해석할 수 있고 필요하다는 우리가 요구하는 것이라고 해석이 가능하다.#@문장구분# 그러면 현재 우리가 가지고 있는 것과 우리가 가지길 원하는 것 중에 우리가 무엇에 더 중요함을 느껴야 할까?#@문장구분# 가지지 않는 것에 중요함을 느끼는 것이 올바르지 않다고 본다.#@문장구분# 가지지 않은 것에 중요하다고 느끼는 것은 소유하지 않은 것에 가치를 부여하는 것이기 때문이다.#@문장구분# 물론 소유하지 않은 것에 가치를 부여해 동기 부여가 되기도 한다.#@문장구분# 예를 들어 나의 미래의 목표에 중요하다고 생각해서 동기부여가 되기도 한다.#@문장구분# 그러면 이걸 더 나아가서 우리가 가지고 있는 것보다 원하는 것, 즉, 요구하는 것에 더 중요하다고 생각하면 그것은 정말 어리석은 짓이다.#@문장구분# 현재 가지고 있는 것보다 요구하는 것에 가치를 부여하는 것이 '익숙함에 속아 소중함을 잊지말자'가 아닐까라는 생각이 든다.#@문장구분# 우리가 가지고있는것은 우리에게 익숙해져있기에 중요함을 느끼지 못한다.#@문장구분# 우리가 너무 당연하듯이 그것을 이용하거나 사용하고 있기 때문이다.#@문장구분# 그러한 것들이 만약에 우리에게 필요한 것이 된다면 즉, 가지지 못한 것이 된다면 그것이 아직까지 중요한것이 아닐까?#@문장구분# 아마도 생각이 달라질 것이다.#@문장구분# 나 또한 이러한 경험이 있다.#@문장구분# 시험 공부를 할때 나는 중요한것에 밑줄을 그으며 공부를 했었다.#@문장구분# 그 밑줄을 긋고 난 뒤에는 다시 볼때 그 밑줄 그은 부분만 보게된다.#@문장구분# 그렇게 몇번을 반복하면 완전히 그것이 공부가 되었다라는 느낌이든다.#@문장구분# 하지만 그것은 단지 그 글에 익숙해진 것일 뿐 이해했다고 보긴 힘들다.#@문장구분# 그렇지만 나는 공부가 되었다고 생각하고 시험을 보면 객관식 문제에서 오류가 많이났다.#@문장구분# 익숙함에 속아 내 소중한 객관식 문제에 점수를 잃은 것이다.#@문장구분# 이러한 경험이 꽤 있고난 뒤에는 반대로 생각해 보았다.#@문장구분# 소중함에 익숙함을 잃어보자 라고 생각해서 반대로 한 두세번을 밑줄 긋지 않고 그냥 읽는다.#@문장구분# 이해하기 위해 읽어야 한다.#@문장구분# 그렇게 하고 진짜 이해가 되었는지 확인하려면 선생님이 알려주시기 전에 먼저 연필로 내가 중요하고 생각한 것에 밑줄을 그어 본다.#@문장구분# 그러고 난 뒤에 선생님이 알려주신 것과 밑줄을 비교하여 공부하면 오류를 고칠 수 있었다.#@문장구분# 그래서 나는 항상 우리가 소중함에 대해 생각을 해야하며 혹시나 익숙함에 속았다면 반대로 소중함에 속아 익숙함을 잃어보는 것도 하나의 방법이라고 생각한다.#@문장구분# 결국 우리가 더 오랜시간 가지고 있는 것은 필요있는 것 보다는 현재 우리가 가지고 있는 것이다.#@문장구분# 그러니 우리는 소중함에 더 많은 가치를 부여해야 한다.#@문장구분#"

    raw_input_ids = tokenizer.encode(raw_infer_data)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

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