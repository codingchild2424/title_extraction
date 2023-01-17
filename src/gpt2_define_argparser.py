    
import argparse
import torch

def define_argparser():
    p = argparse.ArgumentParser()

    #p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=3 if torch.cuda.is_available() else -1)

    # data_path
    p.add_argument('--train_data_path', type=str, default='../datasets/integrated_pre_datasets/train_data.tsv')
    p.add_argument('--valid_data_path', type=str, default='../datasets/integrated_pre_datasets/valid_data.tsv')
    p.add_argument('--test_data_path', type=str, default='../datasets/integrated_pre_datasets/test_data.tsv')
    
    p.add_argument('--pretrained_model_name', type=str, default='skt/kogpt2-base-v2')

    p.add_argument('--ckpt', type=str, default="ckpt")
    p.add_argument('--logs' ,type=str, default="logs")

    p.add_argument('--batch_size_per_device', type=int, default=2) # 2가 최선
    p.add_argument('--gradient_accumulation_steps', type=int, default=32) # 32로 증가
    p.add_argument('--lr', type=float, default=5e-5)

    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--inp_max_len', type=int, default=1023) # 원래는 1024 / 각 문장에 토큰이 1개씩 붙어서, 토큰을 포함해서 chunk하기 위해 1023으로 변경함
    p.add_argument('--tar_max_len', type=int, default=256)
    p.add_argument('--model_fpath', type=str, default="model_records/kogpt2-model.pth")

    p.add_argument('--beam_size', type=int, default=5)

    # 추가
    p.add_argument('--prompt4summary', type=str, default="1줄요약")

    config = p.parse_args()

    return config