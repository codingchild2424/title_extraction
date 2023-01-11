
import random
import torch
from torch.utils.data import Dataset
from .data_utils import TextGenerationDataset
from torch.utils.data import DataLoader

def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []

        '''
        tok = text + [SEP] + label
        '''
        sep_token = '[SEP]'
        toks = []

        for line in lines:

            if line.strip() != '':

                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

                tok = [text + sep_token + label]

                toks += tok

    return labels, texts, toks


def get_datasets(fn, valid_ratio=.2, test_ratio=.2):

    _, _, toks = read_text(fn)

    shuffled = list(toks)
    random.shuffle(shuffled)

    idx1 = int(len(shuffled) * (1 - (valid_ratio + test_ratio))) # 0.6
    idx2 = int(len(shuffled) * (1 - (test_ratio))) # 0.8

    train_dataset = TextGenerationDataset(shuffled[:idx1])
    valid_dataset = TextGenerationDataset(shuffled[idx1:idx2])
    test_dataset = TextGenerationDataset(shuffled[idx2:])

    return train_dataset, valid_dataset, test_dataset
