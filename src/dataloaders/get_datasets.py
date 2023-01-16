
import random
import torch
from torch.utils.data import Dataset
from .data_utils import TextGenerationDataset
from torch.utils.data import DataLoader

import os

def read_text(folder_path):

    # 각 파일을 불러오기
    file_list = os.listdir(folder_path)

    sep_token = '[SEP]'
    labels, texts = [], []
    toks = []
    '''
    tok = text + [SEP] + label
    '''

    for file_name in file_list:

        fn = os.path.join(folder_path, file_name)

        with open(fn, 'r') as f:

            lines = f.readlines()

            for line in lines:

                if line.strip() != '':

                    label = line.split('\t')[0]
                    text = line.split('\t')[1]

                    #label, text = line.strip().split('\t')
                    labels += [label]
                    texts += [text]

                    tok = [text + sep_token + label]

                    toks += tok

    return labels, texts, toks

def get_datasets(fn, valid_ratio=.2, test_ratio=.2):

    labels, texts, toks = read_text(fn)

    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)

    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]

    idx1 = int(len(shuffled) * (1 - (valid_ratio + test_ratio))) # 0.6
    idx2 = int(len(shuffled) * (1 - (test_ratio))) # 0.8

    train_dataset = TextGenerationDataset(texts[:idx1], labels[:idx1])
    valid_dataset = TextGenerationDataset(texts[idx1:idx2], labels[idx1:idx2])
    test_dataset = TextGenerationDataset(texts[idx2:], labels[idx2:])

    return train_dataset, valid_dataset, test_dataset
