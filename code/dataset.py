import random
import json
import re

from pathlib import Path

import torch

from typing import Dict, List, Tuple

class TextDatasetConfig:
    '''
    Dataset の設定用

    data_path : データセットのパス
    vocab_path : 語彙ファイルのパス

    input_length : return する データの長さ
    max_seq_length : 読み込む1文の最大単語数
    is_return_str : id じゃない str のデータを返すかどうか
    '''
    def __init__(self, **kwargs):
        self.data_path = kwargs.pop("data_path", "")
        self.vocab_path = kwargs.pop("vocab_path", "")
        self.input_length = kwargs.pop("input_length", 4096)
        self.max_seq_length = kwargs.pop("max_seq_length", 4094)
        self.min_seq_length = kwargs.pop("min_seq_length", 1)

        self.shift_prob = kwargs.pop("shift_prob", 0.1)

        self.replace_prob = kwargs.pop("replace_prob", 0.1)
        self.unk_mask_prob = kwargs.pop("unk_mask_prob", 0.3)
        self.random_token_prob = kwargs.pop("random_token_prob", 0.7)

        self.is_return_str = kwargs.pop("is_return_str", False)

    @staticmethod
    def from_json(path:str):
        kwargs = json.load(open(path, 'r', encoding='utf-8', errors='ignore'))
        return TextDatasetConfig(**kwargs)

class DarkHistoryDataset(torch.utils.data.Dataset):
    def __init__(self, config:TextDatasetConfig):
        self.block_size     = config.input_length
        self.max_seq_length = config.max_seq_length
        self.min_seq_length = config.min_seq_length

        self.shift_prob = config.shift_prob

        self.replace_prob = config.replace_prob
        self.unk_mask_prob = config.unk_mask_prob
        self.random_token_prob = config.random_token_prob

        self.is_return_str  = config.is_return_str

        self.vocab = {}
        self.vocab_ids = []
        self.vocab_size = 0
        self.load_vocab(config.vocab_path)

        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.mask_id = self.vocab["[MASK]"]

        self.data_len = 0
        self.plain_docs = []
        self.docs = []
        if config.data_path is not None:
            self.load_data(config.data_path)

        self.padding = [self.pad_id] * self.block_size


    def load_data(self, data_path:str):
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().split("\n\n")

        for block in text:
            block = block.rstrip('\n')
            if not block:
                continue

            if len(block) > self.max_seq_length:
                continue

            if len(block) < self.min_seq_length:
                continue

            self.plain_docs.append(block)
            self.docs.append(self.sent_to_ids(block))

        self.data_len = len(self.docs)
        return self.data_len

    def load_vocab(self, vocab_path:str):
        with open(vocab_path, 'r', encoding='utf-8', errors='ignore') as f:
            vocab_data = f.read().split('\n')

        for i, word in enumerate(vocab_data):
            self.vocab[word] = i
            self.vocab_ids.append(word)

        self.vocab_size = len(self.vocab)
        return self.vocab_size

    def sent_to_ids(self, sent:str) -> List[int]:
        return [self.vocab.get(char, self.unk_id) for char in sent if not (char == '\n' or char == '\t')]

    def ids_to_sent(self, ids:List[int]) -> List[str]:
        return [self.vocab_ids[word_id] for word_id in ids if self.mask_id < word_id < self.vocab_size]

    def shift_ids(self, ids:List[int]) -> List[int]:
        id_len = len(ids)
        if(id_len < self.min_seq_length*2):
            return ids

        return ids[random.randrange(int(len(ids)/2)):]

    def replace_ids(self, ids:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = ids.clone()

        prob_matrix = torch.full(labels.shape, self.replace_prob)

        masked_indices = torch.bernoulli(prob_matrix).bool()

        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.replace_prob)).bool() & masked_indices
        ids[indices_replaced] = self.unk_id

        indices_random = torch.bernoulli(torch.full(labels.shape, self.random_token_prob)).bool() & masked_indices
        random_tokens = torch.randint(self.mask_id+1, self.vocab_size-1, labels.shape, dtype=torch.long)
        ids[indices_random] = random_tokens[indices_random]

        return ids, labels

    def make_input(self, ids:List[int]) -> torch.Tensor:
        if random.random() < self.shift_prob:
            ids = self.shift_ids(ids)

        input_mask = [0] + [1]*len(ids) + [0]

        input_ids = [self.cls_id] + ids + [self.sep_id]
        seq_length  = len(input_ids)

        input_ids.extend(self.padding[seq_length:]), input_mask.extend(self.padding[seq_length:])
        # [self.max_seq_length]
        return torch.tensor(input_ids), torch.tensor(input_mask, dtype=torch.bool)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index:int) -> Dict:
        ids = self.docs[index]

        input_ids, input_mask = self.make_input(ids)

        input_ids, labels = self.replace_ids(input_ids)

        output = {  "input_ids"  : input_ids,
                    "labels"     : labels,
                    "input_mask" : input_mask, }

        if self.is_return_str:
            output["plain_text"] = self.plain_docs[index]

        return output