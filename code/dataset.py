import torch
import json
import re

from pathlib import Path

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
        self.is_return_str = kwargs.pop("is_return_str", False)

    @staticmethod
    def from_json(path:str):
        kwargs = json.load(open(path, 'r', encoding='utf-8', errors='ignore'))
        return TextDatasetConfig(**kwargs)

class DarkHistoryDataset(torch.utils.data.Dataset):
    def __init__(self, config:TextDatasetConfig):
        self.block_size     = config.input_length
        self.max_seq_length = config.max_seq_length
        self.is_return_str  = config.is_return_str

        self.vocab = {}
        self.vocab_ids = []
        self.vocab_size = 0
        self.load_vocab(config.vocab_path)

        self.data_len = 0
        self.plain_docs = []
        self.docs = []
        if config.data_path is not None:
            self.load_data(config.data_path)

        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]

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
        return [self.vocab_ids[word_id] for word_id in ids if 0 <= word_id < self.vocab_size]

    def make_input(self, ids:List[int]) -> torch.Tensor:

        input_ids = [self.cls_id] + ids + [self.sep_id]
        seq_length  = len(input_ids)

        input_ids.extend(self.padding[seq_length:])
        # [self.max_seq_length]
        return torch.tensor(input_ids)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index:int) -> Dict:
        ids = self.docs[index]

        input_ids = self.make_input(ids)

        output = { "input_ids" : input_ids}

        if self.is_return_str:
            output["plain_text"] = self.plain_docs[index]

        return output