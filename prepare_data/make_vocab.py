# -*- coding: utf-8 -*-

# コーパスから語彙ファイルをつくる。

import codecs
import json
from collections import defaultdict
from pathlib import Path

import re

class VocabMaker():
    def __init__(self, special_token = []):
        self.vocab = defaultdict(int)
        self.special_words = special_token
        self.del_pattern = re.compile(r"[\t\n]")

    def add_word(self, sentence:str):
        for word in sentence:
            word = self.del_pattern.sub("", word)

            if not word:
                continue

            self.vocab[word] += 1

        return True

    def save_vocab(self, save_path:str):

        vocab = sorted(self.vocab, key=self.vocab.get, reverse=True)

        with open(save_path, mode='w', encoding='utf-8') as f:
            f.write('\n'.join(self.special_words))
            f.write('\n')
            f.write('\n'.join(vocab))
            f.write('\n')

        print("complete_save")

if __name__ == '__main__':

    config = json.load(open("../.json", "r"))

    save_path  = "vocab.txt"
    dataset_path = config["dataset_path"]

    special_token = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    vocab = VocabMaker(special_token = special_token)
    print("made VocabMaker")

    with open(dataset_path, mode='r', encoding='utf-8', errors='ignore') as f:
        sent = f.read()
        print(sent)
        [vocab.add_word(s) for s in sent]

    print('complete ' + dataset_path)
    vocab.save_vocab(save_path)
