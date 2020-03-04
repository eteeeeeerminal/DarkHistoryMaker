import json
import torch

from torch.nn import CrossEntropyLoss

from reformer_pytorch import ReformerLM
from typing import Tupple

class ReformerGenConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.pop("vocab_size", 20000)
        self.hidden_size = kwargs.pop("hidden_size", 768)

        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 4096)

    @staticmethod
    def from_json(path:str) -> ReformerGenConfig:
        kwargs = json.load(open(path, 'r', encoding='utf-8', errors='ignore'))
        return ReformerGenConfig(kwargs)

class ReformerGenModel(nn.Module):
    def __init__(self, config:ReformerGenConfig):
        self.reformer = ReformerLM(
            num_tokens = config.vocab_size,
            dim = config.hidden_size,

            max_seq_len = config.max_position_embeddings
        )

    def random_generate(self) -> str:
        pass

    def generate(self, sent:torch.Tensor) -> List[int]:
        pass

    def forward(
        self,
        input_ids,
        lm_labels=None
    ) -> Tupple:

        output = self.reformer(
            input_ids
        )

        outputs = output,

        if lm_labels is not None:
            output = output[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct()

            outputs = (lm_loss,) + outputs

        return outputs