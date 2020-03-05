import json
import torch

from torch.nn import CrossEntropyLoss

from reformer_pytorch import ReformerLM
from typing import Tuple, List

class ReformerGenConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.pop("vocab_size", 20000)
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.depth = kwargs.pop("depth", 12)

        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 128)

    @staticmethod
    def from_json(path:str):
        kwargs = json.load(open(path, 'r', encoding='utf-8', errors='ignore'))
        return ReformerGenConfig(**kwargs)

class ReformerGenModel(torch.nn.Module):
    def __init__(self, config:ReformerGenConfig):
        self.config = config
        super().__init__()
        self.reformer = ReformerLM(
            num_tokens = self.config.vocab_size,
            dim = self.config.hidden_size,
            depth = self.config.depth,
            max_seq_len = self.config.max_position_embeddings
        )

    @staticmethod
    def from_pretrained(model_path, config:ReformerGenConfig):
        model = ReformerGenModel(config)
        model.load_state_dict(torch.load(model_path))
        return model

    def generate(self, sent:torch.Tensor) -> List[int]:
        pass

    def forward(
        self,
        input_ids,
        lm_labels=None
    ) -> Tuple:

        output = self.reformer(
            input_ids
        )

        outputs = output,

        if lm_labels is not None:
            output = output[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(output.view(-1, output.size(-1)), lm_labels.view(-1))

            outputs = (lm_loss,) + outputs

        return outputs