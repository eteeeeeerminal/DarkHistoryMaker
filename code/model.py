import json
import torch

from torch.nn import CrossEntropyLoss

from reformer_pytorch import ReformerLM
from typing import Tuple, List

class ReformerGenConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.pop("vocab_size", 20000)
        self.hidden_size = kwargs.pop("hidden_size", 512)
        self.emb_dim = kwargs.pop("emb_dim", 512)
        self.depth = kwargs.pop("depth", 12)
        self.heads = kwargs.pop("heads", 8)

        self.causal  = kwargs.pop("causal ", True)

        self.n_hashes = kwargs.pop("n_hashes", 4)

        self.weight_tie = kwargs.pop("weight_tie", False)
        self.full_attn_thres = kwargs.pop("full_attn_thres", 16)

        self.use_scala_norm = kwargs.pop("use_scala_norm", False)

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
            heads = self.config.heads,
            max_seq_len = self.config.max_position_embeddings,

            causal  = self.config.causal,
            emb_dim = self.config.emb_dim,

            n_hashes = self.config.n_hashes,

            weight_tie = self.config.weight_tie,

            full_attn_thres = self.config.full_attn_thres,

            use_scale_norm = self.config.use_scala_norm
        )

    @staticmethod
    def from_pretrained(model_path, config:ReformerGenConfig, device):
        model = ReformerGenModel(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def generate(self, sent_ids:torch.Tensor, eos_id) -> List[int]:
        start_len = sent_ids.shape[0]
        mask = torch.tensor([0]*self.config.max_position_embeddings, dtype=torch.bool, device=sent_ids.device)

        mask = mask.unsqueeze(0)
        mask[..., 1:start_len] = True
        sent_ids = sent_ids.unsqueeze(0)

        padding  = torch.zeros(1, self.config.max_position_embeddings, dtype=torch.long, device=sent_ids.device)
        sent_ids = torch.cat((sent_ids, padding[:, start_len:]), 1)

        for i in range(start_len, self.config.max_position_embeddings):
            output = self.forward(sent_ids, input_mask=mask)[0]
            output = output.argmax(dim=-1)
            sent_ids[..., i] = output[..., i-1]
            mask[..., i] = True
            if output[..., i-1] == eos_id:
                break

        return list(sent_ids.squeeze())

    def forward(
        self,
        input_ids,
        input_mask=None,
        input_attn_mask=None,
        lm_labels=None
    ) -> Tuple:

        output = self.reformer(
            input_ids,
            input_mask=input_mask,
            input_attn_mask=input_attn_mask
        )

        outputs = output,

        if lm_labels is not None:
            output = output[..., :-1, :].contiguous()
            lm_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(output.view(-1, output.size(-1)), lm_labels.view(-1))

            outputs = (lm_loss,) + outputs

        return outputs