import torch
from torch import nn
from torch.nn import Linear, Sequential, ReLU
from transformers import XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class MlpParsingModel(nn.Module):
    def __init__(self, roberta_hidden_dim=768, mlp_dim=500, roberta_id="xlm-roberta-base", dropout=0.1, activation="relu"):
        super().__init__()

        self.roberta_hidden_dim = roberta_hidden_dim
        self.mlp_dim = mlp_dim

        # https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        self.roberta = XLMRobertaModel.from_pretrained(roberta_id)
        self.mlp = Sequential(Linear(roberta_hidden_dim, mlp_dim), ReLU(), Linear(mlp_dim, mlp_dim), ReLU())

        # freeze Roberta parameters
        for name, param in self.named_parameters():
            if name.startswith("roberta"):
                param.requires_grad = False


    def forward(self, x, attention_mask):
        # x: LongTensor (bs, seqlen)
        # attention_mask: IntTensor (bs, seqlen); 1 = real token, 0 = padding
        bs = x.shape[0]
        seqlen = x.shape[1]

        # encode x with Roberta
        out = self.roberta(x, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state  # (bs, seqlen, hidden_size)

        emb = self.mlp(hidden_states) # (bs, seqlen, mlp_dim)
        assert emb.shape == (bs, seqlen, self.mlp_dim)

        scores = torch.einsum("ijk,ilk->ijl", emb, emb)
        assert scores.shape == (bs, seqlen, seqlen)

        return scores