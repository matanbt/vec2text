# Implements transformer encoder

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import LayerNorm

"""
Implementation of the encoder in 'Attention is all you need'
    [https://arxiv.org/pdf/1706.03762].
With the help and inspiration from (1) the paper itself (2) non-official torch implementation of attention is all 
you need and (3) GPT2 implementation from scratch by Andrej Karpathy
    [https://github.com/karpathy/nanoGPT/blob/master/model.py]
    [https://github.com/jadore801120/attention-is-all-you-need-pytorch]
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""


# TODO base it on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
# TODO also on https://github.com/rmokady/CLIP_prefix_caption/blob/main/train.py
# FOLLOW https://github.com/rmokady/CLIP_prefix_caption/blob/1ad805a844a62ab2e5480479aa021bccf0d4d12a/train.py#L203
#       where they reshape the input representation, so it will be a multi-vecotr (input to transformer)
class CausalSelfAttention(nn.Module):
    """ Attention sublayer, as described in Attention paper
    (differ from GPT2's by: not having dropout on the attention matrix and at the end; attending to all tokens)"""

    def __init__(self, hidden_dim, n_head, p_dropout, block_size=1024):
        super().__init__()
        assert hidden_dim % n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        # output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)

        # causal mask to ensure that attention is only applied to the left in the input sequence [DISABLED]
        # self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
        #                      .view(1, 1, block_size, block_size))

        self.n_head = n_head
        self.n_embd = hidden_dim

    def forward(self, x, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (hidden_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Masking the attention:
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # [DISABLED] we allow attending from and to all tokens
        if mask is not None:
            att = att.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)  # a dropout is performed before adding the residual, in the outer model
        return y


class FFN(nn.Module):
    """ 'Positional' feedforward sublayer, as described in Attention paper"""

    def __init__(self, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.llayer1 = nn.Linear(hidden_dim, 4 * hidden_dim)  # ratio used in original Attention paper
        self.relu = nn.ReLU()
        self.llayer2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.llayer1(x)
        x = self.relu(x)
        x = self.llayer2(x)
        return x


class Block(nn.Module):  # residual block
    """ Attention+feedforward block (adds to residual stream), as described in Attention paper"""

    def __init__(self, hidden_dim, n_head, p_dropout):
        super().__init__()
        self.attn = CausalSelfAttention(hidden_dim, n_head, p_dropout)
        self.ln_1 = LayerNorm(hidden_dim)
        self.drop_1 = nn.Dropout(p_dropout)  # todo good?

        self.ffn = FFN(hidden_dim)
        self.ln_2 = LayerNorm(hidden_dim)
        self.drop_2 = nn.Dropout(p_dropout)  # todo good?

    def forward(self, x, mask):
        x = self.ln_1(x + self.drop_1(self.attn(x, mask)))
        x = self.ln_2(x + self.drop_2(self.ffn(x)))
        return x


class TransformerEncoderAligner(nn.Module):
    """ Encoder part of the transformer model, as described in Attention paper"""

    def __init__(self,
                 source_emb_dim: int, target_emb_dim: int,
                 num_blocks: int,
                 hidden_dim: int,
                 dropout_p: float = 0.1,
                 n_head: int = 4,
                 seq_len: int = 4,  # (artificial) sequence length to input the encoder with
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.project_to_input = nn.Linear(source_emb_dim, hidden_dim * seq_len)
        self.blocks = nn.ModuleList([Block(hidden_dim, n_head, dropout_p) for _ in range(num_blocks)])
        self.project_to_output = nn.Linear(hidden_dim, target_emb_dim)

        self.model_kwargs = dict(
            source_emb_dim=source_emb_dim,
            target_emb_dim=target_emb_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
            n_head=n_head,
            seq_len=seq_len,
            model_class_name=self.__class__.__name__
        )

    def forward(self, x):
        # reshape input to (batch_size, sequence_length, hidden_dim), to encode a sequence
        x = self.project_to_input(x).view(x.size(0), -1, self.hidden_dim)
        # apply the transformer blocks
        for block in self.blocks:
            x = block(x, mask=None)  # mask is currently disabled (all interactions are allowed)
        # perform mean pooling and project to output
        x = self.project_to_output(x.mean(dim=1))

        return x
