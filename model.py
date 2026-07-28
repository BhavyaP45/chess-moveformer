import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass



@dataclass
class MoveFormerConfig():
    block_size: int = 768 #Covers 99th percentile of game length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    vocab_size: int = 29 #All chars found in dataset + '\0' padding character (excludes '\n' delimiter)


#Decoded Single-Attention Head
class Head(nn.Module):
    def __init__(self, block_size, n_embd, dropout, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Input shape: (Batch, Time, Channels)
        #Output shape: (Batch, Time, Head size)
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)

        att = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        att = torch.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim = -1)
        att = self.dropout(att)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = att @ v
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads =  nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd) #Trainable projection onto space of embedding dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Input shape: (Batch, Time, Channels)
        #Output shape: (Batch, Time, Embedding Dimension)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MoveFormerModel(nn.Module):
    def __init__(self, block_size, n_embd, vocab_size):
        super().__init__()
        self.token_embedding_tbl = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_tbl = nn.Embedding(block_size, n_embd)
