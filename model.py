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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'



#Decoded Single-Attention Head
class Head(nn.Module):
    def __init__(self, block_size, n_embd, dropout, head_size):
        super().__init__()
        self.head_size = head_size
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
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim = -1)
        att = self.dropout(att)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = att @ v
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, block_size, n_head, n_embd, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [
                Head(block_size, n_embd, dropout, head_size)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd) #Trainable projection onto space of embedding dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Input shape: (Batch, Time, Channels)
        #Output shape: (Batch, Time, Embedding Dimension)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


#Transformer block as per Attention is All You Need
class Block(nn.Module):
    def __init__(self, block_size, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadedAttention(block_size, n_head, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
class MoveFormerModel(nn.Module):
    def __init__(self, block_size, n_layer, n_head, n_embd, dropout, vocab_size):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.block_size = block_size
        self.token_embedding_tbl = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_tbl = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(block_size, n_embd, n_head, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Give every layer a stable, consistent starting point for training.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Small random weights prevent unstable activations while breaking symmetry.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Zero bias starts each layer without an arbitrary offset.
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize token and position vectors on the same small scale.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block size {self.block_size}"
            )

        tok_emb = self.token_embedding_tbl(idx) 
        pos_emb = self.position_embedding_tbl(
            torch.arange(T, device=idx.device)
        )
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            flat_logits = logits.reshape(B * T, C)
            flat_targets = targets.reshape(B * T)
            loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
