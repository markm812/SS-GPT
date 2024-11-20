import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, config) -> None:
        super().__init__()
        self.key = nn.Linear(config.d_model, head_size, bias=False)
        self.query = nn.Linear(config.d_model, head_size, bias=False)
        self.value = nn.Linear(config.d_model, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B, T, C
        q = self.query(x)  # B, T, C

        # compute attention affinities score
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, H) @ (B, H, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # weight aggregation
        v = self.value(x)  # B, T, C
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple self-attention heads"""

    def __init__(self, n_heads, head_size, config) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size=head_size, config=config) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """linear layer followed by a non-linearity"""

    def __init__(self, d_model, config) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.dropout_p),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication -> computation"""

    def __init__(self, n_emdb, n_head, config) -> None:
        super().__init__()
        head_size = n_emdb // n_head
        self.sa = MultiHeadAttention(n_heads=n_head, head_size=head_size, config=config)
        self.ffwd = FeedFoward(d_model=config.d_model, config=config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        # residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, token_size,config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(token_size, config.d_model)
        self.position_embedding_table = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.Sequential(
            *[
                Block(config.d_model, n_head=config.n_head, config=config)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, token_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idea: X holds both the token identities and the position of the token in the sequence
        token_emb = self.token_embedding_table(idx)  # (Batch, Time, d_model)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.config.device)
        )  # (Time, d_model)
        x = token_emb + pos_emb  # (Batch, Time, d_model)
        x = self.blocks(x)  # (Batch, Time, d_model)
        x = self.ln_f(x)  # (Batch, Time, d_model)
        logits = self.lm_head(x)  # (Batch, Time, Vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)  # (Batch * Time)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) in current context
        for _ in range(max_new_tokens):
            # crop the context to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size :]
            # get prediction
            logits, loss = self(idx_cond)
            # only take the last token
            logits = logits[:, -1, :]
            # softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
