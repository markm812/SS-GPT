import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iter = 1000
eval_every = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iter = 200
d_model = 384
# d_head = d_model // n_head
n_head = 6
n_layers = 6
dropout_p = 0.2
# ---------------

torch.manual_seed(1337)

# data: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O ./dataset/data.txt
with open("./dataset/data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# unique characters in the data
unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)

# create char to index and index to char mapping
char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}

# tokenizer
encode = lambda s: [char_to_idx[char] for char in s]
decode = lambda l: "".join([idx_to_char[idx] for idx in l])

# create dataset for train and eval
data = torch.tensor(encode(text), dtype=torch.long)
nm = int(len(data) * 0.9)
train_data = data[:nm]
val_data = data[nm:]


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        lossess = torch.zeros(eval_iter)
        for i in range(eval_iter):
            X, Y = get_batch(split)
            y_pred, loss = model(X, Y)
            lossess[i] = loss.item()
        out[split] = lossess.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout_p)

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

    def __init__(self, n_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """linear layer followed by a non-linearity"""

    def __init__(self, d_model) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication -> computation"""

    def __init__(self, n_emdb, n_head) -> None:
        super().__init__()
        head_size = n_emdb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[Block(d_model, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

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
            torch.arange(T, device=device)
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
            idx_cond = idx[:, -block_size:]
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


# create model and optimizer

print("Using device: ", device)
model = GPTLanguageModel()

# model resuming/evaluating
model = torch.load("model.pth", weights_only=False)

# model training
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for steps in range(max_iter):  # increase number of steps for good results...

    if steps % eval_every == 0 or steps == max_iter - 1:
        losses = estimate_loss()
        print(
            f"Step: {steps}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m, "model.pth")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))