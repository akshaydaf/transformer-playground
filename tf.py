# globals
batch_size = 64
block_size = 512
max_iters = 1000
eval_interval = 500
eval_iters = 200
lr = 1e-3
n_embd = 384

n_layer = 1
n_head = 2
is_dropout = False
dropout = 0.2 if is_dropout else 0
train_path = 'train.csv'
model_path = "weights.pth"
is_layer_norm = False


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = '^'.join(pd.read_csv(train_path, nrows=100000).dropna()['text']) # ^ is the end of document token

char = set()
vocab = sorted(set(dataset))
vocab_length = len(vocab)

# create character level encodings
stoi = {char: idx for idx, char in enumerate(vocab)}
itos = {idx: char for idx, char in enumerate(vocab)}

encode = lambda x: [stoi[i] for i in x]
decode = lambda x: [itos[i] for i in x]

# data into a tensor
data = torch.tensor(encode(dataset), dtype=torch.long)


# train/ val split
n = int(0.9 * len(dataset))
train = data[:n]
val = data[n:]


# Define randomized batches of data

def get_batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def build_doc_mask(idx):
    B, T = idx.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))
    seg = (idx == stoi["^"]).cumsum(dim=1) # represents the document ids 

    same_doc = seg[:, :, None] == seg[:, None, :] 

    not_sep = (idx != stoi["^"])[:, None, :] 
    mask = causal[None, :, :] & same_doc & not_sep ## adding batch dimension
    mask |= torch.eye(T, dtype=torch.bool, device=idx.device)[None] # Mask turns into a square matrix mask[i, j] will be a separator column set to false
    return mask

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss
        out[split] = torch.mean(losses)
    m.train()
    return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        if is_layer_norm:
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask):
        if is_layer_norm:
            x = self.ln1(x)
        x = x + self.sa(x, mask)
        if is_layer_norm:
            x = self.ln2(x)
        return x + self.ff(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, mask):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5

        wei = wei.masked_fill(~mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.drop(out)
        return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_length, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        if is_layer_norm:
            self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_length)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        mask = build_doc_mask(idx)
        for block in self.blocks:
            x = block(x, mask)
        if is_layer_norm:
            x = self.ln_f(x)
        logits = self.lm_head(x) # B, T, vocab_size
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
m = BigramLanguageModel()
m = m.to(device)
idx = torch.ones((1, 1), dtype=torch.long).to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

print(device)
# training_loop
for steps in range(max_iters):

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']}, val loss {losses['val']}")
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))


torch.save(m.state_dict(), model_path)
print(f"Model parameters successfully stored at {model_path}")