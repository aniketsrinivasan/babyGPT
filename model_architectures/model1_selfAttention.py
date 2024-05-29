import torch
import torch.nn as nn
from torch.nn import functional as F


# ~~~~~~~~~~ HYPERPARAMETERS ~~~~~~~~~~
batch_size = 32         # how many independent sequences (blocks) we process in parallel
block_size = 8          # how large each block is
max_iters = 5000
eval_interval = 500
learning_rate = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32              # number of embedding dimensions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


torch.manual_seed(1337)     # for reproducibility

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# All the unique characters that occur in our data:
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoding input text into IDs:
#    creating a mapping:
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#    creating functions (lookup tables):
encode = lambda s: [stoi[c] for c in s]            # encoder: Str -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])   # decoder: list[int] -> Str

# Encode our entire dataset and store as a torch.Tensor:
data = torch.tensor(encode(text), dtype=torch.long)

# We split our dataset into a train:validation split.
#    first 90% will be training data, rest will be validation.
n = int(0.9 * len(data))
#    validation:
val_data = data[n:]
#    training:
train_data = data[:n]


def get_batch(split):
    data = train_data if (split == "train") else val_data
    # We generate batch_size number of random offsets (these are our block starting indices):
    #    starts at len(data)-block_size to avoid IndexError.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stacking our tensors as rows:
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


# Estimating average loss over multiple batches:
#   torch.no_grad(): decorator to tell PyTorch that we will not call loss.backward() here,
#                    allowing better efficiency when running.
@torch.no_grad()
def estimate_loss():
    out = {}
    # Setting the model to evaluation mode:
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Setting the model back to training mode:
    model.train()
    return out


# Single-Head Self-Attention class:
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Creating key, value and query as Linear neural network layers:
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Creating the lower triangular matrix (this is how it is done in practice using PyTorch):
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    # Implementing the forward method based on matrix multiplication (see GPT_inference.ipynb for
    #   specific implementation details):
    def forward(self, x):
        # B:    batch (number of batches)
        # T:    timesteps (number of elements per batch)
        # C:    channels
        B, T, C = x.shape       # getting dimensions based on input

        # Creating key and query by propagating the Linear layers into matrices:
        k = self.key(x)     # (B, T, C)
        q = self.query(x)   # (B, T, C)

        # Compute attention scores ("affinities"):
        weights = q @ k.transpose(-2, -1) * (C ** -0.5)     # (QK^t)/sqrt(d_k) in the paper
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))    # autoregressive
        weights = F.softmax(weights, dim=-1)                # Softmax((QK^t)/sqrt(d_k))

        # Performing the weighted aggregation of the values:
        v = self.value(x)   # (B, T, C)
        output = weights @ v
        return output


# Multi-Head Attention class (parallelized Single-Head):
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Concatenating outputs over the channel dimension:
        return torch.cat([h(x) for h in self.heads], dim=-1)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Each token reads directly off the logits for the next token from a lookup table:
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Creating a position embedding table:
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Creating self-attention heads:
        self.sa_heads = MultiHeadAttention(4, n_embd//4)    # 4 heads of self-attention

        # We need a linear layer:
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (batch, time) tensors of integers:
        #    here:
        #      batch = batch_size
        #      time = block_size
        #      channel = vocab_size
        # Getting token embeddings:
        tok_emb = self.token_embedding_table(idx)  # (batch, time, channel) or (B, T, C)
        # Getting positional embeddings:
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb   # (B, T, C)
        x = self.sa_heads(x)    # apply multi-headed self attention
        logits = self.lm_head(x)             # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # F.cross_entropy expects a (B, C, T) input so we reshape:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # We are predicting what's next based on the identity of an individual token.
            # This is done for every token we have.
            loss = F.cross_entropy(logits, targets)  # we know the next character, how well are we predicting?

        return logits, loss

    # Generating new tokens.
    #    idx is a (B, T) array of indices in the current context.
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):  # we generate 1, 2, ..., max_new_tokens tokens
            # Cropping idx to the last block_size tokens (we can only have at most block_size inputs):
            idx_cond = idx[:, -block_size:]
            # Get predictions:
            logits, loss = self(idx_cond)
            # Focus only on the last time step (a.k.a. our prediction):
            logits = logits[:, -1, :]  # becomes (B, C) from (B, T, C)
            # Softmax to get probabilities:
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution:
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence:
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# We import the AdamW optimizer to begin training and gradient descent:
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop:
for iter in range(max_iters):
    # Every once in a while, evaluate the loss of train and validate sets:
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    # Sample batch of data:
    xb, yb = get_batch("train")

    # Evaluate the loss:
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate from the model:
context = torch.zeros([1, 1], dtype=torch.long, device=device)
pred = m.generate(context, max_new_tokens=500)[0].tolist()
print(pred)
print(decode(pred))
