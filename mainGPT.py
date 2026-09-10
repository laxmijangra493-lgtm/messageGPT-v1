# Let's code messageGPT v2 code...
# I have added if condition and now it was more efficient

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from Mytokenizer import train_bpe


MODE = 'generate' # 'train' or 'generate'
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITER = 5500
EVAL_INTERVAL = 500
NUM_EMB = 384
NUM_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
LEARNING_RATE = 1e-3
CKPT_PATH = "pretrained.pt"

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_size = NUM_EMB // NUM_HEAD
        self.c_attn = nn.Linear(NUM_EMB, 3 * NUM_EMB, bias=False)
        self.proj = nn.Linear(NUM_EMB, NUM_EMB)
        self.resid_dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
    
        k = k.view(B, T, NUM_HEAD, self.head_size).transpose(1, 2)
        q = q.view(B, T, NUM_HEAD, self.head_size).transpose(1, 2)
        v = v.view(B, T, NUM_HEAD, self.head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_EMB, 4 * NUM_EMB),
            nn.GELU(),
            nn.Linear(4 * NUM_EMB, NUM_EMB),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        # Reverted to original names to match your saved checkpoint
        self.layer1 = nn.LayerNorm(NUM_EMB)
        self.MH = MultiHead()
        self.layer2 = nn.LayerNorm(NUM_EMB)
        self.FNN = FeedForward()

    def forward(self, x):
        x = x + self.MH(self.layer1(x))
        x = x + self.FNN(self.layer2(x))
        return x

class MessageGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Reverted to original names to match your saved checkpoint
        self.embedding_table = nn.Embedding(vocab_size, NUM_EMB)
        self.positional_table = nn.Embedding(BLOCK_SIZE, NUM_EMB)
        self.block = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.layernorm = nn.LayerNorm(NUM_EMB)
        self.lm_head = nn.Linear(NUM_EMB, vocab_size)
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
        x = self.embedding_table(idx) + self.positional_table(torch.arange(T, device=idx.device))
        x = self.block(x)
        x = self.layernorm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=0.7, topk=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temp

            if topk is not None:
                v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        
        self.train()
        return idx

def encode_word(word, merges, merge_ranks):
    tokens = tuple(list(word) + ["</w>"])
    while len(tokens) > 1:
        pairs = [(tokens[j], tokens[j + 1]) for j in range(len(tokens) - 1)]
        valid_pairs = [p for p in pairs if p in merge_ranks]
        if not valid_pairs:
            break
        pair_to_merge = min(valid_pairs, key=lambda p: merge_ranks[p])
        merged_val = merges[pair_to_merge]
    
        new_tokens = []
        j = 0
        while j < len(tokens):
            if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == pair_to_merge:
                new_tokens.append(merged_val)
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1
        tokens = tuple(new_tokens)
    return tokens

def encode_text(text, stoi, merges, merge_ranks):
    ids = []
    for w in text.split():
        for token in encode_word(w, merges, merge_ranks):
            if token in stoi:
                ids.append(stoi[token])
            else:
                for char in token:
                    ids.append(stoi.get(char, stoi["<unk>"]))
    return ids

def decode(ids, itos):
    return "".join([itos.get(i, "") for i in ids]).replace("</w>", " ")

@torch.no_grad()
def estimate_loss(model, get_batch_fn):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch_fn(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    if MODE.lower() == 'train':
        with open('tinystories.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        merges, stoi, itos = train_bpe(text, num_merges=1800)

        if "<unk>" not in stoi:
            unk_idx = len(stoi)
            stoi["<unk>"] = unk_idx
            itos[unk_idx] = "<unk>"

        vocab_size = len(stoi)
        merge_ranks = {p: r for r, p in enumerate(merges)}

        print("Tokenizing dataset...")
        data = torch.tensor(encode_text(text, stoi, merges, merge_ranks), dtype=torch.long)
        n_train = int(0.9 * len(data))
        train_data, val_data = data[:n_train], data[n_train:]

        def get_batch(split):
            d = train_data if split == 'train' else val_data
            ix = torch.randint(0, len(d) - BLOCK_SIZE, (BATCH_SIZE,))
            x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
            y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        model = MessageGPT(vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITER, eta_min=1e-4)

        print("Starting training...")
        for iter in range(MAX_ITER):
            if iter % EVAL_INTERVAL == 0:
                losses = estimate_loss(model, get_batch)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if device.type == 'mps': 
                    torch.mps.empty_cache()

            xb, yb = get_batch('train')
            _, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        torch.save({
            "model_state_dict": model.state_dict(),
            "merges": merges,
            "stoi": stoi,
            "itos": itos,
            "vocab_size": vocab_size
        }, CKPT_PATH)
        print(f"Training complete! Saved to {CKPT_PATH}")

    elif MODE.lower() == "generate":
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(f"Checkpoint '{CKPT_PATH}' not found. Set MODE='train' and run first.")

        print("Loading checkpoint...")
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        
        vocab_size = checkpoint.get("vocab_size", len(checkpoint["stoi"]))
        itos = checkpoint["itos"]
        
        model = MessageGPT(vocab_size).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
        print("\n" + decode(generated_ids, itos))

if __name__ == "__main__":
    main()