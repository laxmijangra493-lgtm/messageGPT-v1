import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from mainGPT import MessageGPT, encode_text, decode

BATCH_SIZE = 32          
BLOCK_SIZE = 256
MAX_ITERS = 4000     
LEARNING_RATE = 2e-5     
EVAL_INTERVAL = 200
EOS_DELIMITER = "\n\n<|endoftext|>\n\n"
CKPT_IN = "pretrained.pt"
CKPT_OUT = "finetuned.pt"

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if not os.path.exists(CKPT_IN):
    raise FileNotFoundError(f"Cannot fine-tune without base checkpoint '{CKPT_IN}'.")

print("Loading pre-trained model and tokenizer...")
checkpoint = torch.load(CKPT_IN, map_location=device, weights_only=False)
merges = checkpoint['merges']
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = checkpoint.get("vocab_size", len(stoi))
merge_ranks = {p: r for r, p in enumerate(merges)}

model = MessageGPT(vocab_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])


class InstructionDataset(Dataset):
    def __init__(self, hf_dataset, block_size, stoi, merges, merge_ranks):
        self.samples = []
        pad_token_id = stoi.get("<|pad|>", 0)

        for example in hf_dataset:
            instruction = example["instruction"]
            context = example["context"]
            response = example["response"]

            if context:
                prompt_str = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
            else:
                prompt_str = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            full_str = prompt_str + response + EOS_DELIMITER
            
            prompt_ids = encode_text(prompt_str, stoi, merges, merge_ranks)
            full_ids = encode_text(full_str, stoi, merges, merge_ranks)

            if len(full_ids) < 2:
                continue

            # Truncate if sample exceeds model block size
            full_ids = full_ids[:block_size + 1]
            
            x = torch.tensor(full_ids[:-1], dtype=torch.long)
            y = torch.tensor(full_ids[1:], dtype=torch.long)

            # Target Masking: Mask instruction tokens with -100 so loss is ignore-indexed
            prompt_len = min(len(prompt_ids), len(y))
            y[:prompt_len - 1] = -100  

            # Pad short sequences up to block_size
            pad_len = block_size - len(x)
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])

            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


print("Formatting and encoding dataset...")
raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

split_ds = raw_dataset.train_test_split(test_size=0.05, seed=1337)
train_dataset = InstructionDataset(split_ds['train'], BLOCK_SIZE, stoi, merges, merge_ranks)
val_dataset = InstructionDataset(split_ds['test'], BLOCK_SIZE, stoi, merges, merge_ranks)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=1e-6)

print("Starting fine-tuning...")
model.train()
train_iter = iter(train_loader)

for step in range(MAX_ITERS):
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)

    xb, yb = xb.to(device), yb.to(device)

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        model.eval()
        with torch.no_grad():
            val_xb, val_yb = next(iter(val_loader))
            val_xb, val_yb = val_xb.to(device), val_yb.to(device)
            _, val_loss = model(val_xb, val_yb)
            
        print(f"Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        model.train()

torch.save({
    "model_state_dict": model.state_dict(),
    "merges": merges,
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size
}, CKPT_OUT)
print(f"Fine-tuning complete! Saved to {CKPT_OUT}")

# Inference Test
prompt = "### Instruction:\nExplain gravity in simple terms.\n\n### Response:\n"
prompt_ids = encode_text(prompt, stoi, merges, merge_ranks)
x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
x = x[:, -BLOCK_SIZE:]

model.eval()
with torch.no_grad():
    output_ids = model.generate(x, max_new_tokens=150)[0].tolist()

print("\n--- Model Output ---")
decoded_text = decode(output_ids, itos)

if EOS_DELIMITER.strip() in decoded_text:
    decoded_text = decoded_text.split(EOS_DELIMITER.strip())[0]

print(decoded_text)