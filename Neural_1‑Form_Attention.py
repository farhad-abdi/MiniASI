

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

# ------------------------------
# 1. Neural 1‑Form (core)
# ------------------------------
class NeuralOneForm(nn.Module):
    def __init__(self, d_model, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)

def line_integral_1form(form_mlp, p, q, samples=3):
    device = p.device
    if samples == 3:
        nodes = torch.tensor([0.1127016654, 0.5, 0.8872983346], device=device)
        weights = torch.tensor([0.2777777778, 0.4444444444, 0.2777777778], device=device)
    else:
        nodes = torch.linspace(0, 1, samples, device=device)
        weights = torch.full((samples,), 1.0/samples, device=device)
    direction = q - p
    integral = torch.zeros(p.shape[0], device=device)
    for node, w in zip(nodes, weights):
        t = node
        point = (1 - t) * p + t * q
        vec = form_mlp(point)
        integral += w * torch.sum(vec * direction, dim=-1)
    return integral

# ------------------------------
# 2. Minimal Attention (single head, no FFN, no LayerNorm)
# ------------------------------
class MinimalNeuralFormAttention(nn.Module):
    def __init__(self, d_model, hidden_dim=16):
        super().__init__()
        self.form = NeuralOneForm(d_model, hidden_dim)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, memory=None, mask=None):
        """
        x: (batch, q_len, d_model) – queries
        memory: (batch, kv_len, d_model) – keys/values; if None, use x (self-attention)
        mask: additive mask (e.g., causal)
        """
        batch, q_len, d_model = x.shape
        if memory is None:
            kv = x
        else:
            kv = memory
        kv_len = kv.shape[1]

        # Project values
        V = self.w_v(kv)  # (batch, kv_len, d_model)

        # Compute line‑integral scores (single head)
        scores = torch.zeros(batch, q_len, kv_len, device=x.device)
        form = self.form
        for i in range(q_len):
            p = x[:, i, :]
            for j in range(kv_len):
                q = kv[:, j, :]
                integral = line_integral_1form(form, p, q)
                scores[:, i, j] = integral

        if mask is not None:
            scores = scores + mask

        scores = scores / math.sqrt(d_model)  # scale
        attn = F.softmax(scores, dim=-1)      # (batch, q_len, kv_len)

        # Aggregate values
        out = torch.matmul(attn, V)           # (batch, q_len, d_model)
        out = self.w_o(out)
        return out

# ------------------------------
# 3. Minimal Seq2Seq Model
# ------------------------------
class MinimalNeuralFormSeq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_token_id, d_model=16, hidden_dim=16, max_len=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.max_len = max_len
        self.pad_token_id = pad_token_id

        # Encoder self‑attention (no mask – bidirectional)
        self.enc_attn = MinimalNeuralFormAttention(d_model, hidden_dim)
        # Decoder self‑attention (causal mask applied inside)
        self.dec_attn = MinimalNeuralFormAttention(d_model, hidden_dim)
        # Cross‑attention (decoder queries, encoder keys/values)
        self.cross_attn = MinimalNeuralFormAttention(d_model, hidden_dim)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_ids, decoder_ids):
        # ---- Encoder ----
        enc_seq_len = encoder_ids.size(1)
        enc_pos = self.pos_encoding[:, :enc_seq_len, :]
        enc_x = self.embedding(encoder_ids) + enc_pos
        # No mask – bidirectional
        enc_x = self.enc_attn(enc_x, memory=None, mask=None)   # self‑attention

        # ---- Decoder ----
        dec_seq_len = decoder_ids.size(1)
        dec_pos = self.pos_encoding[:, :dec_seq_len, :]
        dec_x = self.embedding(decoder_ids) + dec_pos

        # Causal mask for decoder self‑attention
        causal_mask = torch.triu(torch.ones(dec_seq_len, dec_seq_len, device=decoder_ids.device), diagonal=1) * -1e9
        # Padding mask (ignore pad tokens)
        pad_mask = (decoder_ids == self.pad_token_id).float() * -1e9
        pad_mask = pad_mask.unsqueeze(1)  # (batch, 1, dec_seq_len)
        self_mask = causal_mask.unsqueeze(0) + pad_mask   # (batch, dec_len, dec_len)

        # Decoder self‑attention
        dec_x = self.dec_attn(dec_x, memory=None, mask=self_mask)

        # Cross‑attention: queries = dec_x, keys/values = enc_x
        # Encoder padding mask for cross‑attention
        enc_pad_mask = (encoder_ids == self.pad_token_id).float() * -1e9
        enc_pad_mask = enc_pad_mask.unsqueeze(1)  # (batch, 1, enc_len)
        # Broadcast to (batch, dec_len, enc_len)
        cross_mask = enc_pad_mask.expand(-1, dec_seq_len, -1)

        dec_x = self.cross_attn(dec_x, memory=enc_x, mask=cross_mask)

        logits = self.head(dec_x)   # (batch, dec_seq_len, vocab_size)
        return logits

# ------------------------------
# 4. Dummy Dataset (same as before)
# ------------------------------
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.idx2word = {0:'<pad>', 1:'<unk>', 2:'<bos>', 3:'<eos>'}

    def add_words(self, sentences):
        for sent in sentences:
            for w in sent.split():
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                    self.idx2word[self.vocab[w]] = w

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.vocab.get(w, self.unk_token_id) for w in text.split()]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special=False):
        words = []
        for i in ids:
            w = self.idx2word.get(i, '<unk>')
            if skip_special and w in ('<pad>','<unk>','<bos>','<eos>'):
                continue
            words.append(w)
        return ' '.join(words)

def build_dummy_data():
    qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Hamlet?", "Shakespeare"),
        ("What is 2+2?", "4"),
        ("What is the largest planet?", "Jupiter"),
        ("What is the smallest country?", "Vatican"),
    ]
    return qa_pairs

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_len=10):
        self.pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        all_text = [q + " " + a for q, a in qa_pairs]
        tokenizer.add_words(all_text)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]
        enc_ids = self.tokenizer.encode(question, add_bos=False, add_eos=False)
        enc_ids = enc_ids[:self.max_len]
        enc_len = len(enc_ids)
        enc_ids = enc_ids + [self.tokenizer.pad_token_id] * (self.max_len - enc_len)

        dec_input = self.tokenizer.encode(answer, add_bos=True, add_eos=False)
        dec_input = dec_input[:self.max_len-1]
        dec_len = len(dec_input)
        dec_input = dec_input + [self.tokenizer.pad_token_id] * (self.max_len - dec_len)

        dec_target = self.tokenizer.encode(answer, add_bos=False, add_eos=True)
        dec_target = dec_target[:self.max_len]
        dec_target = dec_target + [self.tokenizer.pad_token_id] * (self.max_len - len(dec_target))

        return (torch.tensor(enc_ids, dtype=torch.long),
                torch.tensor(dec_input, dtype=torch.long),
                torch.tensor(dec_target, dtype=torch.long))

def collate_fn(batch):
    enc_ids, dec_input, dec_target = zip(*batch)
    return (torch.stack(enc_ids), torch.stack(dec_input), torch.stack(dec_target))

# ------------------------------
# 5. Training and Generation
# ------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for enc_ids, dec_input, dec_target in tqdm(loader, desc='Training'):
        enc_ids, dec_input, dec_target = enc_ids.to(device), dec_input.to(device), dec_target.to(device)
        optimizer.zero_grad()
        logits = model(enc_ids, dec_input)
        loss = criterion(logits.view(-1, logits.size(-1)), dec_target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * dec_target.numel()
        total_tokens += (dec_target != tokenizer.pad_token_id).sum().item()
    return total_loss / total_tokens

@torch.no_grad()
def generate(model, tokenizer, question, max_new_tokens=10, temperature=0.8, device='cpu'):
    model.eval()
    enc_ids = tokenizer.encode(question, add_bos=False, add_eos=False)
    enc_ids = enc_ids[:model.max_len]
    enc_ids = enc_ids + [tokenizer.pad_token_id] * (model.max_len - len(enc_ids))
    enc_ids = torch.tensor([enc_ids], device=device)

    dec_input = torch.tensor([[tokenizer.bos_token_id]], device=device)
    for _ in range(max_new_tokens):
        curr_len = dec_input.size(1)
        if curr_len < model.max_len:
            pad = torch.full((1, model.max_len - curr_len), tokenizer.pad_token_id, device=device)
            dec_full = torch.cat([dec_input, pad], dim=1)
        else:
            dec_full = dec_input[:, -model.max_len:]
        logits = model(enc_ids, dec_full)
        next_token_logits = logits[0, curr_len-1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == tokenizer.eos_token_id:
            break
        dec_input = torch.cat([dec_input, torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(dec_input[0].tolist(), skip_special=True)

# ------------------------------
# 6. Main
# ------------------------------
if __name__ == '__main__':
    tokenizer = SimpleTokenizer()
    qa_pairs = build_dummy_data()
    dataset = QADataset(qa_pairs, tokenizer, max_len=10)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(tokenizer.vocab)
    pad_id = tokenizer.pad_token_id

    model = MinimalNeuralFormSeq2Seq(
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        d_model=16,
        hidden_dim=16,
        max_len=10
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    print("Training minimal model...")
    for epoch in range(50):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    q = "What is the capital of France?"
    ans = generate(model, tokenizer, q, device=device)
    print(f"\nQ: {q}\nA: {ans}")

    q = "Who wrote Hamlet?"
    ans = generate(model, tokenizer, q, device=device)
    print(f"\nQ: {q}\nA: {ans}")

    q = "AI is"
    ans = generate(model, tokenizer, q, device=device)
    print(f"\nQ: {q}\nA: {ans}")

