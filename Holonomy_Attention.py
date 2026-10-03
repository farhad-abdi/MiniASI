
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
from tqdm import tqdm

# ------------------------------
# 1. Non‑Commutative Differential Form
# ------------------------------
class NonCommutativeOneForm(nn.Module):
    def __init__(self, d_model, hidden_dim=32):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model * d_model)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        flat = self.net(x)
        flat = torch.clamp(flat, min=-5.0, max=5.0)
        return flat.view(x.size(0), self.d_model, self.d_model)

# ------------------------------
# 2. Causal Non‑Commutative Attention
# ------------------------------
class CausalNonCommutativeAttention(nn.Module):
    def __init__(self, d_model, hidden_dim=32):
        super().__init__()
        self.form = NonCommutativeOneForm(d_model, hidden_dim)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.temp = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer('nodes', torch.tensor([0.1127, 0.5, 0.8873]))
        self.register_buffer('weights', torch.tensor([0.2778, 0.4444, 0.2778]))

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len, seq_len) with -inf for future positions
        """
        batch, seq_len, d = x.shape
        kv = x  # self‑attention: keys and values are the same as queries
        V = self.w_v(kv)
        out = torch.zeros(batch, seq_len, d, device=x.device)

        for i in range(seq_len):
            p = x[:, i, :]                       # (batch, d)
            P = p.unsqueeze(1).expand(-1, seq_len, -1)   # (batch, seq_len, d)
            Q = kv
            direction = Q - P
            step_norm = torch.norm(direction, dim=-1, keepdim=True)

            batch_seq = batch * seq_len
            P_flat = P.reshape(batch_seq, d)
            Q_flat = Q.reshape(batch_seq, d)
            direction_flat = direction.reshape(batch_seq, d)
            step_norm_flat = step_norm.reshape(batch_seq, 1, 1)

            U = torch.eye(d, device=x.device).unsqueeze(0).expand(batch_seq, -1, -1)

            for node, w in zip(self.nodes, self.weights):
                mid = (1 - node) * P_flat + node * Q_flat
                A_mat = self.form(mid).reshape(batch_seq, d, d)
                term = A_mat * (w * step_norm_flat)
                term = torch.clamp(term, min=-10.0, max=10.0)
                exp_term = torch.matrix_exp(term)
                U = torch.bmm(exp_term, U)

            U = U.reshape(batch, seq_len, d, d)
            scores = torch.diagonal(U, dim1=-2, dim2=-1).sum(-1)   # (batch, seq_len)

            if mask is not None:
                row_mask = mask[:, i, :]  # (batch, seq_len)
                scores = scores + row_mask

            scores = scores / (self.temp.abs() + 1e-6)
            scores = torch.clamp(scores, min=-50.0, max=50.0)
            attn = F.softmax(scores, dim=-1)   # (batch, seq_len)

            # Transform values with U
            V_flat = V.reshape(batch_seq, d, 1)
            U_flat = U.reshape(batch_seq, d, d)
            transformed_v = torch.bmm(U_flat, V_flat).reshape(batch, seq_len, d)

            out[:, i, :] = torch.sum(attn.unsqueeze(-1) * transformed_v, dim=1)

        out = self.w_o(out)
        out = out + x  # residual
        return out

# ------------------------------
# 3. Decoder‑Only Non‑Commutative Model
# ------------------------------
class DecoderOnlyNonCommutative(nn.Module):
    def __init__(self, vocab_size, d_model=16, hidden_dim=32, max_len=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.attn = CausalNonCommutativeAttention(d_model, hidden_dim)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_encoding[:, :seq_len, :]
        # Causal mask: upper triangular with -1e9
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1) * -1e9
        mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)
        x = self.attn(x, mask=mask)
        logits = self.head(x)
        return logits

# ------------------------------
# 4. Dummy Data: next‑integer prediction
# ------------------------------
class ArithmeticDataset(Dataset):
    def __init__(self, num_samples=1000, vocab_size=20, seq_len=8):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data = []
        for _ in range(num_samples):
            # Generate a sequence of integers with a pattern: start random, then increment by 1
            start = random.randint(0, vocab_size - seq_len - 2)  # ensure room
            seq = list(range(start, start + seq_len))
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # input: first seq_len-1 tokens, target: shifted by 1 (next token)
        return seq[:-1], seq[1:]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(inp) for inp in inputs)
    # Pad with 0 (assuming 0 is a valid token, but we'll treat 0 as pad)
    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        padded_targets.append(torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)]))
    return torch.stack(padded_inputs), torch.stack(padded_targets)

# ------------------------------
# 5. Training
# ------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * targets.numel()
        total_tokens += (targets != 0).sum().item()
    return total_loss / total_tokens

# ------------------------------
# 6. Generation (autoregressive)
# ------------------------------
@torch.no_grad()
def generate(model, start_tokens, max_new_tokens=5, temperature=0.8, device='cpu'):
    model.eval()
    input_ids = start_tokens.to(device)  # (1, seq_len)
    for _ in range(max_new_tokens):
        # If sequence grows beyond max_len, crop to last max_len tokens
        if input_ids.size(1) > model.max_len:
            input_ids = input_ids[:, -model.max_len:]
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    return input_ids

# ------------------------------
# 7. Main
# ------------------------------
if __name__ == '__main__':
    vocab_size = 30      # numbers 0..29
    seq_len = 8
    d_model = 16
    hidden_dim = 32
    batch_size = 32
    epochs = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    dataset = ArithmeticDataset(num_samples=2000, vocab_size=vocab_size, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = DecoderOnlyNonCommutative(vocab_size, d_model, hidden_dim, max_len=seq_len).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding (token 0)

    for epoch in range(1, epochs+1):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # Test generation
        start = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)  # known pattern
        gen = generate(model, start, max_new_tokens=5, temperature=0.8, device=device)
        print(f"Start: {start.tolist()} -> Generated: {gen.tolist()}")

