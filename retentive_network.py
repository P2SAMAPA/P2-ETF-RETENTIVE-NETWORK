import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Retention(nn.Module):
    """
    Multi-scale retention mechanism (simplified version).
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # Scale factors for each head
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        batch, seq_len, _ = x.shape
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Retention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.out_proj(out)

class RetNetBlock(nn.Module):
    """RetNet block with retention and feed-forward."""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.retention = Retention(hidden_size, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.retention(x)
        x = self.dropout(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        return x

class RetNet(nn.Module):
    """Retentive Network for ETF prediction."""
    def __init__(self, input_size, hidden_size=64, num_heads=4, num_layers=2, dropout=0.1, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            RetNetBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        # Use the last time step for prediction
        out = self.output_proj(x[:, -1, :])
        return out.squeeze(-1)

def prepare_data(returns, macro_df, seq_len=10):
    """
    Prepare sequences for training.
    returns: pandas Series (single ETF)
    macro_df: pandas DataFrame (macro variables)
    """
    if len(returns) < seq_len + 1:
        return None, None
    # Align with macro
    common_idx = returns.index.intersection(macro_df.index)
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    # Create sequences
    X, y = [], []
    for i in range(seq_len, len(ret_aligned)):
        # Features: lagged returns + lagged macro
        ret_seq = ret_aligned.iloc[i-seq_len:i].values.reshape(-1, 1)  # (seq_len, 1)
        macro_seq = macro_aligned.iloc[i-seq_len:i].values              # (seq_len, n_macros)
        # Concatenate along feature dimension
        seq_features = np.concatenate([ret_seq, macro_seq], axis=1)      # (seq_len, 1 + n_macros)
        X.append(seq_features)
        y.append(ret_aligned.iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def retnet_score(returns, macro_df, hidden_size=64, num_heads=4, num_layers=2, dropout=0.1, seq_len=10, epochs=30, lr=0.001, batch_size=16):
    """
    Train RetNet and return predicted next-day return.
    """
    X, y = prepare_data(returns, macro_df, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetNet(input_size, hidden_size, num_heads, num_layers, dropout, seq_len).to(device)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    # Predict next day
    model.eval()
    with torch.no_grad():
        # Use the last sequence (most recent seq_len days)
        ret_seq = returns.iloc[-seq_len:].values.reshape(-1, 1)
        macro_seq = macro_df.iloc[-seq_len:].values
        last_seq = np.concatenate([ret_seq, macro_seq], axis=1)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(last_seq).item()
    return float(pred)
