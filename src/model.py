#model.py
# This module defines the LSTM model for stock price prediction.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // num_heads
        assert self.head_dim * num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, lstm_outputs):
        batch_size, seq_len, hidden_size = lstm_outputs.shape
        
        # Linear transformations
        Q = self.query(lstm_outputs)  # (batch, seq_len, hidden_size)
        K = self.key(lstm_outputs)    # (batch, seq_len, hidden_size)
        V = self.value(lstm_outputs)  # (batch, seq_len, hidden_size)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        context = self.fc_out(context)
        
        # Global average pooling over sequence dimension
        context = torch.mean(context, dim=1)  # (batch, hidden_size)
        
        return context, attn_weights

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=4, dropout=0.2, output_size=1, use_attention=False, num_heads=8):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # Always unidirectional for real-world forecasting
        )
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads)
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.use_attention:
            context, attn_weights = self.attention(out)
            out = self.fc(context)
            return out
        else:
            out = self.fc(out[:, -1])
            return out
