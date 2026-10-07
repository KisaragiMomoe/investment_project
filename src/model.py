"""
Day 32: 股票预测 Transformer 模型
复用 Day 18 的组件，输入 6 维特征，输出 1 维收益率预测。
"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim = -1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward = 256, dropout = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask = None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class StockTransformer(nn.Module):
    """股票收益率预测模型"""
    def __init__(self, n_features = 6, d_model = 64, num_heads = 4, num_layers = 3, dim_feedforward = 256, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        x = x[:, -1, :]
        return self.output_proj(x)

if __name__ == "__main__":
    model = StockTransformer(n_features = 6, d_model = 64, num_heads = 4, num_layers = 3)
    x = torch.randn(2, 20, 6)
    y = model(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{y.shape}")
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")