"""
Day 32: 数据集构造
把时间序列切成 (seq_len, n_features) 的样本，按时间划分训练/测试集。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

FEATURE_COLS = ["ret_1", "ret_5", "ret_20", "vol_5", "vol_20", "sentiment"]

class StockDataset(Dataset):
    """股票序列数据集"""
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_and_split(seq_len = 20, train_ratio = 0.8):
    """加载数据，构造序列，按时间切分，归一化"""
    # 1. 加载 CSV
    csv_path = os.path.join(RAW_DATA_DIR, "stock_data.csv")
    df = pd.read_csv(csv_path, encoding = "utf-8-sig")
    print(f"加载数据：{len(df)} 条")

    # 2. 提取特征和目标
    x_raw = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df["target"].values.astype(np.float32)

    # 3. 构造序列：用前 seq_len 天预测第 seq_len+1 天
    x_seq, y_seq = [], []
    for i in range(len(x_raw) - seq_len):
        x_seq.append(x_raw[i : i + seq_len])
        y_seq.append(y_raw[i + seq_len])
    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq).reshape(-1, 1)
    print(f"序列样本数：{len(x_seq)}")
    print(f"每个样本形状：{x_seq[0].shape}")

    # 4. 按时间切分
    split = int(len(x_seq) * train_ratio)
    x_train, x_test = x_seq[:split], x_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    print(f"训练集：{len(x_train)} 条")
    print(f"测试集：{len(x_test)} 条")

    # 5. 归一化（只用训练集统计量）
    # 对每个特征，算训练集在所有时间步上的均值和标准差
    x_mean = x_train.mean(axis = (0, 1), keepdims = True)
    x_std = x_train.std(axis = (0, 1), keepdims = True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    print(f"\n归一化后的特征统计（训练集）：")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col}: mean={x_train[:, :, i].mean():.4f}, std={x_train[:, :, i].std():.4f}")

    return (x_train, y_train, x_test, y_test), (x_mean, x_std)

def get_dataloaders(batch_size = 32):
    """返回训练和测试的 DataLoader"""
    (x_train, y_train, x_test, y_test), stats = load_and_split()
    train_dataset = StockDataset(x_train, y_train)
    test_dataset = StockDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader, stats

if __name__ == "__main__":
    train_loader, test_loader, stats = get_dataloaders()
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n训练 batch 形状：X={x_batch.shape}, y={y_batch.shape}")
