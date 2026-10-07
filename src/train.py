"""
Day 32: 训练脚本
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import StockTransformer

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

train_loader, test_loader, stats = get_dataloaders(batch_size = 32)

model = StockTransformer(
    n_features = 6,
    d_model = 64,
    num_heads = 4,
    num_layers = 3,
    dim_feedforward = 256,
    dropout = 0.1,
).to(device)

print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.01)

epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        train_loss += loss.item() * len(x_batch)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            test_loss += loss.item() * len(x_batch)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    if (epoch % 10 == 0):
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

# 算IC
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch).cpu().numpy().flatten()
        all_preds.extend(y_pred)
        all_targets.extend(y_batch.numpy().flatten())
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
ic = np.corrcoef(all_preds, all_targets)[0, 1]
print(f"\n测试集 IC（信息系数）：{ic:.4f}")
print("（IC > 0.05 算不错，> 0.1 很强）")

# ========== 7. 保存模型 ==========
save_path = os.path.join(MODEL_DIR, "stock_transformer.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "stats": {"X_mean": stats[0].tolist(), "X_std": stats[1].tolist()},
}, save_path)
print(f"\n模型已保存到：{save_path}")

# ========== 8. 画图 ==========
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="训练 Loss", color="blue")
plt.plot(test_losses, label="测试 Loss", color="red")
plt.title("训练与测试 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(all_targets, all_preds, alpha=0.3, color="green")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title(f"预测值 vs 真实值（IC={ic:.4f}）")
plt.xlabel("真实收益率")
plt.ylabel("预测收益率")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "day32_training.png"))
plt.show()

print("\n图片已保存")