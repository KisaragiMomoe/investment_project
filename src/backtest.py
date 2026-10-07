"""
Day 33: 回测脚本
加载训练好的模型，在测试集上做回测，扣手续费，对比买入持有。
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dataset import load_and_split, FEATURE_COLS
from model import StockTransformer

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# ========== 1. 设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# ========== 2. 加载数据 ==========
(x_train, y_train, x_test, y_test), (x_mean, x_std) = load_and_split()
print(f"测试集样本数：{len(x_test)}")

model = StockTransformer(
    n_features = 6,
    d_model = 64,
    num_heads = 4,
    num_layers = 3,
    dim_feedforward = 256,
    dropout = 0.1,
).to(device)

checkpoint = torch.load(os.path.join(MODEL_DIR, "stock_transformer.pth"), map_location = device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("模型加载完成")

x_test_t = torch.tensor(x_test, dtype = torch.float32).to(device)
with torch.no_grad():
    y_pred = model(x_test_t).cpu().numpy().flatten()
y_test_real = y_test.flatten()

print(f"\n预测值统计：")
print(f"  均值：{y_pred.mean():.6f}")
print(f"  标准差：{y_pred.std():.6f}")
print(f"  最小值：{y_pred.min():.6f}")
print(f"  最大值：{y_pred.max():.6f}")

print(f"\n真实值统计：")
print(f"  均值：{y_test_real.mean():.6f}")
print(f"  标准差：{y_test_real.std():.6f}")

ic = np.corrcoef(y_pred, y_test_real)[0, 1]
print(f"\nIC（信息系数）：{ic:.4f}")

n_groups = 5
sorted_idx = np.argsort(y_pred)
group_size = len(sorted_idx) // n_groups
print(f"\n按预测值分组（每组 {group_size} 条）：")
for g in range(n_groups):
    start = g * group_size
    end = start + group_size if g < n_groups - 1 else len(sorted_idx)
    group_idx = sorted_idx[start:end]
    avg_return = y_test_real[group_idx].mean()
    print(f"  第 {g+1} 组（预测值{'最低' if g == 0 else '最高' if g == n_groups-1 else '中间'}）：平均真实收益 = {avg_return*100:.4f}%")

signal = (y_pred > 0).astype(int)
threshold = np.percentile(y_pred, 80)
signal_top20 = (y_pred > threshold).astype(int)
signal_reverse = (y_pred < 0).astype(int)

def compute_returns(signal, y_true, cost = 0.001):
    strategy_ret = signal * y_true
    position_change = np.abs(np.diff(signal, prepend = 0))
    cost_array = position_change * cost
    strategy_ret_after_cost = strategy_ret - cost_array
    cum_ret = np.cumprod(1 + strategy_ret_after_cost) - 1
    return cum_ret, strategy_ret_after_cost

cum_buy_hold = np.cumprod(1 + y_test_real) - 1

cum_strategy, ret_strategy = compute_returns(signal, y_test_real)
cum_top20, ret_top20 = compute_returns(signal_top20, y_test_real)
cum_reverse, ret_reverse = compute_returns(signal_reverse, y_test_real)

# ========== 8. 绩效指标 ==========
def compute_metrics(cum_ret, ret_series):
    """计算绩效指标"""
    total_return = cum_ret[-1]
    # 年化收益（假设一年 252 个交易日）
    n_days = len(ret_series)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    # 夏普比率（假设无风险利率为 0）
    sharpe = ret_series.mean() / (ret_series.std() + 1e-8) * np.sqrt(252)
    # 最大回撤
    cum_max = np.maximum.accumulate(cum_ret + 1)
    drawdown = (cum_ret + 1) / cum_max - 1
    max_drawdown = drawdown.min()
    # 胜率
    win_rate = (ret_series > 0).mean()
    return {
        "总收益": total_return,
        "年化收益": annual_return,
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,
        "胜率": win_rate,
    }


print("\n" + "=" * 60)
print("绩效指标对比：")
print("=" * 60)

metrics_bh = compute_metrics(cum_buy_hold, y_test_real)
metrics_strategy = compute_metrics(cum_strategy, ret_strategy)
metrics_top20 = compute_metrics(cum_top20, ret_top20)
metrics_reverse = compute_metrics(cum_reverse, ret_reverse)

print(f"\n{'指标':<12} {'买入持有':<12} {'策略(>0)':<12} {'Top20%':<12} {'反向':<12}")
print("-" * 60)
for key in ["总收益", "年化收益", "夏普比率", "最大回撤", "胜率"]:
    print(f"{key:<12} {metrics_bh[key]:<12.4f} {metrics_strategy[key]:<12.4f} {metrics_top20[key]:<12.4f} {metrics_reverse[key]:<12.4f}")

# ========== 9. 画图 ==========
plt.figure(figsize=(14, 10))

# 子图1：累计收益对比
plt.subplot(2, 2, 1)
plt.plot(cum_buy_hold * 100, label="买入持有", color="gray", linewidth=2)
plt.plot(cum_strategy * 100, label="策略(预测>0)", color="green", linewidth=2)
plt.plot(cum_top20 * 100, label="Top20%", color="blue", linewidth=2)
plt.plot(cum_reverse * 100, label="反向策略", color="red", linewidth=2, alpha=0.6)
plt.title("累计收益对比（扣手续费后）")
plt.xlabel("交易日")
plt.ylabel("累计收益 (%)")
plt.legend()
plt.grid(True)

# 子图2：预测值 vs 真实值散点
plt.subplot(2, 2, 2)
plt.scatter(y_test_real, y_pred, alpha=0.3, color="green")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title(f"预测值 vs 真实值（IC={ic:.4f}）")
plt.xlabel("真实收益率")
plt.ylabel("预测收益率")
plt.grid(True)

# 子图3：分组平均收益
plt.subplot(2, 2, 3)
group_returns = []
for g in range(n_groups):
    start = g * group_size
    end = start + group_size if g < n_groups - 1 else len(sorted_idx)
    group_idx = sorted_idx[start:end]
    group_returns.append(y_test_real[group_idx].mean() * 100)
plt.bar(range(1, n_groups + 1), group_returns, color="steelblue")
plt.title("按预测值分组的平均真实收益")
plt.xlabel("分组（1=预测最低，5=预测最高）")
plt.ylabel("平均真实收益 (%)")
plt.grid(True, axis="y")

# 子图4：回撤曲线
plt.subplot(2, 2, 4)
cum_max = np.maximum.accumulate(cum_strategy + 1)
drawdown = ((cum_strategy + 1) / cum_max - 1) * 100
plt.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.3)
plt.title("策略回撤曲线")
plt.xlabel("交易日")
plt.ylabel("回撤 (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "day33_backtest.png"))
plt.show()

print("\n图片已保存")
