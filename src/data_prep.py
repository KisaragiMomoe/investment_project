"""
Day 31: 数据准备脚本
下载股票数据，构造技术指标，用 FinBERT 做情感分析，保存成 CSV。
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========== 1. 配置路径 ==========
# 项目根目录（当前文件的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

print(f"项目根目录：{PROJECT_ROOT}")
print(f"数据保存目录：{RAW_DATA_DIR}")

# ========== 2. 下载股票数据 ==========
print("\n正在下载股票数据...")
df = yf.download("0700.HK", start="2020-01-01", end="2024-12-31", auto_adjust=True)
df = df.reset_index()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.rename(columns = {"Date": "date", "Close": "close", "Volume": "volume"})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop = True)

print(f"数据量：{len(df)} 条")
print(f"时间范围：{df['date'].iloc[0].date()} 到 {df['date'].iloc[-1].date()}")
print(f"列名：{df.columns.tolist()}")

# ========== 3. 构造技术指标 ==========
df["ret_1"] = df["close"].pct_change(1)
df["ret_5"] = df["close"].pct_change(5)
df["ret_20"] = df["close"].pct_change(20)
df["vol_5"] = df["ret_1"].rolling(5).std()
df["vol_20"] = df["ret_1"].rolling(20).std()
df["volume_change"] = df["volume"].pct_change(5)
df["target"] = df["close"].pct_change(1).shift(-1)
df = df.dropna().reset_index(drop = True)

print(f"\n构造指标后数据量：{len(df)} 条")

# ========== 4. 加载 FinBERT ==========

print("\n正在加载 FinBERT 情感分析模型...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model = sentiment_model.to(device)
sentiment_model.eval()
print(f"使用设备：{device}")

# ========== 5. 情感分析函数 ==========

def get_sentiment(text):
    inputs = tokenizer(text, truncation = True, max_length = 128).to(device)
    inputs = {
        k: torch.tensor(v).unsqueeze(0).to(device)
        for k, v in inputs.items()
    }
    print(f"inputs 类型：{type(inputs)}")
    print(f"input_ids 类型：{type(inputs['input_ids'])}")
    print(f"input_ids 值：{inputs['input_ids']}")
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim = -1)
    neg, neu, pos = probs[0].tolist()
    return pos - neg

# 测试

test_news = [
    "Tencent reports record quarterly earnings, beating expectations",
    "Tencent faces lawsuit over antitrust violations",
    "Tencent announces new game with minor updates",
]
print("\n情感分析测试：")
for news in test_news:
    score = get_sentiment(news)
    label = "利好" if score > 0.1 else ("利空" if score < -0.1 else "中性")
    print(f"  {news[:50]}... → 分数：{score:.4f}（{label}）")

# ========== 6. 模拟每日情感 ==========

print("\n模拟每日新闻情感...")
np.random.seed(42)
df["sentiment"] = df["ret_1"] * 0.3 + np.random.randn(len(df)) * 0.2
df["sentiment"] = df["sentiment"].clip(-1, 1)

print(f"情感分数统计：")
print(f"  均值：{df['sentiment'].mean():.4f}")
print(f"  标准差：{df['sentiment'].std():.4f}")

# ========== 7. 保存数据 ==========

save_path = os.path.join(RAW_DATA_DIR, "stock_data.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n数据已保存到：{save_path}")

# ========== 8. 数据概览 ==========

print("\n" + "=" * 50)
print("数据概览：")
print("=" * 50)
print(df[["date", "close", "ret_1", "sentiment", "target"]].head(10))

feature_cols = ["ret_1", "ret_5", "ret_20", "vol_5", "vol_20", "sentiment"]
print(f"\n特征列统计：")
for col in feature_cols:
    print(f"  {col}: 均值={df[col].mean():.6f}, 标准差={df[col].std():.6f}")

print(f"\n目标列 target：均值={df['target'].mean():.6f}, 标准差={df['target'].std():.6f}")
