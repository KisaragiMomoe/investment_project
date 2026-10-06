# 金融新闻情感分析 + Transformer 股票预测

## 项目简介
用 FinBERT 对金融新闻做情感分析，结合技术指标，用 Transformer 预测次日收益率。

## 环境
- Python 3.10+
- PyTorch, Transformers, yfinance, pandas, numpy

## 如何运行
1. 创建虚拟环境：`python -m venv .venv`
2. 激活：`.venv\Scripts\activate`
3. 安装依赖：`pip install -r requirements.txt`
4. 运行数据准备：`python src/data_prep.py`

## 项目结构
- `src/`：核心代码
- `data/raw/`：原始数据
- `data/processed/`：处理后数据
- `models/`：训练好的模型