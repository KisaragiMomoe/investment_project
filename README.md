# 金融新闻情感分析 + Transformer 股票收益率预测

## 项目简介
用 FinBERT 对金融新闻做情感分析，结合 32 维技术指标，用 Transformer 预测股票次日收益率，并部署为 REST API。

## 技术栈
- Python 3.10+
- PyTorch, Transformers, FinBERT
- pandas, numpy, yfinance
- FastAPI, uvicorn
- matplotlib

## 项目结构
```
investment_project/
├── data/
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后数据
├── src/
│   ├── data_prep.py            # 数据准备 + 情感分析
│   ├── feature_engineering.py  # 32 维特征工程
│   ├── dataset.py              # 数据集构造
│   ├── model.py                # Transformer 模型
│   ├── train.py                # 训练脚本
│   ├── backtest.py             # 回测与评估
│   ├── predictor.py            # 推理封装
│   └── api.py                  # FastAPI 服务
├── models/                     # 训练好的模型
├── requirements.txt
└── README.md
```

## 安装
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 使用
```bash
cd src

# 1. 数据准备
python data_prep.py
python feature_engineering.py

# 2. 训练
python train.py

# 3. 回测
python backtest.py

# 4. 启动 API
python api.py
```

## 特征（32 个）
- 动量：ret_1/3/5/10/20/60
- 波动率：vol_5/10/20
- 均线：ma_dev_5/10/20/60、ma_5_20_cross、ma_20_60_cross
- 成交量：volume_change_1/5、volume_ratio
- 技术指标：RSI、MACD、MACD_hist、ATR_ratio、布林带位置
- 价格形态：intraday_return、high_low_range、close_position
- 市场环境：hsi_ret_1/5/20、relative_strength_5/20
- 情感：FinBERT sentiment

## 模型
- 4 层 Transformer Encoder
- d_model=128, num_heads=8, dim_feedforward=512
- 因果 Mask + 残差连接 + LayerNorm
- 参数量约 80 万

## 评估指标
- IC（信息系数）
- 分组分析
- 夏普比率、最大回撤、胜率
- 对比买入持有基准

## API 示例
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [[...], ...]}'
```

返回：
```json
{"predicted_return": 0.003, "signal": "买入"}
```

## 结果
- 测试集 IC：-0.014（接近 0）
- 结论：股票预测信噪比极低，IC 接近 0 是常态。

## 收获
- 掌握了量化研究的完整流程：数据 → 特征 → 模型 → 回测 → 部署
- 理解了前视偏差、IC、夏普比率、最大回撤等核心概念
- 学会了用 FastAPI 部署模型