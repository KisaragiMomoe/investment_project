"""
Day 34: 特征工程
从原始 OHLCV 数据构造 30+ 个特征。
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def compute_rsi(series, period = 14):
    """RSI 相对强弱指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window = period).mean()
    avg_loss = gain.rolling(window = period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - 100 / (1 + rs)

def compute_macd(series, fast = 12, slow = 26, signal = 9):
    """MACD 指标"""
    ema_fast = series.ewm(span = fast, adjust = False).mean()
    ema_slow = series.ewm(span = slow, adjust = False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span = signal, adjust = False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_atr(high, low, close, period = 14):
    """ATR 平均真实波幅"""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
    return tr.rolling(window = period).mean()

def compute_bollinger(series, period = 20, num_std = 2):
    """布林带"""
    ma = series.rolling(window = period).mean()
    std = series.rolling(window = period).std()
    upper = ma + std * num_std
    lower = ma - std * num_std
    # 价格在布林带中的位置：0 = 下轨，1 = 上轨
    position = (series - lower) / (upper - lower + 1e-8)
    return upper, lower, position

def build_features(ticker = "0700.HK", start = "2020-01-01", end = "2024-12-31"):
    print(f"下载 {ticker} 数据...")
    df = yf.download(ticker, start = start, end = end, auto_adjust = True)
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns = {
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop = True)

    # ========== 1. 动量特征 ==========
    for n in [1, 3, 5, 10, 20, 60]:
        df[f"ret_{n}"] = df["close"].pct_change(n)
    # ========== 2. 波动率特征 ==========
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    # ========== 3. 均线特征 ==========
    for n in [5, 10, 20, 60]:
        df[f"ma_{n}"] = df["close"].rolling(n).mean()
        # 价格相对均线的偏离
        df[f"ma_dev_{n}"] = (df["close"] - df[f"ma_{n}"]) / (df[f"ma_{n}"] + 1e-8)
    # 均线交叉
    df["ma_5_20_cross"] = (df["ma_5"] - df["ma_20"]) / (df["ma_20"] + 1e-8)
    df["ma_20_60_cross"] = (df["ma_20"] - df["ma_60"]) / (df["ma_60"] + 1e-8)
    # ========== 4. 成交量特征 ==========
    df["volume_change_1"] = df["volume"].pct_change(1)
    df["volume_change_5"] = df["volume"].pct_change(5)
    df["volume_ma_5"] = df["volume"].rolling(5).mean()
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
    # RSI
    df["rsi_14"] = compute_rsi(df["close"], 14)
    # MACD
    macd_line, signal_line, histogram = compute_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    # ATR
    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_ratio"] = df["atr_14"] / (df["close"] + 1e-8)
    upper, lower, position = compute_bollinger(df["close"])
    # 布林带
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_position"] = position
    # 当日涨跌
    df["intraday_return"] = (df["close"] - df["open"]) / (df["open"] + 1e-8)
    # 最高最低价差
    df["high_low_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
    # 收盘价在当日区间的位置
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

    # ========== 7. 市场环境 ==========
    # 下载恒生指数作为大盘
    print("下载恒生指数...")
    hsi = yf.download("^HSI", start = start, end = end, auto_adjust = True)
    hsi = hsi.reset_index()
    if isinstance(hsi.columns, pd.MultiIndex):
        hsi.columns = hsi.columns.get_level_values(0)
    hsi = hsi.rename(columns={"Date": "date", "Close": "hsi_close"})
    hsi["date"] = pd.to_datetime(hsi["date"])
    hsi = hsi[["date", "hsi_close"]]

    # 合并
    df = df.merge(hsi, on="date", how="left")
    df["hsi_ret_1"] = df["hsi_close"].pct_change(1)
    df["hsi_ret_5"] = df["hsi_close"].pct_change(5)
    df["hsi_ret_20"] = df["hsi_close"].pct_change(20)

    # 相对强弱（个股相对大盘）
    df["relative_strength_5"] = df["ret_5"] - df["hsi_ret_5"]
    df["relative_strength_20"] = df["ret_20"] - df["hsi_ret_20"]

    # ========== 8. 目标 ==========
    df["target"] = df["close"].pct_change(1).shift(-1)

    # ========== 9. 清理 ==========
    df = df.dropna().reset_index(drop=True)

    # 删除不需要的原始列
    drop_cols = ["open", "high", "low", "volume", "hsi_close"]
    for n in [5, 10, 20, 60]:
        drop_cols.append(f"ma_{n}")
    drop_cols += ["volume_ma_5", "volume_ma_20", "atr_14", "bb_upper", "bb_lower", "macd_signal"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ========== 10. 模拟情感 ==========
    np.random.seed(42)
    df["sentiment"] = df["ret_1"] * 0.3 + np.random.randn(len(df)) * 0.2
    df["sentiment"] = df["sentiment"].clip(-1, 1)

    return df


if __name__ == "__main__":
    df = build_features()
    print(f"\n数据量：{len(df)} 条")
    print(f"特征数：{len(df.columns) - 2}")   # 减去 date 和 target

    print(f"\n所有列：")
    for col in df.columns:
        print(f"  {col}")

    save_path = os.path.join(RAW_DATA_DIR, "stock_data_v2.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n数据已保存到：{save_path}")
