"""
Day 34: 推理封装
加载模型，接收特征序列，返回预测收益率。
"""

import os
import numpy as np
import torch

from dataset import FEATURE_COLS, N_FEATURES
from model import StockTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

class StockPredictor:
    def __init__(self, model_path = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (model_path == None):
            model_path = os.path.join(MODEL_DIR, "stock_transformer.pth")
        #加载模型
        self.model = StockTransformer(
            n_features = N_FEATURES,
            d_model = 128,
            num_heads = 8,
            num_layers = 4,
            dim_feedforward = 512,
            dropout = 0.1
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location = self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.x_mean = np.array(checkpoint["stats"]["x_mean"])
        self.x_std = np.array(checkpoint["stats"]["x_std"])
        print(f"模型加载完成，设备：{self.device}")

    def predict(self, features):
        """
        输入：features，形状 (seq_len, n_features) 或 (1, seq_len, n_features)
        输出：预测收益率（float）
        """
        features = np.array(features, dtype = np.float32)
        if (features.ndim == 2):
            features = features[np.newaxis, :, :]
        features = (features - self.x_mean) / self.x_std
        x = torch.tensor(features, dtype = torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(x).cpu().numpy().flatten()
        return float(y_pred[0])

    def predict_batch(self, features_list):
        features = np.array(features_list, dtype = np.float32)
        features = (features - self.x_mean) / self.x_std
        x = torch.tensor(features, dtype = np.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(x).cpu().numpy().flatten()
        return y_pred.tolist()
if __name__ == "__main__":
    # 测试
    predictor = StockPredictor()

    # 构造一个假的输入：20 天 × 32 特征
    fake_input = np.random.randn(20, N_FEATURES).astype(np.float32)
    pred = predictor.predict(fake_input)
    print(f"预测收益率：{pred:.6f}")