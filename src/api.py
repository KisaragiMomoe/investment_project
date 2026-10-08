"""
Day 34: FastAPI 服务
接收股票特征序列，返回预测收益率。
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

from predictor import StockPredictor
from dataset import FEATURE_COLS, N_FEATURES

app = FastAPI(title="股票收益率预测 API")

# 全局加载模型（只加载一次）
predictor = StockPredictor()

class PredictRequest(BaseModel):
    features: List[List[float]]

class PredictResponse(BaseModel):
    predicted_return: float
    signal: str

@app.get("/")
def root():
    return {"message": "股票收益率预测 API", "features": FEATURE_COLS}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    features = request.features

    # 校验形状
    if len(features) == 0 or len(features[0]) != N_FEATURES:
        return PredictResponse(
            predicted_return=0.0,
            signal=f"输入形状错误，需要 (seq_len, {N_FEATURES})",
        )

    pred = predictor.predict(features)

    if pred > 0.001:
        signal = "买入"
    elif pred < -0.001:
        signal = "卖出"
    else:
        signal = "观望"

    return PredictResponse(predicted_return=pred, signal=signal)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)