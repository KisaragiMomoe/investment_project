import requests
import numpy as np
from dataset import N_FEATURES

# 构造假输入
features = np.random.randn(20, N_FEATURES).tolist()

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": features}
)

print(response.json())