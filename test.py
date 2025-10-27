import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.func import (
    fetch_prices,
    build_features,
    make_supervised,
    LSTMmodel,
    train,
    predict,
    walk_forward_split,
    backtest_daily_full_invest,
    buy_and_hold_dollars,
    metrics,
)

start_date = "2015-01-01"
end_date = None

assets = ["^GSPC"]

capital = 10000
test_size = 0.2
window = 60
horizon = 1
splits = 3
sgn_threshold = 0

fee_buy = 0.0005
fee_sel = 0.0005

lstm_neurons = 64
dropout = 0.2
epochs = 40
batch_size = 64
lr = 0.001


seed = 420
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("==========Testing strategy==============")

data = fetch_prices(assets[0], start_date, end_date)
data = build_features(data)

feature_cols = [c for c in data.columns if (c not in [("ret", "")])]

preds = pd.Series(index=data.index, dtype=float)


features_df = data.copy()
full_features = features_df[feature_cols]
target = features_df["ret"]

curves = []
fold_metrics = []

X_all, y_all, idx_all = make_supervised(
    features_df, features=feature_cols, target="ret", window=window, horizon=horizon
)
print(idx_all)
print(y_all)
