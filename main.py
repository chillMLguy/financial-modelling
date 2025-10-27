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
sgn_threshold = 0.005

fee_buy = 0
fee_sel = 0

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

feature_cols = [c for c in data.columns if c not in ("ret",)]

preds = pd.Series(index=data.index, dtype=float)


features_df = data.copy()
full_features = features_df[feature_cols]
target = features_df["ret"]

curves = []
fold_metrics = []

X_all, y_all, idx_all = make_supervised(
    features_df, features=feature_cols, target="ret", window=window, horizon=horizon
)

seq_df = pd.DataFrame(index=idx_all)
seq_df["y"] = y_all
n = len(seq_df)

for fold, (tr_idx, te_idx) in enumerate(
    walk_forward_split(n, splits, test_size), start=1
):
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]
    idx_tr, idx_te = idx_all[tr_idx], idx_all[te_idx]

    s = X_tr.shape
    scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(s[0], s[1] * s[2])
    X_te_2d = X_te.reshape(X_te.shape[0], X_te.shape[1] * X_te.shape[2])
    scaler.fit(X_tr_2d)
    X_tr = scaler.transform(X_tr_2d).reshape(s)
    X_te = scaler.transform(X_te_2d).reshape(
        X_te.shape[0], X_te.shape[1], X_te.shape[2]
    )

    model = LSTMmodel(
        n_features=X_tr.shape[2], hidden_lstm_units=lstm_neurons, dropout=dropout
    )
    model = train(
        model,
        X_tr.astype(np.float32),
        y_tr.astype(np.float32),
        val_split=test_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    y_hat = predict(
        model, X_te.astype(np.float32), y_te.astype(np.float32), device="cuda"
    )
    fold_m = metrics(y_te, y_hat)
    fold_metrics.append({"fold": fold, **fold_m})

    print(f"Fold {fold}: RMSE={fold_m['RMSE']} MAE={fold_m['MAE']} DA={fold_m['DA']}")

    preds.loc[idx_te] = y_hat

thr = float(sgn_threshold)
signal = pd.Series(0, index=data.index, dtype=int)
signal.loc[preds.index] = np.where(preds > thr, 1, np.where(preds < -thr, -1, 0))

bh_curve = buy_and_hold_dollars(
    data["ret"], initial=capital, fee_buy=fee_buy, fee_sell=fee_sel
)
lstm_curve = backtest_daily_full_invest(
    data["ret"], signal, fee_buy=fee_buy, fee_sell=fee_sel, initial=capital
)

plt.figure(figsize=(10, 5))
bh_curve.plot(label="Buy&Hold ($)")
lstm_curve.plot(label="LSTM daily full‑invest ($)")
plt.title(f"S&P500: LSTM vs Buy&Hold — start ${capital}, fee={fee_buy}")
plt.ylabel("Portfolio value ($)")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
(bh_curve / bh_curve.iloc[0]).plot(label="Buy&Hold")
(lstm_curve / lstm_curve.iloc[0]).plot(label="LSTM strategy")
plt.title(f"S&P500: LSTM vs Buy&Hold (okno={window}, fee={fee_buy})")
plt.legend()
plt.tight_layout()
plt.show()
