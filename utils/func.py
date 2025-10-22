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


def fetch_prices(ticker: str, start: str, end: str = None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
    df.dropna(inplace=True)
    return df


def build_features(df: pd.DataFrame):
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out["logret"] = np.log1p(out["ret"])

    for w in (5, 10, 20, 60):
        out[f"ma{w}"] = out["close"].rolling(w).mean()
        out[f"std{w}"] = out["ret"].rolling(w).std()
        out[f"mom{w}"] = out["close"].pct_change(w)

    out.dropna(inplace=True)
    return out


def make_supervised(
    df: pd.DataFrame, window: int, horizon: int, features: list, target: str
):
    X, y, idx = [], [], []
    values = df[features, target].values
    for i in range(window, len(values) - horizon + 1):
        X.append(values[i - window : i, :-1])
        y.append(values[i + horizon - 1, -1])
        idx.append(df.index[i + horizon - 1])

    X = np.array(X)
    y = np.array(y)
    idx = np.array(idx)

    return X, y, idx


class LSTMmodel(nn.Module):
    def __init__(self, n_features, hidden_lstm_units, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_lstm_units, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm_units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def train(
    model: nn.Module,
    X_tr: np.array,
    y_tr: np.array,
    val_split: int,
    epochs: int,
    batch_size: int,
    lr: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to_device(device)

    n = len(X_tr)
    n_val = int(n * val_split)
    X_train, y_train = X_tr[: n - n_val], y_tr[: n - n_val]
    X_val, y_val = X_tr[n - n_val :], y_tr[n - n_val :]

    ds_tr = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)
    )
    ds_val = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1)
    )
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(ds_tr)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * len(xb)
    val_loss /= len(ds_val)

    return model


def predict(model: nn.Module, X_te: np.array, y_te: np.array, device: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to_device(device)
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_te), torch.zeros((len(X_te), 1)))
    dl = DataLoader(ds, batch_size=1024, shuffle=False)
    out = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            pred = model(xb).squeeze(1).cpu().numpy()
            out.append(pred)
    return np.concatenate(out, axis=0)


def walk_forward_split(n: int, n_splits: int, test_ratio: float):
    test_len = int(n * test_ratio)
    train_start = 0
    for k in range(n_splits):
        train_end = int((n - test_len) * ((k + 1) / n_splits))
        test_start = train_end
        test_end = min(test_start + test_len, n)
        if test_end - test_start < 10:
            break
        yield np.arange(train_start, train_end), np.arange(test_start, test_end)


def backtest_daily_full_invest(
    returns: pd.Series,
    signal: pd.Series,
    fee_buy: float,
    fee_sell: float,
    initial: float,
):
    returns = returns.copy()
    signal = signal.reindex(returns.index).fillna(0).astype(int)
    value = initial
    curve = []
    for dt, (sig, r) in zip(returns.index, zip(signal.values, returns.values)):
        if sig == 1:
            value = value * (1 - fee_buy)
            value = value * (1 + r)
            value = value * (1 - fee_sell)
        else:
            value = value
        curve.append((dt, value))
    curve = pd.Series({dt: v for dt, v in curve}, name="portfolio_value")
    return curve


def buy_and_hold_dollars(
    returns: pd.Series, initial: float, fee_buy: float, fee_sell: float
):
    gross_curve = (1 + returns).cumprod()
    curve = gross_curve * (initial * (1 - fee_buy))

    if len(curve) > 0:
        curve.iloc[-1] = curve.iloc[-1] * (1 - fee_sell)
    return curve


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "DA": float((np.sign(y_true) == np.sign(y_pred)).mean()),  #
    }
