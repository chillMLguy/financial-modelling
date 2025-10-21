import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    df: pd.DataFrame, window: int, horizon: int, features: list, target: str = "ret"
):
    X, y, idx = [], [], []
    values = df[features + [target]].values
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


def train():
    pass


def predict():
    pass


def walk_foward_split():
    pass


def evaluate_strategy():
    pass


def backtest_strategy():
    pass


def buy_and_hold_benchmark():
    pass


def metrics():
    pass


def run_pipeline():
    pass
