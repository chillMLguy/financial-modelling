import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.func import fetch_prices, build_features

start_date = "2015-01-01"
end_date = None

assets = ["^GSPC"]

capital = 10000
test_size = 0.2
window = 60
horizon = 1

fee_buy = 0.0005
fee_sel = 0.0005

seed = 420
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)






data = fetch_prices(assets[0], start_date, end_date)
data = build_features(data)

print(np.shape(data))

