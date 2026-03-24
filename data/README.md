# Data

Stock data files are not included in this repository for copyright reasons.

## Required Files

| File | Description |
| --- | --- |
| `start_dates.xlsx` | Two columns: `Code` and `Start date` |
| `<ticker>.csv` | Daily OHLCV data for each ticker listed in `start_dates.xlsx` |

## CSV Format

Each CSV should include at least:

```text
Date,Open,High,Low,Close,Volume
```

Yahoo Finance exports work with this format.

## Downloading Data

Example with `yfinance`:

```python
import pandas as pd
import yfinance as yf

tickers = pd.read_excel("start_dates.xlsx")["Code"].tolist()
for ticker in tickers:
    yf.download(ticker, start="2010-01-01").to_csv(f"{ticker}.csv")
```

## Dataset Scope

The study focuses on EV-related stocks listed in the US, Hong Kong,
Shanghai, and Shenzhen markets, with a train/test split at `2024-01-01`.
