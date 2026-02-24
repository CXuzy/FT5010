import pandas as pd
import pandas_datareader.data as web

START = "2015-01-01"
END   = "2019-12-31"

# Stooq tickers
tickers = {
    "META": "META.US",     # Meta (covers history; use META instead of FB)
    "AAPL": "AAPL.US",
    "AMZN": "AMZN.US",
    "NFLX": "NFLX.US",
    "GOOGL": "GOOGL.US"
}

def fetch_stooq(symbol: str, start: str, end: str) -> pd.Series:
    df = web.DataReader(symbol, "stooq", start=start, end=end)
    # stooq returns descending dates; sort ascending
    df = df.sort_index()
    # use Close price
    return df["Close"].rename(symbol)

all_close = []
for name, sym in tickers.items():
    print(f"Downloading {name} from Stooq ({sym})...")
    s = fetch_stooq(sym, START, END)
    s.name = name
    all_close.append(s)

prices = pd.concat(all_close, axis=1).dropna(how="all").ffill()

# 保存本地
out = "MAANG_2015_2019_close_stooq.csv"
prices.to_csv(out)

print(f"\nSaved: {out}")
print(prices.head())
print(prices.tail())