from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import akshare as ak

ETF_SYMBOL = "510300"
START = "20150101"
END = "20251231"
ADJUST = "qfq"
OUT_DIR = Path("data/warehouse/prices") / ADJUST

def fetch_with_retry(max_tries=6, base_sleep=1.0):
    last = None
    for i in range(max_tries):
        try:
            df = ak.fund_etf_hist_em(
                symbol=ETF_SYMBOL,
                period="daily",
                start_date=START,
                end_date=END,
                adjust=ADJUST
            )
            if df is None or len(df) == 0:
                raise ValueError("empty dataframe")
            return df
        except Exception as e:
            last = e
            sleep = base_sleep * (2 ** i)
            print(f"[WARN] fetch failed try={i+1}/{max_tries}: {e} sleep={sleep:.1f}s")
            time.sleep(sleep)
    raise last

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{ETF_SYMBOL}.csv"

    df = fetch_with_retry()

    df = df.rename(columns={"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date","close"])

    df[["date","open","high","low","close","volume"]].to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[DONE] saved:", out_path.resolve(), "rows=", len(df))

if __name__ == "__main__":
    main()
