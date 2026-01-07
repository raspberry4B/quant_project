from __future__ import annotations

from pathlib import Path
import pandas as pd

from data_fetch import get_raw_with_cache, fetch_etf_daily_em


CANDIDATES = {
    "date":   ["日期", "交易日期", "时间"],
    "open":   ["开盘"],
    "high":   ["最高"],
    "low":    ["最低"],
    "close":  ["收盘", "最新价"],
    "volume": ["成交量", "成交量(手)", "成交量（手）", "成交量（股）"],
}


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for std, cand in CANDIDATES.items():
        col = pick_col(raw, cand)
        if col is None:
            raise ValueError(f"Missing column for '{std}'. candidates={cand}. got={list(raw.columns)}")
        mapping[col] = std

    df = raw.rename(columns=mapping).copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").set_index("date")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    # Backtrader 典型日线输入
    return df[["open", "high", "low", "close", "volume"]]


def prep_etf_processed(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> Path:
    key = f"ak_etf_em_daily_{symbol}_{start_date}_{end_date}_adj_{adjust}"
    raw = get_raw_with_cache(
        key=key,
        fetch_fn=lambda: fetch_etf_daily_em(symbol, start_date, end_date, adjust=adjust),
        ttl_days=3,
    )

    df = normalize_ohlcv(raw)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"etf_{symbol}_daily_{start_date}_{end_date}_adj_{adjust}.csv"
    df.to_csv(out_path, encoding="utf-8-sig")
    return out_path


if __name__ == "__main__":
    p = prep_etf_processed("510300", "20150101", "20251231", adjust="qfq")
    print("[INFO] saved processed:", p)

    df = pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8-sig")
    print("[INFO] processed head:")
    print(df.head())
    print("[INFO] processed columns:", list(df.columns))
    print("[INFO] rows:", len(df))
