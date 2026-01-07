from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd
import akshare as ak


def _cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.csv"


def _is_cache_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime <= timedelta(days=ttl_days)


def retry_call(fn: Callable[[], pd.DataFrame], max_tries: int = 5, base_sleep: float = 1.0) -> pd.DataFrame:
    last_err: Exception | None = None
    for i in range(max_tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** i)
            print(f"[WARN] fetch failed (try {i+1}/{max_tries}): {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def fetch_etf_daily_em(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    # 东方财富 ETF 历史行情（日线）
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,  # "qfq" / "hfq" / ""(不复权)
    )
    return df


def get_raw_with_cache(
    key: str,
    fetch_fn: Callable[[], pd.DataFrame],
    cache_dir: str = "data/cache",
    ttl_days: int = 3,
) -> pd.DataFrame:
    cache_dir_p = Path(cache_dir)
    p = _cache_path(cache_dir_p, key)

    # 1) 优先使用新鲜缓存
    if _is_cache_fresh(p, ttl_days):
        df = pd.read_csv(p, encoding="utf-8-sig")
        print(f"[INFO] cache hit: {p}")
        return df

    # 2) 拉取 + 重试
    try:
        df = retry_call(fetch_fn, max_tries=5, base_sleep=1.0)
        if df is None or len(df) == 0:
            raise ValueError("empty dataframe")
        df.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[INFO] cache write: {p}")
        return df
    except Exception as e:
        # 3) 降级：如果有旧缓存，仍可用
        if p.exists():
            print(f"[WARN] fetch failed, fallback to stale cache: {p.name}. err={e}")
            return pd.read_csv(p, encoding="utf-8-sig")
        raise


if __name__ == "__main__":
    symbol = "510300"
    start_date = "20150101"
    end_date = "20251231"
    adjust = "qfq"

    key = f"ak_etf_em_daily_{symbol}_{start_date}_{end_date}_adj_{adjust}"
    raw = get_raw_with_cache(
        key=key,
        fetch_fn=lambda: fetch_etf_daily_em(symbol, start_date, end_date, adjust=adjust),
        ttl_days=3,
    )

    print("[INFO] raw head:")
    print(raw.head())
    print("[INFO] raw columns:")
    print(list(raw.columns))
    print(f"[INFO] rows: {len(raw)}")
