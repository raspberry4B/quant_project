from __future__ import annotations

import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak


START = "20150101"
END = "20251231"
ADJUST = "qfq"
MAX_WORKERS = 8        # 并发数：建议 4~12，过高可能被源站限流
SLEEP_EACH = 0.2       # 轻度限速，避免请求过猛
OUT_DIR = Path("data/warehouse/prices") / ADJUST


def get_hs300_codes() -> list[str]:
    cons = ak.index_stock_cons(symbol="000300")  # 沪深300
    code_col = None
    for c in ["con_code", "成分券代码", "品种代码", "代码"]:
        if c in cons.columns:
            code_col = c
            break
    if code_col is None:
        raise ValueError(f"Cannot find code column in hs300 constituents. got={list(cons.columns)}")
    return cons[code_col].astype(str).str.zfill(6).tolist()


def fetch_one(code: str) -> tuple[str, bool, str]:
    """
    return: (code, ok, msg)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{code}.csv"

    # 若已存在且行数足够，跳过（粗略完整性检查）
    if out_path.exists():
        try:
            df0 = pd.read_csv(out_path, encoding="utf-8-sig")
            if len(df0) > 1500 and "date" in df0.columns and "close" in df0.columns:
                return code, True, "skip_cached"
        except Exception:
            pass  # 文件坏了就重拉

    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START, end_date=END, adjust=ADJUST)
        if df is None or len(df) == 0:
            return code, False, "empty"

        # 标准化成统一格式
        rename_map = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        need = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            return code, False, f"missing_cols={missing}, got={list(df.columns)}"

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date", "close"])

        df[need].to_csv(out_path, index=False, encoding="utf-8-sig")
        time.sleep(SLEEP_EACH)
        return code, True, "ok"
    except Exception as e:
        return code, False, f"err={e}"


def main():
    codes = get_hs300_codes()
    print(f"[INFO] hs300 codes: {len(codes)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, c): c for c in codes}
        for fut in as_completed(futures):
            code = futures[fut]
            c, is_ok, msg = fut.result()
            if is_ok:
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 20 == 0:
                print(f"[PROG] done={ok+fail}/{len(codes)} ok={ok} fail={fail}")
            if not is_ok:
                print(f"[FAIL] {c}: {msg}")

    print(f"[DONE] ok={ok}, fail={fail}, elapsed={time.time()-t0:.1f}s")
    print(f"[OUT] {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
