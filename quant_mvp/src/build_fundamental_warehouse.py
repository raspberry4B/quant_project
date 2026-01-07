from __future__ import annotations

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak


MAX_WORKERS = 6
SLEEP_EACH = 0.2
OUT_DIR = Path("data/warehouse/fundamentals/financial_indicator")


def get_hs300_codes() -> list[str]:
    cons = ak.index_stock_cons(symbol="000300")
    code_col = None
    for c in ["con_code", "成分券代码", "品种代码", "代码"]:
        if c in cons.columns:
            code_col = c
            break
    if code_col is None:
        raise ValueError(f"Cannot find code column in hs300 constituents. got={list(cons.columns)}")
    return cons[code_col].astype(str).str.zfill(6).tolist()


def fetch_one(code: str) -> tuple[str, bool, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{code}.csv"

    if out_path.exists():
        try:
            df0 = pd.read_csv(out_path, encoding="utf-8-sig")
            if len(df0) > 0:
                return code, True, "skip_cached"
        except Exception:
            pass

    try:
        df = ak.stock_financial_analysis_indicator(symbol=code)
        if df is None or len(df) == 0:
            return code, False, "empty"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        time.sleep(SLEEP_EACH)
        return code, True, "ok"
    except Exception as e:
        return code, False, f"err={e}"


def main():
    codes = get_hs300_codes()
    print(f"[INFO] hs300 codes: {len(codes)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, c): c for c in codes}
        for fut in as_completed(futures):
            code = futures[fut]
            c, is_ok, msg = fut.result()
            if is_ok:
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {c}: {msg}")
            if (ok + fail) % 50 == 0:
                print(f"[PROG] done={ok+fail}/{len(codes)} ok={ok} fail={fail}")

    print(f"[DONE] ok={ok}, fail={fail}")
    print(f"[OUT] {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
