from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import akshare as ak
import numpy as np

# ========== 参数区（你可直接改这里） ==========
ETF_SYMBOL = "510300"              # 用于总风控
START = "20150101"
END = "20251231"
ADJUST = "qfq"

N_HOLD = 20                        # 持仓数
MA_PERIOD = 120                    # 趋势过滤均线
MCAP_MIN = 50e8                    # 50亿（单位：元）
MCAP_MAX = 500e8                   # 500亿
PB_MAX = 2.5
ROE_MIN = 10.0                     # %
MIN_CASH_BUFFER = 0.02             # 现金缓冲（用于实盘 sizing；回测收益计算不必用）
POOL_MODE = "hs300"                # "hs300" 或 "custom"
CUSTOM_POOL = ["600519", "000001"] # 仅当 POOL_MODE="custom" 时使用（示例）
WAREHOUSE_PRICE_DIR = "data/warehouse/prices/qfq"
PRICE_DIR = Path("data/warehouse/prices") / ADJUST

# ========== 数据获取：交易日与ETF价格（用于总风控） ==========
def fetch_etf_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    p = Path("data/warehouse/prices") / adjust / f"{symbol}.csv"
    if not p.exists():
        raise FileNotFoundError(f"ETF file not found: {p.resolve()}")

    df = pd.read_csv(p, parse_dates=["date"], encoding="utf-8-sig")
    df = df.sort_values("date").set_index("date")

    # 统一列名并确保都是数值
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])

    # 用时间戳切片，start/end 是 YYYYMMDD
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return df.loc[(df.index >= start_dt) & (df.index <= end_dt), ["open", "high", "low", "close", "volume"]]


def month_first_trade_days(trading_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    # trading_index: 已按交易日排序的 DatetimeIndex
    s = pd.Series(1, index=trading_index)
    # 每月取第一个交易日
    m = s.groupby([s.index.year, s.index.month]).head(1)
    return list(m.index)


# ========== 股票池（建议先用沪深300成分，减少数据量） ==========
def get_stock_pool(trade_date: pd.Timestamp) -> List[str]:
    # 离线：直接用本地价格仓库文件名作为股票池
    p = Path("data/warehouse/prices") / ADJUST
    codes = []
    for f in p.glob("*.csv"):
        code = f.stem
        # 跳过 ETF
        if code == "510300":
            continue
        # 只保留 A 股 6 位代码
        if len(code) == 6 and code.isdigit():
            codes.append(code)
    codes.sort()
    return codes


# ========== 基本面与市值/估值数据 ==========
def fetch_spot_snapshot() -> pd.DataFrame:
    """
    东方财富 A股实时快照，字段里通常含：代码、名称、最新价、总市值/流通市值、市盈率、市净率等
    注意：它是“当前时点”的快照，不是历史；用于学习阶段可接受，严格回测需用历史市值/估值口径替代。
    """
    df = ak.stock_zh_a_spot_em()
    return df

#目前测试财务接口都挂了
def fetch_financial_indicators(code: str) -> pd.DataFrame:
    """
    财务分析指标（通常含 ROE、净利润等）。
    不同接口字段会有差异；这里以“先拉到数据”为目标，后续你再精细化字段选择。
    """
    #df = ak.stock_financial_analysis_indicator(symbol=code)
    df = ak.stock_financial_analysis_indicator_em(symbol=code)
    return df


# 策略定义（PB + PE + 市值 + 非ST）
def select_stocks(trade_date: pd.Timestamp) -> List[str]:
    pool = get_stock_pool(trade_date)
    picks = []

    records = []
    for code in pool:
        p = PRICE_DIR / f"{code}.csv"
        if not p.exists():
            continue

        df = pd.read_csv(p, parse_dates=["date"], encoding="utf-8-sig")
        df = df.sort_values("date").set_index("date")
        if "close" not in df.columns:
            continue
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close"])

        # 找到 trade_date 对应的可交易日位置（取不晚于 trade_date 的最后一个交易日）
        idx = df.index.searchsorted(trade_date, side="right") - 1
        if idx < 200:  # 至少需要 200 个交易日用于因子
            continue

        close = df["close"].iloc[: idx + 1]
        volu = df["volume"].iloc[: idx + 1] if "volume" in df.columns else None

        # 因子窗口（交易日）
        # mom: (t-20)/(t-140) - 1 约 6个月动量，跳过近20日
        c_t = close.iloc[-1]
        c_20 = close.iloc[-21]
        c_140 = close.iloc[-141]
        if c_140 <= 0 or c_20 <= 0:
            continue

        mom = c_20 / c_140 - 1.0
        ret20 = c_t / c_20 - 1.0

        # 波动率：60日收益标准差
        r = close.pct_change().dropna()
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 70:
            continue
        vol60 = r.iloc[-60:].std()

        # 流动性过滤：20日平均成交量（代理）
        liq20 = None
        if volu is not None and len(volu.dropna()) >= 30:
            liq20 = volu.iloc[-20:].mean()

        records.append((code, mom, ret20, vol60, liq20))

    if not records:
        return []

    fac = pd.DataFrame(records, columns=["code", "mom", "ret20", "vol60", "liq20"])

    # 过滤：短期不能太差
    fac = fac[fac["ret20"] > -0.15]

    # 流动性过滤（如果有）
    if fac["liq20"].notna().any():
        thr = fac["liq20"].quantile(0.2)
        fac = fac[fac["liq20"] >= thr]

    # 去掉极端高波动（可选）
    fac = fac[fac["vol60"] <= fac["vol60"].quantile(0.7)]

    # 排序：动量优先，波动率次之
    fac = fac.sort_values(["mom", "vol60"], ascending=[False, True])

    return fac.head(N_HOLD)["code"].tolist()



# ========== 回测：月频调仓、等权、趋势过滤 ==========
def backtest():
    etf = fetch_etf_daily(ETF_SYMBOL, START, END, adjust=ADJUST)
    etf["ma"] = etf["close"].rolling(MA_PERIOD).mean()

    rebalance_days = month_first_trade_days(etf.index)
    # 确保 MA 有足够窗口
    rebalance_days = [d for d in rebalance_days if d in etf.index and pd.notna(etf.loc[d, "ma"])]

    # 持仓：{code: weight}
    holdings: Dict[str, float] = {}
    nav = 1.0
    nav_series = []

    # 价格缓存（避免重复拉取）：{code: df_price}
    price_cache: Dict[str, pd.DataFrame] = {}

    def get_stock_price_df(code: str) -> pd.DataFrame:
        if code in price_cache:
            return price_cache[code]
        # 注意：个股日线接口可能需要 symbol 带市场/或不带；这里按 6 位代码尝试
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START, end_date=END, adjust="qfq")
        # 统一
        cand_map = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns={k: v for k, v in cand_map.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        price_cache[code] = df[["close"]]
        return price_cache[code]

    # 回测主循环：按日更新净值（只用收盘价计算）
    dates = etf.index
    reb_set = set(rebalance_days)

    for i, d in enumerate(dates):
        # 1) 若当日是调仓日：先看趋势过滤
        if d in reb_set:
            if etf.loc[d, "close"] <= etf.loc[d, "ma"]:
                holdings = {}  # 空仓
            else:
                picks = select_stocks(d)
                if picks:
                    w = 1.0 / len(picks)
                    holdings = {c: w for c in picks}
                else:
                    holdings = {}

        # 2) 计算当日组合收益（等权、用收盘价相对昨日收盘价）
        if i == 0:
            nav_series.append((d, nav))
            continue

        daily_ret = 0.0
        if holdings:
            for code, w in holdings.items():
                pdf = get_stock_price_df(code)
                if d not in pdf.index or dates[i - 1] not in pdf.index:
                    # 缺失则视为当日收益0（简化；后续可改为剔除或前向填充）
                    continue
                p0 = float(pdf.loc[dates[i - 1], "close"])
                p1 = float(pdf.loc[d, "close"])
                if p0 > 0:
                    daily_ret += w * (p1 / p0 - 1.0)

        nav *= (1.0 + daily_ret)
        nav_series.append((d, nav))

    nav_df = pd.DataFrame(nav_series, columns=["date", "nav"]).set_index("date")
    nav_df["dd"] = nav_df["nav"] / nav_df["nav"].cummax() - 1.0
    print(nav_df.tail())
    print("Max DrawDown:", float(nav_df["dd"].min()))
    return nav_df

def backtest_monthly():
    etf = fetch_etf_daily(ETF_SYMBOL, START, END, adjust=ADJUST)
    etf["ma"] = etf["close"].rolling(MA_PERIOD).mean()

    rebalance_days = month_first_trade_days(etf.index)
    rebalance_days = [d for d in rebalance_days if pd.notna(etf.loc[d, "ma"])]
    print("[INFO] first rebalance day:", rebalance_days[0].date(), "last:", rebalance_days[-1].date(), "count:", len(rebalance_days))

    price_cache: Dict[str, pd.DataFrame] = {}
    
    # 使用本地缓存的价格数据
    def get_stock_price_df(code: str) -> pd.DataFrame:
        if code in price_cache:
            return price_cache[code]
        p = Path(WAREHOUSE_PRICE_DIR) / f"{code}.csv"
        df = pd.read_csv(p, parse_dates=["date"], encoding="utf-8-sig")
        df = df.sort_values("date").set_index("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        price_cache[code] = df[["close"]]
        return price_cache[code]

    nav = 1.0
    out = []

    for i in range(len(rebalance_days) - 1):
        d0 = rebalance_days[i]
        d1 = rebalance_days[i + 1]  # 下一个调仓日（区间右端，不含）

        risk_off = etf.loc[d0, "close"] <= etf.loc[d0, "ma"]
        #risk_off = False  # 临时强制关闭，仅用于调试

        if risk_off:
            out.append((d0, nav))
            continue

        picks = select_stocks(d0)
        print(f"[REB] {d0.date()} picks={len(picks)} risk_off={risk_off}")
        if len(picks) > 0:
            print("   sample:", picks[:5])

        if not picks:
            out.append((d0, nav))
            continue

        rets = []
        for code in picks:
            pdf = get_stock_price_df(code)

            # 区间首尾“可交易日”对齐：d0之后的第一天、d1之前的最后一天
            s = pdf.index.searchsorted(d0, side="left")
            e = pdf.index.searchsorted(d1, side="left") - 1
            if s < 0 or s >= len(pdf) or e < 0 or e <= s:
                continue

            p0 = float(pdf.iloc[s]["close"])
            p1 = float(pdf.iloc[e]["close"])
            if p0 > 0:
                rets.append(p1 / p0 - 1.0)

        if rets:
            period_ret = sum(rets) / len(rets)  # 等权平均
            nav *= (1.0 + period_ret)

        out.append((d0, nav))

    nav_df = pd.DataFrame(out, columns=["date", "nav"]).set_index("date")
    nav_df["dd"] = nav_df["nav"] / nav_df["nav"].cummax() - 1.0
    print(nav_df.tail())
    print("Max DrawDown:", float(nav_df["dd"].min()))
    return nav_df


if __name__ == "__main__":
    nav_df = backtest_monthly()
