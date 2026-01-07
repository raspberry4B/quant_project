from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import akshare as ak


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

# ========== 数据获取：交易日与ETF价格（用于总风控） ==========
def fetch_etf_daily(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust)
    # 标准化
    df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df[["open", "high", "low", "close", "volume"]]


def month_first_trade_days(trading_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    # trading_index: 已按交易日排序的 DatetimeIndex
    s = pd.Series(1, index=trading_index)
    # 每月取第一个交易日
    m = s.groupby([s.index.year, s.index.month]).head(1)
    return list(m.index)


# ========== 股票池（建议先用沪深300成分，减少数据量） ==========
def get_stock_pool(trade_date: pd.Timestamp) -> List[str]:
    if POOL_MODE == "custom":
        return CUSTOM_POOL

    # 沪深300成分（成分会随时间变化；阶段一先用“当期成分”近似）
    # 注意：该接口返回的代码格式可能带市场前缀或不同字段名，你需要按实际返回做一次映射
    cons = ak.index_stock_cons(symbol="000300")  # 沪深300
    # 常见字段：成分券代码
    code_col = None
    for c in ["con_code", "成分券代码", "品种代码", "代码"]:
        if c in cons.columns:
            code_col = c
            break
    if code_col is None:
        raise ValueError(f"Cannot find code column in hs300 constituents. got={list(cons.columns)}")
    codes = cons[code_col].astype(str).str.zfill(6).tolist()
    return codes


# ========== 基本面与市值/估值数据 ==========
def fetch_spot_snapshot() -> pd.DataFrame:
    """
    东方财富 A股实时快照，字段里通常含：代码、名称、最新价、总市值/流通市值、市盈率、市净率等
    注意：它是“当前时点”的快照，不是历史；用于学习阶段可接受，严格回测需用历史市值/估值口径替代。
    """
    df = ak.stock_zh_a_spot_em()
    return df


def fetch_financial_indicators(code: str) -> pd.DataFrame:
    """
    财务分析指标（通常含 ROE、净利润等）。
    不同接口字段会有差异；这里以“先拉到数据”为目标，后续你再精细化字段选择。
    """
    df = ak.stock_financial_analysis_indicator(symbol=code)
    return df


# ========== 选股逻辑（横截面） ==========
def select_stocks(trade_date: pd.Timestamp) -> List[str]:
    # 1) 股票池
    pool = get_stock_pool(trade_date)

    # 2) 快照：估值/市值/名称（用于ST近似过滤）
    spot = fetch_spot_snapshot().copy()

    # 尝试识别关键列
    def find_col(cands: List[str]) -> str:
        for c in cands:
            if c in spot.columns:
                return c
        raise ValueError(f"Cannot find columns {cands}. got={list(spot.columns)}")

    col_code = find_col(["代码", "code", "股票代码"])
    col_name = find_col(["名称", "name", "股票简称"])
    # 市值/估值字段在不同版本可能叫法不同，这里给候选
    col_mcap = None
    for c in ["流通市值", "流通市值(元)", "流通市值（元）", "流通市值(亿)", "流通市值（亿）"]:
        if c in spot.columns:
            col_mcap = c
            break
    col_pb = None
    for c in ["市净率", "市净率PB", "PB", "PB(市净率)"]:
        if c in spot.columns:
            col_pb = c
            break
    if col_mcap is None or col_pb is None:
        raise ValueError(f"Cannot find mcap/pb columns. got={list(spot.columns)}")

    snap = spot[[col_code, col_name, col_mcap, col_pb]].copy()
    snap[col_code] = snap[col_code].astype(str).str.zfill(6)

    # 3) pool 过滤
    snap = snap[snap[col_code].isin(pool)]

    # 4) ST 近似过滤（严格做法后续可替换为专门ST列表）
    snap = snap[~snap[col_name].astype(str).str.contains("ST")]

    # 5) 市值/估值过滤（注意单位：有的字段是“亿”，有的是“元”）
    mcap = pd.to_numeric(snap[col_mcap], errors="coerce")
    pb = pd.to_numeric(snap[col_pb], errors="coerce")

    # 单位处理：若看起来像“亿”，转成元
    # 经验规则：若中位数 < 1e6，通常是“亿”为单位（例如 1234.56 亿）
    med = mcap.median(skipna=True)
    if pd.notna(med) and med < 1e6:
        mcap = mcap * 1e8
    
    print("mcap median:", mcap.median())
    print("mcap min/max:", mcap.min(), mcap.max())

    snap = snap.assign(mcap=mcap, pb=pb).dropna(subset=["mcap", "pb"])
    snap = snap[(snap["mcap"] >= MCAP_MIN) & (snap["mcap"] <= MCAP_MAX)]
    snap = snap[snap["pb"] <= PB_MAX]

    # 6) 财务过滤 + 排序（ROE）
    records = []
    for code in snap[col_code].tolist():
        try:
            fin = fetch_financial_indicators(code)
            if fin is None or len(fin) == 0:
                continue

            # 尝试找到 ROE、净利润字段（候选）
            def find_fin_col(df, cands):
                for c in cands:
                    if c in df.columns:
                        return c
                return None

            roe_col = find_fin_col(fin, ["净资产收益率(ROE)", "净资产收益率", "ROE", "ROE(%)"])
            profit_col = find_fin_col(fin, ["净利润", "归母净利润", "净利润(元)", "归母净利润(元)"])

            if roe_col is None or profit_col is None:
                print(f"[MISS] {code} no ROE col, fin cols={list(fin.columns)}")
                continue

            # 取最新一期（通常第一行就是最新；保险起见按报告期排序）
            fin2 = fin.copy()
            # 常见报告期字段候选
            rpt_col = find_fin_col(fin2, ["报告期", "日期", "截止日期", "period"])
            if rpt_col is not None:
                fin2[rpt_col] = pd.to_datetime(fin2[rpt_col], errors="coerce")
                fin2 = fin2.sort_values(rpt_col, ascending=False)

            roe = pd.to_numeric(fin2.iloc[0][roe_col], errors="coerce")
            profit = pd.to_numeric(fin2.iloc[0][profit_col], errors="coerce")

            if pd.isna(roe) or pd.isna(profit):
                continue
            if roe < ROE_MIN:
                continue
            if profit <= 0:
                continue

            records.append((code, float(roe)))
        except Exception:
            continue

    if not records:
        return []

    sel = pd.DataFrame(records, columns=["code", "roe"]).sort_values("roe", ascending=False)
    return sel.head(N_HOLD)["code"].tolist()


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

    price_cache: Dict[str, pd.DataFrame] = {}

    # 在线获取股票价格数据的辅助函数
    '''def get_stock_price_df(code: str) -> pd.DataFrame:
        if code in price_cache:
            return price_cache[code]
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START, end_date=END, adjust="qfq")
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        price_cache[code] = df[["close"]]
        return price_cache[code]'''
    
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
        print(f"[REB] {d0.date()} picks={len(picks)}")

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
