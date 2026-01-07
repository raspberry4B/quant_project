from __future__ import annotations

import pandas as pd
import backtrader as bt

from strategies.ma_filter import MAFilterStrategy


def run(csv_path: str, cash: float = 100000.0, commission: float = 0.0002):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding="utf-8-sig")

    # 确保列顺序/名字符合 PandasData 默认映射
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Processed data missing columns: {missing}. got={list(df.columns)}")

    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.addstrategy(MAFilterStrategy, ma_period=120)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # 常用分析
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")  # 日收益序列
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print(f"Final Value: {final_value:,.2f}")
    print(f"Max DrawDown (%): {dd.get('max', {}).get('drawdown', None)}")
    print(f"Max DrawDown (len): {dd.get('max', {}).get('len', None)}")
    print(f"Total Trades: {trades.get('total', {}).get('total', None)}")
    print(f"Won Trades: {trades.get('won', {}).get('total', None)}")
    print(f"Lost Trades: {trades.get('lost', {}).get('total', None)}")

    # 如需画图，取消注释（Windows 下可能会弹窗）
    # cerebro.plot()


if __name__ == "__main__":
    csv_path = "data/processed/etf_510300_daily_20150101_20251231_adj_qfq.csv"
    run(csv_path)
