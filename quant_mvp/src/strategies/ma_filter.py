import backtrader as bt


class MAFilterStrategy(bt.Strategy):
    params = dict(ma_period=120, lot=100, printlog=True)

    def log(self, txt):
        if self.p.printlog:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt} {txt}")

    def __init__(self):
        self.ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.ma_period)
        self.order = None

    def next(self):
        if self.order:
            return  # 有未完成订单，先不重复下单

        close = float(self.data.close[0])
        ma = float(self.ma[0])
        cash = float(self.broker.getcash())

        # 观察日志：你会立刻知道信号是否触发
        self.log(f"close={close:.4f}, ma={ma:.4f}, pos={self.position.size}, cash={cash:.2f}")

        if not self.position:
            if close > ma:
                # 全仓买入（ETF 以 100 份为一手）
                size = int(cash / close)
                size = (size // self.p.lot) * self.p.lot
                if size > 0:
                    self.log(f"BUY size={size}")
                    self.order = self.buy(size=size)
        else:
            if close <= ma:
                self.log("CLOSE")
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED price={order.executed.price:.4f}, size={order.executed.size}")
            else:
                self.log(f"SELL EXECUTED price={order.executed.price:.4f}, size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED status={order.getstatusname()}")
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
