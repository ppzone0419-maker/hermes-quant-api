""
HERMES Quant — Base Backtest Engine
- Bar-based 撮合（最壞打算原則防止聖盃假象）
- 支援股票 / 外匯 / 黃金 / 加密貨幣
- 完整績效指標計算
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class Trade:
    symbol: str
    direction: str          # "long" | "short"
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    size: float = 1.0       # units / lots
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0.0
    status: str = "open"    # "open" | "closed" | "sl_hit" | "tp_hit"


@dataclass
class EngineConfig:
    symbol: str
    market_type: str = "equity"    # "equity" | "forex" | "crypto" | "commodity"
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_pips: float = 1.0
    leverage: float = 1.0
    pip_value: float = 0.0001       # forex pip value
    lot_size: float = 100_000.0     # forex standard lot
    risk_per_trade: float = 0.01    # 1% of capital per trade
    worst_case_sl: bool = True      # 最壞打算止損


class BaseBacktestEngine:
    """
    Bar-based 回測引擎
    最壞打算原則：
    - 多單止損時假設在 low 被觸發（最差情況）
    - 空單止損時假設在 high 被觸發（最差情況）
    - 避免假設在確切 SL 價格成交（聖盃假象）
    """

    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.capital = config.initial_capital
        self.equity_curve: List[float] = [config.initial_capital]
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self._df: Optional[pd.DataFrame] = None

    def load_data(self, df: pd.DataFrame):
        """載入 OHLCV 資料"""
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self._df = df.copy().sort_index()

    def _calc_slippage(self, price: float, direction: str) -> float:
        """計算滑點"""
        pip = self.cfg.pip_value if self.cfg.market_type == "forex" else price * 0.0001
        slip = self.cfg.slippage_pips * pip
        return price + slip if direction == "long" else price - slip

    def _calc_commission(self, price: float, size: float) -> float:
        """計算手續費"""
        if self.cfg.market_type == "forex":
            return size * self.cfg.lot_size * self.cfg.commission_rate
        return price * size * self.cfg.commission_rate

    def _calc_position_size(self, entry: float, stop: float) -> float:
        """
        風險導向倉位計算
        讓每筆交易最大虧損 = capital * risk_per_trade
        """
        if not stop or abs(entry - stop) < 1e-8:
            return 1.0
        risk_amount = self.capital * self.cfg.risk_per_trade
        risk_per_unit = abs(entry - stop)
        if self.cfg.market_type == "forex":
            size = risk_amount / (risk_per_unit * self.cfg.lot_size)
            return round(max(0.01, size), 2)
        return max(1.0, round(risk_amount / risk_per_unit))

    def _check_sl_tp(self, trade: Trade, bar: pd.Series) -> bool:
        """
        最壞打算原則撮合止損/止盈
        多單: SL 在 bar low，TP 在 bar high
        空單: SL 在 bar high，TP 在 bar low
        """
        if trade.direction == "long":
            # 最壞打算：先看 low 是否觸及 SL
            if trade.stop_loss and bar["low"] <= trade.stop_loss:
                exit_price = trade.stop_loss if self.cfg.worst_case_sl else bar["low"]
                self._close_trade(trade, bar.name, exit_price, "sl_hit")
                return True
            # 再看 high 是否觸及 TP
            if trade.take_profit and bar["high"] >= trade.take_profit:
                self._close_trade(trade, bar.name, trade.take_profit, "tp_hit")
                return True
        else:  # short
            if trade.stop_loss and bar["high"] >= trade.stop_loss:
                exit_price = trade.stop_loss if self.cfg.worst_case_sl else bar["high"]
                self._close_trade(trade, bar.name, exit_price, "sl_hit")
                return True
            if trade.take_profit and bar["low"] <= trade.take_profit:
                self._close_trade(trade, bar.name, trade.take_profit, "tp_hit")
                return True
        return False

    def _close_trade(self, trade: Trade, time: pd.Timestamp, price: float, status: str):
        """平倉計算 PnL"""
        commission = self._calc_commission(price, trade.size)
        if trade.direction == "long":
            trade.pnl = (price - trade.entry_price) * trade.size - commission
            if self.cfg.market_type == "forex":
                trade.pnl = (price - trade.entry_price) / self.cfg.pip_value * trade.size * 10 - commission
        else:
            trade.pnl = (trade.entry_price - price) * trade.size - commission
            if self.cfg.market_type == "forex":
                trade.pnl = (trade.entry_price - price) / self.cfg.pip_value * trade.size * 10 - commission

        trade.exit_time  = time
        trade.exit_price = price
        trade.status     = status
        self.capital    += trade.pnl
        self.open_trades.remove(trade)
        self.trades.append(trade)

    def open_trade(
        self,
        time: pd.Timestamp,
        price: float,
        direction: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        size: Optional[float] = None,
    ) -> Trade:
        """開倉"""
        entry = self._calc_slippage(price, direction)
        sz = size or self._calc_position_size(entry, stop_loss)
        commission = self._calc_commission(entry, sz)
        self.capital -= commission
        trade = Trade(
            symbol=self.cfg.symbol,
            direction=direction,
            entry_time=time,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=sz,
        )
        self.open_trades.append(trade)
        return trade

    def close_all(self, time: pd.Timestamp, price: float):
        """強制平倉所有持倉"""
        for trade in list(self.open_trades):
            self._close_trade(trade, time, price, "closed")

    def run(self, signals: pd.DataFrame) -> dict:
        """
        執行回測
        signals: DataFrame，index 與 OHLCV 一致，含 signal 欄位 (1=多, -1=空, 0=平倉)
        """
        if self._df is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        df = self._df.join(signals[["signal"]], how="left").fillna(0)
        self.equity_curve = [self.cfg.initial_capital]

        for ts, bar in df.iterrows():
            # 檢查已開倉的止損/止盈
            for trade in list(self.open_trades):
                self._check_sl_tp(trade, bar)

            # 根據訊號操作
            sig = int(bar.get("signal", 0))
            if sig == 1 and not self.open_trades:
                self.open_trade(ts, bar["open"], "long")
            elif sig == -1 and not self.open_trades:
                self.open_trade(ts, bar["open"], "short")
            elif sig == 0 and self.open_trades:
                self.close_all(ts, bar["close"])

            # 記錄浮動 equity
            float_pnl = sum(
                (bar["close"] - t.entry_price) * t.size
                if t.direction == "long"
                else (t.entry_price - bar["close"]) * t.size
                for t in self.open_trades
            )
            self.equity_curve.append(self.capital + float_pnl)

        # 回測結束強制平倉
        if self.open_trades:
            last_ts  = df.index[-1]
            last_bar = df.iloc[-1]
            self.close_all(last_ts, last_bar["close"])

        return self._calc_metrics(df)

    def _calc_metrics(self, df: pd.DataFrame) -> dict:
        """計算完整績效指標"""
        equity = pd.Series(self.equity_curve)
        returns = equity.pct_change().dropna()
        init_cap = self.cfg.initial_capital

        total_return = (equity.iloc[-1] - init_cap) / init_cap * 100
        trading_days = max(1, len(df))
        ann_factor = 252 if self.cfg.market_type in ("equity","crypto") else 252

        ann_return = (1 + total_return/100) ** (ann_factor / trading_days) - 1
        vol = returns.std() * np.sqrt(ann_factor) if len(returns) > 1 else 0
        sharpe = (ann_return - 0.02) / vol if vol > 0 else 0

        # Max Drawdown
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        max_dd = dd.min() * 100

        # Calmar
        calmar = ann_return / abs(max_dd/100) if max_dd != 0 else 0

        # Win rate
        closed = [t for t in self.trades if t.status != "open"]
        wins   = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        win_rate = len(wins) / len(closed) * 100 if closed else 0
        avg_win  = np.mean([t.pnl for t in wins])  if wins   else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 999

        # Equity curve for chart
        eq_dates = [str(df.index[min(i, len(df.index)-1)].date()) for i in range(len(self.equity_curve))]

        return {
            "metrics": {
                "total_return_pct":  round(total_return, 2),
                "ann_return_pct":    round(ann_return * 100, 2),
                "sharpe_ratio":      round(sharpe, 3),
                "calmar_ratio":      round(calmar, 3),
                "max_drawdown_pct":  round(max_dd, 2),
                "volatility_pct":    round(vol * 100, 2),
                "win_rate_pct":      round(win_rate, 2),
                "profit_factor":     round(profit_factor, 3),
                "total_trades":      len(closed),
                "winning_trades":    len(wins),
                "losing_trades":     len(losses),
                "avg_win":           round(avg_win, 2),
                "avg_loss":          round(avg_loss, 2),
                "final_capital":     round(self.capital, 2),
            },
            "equity_curve": {
                "dates":  eq_dates,
                "values": [round(v, 2) for v in self.equity_curve],
            },
            "trades": [
                {
                    "entry_time":  str(t.entry_time),
                    "exit_time":   str(t.exit_time),
                    "direction":   t.direction,
                    "entry_price": round(t.entry_price, 5),
                    "exit_price":  round(t.exit_price, 5) if t.exit_price else None,
                    "pnl":         round(t.pnl, 2),
                    "status":      t.status,
                }
                for t in self.trades[-50:]  # 只傳最後50筆
            ],
        }
