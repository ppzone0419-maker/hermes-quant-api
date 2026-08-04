"""
HERMES Quant - 內建策略集
支援：SMC (Order Block + BOS)、動量、均線交叉、RSI均值回歸、布林通道
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── 1. 均線交叉策略 ──────────────────────────────────────────────────────────
def strategy_ma_cross(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    """快慢均線交叉：金叉做多，死叉做空"""
    fast_ma = _ema(df["close"], fast)
    slow_ma = _ema(df["close"], slow)
    signal  = pd.Series(0, index=df.index)
    signal[fast_ma > slow_ma]  =  1
    signal[fast_ma < slow_ma]  = -1
    signal = signal.diff().fillna(0)
    signal[signal > 0]  =  1
    signal[signal < 0]  = -1
    return pd.DataFrame({"signal": signal})


# ── 2. RSI 均值回歸策略 ──────────────────────────────────────────────────────
def strategy_rsi(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
) -> pd.DataFrame:
    """RSI 超賣做多，超買做空"""
    rsi    = _rsi(df["close"], period)
    signal = pd.Series(0, index=df.index)
    signal[rsi < oversold]    =  1
    signal[rsi > overbought]  = -1
    return pd.DataFrame({"signal": signal})


# ── 3. 布林通道突破策略 ──────────────────────────────────────────────────────
def strategy_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """價格突破布林上軌做多，跌破下軌做空"""
    mid  = df["close"].rolling(period).mean()
    std  = df["close"].rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    signal = pd.Series(0, index=df.index)
    signal[df["close"] > upper] =  1
    signal[df["close"] < lower] = -1
    return pd.DataFrame({"signal": signal})


# ── 4. 動量策略（Berkshire 啟發）────────────────────────────────────────────
def strategy_momentum(
    df: pd.DataFrame,
    lookback: int = 20,
    top_pct: float = 0.3,
) -> pd.DataFrame:
    """
    N 日動量策略
    過去 lookback 日報酬為正且強度在前 top_pct → 做多
    報酬為負且強度在後 top_pct → 做空
    """
    ret    = df["close"].pct_change(lookback)
    signal = pd.Series(0, index=df.index)
    threshold_hi = ret.rolling(lookback * 3).quantile(1 - top_pct)
    threshold_lo = ret.rolling(lookback * 3).quantile(top_pct)
    signal[ret > threshold_hi] =  1
    signal[ret < threshold_lo] = -1
    return pd.DataFrame({"signal": signal})


# ── 5. SMC - Order Block + BOS 策略 ─────────────────────────────────────────
def strategy_smc_ob(
    df: pd.DataFrame,
    ob_lookback: int = 10,
    atr_period: int = 14,
    atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    SMC Order Block + Break of Structure
    1. 識別最近 ob_lookback 根 K 的最高(阻力OB)、最低(支撐OB)
    2. 價格突破結構高點 (BOS 向上) → 確認多頭 OB，回測 OB 上緣做多
    3. 價格跌破結構低點 (BOS 向下) → 確認空頭 OB，反彈至 OB 下緣做空
    4. 止損設在 ATR 乘數外
    """
    atr    = _atr(df, atr_period)
    signal = pd.Series(0, index=df.index)

    highs  = df["high"].rolling(ob_lookback).max()
    lows   = df["low"].rolling(ob_lookback).min()

    # BOS 向上：收盤突破前高
    bos_up   = df["close"] > highs.shift(1)
    # BOS 向下：收盤跌破前低
    bos_down = df["close"] < lows.shift(1)

    # 回測 Order Block 上緣（多單進場）
    ob_bull_top = lows.shift(1) + (highs.shift(1) - lows.shift(1)) * 0.3
    # 回測 Order Block 下緣（空單進場）
    ob_bear_bot = highs.shift(1) - (highs.shift(1) - lows.shift(1)) * 0.3

    # 做多條件：BOS 向上 + 價格在 OB 支撐區
    bull_cond = bos_up & (df["low"] <= ob_bull_top)
    # 做空條件：BOS 向下 + 價格在 OB 阻力區
    bear_cond = bos_down & (df["high"] >= ob_bear_bot)

    signal[bull_cond] =  1
    signal[bear_cond] = -1
    return pd.DataFrame({"signal": signal})


# ── 6. CRT - Candle Range Theory ────────────────────────────────────────────
def strategy_crt(
    df: pd.DataFrame,
    range_bars: int = 3,
) -> pd.DataFrame:
    """
    CRT 蠟燭範圍理論
    1. 計算前 range_bars 根 K 的高低（範圍）
    2. 假突破（sweep）前高後立即收回 → 空單
    3. 假突破（sweep）前低後立即收回 → 多單
    """
    signal = pd.Series(0, index=df.index)
    prev_high = df["high"].rolling(range_bars).max().shift(1)
    prev_low  = df["low"].rolling(range_bars).min().shift(1)

    # 假突破上方後收回（bearish sweep）
    bear_sweep = (df["high"] > prev_high) & (df["close"] < prev_high)
    # 假突破下方後收回（bullish sweep）
    bull_sweep = (df["low"] < prev_low) & (df["close"] > prev_low)

    signal[bull_sweep] =  1
    signal[bear_sweep] = -1
    return pd.DataFrame({"signal": signal})


# ── 策略映射表 ───────────────────────────────────────────────────────────────
STRATEGY_MAP = {
    "ma_cross":  strategy_ma_cross,
    "rsi":       strategy_rsi,
    "bollinger": strategy_bollinger,
    "momentum":  strategy_momentum,
    "smc_ob":    strategy_smc_ob,
    "crt":       strategy_crt,
}


def run_strategy(name: str, df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """執行指定策略，回傳含 signal 欄位的 DataFrame"""
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
    fn = STRATEGY_MAP[name]
    valid_params = {k: v for k, v in params.items() if k in fn.__code__.co_varnames}
    return fn(df, **valid_params)
