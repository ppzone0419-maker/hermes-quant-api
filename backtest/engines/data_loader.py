HERMES Quant — Data Loader
- yfinance 抓取免費數據
- Parquet 本地快取（相同標的+時間不重複抓）
- 自動清洗離群值 (Bad Ticks)
- 支援股票、外匯、黃金、加密貨幣
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Yahoo Finance symbol mapping ────────────────────────────────────────────
SYMBOL_MAP = {
    # 外匯
    "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X", "USD/CHF": "CHF=X", "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X", "USD/TWD": "TWD=X",
    # 黃金/白銀
    "XAUUSD": "GC=F", "XAU/USD": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG/USD": "SI=F",
    # 加密（直接加 -USD）
}


def _to_yf_symbol(symbol: str) -> str:
    """Convert TV/common symbol to Yahoo Finance format."""
    sym = symbol.upper().strip()
    if sym in SYMBOL_MAP:
        return SYMBOL_MAP[sym]
    # NASDAQ:NVDA → NVDA
    if ":" in sym:
        sym = sym.split(":")[-1]
    # BTC-USD, ETH-USD already valid
    return sym


def _cache_path(symbol: str, interval: str, start: str, end: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_").replace("=", "_")
    return CACHE_DIR / f"{safe}_{interval}_{start}_{end}.parquet"


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """清洗離群值 (Bad Ticks)."""
    if df.empty:
        return df
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            continue
        # 移除 IQR 5倍以上的極端值（替換為 NaN 後前向填充）
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        mask = (df[col] < q1 - 5 * iqr) | (df[col] > q3 + 5 * iqr)
        df.loc[mask, col] = np.nan

    # 確保 OHLC 邏輯正確
    if all(c in df.columns for c in ["open","high","low","close"]):
        df["high"] = df[["open","high","low","close"]].max(axis=1)
        df["low"]  = df[["open","high","low","close"]].min(axis=1)

    df = df.ffill().dropna(subset=["close"])
    return df


def load_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV data with local cache.
    Args:
        symbol: e.g. "NVDA", "EUR/USD", "BTC-USD", "GC=F"
        start: "YYYY-MM-DD"
        end:   "YYYY-MM-DD"
        interval: "1d" | "1h" | "1wk"
        force_refresh: ignore cache
    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    yf_sym = _to_yf_symbol(symbol)
    cache_path = _cache_path(yf_sym, interval, start, end)

    # 讀快取
    if not force_refresh and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"[Cache HIT] {symbol} {start}→{end}")
            return df
        except Exception:
            pass

    # 從 yfinance 抓取
    try:
        import yfinance as yf
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning(f"[yfinance] No data for {yf_sym}")
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["open","high","low","close","volume"]].copy()
        df = _clean_ohlcv(df)
        df.to_parquet(cache_path)
        logger.info(f"[yfinance] Loaded {len(df)} bars for {yf_sym}")
        return df

    except Exception as e:
        logger.error(f"[yfinance] Error for {yf_sym}: {e}")
        return pd.DataFrame()
