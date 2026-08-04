"""
HERMES Quant - 倉位管理 & 資金優化
- 風險平價 (Risk Parity)
- 均值-變異數最佳化 (Mean-Variance / Kelly)
- 固定風險倉位計算
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ── 1. 固定風險倉位計算 ──────────────────────────────────────────────────────
def fixed_risk_size(
    capital: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    market_type: str = "equity",
    lot_size: float = 100_000,
) -> float:
    """
    固定風險倉位：每筆最多虧損 capital * risk_pct
    Returns: 建議手數 / 股數
    """
    risk_amount = capital * risk_pct
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit < 1e-10:
        return 1.0
    if market_type == "forex":
        # 外匯：risk = lots * lot_size * pip_diff
        size = risk_amount / (risk_per_unit * lot_size)
        return round(max(0.01, size), 2)
    else:
        size = risk_amount / risk_per_unit
        return max(1.0, round(size))


# ── 2. Kelly Criterion ───────────────────────────────────────────────────────
def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,   # 實務上只用 Kelly 的一部分（保守）
) -> float:
    """
    Kelly 公式: f = W/L - (1-W)/G
    W = win_rate, L = avg_loss, G = avg_win
    fraction: 使用 full Kelly 的比例（0.25 = Quarter Kelly）
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.01
    b = avg_win / avg_loss           # 賠率
    p = win_rate                     # 勝率
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.01, min(fraction * kelly, 0.25))  # 上限 25%


# ── 3. 風險平價 (Risk Parity) ────────────────────────────────────────────────
def risk_parity_weights(
    returns: pd.DataFrame,
    lookback: int = 60,
) -> Dict[str, float]:
    """
    風險平價權重：讓每個資產對組合風險的貢獻相等
    Args:
        returns: DataFrame，每欄為一個資產的日報酬
        lookback: 計算波動率的回看期
    Returns:
        dict {symbol: weight}
    """
    recent = returns.tail(lookback).dropna()
    if recent.empty or len(recent.columns) == 0:
        n = len(returns.columns)
        return {col: 1/n for col in returns.columns}

    vols = recent.std()
    vols = vols.replace(0, np.nan).fillna(vols.mean())
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


# ── 4. 均值-變異數最佳化 (Markowitz) ─────────────────────────────────────────
def mean_variance_weights(
    returns: pd.DataFrame,
    lookback: int = 120,
    target: str = "max_sharpe",   # "max_sharpe" | "min_vol"
    rf: float = 0.02,
    n_sim: int = 3000,
) -> Dict[str, float]:
    """
    蒙地卡羅模擬最佳投資組合
    target:
        "max_sharpe" → 最大夏普比率
        "min_vol"    → 最小波動率
    """
    recent = returns.tail(lookback).dropna()
    syms = list(recent.columns)
    n = len(syms)
    if n == 0:
        return {}
    if n == 1:
        return {syms[0]: 1.0}

    mu  = recent.mean() * 252
    cov = recent.cov()  * 252

    best_metric = -np.inf if target == "max_sharpe" else np.inf
    best_w = np.ones(n) / n

    rng = np.random.default_rng(42)
    for _ in range(n_sim):
        w = rng.dirichlet(np.ones(n))
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < 1e-8:
            continue
        if target == "max_sharpe":
            sharpe = (port_ret - rf) / port_vol
            if sharpe > best_metric:
                best_metric, best_w = sharpe, w
        else:
            if port_vol < best_metric:
                best_metric, best_w = port_vol, w

    return {s: round(float(best_w[i]), 4) for i, s in enumerate(syms)}


# ── 5. 動態倉位調整（根據波動率縮放）────────────────────────────────────────
def vol_scaled_size(
    base_size: float,
    current_vol: float,
    target_vol: float = 0.15,
    max_scale: float = 3.0,
    min_scale: float = 0.1,
) -> float:
    """
    波動率縮放：高波動時降低倉位，低波動時提高倉位
    保持組合波動率穩定在 target_vol
    """
    if current_vol <= 0:
        return base_size
    scale = target_vol / current_vol
    scale = max(min_scale, min(max_scale, scale))
    return round(base_size * scale, 4)
