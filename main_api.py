"""
HERMES Quant — FastAPI 後端主程式
端點：
  POST /api/backtest        → 執行回測
  POST /api/optimize        → 投資組合最佳化
  GET  /api/data/preview    → 預覽歷史資料
  GET  /api/strategies      → 取得可用策略清單
  GET  /health              → 健康檢查
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backtest.loaders.data_loader import load_ohlcv
from backtest.engines.base_engine import BaseBacktestEngine, EngineConfig
from backtest.engines.strategies import run_strategy, STRATEGY_MAP
from backtest.optimizers.position_sizer import (
    risk_parity_weights,
    mean_variance_weights,
    kelly_fraction,
    fixed_risk_size,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HERMES Quant API",
    version="1.0.0",
    description="量化回測 & 投資組合最佳化 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ──────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    symbol:          str   = Field("NVDA",        description="標的代碼")
    start:           str   = Field("2022-01-01",  description="開始日期 YYYY-MM-DD")
    end:             str   = Field("",            description="結束日期（空=今天）")
    interval:        str   = Field("1d",          description="K線週期 1d|1h|1wk")
    strategy:        str   = Field("ma_cross",    description="策略名稱")
    market_type:     str   = Field("equity",      description="市場類型 equity|forex|crypto|commodity")
    initial_capital: float = Field(100_000.0,     description="初始資金")
    risk_per_trade:  float = Field(0.01,          description="每筆最大虧損比例")
    commission_rate: float = Field(0.001,         description="手續費率")
    slippage_pips:   float = Field(1.0,           description="滑點(pips)")
    leverage:        float = Field(1.0,           description="槓桿倍數")
    worst_case_sl:   bool  = Field(True,          description="最壞打算止損原則")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="策略參數")
    force_refresh:   bool  = Field(False,         description="強制重新抓取資料")


class OptimizeRequest(BaseModel):
    symbols:     List[str] = Field(["NVDA","AAPL","MSFT"], description="標的列表")
    start:       str       = Field("2022-01-01")
    end:         str       = Field("")
    method:      str       = Field("risk_parity", description="risk_parity|max_sharpe|min_vol")
    capital:     float     = Field(100_000.0)
    lookback:    int       = Field(60,  description="計算波動率的回看天數")


class DataPreviewRequest(BaseModel):
    symbol:   str = Field("NVDA")
    start:    str = Field("2023-01-01")
    end:      str = Field("")
    interval: str = Field("1d")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _resolve_end(end: str) -> str:
    return end if end else datetime.today().strftime("%Y-%m-%d")


def _detect_market_type(symbol: str) -> str:
    sym = symbol.upper()
    if any(x in sym for x in ["BTC","ETH","SOL","XRP","USDT","BNB"]):
        return "crypto"
    if any(x in sym for x in ["EUR","JPY","GBP","AUD","CHF","CAD","NZD","USD=X","=X"]):
        return "forex"
    if any(x in sym for x in ["XAU","GC=F","SI=F","GOLD","SILVER","XAG"]):
        return "commodity"
    return "equity"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "time": datetime.now().isoformat()}


@app.get("/api/strategies")
def get_strategies():
    """取得所有可用策略與說明"""
    info = {
        "ma_cross":  {"name": "均線交叉", "params": {"fast": 10, "slow": 30}, "desc": "快慢 EMA 交叉，金叉做多死叉做空"},
        "rsi":       {"name": "RSI 均值回歸", "params": {"period": 14, "oversold": 30, "overbought": 70}, "desc": "RSI 超賣做多，超買做空"},
        "bollinger": {"name": "布林通道", "params": {"period": 20, "std_dev": 2.0}, "desc": "突破布林上/下軌進場"},
        "momentum":  {"name": "動量策略", "params": {"lookback": 20, "top_pct": 0.3}, "desc": "N日動量強者做多，弱者做空"},
        "smc_ob":    {"name": "SMC Order Block", "params": {"ob_lookback": 10, "atr_period": 14, "atr_mult": 1.5}, "desc": "BOS + Order Block 回測進場"},
        "crt":       {"name": "CRT 蠟燭範圍", "params": {"range_bars": 3}, "desc": "假突破識別後反向進場"},
    }
    return {"strategies": info}


@app.post("/api/data/preview")
def data_preview(req: DataPreviewRequest):
    """預覽歷史資料（前20筆）"""
    end = _resolve_end(req.end)
    df  = load_ohlcv(req.symbol, req.start, end, req.interval)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"無法取得 {req.symbol} 的資料")
    sample = df.tail(20).copy()
    sample.index = sample.index.astype(str)
    return {
        "symbol":    req.symbol,
        "rows":      len(df),
        "start":     str(df.index[0].date()),
        "end":       str(df.index[-1].date()),
        "interval":  req.interval,
        "preview":   sample.reset_index().rename(columns={"index":"date"}).to_dict(orient="records"),
    }


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """
    執行完整回測流程
    1. 載入 / 快取資料
    2. 產生策略訊號
    3. 引擎撮合（最壞打算止損）
    4. 計算績效指標 + 權益曲線
    5. Kelly / 建議倉位計算
    """
    end         = _resolve_end(req.end)
    market_type = req.market_type or _detect_market_type(req.symbol)

    # 1. 載入資料
    logger.info(f"[Backtest] {req.symbol} {req.start}→{end} strategy={req.strategy}")
    df = load_ohlcv(req.symbol, req.start, end, req.interval, req.force_refresh)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"無法取得 {req.symbol} 的資料，請確認代碼與日期範圍")

    # 2. 產生訊號
    try:
        signals = run_strategy(req.strategy, df, req.strategy_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3. 設定引擎
    pip_value = 0.0001 if market_type == "forex" else 1.0
    cfg = EngineConfig(
        symbol          = req.symbol,
        market_type     = market_type,
        initial_capital = req.initial_capital,
        commission_rate = req.commission_rate,
        slippage_pips   = req.slippage_pips,
        leverage        = req.leverage,
        pip_value       = pip_value,
        risk_per_trade  = req.risk_per_trade,
        worst_case_sl   = req.worst_case_sl,
    )
    engine = BaseBacktestEngine(cfg)
    engine.load_data(df)

    # 4. 執行回測
    result = engine.run(signals)
    metrics = result["metrics"]

    # 5. Kelly 建議倉位
    kelly = kelly_fraction(
        win_rate = metrics["win_rate_pct"] / 100,
        avg_win  = metrics["avg_win"],
        avg_loss = abs(metrics["avg_loss"]) if metrics["avg_loss"] != 0 else 1,
    )
    last_price = float(df["close"].iloc[-1])
    suggested_size = fixed_risk_size(
        capital     = req.initial_capital,
        risk_pct    = kelly,
        entry       = last_price,
        stop_loss   = last_price * 0.98,
        market_type = market_type,
    )

    result["position_advice"] = {
        "kelly_fraction_pct":  round(kelly * 100, 2),
        "suggested_risk_pct":  round(kelly * 100, 2),
        "suggested_size":      suggested_size,
        "last_price":          round(last_price, 5),
        "note": f"建議每筆最大虧損資金的 {round(kelly*100,1)}%，倉位約 {suggested_size} 單位",
    }

    result["meta"] = {
        "symbol":      req.symbol,
        "strategy":    req.strategy,
        "market_type": market_type,
        "start":       req.start,
        "end":         end,
        "interval":    req.interval,
        "bars":        len(df),
    }

    return result


@app.post("/api/optimize")
def optimize_portfolio(req: OptimizeRequest):
    """
    投資組合最佳化
    method: risk_parity | max_sharpe | min_vol
    """
    end = _resolve_end(req.end)

    # 載入所有標的的收盤價
    close_dict = {}
    errors = []
    for sym in req.symbols:
        df = load_ohlcv(sym, req.start, end, "1d")
        if df.empty:
            errors.append(sym)
        else:
            close_dict[sym] = df["close"]

    if not close_dict:
        raise HTTPException(status_code=404, detail=f"所有標的都無法取得資料: {errors}")

    prices  = pd.DataFrame(close_dict).dropna()
    returns = prices.pct_change().dropna()

    # 最佳化
    if req.method == "risk_parity":
        weights = risk_parity_weights(returns, lookback=req.lookback)
    elif req.method in ("max_sharpe", "min_vol"):
        weights = mean_variance_weights(returns, lookback=req.lookback, target=req.method)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    # 計算建議金額
    allocation = {
        sym: {
            "weight_pct":   round(w * 100, 2),
            "amount":       round(req.capital * w, 2),
            "last_price":   round(float(prices[sym].iloc[-1]), 4) if sym in prices.columns else None,
        }
        for sym, w in weights.items()
    }

    # 模擬組合績效
    port_ret = (returns * pd.Series(weights)).sum(axis=1)
    cum_ret  = (1 + port_ret).cumprod()
    ann_ret  = float(port_ret.mean() * 252)
    ann_vol  = float(port_ret.std() * np.sqrt(252))
    sharpe   = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    drawdown = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()

    return {
        "method":     req.method,
        "allocation": allocation,
        "portfolio_metrics": {
            "ann_return_pct":   round(ann_ret * 100, 2),
            "ann_vol_pct":      round(ann_vol * 100, 2),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown_pct": round(float(drawdown) * 100, 2),
        },
        "equity_curve": {
            "dates":  [str(d.date()) for d in cum_ret.index],
            "values": [round(float(v) * req.capital, 2) for v in cum_ret.values],
        },
        "errors": errors,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8001, reload=True)
