"""Motore di simulazione Monte Carlo su dati mensili.

Metodi:
  bootstrap  block bootstrap stazionario (Politis-Romano) dei mesi storici: preserva
             correlazioni, code grasse e un po' di autocorrelazione. Default consigliato.
  normal     normale multivariata sui rendimenti logaritmici (media/covarianza storiche)
  student_t  t di Student multivariata (df=5) con la stessa covarianza: code piu' grasse

Le aspettative possono essere modificate dinamicamente:
  mean_overrides  {id: rendimento annuo atteso}  sostituisce la media storica (forward looking)
  vol_scale       {id: moltiplicatore}           es. 1.2 = +20% di volatilita'
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .portfolio import Portfolio

PCTS = (5, 10, 25, 50, 75, 90, 95)


@dataclass
class MCConfig:
    horizon_years: int = 10
    n_sims: int = 5000
    method: str = "bootstrap"
    block_size: int = 12
    initial: float = 100_000.0
    monthly_flow: float = 0.0           # versamento (>0) o prelievo (<0) mensile
    flow_indexed: bool = False          # flussi rivalutati con l'inflazione simulata
    rebalance: str = "monthly"          # monthly | annual | none
    annual_cost: float = 0.0            # costi annui totali (es. 0.015)
    mean_overrides: dict[str, float] = field(default_factory=dict)
    vol_scale: dict[str, float] = field(default_factory=dict)
    inflation_id: str | None = "CPI_EZ"
    target: float | None = None         # obiettivo patrimoniale (nominale)
    seed: int | None = 42


@dataclass
class MCResult:
    dates: pd.DatetimeIndex
    paths_pct: pd.DataFrame             # percentili del patrimonio nel tempo
    real_paths_pct: pd.DataFrame | None
    final: np.ndarray
    final_real: np.ndarray | None
    invested: float
    cagr: np.ndarray
    max_dd: np.ndarray
    one_year: np.ndarray
    stats: pd.Series
    sample_paths: np.ndarray


def _stationary_bootstrap_idx(n_obs: int, n_sims: int, T: int, block: int, rng) -> np.ndarray:
    idx = np.empty((n_sims, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_sims)
    p = 1.0 / max(block, 1)
    for t in range(1, T):
        jump = rng.random(n_sims) < p
        idx[:, t] = np.where(jump, rng.integers(0, n_obs, n_sims), (idx[:, t - 1] + 1) % n_obs)
    return idx


def prepare_inputs(rets: pd.DataFrame, cfg: MCConfig) -> pd.DataFrame:
    """Applica override di media e scala di volatilita' sui rendimenti log storici."""
    lr = np.log1p(rets)
    for k, g in cfg.mean_overrides.items():
        if k in lr and g is not None:
            lr[k] = lr[k] - lr[k].mean() + np.log1p(g) / 12
    for k, s in cfg.vol_scale.items():
        if k in lr and s:
            m = lr[k].mean()
            lr[k] = m + (lr[k] - m) * s
    return lr


def simulate_asset_logrets(lr: pd.DataFrame, cfg: MCConfig, T: int, rng) -> np.ndarray:
    X = lr.values
    n_obs, N = X.shape
    if cfg.method == "bootstrap":
        idx = _stationary_bootstrap_idx(n_obs, cfg.n_sims, T, cfg.block_size, rng)
        return X[idx]                                           # (n_sims, T, N)
    mu, cov = X.mean(0), np.cov(X, rowvar=False)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(N))
    z = rng.standard_normal((cfg.n_sims, T, N))
    if cfg.method == "student_t":
        df = 5.0
        chi = rng.chisquare(df, (cfg.n_sims, T, 1))
        z = z * np.sqrt((df - 2) / chi)                         # varianza unitaria
    elif cfg.method != "normal":
        raise ValueError(f"Metodo sconosciuto: {cfg.method}")
    return mu + z @ L.T


def run_simulation(rets: pd.DataFrame, portfolio: Portfolio, cfg: MCConfig,
                   start_date: pd.Timestamp | None = None) -> MCResult:
    p = portfolio.normalized()
    cols = list(p.weights)
    use_inf = cfg.inflation_id and cfg.inflation_id in rets.columns
    all_cols = cols + ([cfg.inflation_id] if use_inf and cfg.inflation_id not in cols else [])
    hist = rets[all_cols].dropna()
    if len(hist) < 36:
        raise ValueError(f"Storico comune insufficiente ({len(hist)} mesi) per {all_cols}")
    rng = np.random.default_rng(cfg.seed)
    T = int(cfg.horizon_years * 12)
    lr = prepare_inputs(hist, cfg)
    sims = np.expm1(simulate_asset_logrets(lr, cfg, T, rng))  # rendimenti semplici
    w = np.array([p.weights[c] for c in cols])
    asset_r = sims[:, :, :len(cols)]
    infl = sims[:, :, len(cols)] if use_inf else None
    cost_m = (1 + cfg.annual_cost) ** (1 / 12) - 1

    n = cfg.n_sims
    W = np.empty((n, T + 1))
    W[:, 0] = cfg.initial
    if cfg.rebalance == "monthly":
        port_r = asset_r @ w
        holdings = None
    else:
        holdings = np.tile(w * cfg.initial, (n, 1))
    cpi = np.ones((n, T + 1))
    for t in range(T):
        if infl is not None:
            cpi[:, t + 1] = cpi[:, t] * (1 + infl[:, t])
        flow = cfg.monthly_flow * (cpi[:, t + 1] if cfg.flow_indexed else 1.0)
        if holdings is None:
            W[:, t + 1] = np.maximum(W[:, t] * (1 + port_r[:, t]) * (1 - cost_m) + flow, 0)
        else:
            holdings = holdings * (1 + asset_r[:, t, :]) * (1 - cost_m)
            tot = holdings.sum(1)
            newtot = np.maximum(tot + flow, 0)
            rebal = cfg.rebalance == "annual" and (t + 1) % 12 == 0
            if rebal:
                holdings = newtot[:, None] * w
            else:
                scale = np.divide(newtot, tot, out=np.zeros_like(tot), where=tot > 0)
                holdings = holdings * scale[:, None]
            W[:, t + 1] = newtot

    start = pd.Timestamp(start_date or pd.Timestamp.today()).to_period("M").to_timestamp("M")
    dates = pd.date_range(start, periods=T + 1, freq="ME")
    pct = pd.DataFrame(np.percentile(W, PCTS, axis=0).T, index=dates, columns=[f"P{q}" for q in PCTS])
    real = W / cpi if infl is not None else None
    real_pct = (pd.DataFrame(np.percentile(real, PCTS, axis=0).T, index=dates, columns=[f"P{q}" for q in PCTS])
                if real is not None else None)

    invested = cfg.initial + cfg.monthly_flow * T
    final = W[:, -1]
    years = cfg.horizon_years
    # rendimento annuo (time-weighted) del portafoglio, indipendente dai flussi
    if holdings is None:
        growth = np.prod(1 + port_r, axis=1) * (1 - cost_m) ** T
    else:
        growth = np.prod(1 + asset_r @ w, axis=1) * (1 - cost_m) ** T
    cagr = growth ** (1 / years) - 1
    peak = np.maximum.accumulate(W, axis=1)
    max_dd = np.min(np.divide(W, peak, out=np.ones_like(W), where=peak > 0) - 1, axis=1)
    one_year = W[:, min(12, T)] / W[:, 0] - 1 if cfg.monthly_flow == 0 else np.prod(1 + (asset_r[:, :12] @ w), 1) - 1
    var95 = np.percentile(one_year, 5)

    stats = {
        "Capitale iniziale": cfg.initial,
        "Totale versato/(prelevato)": invested,
        "Patrimonio finale mediano": np.median(final),
        "Patrimonio finale medio": final.mean(),
        "Scenario pessimistico (P5)": np.percentile(final, 5),
        "Scenario prudente (P25)": np.percentile(final, 25),
        "Scenario favorevole (P75)": np.percentile(final, 75),
        "Scenario ottimistico (P95)": np.percentile(final, 95),
        "Rend. annuo mediano": np.median(cagr),
        "Rend. annuo P5": np.percentile(cagr, 5),
        "Rend. annuo P95": np.percentile(cagr, 95),
        "Prob. perdita (finale < versato)": float((final < invested).mean()),
        "Prob. esaurimento capitale": float((final <= 0).mean()),
        "VaR 95% a 1 anno": var95,
        "CVaR 95% a 1 anno": one_year[one_year <= var95].mean(),
        "Max drawdown mediano": np.median(max_dd),
        "Max drawdown P5 (peggiori)": np.percentile(max_dd, 5),
        "Mesi di storico usati": len(hist),
    }
    if real is not None:
        stats["Patrimonio reale mediano (potere d'acquisto)"] = np.median(real[:, -1])
    if cfg.target:
        stats["Prob. raggiungere obiettivo"] = float((final >= cfg.target).mean())

    return MCResult(dates=dates, paths_pct=pct, real_paths_pct=real_pct, final=final,
                    final_real=real[:, -1] if real is not None else None, invested=invested, cagr=cagr,
                    max_dd=max_dd, one_year=one_year, stats=pd.Series(stats),
                    sample_paths=W[: min(50, n)])


def compare(results: dict[str, MCResult]) -> pd.DataFrame:
    return pd.DataFrame({k: v.stats for k, v in results.items()})
