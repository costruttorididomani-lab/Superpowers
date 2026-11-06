"""Rendimenti mensili in valuta base, statistiche di rischio/rendimento e correlazioni."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .db import Database

FX_BY_CCY = {"USD": "EURUSD", "GBP": "EURGBP", "JPY": "EURJPY"}

# Episodi storici di stress per gli stress test "replay"
STRESS_EPISODES = {
    "Crisi finanziaria 2008 (ott-07 / feb-09)": ("2007-11-30", "2009-02-28"),
    "Crisi debito sovrano 2011 (lug-11 / nov-11)": ("2011-07-31", "2011-11-30"),
    "Brexit / Italia 2016 (giu-16)": ("2016-06-30", "2016-06-30"),
    "Q4 2018 (ott-18 / dic-18)": ("2018-10-31", "2018-12-31"),
    "Covid 2020 (feb-20 / mar-20)": ("2020-02-29", "2020-03-31"),
    "Shock tassi 2022 (gen-22 / set-22)": ("2022-01-31", "2022-09-30"),
}


def convexity_from_duration(d: float) -> float:
    """Approssimazione della convessita' di un titolo a tasso fisso a partire dalla duration."""
    return d * d + d


def bond_total_return(yields_pct: pd.Series, duration: float) -> pd.Series:
    """Rendimento totale mensile di un building block obbligazionario a duration costante.

    r_t = y_{t-1}/12 - D * dy + 1/2 * C * dy^2   (y, dy in decimali)
    Approssimazione standard per ricostruire la performance di un portafoglio
    governativo a duration costante da una serie di rendimenti.
    """
    y = yields_pct / 100.0
    dy = y.diff()
    c = convexity_from_duration(duration)
    return (y.shift(1) / 12.0 - duration * dy + 0.5 * c * dy ** 2).iloc[1:]


def asset_returns(db: Database, ids: list[str] | None = None, base_currency: str = "EUR",
                  hedged: bool | dict = False, durations: dict[str, float] | None = None,
                  start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Matrice dei rendimenti mensili (decimali) in valuta base.

    hedged: True/False per tutti, oppure dict {id: bool}. La copertura valutaria e'
    approssimata come rendimento in valuta locale (differenziale di tasso ignorato).
    durations: override della duration per i building block obbligazionari.
    """
    meta = db.instruments()
    ids = list(meta.index) if ids is None else list(ids)
    fx_needed = {FX_BY_CCY[c] for c in meta.loc[ids, "currency"] if c in FX_BY_CCY}
    lv = db.levels(sorted(set(ids) | fx_needed), start=start, end=end)
    durations = durations or {}
    out = {}
    for i in ids:
        if i not in lv.columns:
            continue
        m = meta.loc[i]
        s = lv[i].dropna()
        if len(s) < 3:
            continue
        if m["kind"] == "yield":
            d = float(durations.get(i, m["duration"] if pd.notna(m["duration"]) else 0.25))
            r = bond_total_return(s, d)
        elif m["asset_class"] == "FX":
            r = s.shift(1) / s - 1  # detenere la valuta estera per un investitore EUR
            out[i] = r.iloc[1:]
            continue
        else:
            r = s.pct_change().iloc[1:]
        is_hedged = hedged.get(i, False) if isinstance(hedged, dict) else hedged
        ccy = m["currency"]
        if ccy != base_currency and not is_hedged and m["asset_class"] != "MACRO" and ccy in FX_BY_CCY:
            fx = lv[FX_BY_CCY[ccy]].dropna()
            fx_r = (fx.shift(1) / fx).reindex(r.index)
            r = (1 + r) * fx_r - 1
        out[i] = r
    return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------- statistiche
def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1).min()) if len(w) else np.nan


def summary_stats(rets: pd.DataFrame, rf: pd.Series | None = None) -> pd.DataFrame:
    rows = {}
    for c in rets.columns:
        r = rets[c].dropna()
        if len(r) < 12:
            continue
        n = len(r)
        cagr = (1 + r).prod() ** (12 / n) - 1
        vol = r.std() * np.sqrt(12)
        ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
        sharpe = ex.mean() * 12 / vol if vol > 0 else np.nan
        down = r[r < 0].std() * np.sqrt(12)
        q05 = r.quantile(0.05)
        rows[c] = {
            "Dal": r.index.min().strftime("%Y-%m"), "Mesi": n,
            "Rend. annuo (CAGR)": cagr, "Volatilita": vol, "Sharpe": sharpe,
            "Sortino": (ex.mean() * 12 / down) if down > 0 else np.nan,
            "Max Drawdown": max_drawdown(r), "Peggior mese": r.min(), "Miglior mese": r.max(),
            "VaR 95% mensile": q05, "CVaR 95% mensile": r[r <= q05].mean(),
            "Asimmetria": r.skew(), "Curtosi": r.kurt(), "% mesi positivi": (r > 0).mean(),
        }
    return pd.DataFrame(rows).T


def calendar_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti per anno solare."""
    return (1 + rets).groupby(rets.index.year).prod() - 1


def correlation(rets: pd.DataFrame, method: str = "pearson", min_periods: int = 24) -> pd.DataFrame:
    return rets.corr(method=method, min_periods=min_periods)


def rolling_correlation(rets: pd.DataFrame, a: str, b: str, window: int = 36) -> pd.Series:
    return rets[a].rolling(window).corr(rets[b]).dropna()


def stress_correlation(rets: pd.DataFrame, driver: str, quantile: float = 0.2) -> pd.DataFrame:
    """Correlazioni calcolate solo nei mesi peggiori del 'driver' (es. MSCI World):
    mostra come cambiano le correlazioni quando servono davvero (diversificazione in crisi)."""
    r = rets.dropna(subset=[driver])
    thr = r[driver].quantile(quantile)
    return r[r[driver] <= thr].corr(min_periods=12)


def regime_table(rets: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Rendimento medio mensile di ogni asset nei mesi up/down del driver."""
    r = rets.dropna(subset=[driver])
    up, down = r[r[driver] > 0], r[r[driver] <= 0]
    return pd.DataFrame({
        f"Media mesi {driver} positivo": up.mean(), f"Media mesi {driver} negativo": down.mean(),
        "Beta vs driver": r.apply(lambda c: c.cov(r[driver]) / r[driver].var() if c.notna().sum() > 12 else np.nan),
    })


def risk_free(db: Database, start=None, end=None) -> pd.Series | None:
    lv = db.levels(["EUR_3M"], start=start, end=end)
    if lv.empty:
        return None
    return (lv["EUR_3M"] / 1200).shift(1).dropna()
