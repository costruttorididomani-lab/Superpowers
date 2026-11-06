"""Portafogli di building block e operatori di scenario (le leve dinamiche del simulatore)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

GROWTH_CLASSES = {"EQUITY"}
DEFENSIVE_CLASSES = {"BOND", "CASH"}


@dataclass
class Portfolio:
    weights: dict[str, float]
    durations: dict[str, float] = field(default_factory=dict)   # override duration building block BOND
    hedged: dict[str, bool] = field(default_factory=dict)       # copertura valutaria per strumento
    name: str = "Portafoglio"

    def normalized(self) -> "Portfolio":
        w = {k: float(v) for k, v in self.weights.items() if abs(v) > 1e-9}
        tot = sum(w.values())
        if tot <= 0:
            raise ValueError("Pesi nulli")
        return replace(self, weights={k: v / tot for k, v in w.items()})

    def series(self) -> pd.Series:
        return pd.Series(self.weights, dtype=float).sort_values(ascending=False)


def macro_class(meta: pd.DataFrame, asset_id: str) -> str:
    """Classe 'economica': i fondi vengono ricondotti alla loro natura (fund_type)."""
    row = meta.loc[asset_id]
    if row["asset_class"] == "FUND":
        ft = (row["meta"] or {}).get("fund_type", "EQUITY")
        return "EQUITY" if ft in ("EQUITY", "AZIONARIO") else ("BOND" if ft in ("BOND", "OBBLIGAZIONARIO") else "OTHER")
    return row["asset_class"]


def class_weights(p: Portfolio, meta: pd.DataFrame, by: str = "macro") -> pd.Series:
    out: dict[str, float] = {}
    for k, v in p.weights.items():
        key = macro_class(meta, k) if by == "macro" else meta.loc[k, by]
        out[key] = out.get(key, 0) + v
    return pd.Series(out).sort_values(ascending=False)


def portfolio_duration(p: Portfolio, meta: pd.DataFrame) -> float:
    num = den = 0.0
    for k, v in p.weights.items():
        if meta.loc[k, "asset_class"] in DEFENSIVE_CLASSES:
            d = p.durations.get(k, meta.loc[k, "duration"])
            if pd.notna(d):
                num += v * float(d)
                den += v
    return num / den if den else 0.0


# --------------------------------------------------------------- operatori
def set_equity_share(p: Portfolio, share: float, meta: pd.DataFrame) -> Portfolio:
    """Porta la quota azionaria a `share` (0-1) del totale azioni+obbligazioni,
    riscalando proporzionalmente le componenti; materie prime e altro restano invariati."""
    w = dict(p.weights)
    eq = {k: v for k, v in w.items() if macro_class(meta, k) in GROWTH_CLASSES}
    bd = {k: v for k, v in w.items() if macro_class(meta, k) in DEFENSIVE_CLASSES}
    core = sum(eq.values()) + sum(bd.values())
    if core == 0:
        return p
    se, sb = sum(eq.values()), sum(bd.values())
    if se == 0 or sb == 0:
        raise ValueError("Per cambiare il mix servono sia componenti azionarie che obbligazionarie")
    for k, v in eq.items():
        w[k] = v / se * share * core
    for k, v in bd.items():
        w[k] = v / sb * (1 - share) * core
    return replace(p, weights=w).normalized()


def tilt(p: Portfolio, asset_id: str, delta: float, meta: pd.DataFrame, funded_from: str = "same_class") -> Portfolio:
    """Aumenta (delta>0) o riduce il peso di `asset_id` di `delta` punti (0.05 = +5%).

    funded_from: 'same_class' (pro-quota sugli altri strumenti della stessa macro-classe),
                 'all' (pro-quota su tutto il resto), oppure l'id di uno strumento specifico.
    """
    w = dict(p.weights)
    w.setdefault(asset_id, 0.0)
    if funded_from in ("same_class", "all"):
        cls = macro_class(meta, asset_id)
        pool = {k: v for k, v in w.items() if k != asset_id and v > 0 and
                (funded_from == "all" or macro_class(meta, k) == cls)}
        if not pool:  # nessuno nella stessa classe: finanzia da tutto il resto
            pool = {k: v for k, v in w.items() if k != asset_id and v > 0}
        tot = sum(pool.values())
        delta = min(delta, tot) if delta > 0 else max(delta, -w[asset_id])
        for k, v in pool.items():
            w[k] = v - delta * v / tot
    else:
        w.setdefault(funded_from, 0.0)
        delta = min(delta, w[funded_from]) if delta > 0 else max(delta, -w[asset_id])
        w[funded_from] -= delta
    w[asset_id] += delta
    return replace(p, weights=w).normalized()


def set_duration(p: Portfolio, target: float, meta: pd.DataFrame) -> Portfolio:
    """Imposta la duration di tutti i building block governativi (escluso il monetario)."""
    d = dict(p.durations)
    for k in p.weights:
        if meta.loc[k, "asset_class"] == "BOND":
            d[k] = float(target)
    return replace(p, durations=d)


def shift_duration(p: Portfolio, delta_years: float, meta: pd.DataFrame) -> Portfolio:
    d = dict(p.durations)
    for k in p.weights:
        if meta.loc[k, "asset_class"] == "BOND":
            base = d.get(k, meta.loc[k, "duration"])
            d[k] = max(0.1, float(base) + delta_years)
    return replace(p, durations=d)


def add_fund(p: Portfolio, fund_id: str, weight: float, meta: pd.DataFrame, replace_id: str | None = None) -> Portfolio:
    """Inserisce un fondo con peso `weight` finanziandolo dalla sua stessa macro-classe
    (o da uno strumento specifico `replace_id`, es. sostituire parte dell'indice con il fondo attivo)."""
    return tilt(p, fund_id, weight, meta, funded_from=replace_id or "same_class")


@dataclass
class ScenarioParams:
    """Parametri dinamici del simulatore (tutti facoltativi)."""
    equity_share: float | None = None                         # quota azionaria su azioni+obbligazioni
    tilts: dict[str, float] = field(default_factory=dict)     # {id: +/- punti} es. {"MSCI_WORLD": 0.05}
    duration: float | None = None                              # duration target governativi
    duration_shift: float | None = None                        # oppure +/- anni
    funds: dict[str, float] = field(default_factory=dict)     # {fund_id: peso}
    fund_replace: dict[str, str] = field(default_factory=dict)  # {fund_id: id da sostituire}
    hedge_usd: bool | None = None


def apply_scenario(p: Portfolio, sp: ScenarioParams, meta: pd.DataFrame) -> Portfolio:
    q = p.normalized()
    if sp.equity_share is not None:
        q = set_equity_share(q, sp.equity_share, meta)
    for k, dlt in sp.tilts.items():
        if dlt:
            q = tilt(q, k, dlt, meta)
    for f, wt in sp.funds.items():
        if wt:
            q = add_fund(q, f, wt, meta, sp.fund_replace.get(f))
    if sp.duration is not None:
        q = set_duration(q, sp.duration, meta)
    elif sp.duration_shift:
        q = shift_duration(q, sp.duration_shift, meta)
    if sp.hedge_usd is not None:
        h = dict(q.hedged)
        for k in q.weights:
            if meta.loc[k, "currency"] != "EUR":
                h[k] = sp.hedge_usd
        q = replace(q, hedged=h)
    return replace(q, name=f"{p.name} (scenario)")


# ------------------------------------------------- analisi di portafoglio
def portfolio_returns(rets: pd.DataFrame, p: Portfolio) -> pd.Series:
    """Rendimenti storici con ribilanciamento mensile (solo date con tutti i dati)."""
    cols = list(p.weights)
    r = rets[cols].dropna()
    w = np.array([p.weights[c] for c in cols])
    return pd.Series(r.values @ w, index=r.index, name=p.name)


def risk_contribution(rets: pd.DataFrame, p: Portfolio) -> pd.DataFrame:
    """Contributo di ciascun building block alla volatilita' del portafoglio."""
    cols = list(p.weights)
    r = rets[cols].dropna()
    cov = r.cov().values * 12
    w = np.array([p.weights[c] for c in cols])
    vol = float(np.sqrt(w @ cov @ w))
    mrc = cov @ w / vol
    rc = w * mrc
    return pd.DataFrame({"Peso": w, "Contributo vol.": rc, "% del rischio": rc / vol,
                         "Vol. marginale": mrc}, index=cols).sort_values("% del rischio", ascending=False)


def historical_stress(rets: pd.DataFrame, portfolios: list[Portfolio], episodes: dict) -> pd.DataFrame:
    rows = {}
    for name, (a, b) in episodes.items():
        row = {}
        for p in portfolios:
            cols = list(p.weights)
            sub = rets.loc[a:b, cols]
            if sub.empty or sub.isna().any().any():
                row[p.name] = np.nan
                continue
            pr = sub.values @ np.array([p.weights[c] for c in cols])
            row[p.name] = float(np.prod(1 + pr) - 1)
        rows[name] = row
    return pd.DataFrame(rows).T


def fund_fit(rets: pd.DataFrame, base: Portfolio, fund_ids: list[str], proxies: dict[str, str],
             meta: pd.DataFrame, weight: float = 0.10, rf: pd.Series | None = None) -> pd.DataFrame:
    """Dove e' utile inserire un fondo: per ogni fondo confronta il portafoglio base con
    quello in cui il fondo sostituisce `weight` del proprio indice di riferimento (proxy)."""
    from .analytics import max_drawdown

    base_r = portfolio_returns(rets, base)
    rows = {}
    for f in fund_ids:
        px = proxies.get(f)
        fr = rets[f].dropna()
        if len(fr) < 24:
            continue
        alt = add_fund(base, f, weight, meta, replace_id=px if px in base.weights else None)
        alt_r = portfolio_returns(rets, alt)
        idx = alt_r.index.intersection(base_r.index)
        b, a = base_r.loc[idx], alt_r.loc[idx]
        row = {"Mesi comuni": len(idx), "Corr. con portafoglio": fr.corr(base_r)}
        if px and px in rets:
            j = pd.concat([fr, rets[px]], axis=1).dropna()
            beta = j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var()
            te = (j.iloc[:, 0] - j.iloc[:, 1]).std() * np.sqrt(12)
            alpha = (j.iloc[:, 0].mean() - beta * j.iloc[:, 1].mean()) * 12
            row.update({"Proxy": px, "Beta vs proxy": beta, "Alpha annuo": alpha,
                        "Tracking error": te, "Information ratio": alpha / te if te > 0 else np.nan})
        row.update({
            "Delta rend. portafoglio": ((1 + a).prod() ** (12 / len(a)) - 1) - ((1 + b).prod() ** (12 / len(b)) - 1),
            "Delta volatilita": (a.std() - b.std()) * np.sqrt(12),
            "Delta max drawdown": max_drawdown(a) - max_drawdown(b),
        })
        rows[f] = row
    return pd.DataFrame(rows).T
