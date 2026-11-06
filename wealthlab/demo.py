"""Dataset DIMOSTRATIVO sintetico (source='DEMO').

Serve solo per provare il software quando non c'e' accesso alle fonti dati: le serie
sono generate da un modello fattoriale con parametri plausibili, NON sono dati storici reali.
Il primo `update` con dati reali sostituisce automaticamente i dati DEMO.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .db import Database
from .universe import load_universe

# (rend. annuo, vol annua, beta globale, beta europa, beta tassi, beta commodity, beta usd)
EQ = {
    "MSCI_WORLD": (0.075, 0.15, 1.0, 0.0, 0, 0, 0), "SP500": (0.09, 0.15, 1.0, -0.2, 0, 0, 0),
    "NASDAQ100": (0.13, 0.20, 1.2, -0.3, 0, 0, 0), "EUROSTOXX50": (0.04, 0.18, 0.9, 0.6, 0, 0, 0),
    "FTSEMIB": (0.035, 0.22, 0.9, 0.8, 0, 0, 0), "DAX": (0.075, 0.19, 0.95, 0.55, 0, 0, 0),
    "CAC40": (0.045, 0.18, 0.9, 0.55, 0, 0, 0), "IBEX35": (0.03, 0.21, 0.85, 0.7, 0, 0, 0),
    "FTSE100": (0.035, 0.14, 0.75, 0.3, 0, 0.2, 0), "NIKKEI225": (0.06, 0.19, 0.8, 0, 0, 0, 0),
    "MSCI_EM": (0.05, 0.21, 1.0, 0, 0, 0.3, -0.3),
}
CMD = {"GOLD": (0.07, 0.16, 0.1, 0.0), "SILVER": (0.06, 0.29, 0.4, 0.3), "WTI": (0.02, 0.36, 0.4, 1.0),
       "BRENT": (0.03, 0.33, 0.4, 1.0), "NATGAS": (-0.05, 0.50, 0.2, 0.6), "COPPER": (0.05, 0.25, 0.6, 0.6),
       "GSCI": (0.0, 0.24, 0.4, 0.9)}
Y0 = {"BTP_10Y": 3.6, "BUND_10Y": 3.3, "OAT_10Y": 3.4, "BONOS_10Y": 3.4, "UST_10Y": 4.2, "UST_2Y": 3.0, "EUR_3M": 2.2}


def build_demo(db: Database, start="2005-01-31", end=None, seed=7) -> None:
    cfg = load_universe()
    db.upsert_instruments(cfg["instruments"])
    end = end or (pd.Timestamp.today() - pd.offsets.MonthEnd(1))
    dates = pd.date_range(start, end, freq="ME")
    T = len(dates)
    rng = np.random.default_rng(seed)
    # fattori con regimi di crisi (2008, 2011, 2020, 2022)
    crisis = np.zeros(T)
    for a, b in [("2007-11", "2009-02"), ("2011-07", "2011-11"), ("2020-02", "2020-03"), ("2022-01", "2022-09")]:
        crisis[(dates >= a) & (dates <= pd.Period(b).end_time)] = 1
    g = rng.standard_t(5, T) * 0.042 * np.sqrt(3 / 5) - crisis * 0.03
    eu = rng.standard_normal(T) * 0.025 - crisis * 0.01
    rates = rng.standard_normal(T) * 0.0018 - 0.5 * g * 0.01   # variazione tassi (decimali), risk-off -> tassi giu'
    cmdf = rng.standard_normal(T) * 0.05 + 0.3 * g
    usd = rng.standard_normal(T) * 0.022 + 0.2 * crisis * 0.02

    series = {}
    for k, (mu, vol, bg, be, _, bc, bu) in EQ.items():
        sys_ = bg * g + be * eu + bc * cmdf + bu * usd
        idio_vol = max(vol ** 2 / 12 - np.var(sys_), (0.03 ** 2)) ** 0.5
        r = mu / 12 - np.mean(sys_) + sys_ + rng.standard_normal(T) * idio_vol
        series[k] = 100 * np.cumprod(1 + r)
    for k, (mu, vol, bg, bc) in CMD.items():
        sys_ = bg * g + bc * cmdf
        idio = max(vol ** 2 / 12 - np.var(sys_), 0.02 ** 2) ** 0.5
        r = mu / 12 - np.mean(sys_) + sys_ + rng.standard_normal(T) * idio
        series[k] = 100 * np.cumprod(1 + r)
    # rendimenti: percorso storico stilizzato (alti 2005-08, bassi 2015-21, rialzo 2022) + shock
    t = np.linspace(0, 1, T)
    regime = np.interp(t, [0, .17, .45, .62, .8, .87, 1], [3.5, 4.3, 2.0, 0.6, -0.3, 2.5, 2.7])
    spread_it = 1.0 + 3.5 * np.exp(-((t - .38) / .05) ** 2) + 1.2 * np.exp(-((t - .2) / .03) ** 2)
    noise = np.cumsum(rates * 100) * 0.5
    series["BUND_10Y"] = regime + noise
    series["OAT_10Y"] = regime + noise + 0.4 + rng.standard_normal(T) * 0.05
    series["BTP_10Y"] = regime + noise + spread_it + np.cumsum(rng.standard_normal(T) * 0.06) * 0.3
    series["BONOS_10Y"] = regime + noise + spread_it * 0.8 + rng.standard_normal(T) * 0.08
    series["UST_10Y"] = regime + 1.4 + noise * 1.2 + rng.standard_normal(T) * 0.08
    series["UST_2Y"] = np.clip(series["UST_10Y"] - 1.0 + rng.standard_normal(T) * 0.1, 0.1, None)
    series["EUR_3M"] = np.clip(regime - 0.6, -0.55, None)
    series["EURUSD"] = 1.30 * np.exp(np.cumsum(usd * -0.5))
    series["EURGBP"] = 0.80 * np.exp(np.cumsum(rng.standard_normal(T) * 0.018))
    series["EURJPY"] = 140 * np.exp(np.cumsum(rng.standard_normal(T) * 0.025 + crisis * -0.01))
    infl_ez = 0.0016 + 0.004 * np.exp(-((t - .9) / .04) ** 2) + rng.standard_normal(T) * 0.002
    series["CPI_EZ"] = 90 * np.cumprod(1 + infl_ez)
    series["CPI_US"] = 200 * np.cumprod(1 + infl_ez + 0.0004 + rng.standard_normal(T) * 0.001)

    db.delete_source("DEMO")
    known = set(db.instruments().index)
    for k, v in series.items():
        if k in known and db.last_date(k) is None:   # non sovrascrive dati reali
            db.upsert_series(k, pd.Series(v, index=dates), "DEMO")

    # due fondi dimostrativi (attivi) per testare l'analisi "dove inserire i fondi"
    demo_funds = [
        {"id": "DEMO_FONDO_AZ_GLOBALE", "name": "[DEMO] Fondo Azionario Globale Quality 4*", "asset_class": "FUND",
         "region": "WORLD", "currency": "EUR", "source": "csv", "symbol": "demo", "isin": "XX0000000001",
         "morningstar_stars": 4, "fund_type": "EQUITY", "proxy": "MSCI_WORLD"},
        {"id": "DEMO_FONDO_FLEX_BOND", "name": "[DEMO] Fondo Obbligazionario Flessibile 4*", "asset_class": "FUND",
         "region": "EUROPE", "currency": "EUR", "source": "csv", "symbol": "demo", "isin": "XX0000000002",
         "morningstar_stars": 4, "fund_type": "BOND", "proxy": "BUND_10Y"},
    ]
    db.upsert_instruments(demo_funds)
    fx = pd.Series(series["EURUSD"], index=dates)
    world_eur = pd.Series(series["MSCI_WORLD"], index=dates).pct_change() * (fx.shift(1) / fx)
    fa = 0.85 * world_eur.fillna(0) + 0.0012 + rng.standard_normal(T) * 0.012
    fb = 0.0022 + 0.3 * (-np.diff(np.r_[series["BUND_10Y"][0], series["BUND_10Y"]]) / 100 * 4) + 0.08 * g + rng.standard_normal(T) * 0.006
    db.upsert_series("DEMO_FONDO_AZ_GLOBALE", pd.Series(100 * np.cumprod(1 + fa.values), index=dates), "DEMO")
    db.upsert_series("DEMO_FONDO_FLEX_BOND", pd.Series(100 * np.cumprod(1 + fb), index=dates), "DEMO")
