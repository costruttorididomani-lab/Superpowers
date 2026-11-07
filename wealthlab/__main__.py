"""CLI:  python -m wealthlab <comando>

  init      crea il database e registra l'universo strumenti
  update    aggiorna i dati (incrementale; --full per riscaricare tutto; --only ID1,ID2)
  demo      popola il database con dati DIMOSTRATIVI sintetici (per prove offline)
  status    copertura dati per strumento
  corr      matrice di correlazione (--ids ...)
  export    esporta il database in Excel (data/snapshot/database_mensile.xlsx)
  simulate  simulazione rapida  (--weights MSCI_WORLD=0.5,BTP_10Y=0.5 --years 10)
"""
from __future__ import annotations

import argparse

import pandas as pd

from .analytics import asset_returns, correlation, summary_stats
from .db import Database
from .demo import build_demo
from .montecarlo import MCConfig, run_simulation
from .portfolio import Portfolio
from .universe import load_universe
from .updater import update_database


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wealthlab")
    ap.add_argument("cmd", choices=["init", "update", "demo", "status", "corr", "export", "simulate"])
    ap.add_argument("--db", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--weights", default="MSCI_WORLD=0.5,BTP_10Y=0.3,BUND_10Y=0.2")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--sims", type=int, default=5000)
    a = ap.parse_args(argv)
    db = Database(a.db) if a.db else Database()
    pd.set_option("display.width", 200, "display.max_columns", 30)

    if a.cmd == "init":
        db.upsert_instruments(load_universe()["instruments"])
        print(f"Database pronto: {db.path}")
    elif a.cmd == "update":
        print("Aggiornamento in corso...")
        rep = update_database(db, full=a.full, only=a.only.split(",") if a.only else None)
        ok = (rep["esito"] == "OK").sum()
        print(f"\nCompletato: {ok}/{len(rep)} strumenti aggiornati.")
    elif a.cmd == "demo":
        build_demo(db)
        print("Dati DIMOSTRATIVI caricati (sintetici, non storici). Usa 'update' per i dati reali.")
    elif a.cmd == "status":
        print(db.coverage().to_string(index=False))
    elif a.cmd == "corr":
        ids = a.ids.split(",") if a.ids else None
        print(correlation(asset_returns(db, ids)).round(2).to_string())
    elif a.cmd == "export":
        from .export import export_snapshot
        print(f"Esportato: {export_snapshot(db)}")
    elif a.cmd == "simulate":
        w = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
        rets = asset_returns(db)
        res = run_simulation(rets, Portfolio(w), MCConfig(horizon_years=a.years, n_sims=a.sims))
        print(summary_stats(rets[list(w)]).iloc[:, 2:7].to_string())
        print()
        print(res.stats.to_string())


if __name__ == "__main__":
    main()
