"""Aggiornamento incrementale del database."""
from __future__ import annotations

import pandas as pd

from .db import Database
from .loaders import fetch_csv, fetch_fred, fetch_yahoo
from .universe import load_universe

FETCHERS = {"yahoo": fetch_yahoo, "fred": fetch_fred, "csv": fetch_csv}


def update_database(db: Database, universe_path=None, full: bool = False, only: list[str] | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """Scarica i dati mancanti. Di default riparte dagli ultimi 3 mesi gia' presenti
    (per catturare revisioni e il mese corrente non ancora chiuso)."""
    cfg = load_universe(universe_path) if universe_path else load_universe()
    db.upsert_instruments(cfg["instruments"])
    report = []
    for ins in cfg["instruments"]:
        if only and ins["id"] not in only:
            continue
        start = cfg.get("history_start", "2005-01-01")
        last = None if full else db.last_date(ins["id"])
        if last is not None:
            start = (last - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        try:
            s = FETCHERS[ins["source"]](str(ins["symbol"]), start)
            # i dati reali sostituiscono eventuali dati DEMO dello stesso strumento
            db.delete_source("DEMO", ins["id"])
            n = db.upsert_series(ins["id"], s, ins["source"])
            db.log(ins["id"], n, "OK")
            report.append({"id": ins["id"], "righe": n, "esito": "OK", "ultimo": s.index.max().date() if n else None})
        except Exception as e:  # noqa: BLE001 - un errore su una fonte non deve bloccare le altre
            db.log(ins["id"], 0, "ERRORE", str(e)[:500])
            report.append({"id": ins["id"], "righe": 0, "esito": f"ERRORE: {str(e)[:120]}", "ultimo": None})
        if verbose:
            r = report[-1]
            print(f"  {r['id']:<14} {r['esito']:<10} righe={r['righe']}")
    return pd.DataFrame(report)
