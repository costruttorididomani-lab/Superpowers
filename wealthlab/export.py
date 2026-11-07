"""Esportazione completa del database in un file Excel consultabile senza software."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analytics import (asset_returns, calendar_returns, correlation, risk_free, stress_correlation,
                        summary_stats)
from .db import ROOT, Database
from .report import to_excel


def export_snapshot(db: Database, path: str | Path | None = None) -> Path:
    path = Path(path or ROOT / "data" / "snapshot" / "database_mensile.xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = db.instruments()
    inv = [i for i in meta.index if meta.loc[i, "asset_class"] not in ("FX", "MACRO")]
    rets = asset_returns(db, inv)
    anag = meta.drop(columns=["meta"]).join(db.coverage().set_index("id")[["dal", "al", "mesi", "fonte_dati"]])
    lv = db.levels()
    lv.index = lv.index.strftime("%Y-%m-%d")
    to_excel({
        "Anagrafica": anag,
        "Livelli mensili": lv,
        "Rendimenti mensili EUR": rets,
        "Statistiche EUR": summary_stats(rets, risk_free(db)),
        "Rendimenti annuali EUR": calendar_returns(rets.dropna(how="all")),
        "Correlazioni": correlation(rets).round(3),
        "Correlazioni crisi azionaria": stress_correlation(rets, "MSCI_WORLD", 0.2).round(3),
    }, path)
    return path
