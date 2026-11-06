"""Import e analisi della posizione di un cliente."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .portfolio import Portfolio

COLUMN_ALIASES = {
    "description": ["description", "descrizione", "titolo", "strumento", "nome"],
    "isin": ["isin", "codice isin"],
    "amount": ["amount", "controvalore", "valore", "importo", "ctv"],
    "instrument_id": ["instrument_id", "building_block", "proxy", "id", "classe"],
}


def parse_amount(v) -> float:
    """Accetta 150000 | 150000.50 | 150.000 | 150.000,50 | 150,000.50 | '€ 1.234'."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace("€", "").replace(" ", "").strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) != 3 else s.replace(",", "")
    elif s.count(".") > 1 or (s.count(".") == 1 and len(s.split(".")[-1]) == 3):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_positions(src) -> pd.DataFrame:
    """Legge CSV/XLSX con colonne: descrizione, isin, controvalore, building_block."""
    if isinstance(src, (str, Path)) and str(src).endswith(".xlsx"):
        df = pd.read_excel(src)
    elif hasattr(src, "name") and str(src.name).endswith(".xlsx"):
        df = pd.read_excel(src)
    else:
        raw = src.read() if hasattr(src, "read") else Path(src).read_bytes()
        txt = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        sep = ";" if txt.count(";") > txt.count(",") else ","
        df = pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, keep_default_na=False)
        df = df.replace("", pd.NA)
    cols = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    for std, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in cols:
                out[std] = df[cols[a]]
                break
    if "amount" not in out:
        raise ValueError("Colonna del controvalore mancante (controvalore/amount)")
    out["amount"] = out["amount"].map(parse_amount)
    for c in ("description", "isin", "instrument_id"):
        if c not in out:
            out[c] = ""
    return out.fillna({"description": "", "isin": "", "instrument_id": ""})


def map_positions(pos: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Associa ogni posizione a un building block del database: prima per id esplicito,
    poi per ISIN registrato nell'anagrafica fondi. Restituisce (mappate, non mappate)."""
    isin_map = {str((m or {}).get("isin", "")).upper(): i for i, m in meta["meta"].items() if (m or {}).get("isin")}
    pos = pos.copy()
    ids = []
    for _, r in pos.iterrows():
        iid = str(r["instrument_id"]).strip().upper()
        if iid in meta.index:
            ids.append(iid)
        elif str(r["isin"]).upper() in isin_map:
            ids.append(isin_map[str(r["isin"]).upper()])
        else:
            ids.append(None)
    pos["instrument_id"] = ids
    return pos[pos["instrument_id"].notna()], pos[pos["instrument_id"].isna()]


def positions_to_portfolio(pos: pd.DataFrame, name: str = "Cliente") -> Portfolio:
    w = pos.groupby("instrument_id")["amount"].sum()
    return Portfolio(weights=w.to_dict(), name=name).normalized()
