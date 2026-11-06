"""Livello di persistenza: database SQLite con serie storiche mensili.

Schema (scalabile: aggiungere strumenti non richiede modifiche allo schema):
  instruments  anagrafica strumenti (sincronizzata da config/universe.yaml)
  observations serie storiche mensili (instrument_id, date fine mese, value)
  update_log   storico degli aggiornamenti
  clients / positions  posizioni dei clienti
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "wealthlab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    asset_class  TEXT NOT NULL,
    region       TEXT,
    currency     TEXT,
    source       TEXT,
    symbol       TEXT,
    kind         TEXT DEFAULT 'price',
    duration     REAL,
    meta         TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    instrument_id TEXT NOT NULL REFERENCES instruments(id),
    date          TEXT NOT NULL,          -- fine mese YYYY-MM-DD
    value         REAL NOT NULL,
    source        TEXT,                   -- yahoo | fred | csv | DEMO
    updated_at    TEXT,
    PRIMARY KEY (instrument_id, date)
);
CREATE INDEX IF NOT EXISTS idx_obs_date ON observations(date);
CREATE TABLE IF NOT EXISTS update_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT,
    instrument_id TEXT,
    rows          INTEGER,
    status        TEXT,
    message       TEXT
);
CREATE TABLE IF NOT EXISTS clients (
    id         TEXT PRIMARY KEY,
    name       TEXT,
    profile    TEXT,
    notes      TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    client_id     TEXT REFERENCES clients(id),
    instrument_id TEXT,           -- id del building block / proxy nel database
    description   TEXT,
    isin          TEXT,
    amount        REAL,           -- controvalore in valuta base
    as_of         TEXT,
    PRIMARY KEY (client_id, instrument_id, description)
);
"""

CORE_COLS = {"id", "name", "asset_class", "region", "currency", "source", "symbol", "kind", "duration"}


class Database:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------ anagrafica
    def upsert_instruments(self, instruments: list[dict]) -> None:
        rows = []
        for ins in instruments:
            meta = {k: v for k, v in ins.items() if k not in CORE_COLS}
            rows.append((
                ins["id"], ins["name"], ins["asset_class"], ins.get("region"), ins.get("currency", "EUR"),
                ins.get("source"), str(ins.get("symbol", "")), ins.get("kind", "price"), ins.get("duration"),
                json.dumps(meta),
            ))
        with self.connect() as con:
            con.executemany(
                """INSERT INTO instruments (id,name,asset_class,region,currency,source,symbol,kind,duration,meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, asset_class=excluded.asset_class,
                     region=excluded.region, currency=excluded.currency, source=excluded.source,
                     symbol=excluded.symbol, kind=excluded.kind, duration=excluded.duration, meta=excluded.meta""",
                rows,
            )

    def instruments(self) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql("SELECT * FROM instruments ORDER BY asset_class, id", con)
        df["meta"] = df["meta"].apply(lambda s: json.loads(s) if s else {})
        return df.set_index("id")

    # ---------------------------------------------------------------- serie
    def upsert_series(self, instrument_id: str, series: pd.Series, source: str) -> int:
        s = series.dropna()
        if s.empty:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        rows = [(instrument_id, pd.Timestamp(d).strftime("%Y-%m-%d"), float(v), source, now) for d, v in s.items()]
        with self.connect() as con:
            con.executemany(
                """INSERT INTO observations (instrument_id,date,value,source,updated_at) VALUES (?,?,?,?,?)
                   ON CONFLICT(instrument_id,date) DO UPDATE SET value=excluded.value,
                     source=excluded.source, updated_at=excluded.updated_at""",
                rows,
            )
        return len(rows)

    def last_date(self, instrument_id: str, exclude_source: str | None = "DEMO") -> pd.Timestamp | None:
        q = "SELECT MAX(date) FROM observations WHERE instrument_id=?"
        params: list = [instrument_id]
        if exclude_source:
            q += " AND source<>?"
            params.append(exclude_source)
        with self.connect() as con:
            r = con.execute(q, params).fetchone()[0]
        return pd.Timestamp(r) if r else None

    def delete_source(self, source: str, instrument_id: str | None = None) -> None:
        with self.connect() as con:
            if instrument_id:
                con.execute("DELETE FROM observations WHERE source=? AND instrument_id=?", (source, instrument_id))
            else:
                con.execute("DELETE FROM observations WHERE source=?", (source,))

    def levels(self, ids: list[str] | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Matrice date x strumenti dei livelli (prezzi, rendimenti %, indici CPI)."""
        q = "SELECT instrument_id, date, value FROM observations WHERE 1=1"
        params: list = []
        if ids:
            q += f" AND instrument_id IN ({','.join('?' * len(ids))})"
            params += list(ids)
        if start:
            q += " AND date>=?"
            params.append(start)
        if end:
            q += " AND date<=?"
            params.append(end)
        with self.connect() as con:
            df = pd.read_sql(q, con, params=params)
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot(index="date", columns="instrument_id", values="value").sort_index()

    def coverage(self) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql(
                """SELECT i.id, i.name, i.asset_class, i.region, i.source AS fonte_cfg,
                          MIN(o.date) AS dal, MAX(o.date) AS al, COUNT(o.date) AS mesi,
                          GROUP_CONCAT(DISTINCT o.source) AS fonte_dati
                   FROM instruments i LEFT JOIN observations o ON o.instrument_id=i.id
                   GROUP BY i.id ORDER BY i.asset_class, i.id""", con)
        return df

    def log(self, instrument_id: str, rows: int, status: str, message: str = "") -> None:
        with self.connect() as con:
            con.execute("INSERT INTO update_log (run_at,instrument_id,rows,status,message) VALUES (?,?,?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), instrument_id, rows, status, message))

    def update_history(self, limit: int = 200) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql("SELECT * FROM update_log ORDER BY id DESC LIMIT ?", con, params=[limit])

    # -------------------------------------------------------------- clienti
    def save_client(self, client_id: str, name: str, positions: pd.DataFrame, profile: str = "", notes: str = "") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as con:
            con.execute("""INSERT INTO clients (id,name,profile,notes,created_at) VALUES (?,?,?,?,?)
                           ON CONFLICT(id) DO UPDATE SET name=excluded.name, profile=excluded.profile, notes=excluded.notes""",
                        (client_id, name, profile, notes, now))
            con.execute("DELETE FROM positions WHERE client_id=?", (client_id,))
            con.executemany(
                "INSERT INTO positions (client_id,instrument_id,description,isin,amount,as_of) VALUES (?,?,?,?,?,?)",
                [(client_id, r.instrument_id, r.get("description", r.instrument_id), r.get("isin", ""),
                  float(r.amount), now[:10]) for _, r in positions.iterrows()],
            )

    def clients(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql("SELECT * FROM clients ORDER BY name", con)

    def client_positions(self, client_id: str) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql("SELECT * FROM positions WHERE client_id=?", con, params=[client_id])
