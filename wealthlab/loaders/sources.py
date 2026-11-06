from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IMPORT_DIR = ROOT / "data" / "import"


def to_month_end(s: pd.Series, how: str = "last") -> pd.Series:
    """Ricampiona a fine mese (ultimo valore del mese)."""
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s.resample("ME").last() if how == "last" else s.resample("ME").mean()
    return s.dropna()


def fetch_yahoo(symbol: str, start: str) -> pd.Series:
    """Prezzi (Adj Close se disponibile, include dividendi per ETF/fondi) da Yahoo Finance."""
    import yfinance as yf

    df = yf.download(symbol, start=start, interval="1mo", auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        # alcuni simboli rispondono solo in daily
        df = yf.download(symbol, start=start, interval="1d", auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"Nessun dato Yahoo per {symbol}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = to_month_end(close)
    # con interval=1mo Yahoo etichetta il mese con il primo giorno: dopo il resample e' gia' fine mese
    return s[s.index >= pd.Timestamp(start)]


def fetch_fred(code: str, start: str) -> pd.Series:
    """Serie FRED (St. Louis Fed) senza API key, via endpoint CSV pubblico."""
    import requests

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index(pd.to_datetime(df["date"]))["value"]
    return to_month_end(s)


def fetch_csv(name: str, start: str | None = None) -> pd.Series:
    """CSV locale in data/import/<name>.csv con colonne date,value (separatore , o ;, decimali . o ,)."""
    path = Path(name)
    if not path.is_absolute():
        path = IMPORT_DIR / (name if name.endswith((".csv", ".xlsx")) else f"{name}.csv")
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        raw = path.read_text(encoding="utf-8-sig")
        sep = ";" if raw.count(";") > raw.count(",") else ","
        df = pd.read_csv(io.StringIO(raw), sep=sep, decimal="," if sep == ";" else ".")
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next(c for c in df.columns if c in ("date", "data"))
    val_col = next(c for c in df.columns if c in ("value", "valore", "nav", "close", "prezzo"))
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col], dayfirst=True))
    s = to_month_end(s)
    return s[s.index >= pd.Timestamp(start)] if start else s
