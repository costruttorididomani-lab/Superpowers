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


def fetch_yahoo(symbol: str, start: str, retries: int = 3) -> pd.Series:
    """Chiusure di fine mese da Yahoo Finance, calcolate dai dati GIORNALIERI (prezzi rettificati:
    per ETF/fondi includono i dividendi). Le barre mensili di Yahoo hanno buchi sui future."""
    import time

    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, interval="1d", auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                time.sleep(0.5)  # cortesia verso il servizio (evita blocchi per troppe richieste)
                return to_month_end(close)
            last_err = ValueError(f"Nessun dato Yahoo per {symbol}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise last_err


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


def fetch_ecb(key: str, start: str) -> pd.Series:
    """Serie del Data Portal BCE (es. 'IRS/M.IT.L.L40.CI.0000.EUR.N.Z' = rendimento BTP 10 anni,
    media mensile). Nessuna API key necessaria."""
    import requests

    url = f"https://data-api.ecb.europa.eu/service/data/{key}"
    r = requests.get(url, params={"format": "csvdata", "startPeriod": start[:7]}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.PeriodIndex(df["TIME_PERIOD"], freq="M").to_timestamp(how="end").normalize())
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
                  index=pd.to_datetime(df[date_col], dayfirst=not df[date_col].astype(str).str.match(r"^\d{4}-").all()))
    s = to_month_end(s)
    return s[s.index >= pd.Timestamp(start)] if start else s
