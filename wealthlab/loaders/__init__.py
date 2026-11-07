"""Connettori alle fonti dati. Ogni loader restituisce una pd.Series mensile (fine mese)."""
from .sources import fetch_csv, fetch_ecb, fetch_fred, fetch_yahoo, to_month_end

__all__ = ["fetch_yahoo", "fetch_fred", "fetch_ecb", "fetch_csv", "to_month_end"]
