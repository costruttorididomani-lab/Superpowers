"""Esportazione Excel delle analisi (per archivio consulenza / allegato al cliente)."""
from __future__ import annotations

import io

import pandas as pd


def to_excel(sheets: dict[str, pd.DataFrame | pd.Series], path=None) -> bytes:
    buf = io.BytesIO() if path is None else path
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Nota": [
            "Report generato con WealthLab.",
            "Le simulazioni Monte Carlo sono proiezioni statistiche basate su dati storici e ipotesi:",
            "non costituiscono garanzia di rendimenti futuri.",
        ]}).to_excel(xw, sheet_name="Note", index=False)
        for name, df in sheets.items():
            obj = df.to_frame() if isinstance(df, pd.Series) else df
            if isinstance(obj.index, pd.DatetimeIndex):
                obj = obj.copy()
                obj.index = obj.index.strftime("%Y-%m-%d")
            obj.to_excel(xw, sheet_name=name[:31])
            ws = xw.sheets[name[:31]]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(45, max(12, max(len(str(c.value or "")) for c in col) + 2))
    return buf.getvalue() if path is None else b""
