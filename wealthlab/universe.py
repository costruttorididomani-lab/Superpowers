"""Lettura della configurazione dell'universo strumenti."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSE = ROOT / "config" / "universe.yaml"


def load_universe(path: str | Path = DEFAULT_UNIVERSE) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ids = [i["id"] for i in cfg["instruments"]]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise ValueError(f"ID duplicati in universe.yaml: {dup}")
    for ins in cfg["instruments"]:
        missing = {"id", "name", "asset_class", "source", "symbol"} - ins.keys()
        if missing:
            raise ValueError(f"Strumento {ins.get('id')} privo dei campi {missing}")
    return cfg
