import numpy as np
import pandas as pd
import pytest

from wealthlab.analytics import asset_returns, bond_total_return, correlation, summary_stats
from wealthlab.client import map_positions, positions_to_portfolio, read_positions
from wealthlab.db import Database
from wealthlab.demo import build_demo
from wealthlab.montecarlo import MCConfig, run_simulation
from wealthlab.portfolio import (Portfolio, ScenarioParams, apply_scenario, class_weights, fund_fit,
                                 historical_stress, portfolio_duration, risk_contribution, set_equity_share, tilt)
from wealthlab.analytics import STRESS_EPISODES


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    d = Database(tmp_path_factory.mktemp("db") / "t.db")
    build_demo(d, end="2025-12-31")
    return d


@pytest.fixture(scope="module")
def meta(db):
    return db.instruments()


@pytest.fixture(scope="module")
def rets(db):
    return asset_returns(db)


BASE = Portfolio({"MSCI_WORLD": 0.4, "FTSEMIB": 0.1, "BTP_10Y": 0.3, "BUND_10Y": 0.15, "GOLD": 0.05})


def test_bond_return_formula():
    y = pd.Series([3.0, 4.0], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    r = bond_total_return(y, 7.0).iloc[0]
    assert r == pytest.approx(0.03 / 12 - 7 * 0.01 + 0.5 * 56 * 0.0001)


def test_upsert_is_idempotent(tmp_path):
    d = Database(tmp_path / "x.db")
    d.upsert_instruments([{"id": "A", "name": "A", "asset_class": "EQUITY", "source": "csv", "symbol": "a"}])
    s = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    d.upsert_series("A", s, "csv")
    d.upsert_series("A", s * 2, "csv")
    lv = d.levels(["A"])
    assert len(lv) == 2 and lv["A"].iloc[-1] == 4.0


def test_returns_and_stats(rets):
    assert {"MSCI_WORLD", "BTP_10Y", "GOLD"} <= set(rets.columns)
    st = summary_stats(rets[["MSCI_WORLD", "BTP_10Y"]])
    assert st.loc["MSCI_WORLD", "Volatilita"] > st.loc["BTP_10Y", "Volatilita"]
    c = correlation(rets)
    assert np.allclose(np.diag(c.values), 1)


def test_equity_share_and_tilt(meta):
    p = set_equity_share(BASE, 0.7, meta)
    cw = class_weights(p, meta)
    assert cw["EQUITY"] / (cw["EQUITY"] + cw["BOND"]) == pytest.approx(0.7)
    assert p.weights["GOLD"] == pytest.approx(0.05)
    q = tilt(BASE, "FTSEMIB", 0.05, meta)
    assert q.weights["FTSEMIB"] == pytest.approx(0.15)
    assert class_weights(q, meta)["EQUITY"] == pytest.approx(0.5)  # finanziato dall'azionario
    assert sum(q.weights.values()) == pytest.approx(1)


def test_scenario_duration_and_funds(meta):
    sp = ScenarioParams(duration=3.0, funds={"DEMO_FONDO_AZ_GLOBALE": 0.1},
                        fund_replace={"DEMO_FONDO_AZ_GLOBALE": "MSCI_WORLD"})
    s = apply_scenario(BASE, sp, meta)
    assert portfolio_duration(s, meta) == pytest.approx(3.0)
    assert s.weights["MSCI_WORLD"] == pytest.approx(0.3)


def test_shorter_duration_lowers_rate_risk(db):
    r_long = asset_returns(db, ["BUND_10Y"], durations={"BUND_10Y": 15})
    r_short = asset_returns(db, ["BUND_10Y"], durations={"BUND_10Y": 2})
    assert r_long["BUND_10Y"].std() > r_short["BUND_10Y"].std()


@pytest.mark.parametrize("method", ["bootstrap", "normal", "student_t"])
@pytest.mark.parametrize("rebalance", ["monthly", "annual"])
def test_monte_carlo(rets, method, rebalance):
    cfg = MCConfig(horizon_years=5, n_sims=500, method=method, rebalance=rebalance, target=120_000)
    res = run_simulation(rets, BASE, cfg)
    assert res.paths_pct.shape == (61, 7)
    assert (res.paths_pct["P5"] <= res.paths_pct["P95"]).all()
    assert 0 <= res.stats["Prob. raggiungere obiettivo"] <= 1
    assert res.real_paths_pct is not None


def test_mc_withdrawals_and_overrides(rets):
    cfg = MCConfig(horizon_years=10, n_sims=500, monthly_flow=-2000, mean_overrides={"MSCI_WORLD": 0.0})
    res = run_simulation(rets, BASE, cfg)
    assert res.stats["Totale versato/(prelevato)"] == 100_000 - 240_000
    assert res.final.min() >= 0
    hi = run_simulation(rets, BASE, MCConfig(horizon_years=10, n_sims=500, mean_overrides={"MSCI_WORLD": 0.15}))
    lo = run_simulation(rets, BASE, MCConfig(horizon_years=10, n_sims=500, mean_overrides={"MSCI_WORLD": -0.05}))
    assert hi.stats["Patrimonio finale mediano"] > lo.stats["Patrimonio finale mediano"]


def test_client_flow(tmp_path, db, meta, rets):
    f = tmp_path / "pos.csv"
    f.write_text("descrizione;isin;controvalore;building_block\nETF World;;150.000;MSCI_WORLD\n"
                 "Fondo;XX0000000001;50000;\nBTP;;100000;BTP_10Y\nIgnoto;ZZ;10000;\n")
    pos = read_positions(f)
    assert pos["amount"].sum() == 310_000
    m, u = map_positions(pos, meta)
    assert len(u) == 1 and "DEMO_FONDO_AZ_GLOBALE" in set(m["instrument_id"])
    p = positions_to_portfolio(m)
    r = asset_returns(db, list(p.weights))
    rc = risk_contribution(r, p)
    assert rc["% del rischio"].sum() == pytest.approx(1)
    st = historical_stress(r, [p], STRESS_EPISODES)
    assert st.notna().all().all()
    ff = fund_fit(asset_returns(db), BASE, ["DEMO_FONDO_AZ_GLOBALE"], {"DEMO_FONDO_AZ_GLOBALE": "MSCI_WORLD"}, meta)
    assert "Alpha annuo" in ff.columns


def test_completed_months_excludes_current():
    from wealthlab.updater import completed_months
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-09-30", "2026-10-31", "2026-11-30"]))
    out = completed_months(s, today=pd.Timestamp("2026-11-07"))
    assert list(out.index.strftime("%Y-%m")) == ["2026-09", "2026-10"]


def test_fill_missing_never_overwrites(tmp_path):
    d = Database(tmp_path / "f.db")
    d.upsert_instruments([{"id": "A", "name": "A", "asset_class": "EQUITY", "source": "csv", "symbol": "a"}])
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    d.upsert_series("A", pd.Series([1.0, 2.0], index=idx[1:]), "yahoo")
    n = d.fill_missing("A", pd.Series([9.0, 9.0, 9.0], index=idx), "seed")
    lv = d.levels(["A"])["A"]
    assert n == 1 and lv.iloc[0] == 9.0 and lv.iloc[1] == 1.0


def test_seed_files_are_valid():
    from wealthlab.loaders import fetch_csv
    from wealthlab.updater import SEED_DIR
    for f in SEED_DIR.glob("*.csv"):
        s = fetch_csv(str(f))
        assert len(s) > 200 and s.index.is_monotonic_increasing and (s > 0).all()
