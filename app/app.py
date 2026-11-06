"""WealthLab - interfaccia Streamlit.   Avvio:  streamlit run app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wealthlab.analytics import (STRESS_EPISODES, asset_returns, calendar_returns, correlation,  # noqa: E402
                                 risk_free, rolling_correlation, stress_correlation, summary_stats)
from wealthlab.client import map_positions, positions_to_portfolio, read_positions  # noqa: E402
from wealthlab.db import Database  # noqa: E402
from wealthlab.demo import build_demo  # noqa: E402
from wealthlab.montecarlo import MCConfig, compare, run_simulation  # noqa: E402
from wealthlab.portfolio import (Portfolio, ScenarioParams, apply_scenario, class_weights, fund_fit,  # noqa: E402
                                 historical_stress, macro_class, portfolio_duration, portfolio_returns,
                                 risk_contribution)
from wealthlab.report import to_excel  # noqa: E402
from wealthlab.updater import update_database  # noqa: E402

st.set_page_config(page_title="WealthLab", page_icon="📊", layout="wide")
PCT = "{:.2%}".format
def EUR(v):
    return f"€ {v:,.0f}".replace(",", ".")
NAVY, GOLD, GREY = "#14365d", "#c9a227", "#8a94a6"


@st.cache_resource
def get_db() -> Database:
    return Database()


db = get_db()


@st.cache_data(ttl=600, show_spinner=False)
def load_returns(_v: int, hedged: bool, durations: tuple) -> pd.DataFrame:
    return asset_returns(db, hedged=hedged, durations=dict(durations))


def data_version() -> int:
    return st.session_state.get("data_version", 0)


def bump():
    st.session_state["data_version"] = data_version() + 1
    st.cache_data.clear()


meta = db.instruments()
label = {i: f"{r['name']} [{i}]" for i, r in meta.iterrows()}
investable = [i for i in meta.index if meta.loc[i, "asset_class"] not in ("FX", "MACRO")]
cov = db.coverage()
is_demo = cov["fonte_dati"].fillna("").str.contains("DEMO").any()

st.sidebar.title("📊 WealthLab")
page = st.sidebar.radio("Sezione", ["Database", "Analisi storica", "Simulatore Monte Carlo", "Analisi cliente"])
hedged_global = st.sidebar.checkbox("Strumenti in valuta estera coperti dal cambio", value=False,
                                    help="Se disattivo, i rendimenti sono convertiti in EUR (rischio cambio incluso).")
if is_demo:
    st.sidebar.warning("⚠️ Nel database ci sono dati **DIMOSTRATIVI** sintetici. Esegui l'aggiornamento per i dati reali.")
st.sidebar.caption("Le simulazioni sono proiezioni statistiche e non garantiscono rendimenti futuri.")


# =========================================================== componenti comuni
def portfolio_editor(key: str, default: dict[str, float]) -> Portfolio:
    df = pd.DataFrame({"Strumento": list(default), "Peso %": [v * 100 for v in default.values()]})
    ed = st.data_editor(df, key=key, num_rows="dynamic", width="stretch",
                        column_config={"Strumento": st.column_config.SelectboxColumn(options=investable, required=True),
                                       "Peso %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5)})
    ed = ed.dropna()
    ed = ed[ed["Peso %"] > 0]
    tot = ed["Peso %"].sum()
    if abs(tot - 100) > 0.01:
        st.caption(f"Totale pesi {tot:.1f}% → verranno normalizzati a 100%.")
    return Portfolio(weights=ed.groupby("Strumento")["Peso %"].sum().to_dict(), name="Attuale").normalized()


def scenario_levers(base: Portfolio, key: str) -> ScenarioParams:
    cw = class_weights(base, meta)
    eq_now = cw.get("EQUITY", 0) / max(cw.get("EQUITY", 0) + cw.get("BOND", 0) + cw.get("CASH", 0), 1e-9)
    st.markdown("**Leve di scenario**")
    c1, c2, c3 = st.columns(3)
    eq = c1.slider("Quota azionaria (su azioni+obbligazioni) %", 0, 100, int(round(eq_now * 100)), key=f"{key}_eq")
    d_now = portfolio_duration(base, meta)
    dur = c2.slider("Duration obbligazionaria (anni)", 0.5, 20.0, float(round(d_now, 1)) if d_now else 7.0, 0.5,
                    key=f"{key}_dur")
    hedge = c3.checkbox("Copri il rischio cambio", value=hedged_global, key=f"{key}_hedge")
    c4, c5, c6 = st.columns(3)
    t_world = c4.slider("Δ Azionario Mondo (MSCI World) punti %", -30, 30, 0, key=f"{key}_w")
    t_ita = c5.slider("Δ Azionario Italia (FTSE MIB) punti %", -20, 20, 0, key=f"{key}_it")
    t_gold = c6.slider("Δ Oro punti %", -10, 15, 0, key=f"{key}_gold")
    with st.expander("Tilt su altri strumenti e inserimento fondi"):
        extra = st.multiselect("Altri strumenti da sovra/sotto-pesare", [i for i in investable if i not in
                               ("MSCI_WORLD", "FTSEMIB", "GOLD")], key=f"{key}_extra", format_func=lambda i: label[i])
        tilts = {"MSCI_WORLD": t_world / 100, "FTSEMIB": t_ita / 100, "GOLD": t_gold / 100}
        for i in extra:
            tilts[i] = st.slider(f"Δ {label[i]} punti %", -30, 30, 0, key=f"{key}_x_{i}") / 100
        funds = [i for i in meta.index if meta.loc[i, "asset_class"] == "FUND"]
        fsel = st.multiselect("Fondi da inserire", funds, key=f"{key}_funds", format_func=lambda i: label[i])
        fw, frep = {}, {}
        for f in fsel:
            a, b = st.columns(2)
            fw[f] = a.slider(f"Peso {label[f]} %", 0, 50, 10, key=f"{key}_fw_{f}") / 100
            proxy = (meta.loc[f, "meta"] or {}).get("proxy")
            opts = ["(stessa macro-classe pro-quota)"] + investable
            choice = b.selectbox("Finanziato da", opts, index=opts.index(proxy) if proxy in opts else 0,
                                 key=f"{key}_fr_{f}")
            if choice != opts[0]:
                frep[f] = choice
    return ScenarioParams(equity_share=eq / 100 if abs(eq / 100 - eq_now) > 0.004 else None,
                          tilts={k: v for k, v in tilts.items() if v}, duration=dur if abs(dur - d_now) > 0.04 else None,
                          funds=fw, fund_replace=frep, hedge_usd=hedge)


def mc_settings(key: str) -> MCConfig:
    with st.expander("Parametri della simulazione", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        cfg = MCConfig(
            horizon_years=c1.slider("Orizzonte (anni)", 1, 40, 10, key=f"{key}_h"),
            n_sims=c2.select_slider("Numero simulazioni", [1000, 2000, 5000, 10000], 5000, key=f"{key}_n"),
            method=c3.selectbox("Metodo", ["bootstrap", "student_t", "normal"], key=f"{key}_m",
                                help="bootstrap = ricampionamento a blocchi dei mesi storici (consigliato)"),
            block_size=c4.slider("Blocco bootstrap (mesi)", 1, 36, 12, key=f"{key}_b"),
        )
        c1, c2, c3, c4 = st.columns(4)
        cfg.initial = c1.number_input("Capitale iniziale €", 0.0, 1e9, 100_000.0, 10_000.0, key=f"{key}_i")
        cfg.monthly_flow = c2.number_input("Versamento(+)/prelievo(-) mensile €", -1e6, 1e6, 0.0, 100.0, key=f"{key}_f")
        cfg.annual_cost = c3.number_input("Costi annui totali %", 0.0, 5.0, 0.0, 0.1, key=f"{key}_c") / 100
        cfg.rebalance = c4.selectbox("Ribilanciamento", ["monthly", "annual", "none"], key=f"{key}_r",
                                     format_func={"monthly": "Mensile", "annual": "Annuale", "none": "Nessuno"}.get)
        c1, c2 = st.columns(2)
        tgt = c1.number_input("Obiettivo patrimoniale € (0 = nessuno)", 0.0, 1e10, 0.0, 10_000.0, key=f"{key}_t")
        cfg.target = tgt or None
        cfg.flow_indexed = c2.checkbox("Flussi indicizzati all'inflazione", key=f"{key}_fi")
        st.markdown("*Aspettative forward-looking (facoltative): sostituiscono la media storica*")
        ov = st.data_editor(pd.DataFrame({"Strumento": pd.Series(dtype=str), "Rend. annuo atteso %": pd.Series(dtype=float),
                                          "Moltiplicatore vol.": pd.Series(dtype=float)}),
                            num_rows="dynamic", key=f"{key}_ov", width="stretch",
                            column_config={"Strumento": st.column_config.SelectboxColumn(options=investable)})
        for _, r in ov.dropna(subset=["Strumento"]).iterrows():
            if pd.notna(r["Rend. annuo atteso %"]):
                cfg.mean_overrides[r["Strumento"]] = r["Rend. annuo atteso %"] / 100
            if pd.notna(r["Moltiplicatore vol."]):
                cfg.vol_scale[r["Strumento"]] = r["Moltiplicatore vol."]
    return cfg


def returns_for(portfolios: list[Portfolio]) -> pd.DataFrame:
    """Rendimenti con le duration/coperture specifiche di ciascun portafoglio (colonne rinominate se serve)."""
    frames = []
    for p in portfolios:
        r = asset_returns(db, ids=list(p.weights) + ["CPI_EZ", "EUR_3M"], hedged=p.hedged or hedged_global,
                          durations=p.durations)
        frames.append(r)
    return frames


def fan_chart(results: dict, real: bool = False) -> go.Figure:
    fig = go.Figure()
    colors = [NAVY, GOLD, "#2e8b57", "#b03a2e"]
    rgba = lambda h, a: f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"  # noqa: E731
    for (name, res), col in zip(results.items(), colors):
        d = res.real_paths_pct if real and res.real_paths_pct is not None else res.paths_pct
        fig.add_trace(go.Scatter(x=d.index, y=d["P95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.index, y=d["P5"], fill="tonexty", line=dict(width=0),
                                 fillcolor=rgba(col, 0.13), name=f"{name} P5–P95"))
        fig.add_trace(go.Scatter(x=d.index, y=d["P25"], line=dict(color=col, dash="dot", width=1), name=f"{name} P25"))
        fig.add_trace(go.Scatter(x=d.index, y=d["P50"], line=dict(color=col, width=3), name=f"{name} mediana"))
    fig.update_layout(height=460, yaxis_title="Patrimonio €", hovermode="x unified", legend=dict(orientation="h"))
    return fig


def fmt_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for idx in out.index:
        for c in out.columns:
            v = df.loc[idx, c]
            if pd.isna(v):
                out.loc[idx, c] = "–"
            elif any(k in idx for k in ("Prob.", "Rend.", "VaR", "drawdown", "Delta")):
                out.loc[idx, c] = PCT(v)
            elif "Mesi" in idx:
                out.loc[idx, c] = f"{v:.0f}"
            else:
                out.loc[idx, c] = EUR(v)
    return out


def run_compare(base: Portfolio, scen: Portfolio, cfg: MCConfig):
    rb, rs = returns_for([base, scen])
    res = {"Attuale": run_simulation(rb, base, cfg), "Scenario": run_simulation(rs, scen, cfg)}
    return res, rb, rs


def show_mc(res: dict, rb: pd.DataFrame, rs: pd.DataFrame, base: Portfolio, scen: Portfolio, key: str):
    tab1, tab2, tab3, tab4 = st.tabs(["Proiezione", "Distribuzioni", "Stress test storici", "Composizione e rischio"])
    with tab1:
        real = st.toggle("Valori reali (al netto dell'inflazione)", key=f"real_{key}")
        st.plotly_chart(fan_chart(res, real), width="stretch")
        st.dataframe(fmt_stats(compare(res)), width="stretch")
    with tab2:
        c1, c2 = st.columns(2)
        h = pd.concat([pd.DataFrame({"Patrimonio finale": r.final, "Portafoglio": k}) for k, r in res.items()])
        c1.plotly_chart(px.histogram(h, x="Patrimonio finale", color="Portafoglio", barmode="overlay", nbins=80,
                                     color_discrete_sequence=[NAVY, GOLD], title="Distribuzione patrimonio finale"),
                        width="stretch")
        d = pd.concat([pd.DataFrame({"Max drawdown": r.max_dd, "Portafoglio": k}) for k, r in res.items()])
        c2.plotly_chart(px.histogram(d, x="Max drawdown", color="Portafoglio", barmode="overlay", nbins=60,
                                     color_discrete_sequence=[NAVY, GOLD], title="Distribuzione max drawdown"),
                        width="stretch")
    with tab3:
        sb = historical_stress(rb, [base], STRESS_EPISODES)
        ss = historical_stress(rs, [scen], STRESS_EPISODES)
        stress = pd.concat([sb, ss], axis=1)
        stress.columns = ["Attuale", "Scenario"]
        st.dataframe(stress.style.format(PCT, na_rep="dati insufficienti"), width="stretch")
        st.caption("Performance che il portafoglio avrebbe avuto negli episodi storici (ribilanciamento mensile).")
    with tab4:
        c1, c2 = st.columns(2)
        comp = pd.DataFrame({"Attuale": base.series(), "Scenario": scen.series()}).fillna(0)
        comp.index = [label.get(i, i) for i in comp.index]
        c1.dataframe(comp.style.format(PCT), width="stretch")
        cw = pd.DataFrame({"Attuale": class_weights(base, meta), "Scenario": class_weights(scen, meta)}).fillna(0)
        c1.dataframe(cw.style.format(PCT), width="stretch")
        c1.metric("Duration obbligazionaria", f"{portfolio_duration(base, meta):.1f} → {portfolio_duration(scen, meta):.1f} anni")
        rc = risk_contribution(rs, scen)
        c2.plotly_chart(px.bar(rc.reset_index(), x="index", y="% del rischio", title="Contributo al rischio (scenario)",
                               color_discrete_sequence=[NAVY]), width="stretch")
        hb, hs = portfolio_returns(rb, base), portfolio_returns(rs, scen)
        stt = summary_stats(pd.concat([hb.rename("Attuale"), hs.rename("Scenario")], axis=1), risk_free(db))
        c2.dataframe(stt[["Dal", "Rend. annuo (CAGR)", "Volatilita", "Sharpe", "Max Drawdown"]], width="stretch")


# =================================================================== PAGINE
if page == "Database":
    st.title("Database storico mensile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Strumenti", len(meta))
    c2.metric("Osservazioni", int(cov["mesi"].sum()))
    c3.metric("Ultimo dato", str(cov["al"].max()))
    a, b, c = st.columns(3)
    if a.button("🔄 Aggiorna dati (incrementale)", type="primary"):
        with st.spinner("Download da Yahoo Finance / FRED..."):
            rep = update_database(db, verbose=False)
        st.dataframe(rep, width="stretch")
        bump()
    if b.button("⬇️ Riscarica tutto lo storico"):
        with st.spinner("Download completo..."):
            rep = update_database(db, full=True, verbose=False)
        st.dataframe(rep, width="stretch")
        bump()
    if c.button("🧪 Carica dati dimostrativi"):
        build_demo(db)
        bump()
        st.rerun()
    st.dataframe(cov, width="stretch", hide_index=True)
    with st.expander("Importa NAV di un fondo da CSV/Excel"):
        st.markdown("File con colonne **date** e **value** (o data/nav). Per registrarlo in modo permanente "
                    "aggiungilo anche a `config/universe.yaml` con `source: csv`.")
        up = st.file_uploader("File NAV", type=["csv", "xlsx"])
        c1, c2, c3 = st.columns(3)
        fid = c1.text_input("ID fondo (es. FONDO_XYZ)")
        fname = c2.text_input("Nome fondo")
        ftype = c3.selectbox("Natura", ["EQUITY", "BOND", "OTHER"])
        c1, c2, c3 = st.columns(3)
        isin = c1.text_input("ISIN")
        stars = c2.selectbox("Rating Morningstar", [5, 4, 3, 2, 1, None], index=1)
        proxy = c3.selectbox("Indice di riferimento (proxy)", [None] + investable)
        if up and fid and st.button("Importa fondo"):
            from wealthlab.loaders.sources import IMPORT_DIR, fetch_csv
            dest = IMPORT_DIR / f"{fid.lower()}{Path(up.name).suffix}"
            dest.write_bytes(up.getvalue())
            s = fetch_csv(str(dest))
            db.upsert_instruments([{"id": fid.upper(), "name": fname or fid, "asset_class": "FUND", "region": "WORLD",
                                    "currency": "EUR", "source": "csv", "symbol": dest.stem, "isin": isin,
                                    "morningstar_stars": stars, "fund_type": ftype, "proxy": proxy}])
            n = db.upsert_series(fid.upper(), s, "csv")
            st.success(f"Importati {n} mesi per {fid.upper()}")
            bump()
    with st.expander("Log aggiornamenti"):
        st.dataframe(db.update_history(), width="stretch")

elif page == "Analisi storica":
    st.title("Analisi storica, rischio e correlazioni")
    rets = load_returns(data_version(), hedged_global, ())
    defaults = [i for i in ["MSCI_WORLD", "SP500", "EUROSTOXX50", "FTSEMIB", "BTP_10Y", "BUND_10Y", "UST_10Y", "GOLD", "BRENT"]
                if i in rets]
    sel = st.multiselect("Strumenti", [c for c in rets.columns], defaults, format_func=lambda i: label.get(i, i))
    if not sel:
        st.stop()
    y0, y1 = rets.index.min().year, rets.index.max().year
    yr = st.slider("Periodo", y0, y1, (max(y0, y1 - 20), y1))
    r = rets.loc[str(yr[0]):str(yr[1]), sel]
    rf = risk_free(db)
    t1, t2, t3, t4 = st.tabs(["Performance", "Correlazioni", "Correlazioni nel tempo", "Rendimenti annuali"])
    with t1:
        cum = (1 + r.fillna(0)).cumprod() * 100
        st.plotly_chart(px.line(cum, labels={"value": "Base 100", "date": ""}, title="Performance cumulata in EUR"),
                        width="stretch")
        stt = summary_stats(r, rf)
        pc = [c for c in stt.columns if c not in ("Dal", "Mesi", "Sharpe", "Sortino", "Asimmetria", "Curtosi")]
        st.dataframe(stt.style.format({c: PCT for c in pc} | {"Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                                                               "Asimmetria": "{:.2f}", "Curtosi": "{:.2f}"}),
                     width="stretch")
    with t2:
        c1, c2 = st.columns(2)
        corr = correlation(r)
        c1.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                  title="Correlazione (intero periodo)"), width="stretch")
        drv = c2.selectbox("Driver di stress", sel, index=0)
        sc = stress_correlation(r, drv, 0.2)
        c2.plotly_chart(px.imshow(sc, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                  title=f"Correlazione nel 20% dei mesi peggiori di {drv}"), width="stretch")
        st.caption("Confrontare le due matrici mostra quali asset diversificano davvero nelle fasi di crisi.")
    with t3:
        c1, c2, c3 = st.columns(3)
        a = c1.selectbox("Asset A", sel, 0)
        b = c2.selectbox("Asset B", sel, min(1, len(sel) - 1))
        w = c3.slider("Finestra mobile (mesi)", 12, 60, 36)
        st.plotly_chart(px.line(rolling_correlation(r, a, b, w), title=f"Correlazione mobile {w}m {a} / {b}")
                        .add_hline(y=0, line_dash="dot"), width="stretch")
    with t4:
        st.dataframe(calendar_returns(r).style.format(PCT).background_gradient(cmap="RdYlGn", axis=None),
                     width="stretch")
    st.download_button("📥 Esporta Excel", to_excel({"Statistiche": summary_stats(r, rf), "Correlazioni": correlation(r),
                                                     "Rendimenti mensili": r, "Rendimenti annuali": calendar_returns(r)}),
                       "analisi_storica.xlsx")

elif page == "Simulatore Monte Carlo":
    st.title("Simulatore Monte Carlo di scenario")
    st.markdown("Definisci un **portafoglio modello**, muovi le **leve** e confronta l'impatto su rendimento, rischio e probabilità.")
    default = {"MSCI_WORLD": 0.35, "EUROSTOXX50": 0.10, "FTSEMIB": 0.05, "BTP_10Y": 0.20, "BUND_10Y": 0.15,
               "UST_10Y": 0.05, "GOLD": 0.05, "EUR_3M": 0.05}
    default = {k: v for k, v in default.items() if k in meta.index}
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Portafoglio di partenza")
        base = portfolio_editor("mc_base", default)
    with c2:
        sp = scenario_levers(base, "mc")
        cfg = mc_settings("mc")
    scen = apply_scenario(base, sp, meta)
    if st.button("▶️ Esegui simulazione", type="primary"):
        with st.spinner("Simulazione in corso..."):
            try:
                st.session_state["mc_out"] = (*run_compare(base, scen, cfg), base, scen)
            except ValueError as e:
                st.error(str(e))
    if "mc_out" in st.session_state:
        res, rb, rs, b0, s0 = st.session_state["mc_out"]
        show_mc(res, rb, rs, b0, s0, "mc")
        st.download_button("📥 Esporta Excel", to_excel({"Confronto": compare(res), "Percentili attuale": res["Attuale"].paths_pct,
                                                         "Percentili scenario": res["Scenario"].paths_pct,
                                                         "Pesi": pd.DataFrame({"Attuale": b0.series(), "Scenario": s0.series()})}),
                           "simulazione.xlsx")

else:
    st.title("Analisi posizione cliente")
    clients = db.clients()
    c1, c2 = st.columns([2, 1])
    src = c1.radio("Origine posizione", ["Carica file", "Cliente salvato"], horizontal=True)
    pos = None
    if src == "Carica file":
        up = c1.file_uploader("File posizioni (CSV/XLSX: descrizione, isin, controvalore, building_block)", ["csv", "xlsx"])
        tpl = "descrizione;isin;controvalore;building_block\nETF MSCI World;IE00B4L5Y983;150000;MSCI_WORLD\n" \
              "BTP 2033;IT0005542359;80000;BTP_10Y\nBund 2034;DE000BU2Z023;40000;BUND_10Y\n" \
              "Azioni Italia;;30000;FTSEMIB\nOro fisico;;20000;GOLD\nLiquidità;;30000;EUR_3M\n"
        c2.download_button("Scarica modello file", tpl, "modello_posizioni.csv")
        if up:
            pos = read_positions(up)
    elif not clients.empty:
        cid = c1.selectbox("Cliente", clients["id"], format_func=lambda i: clients.set_index("id").loc[i, "name"])
        pos = db.client_positions(cid)
    if pos is None:
        st.info("Carica il file delle posizioni del cliente. Ogni riga va associata a un building block del database "
                "(colonna building_block) oppure a un fondo registrato tramite ISIN.")
        st.dataframe(meta[["name", "asset_class", "region"]], width="stretch")
        st.stop()
    mapped, unmapped = map_positions(pos, meta)
    if not unmapped.empty:
        st.warning(f"{len(unmapped)} posizioni non associate (€ {unmapped['amount'].sum():,.0f}): assegna un building block.")
        fix = st.data_editor(unmapped[["description", "isin", "amount", "instrument_id"]], key="fix",
                             column_config={"instrument_id": st.column_config.SelectboxColumn(options=investable)})
        mapped = pd.concat([mapped, fix.dropna(subset=["instrument_id"])])
    total = mapped["amount"].sum()
    base = positions_to_portfolio(mapped, "Attuale")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patrimonio analizzato", EUR(total))
    cw = class_weights(base, meta)
    k2.metric("Azionario", PCT(cw.get("EQUITY", 0)))
    k3.metric("Obbligazionario + liquidità", PCT(cw.get("BOND", 0) + cw.get("CASH", 0)))
    k4.metric("Duration obbligazionaria", f"{portfolio_duration(base, meta):.1f} anni")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.pie(names=cw.index, values=cw.values, title="Per classe", hole=.45), width="stretch")
    rg = class_weights(base, meta, by="region")
    c2.plotly_chart(px.pie(names=rg.index, values=rg.values, title="Per area geografica", hole=.45), width="stretch")
    cc = class_weights(base, meta, by="currency")
    c3.plotly_chart(px.pie(names=cc.index, values=cc.values, title="Per valuta", hole=.45), width="stretch")

    with st.expander("Salva cliente nel database"):
        a, b, c = st.columns(3)
        cid = a.text_input("Codice cliente")
        cname = b.text_input("Nome")
        prof = c.selectbox("Profilo", ["Prudente", "Moderato", "Bilanciato", "Dinamico", "Aggressivo"])
        if st.button("Salva") and cid:
            db.save_client(cid, cname or cid, mapped, prof)
            st.success("Cliente salvato")

    st.subheader("Proposta: leve di scenario")
    sp = scenario_levers(base, "cl")
    cfg = mc_settings("cl")
    cfg.initial = float(total) if cfg.initial == 100_000.0 else cfg.initial
    scen = apply_scenario(base, sp, meta)
    if st.button("▶️ Analizza attuale vs proposta", type="primary"):
        with st.spinner("Calcolo in corso..."):
            try:
                st.session_state["cl_out"] = (*run_compare(base, scen, cfg), base, scen)
            except ValueError as e:
                st.error(str(e))
    if "cl_out" in st.session_state:
        res, rb, rs, b0, s0 = st.session_state["cl_out"]
        show_mc(res, rb, rs, b0, s0, "cl")
        funds = [i for i in meta.index if meta.loc[i, "asset_class"] == "FUND"]
        if funds:
            st.subheader("Dove inserire i fondi")
            proxies = {f: (meta.loc[f, "meta"] or {}).get("proxy") for f in funds}
            ids = set(funds) | set(b0.weights) | {p for p in proxies.values() if p in meta.index}
            rr = asset_returns(db, ids=list(ids), hedged=hedged_global)
            ff = fund_fit(rr, b0, funds, proxies, meta, weight=0.10)
            if not ff.empty:
                ff.insert(0, "Fondo", [label[i] for i in ff.index])
                ff.insert(1, "Stelle MS", [(meta.loc[i, "meta"] or {}).get("morningstar_stars") for i in ff.index])
                pc = ["Corr. con portafoglio", "Beta vs proxy", "Information ratio"]
                st.dataframe(ff.style.format({c: PCT for c in ff.columns if c.startswith(("Alpha", "Tracking", "Delta"))}
                                             | {c: "{:.2f}" for c in pc if c in ff}, na_rep="–"), width="stretch")
                st.caption("Impatto storico sostituendo il 10% del portafoglio (dall'indice proxy del fondo) con il fondo.")
        st.download_button("📥 Report cliente (Excel)", to_excel({
            "Posizioni": mapped, "Confronto MC": compare(res), "Pesi": pd.DataFrame({"Attuale": b0.series(), "Proposta": s0.series()}),
            "Stress test": pd.concat([historical_stress(rb, [b0], STRESS_EPISODES), historical_stress(rs, [s0], STRESS_EPISODES)], axis=1),
            "Contributo rischio": risk_contribution(rb, b0)}), "report_cliente.xlsx")
