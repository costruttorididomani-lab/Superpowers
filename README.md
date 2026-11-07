# WealthLab

Database storico **mensile** (20+ anni) di indici azionari, governativi, materie prime, cambi,
inflazione e fondi selezionati, con analisi di correlazione, **simulazioni Monte Carlo di scenario**
e **analisi della posizione del cliente**. Pensato per la consulenza patrimoniale, in valuta base EUR.

> Le simulazioni sono proiezioni statistiche basate su dati storici e ipotesi: non costituiscono
> garanzia di rendimenti futuri né raccomandazione personalizzata.

## Stato del database incluso

Il repository contiene già il database popolato con **dati reali** (`data/wealthlab.db`) e la sua
esportazione Excel (`data/snapshot/database_mensile.xlsx`, generabile con `python -m wealthlab export`):
30 serie mensili da **gennaio 2005 a ottobre 2026** (governativi area euro e inflazione fino a settembre 2026,
ultimo dato pubblicato). Si conservano solo mesi **chiusi**: il mese in corso entra al primo
aggiornamento successivo alla sua chiusura.

Fonti: Yahoo Finance (indici, future su materie prime, cambi), BCE (rendimenti decennali IT/DE/FR/ES,
Euribor 3M), FRED (Treasury USA, inflazione), più integrazioni storiche documentate in `data/seed/`.
Unico dato assente: inflazione USA di ottobre 2025, mai pubblicata a causa dello shutdown federale.

## Avvio rapido

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m wealthlab update        # scarica ~20 anni di storico (Yahoo Finance + FRED)
streamlit run app/app.py          # apre l'interfaccia nel browser
```

Senza connessione alle fonti si può provare tutto con `python -m wealthlab demo`
(dati **sintetici**, segnalati in giallo nell'interfaccia e sostituiti automaticamente al primo `update`).

## Cosa contiene il database

| Classe | Strumenti (modificabili in `config/universe.yaml`) | Fonte |
|---|---|---|
| Azionario | MSCI World, S&P 500, Nasdaq 100, Euro Stoxx 50, FTSE MIB, DAX, CAC 40, IBEX 35, FTSE 100, Nikkei 225, MSCI EM | Yahoo Finance |
| Governativi | BTP, Bund, OAT, Bonos 10Y; Euribor 3M | BCE |
| | Treasury 10Y e 2Y | FRED |
| Materie prime | Oro, Argento, WTI, Brent, Gas naturale, Rame, S&P GSCI | Yahoo Finance |
| Cambi / macro | EUR/USD, EUR/GBP, EUR/JPY; inflazione Area Euro (HICP) e USA (CPI) | Yahoo / FRED |
| Integrazioni | Euro Stoxx 50 2005-2007, Brent 2005-2007 e mesi mancanti | `data/seed/` (FMP, EIA) |
| Fondi | i fondi 4-5 stelle Morningstar che selezioni | Yahoo (codice `0P...`) o CSV |

**Aggiornabile e scalabile**: `python -m wealthlab update` scarica solo i mesi mancanti (riprende dagli
ultimi 3 mesi per catturare revisioni). Per aggiungere uno strumento basta una riga in `universe.yaml`:
lo schema SQLite (`data/wealthlab.db`) non cambia. Tutte le serie sono a fine mese.

### Governativi e duration
Le fonti pubbliche danno i **rendimenti** dei titoli di Stato, non i prezzi. Ogni governativo è quindi un
*building block a duration costante* il cui rendimento totale mensile è ricostruito come:

    r = y(t-1)/12 − Duration × Δy + ½ × Convessità × Δy²

Questo rende la **duration un parametro**: nel simulatore si può allungarla o accorciarla e l'effetto
viene ricalcolato sull'intero storico dei tassi.

### Fondi Morningstar
I rating Morningstar sono dati proprietari e non scaricabili liberamente: la selezione dei fondi a
4-5 stelle va fatta da Morningstar Direct / piattaforma interna. Due modi per inserirli:
1. **Yahoo**: molti fondi UE hanno ticker uguale al codice Morningstar (es. `0P0000XXXX.F`), `source: yahoo`.
2. **CSV/Excel**: esporta i NAV (colonne `date;value`, vedi `data/import/esempio_nav_fondo.csv`)
   e importali da *Database → Importa NAV* oppure con `source: csv` in `universe.yaml`.

Campi utili: `isin` (abbinamento automatico alle posizioni cliente), `morningstar_stars`,
`fund_type` (EQUITY/BOND/OTHER) e `proxy` (indice di riferimento, es. `MSCI_WORLD`).

## Interfaccia (`streamlit run app/app.py`)

1. **Database** – copertura dati, aggiornamento con un click, import NAV fondi, log aggiornamenti.
2. **Analisi storica** – performance in EUR (con/senza copertura cambio), CAGR, volatilità, Sharpe,
   Sortino, max drawdown, VaR/CVaR; matrice di correlazione completa **e nei mesi di crisi**;
   correlazioni mobili; rendimenti per anno solare; export Excel.
3. **Simulatore Monte Carlo** – portafoglio modello + leve dinamiche:
   - quota azionaria su azioni+obbligazioni;
   - Δ azionario mondo, Δ azionario Italia, Δ oro, tilt su qualunque altro strumento;
   - duration obbligazionaria target;
   - inserimento di fondi, finanziati dalla stessa classe o da un indice specifico;
   - copertura del cambio;
   - orizzonte, numero simulazioni, metodo (bootstrap a blocchi / normale / t di Student),
     versamenti o prelievi mensili (anche indicizzati), costi, ribilanciamento, obiettivo;
   - **aspettative forward-looking**: rendimento atteso e moltiplicatore di volatilità per strumento.
   Output: ventaglio dei percentili (nominale e reale), probabilità di perdita e di raggiungere l'obiettivo,
   VaR/CVaR a 1 anno, distribuzione del max drawdown, **stress test storici** (2008, 2011, 2016, 2018,
   Covid, 2022), contributo al rischio.
4. **Analisi cliente** – carica il file delle posizioni (`data/import/esempio_posizioni_cliente.csv`),
   abbinamento automatico per ISIN/building block, allocazione per classe/area/valuta, duration,
   confronto **attuale vs proposta** con le stesse leve, **dove inserire i fondi** (alpha, beta,
   tracking error, information ratio e impatto sul portafoglio sostituendo il 10% dell'indice proxy),
   salvataggio cliente, report Excel.

## Riga di comando

```bash
python -m wealthlab status                       # copertura dati
python -m wealthlab update --only BTP_10Y,GOLD   # aggiorna alcuni strumenti
python -m wealthlab update --full                # riscarica tutto
python -m wealthlab corr --ids MSCI_WORLD,BTP_10Y,GOLD
python -m wealthlab export                       # Excel completo del database
python -m wealthlab simulate --weights MSCI_WORLD=0.6,BTP_10Y=0.4 --years 15
```

## Note di metodo
- Rendimenti convertiti in EUR con il cambio storico; la copertura valutaria è approssimata con il
  rendimento in valuta locale (non include il differenziale dei tassi).
- I rendimenti dei governativi sono **medie mensili** (convenzione BCE/FRED), non valori di fine mese:
  attenuano leggermente la volatilità mensile dei building block obbligazionari.
- Le materie prime sono future "front month" continui: i cambi di scadenza (soprattutto gas naturale)
  possono generare salti mensili ampi.
- Gli indici azionari Yahoo sono in genere *price index* (DAX è total return): per includere i dividendi
  usare ETF ad accumulazione come proxy (`symbol` dell'ETF in `universe.yaml`).
- Il bootstrap stazionario (blocchi medi di 12 mesi) usa solo i mesi in cui tutti gli strumenti del
  portafoglio hanno dati: strumenti con storia breve accorciano il campione (indicato nei risultati).

## Struttura
```
config/universe.yaml     universo strumenti
wealthlab/db.py          database SQLite
wealthlab/loaders/       connettori Yahoo, FRED, BCE, CSV
wealthlab/export.py      esportazione Excel del database
data/seed/               integrazioni storiche (riempiono solo i mesi mancanti)
wealthlab/updater.py     aggiornamento incrementale
wealthlab/analytics.py   rendimenti, statistiche, correlazioni, stress
wealthlab/portfolio.py   portafogli, leve di scenario, contributo al rischio, analisi fondi
wealthlab/montecarlo.py  motore Monte Carlo
wealthlab/client.py      import posizioni cliente
app/app.py               interfaccia Streamlit
tests/                   test automatici (pytest)
```
