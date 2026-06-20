# Bullish Scanners (NSE / BSE)

Two scanners in one Streamlit app:

1. **Reversal patterns** — 16 bullish reversal candle/volume patterns detected at
   *lows* (after a downtrend), tabs ordered by historical reversal frequency.
2. **VCP breakout** — high-grade Minervini-style volatility-contraction bases near
   the *highs*: tight, contracting, quiet bases that are **Coiling** (ready under the
   pivot) or **Breaking out** (clearing the pivot today on a volume surge).

Pick the scanner with the **Mode** toggle at the top of the sidebar. Both share the
Exchange, Timeframe (Daily/Weekly), sector filter, and scan-all controls.

## Repo layout (put all of these in the GitHub repo root)

```
streamlit_app.py              # the Streamlit UI (both scanners, Dhan/Yahoo)
screener_core.py              # reversal-pattern engine
vcp_core.py                   # VCP engine
dhan_data.py                  # DhanHQ v2 data layer (scrip-master + historical fetch)
bullish_reversal_screener.py  # optional command-line reversal runner
requirements.txt
EQUITY_L_2.csv                # NSE list  (cols: companyId, Name, Sector, Industry)
bse_stocks.csv                # BSE list  (cols: Scrip Code, Scrip ID, Scrip Name, Status, ...)
```

> The two CSV filenames are referenced at the top of `streamlit_app.py`
> (`NSE_CSV`, `BSE_CSV`). If you rename the files, update those two lines.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push the repo (all files above) to GitHub.
2. Go to https://share.streamlit.io → **New app** → pick your repo/branch.
3. Set **Main file path** to `streamlit_app.py`. Deploy.

That's it — `requirements.txt` is picked up automatically.

## Data source: Dhan (default) or Yahoo

Pick the source with the **Data source** toggle in the sidebar.

**Dhan (default).** Uses the DhanHQ v2 historical API (`/v2/charts/historical`). Dhan
identifies stocks by numeric **security IDs**, so the app downloads Dhan's public
scrip-master once (cached for a day) and maps your NSE/BSE symbols automatically
(~99% of your lists map; anything Dhan doesn't list is skipped and counted).

You need a Dhan **access-token** (JWT) from an account with an **active Data API
subscription** (the data/historical endpoints are a paid add-on; order/portfolio APIs
are free). Provide it either way:

- **Recommended — Streamlit secrets.** In your app's *Settings → Secrets*, add:
  ```
  DHAN_ACCESS_TOKEN = "your_jwt_here"
  DHAN_CLIENT_ID = "your_client_id"   # optional
  ```
  The app picks these up automatically and never shows the token on screen.
- **Or paste it** into the sidebar's *Dhan access-token* box (kept only for the session).

Daily candles come straight from Dhan; **weekly** is resampled from daily (the Dhan
historical endpoint returns daily bars). Dhan rate-limits the data API, so keep
**Fetch threads** modest (default 5) — large scans auto-throttle and cache for the
day, so the first full-universe scan takes several minutes and re-runs are instant.
If every stock returns "no data," your token is wrong/expired or the Data API
subscription isn't active (the app says so).

**Yahoo (fallback).** No token needed — handy if your Dhan subscription isn't set up.
Same engines, same output.

## CLI data source

The command-line runner uses Yahoo by default. To use Dhan, export your token first:
```bash
export DHAN_ACCESS_TOKEN="your_jwt_here"
export DHAN_CLIENT_ID="your_client_id"   # optional
python3 bullish_reversal_screener.py --all
```

## Using it

- **Exchange**: NSE, BSE, or Both (Both prefers NSE for dual-listed names).
- **Timeframe**: Daily or Weekly candles. Weekly applies the same patterns and
  downtrend gate to weekly bars (fetches ~5y of history) — fewer, slower, but more
  significant signals. The latest weekly candle is the in-progress week.
- **Sectors**: optional filter (NSE Sector / BSE Industry).
- **Max stocks to scan**: cap for a quick run, or tick **Scan ALL matching stocks**
  to scan the entire filtered list in one pass. Large scans (>500) auto-throttle
  (fewer threads + a small per-request delay) to respect data-source rate limits
  (stricter on Dhan) — they take a few minutes the first time, then cache for the
  day so re-runs are instant.
- **Scan last N bars**: 1 = only the freshest completed bar (default). 2–3 catches
  signals from the last few sessions.
- **Volume-confirmed only**: show only signals whose signal-bar volume >=1.5x its
  20-day average.
- **Run scan** -> results appear. Each **stock symbol is a clickable link that opens
  its TradingView chart** in a new tab. There's also a button to download a full
  standalone HTML report (symbols clickable there too).

## VCP scanner settings

- **Within % of 52-period high** (default 25%, Minervini standard) — only stocks near
  their highs qualify.
- **Max base tightness %** (default 5%) — the final contraction must be no wider.
- **Min base length** — minimum bars in the tight base.
- **Trend strictness** — *Strict* = textbook Stage 2 (price > 50 > 150 > 200 MA, 200-MA
  rising); *Standard*/*Relaxed* loosen it for recovery setups.
- **Minimum grade** (A/B/C) and **Status** (Coiling/Breakout) filters.

The VCP scan also gates on **volatility contraction** (recent range narrower than
earlier) and **volume dry-up** (base quieter than its 50-bar average), and grades each
base A/B/C from tightness, contraction, dry-up, proximity to high, base length, and
relative strength vs Nifty. It is intentionally strict — on any given day only a
handful of stocks are in a genuine high-grade VCP. Stocks well below their 52-week high
(recovery breakouts) are excluded by design.

## Notes on accuracy

- Symbol mapping: NSE → `SYMBOL.NS`; BSE → `SCRIP_ID.BO` (the alphabetic Scrip ID,
  not the numeric code — that's what Yahoo serves reliably).
- Detection logic lives only in `screener_core.py` and is covered by synthetic
  unit tests: `python3 screener_core.py` (or `--selftest` on the CLI) checks that
  every detector fires after a downtrend and is *rejected* in an uptrend.
- The win-% on each tab is a **historical best-case reversal frequency** (a ranking
  aid), not a tradeable win-rate after costs. This is a research/screening tool,
  not investment advice.
```
