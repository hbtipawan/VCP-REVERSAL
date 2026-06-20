#!/usr/bin/env python3
"""
streamlit_app.py — Bullish scanners for NSE/BSE (Dhan or Yahoo data).
Scanners:
  * Reversal patterns  — 16 bullish reversal candle/volume patterns at lows.
  * VCP breakout       — high-grade Minervini-style volatility-contraction bases near highs.
Data source:
  * Dhan (default)     — DhanHQ v2 historical API. Needs an access-token with an active
                         Data API subscription. Set DHAN_ACCESS_TOKEN (and optionally
                         DHAN_CLIENT_ID) in Streamlit secrets, or paste in the sidebar.
  * Yahoo (fallback)   — no token needed.
Run:    streamlit run streamlit_app.py
Deploy: push this + screener_core.py + vcp_core.py + dhan_data.py + EQUITY_L_2.csv
        + bse_stocks.csv + requirements.txt to GitHub, then deploy on share.streamlit.io.
"""
import os, threading, datetime as dt
import pandas as pd
import streamlit as st
import screener_core as sc
import vcp_core as vc
import dhan_data as dd

st.set_page_config(page_title="Bullish Scanners", layout="wide", page_icon=":chart_with_upwards_trend:")
NSE_CSV = "EQUITY_L_2.csv"
BSE_CSV = "bse_stocks.csv"

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px; }
.block-container { padding-top: 1.4rem; max-width: 1340px; }
h1 { font-size: 33px !important; }
button[data-baseweb="tab"] { font-size: 17px !important; font-weight: 700; padding: 8px 12px; }
[data-testid="stMetricValue"] { font-size: 28px; }
table.res { width:100%; border-collapse:collapse; font-size:16px; margin-top:6px; }
table.res th { background:#f4f7fb; color:#3a4654; text-transform:uppercase; font-size:12px;
  letter-spacing:.3px; text-align:left; padding:9px 9px; border-bottom:2px solid #d7dde5; white-space:nowrap;}
table.res td { padding:9px 9px; border-bottom:1px solid #e6ebf1; text-align:left; white-space:nowrap;}
table.res td.sym { font-weight:800; font-size:18px; }
table.res td.sym a { color:#0d47a1; text-decoration:none; }
table.res td.sym a:hover { text-decoration:underline; }
table.res td.sym .ex { font-size:11px; color:#3a4654; margin-left:5px; border:1px solid #d7dde5;
  border-radius:5px; padding:1px 5px; }
table.res td.nm { color:#3a4654; font-size:13px; white-space:normal; }
table.res td.note { color:#3a4654; font-size:14px; white-space:normal; }
table.res .ago { color:#8a93a0; font-size:12px; }
.vok { color:#1b7a2f; font-weight:800; } .vno { color:#aaa; }
.grade { color:#fff; font-weight:800; padding:3px 11px; border-radius:8px; }
.stt { font-weight:800; padding:3px 9px; border-radius:7px; font-size:14px; }
.stt.bk { background:#e7f6ea; color:#1b7a2f; } .stt.co { background:#eef2f8; color:#0d47a1; }
.litline { color:#3a4654; font-size:17px; margin:0 0 8px; }
.gateline { color:#8a6d00; font-size:14px; margin:0 0 10px; }
</style>
""", unsafe_allow_html=True)

def get_secret(k, default=""):
    try: return st.secrets.get(k, default)
    except Exception: return default

# ----------------------------------- list loaders -----------------------------------
def parse_nse(raw):
    raw = raw.rename(columns=lambda c: str(c).strip())
    out = pd.DataFrame({"symbol": raw["companyId"].astype(str).str.strip(),
        "name": raw["Name"].astype(str).str.strip(),
        "sector": raw["Sector"].astype(str).str.strip() if "Sector" in raw else "", "exch": "NSE"})
    out["yahoo"] = out["symbol"] + ".NS"
    return out[out["symbol"].str.len() > 0]

def parse_bse(raw):
    raw = raw.rename(columns=lambda c: str(c).strip())
    if "Status" in raw:
        raw = raw[raw["Status"].astype(str).str.strip().str.lower() == "active"]
    out = pd.DataFrame({"symbol": raw["Scrip ID"].astype(str).str.strip(),
        "name": raw["Scrip Name"].astype(str).str.strip(),
        "sector": raw["Industry"].astype(str).str.strip() if "Industry" in raw else "", "exch": "BSE"})
    out["yahoo"] = out["symbol"] + ".BO"
    return out[out["symbol"].str.len() > 0]

@st.cache_data(show_spinner=False)
def load_bundled(kind):
    path = NSE_CSV if kind == "NSE" else BSE_CSV
    if not os.path.exists(path): return None
    raw = pd.read_csv(path)
    return parse_nse(raw) if kind == "NSE" else parse_bse(raw)

def get_list(kind):
    df = load_bundled(kind)
    if df is not None: return df
    up = st.sidebar.file_uploader(f"{kind} list not found — upload its CSV", type="csv", key=f"up_{kind}")
    if up is not None:
        raw = pd.read_csv(up)
        return parse_nse(raw) if kind == "NSE" else parse_bse(raw)
    return None

@st.cache_data(show_spinner="Loading Dhan instrument master...", ttl=24*3600)
def dhan_maps():
    return dd.build_symbol_maps(dd.load_scrip_master())

# --------------------------- persistent, thread-safe fetch cache --------------------
@st.cache_resource
def _fetch_cache():
    return {}, threading.Lock()

def make_fetch(source, weekly, years, yahoo_rng, yahoo_interval, token, client, daystamp):
    cache, lock = _fetch_cache()
    def f(row):
        rid = row.get("security_id") if source == "Dhan" else row.get("yahoo")
        key = (source, rid, weekly, daystamp)
        with lock:
            if key in cache: return cache[key]
        if source == "Dhan":
            to_d = dt.date.today(); from_d = to_d - dt.timedelta(days=int(years*365)+15)
            df = dd.fetch_daily(row["security_id"], row["exchange_segment"],
                                row.get("instrument","EQUITY"), from_d.isoformat(),
                                to_d.isoformat(), token, client)
            if df is not None and weekly: df = dd.to_weekly(df)
        else:
            df = sc.fetch_ohlcv(row["yahoo"], rng=yahoo_rng, interval=yahoo_interval)
        with lock: cache[key] = df
        return df
    return f

# ----------------------------------- table renderers --------------------------------
def reversal_table(rows, unit="d"):
    trs = ""
    for r in rows:
        badge = "<span class='vok'>&#10003; vol</span>" if r["vol_confirmed"] else "<span class='vno'>&mdash;</span>"
        ago = "today" if r["bars_ago"] == 0 else f"{r['bars_ago']}{unit} ago"
        vr = f"{r['vol_ratio']}x" if r["vol_ratio"] is not None else "&mdash;"
        trs += (f"<tr><td class='sym'><a href='{sc.tv_url(r['exch'], r['symbol'])}' target='_blank' "
                f"rel='noopener'>{r['symbol']}</a><span class='ex'>{r['exch']}</span></td>"
                f"<td>{r['close']}</td><td>{r['date']}<br><span class='ago'>{ago}</span></td>"
                f"<td>{vr} {badge}</td><td>{r['stop']}</td><td>{r['risk']}%</td>"
                f"<td>{r['target']}</td><td class='nm'>{r['name']}</td><td class='note'>{r['note']}</td></tr>")
    return ("<table class='res'><tr><th>Symbol</th><th>Close</th><th>Signal</th><th>Vol vs avg</th>"
            "<th>Stop</th><th>Risk</th><th>2R target</th><th>Company</th><th>Read</th></tr>" + trs + "</table>")

def vcp_table(rows):
    gcol = {"A": "#1b7a2f", "B": "#0d47a1", "C": "#8a6d00"}
    trs = ""
    for r in rows:
        st_ = "Breakout" if r['status'] == "Breakout" else "Coiling"
        stcss = "bk" if r['status'] == "Breakout" else "co"
        trs += (f"<tr><td><span class='grade' style='background:{gcol.get(r['grade'],'#445')}'>{r['grade']}</span></td>"
                f"<td><span class='stt {stcss}'>{st_}</span></td>"
                f"<td class='sym'><a href='{sc.tv_url(r['exch'],r['symbol'])}' target='_blank' rel='noopener'>"
                f"{r['symbol']}</a><span class='ex'>{r['exch']}</span></td>"
                f"<td>{r['close']}</td><td>{r['tightness']}%</td><td>{r['base_len']}</td>"
                f"<td>{r['contraction']}</td><td>{r['dryup']}</td><td>{r['near_high']}%</td>"
                f"<td>{r['rs']}%</td><td>{r['pivot']}</td><td>{r['dist_pivot']}%</td>"
                f"<td>{r['vol_surge']}x</td><td>{r['stop']}</td><td>{r['target']}</td><td class='nm'>{r['name']}</td></tr>")
    return ("<table class='res'><tr><th>Grade</th><th>Status</th><th>Symbol</th><th>Close</th><th>Tight</th>"
            "<th>Base</th><th>Contr</th><th>Dry</th><th>Near hi</th><th>RS</th><th>Pivot</th><th>To pivot</th>"
            "<th>Vol</th><th>Stop</th><th>2R tgt</th><th>Company</th></tr>" + trs + "</table>")

# =================================== SIDEBAR ========================================
st.sidebar.title("Scanner")
scanner = st.sidebar.radio("Mode", ["Reversal patterns", "VCP breakout"], index=0)
source = st.sidebar.radio("Data source", ["Dhan", "Yahoo"], index=0,
            help="Dhan = DhanHQ v2 historical API (needs access-token + Data API subscription). "
                 "Yahoo = no token needed (fallback).")
dhan_token = dhan_client = None
if source == "Dhan":
    dhan_token = get_secret("DHAN_ACCESS_TOKEN", "")
    dhan_client = get_secret("DHAN_CLIENT_ID", "")
    if not dhan_token:
        dhan_token = st.sidebar.text_input("Dhan access-token", type="password",
            help="JWT from your Dhan account (Data API subscription required). "
                 "Better: store it as DHAN_ACCESS_TOKEN in Streamlit secrets.")
    else:
        st.sidebar.caption("Using Dhan token from secrets.")
exch = st.sidebar.radio("Exchange", ["NSE", "BSE", "Both"], index=0)
timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"], index=0)

frames = []
if exch in ("NSE", "Both"):
    nse = get_list("NSE")
    if nse is not None: frames.append(nse)
if exch in ("BSE", "Both"):
    bse = get_list("BSE")
    if bse is not None: frames.append(bse)
if not frames:
    st.title("Bullish Scanners")
    st.warning("Stock-list CSV(s) not found. Place **EQUITY_L_2.csv** (NSE) and/or "
               "**bse_stocks.csv** (BSE) next to this app, or upload them in the sidebar.")
    st.stop()

uni = pd.concat(frames, ignore_index=True)
if exch == "Both":
    nse_syms = set(uni[uni["exch"] == "NSE"]["symbol"].str.upper())
    uni = uni[~((uni["exch"] == "BSE") & (uni["symbol"].str.upper().isin(nse_syms)))]
uni = uni.drop_duplicates(subset=["symbol", "exch"]).reset_index(drop=True)

# Map symbols -> Dhan security IDs (drops anything Dhan doesn't list)
unmapped = []
if source == "Dhan":
    try:
        nse_map, bse_map = dhan_maps()
        uni, unmapped = dd.map_universe(uni, nse_map, bse_map)
    except Exception as e:
        st.title("Bullish Scanners")
        st.error(f"Could not load the Dhan instrument master: {e}")
        st.stop()

sectors = sorted(s for s in uni["sector"].dropna().unique() if s and s != "nan")
chosen = st.sidebar.multiselect("Sectors / industries (optional)", sectors)
if chosen:
    uni = uni[uni["sector"].isin(chosen)]

st.sidebar.markdown(f"**{len(uni)}** stocks match"
                    + (f" ({len(unmapped)} not on Dhan, skipped)" if source == "Dhan" and unmapped else ""))
total_uni = len(uni)
scan_all = st.sidebar.checkbox("Scan ALL matching stocks (no limit)", value=False,
            help="Scans every matching stock. Large scans auto-throttle and cache for the day.")
if scan_all:
    max_n, shuffle = total_uni, False
else:
    slider_max = min(max(total_uni, 50), 2000)
    max_n = st.sidebar.slider("Max stocks to scan", 20, slider_max, min(150, slider_max), step=10)
    shuffle = st.sidebar.checkbox("Random sample (vs first N alphabetically)", value=False)

scan_n = 1; vol_only = False
near_high = 25; max_tight = 5; min_base = 3; strictness = "Strict"; min_grade = "All (A/B/C)"; status_f = "All"
if scanner == "Reversal patterns":
    scan_n  = st.sidebar.slider("Scan signals from last N bars", 1, 3, 1)
    vol_only = st.sidebar.checkbox("Show volume-confirmed signals only", value=False)
else:
    st.sidebar.markdown("**VCP settings**")
    near_high = st.sidebar.slider("Within % of 52-period high", 10, 40, 25)
    max_tight = st.sidebar.slider("Max base tightness (%)", 2, 10, 5)
    min_base  = st.sidebar.slider("Min base length (bars)", 3, 15, 3)
    strictness = st.sidebar.selectbox("Trend strictness", ["Strict", "Standard", "Relaxed"], index=0)
    min_grade = st.sidebar.selectbox("Minimum grade", ["A only", "A & B", "All (A/B/C)"], index=2)
    status_f  = st.sidebar.selectbox("Status", ["All", "Coiling", "Breakout"], index=0)

dft_workers = 5 if source == "Dhan" else 8
workers = st.sidebar.slider("Fetch threads", 1, 16, dft_workers,
            help="Dhan rate-limits the data API; keep this modest. Auto-reduced for very large scans.")
run = st.sidebar.button("Run scan", type="primary", use_container_width=True)

if not scan_all and total_uni > max_n:
    uni = uni.sample(max_n) if shuffle else uni.head(max_n)

# =================================== HEADER =========================================
if scanner == "Reversal patterns":
    st.title("Bullish Reversal Scanner")
    st.markdown("Long-only &middot; tabs ordered by historical reversal frequency. Win-% are best-case "
                "historical frequencies (a ranking aid, not a tradeable win-rate). Every signal passes a "
                "prior-downtrend location gate.")
else:
    st.title("VCP Breakout Scanner")
    st.markdown("High-grade volatility-contraction bases near the highs (Minervini-style). "
                "**Coiling** = tight base under the pivot; **Breakout** = today cleared the pivot on a "
                "volume surge. Strict by design.")
st.caption(f"Data source: **{source}**" + ("  ·  Dhan historical API (Data API subscription required)"
           if source == "Dhan" else "  ·  Yahoo Finance"))

# =================================== RUN ============================================
if run:
    if source == "Dhan" and not dhan_token:
        st.error("Enter your Dhan access-token in the sidebar (or set DHAN_ACCESS_TOKEN in secrets).")
        st.stop()
    rows = uni.to_dict("records")
    daystamp = str(dt.date.today())
    n_scan = len(rows)
    base_workers = workers
    if n_scan > 500:
        base_workers, req_delay = min(workers, 4 if source == "Dhan" else 6), 0.2 if source == "Dhan" else 0.15
        st.info(f"Scanning {n_scan} stocks via {source} with throttling — this can take several minutes"
                + (" (Dhan rate-limits the data API)." if source == "Dhan" else ".")
                + " Results cache for the day, so the next run is instant.")
    else:
        req_delay = 0.05 if source == "Dhan" else 0.0
    weekly = (timeframe == "Weekly")
    yahoo_interval = "1wk" if weekly else "1d"

    bar = st.progress(0.0, text="Starting...")
    def prog(done, total, sym): bar.progress(done / total, text=f"Fetched {done}/{total}  ({sym})")

    if scanner == "Reversal patterns":
        years = 2.5 if weekly else 1.2
        yahoo_rng = "5y" if weekly else "1y"
        fetch_fn = make_fetch(source, weekly, years, yahoo_rng, yahoo_interval, dhan_token, dhan_client, daystamp)
        with st.spinner(f"Screening {n_scan} stocks ({timeframe.lower()})..."):
            try:
                results, scanned, failed = sc.run_screen(rows, fetch_fn=fetch_fn, scan_last_n=scan_n,
                                            max_workers=base_workers, progress=prog, request_delay=req_delay)
            except PermissionError as e:
                bar.empty(); st.error(str(e)); st.stop()
        bar.empty()
        st.session_state["res"] = dict(mode="reversal", results=results, scanned=scanned, failed=failed,
                                       scan_n=scan_n, timeframe=timeframe, source=source,
                                       when=dt.datetime.now().strftime("%d %b %Y %H:%M"))
    else:
        years = 3.5 if weekly else 2.2
        yahoo_rng = vc.tf_params(timeframe)["rng"]
        fetch_fn = make_fetch(source, weekly, years, yahoo_rng, yahoo_interval, dhan_token, dhan_client, daystamp)
        nrow = dd.NIFTY_ROW if source == "Dhan" else {"yahoo": "^NSEI", "symbol": "NIFTY", "exch": "NSE"}
        nret = vc.nifty_mom(timeframe, fetch_fn=fetch_fn, nifty_row=nrow)
        with st.spinner(f"Scanning {n_scan} stocks for VCP bases ({timeframe.lower()})..."):
            try:
                cands, scanned, failed = vc.run_vcp_screen(rows, fetch_fn=fetch_fn, timeframe=timeframe,
                                            near_high_pct=near_high/100, max_tight=max_tight/100,
                                            min_base=min_base, strictness=strictness, max_workers=base_workers,
                                            progress=prog, request_delay=req_delay, nifty_ret=nret)
            except PermissionError as e:
                bar.empty(); st.error(str(e)); st.stop()
        bar.empty()
        st.session_state["res"] = dict(mode="vcp", cands=cands, scanned=scanned, failed=failed,
                                       timeframe=timeframe, source=source,
                                       when=dt.datetime.now().strftime("%d %b %Y %H:%M"))

# =================================== RENDER =========================================
R = st.session_state.get("res")
if R and R["mode"] == "reversal" and scanner == "Reversal patterns":
    results, scanned, failed = R["results"], R["scanned"], R["failed"]
    tf = R.get("timeframe", "Daily"); unit = "w" if tf == "Weekly" else "d"
    disp = {n: [r for r in results[n] if (r["vol_confirmed"] or not vol_only)] for n in sc.PATTERN_NAMES}
    total = sum(len(v) for v in disp.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", scanned); c2.metric("Total signals", total)
    c3.metric("No data / failed", len(failed)); c4.metric("Timeframe", tf)
    st.caption(f"Last run {R['when']} \u00b7 {R.get('source','')} \u00b7 last {R['scan_n']} {tf.lower()} bar(s)")
    if scanned == 0 and failed:
        st.error("No data returned for any stock. If using Dhan, check your access-token and that your "
                 "Data API subscription is active.")
    labels = [f"{wp}% \u00b7 {name} ({len(disp[name])})" for (name, wp, *_ ) in sc.PATTERNS]
    tabs = st.tabs(labels)
    for tab, (name, wp, src, tier, fn, plen, lit) in zip(tabs, sc.PATTERNS):
        with tab:
            st.markdown(f"### {name} &nbsp; <span style='font-size:16px;color:#445'>Tier {tier} \u00b7 "
                        f"{wp}% historical{'' if src else ' (est)'}</span>", unsafe_allow_html=True)
            st.markdown(f"<p class='litline'>{lit}</p>", unsafe_allow_html=True)
            gate = "prior downtrend required" + (" + near a recent low" if plen == 1 else "")
            st.markdown(f"<p class='gateline'>Location gate: {gate}. Volume-confirmed first.</p>", unsafe_allow_html=True)
            if disp[name]: st.markdown(reversal_table(disp[name], unit), unsafe_allow_html=True)
            else: st.info("No fresh signals in this category.")
    html = sc.build_html(results, scanned, failed, R["scan_n"], timeframe=tf)
    st.download_button("Download full HTML report", data=html,
                       file_name=f"reversal_{tf.lower()}_{dt.date.today()}.html", mime="text/html",
                       use_container_width=True)
    if failed:
        with st.expander(f"{len(failed)} symbols returned no data"):
            st.write(", ".join(failed))

elif R and R["mode"] == "vcp" and scanner == "VCP breakout":
    cands, scanned, failed = R["cands"], R["scanned"], R["failed"]
    tf = R.get("timeframe", "Daily")
    allow = {"A only": {"A"}, "A & B": {"A", "B"}, "All (A/B/C)": {"A", "B", "C"}}[min_grade]
    disp = [r for r in cands if r["grade"] in allow and (status_f == "All" or r["status"] == status_f)]
    nA = sum(1 for r in disp if r["grade"] == "A"); nB = sum(1 for r in disp if r["grade"] == "B")
    nBO = sum(1 for r in disp if r["status"] == "Breakout")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", scanned); c2.metric("VCP candidates", len(disp))
    c3.metric("Grade A / B", f"{nA} / {nB}"); c4.metric("Breakouts", nBO)
    st.caption(f"Last run {R['when']} \u00b7 {R.get('source','')} \u00b7 {tf} \u00b7 ranked by grade then score")
    if scanned == 0 and failed:
        st.error("No data returned for any stock. If using Dhan, check your access-token and Data API subscription.")
    if disp:
        st.markdown(vcp_table(disp), unsafe_allow_html=True)
    elif scanned:
        st.info("No VCP candidates matched your filters. VCP is strict by design — loosen the near-high %, "
                "trend strictness, or grade filter, or scan a wider universe.")
    summ = f"{nA} A, {nB} B, {nBO} breakouts"
    html = vc.build_vcp_html(disp, scanned, failed, timeframe=tf, summary=summ)
    st.download_button("Download VCP HTML report", data=html,
                       file_name=f"vcp_{tf.lower()}_{dt.date.today()}.html", mime="text/html",
                       use_container_width=True)
    if failed:
        with st.expander(f"{len(failed)} symbols returned no data"):
            st.write(", ".join(failed))
else:
    st.info("Set your universe and options in the sidebar, then press **Run scan**.")
