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
import os, threading, datetime as dt, io, re, urllib.parse
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import screener_core as sc
import vcp_core as vc
import dhan_data as dd
import upstox_data as ud
import kite_data as kd
import marketcap as mcap

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
def _pick_col(raw, candidates):
    """Return the actual column whose (stripped, lowercased) name matches a candidate."""
    low = {str(c).strip().lower(): c for c in raw.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

_JUNK_SYMS = {"", "NAN", "NONE", "SYMBOL", "STOCK NAME", "SCRIP ID", "TICKER", "COMPANYID"}

def _build_list(raw, exch, sym_cands, name_cands, sec_cands):
    raw = raw.rename(columns=lambda c: str(c).strip())
    sym_col = _pick_col(raw, sym_cands) or raw.columns[0]      # single-column list -> first col
    name_col = _pick_col(raw, name_cands)
    sec_col = _pick_col(raw, sec_cands)
    sym = raw[sym_col].astype(str).str.strip().str.upper()
    out = pd.DataFrame({
        "symbol": sym,
        "name": (raw[name_col].astype(str).str.strip()
                 if (name_col and name_col != sym_col) else sym),
        "sector": (raw[sec_col].astype(str).str.strip() if sec_col else ""),
        "exch": exch})
    out["yahoo"] = out["symbol"] + (".NS" if exch == "NSE" else ".BO")
    out = out[(out["symbol"].str.len() > 0) & (~out["symbol"].isin(_JUNK_SYMS))]
    return out.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

def parse_nse(raw):
    return _build_list(raw, "NSE",
        sym_cands=["companyId", "SYMBOL", "Symbol", "Stock Name", "Ticker", "Security Id", "Scrip ID"],
        name_cands=["Name", "NAME OF COMPANY", "Company Name", "Security Name", "Stock Name"],
        sec_cands=["Sector", "Industry"])

def parse_bse(raw):
    raw = raw.rename(columns=lambda c: str(c).strip())
    if "Status" in raw:                                        # keep only active scrips when present
        raw = raw[raw["Status"].astype(str).str.strip().str.lower() == "active"]
    return _build_list(raw, "BSE",
        sym_cands=["Scrip ID", "Security Id", "SYMBOL", "Symbol", "Stock Name", "Ticker"],
        name_cands=["Scrip Name", "Security Name", "Name", "Company Name", "Stock Name"],
        sec_cands=["Industry", "Sector"])

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

def _clean_sym(v):
    """Normalise one cell into a bare ticker, or '' if it isn't one."""
    s = str(v).strip().upper()
    if not s or s in ("NAN", "SYMBOL", "TICKER", "NAME", "CODE", "SCRIP", "SCRIPCODE", "STOCK",
                      "STOCKS", "COMPANY", "COMPANYID", "SECURITY", "ISIN", "SERIES", "EXCHANGE",
                      "SECTOR", "INDUSTRY", "TRADINGSYMBOL", "NSESYMBOL"):
        return ""
    for suf in (".NS", ".BO", ".NSE", ".BSE"):
        if s.endswith(suf): s = s[:-len(suf)]
    return s.strip()

def parse_symbol_upload(file, exch_default):
    """Robustly read an uploaded CSV/TXT of stock symbols. Detects a symbol column by name,
    else uses the first column; falls back to whitespace/comma-split for header-less lists.
    Returns a list of {symbol,name,sector,exch} dicts (deduped, order preserved)."""
    raw = file.getvalue()
    txt = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    out = []
    parsed = False
    try:
        df = pd.read_csv(io.StringIO(txt))
        if len(df.columns) >= 1 and len(df) >= 1:
            cols = {str(c).lower().strip(): c for c in df.columns}
            keys = ("symbol", "ticker", "companyid", "sc_name", "scrip", "scripcode",
                    "code", "nse symbol", "nsesymbol", "yahoo", "tradingsymbol", "name")
            matched = [cols[k] for k in keys if k in cols]
            if matched:
                pick = matched[0]
            elif len(df.columns) == 1:
                # no recognised header on a single column -> the 'header' is really the first symbol;
                # re-read with no header so that line becomes data too
                df = pd.read_csv(io.StringIO(txt), header=None)
                pick = df.columns[0]
            else:
                pick = df.columns[0]
            seccol = next((cols[k] for k in ("sector", "industry") if k in cols), None)
            exccol = next((cols[k] for k in ("exch", "exchange") if k in cols), None)
            col_vals = df[pick].astype(str).tolist()
            sec_vals = df[seccol].astype(str).tolist() if seccol else None
            exc_vals = df[exccol].astype(str).tolist() if exccol else None
            for j, val in enumerate(col_vals):
                s = _clean_sym(val)
                if not s: continue
                ex = (exc_vals[j].upper().strip() if exc_vals else exch_default)
                ex = ex if ex in ("NSE", "BSE") else exch_default
                out.append((s, (sec_vals[j] if sec_vals else ""), ex))
            parsed = True
    except Exception:
        parsed = False
    if not parsed:                                   # header-less list: split on space/comma/semicolon/newline
        for tok in re.split(r"[\s,;]+", txt):
            s = _clean_sym(tok)
            if s: out.append((s, "", exch_default))
    seen, rows = set(), []
    for s, sec, ex in out:
        if (s, ex) in seen: continue
        seen.add((s, ex))
        rows.append({"symbol": s, "name": s, "sector": (sec if sec and sec != "nan" else ""), "exch": ex})
    return rows

@st.cache_data(show_spinner="Loading Dhan instrument master...", ttl=24*3600)
def dhan_maps():
    return dd.build_symbol_maps(dd.load_scrip_master())

@st.cache_data(show_spinner="Loading Upstox instruments...", ttl=24*3600)
def upstox_maps():
    return ud.build_symbol_maps()

@st.cache_data(show_spinner="Loading Kite instrument list...", ttl=24*3600)
def kite_maps():
    return kd.build_symbol_maps()

@st.cache_data(show_spinner="Logging in to Kite (TOTP)...", ttl=8*3600)
def get_kite_auto_token(api_key, api_secret, user_id, password, totp_secret, daystamp):
    """Generate + cache a fresh Kite access-token via TOTP. daystamp forces one
    regeneration per day. Returns the token string; raises on failure."""
    return kd.auto_login(api_key, api_secret, user_id, password, totp_secret)

@st.cache_data(show_spinner=False, ttl=24*3600)
def market_caps(pairs, daystamp):
    return mcap.get_marketcaps([tuple(p) for p in pairs])

@st.cache_data(show_spinner="Generating Dhan access token...", ttl=23*3600)
def get_dhan_auto_token(client_id, pin, totp_secret, daystamp):
    """Generate (and cache for the day) a fresh 24h token via TOTP. daystamp forces
    one regeneration per day. Returns the token string; raises on failure."""
    tok, _exp = dd.generate_token(client_id, pin, totp_secret)
    return tok

# --------------------------- persistent, thread-safe fetch cache --------------------
def _source_maps(source):
    if source == "Kite":   return kite_maps()
    if source == "Upstox": return upstox_maps()
    if source == "Dhan":   return dhan_maps()
    return None, None

def _source_mapper(source):
    return {"Kite": kd.map_universe, "Upstox": ud.map_universe, "Dhan": dd.map_universe}.get(source)

def map_smart(source, uni, yahoo_fallback):
    """Map an uploaded/bundled universe to the chosen broker with two safety nets so
    nothing is silently dropped:
      1) a symbol that misses on its tagged exchange is retried on the OTHER exchange
         (many names are tagged NSE in the CSV but the broker lists them on BSE, or vice-versa);
      2) symbols still absent from the broker's instrument master (typically NSE-Emerge /
         BSE-SME scrips the broker API doesn't carry) are optionally kept as Yahoo-fetch rows,
         so they are still scanned rather than skipped.
    Returns (mapped_df, unmapped_list, n_yahoo_fallback)."""
    nse_map, bse_map = _source_maps(source)
    mapper = _source_mapper(source)
    base = uni.copy()
    base["symbol"] = base["symbol"].astype(str).str.upper()
    def _got(df): return set(df["symbol"].astype(str).str.upper()) if len(df) else set()
    mapped, _ = mapper(base, nse_map, bse_map)
    got = _got(mapped)
    # 1) retry the misses on the opposite exchange
    miss = base[~base["symbol"].isin(got)].copy()
    if len(miss):
        miss["exch"] = miss["exch"].map({"NSE": "BSE", "BSE": "NSE"}).fillna(miss["exch"])
        m2, _ = mapper(miss, nse_map, bse_map)
        if len(m2):
            mapped = pd.concat([mapped, m2], ignore_index=True); got |= _got(m2)
    # 2) whatever the broker still can't map -> Yahoo rows (or reported as skipped)
    still = base[~base["symbol"].isin(got)].copy()
    n_yahoo = 0; unmapped = []
    if len(still):
        if yahoo_fallback:
            still["yahoo"] = still.apply(
                lambda r: str(r["symbol"]).upper() + (".NS" if r["exch"] == "NSE" else ".BO"), axis=1)
            for c in ("instrument_token", "instrument_key", "security_id"):
                if c in mapped.columns and c not in still.columns:
                    still[c] = None
            mapped = pd.concat([mapped, still], ignore_index=True); n_yahoo = len(still)
        else:
            unmapped = [f"{r.symbol}.{r.exch}" for r in still.itertuples()]
    return mapped.reset_index(drop=True), unmapped, n_yahoo

@st.cache_resource
def _fetch_cache():
    return {}, threading.Lock()

def make_fetch(source, weekly, years, yahoo_rng, yahoo_interval, token, client, daystamp,
               kite_at=None, kite_key=None):
    cache, lock = _fetch_cache()
    def _has(v):
        if v is None: return False
        try:
            if isinstance(v, float) and pd.isna(v): return False
        except Exception: pass
        return str(v).strip() not in ("", "nan", "None")
    def f(row):
        bid = (row.get("security_id") if source == "Dhan"
               else row.get("instrument_key") if source == "Upstox"
               else row.get("instrument_token") if source == "Kite"
               else None)
        # broker row with no id (SME/Emerge the broker doesn't list) -> Yahoo fallback
        use_yahoo = (source == "Yahoo") or (source in ("Dhan", "Upstox", "Kite") and not _has(bid))
        rid = ("Y:" + str(row.get("yahoo"))) if use_yahoo else bid
        key = (source, rid, weekly, daystamp)
        with lock:
            if key in cache: return cache[key]
        if use_yahoo:
            df = sc.fetch_ohlcv(row["yahoo"], rng=yahoo_rng, interval=yahoo_interval)
        elif source == "Dhan":
            to_d = dt.date.today(); from_d = to_d - dt.timedelta(days=int(years*365)+15)
            df = dd.fetch_daily(row["security_id"], row["exchange_segment"],
                                row.get("instrument","EQUITY"), from_d.isoformat(),
                                to_d.isoformat(), token, client)
            if df is not None and weekly: df = dd.to_weekly(df)
        elif source == "Upstox":
            to_d = dt.date.today(); from_d = to_d - dt.timedelta(days=int(years*365)+15)
            df = ud.fetch_daily(row["instrument_key"], from_d.isoformat(), to_d.isoformat())
            if df is not None and weekly: df = ud.to_weekly(df)
        elif source == "Kite":
            to_d = dt.date.today(); from_d = to_d - dt.timedelta(days=int(years*365)+15)
            df = kd.fetch_daily(row["instrument_token"], from_d.isoformat(),
                                to_d.isoformat(), kite_at, kite_key)
            if df is not None and weekly: df = kd.to_weekly(df)
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
                f"<td>{r['target']}</td><td>{mcap.fmt_cr(r.get('mcap_cr'))}</td>"
                f"<td class='nm'>{r['name']}</td><td class='note'>{r['note']}</td></tr>")
    return ("<table class='res'><tr><th>Symbol</th><th>Close</th><th>Signal</th><th>Vol vs avg</th>"
            "<th>Stop</th><th>Risk</th><th>2R target</th><th>Mkt Cap</th><th>Company</th><th>Read</th></tr>" + trs + "</table>")

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
                f"<td>{r['vol_surge']}x</td><td>{r['stop']}</td><td>{r['target']}</td>"
                f"<td>{mcap.fmt_cr(r.get('mcap_cr'))}</td><td class='nm'>{r['name']}</td></tr>")
    return ("<table class='res'><tr><th>Grade</th><th>Status</th><th>Symbol</th><th>Close</th><th>Tight</th>"
            "<th>Base</th><th>Contr</th><th>Dry</th><th>Near hi</th><th>RS</th><th>Pivot</th><th>To pivot</th>"
            "<th>Vol</th><th>Stop</th><th>2R tgt</th><th>Mkt Cap</th><th>Company</th></tr>" + trs + "</table>")

# =============================== SORTABLE TABLE ====================================
_SORT_CSS = """<style>
*{box-sizing:border-box}
body{margin:0;background:#0e1117;color:#e7ebf2;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif}
table{border-collapse:collapse;width:100%;font-size:16px}
thead th{position:sticky;top:0;background:#0f1622;color:#bcd6ff;z-index:2;padding:12px 12px;
  text-align:right;white-space:nowrap;cursor:pointer;border-bottom:2px solid #2c3a55;font-weight:700;
  user-select:none;font-size:15px}
thead th:hover{background:#15203046;color:#fff}
th.l,td.l{text-align:left}
th .ar{color:#5cc8ff;font-size:13px}
tbody td{padding:11px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid #1b2430}
tbody tr:nth-child(even){background:#121822}
tbody tr:hover{background:#1b2740}
a{color:#6cb4ff;text-decoration:none;font-weight:700}
a:hover{text-decoration:underline}
.ex{color:#8190a6;font-size:11px;margin-left:6px;padding:1px 5px;border:1px solid #36435c;border-radius:4px}
.grade{display:inline-block;min-width:22px;padding:3px 9px;border-radius:6px;color:#fff;font-weight:800;text-align:center}
.stt{padding:3px 10px;border-radius:12px;font-size:13px;font-weight:700}
.bk{background:#0f3d20;color:#5be08a;border:1px solid #1f6e3a}
.co{background:#172033;color:#9fb4d4;border:1px solid #2f3e57}
.nm{color:#9aa7bd;font-size:14px;white-space:normal}
.note{color:#8ea0ba;font-size:13px;white-space:normal;max-width:340px}
.vok{color:#5be08a;font-weight:700}.vno{color:#6b7688}
.hint{color:#7e8aa0;font-size:13px;padding:6px 2px 10px}
</style>"""

_SORT_JS = """<script>
const tbl=document.getElementById('tbl'), tb=tbl.tBodies[0], ths=tbl.tHead.rows[0].cells;
let cur=-1, asc=true;
for(let i=0;i<ths.length;i++){ths[i].addEventListener('click',()=>{
  asc=(cur===i)?!asc:true; cur=i;
  const num=ths[i].dataset.t==='num';
  [...tb.rows].sort((a,b)=>{
    let x=a.cells[i].dataset.v, y=b.cells[i].dataset.v;
    if(num){x=parseFloat(x);y=parseFloat(y);if(isNaN(x))x=-1e18;if(isNaN(y))y=-1e18;return asc?x-y:y-x;}
    x=(x||'').toString().toLowerCase();y=(y||'').toString().toLowerCase();
    return asc?(x<y?-1:x>y?1:0):(x>y?-1:x<y?1:0);
  }).forEach(r=>tb.appendChild(r));
  for(let j=0;j<ths.length;j++){const s=ths[j].querySelector('.ar');if(s)s.textContent=(j===i)?(asc?' \u25B2':' \u25BC'):'';}
});}
</script>"""

def _attr(v):
    return str(v).replace('&','&amp;').replace('"','&quot;').replace('<','&lt;').replace('>','&gt;')

def render_sortable(columns, rows, height=None):
    head = "".join(f"<th data-t='{c['type']}' class='{c.get('cls','')}'>{c['label']}<span class='ar'></span></th>"
                   for c in columns)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td data-v=\"{_attr(c['v'](r))}\" class='{c.get('cls','')}'>{c['c'](r)}</td>" for c in columns) + "</tr>"
    html = (_SORT_CSS + "<div class='hint'>Tap any column header to sort &mdash; tap again to reverse.</div>"
            + f"<table id='tbl'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>" + _SORT_JS)
    components.html(html, height=height or min(820, 150 + 44*len(rows)), scrolling=True)

_GCOL = {"A": "#1b7a2f", "B": "#0d47a1", "C": "#8a6d00"}
def _sym_cell(r):
    return (f"<a href='{sc.tv_url(r['exch'], r['symbol'])}' target='_blank' rel='noopener'>{r['symbol']}</a>"
            f"<span class='ex'>{r['exch']}</span>")
def _mc_v(r):
    v = r.get('mcap_cr'); return v if v is not None else -1

def vcp_columns():
    return [
        {"label":"Grade","type":"text","cls":"l","v":lambda r:r['grade'],
         "c":lambda r:f"<span class='grade' style='background:{_GCOL.get(r['grade'],'#445')}'>{r['grade']}</span>"},
        {"label":"Status","type":"text","cls":"l","v":lambda r:r['status'],
         "c":lambda r:f"<span class='stt {'bk' if r['status']=='Breakout' else 'co'}'>{r['status']}</span>"},
        {"label":"Type","type":"text","cls":"l","v":lambda r:r.get('base_type','Flat'),
         "c":lambda r:r.get('base_type','Flat')},
        {"label":"Slope/Pole","type":"num",
         "v":lambda r:(r.get('pole') if r.get('base_type')=='HiTightFlag' and r.get('pole') is not None
                       else (r.get('slope') if r.get('slope') is not None else 0)),
         "c":lambda r:(f"+{r['pole']:.0f}% pole" if r.get('base_type')=='HiTightFlag' and r.get('pole') is not None
                       else (f"{r['slope']:+.2f}/bar" if r.get('slope') is not None
                       else (r['note'] if r.get('note') else "&mdash;")))},
        {"label":"Symbol","type":"text","cls":"l","v":lambda r:r['symbol'],"c":_sym_cell},
        {"label":"Close","type":"num","v":lambda r:r['close'],"c":lambda r:f"{r['close']}"},
        {"label":"Tight","type":"num","v":lambda r:r['tightness'],"c":lambda r:f"{r['tightness']}%"},
        {"label":"Base","type":"num","v":lambda r:r['base_len'],"c":lambda r:f"{r['base_len']}"},
        {"label":"Contr","type":"num","v":lambda r:r['contraction'],"c":lambda r:f"{r['contraction']}"},
        {"label":"Dry","type":"num","v":lambda r:r['dryup'],"c":lambda r:f"{r['dryup']}"},
        {"label":"Near hi","type":"num","v":lambda r:r['near_high'],"c":lambda r:f"{r['near_high']}%"},
        {"label":"From low","type":"num","v":lambda r:(r.get('low_dist') if r.get('low_dist') is not None else -1),
         "c":lambda r:(f"{r['low_dist']}%" if r.get('low_dist') is not None else "&mdash;")},
        {"label":"RS","type":"num","v":lambda r:r['rs'],"c":lambda r:f"{r['rs']}%"},
        {"label":"Pivot","type":"num","v":lambda r:r['pivot'],"c":lambda r:f"{r['pivot']}"},
        {"label":"To pivot","type":"num","v":lambda r:r['dist_pivot'],"c":lambda r:f"{r['dist_pivot']}%"},
        {"label":"Vol","type":"num","v":lambda r:r['vol_surge'],"c":lambda r:f"{r['vol_surge']}x"},
        {"label":"Stop","type":"num","v":lambda r:r['stop'],"c":lambda r:f"{r['stop']}"},
        {"label":"2R tgt","type":"num","v":lambda r:r['target'],"c":lambda r:f"{r['target']}"},
        {"label":"Mkt Cap","type":"num","v":_mc_v,"c":lambda r:mcap.fmt_cr(r.get('mcap_cr'))},
        {"label":"Company","type":"text","cls":"l nm","v":lambda r:r['name'],"c":lambda r:_attr(r['name'])},
    ]

def rev_columns(unit="d"):
    return [
        {"label":"Symbol","type":"text","cls":"l","v":lambda r:r['symbol'],"c":_sym_cell},
        {"label":"Close","type":"num","v":lambda r:r['close'],"c":lambda r:f"{r['close']}"},
        {"label":"When","type":"num","v":lambda r:r['bars_ago'],
         "c":lambda r:("today" if r['bars_ago']==0 else f"{r['bars_ago']}{unit} ago")},
        {"label":"Vol vs avg","type":"num","v":lambda r:(r['vol_ratio'] if r['vol_ratio'] is not None else -1),
         "c":lambda r:(f"{r['vol_ratio']}x" if r['vol_ratio'] is not None else "&mdash;")
                      + (" <span class='vok'>&#10003;</span>" if r['vol_confirmed'] else "")},
        {"label":"Stop","type":"num","v":lambda r:r['stop'],"c":lambda r:f"{r['stop']}"},
        {"label":"Risk","type":"num","v":lambda r:r['risk'],"c":lambda r:f"{r['risk']}%"},
        {"label":"2R tgt","type":"num","v":lambda r:r['target'],"c":lambda r:f"{r['target']}"},
        {"label":"Mkt Cap","type":"num","v":_mc_v,"c":lambda r:mcap.fmt_cr(r.get('mcap_cr'))},
        {"label":"Company","type":"text","cls":"l nm","v":lambda r:r['name'],"c":lambda r:_attr(r['name'])},
        {"label":"Read","type":"text","cls":"l note","v":lambda r:r.get('note',''),"c":lambda r:_attr(r.get('note',''))},
    ]

# =================================== SIDEBAR ========================================
st.sidebar.title("Scanner")
scanner = st.sidebar.radio("Mode", ["Reversal patterns", "VCP breakout"], index=0)
source = st.sidebar.radio("Data source", ["Upstox", "Dhan", "Kite", "Yahoo"], index=0,
            help="Upstox = free, accurate, no token or login needed (recommended). "
                 "Dhan = needs access-token + paid Data API subscription. "
                 "Kite = Zerodha Kite Connect (₹500/mo); daily login (paste token or TOTP auto-login). "
                 "Yahoo = free but less accurate for Indian stocks.")
dhan_token = dhan_client = None
dhan_auth_err = None
kite_at = kite_key = None           # Kite access-token + api_key, set below if source==Kite
kite_auth_err = None
if source == "Dhan":
    dhan_client = get_secret("DHAN_CLIENT_ID", "")
    sec_token = get_secret("DHAN_ACCESS_TOKEN", "")
    sec_pin = get_secret("DHAN_PIN", ""); sec_totp = get_secret("DHAN_TOTP_SECRET", "")
    # auth mode: auto-TOTP if a totp secret is configured, else manual token
    default_mode = "Auto-login (TOTP)" if (sec_totp and dhan_client) else "Paste token"
    auth_mode = st.sidebar.radio("Dhan login", ["Paste token", "Auto-login (TOTP)"],
                    index=0 if default_mode == "Paste token" else 1,
                    help="Auto-login generates a fresh 24h token each day from your Client ID + PIN "
                         "+ TOTP secret, so you never paste a token. Requires TOTP enabled on Dhan.")
    if auth_mode == "Paste token":
        dhan_token = sec_token
        if not dhan_token:
            dhan_token = st.sidebar.text_input("Dhan access-token", type="password",
                help="24h JWT from web.dhan.co. Better: store as DHAN_ACCESS_TOKEN in secrets.")
        elif sec_token:
            st.sidebar.caption("Using token from secrets.")
    else:
        cid = dhan_client or st.sidebar.text_input("Dhan Client ID", value="")
        pin = sec_pin or st.sidebar.text_input("Dhan PIN", type="password")
        totp_secret = sec_totp or st.sidebar.text_input("TOTP secret", type="password",
                        help="The text string shown under the QR code at "
                             "My Profile -> Access DhanHQ APIs -> Setup TOTP.")
        if cid and pin and totp_secret:
            try:
                dhan_token = get_dhan_auto_token(cid, pin, totp_secret, str(dt.date.today()))
                dhan_client = cid
                st.sidebar.caption("Auto-login token ready (refreshes daily).")
            except Exception as e:
                dhan_auth_err = str(e)
                st.sidebar.error("Auto-login failed — see message in main panel.")

elif source == "Kite":
    # API key/secret are app-level (set once). The access-token is what changes daily.
    kite_key    = get_secret("KITE_API_KEY", "")
    kite_secret = get_secret("KITE_API_SECRET", "")
    sec_ktoken  = get_secret("KITE_ACCESS_TOKEN", "")
    sec_kuser   = get_secret("KITE_USER_ID", "")
    sec_kpass   = get_secret("KITE_PASSWORD", "")
    sec_ktotp   = get_secret("KITE_TOTP_SECRET", "")
    today = str(dt.date.today())
    if not kite_key:
        kite_key = st.sidebar.text_input("Kite API key", value="",
                        help="From kite.trade -> your app. Best: store as KITE_API_KEY in secrets.")
    if not kite_secret:
        kite_secret = st.sidebar.text_input("Kite API secret", type="password",
                        help="From kite.trade -> your app. Best: store as KITE_API_SECRET in secrets.")
    # a token generated earlier today survives Streamlit reruns via session_state
    if st.session_state.get("kite_token_day") != today:
        st.session_state.pop("kite_access_token", None)     # expired overnight — drop it
    kite_at = sec_ktoken or st.session_state.get("kite_access_token")

    default_kmode = "Auto-login (TOTP)" if (sec_kuser and sec_kpass and sec_ktotp) else "Paste token"
    kauth_mode = st.sidebar.radio("Kite login", ["Paste token", "Auto-login (TOTP)"],
                    index=0 if default_kmode == "Paste token" else 1,
                    help="Kite needs a fresh login each day. Paste token = click the link, log in, "
                         "paste the request_token from the redirected URL. Auto-login = enter "
                         "User ID + password + TOTP secret once; it logs in for you each day.")

    if not kite_key or not kite_secret:
        st.sidebar.warning("Enter your Kite API key & secret (or set them in secrets) to continue.")
    elif kite_at:
        st.sidebar.success("Kite token ready for today.")
        if st.sidebar.button("Log out of Kite (clear token)"):
            for k in ("kite_access_token", "kite_token_day"): st.session_state.pop(k, None)
            st.rerun()
    elif kauth_mode == "Paste token":
        st.sidebar.markdown(f"**1.** [Log in to Kite →]({kd.login_url(kite_key)})")
        st.sidebar.caption("After login you land on your Redirect URL like "
                           "`https://127.0.0.1/?request_token=XXXX&action=login`. Paste XXXX below "
                           "(or the whole URL).")
        rt_raw = st.sidebar.text_input("2. request_token (or full redirect URL)", value="")
        if rt_raw:
            rt = rt_raw.strip()
            if "request_token=" in rt:                # user pasted the whole URL — extract it
                rt = urllib.parse.parse_qs(urllib.parse.urlparse(rt).query).get("request_token", [rt])[0]
            try:
                kite_at = kd.generate_session(kite_key, kite_secret, rt)
                st.session_state["kite_access_token"] = kite_at
                st.session_state["kite_token_day"] = today
                st.sidebar.success("Logged in — token cached for today.")
            except Exception as e:
                kite_auth_err = str(e)
                st.sidebar.error("Login failed — see message in main panel.")
    else:   # Auto-login (TOTP)
        kuser = sec_kuser or st.sidebar.text_input("Zerodha Client ID (e.g. AB1234)", value="")
        kpass = sec_kpass or st.sidebar.text_input("Zerodha password", type="password")
        ktotp = sec_ktotp or st.sidebar.text_input("TOTP secret", type="password",
                        help="The text key behind the external-2FA TOTP QR on kite.zerodha.com "
                             "(My Profile -> Settings -> account security -> External TOTP).")
        if kuser and kpass and ktotp:
            try:
                kite_at = get_kite_auto_token(kite_key, kite_secret, kuser, kpass, ktotp, today)
                st.session_state["kite_access_token"] = kite_at
                st.session_state["kite_token_day"] = today
                st.sidebar.caption("Auto-login token ready (refreshes daily).")
            except Exception as e:
                kite_auth_err = str(e)
                st.sidebar.error("Auto-login failed — see message in main panel.")
        else:
            st.sidebar.caption("Enter Client ID, password and TOTP secret (or set them in secrets).")
exch = st.sidebar.radio("Exchange", ["NSE", "BSE", "Both"], index=0)
timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"], index=0)

use_upload = st.sidebar.checkbox("Search only within my uploaded stock list", value=False,
                    help="Scan ONLY the symbols in a CSV/TXT you upload (overrides the bundled list and the "
                         "full-universe option). Toggle off to go back to the normal list.")
up_file = None
if use_upload:
    up_file = st.sidebar.file_uploader("Upload symbols (CSV with a 'Symbol' column, or one per line)",
                                       type=["csv", "txt"], key="symbol_universe")
    if up_file is None:
        st.sidebar.warning("Upload a file to activate — using the normal list until you do.")

full_universe = st.sidebar.checkbox("Scan entire NSE/BSE universe (ignore my list)", value=False,
                    help="Screens every cash-equity on the selected exchange(s) straight from the "
                         "data source, instead of your uploaded CSVs.")

yahoo_fill = st.sidebar.checkbox("Scan broker-missing symbols via Yahoo", value=True,
                    help="If the selected broker (Upstox/Dhan/Kite) doesn't list a symbol — typically "
                         "NSE-Emerge / BSE-SME scrips — fetch it from Yahoo instead of skipping it, so "
                         "every stock in your list is scanned. Turn off to only use the broker's own data.")

unmapped = []
n_yahoo = 0
cap_note = None
if use_upload and up_file is not None:
    rows_up = parse_symbol_upload(up_file, "BSE" if exch == "BSE" else "NSE")
    if not rows_up:
        st.title("Bullish Scanners")
        st.error("No stock symbols found in that file. Use a CSV with a **Symbol** column "
                 "(Ticker / companyId / NSE Symbol also work), or a plain list with one symbol per line.")
        st.stop()
    uni = pd.DataFrame(rows_up)
    if source in ("Dhan", "Upstox", "Kite"):
        try:
            uni, unmapped, n_yahoo = map_smart(source, uni, yahoo_fill)
        except Exception as e:
            st.title("Bullish Scanners"); st.error(f"Could not load the {source} instrument list: {e}"); st.stop()
    if "yahoo" not in uni.columns:
        uni["yahoo"] = uni.apply(lambda r: r["symbol"] + (".NS" if r["exch"] == "NSE" else ".BO"), axis=1)
    cap_note = f"your uploaded list ({len(uni)} mapped)"
elif full_universe:
    try:
        if source == "Dhan":
            uni = dd.full_universe(exch, dd.load_scrip_master())
        elif source == "Upstox":
            uni = ud.full_universe(exch)
        elif source == "Kite":
            uni = kd.full_universe(exch)
        else:                                    # Yahoo: use Upstox's symbol list
            uni = ud.full_universe(exch)
    except Exception as e:
        st.title("Bullish Scanners"); st.error(f"Could not load the full universe: {e}"); st.stop()
    if "yahoo" not in uni.columns:
        uni["yahoo"] = uni.apply(lambda r: r["symbol"] + (".NS" if r["exch"] == "NSE" else ".BO"), axis=1)
    cap_note = f"full {exch} universe"
else:
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
    # Map symbols -> source IDs, with other-exchange retry + optional Yahoo fallback
    if source in ("Dhan", "Upstox", "Kite"):
        try:
            uni, unmapped, n_yahoo = map_smart(source, uni, yahoo_fill)
        except Exception as e:
            st.title("Bullish Scanners"); st.error(f"Could not load the {source} instrument list: {e}"); st.stop()

if exch == "Both":
    nse_syms = set(uni[uni["exch"] == "NSE"]["symbol"].str.upper())
    uni = uni[~((uni["exch"] == "BSE") & (uni["symbol"].str.upper().isin(nse_syms)))]
uni = uni.drop_duplicates(subset=["symbol", "exch"]).reset_index(drop=True)

sectors = sorted(s for s in uni["sector"].dropna().unique() if s and s != "nan")
if sectors:
    chosen = st.sidebar.multiselect("Sectors / industries (optional)", sectors)
    if chosen:
        uni = uni[uni["sector"].isin(chosen)]

st.sidebar.markdown(f"**{len(uni)}** stocks "
                    + ("from your uploaded list" if (use_upload and up_file is not None)
                       else "in the full universe" if full_universe else "match")
                    + (f" · {n_yahoo} via Yahoo (not in {source}'s list)"
                       if (not full_universe and n_yahoo) else "")
                    + (f" ({len(unmapped)} not listed, skipped)"
                       if (not full_universe and source in ("Dhan","Upstox","Kite") and unmapped) else ""))
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
low_on = True; low_min = 30.0; low_max = None
wedge_on = False; wedge_lo = 0.05; wedge_hi = 0.80; base_f = "All"
htf_on = False; htf_thrust = 80; htf_flag = 25
cup_on = False; cup_dmin = 12; cup_dmax = 40; cup_hmax = 15
darvas_on = False; darvas_dmin = 3; darvas_dmax = 40
pp_on = False; pp_ext = 5
ep_on = False; ep_move = 10; ep_gap = 5; ep_vol = 3.0; ep_dorm = 30
if scanner == "Reversal patterns":
    scan_n  = st.sidebar.slider("Scan signals from last N bars", 1, 3, 1)
    vol_only = st.sidebar.checkbox("Show volume-confirmed signals only", value=False)
else:
    st.sidebar.markdown("**VCP settings**")
    near_high = st.sidebar.slider("Within % of 52-period high", 5, 40, 12)
    max_tight = st.sidebar.slider("Max base tightness (%)", 2, 10, 5)
    min_base  = st.sidebar.slider("Min base length (bars)", 3, 15, 3)
    strictness = st.sidebar.selectbox("Trend strictness", ["Strict", "Standard", "Relaxed"], index=0)
    min_grade = st.sidebar.selectbox("Minimum grade", ["A only", "A & B", "All (A/B/C)"], index=2)
    status_f  = st.sidebar.selectbox("Status", ["All", "Coiling", "Breakout"], index=0)
    use_low = st.sidebar.checkbox("Filter by distance from 52-week low", value=True,
                help="Keep only stocks whose price sits within this % band above their 52-period low. "
                     "Untick to drop the filter entirely (removes the old \u226530%-above-low floor).")
    if use_low:
        low_lo, low_hi = st.sidebar.slider("Distance above 52-week low (%)", 0, 500, (30, 500), step=5,
                help="Lower handle screens out names still hugging their low (too early). "
                     "Upper handle screens out names that have already run far off the low. "
                     "Leave the upper handle at 500 for no upper cap.")
        low_on = True; low_min = float(low_lo); low_max = None if low_hi >= 500 else float(low_hi)
    else:
        low_on = False; low_min = 30.0; low_max = None
    wedge_on = st.sidebar.checkbox("Also find sloping / wedge bases", value=False,
                help="Adds Ascending (rising-trendline accumulation) and Descending (falling-resistance coil) "
                     "bases that the flat-box detector can't see. Runs only when the flat scan finds nothing, "
                     "so it never changes a flat result. Grades for these are heuristic \u2014 backtest before trusting.")
    if wedge_on:
        wlo, whi = st.sidebar.slider("Wedge trendline slope (% per bar)", 0.0, 1.5, (0.05, 0.80), step=0.05,
                help="Allowed slope of the dominant trendline. Lower handle keeps out near-flat lines "
                     "(use the flat detector for those); upper handle keeps out too-steep, parabolic moves.")
        wedge_lo, wedge_hi = float(wlo), float(whi)
    else:
        wedge_lo, wedge_hi = 0.05, 0.80
    htf_on = st.sidebar.checkbox("Also find High Tight Flags", value=False,
                help="Adds HiTightFlag \u2014 a tight, shallow consolidation atop a recent powerful pole "
                     "(a continuation pattern). Runs only when flat + wedge find nothing, so it never changes "
                     "those results. Grades are heuristic \u2014 backtest before trusting.")
    if htf_on:
        htf_thrust = st.sidebar.slider("Min pole gain into the flag (%)", 40, 150, 80, step=5,
                help="How far price must have run up over ~9 weeks before the flag. Classic HTF \u2248 90\u2013100%+.")
        htf_flag = st.sidebar.slider("Max flag depth (%)", 5, 35, 25, step=1,
                help="Maximum high-to-low retrace inside the flag. Tighter (lower) = higher quality.")
    cup_on = st.sidebar.checkbox("Also find Cup-with-Handle", value=False,
                help="O'Neil/Minervini rounded U base + small upper-half handle on dry volume. "
                     "Buy above the handle high. Runs only when the above find nothing.")
    if cup_on:
        cup_dmin, cup_dmax = st.sidebar.slider("Cup depth (% from rim)", 8, 60, (12, 40), step=1,
                help="Allowed cup depth. O'Neil's classic range is ~12\u201333%; deeper cups are weaker.")
        cup_hmax = st.sidebar.slider("Max handle depth (%)", 5, 25, 15, step=1,
                help="Handle must be shallow and sit in the cup's upper half. Tighter is stronger.")
    darvas_on = st.sidebar.checkbox("Also find Darvas Boxes", value=False,
                help="A confirmed ceiling (high not exceeded for 3 bars) + floor (low not broken for 3 bars). "
                     "Buy the break above the box top. Allows wider boxes than the flat detector.")
    if darvas_on:
        darvas_dmin, darvas_dmax = st.sidebar.slider("Box height (%)", 2, 60, (3, 40), step=1,
                help="Allowed box height (top vs bottom). Too narrow = noise; too wide = loose.")
    pp_on = st.sidebar.checkbox("Also find Pocket Pivots", value=False,
                help="Morales/Kacher early in-base entry: an up-day whose volume tops the highest down-day "
                     "volume of the prior 10 days, near/through the 10dma. Buy the day's close (not a breakout).")
    if pp_on:
        pp_ext = st.sidebar.slider("Max extension above 10dma (%)", 1, 12, 5, step=1,
                help="Skip pivots too far above the 10-day MA (extended). Lower = stricter, earlier entries.")
    ep_on = st.sidebar.checkbox("Also find Episodic Pivots (Bonde / Qullamaggie)", value=False,
                help="A catalyst-driven gap/surge on massive volume out of a months-long DORMANT base. "
                     "Independent of the base-pattern gates (EPs come from stocks below their MAs / far from "
                     "highs). Confirm the earnings/news catalyst yourself; buy the break of the event-day high.")
    if ep_on:
        ep_move = st.sidebar.slider("Min day move (%)", 4, 25, 10, step=1,
                help="Close-to-close surge on the event day. Qullamaggie's classic EP is a 10%+ gap; lower it "
                     "to ~5-6% to catch more NSE mid/small-cap EPs.")
        ep_gap = st.sidebar.slider("Min opening gap (%)", 0, 20, 5, step=1,
                help="The opening gap vs the prior close. 0 also allows strong intraday-ramp EPs (no gap).")
        ep_vol = st.sidebar.slider("Min volume vs 50-day avg (x)", 1.5, 10.0, 3.0, step=0.5,
                help="Volume explosion on the event day. Must also be the highest in the prior 10 sessions.")
        ep_dorm = st.sidebar.slider("Max prior run before the event (%)", 5, 80, 30, step=5,
                help="The stock must NOT have rallied into the event. Caps the gain over the prior ~3 months "
                     "(quarter). Lower = stricter 'dormant base' requirement (the best EPs are flat for 3-6 months).")
    base_f = st.sidebar.selectbox("Base type",
                ["All", "Flat", "Ascending", "Descending", "HiTightFlag", "CupHandle", "DarvasBox", "PocketPivot", "EpisodicPivot"],
                index=0, help="Filter results by base shape. Each non-Flat type needs its own toggle enabled above.")

dft_workers = 3 if source == "Kite" else 5 if source == "Dhan" else 8
workers = st.sidebar.slider("Fetch threads", 1, 16, dft_workers,
            help="Dhan & Kite rate-limit the data API (Kite ~3 req/s); keep modest. "
                 "Auto-reduced for very large scans.")
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
st.caption(f"Data source: **{source}**" + {
    "Dhan": "  ·  Dhan historical API (Data API subscription required)",
    "Upstox": "  ·  Upstox historical API (free, no token or login needed)",
    "Kite": "  ·  Zerodha Kite Connect (₹500/mo; daily login required)",
    "Yahoo": "  ·  Yahoo Finance",
}.get(source, ""))

if source == "Dhan":
    if dhan_auth_err:
        st.error(f"Auto-login error: {dhan_auth_err}")
    cc1, cc2 = st.columns([1, 3])
    if cc1.button("Check Dhan connection", use_container_width=True):
        if not dhan_token:
            st.warning("No token yet — paste a token or set up auto-login in the sidebar first.")
        else:
            prof = dd.get_profile(dhan_token, dhan_client)
            if prof.get("errorCode") or prof.get("errorType"):
                st.error(f"Token rejected: {prof.get('errorMessage', prof)}. The token is invalid or "
                         "expired — regenerate it (tokens last 24h) or fix your credentials.")
            else:
                data_ok = str(prof.get("dataPlan", "")).lower() == "active"
                st.success(f"Token valid for client {prof.get('dhanClientId','?')} "
                           f"(expires {prof.get('tokenValidity','?')}).")
                cols = st.columns(3)
                cols[0].metric("Data API plan", prof.get("dataPlan", "—"))
                cols[1].metric("Data valid till", str(prof.get("dataValidity", "—"))[:10])
                cols[2].metric("Segments", "✓" if prof.get("activeSegment") else "—")
                if not data_ok:
                    st.error("Your **Data API plan is not Active** — that's why historical data fails. "
                             "Historical/EOD data needs the Data API subscription (₹499 + tax/month, or "
                             "free if you've done 25+ trades in the last 30 days). Activate it at "
                             "web.dhan.co → My Profile → Access DhanHQ APIs → **Data APIs** tab, then retry.")
                else:
                    st.info("Data API is active — you're good to run the scan.")

if source == "Kite":
    if kite_auth_err:
        st.error(f"Kite login error: {kite_auth_err}")
    cc1, cc2 = st.columns([1, 3])
    if cc1.button("Check Kite connection", use_container_width=True):
        if not (kite_at and kite_key):
            st.warning("No token yet — log in (paste token or TOTP auto-login) in the sidebar first.")
        else:
            prof = kd.get_profile(kite_key, kite_at)
            if prof.get("status") == "error" or "user_id" not in prof:
                st.error(f"Token rejected: {prof.get('message', prof)}. Log in again — "
                         "Kite tokens expire daily (~6 AM).")
            else:
                st.success(f"Token valid for {prof.get('user_name','?')} ({prof.get('user_id','?')}).")
                cols = st.columns(3)
                cols[0].metric("Broker", prof.get("broker", "—"))
                cols[1].metric("Exchanges", str(len(prof.get("exchanges", []) or [])))
                cols[2].metric("Type", prof.get("user_type", "—"))
                st.info("Connected — you're good to run the scan.")

# =================================== RUN ============================================
if run:
    if source == "Dhan" and not dhan_token:
        st.error("Enter your Dhan access-token in the sidebar (or set DHAN_ACCESS_TOKEN in secrets).")
        st.stop()
    if source == "Kite" and not (kite_at and kite_key):
        st.error("Log in to Kite in the sidebar first (paste the request_token, or set up TOTP auto-login).")
        st.stop()
    rows = uni.to_dict("records")
    daystamp = str(dt.date.today())
    n_scan = len(rows)
    base_workers = workers
    if n_scan > 500:
        if source == "Kite":
            base_workers, req_delay = min(workers, 3), 0.34          # Kite ~3 req/s historical cap
        elif source == "Dhan":
            base_workers, req_delay = min(workers, 4), 0.2
        else:
            base_workers, req_delay = min(workers, 6), 0.15
        st.info(f"Scanning {n_scan} stocks via {source} with throttling — this can take several minutes"
                + (" (Dhan rate-limits the data API)." if source == "Dhan"
                   else " (Kite caps historical calls at ~3/sec)." if source == "Kite" else ".")
                + " Results cache for the day, so the next run is instant.")
    else:
        req_delay = 0.34 if source == "Kite" else 0.05 if source == "Dhan" else 0.0
    weekly = (timeframe == "Weekly")
    yahoo_interval = "1wk" if weekly else "1d"

    bar = st.progress(0.0, text="Starting...")
    def prog(done, total, sym): bar.progress(done / total, text=f"Fetched {done}/{total}  ({sym})")

    if scanner == "Reversal patterns":
        years = 2.5 if weekly else 1.2
        yahoo_rng = "5y" if weekly else "1y"
        fetch_fn = make_fetch(source, weekly, years, yahoo_rng, yahoo_interval, dhan_token, dhan_client, daystamp, kite_at=kite_at, kite_key=kite_key)
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
        fetch_fn = make_fetch(source, weekly, years, yahoo_rng, yahoo_interval, dhan_token, dhan_client, daystamp, kite_at=kite_at, kite_key=kite_key)
        nrow = (dd.NIFTY_ROW if source == "Dhan" else ud.NIFTY_ROW if source == "Upstox"
                else kd.NIFTY_ROW if source == "Kite"
                else {"yahoo": "^NSEI", "symbol": "NIFTY", "exch": "NSE"})
        nret = vc.nifty_mom(timeframe, fetch_fn=fetch_fn, nifty_row=nrow)
        with st.spinner(f"Scanning {n_scan} stocks for VCP bases ({timeframe.lower()})..."):
            try:
                cands, scanned, failed = vc.run_vcp_screen(rows, fetch_fn=fetch_fn, timeframe=timeframe,
                                            near_high_pct=near_high/100, max_tight=max_tight/100,
                                            min_base=min_base, strictness=strictness, max_workers=base_workers,
                                            progress=prog, request_delay=req_delay, nifty_ret=nret,
                                            low_dist_on=low_on, low_dist_min=low_min, low_dist_max=low_max,
                                            wedge_on=wedge_on, wedge_slope_min=wedge_lo, wedge_slope_max=wedge_hi,
                                            htf_on=htf_on, htf_thrust_min=htf_thrust/100, htf_flag_max=htf_flag/100,
                                            cup_on=cup_on, cup_min_depth=cup_dmin/100, cup_max_depth=cup_dmax/100,
                                            cup_handle_max=cup_hmax/100,
                                            darvas_on=darvas_on, darvas_min_pct=darvas_dmin/100, darvas_max_pct=darvas_dmax/100,
                                            pp_on=pp_on, pp_max_ext=pp_ext/100,
                                            ep_on=ep_on, ep_move_min=ep_move/100, ep_gap_min=ep_gap/100,
                                            ep_vol_min=ep_vol, ep_dorm_max=ep_dorm/100)
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
    _allrows = [r for nm in sc.PATTERN_NAMES for r in results[nm]]
    if _allrows and not _allrows[0].get("mcap_done"):
        _caps = market_caps([(r["symbol"], r["exch"]) for r in _allrows], str(dt.date.today()))
        for r in _allrows: r["mcap_cr"] = _caps.get((r["symbol"], r["exch"])); r["mcap_done"] = True
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
            if disp[name]: render_sortable(rev_columns(unit), disp[name])
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
    if cands and not cands[0].get("mcap_done"):
        _caps = market_caps([(r["symbol"], r["exch"]) for r in cands], str(dt.date.today()))
        for r in cands: r["mcap_cr"] = _caps.get((r["symbol"], r["exch"])); r["mcap_done"] = True
    tf = R.get("timeframe", "Daily")
    allow = {"A only": {"A"}, "A & B": {"A", "B"}, "All (A/B/C)": {"A", "B", "C"}}[min_grade]
    disp = [r for r in cands if r["grade"] in allow and (status_f == "All" or r["status"] == status_f)
            and (base_f == "All" or r.get("base_type", "Flat") == base_f)]
    nA = sum(1 for r in disp if r["grade"] == "A"); nB = sum(1 for r in disp if r["grade"] == "B")
    nBO = sum(1 for r in disp if r["status"] == "Breakout")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", scanned); c2.metric("VCP candidates", len(disp))
    c3.metric("Grade A / B", f"{nA} / {nB}"); c4.metric("Breakouts", nBO)
    st.caption(f"Last run {R['when']} \u00b7 {R.get('source','')} \u00b7 {tf} \u00b7 ranked by grade then score")
    if scanned == 0 and failed:
        st.error("No data returned for any stock. If using Dhan, check your access-token and Data API subscription.")
    if disp:
        render_sortable(vcp_columns(), disp)
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
