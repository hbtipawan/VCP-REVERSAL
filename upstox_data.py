#!/usr/bin/env python3
"""
upstox_data.py — Upstox data layer for the scanners.

Upstox's historical-candle API is FREE and needs NO authentication, no daily login,
and no paid subscription. It is exchange-grade and adjusted for splits/bonuses.
We map your NSE/BSE symbols to Upstox instrument_keys (segment|ISIN) via Upstox's
public instrument JSON, then pull daily candles from the v3 historical endpoint.
Weekly candles are resampled from daily.

Docs: https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/
"""
import io, gzip, json, time, urllib.request, urllib.error, urllib.parse, datetime as dt
import numpy as np, pandas as pd

NSE_JSON = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
BSE_JSON = "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"
HIST_V3  = "https://api.upstox.com/v3/historical-candle"   # /{key}/days/1/{to}/{from}
UA = "Mozilla/5.0 (compatible; screener/1.0)"

# Nifty 50 index for the relative-strength benchmark
NIFTY_ROW = {"symbol": "NIFTY 50", "name": "Nifty 50", "exch": "NSE",
             "instrument_key": "NSE_INDEX|Nifty 50"}

def _get_json_gz(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(gzip.decompress(r.read()))

def load_instruments():
    rows = []
    for url in (NSE_JSON, BSE_JSON):
        try: rows += _get_json_gz(url)
        except Exception: pass
    return rows

def build_symbol_maps(instruments=None):
    """Return (nse_map, bse_map): {UPPER_SYMBOL: instrument_key} for cash equities.
    NSE equities are instrument_type 'EQ'; BSE equities use group codes (A/B/T/X/...),
    so for BSE we take all BSE_EQ rows except debt (F/IF)."""
    instruments = instruments or load_instruments()
    nse, bse = {}, {}
    for d in instruments:
        seg = d.get("segment"); itype = d.get("instrument_type")
        sym = str(d.get("trading_symbol", "")).strip().upper()
        key = d.get("instrument_key")
        if not sym or not key: continue
        if seg == "NSE_EQ" and itype == "EQ" and sym not in nse:
            nse[sym] = key
        elif seg == "BSE_EQ" and itype not in ("F", "IF") and sym not in bse:
            bse[sym] = key
    return nse, bse

def map_universe(uni, nse_map, bse_map):
    keys = []
    for s, e in zip(uni["symbol"].astype(str).str.upper(), uni["exch"]):
        keys.append((nse_map if e == "NSE" else bse_map).get(s))
    out = uni.copy(); out["instrument_key"] = keys
    mapped = out[out["instrument_key"].notna()].reset_index(drop=True)
    unmapped = sorted(out[out["instrument_key"].isna()]["symbol"].tolist())
    return mapped, unmapped

# --------------------------------------------------------- daily fetch ----
def fetch_daily(instrument_key, from_date, to_date, token=None, retries=2):
    ik = urllib.parse.quote(instrument_key, safe="")
    url = f"{HIST_V3}/{ik}/days/1/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "User-Agent": UA}
    if token: headers["Authorization"] = f"Bearer {token}"   # optional; not required
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            candles = (d.get("data") or {}).get("candles") or []
            if not candles: return None
            df = pd.DataFrame(candles, columns=["ts","open","high","low","close","volume","oi"])
            df["date"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).dt.normalize()
            df = df[["date","open","high","low","close","volume"]].iloc[::-1].reset_index(drop=True)
            for c in ("open","high","low","close","volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna(subset=["close"]).reset_index(drop=True)
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(1.2 * (attempt + 1)); continue
            return None
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None

def to_weekly(df):
    if df is None or df.empty: return df
    s = df.set_index("date").sort_index()
    w = s.resample("W-FRI").agg({"open":"first","high":"max","low":"min",
                                 "close":"last","volume":"sum"}).dropna(subset=["close"])
    return w.reset_index()

def make_fetch_fn(weekly=False, years=2.0, token=None):
    to_d = dt.date.today(); from_d = to_d - dt.timedelta(days=int(years*365)+15)
    fs, ts = from_d.isoformat(), to_d.isoformat()
    def f(row):
        df = fetch_daily(row["instrument_key"], fs, ts, token)
        return to_weekly(df) if (weekly and df is not None) else df
    return f
