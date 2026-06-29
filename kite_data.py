#!/usr/bin/env python3
"""
kite_data.py — Zerodha Kite Connect data source for the Bullish Scanners app.

Mirrors the public contract used by dhan_data.py / upstox_data.py so the screener
engine (screener_core / vcp_core) needs ZERO changes. Returns the same tidy
DataFrame everywhere: columns = date, open, high, low, close, volume (lowercase),
date normalised to midnight, NaNs dropped.

Auth model (why Kite differs from Upstox):
  Kite's historical API REQUIRES a daily access-token. Two ways to get one:
    1. MANUAL  — click the login link, log in on Zerodha, you're redirected to your
                 redirect URL with ?request_token=XXX. Paste that token; we exchange
                 it (request_token + checksum) for a 1-day access-token.
    2. AUTO    — give user_id + password + TOTP secret; we drive the Kite web-login
                 endpoints and grab the request_token automatically (regenerates daily).
  The instrument list / symbol-mapping need NO auth (public CSV dump), so the
  universe loads instantly; only the candle fetch needs the token.

Kite Connect v3 endpoints used:
  Instruments dump : https://api.kite.trade/instruments/NSE  (and /BSE)  [public CSV]
  Token exchange   : POST https://api.kite.trade/session/token
  Historical       : GET  https://api.kite.trade/instruments/historical/{token}/day
  Profile          : GET  https://api.kite.trade/user/profile
  Login (manual)   : https://kite.zerodha.com/connect/login?v=3&api_key=XXX
  Login (TOTP)     : api/login -> api/twofa -> connect/login (capture request_token)
"""
import io, csv, json, time, hashlib, urllib.request, urllib.parse, urllib.error, http.cookiejar
import datetime as dt
import pandas as pd

try:
    import pyotp
except Exception:                      # pyotp only needed for the TOTP auto-login path
    pyotp = None

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
API   = "https://api.kite.trade"
KITE  = "https://kite.zerodha.com"

# NIFTY 50 index — fixed Kite instrument_token (NSE indices segment).
NIFTY_ROW = {"instrument_token": 256265, "symbol": "NIFTY", "exch": "NSE", "yahoo": "^NSEI"}


# ============================================================== auth / session ====
def login_url(api_key):
    """The page the user clicks to log in. On success Zerodha redirects to the app's
    registered Redirect URL with ?request_token=... appended."""
    return f"{KITE}/connect/login?v=3&api_key={urllib.parse.quote(str(api_key))}"


def _checksum(api_key, request_token, api_secret):
    return hashlib.sha256((str(api_key) + str(request_token) + str(api_secret)).encode()).hexdigest()


def generate_session(api_key, api_secret, request_token):
    """Exchange a one-time request_token for a 1-day access_token.
    Returns the access_token string; raises RuntimeError with Kite's message on failure."""
    rt = str(request_token).strip()
    body = urllib.parse.urlencode({
        "api_key": api_key, "request_token": rt,
        "checksum": _checksum(api_key, rt, api_secret),
    }).encode()
    req = urllib.request.Request(API + "/session/token", data=body,
                                 headers={"X-Kite-Version": "3", "User-Agent": UA})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=20).read().decode())
    except urllib.error.HTTPError as e:
        try:    msg = json.loads(e.read().decode()).get("message", str(e))
        except Exception: msg = str(e)
        raise RuntimeError(f"Token exchange failed: {msg}")
    except Exception as e:
        raise RuntimeError(f"Token exchange failed: {e}")
    if d.get("status") != "success" or "access_token" not in (d.get("data") or {}):
        raise RuntimeError(f"Token exchange failed: {d.get('message', d)}")
    return d["data"]["access_token"]


def auto_login(api_key, api_secret, user_id, password, totp_secret):
    """Fully automated daily login via TOTP (no manual paste). Drives the Kite web
    endpoints, captures the request_token from the redirect, then exchanges it.
    Requires `pyotp` and external-2FA TOTP set up on the Zerodha account."""
    if pyotp is None:
        raise RuntimeError("pyotp not installed — add `pyotp` to requirements.txt for TOTP auto-login.")
    captured = {"rt": None}

    class _Catch(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if "request_token=" in (newurl or ""):
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(newurl).query)
                rt = qs.get("request_token", [None])[0]
                if rt:
                    captured["rt"] = rt
                    return None                      # stop following — we have it
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj), _Catch())
    op.addheaders = [("User-Agent", UA)]

    # 1) password login -> request_id
    d1 = urllib.parse.urlencode({"user_id": str(user_id).strip(), "password": str(password)}).encode()
    try:
        j1 = json.loads(op.open(KITE + "/api/login", d1, timeout=15).read().decode())
    except Exception as e:
        raise RuntimeError(f"Password login failed: {e}")
    request_id = (j1.get("data") or {}).get("request_id")
    if not request_id:
        raise RuntimeError(f"Password login failed: {j1.get('message', j1)}")

    # 2) TOTP two-factor
    totp = pyotp.TOTP(str(totp_secret).strip()).now()
    d2 = urllib.parse.urlencode({"user_id": str(user_id).strip(), "request_id": request_id,
                                 "twofa_value": totp, "twofa_type": "totp"}).encode()
    try:
        op.open(KITE + "/api/twofa", d2, timeout=15).read()
    except urllib.error.HTTPError as e:
        try:    msg = json.loads(e.read().decode()).get("message", str(e))
        except Exception: msg = str(e)
        raise RuntimeError(f"TOTP step failed: {msg} (check the TOTP secret / device clock).")
    except Exception as e:
        raise RuntimeError(f"TOTP step failed: {e}")

    # 3) hit connect/login — it 302s to the redirect URL carrying the request_token
    try:
        op.open(login_url(api_key), timeout=15)
    except Exception:
        pass                                          # final hop to redirect URL may error; token already captured
    if not captured["rt"]:
        raise RuntimeError("Could not capture request_token — your app's Redirect URL may be "
                           "misconfigured, or Kite changed its login flow. Use 'Paste token' instead.")
    return generate_session(api_key, api_secret, captured["rt"])


def get_profile(api_key, access_token):
    """GET /user/profile — used by the sidebar 'Check Kite connection' button."""
    req = urllib.request.Request(API + "/user/profile",
            headers={"X-Kite-Version": "3", "User-Agent": UA,
                     "Authorization": f"token {api_key}:{access_token}"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
        return d.get("data") or d
    except urllib.error.HTTPError as e:
        try:    return json.loads(e.read().decode())
        except Exception: return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================ instruments / maps ===
def _dump(exchange, retries=3):
    """Download the public instruments CSV for an exchange ('NSE'/'BSE'). No auth.
    Retries on transient 503/429 (Kite occasionally rate-limits the dump endpoint)."""
    url = f"{API}/instruments/{exchange}"
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            return list(csv.DictReader(io.StringIO(raw)))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (503, 429, 502):
                time.sleep(1.0 * (attempt + 1)); continue
            raise
        except Exception as e:
            last = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Could not download Kite instruments for {exchange}: {last}")


def build_symbol_maps():
    """Returns (nse_map, bse_map), each {UPPER_SYMBOL: instrument_token(int)} for cash
    equity only. No auth required."""
    nse_map, bse_map = {}, {}
    for rec in _dump("NSE"):
        if rec.get("instrument_type") == "EQ" and rec.get("segment") == "NSE":
            ts, tok = (rec.get("tradingsymbol") or "").strip().upper(), rec.get("instrument_token")
            if ts and tok: nse_map[ts] = int(tok)
    for rec in _dump("BSE"):
        if rec.get("instrument_type") == "EQ" and rec.get("segment") == "BSE":
            ts, tok = (rec.get("tradingsymbol") or "").strip().upper(), rec.get("instrument_token")
            if ts and tok: bse_map[ts] = int(tok)
    return nse_map, bse_map


def map_universe(uni, nse_map, bse_map):
    """Add `instrument_token` to the universe DataFrame, dropping rows Kite doesn't list.
    Returns (mapped_df, unmapped_symbol_list) — same shape as dd/ud.map_universe."""
    rows, unmapped = [], []
    for r in uni.to_dict("records"):
        sym = str(r.get("symbol", "")).strip().upper()
        ex = r.get("exch", "NSE")
        tok = (nse_map.get(sym) if ex == "NSE" else bse_map.get(sym))
        if tok is None:
            unmapped.append(f"{sym}.{ex}"); continue
        r["instrument_token"] = tok
        r["yahoo"] = sym + (".NS" if ex == "NSE" else ".BO")
        rows.append(r)
    return pd.DataFrame(rows), unmapped


def full_universe(exch):
    """Build a universe DataFrame straight from Kite's instrument dump (all EQ names).
    `exch` in {'NSE','BSE','Both'}. Columns: symbol,name,sector,exch,instrument_token,yahoo."""
    out = []
    if exch in ("NSE", "Both"):
        for rec in _dump("NSE"):
            if rec.get("instrument_type") == "EQ" and rec.get("segment") == "NSE":
                sym = (rec.get("tradingsymbol") or "").strip().upper()
                if sym:
                    out.append({"symbol": sym, "name": (rec.get("name") or sym).strip(), "sector": "",
                                "exch": "NSE", "instrument_token": int(rec["instrument_token"]),
                                "yahoo": sym + ".NS"})
    if exch in ("BSE", "Both"):
        for rec in _dump("BSE"):
            if rec.get("instrument_type") == "EQ" and rec.get("segment") == "BSE":
                sym = (rec.get("tradingsymbol") or "").strip().upper()
                if sym:
                    out.append({"symbol": sym, "name": (rec.get("name") or sym).strip(), "sector": "",
                                "exch": "BSE", "instrument_token": int(rec["instrument_token"]),
                                "yahoo": sym + ".BO"})
    return pd.DataFrame(out)


# ==================================================================== candles ======
def fetch_daily(instrument_token, from_iso, to_iso, access_token, api_key, retries=2):
    """Fetch DAILY candles for one instrument_token between from_iso/to_iso (YYYY-MM-DD).
    Returns df[date,open,high,low,close,volume] (date normalised) or None.
    Kite's day-interval limit (~2000 days/request) comfortably covers our 1-3.5y window."""
    if not (access_token and api_key):
        return None
    url = (f"{API}/instruments/historical/{int(instrument_token)}/day"
           f"?from={urllib.parse.quote(str(from_iso))}&to={urllib.parse.quote(str(to_iso))}")
    headers = {"X-Kite-Version": "3", "User-Agent": UA,
               "Authorization": f"token {api_key}:{access_token}"}
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            d = json.loads(urllib.request.urlopen(req, timeout=25).read().decode())
            candles = ((d.get("data") or {}).get("candles")) or []
            if not candles:
                return None
            cols = ["date", "open", "high", "low", "close", "volume"]
            df = pd.DataFrame([row[:6] for row in candles], columns=cols)
            df["date"] = (pd.to_datetime(df["date"], utc=True)
                          .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize())
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna().reset_index(drop=True)
            return df if len(df) else None
        except urllib.error.HTTPError as e:
            if e.code == 429:                          # rate-limited — back off and retry
                time.sleep(0.7 * (attempt + 1)); continue
            if e.code in (401, 403):                    # token dead — don't waste retries
                return None
            time.sleep(0.4 * (attempt + 1))
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


def to_weekly(df):
    """Resample a daily df to weekly (W-FRI) OHLCV — identical convention to dd/ud."""
    if df is None or df.empty:
        return df
    g = (df.set_index("date").resample("W-FRI")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
           .dropna().reset_index())
    return g if len(g) else None
