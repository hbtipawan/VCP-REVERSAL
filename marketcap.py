#!/usr/bin/env python3
"""
marketcap.py — fetch market capitalisation (in ₹ Crore) for a small list of stocks.

Only the screener's RESULT stocks are looked up (tens, not thousands), so a couple of
batched Yahoo quote calls suffice. Uses the cookie+crumb flow; degrades to None on any
failure (the column just shows a dash). Works regardless of the chosen price source.
"""
import urllib.request, urllib.parse, http.cookiejar, json, time
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

def _session():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*")]
    for u in ("https://fc.yahoo.com", "https://finance.yahoo.com/quote/RELIANCE.NS"):
        try: op.open(u, timeout=10).read()
        except Exception: pass
    crumb = op.open("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10).read().decode().strip()
    if not crumb or "<" in crumb: raise RuntimeError("no crumb")
    return op, crumb

def get_marketcaps(pairs):
    """pairs: iterable of (symbol, exch in {'NSE','BSE'}). Returns {(symbol,exch): cr|None}."""
    pairs = list(dict.fromkeys(pairs))
    out = {}
    if not pairs: return out
    try:
        op, crumb = _session()
    except Exception:
        return {p: None for p in pairs}
    ymap = {}                                   # yahoo symbol -> (symbol,exch)
    for sym, exch in pairs:
        ymap[f"{sym}.{'NS' if exch=='NSE' else 'BO'}"] = (sym, exch)
    ys = list(ymap.keys())
    for k in range(0, len(ys), 40):
        chunk = ys[k:k+40]
        url = ("https://query2.finance.yahoo.com/v7/finance/quote?symbols="
               + urllib.parse.quote(",".join(chunk)) + "&crumb=" + urllib.parse.quote(crumb))
        try:
            d = json.loads(op.open(urllib.request.Request(url), timeout=20).read().decode())
            res = (d.get("quoteResponse") or d.get("quoteResult") or {}).get("result") or []
            for r in res:
                mc = r.get("marketCap")
                if mc is None:
                    sh, px = r.get("sharesOutstanding"), r.get("regularMarketPrice")
                    mc = sh*px if (sh and px) else None
                key = ymap.get(r.get("symbol"))
                if key: out[key] = round(mc/1e7, 1) if mc else None
            time.sleep(0.25)
        except Exception:
            continue
    for p in pairs: out.setdefault(p, None)
    return out

def fmt_cr(v):
    if v is None: return "&mdash;"
    if v >= 1e5: return f"{v/1e5:.2f}L Cr"      # >= 1 lakh crore
    if v >= 1000: return f"{v:,.0f} Cr"
    return f"{v:,.1f} Cr"
