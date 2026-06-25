#!/usr/bin/env python3
"""
vcp_core.py — Volatility Contraction Pattern (VCP) engine.
Finds tight, high-grade bases near the highs that are coiling or breaking out
(TARIL-type). Daily or weekly. Imported by the Streamlit app.

Quality logic (Minervini SEPA-inspired):
  GATES: established uptrend (price above rising medium/long MAs, short>medium MA),
         near the recent high (within near_high_pct), >=30% above the 52-period low,
         a TIGHT final base (span <= max_tight, default 5%).
  CLASSIFY: "Coiling" (price sitting just under the pivot) or
            "Breakout" (today clears a prior tight base on a volume surge).
  GRADE A/B/C from: tightness, volatility contraction, volume dry-up,
            proximity to high, base length, relative strength vs Nifty.
"""
import numpy as np, pandas as pd, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import screener_core as sc   # reuse fetch_ohlcv + tv_url

def tf_params(timeframe):
    if timeframe == "Weekly":
        return dict(maf=10, mam=30, mal=40, win=52, mom=26, funnel=15, rng="5y")
    return dict(maf=50, mam=150, mal=200, win=252, mom=126, funnel=30, rng="2y")

def vcp_arrays(df, p):
    c=df['close'].values.astype(float); h=df['high'].values.astype(float)
    l=df['low'].values.astype(float);  v=df['volume'].values.astype(float)
    sma=lambda x,n: pd.Series(x).rolling(n).mean().values
    return dict(c=c,h=h,l=l,v=v, maf=sma(c,p['maf']), mam=sma(c,p['mam']),
        mal=sma(c,p['mal']), vol50=sma(v,50), n=len(c))

def _ok(x): return x is not None and not (isinstance(x,float) and np.isnan(x))

def tight_base(A, end, max_tight):
    """Walk backward from `end` while the running high-low span stays <= max_tight*close.
    Only bars within the tight band are counted. Returns (#bars, pivot_high, base_low)."""
    c=A['c'][end]; hi=A['h'][end]; lo=A['l'][end]; k=1
    if c<=0 or (hi-lo)/c > max_tight: return 0, hi, lo
    for j in range(end-1, max(-1, end-40), -1):
        nhi=max(hi,A['h'][j]); nlo=min(lo,A['l'][j])
        if (nhi-nlo)/c > max_tight: break
        hi,lo=nhi,nlo; k+=1
    return k, hi, lo

def _funnel(A, end, F, c):
    """Range% over three consecutive thirds of the base ending at `end` (old->new)."""
    w=max(F//3,1)
    def rp(a,b):
        a=max(a,0)
        if b<a: return np.nan
        return (np.nanmax(A['h'][a:b+1])-np.nanmin(A['l'][a:b+1]))/c
    return rp(end-F+1,end-2*w), rp(end-2*w+1,end-w), rp(end-w+1,end)

def _fit(x, y):
    """Least-squares line y = slope*x + intercept. Returns (slope, intercept, r2)."""
    n=len(x)
    if n<2: return 0.0, float(y[-1]), 0.0
    xm=x.mean(); ym=y.mean()
    sxx=float(((x-xm)**2).sum())
    if sxx<=0: return 0.0, float(ym), 0.0
    sxy=float(((x-xm)*(y-ym)).sum())
    slope=sxy/sxx; intercept=float(ym-slope*xm)
    yhat=slope*x+intercept
    ss_res=float(((y-yhat)**2).sum()); ss_tot=float(((y-ym)**2).sum())
    r2=1.0-ss_res/ss_tot if ss_tot>0 else 0.0
    return slope, intercept, float(r2)

def wedge_base(A, end, c, max_tight, min_base, slope_lo, slope_hi, vol50,
               max_len=40, min_r2=0.35):
    """Detect a SLOPING / converging base ending at `end` that the flat-box detector
    (tight_base) structurally cannot see. Fits trendlines to the lows (support) and highs
    (resistance) over candidate window lengths and keeps the best CONVERGING wedge whose
    dominant trendline slope (in %/bar of price) sits inside [slope_lo, slope_hi]:
      Ascending  = higher lows hugging a rising support line, top flat-or-converging.
      Descending = falling resistance over holding lows, narrowing into the apex.
    Returns a base dict with the SAME keys the flat path uses (bk,piv,blo,tight,contr,dry)
    plus type/slope/lineq, or None. NOTE: heuristic — not yet validated by the backtest."""
    lo_all=A['l']; hi_all=A['h']; v_all=A['v']
    best=None
    for L in range(min_base+2, min(max_len, end+1)+1):
        s=end-L+1
        if s<0: break
        lo=lo_all[s:end+1].astype(float); hi=hi_all[s:end+1].astype(float)
        if np.isnan(lo).any() or np.isnan(hi).any(): continue
        x=np.arange(L, dtype=float)
        sl_lo,ic_lo,r2_lo=_fit(x,lo)            # support trendline (lows)
        sl_hi,ic_hi,r2_hi=_fit(x,hi)            # resistance trendline (highs)
        w_start=ic_hi-ic_lo
        w_end=(sl_hi*(L-1)+ic_hi)-(sl_lo*(L-1)+ic_lo)
        if w_start<=0 or w_end<=0: continue
        contr=w_end/w_start
        if contr>0.92: continue                 # require genuine convergence (channel narrows)
        band=w_end/c
        if band>max_tight: continue             # apex must be tight enough to act as a pivot
        slp_sup=sl_lo/c*100.0; slp_res=sl_hi/c*100.0   # slopes as %/bar of price
        asc  = (slp_sup>=slope_lo and r2_lo>=min_r2 and sl_hi<=sl_lo*1.05)
        desc = ((-slp_res)>=slope_lo and r2_hi>=min_r2 and sl_lo>=-abs(sl_hi)*0.5)
        if asc and slp_sup<=slope_hi:
            btype,key,lineq = "Ascending", slp_sup, r2_lo
        elif desc and (-slp_res)<=slope_hi:
            btype,key,lineq = "Descending", slp_res, r2_hi
        else:
            continue
        piv=float(np.nanmax(hi)); blo=float(np.nanmin(lo))
        if piv<=0: continue
        dry=float(np.nanmean(v_all[s:end+1]))/vol50 if (_ok(vol50) and vol50>0) else 1.0
        if dry>1.0: continue                    # base must be quiet (volume dry-up)
        cand=dict(bk=L, piv=piv, blo=blo, tight=band, contr=contr, dry=dry,
                  type=btype, slope=round(key,3), lineq=round(lineq,2))
        rank=(lineq, 1.0-contr, L/float(max_len))   # cleanest line, then most contracted, then longest
        if best is None or rank>best[0]: best=(rank,cand)
    return best[1] if best else None

def analyze_vcp(df, row, timeframe="Daily", near_high_pct=0.12, max_tight=0.05,
                min_base=3, strictness="Strict", nifty_mom_ret=0.0,
                low_dist_on=True, low_dist_min=30.0, low_dist_max=None,
                wedge_on=False, wedge_slope_min=0.05, wedge_slope_max=0.80):
    p=tf_params(timeframe); A=vcp_arrays(df,p); n=A['n']
    if n < max(p['mal'], p['win'])+5: return None
    i=n-1; c=A['c'][i]
    maf,mam,mal=A['maf'][i],A['mam'][i],A['mal'][i]
    if not all(_ok(x) for x in (c,maf,mam,mal)) or c<=0: return None
    # --- trend template gates (Relaxed / Standard / Strict). Strict = textbook Stage 2 ---
    maf_rising = maf > A['maf'][i-10]
    mal_rising = mal > A['mal'][i-20]
    if strictness=="Strict":
        if not (c>maf>mam>mal and mal_rising): return None
    elif strictness=="Relaxed":
        if not (c>mam and maf>mam and maf_rising): return None
    else:  # Standard
        if not (c>maf and c>mam and c>mal and maf>mam and maf_rising): return None
    hi52=np.nanmax(A['h'][i-p['win']+1:i+1]); lo52=np.nanmin(A['l'][i-p['win']+1:i+1])
    nh=(hi52-c)/hi52
    if nh > near_high_pct: return None                       # within near_high_pct of 52-period high
    dist_low=(c/lo52-1.0)*100.0 if (_ok(lo52) and lo52>0) else None   # % above the 52-period low
    if low_dist_on and dist_low is not None:                 # optional distance-from-low band (toggleable)
        if dist_low < low_dist_min: return None              #   floor: keep out names too close to the low
        if low_dist_max is not None and dist_low > low_dist_max: return None  # cap: keep out over-extended names
    vol50=A['vol50'][i]; vol_surge=A['v'][i]/vol50 if _ok(vol50) and vol50>0 else 0
    F=p['funnel']
    def assess(end):
        """Validate a proper VCP base ending at `end`: tight, contracting, quiet."""
        bk,piv,blo=tight_base(A,end,max_tight)
        if bk<min_base or piv<=0: return None
        tight=(piv-blo)/c
        r1,r2,r3=_funnel(A,end,F,c)
        if not _ok(r1) or r1<=0 or not _ok(r3): return None
        contr=r3/r1
        if contr>1.0: return None                           # base must have net-contracted
        seg=A['v'][end-bk+1:end+1]
        dry=float(np.nanmean(seg))/vol50 if _ok(vol50) and vol50>0 else 1.0
        if dry>1.0: return None                             # base must be quiet (volume dry-up)
        return dict(bk=bk,piv=piv,blo=blo,tight=tight,contr=contr,dry=dry)
    status=base=None
    co=assess(i)                                            # coiling: base ends today, price under pivot
    if co and c<=co['piv']*1.001 and c>=co['piv']*(1-0.06):
        status,base="Coiling",co
    else:                                                   # breakout: base ended yesterday, clears pivot on volume
        bo=assess(i-1)
        if bo and c>bo['piv'] and vol_surge>=2.5:           # 2.5x+ surge (backtest: 2.5-4x best)
            status,base="Breakout",bo
    base_type="Flat"; slope=None
    if base is None:
        # flat box found nothing -> optionally look for a sloping/converging wedge base.
        # This only ADDS candidates the flat detector missed; it never alters a flat result.
        if not wedge_on: return None
        wc=wedge_base(A,i,c,max_tight,min_base,wedge_slope_min,wedge_slope_max,vol50)
        if wc and c<=wc['piv']*1.001 and c>=wc['piv']*(1-0.06):
            status,base="Coiling",wc                        # coiling under the wedge pivot
        else:
            wb=wedge_base(A,i-1,c,max_tight,min_base,wedge_slope_min,wedge_slope_max,vol50)
            if wb and c>wb['piv'] and vol_surge>=2.5:
                status,base="Breakout",wb                   # broke the wedge pivot today on volume
        if base is None: return None
        base_type=base['type']; slope=base['slope']
    mom=p['mom']; sret=(c/A['c'][i-mom]-1) if i-mom>=0 and A['c'][i-mom]>0 else 0.0
    rs=sret-(nifty_mom_ret or 0.0)
    # weights re-tuned from a 2,072-trade breakout backtest: proximity to 52w high,
    # relative strength, and a 2.5-4x breakout surge were the strongest return drivers.
    s_tight=14*max(0,1-base['tight']/max_tight)
    s_contr=12*float(np.clip(1-base['contr'],0,1))
    s_dry  =10*float(np.clip((1-base['dry'])/0.5,0,1))
    s_near =18*max(0,1-nh/near_high_pct)
    s_base = 8*float(np.clip((base['bk']-min_base)/8,0,1))
    s_rs   =18*float(np.clip(0.5+rs/0.5,0,1))
    s_surge=10*(1.0 if 2.5<=vol_surge<=4.0 else (0.5 if vol_surge>1.5 else 0.0)) if status=="Breakout" \
            else 10*float(np.clip((1-base['dry'])/0.5,0,1))   # coiling: reward quiet base instead
    s_trend=10
    score=round(s_tight+s_contr+s_dry+s_near+s_base+s_rs+s_surge+s_trend)
    grade="A" if score>=75 else "B" if score>=60 else "C"
    return dict(symbol=row['symbol'], name=row.get('name',''), exch=row.get('exch',''),
        sector=row.get('sector',''), close=round(c,2), status=status, grade=grade, score=score,
        base_len=int(base['bk']), tightness=round(base['tight']*100,1),
        contraction=round(base['contr'],2), dryup=round(base['dry'],2),
        near_high=round(nh*100,1), low_dist=(round(dist_low,1) if dist_low is not None else None),
        rs=round(rs*100,1), pivot=round(base['piv'],2),
        stop=round(base['blo'],2), dist_pivot=round((base['piv']-c)/c*100,2),
        vol_surge=round(vol_surge,2), target=round(c+2*(c-base['blo']),2),
        base_type=base_type, slope=slope,
        date=str(df['date'].iloc[i].date()))

def nifty_mom(timeframe="Daily", fetch_fn=None, nifty_row=None):
    """Benchmark momentum return over the lookback. fetch_fn(row)->df; nifty_row identifies
    the index for the chosen data source. Returns 0.0 if unavailable (RS just neutralises)."""
    p=tf_params(timeframe)
    if fetch_fn is None:
        fetch_fn=lambda row: sc.fetch_ohlcv(row["yahoo"], rng=p['rng'],
                              interval="1wk" if timeframe=="Weekly" else "1d")
    if nifty_row is None:
        nifty_row={"yahoo":"^NSEI","symbol":"NIFTY","exch":"NSE",
                   "security_id":"13","exchange_segment":"IDX_I","instrument":"INDEX"}
    try:
        df=fetch_fn(nifty_row)
    except Exception:
        return 0.0
    if df is None or len(df)<p['mom']+2: return 0.0
    c=df['close'].values
    return float(c[-1]/c[-1-p['mom']]-1)

def build_vcp_html(rows, scanned, failed, timeframe="Daily", summary=""):
    from datetime import datetime
    today=datetime.now().strftime("%d %b %Y %H:%M")
    gcol={"A":"#1b7a2f","B":"#0d47a1","C":"#8a6d00"}
    def _mc(v):
        if v is None: return "&mdash;"
        return f"{v/1e5:.2f}L Cr" if v>=1e5 else (f"{v:,.0f} Cr" if v>=1000 else f"{v:,.1f} Cr")
    trs=""
    for r in rows:
        g=r['grade']; st_="Breakout" if r['status']=="Breakout" else "Coiling"
        stcss="bk" if r['status']=="Breakout" else "co"
        ld_=f"{r['low_dist']}%" if r.get('low_dist') is not None else "&mdash;"
        bt_=r.get('base_type','Flat'); btcss={"Ascending":"asc","Descending":"desc"}.get(bt_,"flat")
        sl_=f"{r['slope']:+.2f}%/bar" if r.get('slope') is not None else "&mdash;"
        trs+=(f"<tr><td><span class='grade' style='background:{gcol.get(g,'#445')}'>{g}</span></td>"
              f"<td><span class='stt {stcss}'>{st_}</span></td>"
              f"<td><span class='bt {btcss}'>{bt_}</span></td><td>{sl_}</td>"
              f"<td class='sym'><a href='{sc.tv_url(r['exch'],r['symbol'])}' target='_blank' "
              f"rel='noopener'>{r['symbol']}</a><span class='ex'>{r['exch']}</span></td>"
              f"<td>{r['close']}</td><td>{r['tightness']}%</td><td>{r['base_len']}</td>"
              f"<td>{r['contraction']}</td><td>{r['dryup']}</td><td>{r['near_high']}%</td><td>{ld_}</td>"
              f"<td>{r['rs']}%</td><td>{r['pivot']}</td><td>{r['dist_pivot']}%</td>"
              f"<td>{r['vol_surge']}x</td><td>{r['stop']}</td><td>{r['target']}</td>"
              f"<td>{_mc(r.get('mcap_cr'))}</td><td class='nm'>{r['name']}</td></tr>")
    body=(f"<table><tr><th>Grade</th><th>Status</th><th>Type</th><th>Slope</th><th>Symbol</th><th>Close</th><th>Tight</th>"
          f"<th>Base</th><th>Contr</th><th>Dry</th><th>Near hi</th><th>From low</th><th>RS</th><th>Pivot</th>"
          f"<th>To pivot</th><th>Vol</th><th>Stop</th><th>2R tgt</th><th>Mkt Cap</th><th>Company</th></tr>{trs}</table>"
          if rows else "<div class='empty'>No VCP candidates matched.</div>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VCP Scanner</title><style>
:root{{--ink:#11161d;--mut:#3a4654;--line:#d7dde5;--panel:#f4f7fb}}*{{box-sizing:border-box}}
body{{margin:0;background:#fff;color:var(--ink);font-size:17px;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif}}
.wrap{{max-width:1300px;margin:0 auto;padding:26px 20px 80px}}h1{{font-size:32px;margin:0 0 4px}}
.sub{{font-size:18px;color:var(--mut);margin:0 0 6px}}
.meta{{font-size:16px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:12px 16px;margin:14px 0 18px}}
table{{width:100%;border-collapse:collapse;font-size:16px}}
th,td{{padding:9px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{background:var(--panel);font-size:13px;text-transform:uppercase;color:var(--mut)}}
.sym{{font-weight:800}}.sym a{{color:#0d47a1;text-decoration:none}}.sym a:hover{{text-decoration:underline}}
.ex{{font-size:11px;color:var(--mut);margin-left:5px}}.nm{{color:var(--mut);font-size:13px;white-space:normal}}
.grade{{color:#fff;font-weight:800;padding:3px 11px;border-radius:8px}}
.stt{{font-weight:800;padding:3px 9px;border-radius:7px;font-size:14px}}
.stt.bk{{background:#e7f6ea;color:#1b7a2f}}.stt.co{{background:#eef2f8;color:#0d47a1}}
.bt{{font-weight:800;padding:3px 9px;border-radius:7px;font-size:14px}}
.bt.flat{{background:#eef2f8;color:#3a4654}}.bt.asc{{background:#e7f6ea;color:#1b7a2f}}
.bt.desc{{background:#fff4e0;color:#8a6d00}}
.empty{{padding:22px;background:var(--panel);border:1px dashed var(--line);border-radius:12px;color:var(--mut)}}
.foot{{margin-top:28px;font-size:15px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:16px 18px}}</style></head><body><div class="wrap">
<h1>VCP Breakout Scanner</h1><p class="sub">Tight bases near the highs &middot; {timeframe} &middot; ranked by grade</p>
<div class="meta"><b>{today}</b> &middot; {scanned} scanned &middot; <b>{len(rows)}</b> candidates &middot; {summary}</div>
{body}
<div class="foot"><b>How to read:</b> <b>Coiling</b> = tight base under the pivot, ready; <b>Breakout</b> = today
cleared the pivot on a volume surge (TARIL-type). <b>Type</b>: <b>Flat</b> = horizontal tight box;
<b>Ascending</b> = higher lows on a rising support trendline (accumulation); <b>Descending</b> = falling
resistance over holding lows, coiling into the apex. <b>Slope</b> = trendline slope in %/bar (Flat = &mdash;).
Tight = base span %; Dry = base volume vs 50-bar avg (lower = quieter base); Contr = recent vs earlier range
(lower = contracting); enter on a move through the Pivot, stop at base low. Grade reflects base quality;
Ascending/Descending grades are heuristic and not yet backtested. Research tool, not investment advice.</div>
</div></body></html>"""

def run_vcp_screen(rows, fetch_fn=None, timeframe="Daily", near_high_pct=0.12,
                   max_tight=0.05, min_base=3, strictness="Strict", max_workers=8,
                   progress=None, request_delay=0.0, nifty_ret=0.0,
                   low_dist_on=True, low_dist_min=30.0, low_dist_max=None,
                   wedge_on=False, wedge_slope_min=0.05, wedge_slope_max=0.80):
    fetch_fn = fetch_fn or (lambda row: sc.fetch_ohlcv(row["yahoo"]))
    out=[]; scanned=0; failed=[]; total=len(rows); done=0
    def work(r):
        if request_delay: time.sleep(request_delay)
        return r, fetch_fn(r)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs=[ex.submit(work,r) for r in rows]
        for fut in as_completed(futs):
            r,df=fut.result(); done+=1
            if df is None or len(df)<60: failed.append(r["symbol"])
            else:
                scanned+=1
                try: m=analyze_vcp(df,r,timeframe,near_high_pct,max_tight,min_base,strictness,nifty_ret,
                                   low_dist_on,low_dist_min,low_dist_max,
                                   wedge_on,wedge_slope_min,wedge_slope_max)
                except Exception: m=None
                if m: out.append(m)
            if progress: progress(done,total,r["symbol"])
    grade_rank={"A":0,"B":1,"C":2}
    out.sort(key=lambda x:(grade_rank.get(x['grade'],9), -x['score']))
    return out, scanned, failed


# ----------------------------------------------------------------- test ----
if __name__=="__main__":
    import sys
    syms=[("TARIL","Transformers & Rectifiers","NSE"),("RELIANCE","Reliance","NSE"),
          ("BHARTIARTL","Bharti Airtel","NSE"),("SIEMENS","Siemens","NSE"),
          ("ABB","ABB","NSE"),("BEL","BEL","NSE"),("HAL","HAL","NSE"),
          ("TITAN","Titan","NSE"),("DIXON","Dixon","NSE"),("POLYCAB","Polycab","NSE"),
          ("APLAPOLLO","APL Apollo","NSE"),("PERSISTENT","Persistent","NSE"),
          ("TRENT","Trent","NSE"),("MAZDOCK","Mazagon","NSE"),("CDSL","CDSL","NSE")]
    tf = "Daily"
    nret = nifty_mom(tf)
    print(f"Nifty {tf} momentum return: {nret*100:.1f}%\n")
    p=tf_params(tf)
    ffn=lambda y: sc.fetch_ohlcv(y, rng=p['rng'], interval="1d")
    for s,nm,ex in syms:
        df=ffn(s+".NS")
        if df is None: print(f"{s}: no data"); continue
        # show trend diagnostics for TARIL specifically
        m=analyze_vcp(df,{"symbol":s,"name":nm,"exch":ex},tf,0.25,0.05,3,"Strict",nret)
        if m:
            print(f"{s:<11} {m['status']:<9} grade {m['grade']} ({m['score']})  "
                  f"tight {m['tightness']}%  base {m['base_len']}  contr {m['contraction']}  "
                  f"dry {m['dryup']}  nearHi {m['near_high']}%  RS {m['rs']}%  pivot {m['pivot']}")
        else:
            A=vcp_arrays(df,p); i=A['n']-1; c=A['c'][i]
            hi=np.nanmax(A['h'][i-p['win']+1:i+1])
            print(f"{s:<11} -- no VCP. c{c:.0f} 50ma{A['maf'][i]:.0f} 150ma{A['mam'][i]:.0f} "
                  f"200ma{A['mal'][i]:.0f} nearHi{(hi-c)/hi*100:.0f}% stacked:"
                  f"{c>A['maf'][i]>A['mam'][i]>A['mal'][i]} 200rising:{A['mal'][i]>A['mal'][i-20]}")
