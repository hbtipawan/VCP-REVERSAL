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
        return dict(maf=10, mam=30, mal=40, win=52, mom=26, funnel=15, rng="5y", epw=13)
    return dict(maf=50, mam=150, mal=200, win=252, mom=126, funnel=30, rng="2y", epw=65)

def vcp_arrays(df, p):
    c=df['close'].values.astype(float); h=df['high'].values.astype(float)
    l=df['low'].values.astype(float);  v=df['volume'].values.astype(float)
    o=df['open'].values.astype(float) if 'open' in df else c.copy()
    sma=lambda x,n: pd.Series(x).rolling(n).mean().values
    return dict(c=c,h=h,l=l,v=v,o=o, maf=sma(c,p['maf']), mam=sma(c,p['mam']),
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

def flat_base(A, end, c, max_tight, min_base, F, vol50):
    """Validate a proper FLAT VCP base ending at `end`: tight, contracting, quiet.
    Module-level so the live engine and the backtest share ONE implementation."""
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
    return dict(bk=bk,piv=piv,blo=blo,tight=tight,contr=contr,dry=dry,type="Flat")

def htf_base(A, end, c, max_tight, min_base, vol50,
             thrust_min=0.80, flag_max=0.25, flag_min_len=4, flag_max_len=30, pole_win=45):
    """High Tight Flag: a tight, shallow consolidation (the flag) sitting atop a recent
    POWERFUL pole (a >= thrust_min advance into the flag over <= pole_win bars). A continuation
    pattern — the flag may be deeper than a flat base (up to flag_max) but the pole is what
    defines it. Returns a base dict (bk,piv,blo,tight,contr,dry + type/pole) or None.
    Heuristic — NOT yet validated by the backtest."""
    h=A['h']; l=A['l']; v=A['v']
    best=None
    for Lf in range(flag_min_len, min(flag_max_len, end-pole_win)+1):
        s=end-Lf+1
        if s-pole_win<0: continue
        fhi=float(np.nanmax(h[s:end+1])); flo=float(np.nanmin(l[s:end+1]))
        if fhi<=0: continue
        depth=(fhi-flo)/fhi
        if depth>flag_max: continue                     # flag must be shallow
        pole_low=float(np.nanmin(l[s-pole_win:s]))
        if pole_low<=0: continue
        thrust=fhi/pole_low-1.0
        if thrust<thrust_min: continue                  # need a powerful, recent pole
        contr=(fhi-flo)/(fhi-pole_low) if fhi>pole_low else 1.0
        if contr>0.5: continue                          # flag range must be small vs the pole
        dry=float(np.nanmean(v[s:end+1]))/vol50 if (_ok(vol50) and vol50>0) else 1.0
        if dry>1.0: continue                            # quiet flag
        cand=dict(bk=Lf,piv=fhi,blo=flo,tight=depth,contr=round(contr,2),dry=dry,
                  type="HiTightFlag",pole=round(thrust*100,1))
        rank=(thrust,1.0-depth,1.0-contr)               # biggest pole, shallowest, tightest
        if best is None or rank>best[0]: best=(rank,cand)
    return best[1] if best else None

def cup_handle(A, end, c, vol50, min_depth=0.12, max_depth=0.40, handle_max=0.15,
               hmin=3, hmax=25, cmin=18, cmax=130):
    """Cup-with-handle (O'Neil / Minervini). A rounded U cup (depth min..max from the rim,
    rims roughly level, U-shaped not V) followed by a small handle in the cup's UPPER HALF
    that drifts down/flat on volume dry-up. Pivot = handle high, stop = handle low. Returns a
    base dict (bk,piv,blo,tight,contr,dry + type/note/depth) or None. Heuristic \u2014 backtest first."""
    h=A['h']; l=A['l']; v=A['v']
    best=None
    for Hf in range(hmin, hmax+1):
        hs=end-Hf+1
        if hs-cmin<0: break
        h_hi=float(np.nanmax(h[hs:end+1])); h_lo=float(np.nanmin(l[hs:end+1]))
        if h_hi<=0: continue
        h_depth=(h_hi-h_lo)/h_hi
        if h_depth>handle_max: continue                  # handle must be shallow
        for Lc in range(cmin, min(cmax, hs)+1):
            cs=hs-Lc
            if cs<0: break
            seg_h=h[cs:hs]; seg_l=l[cs:hs]               # cup span (left rim -> right rim, no handle)
            if len(seg_l)<3 or np.isnan(seg_l).any(): continue
            lrim=float(np.nanmax(seg_h[:3])); rrim=h_hi
            cup_hi=max(lrim,rrim); cup_lo=float(np.nanmin(seg_l))
            if cup_hi<=0 or cup_lo<=0: continue
            depth=(cup_hi-cup_lo)/cup_hi
            if depth<min_depth or depth>max_depth: continue
            if not (0.88*lrim<=rrim<=1.08*lrim): continue        # rims roughly level
            bottoff=int(np.argmin(seg_l))                        # U not V: bottom near the middle
            if bottoff<0.25*Lc or bottoff>0.75*Lc: continue
            if np.sum(seg_l<=cup_lo+0.20*depth*cup_hi)<max(0.30*Lc,4): continue   # broad, rounded base (rejects V)
            mid=seg_l[int(0.30*Lc):int(0.70*Lc)+1]               # middle third must sit low (no sharp spike)
            if len(mid)==0 or (float(np.nanmean(mid))-cup_lo)>0.35*depth*cup_hi: continue
            if h_lo<cup_lo+0.5*depth*cup_hi: continue            # handle in the upper half of the cup
            if h_depth>0.5*depth: continue                       # handle shallower than the cup
            dry=float(np.nanmean(v[hs:end+1]))/vol50 if (_ok(vol50) and vol50>0) else 1.0
            if dry>1.0: continue                                 # handle volume dry-up
            contr=h_depth/depth
            cand=dict(bk=Hf,piv=h_hi,blo=h_lo,tight=h_depth,contr=round(contr,2),dry=dry,
                      type="CupHandle",depth=round(depth*100,1),note=f"cup {depth*100:.0f}%")
            rank=(depth,1.0-contr,Lc/float(cmax))                # deeper cup, tighter handle, longer cup
            if best is None or rank>best[0]: best=(rank,cand)
            break                                                # first valid cup per handle, then next handle
    return best[1] if best else None

def darvas_box(A, end, c, vol50, conf=3, max_age=60, min_pct=0.03, max_pct=0.40):
    """Darvas box (How I Made $2,000,000): a CEILING (a high not exceeded for `conf` bars) and
    a FLOOR (a low not broken for `conf` bars) bound a consolidation box near the highs. Buy the
    break above the ceiling. Pivot = box top, stop = box bottom. Returns a base dict or None."""
    h=A['h']; l=A['l']; v=A['v']
    ct=None
    for j in range(end-conf, max(end-conf-max_age, conf), -1):           # most recent confirmed ceiling
        if h[j]>=h[j-1] and all(h[j]>=h[j+k] for k in range(1,conf+1)):
            ct=j; break
    if ct is None: return None
    box_top=float(h[ct]); cf=None
    for j in range(ct+1, end-conf+1):                                    # floor after the ceiling
        if all(l[j]<=l[j+k] for k in range(1,conf+1)):
            cf=j; break
    if cf is None: return None
    box_bot=float(l[cf])
    if box_bot<=0 or box_top<=box_bot: return None
    box_pct=(box_top-box_bot)/box_bot
    if box_pct<min_pct or box_pct>max_pct: return None
    dry=float(np.nanmean(v[ct:end+1]))/vol50 if (_ok(vol50) and vol50>0) else 1.0
    return dict(bk=int(max(end-ct,conf)),piv=box_top,blo=box_bot,tight=box_pct,
                contr=round(min(box_pct/max_pct,1.0),2),dry=round(dry,2),
                type="DarvasBox",note=f"box {box_pct*100:.0f}%")

def pocket_pivot(A, end, c, vol50, max_ext=0.05, lookback=10):
    """Pocket pivot (Morales / Kacher): an UP day whose volume exceeds the highest DOWN-day
    volume of the prior `lookback` days, occurring near or through the 10-day MA in an uptrend.
    An early in-base entry (buy today's close), NOT a pivot breakout. Entry = close, stop =
    today's low. Returns a base dict or None."""
    cl=A['c']; lo=A['l']; vv=A['v']
    if end<lookback+1: return None
    if cl[end]<=cl[end-1]: return None                                   # must be an up day
    dv=[vv[j] for j in range(end-lookback, end) if cl[j]<cl[j-1]]
    thr=max(dv) if dv else 0.0
    if not (vv[end]>thr): return None                                    # volume signature
    ma10=float(np.nanmean(cl[end-9:end+1]))
    if ma10<=0: return None
    if not ((lo[end]<=ma10<=c) or abs((c-ma10)/ma10)<=max_ext): return None   # near/through 10dma, not extended
    dayrange=(c-float(lo[end]))/c if c>0 else 0
    if dayrange<=0: return None
    volratio=vv[end]/thr if thr>0 else 2.0
    return dict(bk=1,piv=round(c,2),blo=round(float(lo[end]),2),tight=dayrange,contr=0.5,
                dry=round((thr/vol50) if (_ok(vol50) and vol50>0) else 1.0,2),
                type="PocketPivot",note=f"vol {volratio:.1f}x")

def episodic_pivot(A, end, vol50, move_min=0.10, gap_min=0.05, vol_min=3.0,
                   dorm_win=65, dorm_max=0.30, vol_lookback=10):
    """Episodic Pivot (Bonde / Qullamaggie): a large catalyst-driven gap-up / surge on massive
    volume, OUT OF a stock that has been dormant (flat, not rallied) for months, closing strong.
    OHLCV footprint only -- the catalyst (earnings / news) must be confirmed by the user.
    Entry = break of the event-day HIGH; stop = the event-day LOW. Returns a dict or None."""
    o=A['o']; cl=A['c']; hi=A['h']; lo=A['l']; vv=A['v']
    if end < dorm_win+5: return None
    pc=cl[end-1]
    if pc<=0 or not (_ok(vol50) and vol50>0): return None
    c=cl[end]; H=float(hi[end]); L=float(lo[end])
    move=c/pc-1.0                                   # close-to-close surge
    gap =o[end]/pc-1.0                               # opening gap
    volr=vv[end]/vol50                               # volume vs 50-day average
    rng=H-L
    if rng<=0 or c<=0: return None
    close_pos=(c-L)/rng                              # where it closed in the day's range
    # --- the episode: a big up day, on massive volume, closing strong ---
    if move < move_min: return None
    if volr < vol_min: return None
    if close_pos < 0.5: return None                  # must close in the upper half (held the move)
    if gap < gap_min and move < move_min*1.5: return None   # a real gap, or an exceptionally large surge
    if vv[end] < np.nanmax(vv[end-vol_lookback:end]):       # volume must be a genuine expansion (recent high)
        return None
    # --- dormancy: the stock must NOT have already rallied into the event (flat for months) ---
    base0=cl[end-1-dorm_win]
    pre_run=(pc/base0-1.0) if base0>0 else 0.0
    if pre_run > dorm_max: return None               # already ran up -> not a fresh episodic pivot
    pre_adr=float(np.nanmean((hi[end-dorm_win:end]-lo[end-dorm_win:end])/
                             np.where(cl[end-dorm_win:end]>0, cl[end-dorm_win:end], np.nan)))
    return dict(type="EpisodicPivot", piv=round(H,2), blo=round(L,2), bk=dorm_win,
                tight=rng/c, contr=pre_run, dry=pre_adr, gap=gap, move=move, volr=volr,
                close_pos=close_pos,
                note=(f"gap +{gap*100:.0f}% \u00b7 +{move*100:.0f}% \u00b7 {volr:.1f}x vol"
                      if gap>=0.01 else f"+{move*100:.0f}% \u00b7 {volr:.1f}x vol"))

def analyze_vcp(df, row, timeframe="Daily", near_high_pct=0.12, max_tight=0.05,
                min_base=3, strictness="Strict", nifty_mom_ret=0.0,
                low_dist_on=True, low_dist_min=30.0, low_dist_max=None,
                wedge_on=False, wedge_slope_min=0.05, wedge_slope_max=0.80,
                htf_on=False, htf_thrust_min=0.80, htf_flag_max=0.25,
                cup_on=False, cup_min_depth=0.12, cup_max_depth=0.40, cup_handle_max=0.15,
                darvas_on=False, darvas_min_pct=0.03, darvas_max_pct=0.40,
                pp_on=False, pp_max_ext=0.05,
                ep_on=False, ep_move_min=0.10, ep_gap_min=0.05, ep_vol_min=3.0, ep_dorm_max=0.30):
    p=tf_params(timeframe); A=vcp_arrays(df,p); n=A['n']
    if n < max(p['mal'], p['win'])+5: return None
    i=n-1; c=A['c'][i]
    maf,mam,mal=A['maf'][i],A['mam'][i],A['mal'][i]
    if not all(_ok(x) for x in (c,maf,mam,mal)) or c<=0: return None
    # --- Episodic Pivot: an INDEPENDENT path that runs ONLY when ep_on. EPs erupt from dormant
    # stocks (often below their MAs / far from the highs), so they deliberately bypass the
    # Stage-2 trend and near-high gates below. With ep_on=False this whole block is skipped,
    # so the base-pattern logic that follows is completely unchanged. ---
    if ep_on:
        ep=episodic_pivot(A,i,A['vol50'][i],ep_move_min,ep_gap_min,ep_vol_min,p['epw'],ep_dorm_max)
        if ep:
            hi52e=np.nanmax(A['h'][i-p['win']+1:i+1]); lo52e=np.nanmin(A['l'][i-p['win']+1:i+1])
            nh_e=(hi52e-c)/hi52e if hi52e>0 else 0.0
            dl_e=(c/lo52e-1.0)*100.0 if (_ok(lo52e) and lo52e>0) else None
            mom=p['mom']; sret=(c/A['c'][i-mom]-1) if i-mom>=0 and A['c'][i-mom]>0 else 0.0
            rs_e=sret-(nifty_mom_ret or 0.0)
            sc=round(25*float(np.clip(ep['move']/0.20,0,1)) + 25*float(np.clip(ep['volr']/6.0,0,1))
                     + 15*float(np.clip(max(ep['gap'],0)/0.12,0,1))
                     + 15*float(np.clip(1-ep['contr']/max(ep_dorm_max,1e-9),0,1))
                     + 20*float(np.clip(ep['close_pos'],0,1)))
            gr="A" if sc>=75 else "B" if sc>=60 else "C"
            return dict(symbol=row['symbol'], name=row.get('name',''), exch=row.get('exch',''),
                sector=row.get('sector',''), close=round(c,2), status="Breakout", grade=gr, score=sc,
                base_len=int(ep['bk']), tightness=round(ep['tight']*100,1),
                contraction=round(ep['contr'],2), dryup=round(ep['dry'],2),
                near_high=round(nh_e*100,1), low_dist=(round(dl_e,1) if dl_e is not None else None),
                rs=round(rs_e*100,1), pivot=ep['piv'], stop=ep['blo'],
                dist_pivot=round((ep['piv']-c)/c*100,2), vol_surge=round(ep['volr'],2),
                target=round(c+2*(c-ep['blo']),2), base_type="EpisodicPivot",
                slope=None, pole=None, note=ep['note'], date=str(df['date'].iloc[i].date()))
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
    status=base=None
    co=flat_base(A,i,c,max_tight,min_base,F,vol50)          # coiling: base ends today, price under pivot
    if co and c<=co['piv']*1.001 and c>=co['piv']*(1-0.06):
        status,base="Coiling",co
    else:                                                   # breakout: base ended yesterday, clears pivot on volume
        bo=flat_base(A,i-1,c,max_tight,min_base,F,vol50)
        if bo and c>bo['piv'] and vol_surge>=2.5:           # 2.5x+ surge (backtest: 2.5-4x best)
            status,base="Breakout",bo
    base_type="Flat"; slope=None; pole=None; note=None
    # Fallbacks below ONLY run when the flat box found nothing, and only when toggled on.
    # They can only ADD candidates the flat detector missed; they never alter a flat result.
    if base is None and wedge_on:                           # --- sloping / converging wedge ---
        wc=wedge_base(A,i,c,max_tight,min_base,wedge_slope_min,wedge_slope_max,vol50)
        if wc and c<=wc['piv']*1.001 and c>=wc['piv']*(1-0.06):
            status,base="Coiling",wc
        else:
            wb=wedge_base(A,i-1,c,max_tight,min_base,wedge_slope_min,wedge_slope_max,vol50)
            if wb and c>wb['piv'] and vol_surge>=2.5:
                status,base="Breakout",wb
        if base is not None:
            base_type=base['type']; slope=base['slope']
    if base is None and htf_on:                             # --- high tight flag (continuation) ---
        hc=htf_base(A,i,c,max_tight,min_base,vol50,htf_thrust_min,htf_flag_max)
        if hc and c<=hc['piv']*1.001 and c>=hc['piv']*(1-0.08):
            status,base="Coiling",hc
        else:
            hb=htf_base(A,i-1,c,max_tight,min_base,vol50,htf_thrust_min,htf_flag_max)
            if hb and c>hb['piv'] and vol_surge>=2.5:
                status,base="Breakout",hb
        if base is not None:
            base_type="HiTightFlag"; pole=base.get('pole')
    if base is None and cup_on:                             # --- cup with handle ---
        weekly=(timeframe=="Weekly")
        hmn,hmx,cmn,cmx=(2,8,6,45) if weekly else (3,25,18,130)
        cc=cup_handle(A,i,c,vol50,cup_min_depth,cup_max_depth,cup_handle_max,hmn,hmx,cmn,cmx)
        if cc and c<=cc['piv']*1.001 and c>=cc['piv']*(1-0.06):
            status,base="Coiling",cc
        else:
            cb=cup_handle(A,i-1,c,vol50,cup_min_depth,cup_max_depth,cup_handle_max,hmn,hmx,cmn,cmx)
            if cb and c>cb['piv'] and vol_surge>=2.5:
                status,base="Breakout",cb
        if base is not None:
            base_type="CupHandle"; note=base.get('note')
    if base is None and darvas_on:                          # --- darvas box ---
        dc=darvas_box(A,i,c,vol50,3,60,darvas_min_pct,darvas_max_pct)
        if dc and c<=dc['piv']*1.001 and c>=dc['piv']*(1-0.06):
            status,base="Coiling",dc
        else:
            db=darvas_box(A,i-1,c,vol50,3,60,darvas_min_pct,darvas_max_pct)
            if db and c>db['piv'] and c<=db['piv']*1.04 and vol_surge>=2.5:   # fresh break, not extended
                status,base="Breakout",db
        if base is not None:
            base_type="DarvasBox"; note=base.get('note')
    if base is None and pp_on:                              # --- pocket pivot (early in-base entry) ---
        pp=pocket_pivot(A,i,c,vol50,pp_max_ext)
        if pp:
            status,base="Breakout",pp                        # actionable today (buy the close)
            base_type="PocketPivot"; note=pp.get('note')
    if base is None: return None
    mom=p['mom']; sret=(c/A['c'][i-mom]-1) if i-mom>=0 and A['c'][i-mom]>0 else 0.0
    rs=sret-(nifty_mom_ret or 0.0)
    # weights re-tuned from a 2,072-trade breakout backtest: proximity to 52w high,
    # relative strength, and a 2.5-4x breakout surge were the strongest return drivers.
    tight_ref = {"HiTightFlag":htf_flag_max, "CupHandle":cup_handle_max,
                 "DarvasBox":darvas_max_pct}.get(base_type, max_tight)
    s_tight=14*max(0,1-base['tight']/tight_ref)
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
        base_type=base_type, slope=slope, pole=pole, note=note,
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
        bt_=r.get('base_type','Flat')
        btcss={"Ascending":"asc","Descending":"desc","HiTightFlag":"htf",
               "CupHandle":"cup","DarvasBox":"dvb","PocketPivot":"pp",
               "EpisodicPivot":"ep"}.get(bt_,"flat")
        # numeric sort key for the Slope/Pole column (pole% if HTF, else slope)
        if bt_=="HiTightFlag" and r.get('pole') is not None:
            sl_=f"+{r['pole']:.0f}% pole"; sl_v=r['pole']
        elif r.get('slope') is not None:
            sl_=f"{r['slope']:+.2f}%/bar"; sl_v=r['slope']
        elif r.get('note'):
            sl_=r['note']; sl_v=None
        else:
            sl_="&mdash;"; sl_v=None
        mc_cr=r.get('mcap_cr')
        d=sc._dv
        trs+=(f"<tr><td data-v=\"{d(g)}\"><span class='grade' style='background:{gcol.get(g,'#445')}'>{g}</span></td>"
              f"<td data-v=\"{d(st_)}\"><span class='stt {stcss}'>{st_}</span></td>"
              f"<td data-v=\"{d(bt_)}\"><span class='bt {btcss}'>{bt_}</span></td>"
              f"<td data-v=\"{d(sl_v)}\">{sl_}</td>"
              f"<td class='sym' data-v=\"{d(r['symbol'])}\"><a href='{sc.tv_url(r['exch'],r['symbol'])}' target='_blank' "
              f"rel='noopener'>{r['symbol']}</a><span class='ex'>{r['exch']}</span></td>"
              f"<td data-v=\"{d(r['close'])}\">{r['close']}</td>"
              f"<td data-v=\"{d(r['tightness'])}\">{r['tightness']}%</td>"
              f"<td data-v=\"{d(r['base_len'])}\">{r['base_len']}</td>"
              f"<td data-v=\"{d(r['contraction'])}\">{r['contraction']}</td>"
              f"<td data-v=\"{d(r['dryup'])}\">{r['dryup']}</td>"
              f"<td data-v=\"{d(r['near_high'])}\">{r['near_high']}%</td>"
              f"<td data-v=\"{d(r.get('low_dist'))}\">{ld_}</td>"
              f"<td data-v=\"{d(r['rs'])}\">{r['rs']}%</td>"
              f"<td data-v=\"{d(r['pivot'])}\">{r['pivot']}</td>"
              f"<td data-v=\"{d(r['dist_pivot'])}\">{r['dist_pivot']}%</td>"
              f"<td data-v=\"{d(r['vol_surge'])}\">{r['vol_surge']}x</td>"
              f"<td data-v=\"{d(r['stop'])}\">{r['stop']}</td>"
              f"<td data-v=\"{d(r['target'])}\">{r['target']}</td>"
              f"<td data-v=\"{d(mc_cr)}\">{_mc(mc_cr)}</td>"
              f"<td class='nm' data-v=\"{d(r['name'])}\">{r['name']}</td></tr>")
    def _hh(label, t):
        return f"<th data-t='{t}'>{label}<span class='ar'></span></th>"
    _head=("".join([_hh("Grade","str"),_hh("Status","str"),_hh("Type","str"),_hh("Slope/Pole","num"),
            _hh("Symbol","str"),_hh("Close","num"),_hh("Tight","num"),_hh("Base","num"),_hh("Contr","num"),
            _hh("Dry","num"),_hh("Near hi","num"),_hh("From low","num"),_hh("RS","num"),_hh("Pivot","num"),
            _hh("To pivot","num"),_hh("Vol","num"),_hh("Stop","num"),_hh("2R tgt","num"),
            _hh("Mkt Cap","num"),_hh("Company","str")]))
    body=(f"<table class='sortable'><thead><tr>{_head}</tr></thead><tbody>{trs}</tbody></table>"
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
.bt.desc{{background:#fff4e0;color:#8a6d00}}.bt.htf{{background:#fdeaea;color:#b3261e}}
.bt.cup{{background:#eeedfe;color:#3c3489}}.bt.dvb{{background:#e1f5ee;color:#0f6e56}}
.bt.pp{{background:#fbeaf0;color:#993556}}
.bt.ep{{background:#fff3cd;color:#7a4f00}}
.empty{{padding:22px;background:var(--panel);border:1px dashed var(--line);border-radius:12px;color:var(--mut)}}
.foot{{margin-top:28px;font-size:15px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:16px 18px}}</style>{sc.SORT_CSS}</head><body><div class="wrap">
<h1>VCP Breakout Scanner</h1><p class="sub">Tight bases near the highs &middot; {timeframe} &middot; ranked by grade</p>
<div class="meta"><b>{today}</b> &middot; {scanned} scanned &middot; <b>{len(rows)}</b> candidates &middot; {summary}</div>
<div class="sorthint">Tip: <b>tap any column header</b> to sort &mdash; tap again to reverse.</div>
{body}
<div class="foot"><b>How to read:</b> <b>Coiling</b> = tight base under the pivot, ready; <b>Breakout</b> = today
cleared the pivot on a volume surge (TARIL-type). <b>Type</b>: <b>Flat</b> = horizontal tight box;
<b>Ascending</b> = higher lows on a rising support trendline (accumulation); <b>Descending</b> = falling
resistance over holding lows, coiling into the apex; <b>HiTightFlag</b> = a tight shallow flag atop a recent
powerful pole (the Slope/Pole column shows the pole gain %); <b>CupHandle</b> = a rounded U base + small
upper-half handle (column shows cup depth); <b>DarvasBox</b> = a confirmed ceiling/floor box (column shows
box %); <b>PocketPivot</b> = an early in-base entry on an up-day whose volume tops the prior 10 down-days
(column shows the volume ratio; buy the day's close, not a pivot breakout);
<b>EpisodicPivot</b> = a large catalyst-driven gap/surge on massive volume out of a months-long dormant base
(column shows gap %, move % and volume; confirm the earnings/news catalyst yourself, buy the break of the
event-day high, stop at the event-day low). <b>Slope</b> = trendline slope in %/bar (Flat = &mdash;).
Tight = base span %; Dry = base volume vs 50-bar avg (lower = quieter base); Contr = recent vs earlier range
(lower = contracting); enter on a move through the Pivot, stop at base low. Grade reflects base quality;
Ascending/Descending grades are heuristic and not yet backtested. Research tool, not investment advice.</div>
</div>{sc.SORT_JS}</body></html>"""

def run_vcp_screen(rows, fetch_fn=None, timeframe="Daily", near_high_pct=0.12,
                   max_tight=0.05, min_base=3, strictness="Strict", max_workers=8,
                   progress=None, request_delay=0.0, nifty_ret=0.0,
                   low_dist_on=True, low_dist_min=30.0, low_dist_max=None,
                   wedge_on=False, wedge_slope_min=0.05, wedge_slope_max=0.80,
                   htf_on=False, htf_thrust_min=0.80, htf_flag_max=0.25,
                   cup_on=False, cup_min_depth=0.12, cup_max_depth=0.40, cup_handle_max=0.15,
                   darvas_on=False, darvas_min_pct=0.03, darvas_max_pct=0.40,
                   pp_on=False, pp_max_ext=0.05,
                   ep_on=False, ep_move_min=0.10, ep_gap_min=0.05, ep_vol_min=3.0, ep_dorm_max=0.30):
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
                                   wedge_on,wedge_slope_min,wedge_slope_max,
                                   htf_on,htf_thrust_min,htf_flag_max,
                                   cup_on,cup_min_depth,cup_max_depth,cup_handle_max,
                                   darvas_on,darvas_min_pct,darvas_max_pct,
                                   pp_on,pp_max_ext,
                                   ep_on,ep_move_min,ep_gap_min,ep_vol_min,ep_dorm_max)
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
