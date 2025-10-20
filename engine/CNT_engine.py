# === Single-cell CNT quick-look + SAFE checks + Core-vs-Wide (17 µm) ===


# Colab-ready. Incolla i dati in DATA come sequenza di quartine: λ  flux(Jy)  err(Jy)  flag.


# Parser tollerante: ignora i numeri di pagina 1/2/3/4 e gestisce flag attaccati al λ successivo.


# -------------------- PASTE YOUR DATA HERE --------------------


DATA = """

# Replace with your numbers. Tolerated format: a flat sequence λ F σ flag λ F σ flag ...



"""



OBJECT_ID = "object name"



# -------------------- USER OPTIONS --------------------


REBIN = 2                   # 0 (off) o 2 (consigliato) per S/N


SIGCLIP_ANCHORS = 3.0       # sigma-clip sugli anchors (0 = off) [placeholder se vorrai attivarlo]


DYNAMIC_MASK_Z  = None      # None = off; oppure numero (es. 3.0, 6.0) per mascherare outlier globali


ACCEPTED_FLAGS  = {"ALL"}   # {"ALL"} per accettare tutto; altrimenti es. {0,1}



# Soglie detection


DET_THRESH  = 3.0           # YES se S/N >= 3


MARG_THRESH = 2.0           # MARG se 2 <= S/N < 3



# CNT/HNT bands (standard)


BANDS = [


    dict(name="CNT_5p20_5p70",   core=(5.20, 5.70),   anchors=[(4.90, 5.10), (5.80, 6.00)],   pivot=5.45,


         label="CNT 5.3 µm"),


    dict(name="CNT_10p60_10p90", core=(10.60, 10.90), anchors=[(10.00, 10.40), (11.00, 11.40)], pivot=10.75,


         label="HNT/HNT+ 10.6–10.9 µm"),


    dict(name="CNT_12p86_13p40", core=(12.86, 13.40), anchors=[(12.20, 12.70), (13.50, 13.90)], pivot=13.00,


         label="HNT/HNT+ 12.9–13.4 µm"),


    dict(name="CNT_16p80_17p20", core=(16.80, 17.20), anchors=[(16.00, 16.50), (17.60, 18.00)], pivot=17.00,


         label="HNT/HNT+ ~17 µm (core)"),


    dict(name="CNT_16p50_17p50_wide", core=(16.50, 17.50), anchors=[(16.00, 16.50), (17.60, 18.00)], pivot=17.00,


         label="HNT/HNT+ ~17 µm (wide)"),


]

# === Wide-band extensions for Core vs Wide confirmation ===
BANDS += [
    dict(name="CNT_5p00_5p80",
         core=(5.00, 5.80),
         anchors=[(4.80, 5.00), (5.80, 6.00)],
         pivot=5.40,
         label="CNT ~5.3 μm (wide)"),

    dict(name="CNT_10p40_11p00",
         core=(10.40, 11.00),
         anchors=[(10.00, 10.40), (11.00, 11.30)],
         pivot=10.70,
         label="HNT/HNT+ ~10.7 μm (wide)"),

    dict(name="CNT_12p60_13p50",
         core=(12.60, 13.50),
         anchors=[(12.20, 12.60), (13.50, 13.90)],
         pivot=13.00,
         label="HNT/HNT+ ~12.9 μm (wide)"),
]

# SAFE variants per bande critiche (solo anchors cambiano; core identico)


SAFE_BANDS = [


    # 5.3 µm: evita coda H2O ~6 µm restringendo ancora alta


    dict(base="CNT_5p20_5p70",


         name="CNT_5p20_5p70_SAFE_water",


         anchors_safe=[(4.90, 5.10), (5.75, 5.85)],  # più vicino al core, evita 5.9–6.0


         note="water-safe"),


    # 10.6–10.9 µm: evita PAH 11.2–11.3 µm


    dict(base="CNT_10p60_10p90",


         name="CNT_10p60_10p90_SAFE_PAH",


         anchors_safe=[(10.00, 10.40), (10.95, 10.99)],


         note="PAH-safe"),


    # 12.86–13.40 µm: evita PAH 12.7 (e bump 13.04 resta mascherato via MASKS_GLOBAL)


    dict(base="CNT_12p86_13p40",


         name="CNT_12p86_13p40_SAFE_PAH127",


         anchors_safe=[(12.30, 12.55), (13.50, 13.90)],


         note="PAH12.7-safe"),


]



# Maschere globali (linee strette / features note da escludere sempre)


MASKS_GLOBAL = [


    (10.50, 10.52),  # [S IV]


    (12.35, 12.39),  # H I 12.37


    (12.80, 12.84),  # [Ne II]


    (13.02, 13.06),  # bump 13.04


    (17.02, 17.05),  # H2 S(1) 17.03 (più stretta qui)


    (17.35, 17.45),  # C60 ~17.4


]



# ---- OUTPUT OPTIONS ----


SAVE_CSV  = True


SAVE_PDF  = True


SAVE_JSON = True



# -------------------- CODE (no edits below) --------------------

import os
OBJECT_ID = os.environ.get("SHIM_OBJECT_ID", "object_name").replace("_", " ")


import re, math, json, numpy as np, pandas as pd


import matplotlib.pyplot as plt


from IPython.display import display


from matplotlib.backends.backend_pdf import PdfPages



plt.rcParams["figure.dpi"] = 120



CORE_COL = {"green": "#2ca02c", "orange": "#ff7f0e", "gray": "#9aa0a6"}


YES_COL  = "#2ca02c"; MARG_COL = "#ff7f0e"; NO_COL = "#9aa0a6"; SPEC_COL = "#1f77b4"



def verdict_from_sn(sn):


    if not (np.isfinite(sn)):


        return "NO (insufficient data)", "gray"


    if sn >= DET_THRESH:


        return "YES (detection)", "green"


    if sn >= MARG_THRESH:


        return "MARGINAL", "orange"


    return "NO", "gray"



def parse_text_table(txt):


    txt = txt.replace(",", " ")


    page_markers = {1,2,3,4}


    rows = []


    for line in txt.splitlines():


        line = line.strip()


        if not line or line.startswith("#"): continue


        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)


        if len(nums) == 1:


            x = float(nums[0])


            if abs(x-round(x)) < 1e-12 and int(round(x)) in page_markers:


                continue


        if len(nums) == 4:


            lam, f, s, flg = map(float, nums)


            rows.append((lam, f, s, int(round(flg))))


    if not rows:


        toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)


        vals = list(map(float, toks))


        cleaned = []


        for v in vals:


            if abs(v-round(v)) < 1e-12 and int(round(v)) in page_markers:


                continue


            cleaned.append(v)


        i=0


        while i+3 < len(cleaned):


            lam,f,s,flg = cleaned[i:i+4]


            rows.append((lam,f,s,int(round(flg))))


            i+=4


    if not rows:


        raise ValueError("No valid rows found.")


    arr=np.array(rows,float)


    lam, flux, sig = arr[:,0], arr[:,1], arr[:,2]


    flag = np.array([int(x) for x in arr[:,3]])


    ok = np.isfinite(lam)&np.isfinite(flux)&np.isfinite(sig)&(sig>0)


    return lam[ok],flux[ok],sig[ok],flag[ok]



def rebin_series(lam, flux, sig, factor):


    if not factor or factor<=1: return lam,flux,sig


    n=(lam.size//factor)*factor


    lam=lam[:n].reshape(-1,factor).mean(1)


    W=1.0/(sig[:n].reshape(-1,factor)**2)


    flux=(W*flux[:n].reshape(-1,factor)).sum(1)/W.sum(1)


    sig=1.0/np.sqrt(W.sum(1))


    return lam,flux,sig



def in_windows(x,windows):


    if not windows: return np.zeros_like(x,bool)


    m=np.zeros_like(x,bool)


    for a,b in windows: m|=(x>=a)&(x<=b)


    return m



# --- weighted least squares helpers ---


def _safe_wls(X,y,S):


    S=np.maximum(S,1e-20)


    w=1.0/(S**2)


    W12=np.sqrt(w)[:,None]


    Xw=W12*X; yw=(W12[:,0])*y


    beta,*_=np.linalg.lstsq(Xw,yw,rcond=None)


    XtWX=X.T@(w[:,None]*X)


    try: cov=np.linalg.inv(XtWX)


    except np.linalg.LinAlgError: cov=np.linalg.pinv(XtWX)


    return beta,cov



def wls_linear(L,F,S,pivot):


    x=(L-pivot)


    X=np.column_stack([np.ones_like(x),x])


    beta,cov=_safe_wls(X,F,S)


    a,b=beta; sa,sb=np.sqrt(np.maximum(np.diag(cov),0.0))


    return float(a),float(b),float(sa),float(sb)



def wls_poly2(L,F,S,pivot):


    x=(L-pivot)


    X=np.column_stack([np.ones_like(x),x,x**2])


    beta,cov=_safe_wls(X,F,S)


    a,b,c=beta; sa,sb,sc=np.sqrt(np.maximum(np.diag(cov),0.0))


    return (float(a),float(b),float(c)),(float(sa),float(sb),float(sc))



def fit_continuum(lam,flux,sig,anchors_mask,pivot,model="linear"):


    L,F,S=lam[anchors_mask],flux[anchors_mask],sig[anchors_mask]


    if L.size<2:


        W=1.0/(np.maximum(S,1e-20)**2)


        a=float(np.sum(W*F)/np.sum(W)); sa=float(1.0/np.sqrt(np.sum(W)))


        C=lambda x:a+0*(x-pivot)


        return dict(a=a,b=0,sa=sa,sb=0,c=np.nan,sc=np.nan,C=C,anchors_used=int(L.size),model="flat")


    if model=="poly2" and L.size>=3:


        (a,b,c),(sa,sb,sc)=wls_poly2(L,F,S,pivot)


        C=lambda x:a+b*(x-pivot)+c*(x-pivot)**2


        return dict(a=a,b=b,c=c,sa=sa,sb=sb,sc=sc,C=C,anchors_used=int(L.size),model="poly2")


    # default linear


    a,b,sa,sb=wls_linear(L,F,S,pivot)


    C=lambda x:a+b*(x-pivot)


    return dict(a=a,b=b,c=np.nan,sa=sa,sb=sb,sc=np.nan,C=C,anchors_used=int(L.size),model="linear")



def metrics_core(lam,flux,sig,C,core_mask):


    L,F,S=lam[core_mask],flux[core_mask],sig[core_mask]


    if L.size==0:


        return dict(N=0,SN=np.nan,EW=np.nan,TAU=np.nan),core_mask


    Cval=C(L); D=Cval-F


    SN=(D/S).sum()/np.sqrt(L.size)


    r=np.clip(D/np.maximum(Cval,1e-30),-0.8,0.8)


    EW=np.trapz(r,L) if L.size>=2 else 0.0


    tau=-np.log1p(-r); w=1.0/(S**2)


    TAU=(tau*w).sum()/w.sum()


    return dict(N=int(L.size),SN=float(SN),EW=float(EW),TAU=float(TAU)),core_mask



def delta_chi2_notch(lam,flux,sig,C,core_mask):


    w=1.0/(sig**2); T=core_mask.astype(float)


    Cfull=C(lam)


    num=np.sum(w*(flux-Cfull)*(-Cfull*T))


    den=np.sum(w*(Cfull*T)**2)


    s_hat=0.0 if den<=0 else float(num/den)


    chi2_null=float(np.sum(((flux-Cfull)/sig)**2))


    chi2_notch=float(np.sum(((flux-Cfull*(1.0-s_hat*T))/sig)**2))


    dchi2=chi2_null-chi2_notch


    zsig=math.sqrt(max(dchi2,0.0))


    return dchi2,zsig,s_hat



# ---------- RUN BASE QUICK-LOOK ----------


lam,flux,sig,flag=parse_text_table(DATA)


if ACCEPTED_FLAGS != {"ALL"}:


    ok=np.array([f in ACCEPTED_FLAGS for f in flag])


else:


    ok=np.ones_like(flag,bool)


lam,flux,sig=lam[ok],flux[ok],sig[ok]


lam,flux,sig=rebin_series(lam,flux,sig,REBIN)



results=[]; plots_for_pdf=[]



for band in BANDS:


    name=band["name"]; c0,c1=band["core"]; pivot=band["pivot"]; anchors=band["anchors"]


    mask_lines=in_windows(lam,MASKS_GLOBAL)


    anchors_mask=in_windows(lam,anchors)&(~mask_lines)


    core_mask=((lam>=c0)&(lam<=c1))&(~mask_lines)



    fit=fit_continuum(lam,flux,sig,anchors_mask,pivot,model="linear")


    C=fit["C"]



    M,core_used=metrics_core(lam,flux,sig,C,core_mask)


    dchi2,zsig,s_hat=delta_chi2_notch(lam,flux,sig,C,core_used)


    verdict_label,verdict_key=verdict_from_sn(M["SN"])



    results.append(dict(


        band=name,core=f"{c0:.2f}-{c1:.2f}",


        a=fit.get("a",np.nan),sa=fit.get("sa",np.nan),b=fit.get("b",np.nan),sb=fit.get("sb",np.nan),


        c=fit.get("c",np.nan),


        N_core=int(core_used.sum()),SN=M["SN"],EW=M["EW"],tau_bar=M["TAU"],


        DeltaChi2=dchi2,zsig=zsig,s_hat=s_hat,


        anchors_used=int(anchors_mask.sum()),


        verdict=verdict_label,verdict_key=verdict_key


    ))



    # --- Plot ---


    vcol=CORE_COL.get(verdict_key,"#9aa0a6")


    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,5),sharex=True,gridspec_kw={'height_ratios':[3,1]})


    ax1.errorbar(lam,flux,yerr=sig,fmt='o',ms=3,ecolor='#cccccc',mec=SPEC_COL,mfc=SPEC_COL,alpha=0.9)


    Lg=np.linspace(c0-0.6,c1+0.6,400); ax1.plot(Lg, C(Lg),lw=1.4,color="#3366cc")


    ax1.axvspan(c0,c1,color=vcol,alpha=0.22,lw=0)


    # linea verticale su YES/MARG


    if verdict_key=="green":


        ax1.axvline((c0+c1)/2,color="green",linestyle="--",lw=1.2)


    elif verdict_key=="orange":


        ax1.axvline((c0+c1)/2,color="orange",linestyle="--",lw=1.2)


    ax1.set_title(f"{OBJECT_ID} — {band.get('label', name)} {verdict_label}")


    ax1.set_ylabel("Flux (Jy)")



    R=(C(lam)-flux)/np.maximum(C(lam),1e-30)


    sel=(lam>=c0-0.4)&(lam<=c1+0.4)


    ax2.scatter(lam[sel],R[sel],s=12,color="#444444",alpha=0.9)


    ax2.axvspan(c0,c1,color=vcol,alpha=0.15,lw=0)


    ax2.axhline(0,lw=1,color="#666666")


    ax2.set_xlabel("Wavelength (μm)"); ax2.set_ylabel("(C−F)/C")


    plt.tight_layout(); plt.show()


    plots_for_pdf.append((fig,name))



df=pd.DataFrame(results)


print(f"\n{OBJECT_ID} — CNT quick-look")


display(df)



print("\n--- CNT quick-look interpretation (per band) ---")


for row in results:


    band=row["band"]; core=row["core"]


    SNv=row["SN"]; d2=row["DeltaChi2"]; EWv=row["EW"]; tauv=row["tau_bar"]


    verdict=row["verdict"]


    msg=f"[{band} core {core}]: verdict={verdict}; S/N={SNv:.2f}σ; Δχ²={d2:.2f} (~{np.sqrt(max(d2,0)):.2f}σ); EW={EWv:.3e} μm; mean τ={tauv:.3e}"


    if SNv>=DET_THRESH and np.sqrt(max(d2,0))>=DET_THRESH: msg+=" → Interpretation: robust absorption CNT/HNT."


    elif SNv>=MARG_THRESH: msg+=" → Interpretation: marginal feature."


    else: msg+=" → Interpretation: no CNT absorption."


    print(msg)



# Save base outputs


safe_obj=OBJECT_ID.replace(" ","_")


if SAVE_CSV: df.to_csv(f"{safe_obj}_quicklook.csv",index=False)


if SAVE_PDF and len(plots_for_pdf)>0:


    pdf_path=f"{safe_obj}_plots.pdf"


    with PdfPages(pdf_path) as pdf:


        for fig,name in plots_for_pdf: pdf.savefig(fig,bbox_inches='tight')


    print(f"[Saved PDF] {pdf_path}")


if SAVE_JSON:


    with open(f"{safe_obj}_report.json","w",encoding="utf-8") as f:


        json.dump(df.to_dict(orient="records"),f,ensure_ascii=False,indent=2)


    print(f"[Saved JSON] {safe_obj}_report.json")



# =====================================================================


# === NEW SECTION === STANDARD vs SAFE (5.3, 10.6–10.9, 12.9–13.4)


# =====================================================================


# Costruisci un indice delle bande standard


by_name = {b["name"]: b for b in BANDS}


std_results = {r["band"]: r for r in results}



safe_rows = []


safe_plots = []



for sb in SAFE_BANDS:


    base_name = sb["base"]


    if base_name not in by_name or base_name not in std_results:


        continue


    base = by_name[base_name]


    c0,c1 = base["core"]; pivot = base["pivot"]


    # maschere


    mask_lines = in_windows(lam, MASKS_GLOBAL)


    core_mask  = ((lam>=c0)&(lam<=c1)) & (~mask_lines)


    anchors_std = in_windows(lam, base["anchors"]) & (~mask_lines)


    anchors_safe= in_windows(lam, sb["anchors_safe"]) & (~mask_lines)



    # fit standard (linear)


    fit_std  = fit_continuum(lam,flux,sig,anchors_std,pivot,model="linear")


    C_std    = fit_std["C"]


    M_std,_  = metrics_core(lam,flux,sig,C_std,core_mask)


    d2_std,z_std,s_std = delta_chi2_notch(lam,flux,sig,C_std,core_mask)


    verdict_std,key_std= verdict_from_sn(M_std["SN"])



    # fit SAFE (linear)


    fit_safe = fit_continuum(lam,flux,sig,anchors_safe,pivot,model="linear")


    C_safe   = fit_safe["C"]


    M_safe,_ = metrics_core(lam,flux,sig,C_safe,core_mask)


    d2_safe,z_safe,s_safe = delta_chi2_notch(lam,flux,sig,C_safe,core_mask)


    verdict_safe,key_safe = verdict_from_sn(M_safe["SN"])



    # regola di promozione


    same_sign = np.sign(M_std["EW"]) == np.sign(M_safe["EW"])


    robust_any = ((M_std["SN"]>=DET_THRESH and z_std>=DET_THRESH) or


                  (M_safe["SN"]>=DET_THRESH and z_safe>=DET_THRESH))


    if same_sign and robust_any:


        verdict_pair = "YES (robust)"


        key_pair = "green"


    elif (M_std["SN"]>=MARG_THRESH or M_safe["SN"]>=MARG_THRESH):


        verdict_pair = "MARGINAL"


        key_pair = "orange"


    else:


        verdict_pair = "NO"


        key_pair = "gray"



    safe_rows.append(dict(


        band_base=base_name, core=f"{c0:.2f}-{c1:.2f}",


        std_SN=M_std["SN"], std_z=z_std, std_EW=M_std["EW"], std_tau=M_std["TAU"],


        std_anchors=int(anchors_std.sum()), std_verdict=verdict_std,


        safe_name=sb["name"], safe_mode=sb["note"],


        safe_SN=M_safe["SN"], safe_z=z_safe, safe_EW=M_safe["EW"], safe_tau=M_safe["TAU"],


        safe_anchors=int(anchors_safe.sum()), safe_verdict=verdict_safe,


        pair_verdict=verdict_pair, pair_key=key_pair


    ))



    # plot locale comparativo (standard vs SAFE sullo stesso pannello)


    vcol_std  = CORE_COL.get(key_std,"#9aa0a6")


    vcol_safe = CORE_COL.get(key_safe,"#9aa0a6")


    xm, xM = c0-0.6, c1+0.6


    sel_loc = (lam>=xm) & (lam<=xM)


    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,5),sharex=True,gridspec_kw={'height_ratios':[3,1]})


    ax1.errorbar(lam[sel_loc], flux[sel_loc], yerr=sig[sel_loc], fmt='o', ms=3,


                 ecolor='#cccccc', mec=SPEC_COL, mfc=SPEC_COL, alpha=0.9)


    Lg=np.linspace(xm,xM,500)


    ax1.plot(Lg, C_std(Lg),  lw=1.2, color="#3366cc", label="continuum (std)")


    ax1.plot(Lg, C_safe(Lg), lw=1.2, color="#444488", linestyle="--", label="continuum (SAFE)")


    ax1.axvspan(c0,c1,color=vcol_std, alpha=0.18, lw=0, label=f"core std ({verdict_std})")


    ax1.axvspan(c0,c1,color=vcol_safe,alpha=0.10, lw=0, label=f"core SAFE ({verdict_safe})")


    xc=0.5*(c0+c1)


    if key_std=="green": ax1.axvline(xc,color="green",linestyle="--",lw=1.0)


    elif key_std=="orange": ax1.axvline(xc,color="orange",linestyle="--",lw=1.0)


    ax1.set_title(f"{OBJECT_ID} — {base.get('label', base_name)}: standard vs SAFE ({sb['note']}) → {verdict_pair}")


    ax1.set_ylabel("Flux (Jy)"); ax1.legend(loc="best",frameon=True)



    R_std  = (C_std(lam)  - flux)/np.maximum(C_std(lam),1e-30)


    R_safe = (C_safe(lam) - flux)/np.maximum(C_safe(lam),1e-30)


    ax2.scatter(lam[sel_loc], R_std[sel_loc],  s=10, alpha=0.85, label="(Cstd−F)/Cstd",  color="#444444")


    ax2.scatter(lam[sel_loc], R_safe[sel_loc], s=10, alpha=0.60, label="(Csafe−F)/Csafe", color="#777777")


    ax2.axvspan(c0,c1,color=vcol_std, alpha=0.12, lw=0)


    ax2.axhline(0,lw=1,color="#666666")


    ax2.set_xlabel("Wavelength (μm)"); ax2.set_ylabel("Residual")


    ax2.legend(loc="best",frameon=True)


    plt.tight_layout(); plt.show()


    safe_plots.append(fig)



if safe_rows:


    safe_df = pd.DataFrame(safe_rows, columns=[


        "band_base","core",


        "std_SN","std_z","std_EW","std_tau","std_anchors","std_verdict",


        "safe_name","safe_mode","safe_SN","safe_z","safe_EW","safe_tau","safe_anchors","safe_verdict",


        "pair_verdict","pair_key"


    ])


    print("\n=== STANDARD vs SAFE summary (critical bands) ===")


    display(safe_df)


    if SAVE_CSV:


        path=f"{safe_obj}_standard_vs_safe.csv"; safe_df.to_csv(path,index=False); print(f"[Saved CSV] {path}")


else:


    print("\n=== STANDARD vs SAFE summary ===\n(no critical bands available in this run)")
# =====================================================================
# === Core vs Wide confirmation (all major bands: 5.2, 10.7, 12.9, 17 µm)
# =====================================================================
# ============================================================
# Core vs Wide cross-confirmation — FIX robusto (tutte le bande)
# (usa sia `results` list che `df` DataFrame se disponibili)
# ============================================================
import numpy as np
import pandas as pd
from IPython.display import display

# Definizione gruppi core/wide
CORE_WIDE_GROUPS = [
    dict(label="CNT ~5.3 µm",       core_band="CNT_5p20_5p70",     wide_band="CNT_5p00_5p80"),
    dict(label="HNT/HNT+ ~10.7 µm", core_band="CNT_10p60_10p90",   wide_band="CNT_10p40_11p00"),
    dict(label="HNT/HNT+ ~12.9 µm", core_band="CNT_12p86_13p40",   wide_band="CNT_12p60_13p50"),
    dict(label="HNT/HNT+ ~17 µm",   core_band="CNT_16p80_17p20",   wide_band="CNT_16p50_17p50_wide"),
]

# 1) Costruisci un mapper nome_banda -> riga (dict) da qualsiasi sorgente esista
band_rows = {}

# a) Se esiste la lista 'results'
if 'results' in globals() and isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
    for r in results:
        band_rows[str(r.get('band',''))] = r.copy()

# b) Se esiste il DataFrame 'df' (quick-look)
if 'df' in globals() and isinstance(df, pd.DataFrame) and not df.empty:
    for _, r in df.iterrows():
        name = str(r.get('band', ''))
        if name and name not in band_rows:
            band_rows[name] = {k: r[k] for k in r.index if k in r}

def _pull_metric(rowdict, key, default=np.nan):
    try:
        v = rowdict.get(key, default)
        return float(v) if v is not None else default
    except Exception:
        return default

rows = []
for g in CORE_WIDE_GROUPS:
    label = g["label"]
    core_name, wide_name = g["core_band"], g["wide_band"]

    c = band_rows.get(core_name, None)
    w = band_rows.get(wide_name, None)

    if c is None or w is None:
        # riga placeholder per capire subito se mancano nomi
        rows.append(dict(Band=label, Core_SNR=np.nan, Wide_SNR=np.nan,
                         Core_z=np.nan, Wide_z=np.nan,
                         ΔSN=np.nan, Δz=np.nan,
                         Verdict="n/a", Comment=f"missing: {('core ' if c is None else '')}{('wide' if w is None else '')}"))
        continue

    cSN = _pull_metric(c, "SN");  wSN = _pull_metric(w, "SN")
    cz  = _pull_metric(c, "zsig"); wz  = _pull_metric(w, "zsig")
    cEW = _pull_metric(c, "EW");  wEW = _pull_metric(w, "EW")
    cTAU=_pull_metric(c, "tau_bar"); wTAU=_pull_metric(w, "tau_bar")

    dSN = (cSN - wSN) if (np.isfinite(cSN) and np.isfinite(wSN)) else np.nan
    dz  = (cz  - wz ) if (np.isfinite(cz)  and np.isfinite(wz )) else np.nan

    same_sign = (np.sign(cEW) == np.sign(wEW)) if (np.isfinite(cEW) and np.isfinite(wEW)) else False

    # criteri: confermato se stesso segno e differenze ragionevoli
    ok       = (same_sign and (np.isnan(dSN) or abs(dSN) < 5) and (np.isnan(dz) or abs(dz) < 5))
    marginal = (same_sign and ((np.isnan(dSN) or abs(dSN) < 8) or (np.isnan(dz) or abs(dz) < 8)))

    verdict = "IT FITS" if ok else ("MARGINAL" if marginal else "INCONSISTENT")

    rows.append(dict(
        Band=label,
        Core_SNR=f"{cSN:.2f}" if np.isfinite(cSN) else "nan",
        Wide_SNR=f"{wSN:.2f}" if np.isfinite(wSN) else "nan",
        Core_z  =f"{cz:.2f}"  if np.isfinite(cz)  else "nan",
        Wide_z  =f"{wz:.2f}"  if np.isfinite(wz)  else "nan",
        ΔSN     =f"{dSN:+.2f}" if np.isfinite(dSN) else "nan",
        Δz      =f"{dz:+.2f}"  if np.isfinite(dz)  else "nan",
        Verdict=verdict,
        Comment="same sign" if same_sign else "sign flip"
    ))

# 2) DataFrame finale + evidenziazione
df_corewide = pd.DataFrame(rows, columns=["Band","Core_SNR","Wide_SNR","Core_z","Wide_z","ΔSN","Δz","Verdict","Comment"])

print("\n=== Core vs Wide cross-confirmation across all bands (FIX) ===")
def _color_verdict(v):
    if v == "IT FITS": return "background-color:#b6f2b6"
    if v == "MARGINAL":  return "background-color:#fce8b2"
    if v == "INCONSISTENT": return "background-color:#f4b6b6"
    return ""
display(df_corewide.style.map(_color_verdict, subset=["Verdict"]))

# =====================================================================
# === Core vs Wide: grafici + salvataggio multipagina in PDF
#     (da usare dopo che esiste df_corewide)
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ------------- Opzioni -------------
SAVE_PDF = True
OUT_BASENAME = f"{OBJECT_ID}_corewide_summary" if 'OBJECT_ID' in globals() else "corewide_summary"
PDF_PATH = f"{OUT_BASENAME}.pdf"
CSV_PATH = f"{OUT_BASENAME}.csv"
TEX_PATH = f"{OUT_BASENAME}.tex"

# ------------- Utility plotting -------------
def _bar_compare(core_val, wide_val, title, ylabel):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    xpos = np.arange(2)
    vals = [core_val, wide_val]
    ax.bar(xpos, vals, width=0.55)
    ax.set_xticks(xpos, ["core", "wide"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', lw=0.4, alpha=0.6)
    return fig

def _table_page(df, title="Core vs Wide summary"):
    # rende la tabella in una pagina matplotlib
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=12, pad=8)
    # limita la larghezza testi per la colonna Note/Comment
    df_disp = df.copy()
    if "Comment" in df_disp.columns:
        df_disp["Comment"] = df_disp["Comment"].astype(str)
    tbl = ax.table(cellText=df_disp.values,
                   colLabels=df_disp.columns,
                   cellLoc='center',
                   loc='upper left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1.0, 1.25)
    return fig

# ------------- Salva versione tabellare su disco -------------
try:
    df_corewide.to_csv(CSV_PATH, index=False)
    with open(TEX_PATH, "w") as f:
        f.write(df_corewide.to_latex(index=False, escape=True))
    print(f"[OK] Salvati tabella CSV → {CSV_PATH} e LaTeX → {TEX_PATH}")
except Exception as e:
    print("Salvataggio tabella (CSV/LaTeX) saltato:", e)

# ------------- Generazione grafici e PDF -------------
# --- Generazione grafici e PDF ---
figs = []

# Se la tabella è vuota, non fare niente
if df_corewide.empty:
    print("[WARN] df_corewide is empty — skipping plots.")
else:
    # forza numerici prima di plottare
    for col in ["Core_SNR", "Wide_SNR", "Core_z", "Wide_z", "ΔSN", "Δz"]:
        if col in df_corewide.columns:
            df_corewide[col] = pd.to_numeric(df_corewide[col], errors="coerce")

    # pagina iniziale: tabella riassuntiva
    try:
        figs.append(_table_page(df_corewide, title=f"{OBJECT_ID} — Core vs Wide summary" if 'OBJECT_ID' in globals() else "Core vs Wide summary"))
    except Exception as e:
        print("Pagina tabella saltata:", e)

    # loop per banda: crea grafici S/N e z
    for _, row in df_corewide.iterrows():
        label = str(row["Band"])

        def _num(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        core_sn = _num(row.get("Core_SNR", np.nan))
        wide_sn = _num(row.get("Wide_SNR", np.nan))
        core_z  = _num(row.get("Core_z", np.nan))
        wide_z  = _num(row.get("Wide_z", np.nan))

        if np.isfinite(core_sn) and np.isfinite(wide_sn):
            figs.append(_bar_compare(core_sn, wide_sn, title=f"{label} — S/N (core vs wide)", ylabel="S/N"))
        if np.isfinite(core_z) and np.isfinite(wide_z):
            figs.append(_bar_compare(core_z, wide_z, title=f"{label} — z = √Δχ² (core vs wide)", ylabel="z"))

    # mostra i grafici in Colab
    for fig in figs:
        plt.show(fig)

    # salva PDF multipagina
    if SAVE_PDF and len(figs) > 0:
        try:
            with PdfPages(PDF_PATH) as pdf:
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
            print(f"[OK] PDF salvato → {PDF_PATH}")
        except Exception as e:
            print("Salvataggio PDF fallito:", e)
# === EXTRA VERIFICA AUTO su bande "YES": rebin, core-edge, jackknife, poly2


#      (Incolla questa sezione dopo la "Core vs Wide confirmation")


# =====================================================================



def wls_poly2(L, F, S, pivot):


    # Fit quadratico pesato: a + b (x) + c (x^2), con x = (L - pivot)


    S = np.maximum(S, 1e-20)


    w = 1.0 / (S**2)


    x = (L - pivot)


    X = np.column_stack([np.ones_like(x), x, x**2])


    W12 = np.sqrt(w)[:, None]


    Xw = W12 * X


    yw = W12[:, 0] * F


    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)


    XtWX = X.T @ (w[:, None] * X)


    try:


        cov = np.linalg.inv(XtWX)


    except np.linalg.LinAlgError:


        cov = np.linalg.pinv(XtWX)


    a, b, c = beta


    sa, sb, sc = np.sqrt(np.maximum(np.diag(cov), 0.0))


    C = lambda xgrid: a + b*(xgrid - pivot) + c*(xgrid - pivot)**2


    return dict(a=float(a), b=float(b), c=float(c),


                sa=float(sa), sb=float(sb), sc=float(sc),


                C=C, model="poly2")



def run_band_eval(lam, flux, sig, band, masks_global, model="linear",


                  rebin_factor=None, core_shift=(0.0, 0.0)):


    # Opzionale rebin locale


    if rebin_factor and rebin_factor > 1:


        Lr, Fr, Sr = rebin_series(lam, flux, sig, rebin_factor)


    else:


        Lr, Fr, Sr = lam, flux, sig



    c0, c1 = band["core"]


    dl, dr = core_shift


    c0s, c1s = c0 + dl, c1 + dr



    # Maschere


    m_lines   = in_windows(Lr, masks_global)


    m_anchors = in_windows(Lr, band["anchors"]) & (~m_lines)


    m_core    = ((Lr >= c0s) & (Lr <= c1s)) & (~m_lines)



    # Fit continuo


    if model == "linear":


        fit = fit_continuum(Lr, Fr, Sr, m_anchors, band["pivot"], model="linear")


    elif model == "poly2":


        La, Fa, Sa = Lr[m_anchors], Fr[m_anchors], Sr[m_anchors]


        if La.size < 3:  # fall back a lineare se anchor insufficienti


            fit = fit_continuum(Lr, Fr, Sr, m_anchors, band["pivot"], model="linear")


        else:


            fit = wls_poly2(La, Fa, Sa, band["pivot"])


    else:


        raise ValueError("model must be 'linear' or 'poly2'")



    C = fit["C"]



    # Metriche sul core


    M, core_used = metrics_core(Lr, Fr, Sr, C, m_core)


    dchi2, zsig, s_hat = delta_chi2_notch(Lr, Fr, Sr, C, core_used)



    return dict(


        SN=M["SN"], zsig=zsig, EW=M["EW"], tau=M["TAU"],


        DeltaChi2=dchi2, s_hat=s_hat,


        anchors_used=int(m_anchors.sum()),


        N_core=int(m_core.sum()),


        c0=c0s, c1=c1s, model=model, rebin=rebin_factor


    ), (Lr, Fr, Sr, m_core, C)



# Trova le righe verdi (YES) dal quick-look


yes_rows = [r for r in results if r["verdict"].startswith("YES")]



if not yes_rows:


    print("\n[Extra verifica] Nessuna banda 'YES' da verificare.")


else:


    print("\n================ EXTRA VERIFY ON GREEN SIGNAL YES ================\n")


    extra_tables = []



    # Mappa rapida band_name -> definizione banda


    band_defs = {b["name"]: b for b in BANDS}



    # Parametri del test


    edge_steps = [(-0.05, -0.05), (0.0, -0.05), (-0.05, 0.0),


                  (0.0, 0.0),  (0.05, 0.0),  (0.0, 0.05), (0.05, 0.05)]


    rebin_set = [1, 2]



    for row in yes_rows:


        bname = row["band"]


        bdef  = band_defs.get(bname)


        if bdef is None:


            continue



        print(f"→ {OBJECT_ID} | {bname} | core {row['core']} — verifica extra")



        # 1) REBIN test (R=1,2) con modello lineare


        rebin_results = []


        for R in rebin_set:


            out, _ = run_band_eval(lam, flux, sig, bdef, MASKS_GLOBAL, model="linear",


                                   rebin_factor=R, core_shift=(0.0, 0.0))


            rebin_results.append((R, out))



        # 2) Core-edge scan ±0.05 μm (usa REBIN=REBIN corrente dell'utente)


        edge_results = []


        for (dl, dr) in edge_steps:


            out, _ = run_band_eval(lam, flux, sig, bdef, MASKS_GLOBAL, model="linear",


                                   rebin_factor=REBIN, core_shift=(dl, dr))


            edge_results.append(((dl, dr), out))



        # 3) Jackknife sul core (REBIN=REBIN, lineare)


        base_eval, (Lr, Fr, Sr, m_core, Cbase) = run_band_eval(


            lam, flux, sig, bdef, MASKS_GLOBAL, model="linear",


            rebin_factor=REBIN, core_shift=(0.0, 0.0)


        )


        core_idx = np.where(m_core)[0]


        SN_jk = []


        z_jk  = []


        if core_idx.size >= 3:


            # Stesso continuo; ricalcola SN/Δχ² togliendo un punto alla volta


            Cvals = Cbase(Lr)


            for drop in core_idx:


                mc = m_core.copy()


                mc[drop] = False


                # SN jackknife


                Lc = Lr[mc]; Fc = Fr[mc]; Sc = Sr[mc]


                if Lc.size == 0:


                    continue


                Cv = Cbase(Lc); D = Cv - Fc


                SN = (D/Sc).sum() / np.sqrt(Lc.size)


                # z jackknife (Δχ²)


                T = mc.astype(float)


                w = 1.0 / (Sr**2)


                num = np.sum(w * (Fr - Cvals) * (-Cvals*T))


                den = np.sum(w * (Cvals*T)**2)


                s_hat_j = 0.0 if den <= 0 else float(num/den)


                chi2_null = float(np.sum(((Fr - Cvals)/Sr)**2))


                chi2_notch = float(np.sum(((Fr - Cvals*(1.0 - s_hat_j*T))/Sr)**2))


                dchi2_j = chi2_null - chi2_notch


                zsig_j = math.sqrt(max(dchi2_j, 0.0))


                SN_jk.append(SN); z_jk.append(zsig_j)



        # 4) Confronto poly2 (REBIN=REBIN, stesso core)


        poly2_eval, _ = run_band_eval(


            lam, flux, sig, bdef, MASKS_GLOBAL, model="poly2",


            rebin_factor=REBIN, core_shift=(0.0, 0.0)


        )



        # ---- Riassunti numerici ----


        # REBIN summary


        sn_r = {R: ev["SN"] for (R, ev) in rebin_results}


        z_r  = {R: ev["zsig"] for (R, ev) in rebin_results}


        rebin_ok = (


            (sn_r.get(1, -np.inf) >= MARG_THRESH and z_r.get(1, -np.inf) >= MARG_THRESH) and


            (sn_r.get(2, -np.inf) >= MARG_THRESH and z_r.get(2, -np.inf) >= MARG_THRESH)


        )



        # Edge summary


        sn_edge = [ev["SN"] for _, ev in edge_results]


        z_edge  = [ev["zsig"] for _, ev in edge_results]


        edge_ok = (min(sn_edge) >= MARG_THRESH) and (min(z_edge) >= MARG_THRESH)



        # Jackknife summary


        if len(SN_jk) >= 3:


            SN_jk_mean = float(np.mean(SN_jk)); SN_jk_std = float(np.std(SN_jk))


            z_jk_mean  = float(np.mean(z_jk));  z_jk_std  = float(np.std(z_jk))


            jk_ok = (SN_jk_mean >= MARG_THRESH) and (z_jk_mean >= MARG_THRESH)


        else:


            SN_jk_mean = np.nan; SN_jk_std = np.nan


            z_jk_mean  = np.nan; z_jk_std  = np.nan


            jk_ok = False if core_idx.size>0 else True  # se pochissimi punti, non penalizzare



        # Poly2 consistency: deve avere stesso segno (assorbimento) e ≥ MARG


        poly2_ok = (poly2_eval["SN"] >= MARG_THRESH) and (poly2_eval["zsig"] >= MARG_THRESH)



        # Decisione suggerita (NON cambia il tuo verdetto, è solo un flag)


        confirm_strong = rebin_ok and edge_ok and jk_ok


        status = "CONFIRM_STRONG" if confirm_strong else "REVIEW"



        # Stampa breve


        print(f"   REBIN test:  R1 S/N={sn_r.get(1, np.nan):.2f}, √Δχ²={z_r.get(1, np.nan):.2f}  |  "


              f"R2 S/N={sn_r.get(2, np.nan):.2f}, √Δχ²={z_r.get(2, np.nan):.2f}  → "


              f"{'OK' if rebin_ok else 'check'}")


        print(f"   Core-edge (±0.05): min S/N={min(sn_edge):.2f}, min √Δχ²={min(z_edge):.2f}  → "


              f"{'OK' if edge_ok else 'check'}")


        if np.isfinite(SN_jk_mean):


            print(f"   Jackknife:  S/N={SN_jk_mean:.2f}±{SN_jk_std:.2f}, √Δχ²={z_jk_mean:.2f}±{z_jk_std:.2f}  → "


                  f"{'OK' if jk_ok else 'check'}")


        else:


            print("   Jackknife:  (non applicabile: core con <3 punti)")


        print(f"   Poly2 vs Linear:  S/N_poly2={poly2_eval['SN']:.2f}, √Δχ²_poly2={poly2_eval['zsig']:.2f}  → "


              f"{'OK' if poly2_ok else 'check'}")


        print(f"   ⇒ Extra-verifica status: {status}\n")



        extra_tables.append(dict(


            band=bname, core=row["core"],


            SN_R1=sn_r.get(1, np.nan), z_R1=z_r.get(1, np.nan),


            SN_R2=sn_r.get(2, np.nan), z_R2=z_r.get(2, np.nan),


            SN_edge_min=min(sn_edge), z_edge_min=min(z_edge),


            SN_jk_mean=SN_jk_mean, SN_jk_std=SN_jk_std,


            z_jk_mean=z_jk_mean, z_jk_std=z_jk_std,


            SN_poly2=poly2_eval["SN"], z_poly2=poly2_eval["zsig"],


            anchors_used_base=base_eval["anchors_used"],


            N_core_base=base_eval["N_core"],


            status=status


        ))



    # Tabella riassuntiva di tutte le bande YES


    extra_df = pd.DataFrame(extra_tables, columns=[


        "band","core",


        "SN_R1","z_R1","SN_R2","z_R2",


        "SN_edge_min","z_edge_min",


        "SN_jk_mean","SN_jk_std","z_jk_mean","z_jk_std",


        "SN_poly2","z_poly2",


        "anchors_used_base","N_core_base",


        "status"


    ])


    print("\n=== EXTRA VERIFY SUMMARY (solo bande YES) ===")


    display(extra_df)



    if SAVE_CSV:


        path_extra = f"{safe_obj}_extra_verify.csv"


        extra_df.to_csv(path_extra, index=False)


        print(f"[Saved CSV] {path_extra}")
# =========================== #
#  EXTRA: LSF check + Injection–Recovery + Plots (final pro version)
#  Colab-ready, same logic, adds optional LaTeX/CSV/PDF and visual summaries
# ===========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---- USER OPTIONS ----
INSTR_R = 300
INJ_DEPTHS = [0.005, 0.01, 0.02]
N_BOOT = 200
R_TESTS = [1, 2]
SAVE_PDF = True
SAVE_LATEX = True
OUT_BASE = OBJECT_ID.replace(" ", "_") if 'OBJECT_ID' in globals() else "target"

# ---- Utility identical to your base code ----
def _fit_band_continuum(band, lam, flux, sig, masks_extra=None):
    anchors = band["anchors"]; pivot = band["pivot"]
    mask_lines = in_windows(lam, MASKS_GLOBAL)
    anchors_mask = in_windows(lam, anchors) & (~mask_lines)
    if masks_extra is not None:
        for w in masks_extra:
            anchors_mask &= ~in_windows(lam, [w])
    return fit_continuum(lam, flux, sig, anchors_mask, pivot, model="linear")

def _measure_band_once(band, lam, flux, sig, masks_extra=None):
    c0, c1 = band["core"]
    fit = _fit_band_continuum(band, lam, flux, sig, masks_extra=masks_extra)
    C = fit["C"]
    mask_lines = in_windows(lam, MASKS_GLOBAL)
    core_mask = ((lam >= c0) & (lam <= c1)) & (~mask_lines)
    M, used = metrics_core(lam, flux, sig, C, core_mask)
    dchi2, zsig, s_hat = delta_chi2_notch(lam, flux, sig, C, used)
    return dict(SN=M["SN"], z=zsig, EW=M["EW"], TAU=M["TAU"],
                N_core=int(core_mask.sum()), anchors_used=fit.get("anchors_used", np.nan),
                s_hat=s_hat)

def _inject_notch_top_hat(band, lam, flux, depth):
    c0, c1 = band["core"]
    core_mask = (lam >= c0) & (lam <= c1)
    Fmod = flux.copy()
    Fmod[core_mask] = flux[core_mask] * (1.0 - depth)
    return Fmod

def _rebin_if_needed(lam, flux, sig, R):
    if R and R > 1:
        return rebin_series(lam, flux, sig, R)
    return lam, flux, sig

# ---- Core process ----
lsf_rows, inj_rows = [], []
inj_distributions = {}

for band in BANDS:
    name = band["name"]
    c0, c1 = band["core"]
    lam_c = 0.5 * (c0 + c1)
    core_width = (c1 - c0)
    fwhm_lsf = lam_c / INSTR_R if INSTR_R and INSTR_R > 0 else np.nan
    resolved = (core_width >= fwhm_lsf) if np.isfinite(fwhm_lsf) else False

    base = _measure_band_once(band, lam, flux, sig)
    lsf_rows.append(dict(
        band=name, core=f"{c0:.2f}-{c1:.2f}",
        lambda_c=lam_c, core_width=core_width,
        FWHM_LSF=fwhm_lsf, width_to_LSF=(core_width/fwhm_lsf if np.isfinite(fwhm_lsf) and fwhm_lsf>0 else np.nan),
        resolved=bool(resolved),
        SN_base=base["SN"], z_base=base["z"],
        N_core=base["N_core"], anchors_used=base["anchors_used"]
    ))

    inj_distributions[name] = {"SN": [], "z": []}

    # Injection–Recovery
    for depth in INJ_DEPTHS:
        F_inj_clean = _inject_notch_top_hat(band, lam, flux, depth)
        rec_SN, rec_Z = {R: [] for R in R_TESTS}, {R: [] for R in R_TESTS}
        for _ in range(N_BOOT):
            noise = np.random.normal(0.0, sig)
            F_noisy = F_inj_clean + noise
            for R in R_TESTS:
                Lb, Fb, Sb = _rebin_if_needed(lam, F_noisy, sig, R)
                met = _measure_band_once(band, Lb, Fb, Sb)
                rec_SN[R].append(met["SN"]); rec_Z[R].append(met["z"])
        for R in R_TESTS:
            SNa, Za = np.array(rec_SN[R], float), np.array(rec_Z[R], float)
            ok = (SNa >= DET_THRESH) & (Za >= DET_THRESH)
            inj_rows.append(dict(
                band=name, core=f"{c0:.2f}-{c1:.2f}",
                depth=depth, rebin=R,
                rec_rate=float(ok.mean()),
                SN_med=float(np.median(SNa)), SN_iqr=float(np.percentile(SNa,75)-np.percentile(SNa,25)),
                z_med=float(np.median(Za)),  z_iqr=float(np.percentile(Za,75)-np.percentile(Za,25)),
                N_boot=N_BOOT
            ))
            inj_distributions[name]["SN"].extend(SNa)
            inj_distributions[name]["z"].extend(Za)

# ---- Output tables ----
lsf_df = pd.DataFrame(lsf_rows)
inj_df = pd.DataFrame(inj_rows)
display(lsf_df); display(inj_df)

lsf_csv = f"{OUT_BASE}_LSF_check.csv"
inj_csv = f"{OUT_BASE}_injection_recovery.csv"
lsf_tex = f"{OUT_BASE}_LSF_check.tex"
inj_tex = f"{OUT_BASE}_injection_recovery.tex"
pdf_path = f"{OUT_BASE}_LSF_IR_summary.pdf"

lsf_df.to_csv(lsf_csv, index=False)
inj_df.to_csv(inj_csv, index=False)
print(f"[OK] CSV saved → {lsf_csv}, {inj_csv}")

if SAVE_LATEX:
    with open(lsf_tex, "w") as f: f.write(lsf_df.to_latex(index=False))
    with open(inj_tex, "w") as f: f.write(inj_df.to_latex(index=False))
    print(f"[OK] LaTeX saved → {lsf_tex}, {inj_tex}")

# ---- PDF summary (tables + histograms) ----
if SAVE_PDF:
    figs = []

    # Page 1: LSF table
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.axis('off')
    ax1.set_title(f"{OUT_BASE} — LSF Check", loc='left')
    ax1.table(cellText=lsf_df.values, colLabels=lsf_df.columns, cellLoc='center', loc='upper left')
    figs.append(fig1)

    # Page 2: Injection–Recovery table
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.axis('off')
    ax2.set_title(f"{OUT_BASE} — Injection–Recovery Summary", loc='left')
    ax2.table(cellText=inj_df.values, colLabels=inj_df.columns, cellLoc='center', loc='upper left')
    figs.append(fig2)

    # Pages 3+: Histograms (S/N and z distributions)
    for band_name, data in inj_distributions.items():
        sn = np.array(data["SN"], float)
        z  = np.array(data["z"], float)

        # Histogram S/N
        fig_sn, ax_sn = plt.subplots(figsize=(6,4))
        ax_sn.hist(sn, bins=25, color='steelblue', alpha=0.75)
        ax_sn.set_title(f"{band_name} — Injection–Recovery: S/N distribution")
        ax_sn.set_xlabel("Recovered S/N per bootstrap")
        ax_sn.set_ylabel("Count")
        ax_sn.text(0.02, 0.95,
                   "Useful to assess noise stability and bias.\n"
                   "Uniform, narrow distributions = robust recovery.",
                   transform=ax_sn.transAxes, fontsize=8, va='top')
        figs.append(fig_sn)

        # Histogram z (√Δχ²)
        fig_z, ax_z = plt.subplots(figsize=(6,4))
        ax_z.hist(z, bins=25, color='orange', alpha=0.75)
        ax_z.set_title(f"{band_name} — Injection–Recovery: √Δχ² distribution")
        ax_z.set_xlabel("Recovered √Δχ² per bootstrap")
        ax_z.set_ylabel("Count")
        ax_z.text(0.02, 0.95,
                   "Useful to verify consistency of detection metric.\n"
                   "Sharp peak = consistent retrieval; wide = unstable.",
                   transform=ax_z.transAxes, fontsize=8, va='top')
        figs.append(fig_z)

    with PdfPages(pdf_path) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches='tight')
    print(f"[OK] PDF saved → {pdf_path}")

# ---- Referee summary ----
for _, r in lsf_df.iterrows():
    tag = "RESOLVED" if r["resolved"] else "UNRESOLVED?"
    print(f"[{r['band']}] core {r['core']} @λc≈{r['lambda_c']:.2f} µm | width={r['core_width']:.3f} µm, "
          f"LSF≈{r['FWHM_LSF']:.3f} µm → {tag}; base: S/N={r['SN_base']:.2f}, √Δχ²={r['z_base']:.2f}")
print("\nNote: a ‘green’ detection becomes robust if:")
print(" • width/LSF ≳ 1 (not <<1)")
print(" • rec_rate ≥ 0.8 for similar depth in ≥1 rebin test")
# ============================
# EXTRA: PAH/SIL masks + Broad-Deficit checks (11.6 µm, ~17 µm)
#  — Identico nelle misure/verdetti; aggiunge salvataggi CSV/TeX/PDF e note sui grafici.
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Saving options ----------
SAVE_CSV  = True
SAVE_TEX  = True
SAVE_PDF  = True
OUT_BASE  = (OBJECT_ID.replace(" ", "_") if 'OBJECT_ID' in globals() else "target") + "_broaddef"

# ---------- 1) PAH & Silicate masks (come prima) ----------
PAH_WINDOWS = [
    (3.25, 3.35), (6.17, 6.27), (7.55, 7.95), (8.55, 8.70),
    (11.18, 11.36), (12.62, 12.78), (16.35, 16.50), (17.30, 17.48),
]
SIL_WINDOWS = [
    (9.0, 10.2), (10.2, 10.4), (17.8, 18.6),
]

# ---------- 2) Broad-deficit windows (come prima) ----------
BROAD_DEFICITS = [
    dict(
        name="BD_11p50_11p70",
        core=(11.50, 11.70),
        anchors=[(10.90, 11.10), (12.10, 12.30)],
        pivot=11.60,
        label="Broad-deficit ~11.6 µm"
    ),
    dict(
        name="BD_16p60_17p60",
        core=(16.60, 17.60),
        anchors=[(16.00, 16.40), (17.80, 18.00)],
        pivot=17.10,
        label="Broad-deficit ~17 µm"
    ),
]

# ---------- 3) Utility locali (identiche nella logica) ----------
def _local_mask(lam, extra_windows):
    return in_windows(lam, extra_windows) if extra_windows else np.zeros_like(lam, bool)

def measure_window(lam, flux, sig, win_def, extra_masks=None, model="linear", rebin=None):
    if rebin and rebin > 1:
        L, F, S = rebin_series(lam, flux, sig, rebin)
    else:
        L, F, S = lam, flux, sig

    c0, c1 = win_def["core"]
    pivot   = win_def["pivot"]
    anchors = win_def["anchors"]

    m_lines  = in_windows(L, MASKS_GLOBAL)
    m_pah    = _local_mask(L, PAH_WINDOWS)
    m_sil    = _local_mask(L, SIL_WINDOWS)
    m_extras = _local_mask(L, extra_masks) if extra_masks else np.zeros_like(L, bool)

    m_anchors = in_windows(L, anchors) & ~(m_lines | m_pah | m_sil | m_extras)
    m_core    = ((L >= c0) & (L <= c1)) & ~(m_lines | m_pah | m_sil | m_extras)

    fit = fit_continuum(L, F, S, m_anchors, pivot, model=model)
    C   = fit["C"]

    M, used = metrics_core(L, F, S, C, m_core)
    d2, z, s_hat = delta_chi2_notch(L, F, S, C, used)

    if not np.isfinite(M["SN"]):
        verdict, key = "NO (insufficient data)", "gray"
    elif M["SN"] >= DET_THRESH and z >= DET_THRESH:
        verdict, key = "YES (broad-deficit)", "green"
    elif M["SN"] >= MARG_THRESH or z >= MARG_THRESH:
        verdict, key = "MARGINAL", "orange"
    else:
        verdict, key = "NO", "gray"

    out = dict(
        band=win_def["name"],
        label=win_def.get("label", win_def["name"]),
        core=f"{c0:.2f}-{c1:.2f}",
        anchors_used=int(m_anchors.sum()),
        N_core=int(m_core.sum()),
        SN=float(M["SN"]),
        EW=float(M["EW"]),
        tau_bar=float(M["TAU"]),
        DeltaChi2=float(d2),
        zsig=float(z),
        s_hat=float(s_hat),
        verdict=verdict,
        verdict_key=key,
        model=model,
        rebin=(rebin or 1)
    )
    return out, (L, F, S, m_core, m_anchors, C)

def plot_window(L, F, S, win_def, masks_tuple, C, verdict_key, title_note=""):
    c0, c1 = win_def["core"]
    (m_core, m_anchors) = masks_tuple
    vcol = {"green":"#2ca02c", "orange":"#ff7f0e", "gray":"#9aa0a6"}.get(verdict_key, "#9aa0a6")

    xm, xM = c0 - 0.8, c1 + 0.8
    sel = (L >= xm) & (L <= xM)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax1.errorbar(L[sel], F[sel], yerr=S[sel], fmt='o', ms=3, ecolor='#cccccc', mec="#1f77b4", mfc="#1f77b4", alpha=0.9)
    Lg = np.linspace(xm, xM, 500)
    ax1.plot(Lg, C(Lg), lw=1.4, color="#3366cc")
    ax1.axvspan(c0, c1, color=vcol, alpha=0.20, lw=0, label="core")
    for (a,b) in win_def["anchors"]:
        ax1.axvspan(a, b, color="#777777", alpha=0.07, lw=0)
    ax1.set_title(f"{OBJECT_ID} — {win_def.get('label', win_def['name'])} {title_note}")
    ax1.set_ylabel("Flux (Jy)")

    R = (C(L) - F)/np.maximum(C(L), 1e-30)
    ax2.scatter(L[sel], R[sel], s=12, color="#444444", alpha=0.9)
    ax2.axvspan(c0, c1, color=vcol, alpha=0.15, lw=0)
    ax2.axhline(0, lw=1, color="#666666")
    ax2.set_xlabel("Wavelength (μm)")
    ax2.set_ylabel("(C−F)/C")

    # Nota esplicativa (in inglese) per referee
    ax2.text(0.01, 1.02,
             "Diagnostic: broad-deficit after PAH/silicate masking.\n"
             "If this is null while narrow bands are significant, the narrow signal is unlikely to be a wing.",
             transform=ax2.transAxes, fontsize=8, va='bottom')

    plt.tight_layout()
    plt.show()
    return fig  # <- restituisco la figura per salvataggio PDF

# ---------- 4) RUN + raccolta figure per PDF ----------
broad_rows = []
figs_for_pdf = []

for bd in BROAD_DEFICITS:
    outs = []

    out1, (L1, F1, S1, mcore1, manch1, C1) = measure_window(lam, flux, sig, bd, extra_masks=None, model="linear", rebin=1)
    outs.append(out1)
    fig1 = plot_window(L1, F1, S1, bd, (mcore1, manch1), C1, out1["verdict_key"], title_note=f"[rebin=1, {out1['verdict']}]")
    figs_for_pdf.append(fig1)

    try:
        REBIN_LOCAL = int(REBIN) if 'REBIN' in globals() else 1
    except Exception:
        REBIN_LOCAL = 1
    if REBIN_LOCAL and REBIN_LOCAL > 1:
        out2, (L2, F2, S2, mcore2, manch2, C2) = measure_window(lam, flux, sig, bd, extra_masks=None, model="linear", rebin=REBIN_LOCAL)
        outs.append(out2)
        fig2 = plot_window(L2, F2, S2, bd, (mcore2, manch2), C2, out2["verdict_key"], title_note=f"[rebin={REBIN_LOCAL}, {out2['verdict']}]")
        figs_for_pdf.append(fig2)

    best = max(outs, key=lambda d: (np.nan_to_num(d["SN"], nan=-1e9), np.nan_to_num(d["zsig"], nan=-1e9)))
    broad_rows.append(best)

# ---------- 5) Tabella riassuntiva (a schermo + salvataggi opzionali) ----------
df_broad = None
if broad_rows:
    cols = ["band","label","core","anchors_used","N_core","SN","zsig","EW","tau_bar","DeltaChi2","s_hat","model","rebin","verdict"]
    df_broad = pd.DataFrame(broad_rows, columns=cols)
    print("\n=== Broad-deficit summary (PAH + Silicate masked) ===")
    display(df_broad.style.background_gradient(subset=["SN","zsig"], cmap="Greens"))
else:
    print("\n=== Broad-deficit summary ===\n(Nessuna finestra valutabile)")

# ---------- 6) Overview delle maschere PAH/SIL su tutto lo spettro ----------
overview_fig = None
try:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.errorbar(lam, flux, yerr=sig, fmt='o', ms=2.8,
                ecolor='#cccccc', mec="#1f77b4", mfc="#1f77b4", alpha=0.9)

    # PAH = violet (lavender), Silicate = soft yellow
    for (a,b) in PAH_WINDOWS:
        ax.axvspan(a, b, color="#c084fc", alpha=0.12, lw=0)  # light violet for PAH
    for (a,b) in SIL_WINDOWS:
        ax.axvspan(a, b, color="#fff59d", alpha=0.20, lw=0)  # pastel yellow for silicates

    ax.set_title(f"{OBJECT_ID} — overview: PAH (violet) + Silicate (yellow) masks")
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Flux (Jy)")
    ax.text(0.01, 1.02,
            "Overview: PAH (violet) and silicate (yellow) masks exclude contaminated regions\n"
            "to ensure local continua and broad-deficit fits are clean.",
            transform=ax.transAxes, fontsize=8, va='bottom')

    plt.tight_layout()
    plt.show()
    overview_fig = fig
    figs_for_pdf.append(fig)
except Exception as e:
    print("Overview plot skipped:", e)

# ---------- 7) Salvataggi (CSV/TeX/PDF multipagina) ----------
if df_broad is not None and SAVE_CSV:
    csv_path = f"{OUT_BASE}_summary.csv"
    df_broad.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved → {csv_path}")

if df_broad is not None and SAVE_TEX:
    tex_path = f"{OUT_BASE}_summary.tex"
    with open(tex_path, "w") as f:
        f.write(df_broad.to_latex(index=False))
    print(f"[OK] LaTeX saved → {tex_path}")

if SAVE_PDF and len(figs_for_pdf) > 0:
    # prima pagina: tabella come immagine matplotlib
    if df_broad is not None:
        fig_tab, ax_tab = plt.subplots(figsize=(10, 3.8))
        ax_tab.axis('off')
        title = f"{OBJECT_ID} — Broad-deficit summary (PAH/Silicate masked)" if 'OBJECT_ID' in globals() else "Broad-deficit summary"
        ax_tab.set_title(title, loc='left')
        ax_tab.table(cellText=df_broad.values, colLabels=df_broad.columns, cellLoc='center', loc='upper left')
        figs_for_pdf.insert(0, fig_tab)

    pdf_path = f"{OUT_BASE}.pdf"
    with PdfPages(pdf_path) as pdf:
        for f in figs_for_pdf:
            pdf.savefig(f, bbox_inches='tight')
    print(f"[OK] PDF saved → {pdf_path}")

# ---------- Note operative (immutate) ----------
# - Non altera le misure CNT strette: è un check extra "broad-deficit" con PAH/SIL esclusi.
# - Verdetti con le stesse soglie DET_THRESH/MARG_THRESH e √Δχ² del pipeline.
# - Per cambiare/estendere maschere o finestre, modifica PAH_WINDOWS / SIL_WINDOWS o BROAD_DEFICITS.
# CERTIFICAZIONE PER BANDA (report compatto "da referee")

# Usa: results, safe_df (se esiste), lsf_df, inj_df, extra_df (se esiste), DET_THRESH, MARG_THRESH, INJ_DEPTHS

# ===========================

import numpy as np

import pandas as pd


print("\n================ CNT/HNT — OVERVIEW PER EACH BAND ================\n")


# helper: pick nearest depth in injection grid

def _nearest_depth(target, grid):

    grid = np.array(list(grid), float)

    if grid.size == 0 or not np.isfinite(target):

        return None

    idx = int(np.argmin(np.abs(grid - target)))

    return float(grid[idx])


# helper: safe lookup

def _safe_pair_for(band_name):

    try:

        if 'safe_df' in globals() and isinstance(safe_df, pd.DataFrame):

            m = safe_df['band_base'] == band_name

            if m.any():

                row = safe_df[m].iloc[0]

                return dict(

                    pair=row.get('pair_verdict','NO'),

                    key =row.get('pair_key','gray'),

                    std=row.get('std_verdict',''),

                    saf=row.get('safe_verdict','')

                )

    except Exception:

        pass

    return dict(pair='n/a', key='gray', std='', saf='')


# helper: lsf lookup

def _lsf_for(band_name):

    try:

        row = lsf_df[lsf_df['band']==band_name].iloc[0]

        w2l = float(row.get('width_to_LSF', np.nan))

        resolved = bool(row.get('resolved', False))

        snb = float(row.get('SN_base', np.nan))

        zb  = float(row.get('z_base', np.nan))

        return dict(width_to_LSF=w2l, resolved=resolved, SN_base=snb, z_base=zb)

    except Exception:

        return dict(width_to_LSF=np.nan, resolved=False, SN_base=np.nan, z_base=np.nan)


# helper: injection lookup (nearest depth to observed)

def _inj_for(band_name, depth_obs):

    try:

        sub = inj_df[inj_df['band']==band_name]

        if sub.empty:

            return dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)

        depth_grid = sorted(sub['depth'].unique())

        dsel = _nearest_depth(depth_obs, depth_grid)

        if dsel is None:

            return dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)

        subd = sub[sub['depth']==dsel]

        # scegli il rebin con rec_rate max

        idx = subd['rec_rate'].astype(float).idxmax()

        row = subd.loc[idx]

        return dict(depth_sel=float(dsel),

                    best_rate=float(row['rec_rate']),

                    best_R=int(row['rebin']),

                    SN_med=float(row['SN_med']),

                    z_med=float(row['z_med']))

    except Exception:

        return dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)


# helper: stability lookup (rebin/edge/jk/poly2) — disponibile per bande YES nel tuo "extra_df"

def _stability_for(band_name):

    if 'extra_df' not in globals() or not isinstance(extra_df, pd.DataFrame) or extra_df.empty:

        return dict(has=False, rebin_ok=False, edge_ok=False, jk_ok=False, poly2_ok=False,

                    SN_R1=np.nan, z_R1=np.nan, SN_R2=np.nan, z_R2=np.nan,

                    SN_edge_min=np.nan, z_edge_min=np.nan,

                    SN_jk_mean=np.nan, z_jk_mean=np.nan,

                    SN_poly2=np.nan, z_poly2=np.nan)

    m = extra_df['band']==band_name

    if not m.any():

        return dict(has=False, rebin_ok=False, edge_ok=False, jk_ok=False, poly2_ok=False,

                    SN_R1=np.nan, z_R1=np.nan, SN_R2=np.nan, z_R2=np.nan,

                    SN_edge_min=np.nan, z_edge_min=np.nan,

                    SN_jk_mean=np.nan, z_jk_mean=np.nan,

                    SN_poly2=np.nan, z_poly2=np.nan)

    r = extra_df[m].iloc[0]

    SN_R1 = float(r.get('SN_R1', np.nan)); z_R1 = float(r.get('z_R1', np.nan))

    SN_R2 = float(r.get('SN_R2', np.nan)); z_R2 = float(r.get('z_R2', np.nan))

    SN_edge_min = float(r.get('SN_edge_min', np.nan)); z_edge_min = float(r.get('z_edge_min', np.nan))

    SN_jk_mean = float(r.get('SN_jk_mean', np.nan)); z_jk_mean = float(r.get('z_jk_mean', np.nan))

    SN_poly2 = float(r.get('SN_poly2', np.nan)); z_poly2 = float(r.get('z_poly2', np.nan))


    rebin_ok = (SN_R1>=MARG_THRESH and z_R1>=MARG_THRESH and

                SN_R2>=MARG_THRESH and z_R2>=MARG_THRESH) if np.isfinite(SN_R1) and np.isfinite(SN_R2) else False

    edge_ok  = (SN_edge_min>=MARG_THRESH and z_edge_min>=MARG_THRESH) if np.isfinite(SN_edge_min) else False

    # jackknife: se mancano dati, non penalizzare (come nel tuo extra)

    jk_ok    = True if not np.isfinite(SN_jk_mean) else (SN_jk_mean>=MARG_THRESH and z_jk_mean>=MARG_THRESH)

    poly2_ok = (SN_poly2>=MARG_THRESH and z_poly2>=MARG_THRESH) if np.isfinite(SN_poly2) else False


    return dict(has=True, rebin_ok=rebin_ok, edge_ok=edge_ok, jk_ok=jk_ok, poly2_ok=poly2_ok,

                SN_R1=SN_R1, z_R1=z_R1, SN_R2=SN_R2, z_R2=z_R2,

                SN_edge_min=SN_edge_min, z_edge_min=z_edge_min,

                SN_jk_mean=SN_jk_mean, z_jk_mean=z_jk_mean,

                SN_poly2=SN_poly2, z_poly2=z_poly2)


def _bool2txt(ok):

    if ok is True:  return "OK"

    if ok is False: return "check"

    return "n/a"


def _final_decision(base_sn_ok, base_z_ok, safe_ok, lsf_ok, inj_ok, stab_ok):

    # regole: robust se base ok e tutti i blocchi OK; detection se base ok e >=3 blocchi OK

    blocks_ok = sum([1 if x else 0 for x in [safe_ok, lsf_ok, inj_ok, stab_ok]])

    if base_sn_ok and base_z_ok and all([safe_ok, lsf_ok, inj_ok, stab_ok]):

        return "YES (robust)"

    if base_sn_ok and base_z_ok and blocks_ok >= 3:

        return "YES (detection)"

    if base_sn_ok or base_z_ok:

        return "MARGINAL/REVIEW"

    return "NO"


for R in results:

    bname = R["band"]; core_rng = R["core"]

    SNv = float(R["SN"]); zv = float(R["zsig"]); EWv = float(R["EW"]); TAU = float(R["tau_bar"])

    base_verdict = R["verdict"]

    base_sn_ok = np.isfinite(SNv) and (SNv >= DET_THRESH)

    base_z_ok  = np.isfinite(zv)  and (zv  >= DET_THRESH)


    # SAFE

    safe_info = _safe_pair_for(bname)

    safe_ok = (safe_info['pair'] in ["YES (robust)", "YES (detection)"])


    # LSF

    lsf = _lsf_for(bname)

    w2lsf = lsf['width_to_LSF']

    lsf_ok = bool(np.isfinite(w2lsf) and w2lsf >= 1.0)


    # InRec (profondità "simile" all'osservata): usa TAU come proxy di depth (piccole tau ~ depth)

    if not ('INJ_DEPTHS' in globals() and len(INJ_DEPTHS)>0):

        inj_ok = False

        inj = dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)

    else:

        # clip su griglia disponibile (non forzare fuori range)

        dmin, dmax = min(INJ_DEPTHS), max(INJ_DEPTHS)

        depth_obs = float(np.clip(TAU if np.isfinite(TAU) else dmin, dmin, dmax))

        inj = _inj_for(bname, depth_obs)

        inj_ok = bool(np.isfinite(inj['best_rate']) and (inj['best_rate'] >= 0.8))


    # Stability (se disponibile da extra_df)

    stab = _stability_for(bname)

    stab_ok = bool(stab['rebin_ok'] and stab['edge_ok'] and stab['jk_ok'] and stab['poly2_ok']) if stab['has'] else False


    final = _final_decision(base_sn_ok, base_z_ok, safe_ok, lsf_ok, inj_ok, stab_ok)


    # stampa sintetica

    print(f"{bname} [{core_rng}] — base: S/N={SNv:.2f}, √Δχ²={zv:.2f}; EW={EWv:.3e} μm; ⟨τ⟩={TAU:.3e}")

    print(f"  SAFE: {safe_info['pair']}  |  LSF width/LSF={w2lsf:.2f} → {_bool2txt(lsf_ok)}")

    if inj['depth_sel'] is not None:

        print(f"  InRec: depth≈{inj['depth_sel']:.3f}, best rec={inj['best_rate']:.2f} @R={inj['best_R']} → {_bool2txt(inj_ok)}")

    else:

        print(f"  InRec: n/a → {_bool2txt(False)}")

    if stab['has']:

        print(f"  Stability: rebin(1,2)→{_bool2txt(stab['rebin_ok'])}; edge±0.05→{_bool2txt(stab['edge_ok'])}; "

              f"jackknife→{_bool2txt(stab['jk_ok'])}; poly2→{_bool2txt(stab['poly2_ok'])}")

    else:

        print(f"  Stability: n/a (extra checks non applicati per questa banda)")

    print(f"  ⇒ Final result: {final}  (base={base_verdict})\n")
# ============================
# APPEND-ONLY — Compact summary (table + plots + optional export)
# Dipendenze: results, lsf_df, inj_df, (safe_df opz.), (extra_df opz.), DET_THRESH, MARG_THRESH
# NON modifica il codice precedente: ricalcola tutto in lettura.
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Opzioni utente ----------
SAVE_CSV   = True     # salva tabella in CSV
SAVE_TEX   = True     # salva tabella in LaTeX
SAVE_PDF   = True     # salva multipagina con tabella + miniplot
SAVE_PNG   = True     # salva PNG della tabella
MAX_ROWS_PER_TABLE_FIG = 12

OUT_BASENAME = f"{OBJECT_ID.replace(' ', '_')}_band_cert" if 'OBJECT_ID' in globals() else "band_cert"

# ---------- Helper locali: copie "read-only" delle tue funzioni ----------
def _safe_pair_for_read(band_name):
    try:
        if 'safe_df' in globals() and isinstance(safe_df, pd.DataFrame):
            m = (safe_df['band_base'] == band_name)
            if m.any():
                row = safe_df[m].iloc[0]
                return dict(pair=row.get('pair_verdict','NO'),
                            key=row.get('pair_key','gray'),
                            std=row.get('std_verdict',''),
                            saf=row.get('safe_verdict',''))
    except Exception:
        pass
    return dict(pair='n/a', key='gray', std='', saf='')

def _lsf_for_read(band_name):
    try:
        row = lsf_df[lsf_df['band']==band_name].iloc[0]
        return dict(width_to_LSF=float(row.get('width_to_LSF', np.nan)),
                    resolved=bool(row.get('resolved', False)),
                    SN_base=float(row.get('SN_base', np.nan)),
                    z_base=float(row.get('z_base', np.nan)))
    except Exception:
        return dict(width_to_LSF=np.nan, resolved=False, SN_base=np.nan, z_base=np.nan)

def _nearest_depth_read(target, grid):
    g = np.array(list(grid), float)
    if g.size == 0 or not np.isfinite(target): return None
    idx = int(np.argmin(np.abs(g - target)))
    return float(g[idx])

def _inj_for_read(band_name, depth_obs):
    try:
        sub = inj_df[inj_df['band']==band_name]
        if sub.empty:
            return dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)
        depth_grid = sorted(sub['depth'].unique())
        dsel = _nearest_depth_read(depth_obs, depth_grid)
        if dsel is None:
            return dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)
        subd = sub[sub['depth']==dsel]
        idx = subd['rec_rate'].astype(float).idxmax()
        row = subd.loc[idx]
        return dict(depth_sel=float(dsel),
                    best_rate=float(row['rec_rate']),
                    best_R=int(row['rebin']),
                    SN_med=float(row['SN_med']),
                    z_med=float(row['z_med']))
    except Exception:
        return dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)

def _stability_for_read(band_name):
    if 'extra_df' not in globals() or not isinstance(extra_df, pd.DataFrame) or extra_df.empty:
        return dict(has=False, rebin_ok=False, edge_ok=False, jk_ok=False, poly2_ok=False)
    m = (extra_df['band']==band_name)
    if not m.any():
        return dict(has=False, rebin_ok=False, edge_ok=False, jk_ok=False, poly2_ok=False)
    r = extra_df[m].iloc[0]
    def _f(x):
        try:
            return float(r.get(x, np.nan))
        except Exception:
            return np.nan
    SN_R1, z_R1 = _f('SN_R1'), _f('z_R1')
    SN_R2, z_R2 = _f('SN_R2'), _f('z_R2')
    SN_edge_min, z_edge_min = _f('SN_edge_min'), _f('z_edge_min')
    SN_jk_mean, z_jk_mean = _f('SN_jk_mean'), _f('z_jk_mean')
    SN_poly2, z_poly2 = _f('SN_poly2'), _f('z_poly2')

    rebin_ok = (SN_R1>=MARG_THRESH and z_R1>=MARG_THRESH and SN_R2>=MARG_THRESH and z_R2>=MARG_THRESH) if np.isfinite(SN_R1) and np.isfinite(SN_R2) else False
    edge_ok  = (SN_edge_min>=MARG_THRESH and z_edge_min>=MARG_THRESH) if np.isfinite(SN_edge_min) else False
    jk_ok    = True if not np.isfinite(SN_jk_mean) else (SN_jk_mean>=MARG_THRESH and z_jk_mean>=MARG_THRESH)
    poly2_ok = (SN_poly2>=MARG_THRESH and z_poly2>=MARG_THRESH) if np.isfinite(SN_poly2) else False

    return dict(has=True, rebin_ok=rebin_ok, edge_ok=edge_ok, jk_ok=jk_ok, poly2_ok=poly2_ok)

def _final_decision_read(base_sn_ok, base_z_ok, safe_ok, lsf_ok, inj_ok, stab_ok):
    blocks_ok = sum([1 if x else 0 for x in [safe_ok, lsf_ok, inj_ok, stab_ok]])
    if base_sn_ok and base_z_ok and all([safe_ok, lsf_ok, inj_ok, stab_ok]):
        return "YES (robust)"
    if base_sn_ok and base_z_ok and blocks_ok >= 3:
        return "YES (detection)"
    if base_sn_ok or base_z_ok:
        return "MARGINAL/REVIEW"
    return "NO"

# ---------- Costruisci la tabella riassuntiva senza toccare il codice esistente ----------
rows = []
for R in results:
    bname = R["band"]; core_rng = R["core"]
    SNv = float(R["SN"]); zv = float(R["zsig"]); EWv = float(R["EW"]); TAU = float(R["tau_bar"])
    base_verdict = R["verdict"]
    base_sn_ok = np.isfinite(SNv) and (SNv >= DET_THRESH)
    base_z_ok  = np.isfinite(zv)  and (zv  >= DET_THRESH)

    safe_info = _safe_pair_for_read(bname)
    safe_ok = (safe_info['pair'] in ["YES (robust)", "YES (detection)"])

    lsf = _lsf_for_read(bname)
    w2lsf = lsf['width_to_LSF']
    lsf_ok = bool(np.isfinite(w2lsf) and w2lsf >= 1.0)

    if not ('INJ_DEPTHS' in globals() and len(INJ_DEPTHS)>0):
        inj = dict(depth_sel=None, best_rate=np.nan, best_R=None, SN_med=np.nan, z_med=np.nan)
        inj_ok = False
    else:
        dmin, dmax = min(INJ_DEPTHS), max(INJ_DEPTHS)
        depth_obs = float(np.clip(TAU if np.isfinite(TAU) else dmin, dmin, dmax))
        inj = _inj_for_read(bname, depth_obs)
        inj_ok = bool(np.isfinite(inj['best_rate']) and (inj['best_rate'] >= 0.8))

    stab = _stability_for_read(bname)
    stab_ok = bool(stab['rebin_ok'] and stab['edge_ok'] and stab['jk_ok'] and stab['poly2_ok']) if stab['has'] else False

    final = _final_decision_read(base_sn_ok, base_z_ok, safe_ok, lsf_ok, inj_ok, stab_ok)

    rows.append(dict(
        Band=bname, Core=core_rng,
        SN=SNv, z=zv, EW=EWv, tau=TAU,
        Base=base_verdict,
        SAFE_pair=safe_info['pair'],
        LSF_ratio=w2lsf, LSF_ok=lsf_ok,
        InRec_depth=inj['depth_sel'], InRec_best=inj['best_rate'], InRec_R=inj['best_R'], InRec_ok=inj_ok,
        Rebin_ok=stab.get('rebin_ok', False), Edge_ok=stab.get('edge_ok', False),
        Jack_ok=stab.get('jk_ok', False), Poly2_ok=stab.get('poly2_ok', False),
        Final=final
    ))

df_cert = pd.DataFrame(rows, columns=[
    "Band","Core","SN","z","EW","tau","Base",
    "SAFE_pair","LSF_ratio","LSF_ok",
    "InRec_depth","InRec_best","InRec_R","InRec_ok",
    "Rebin_ok","Edge_ok","Jack_ok","Poly2_ok",
    "Final"
])

# ---------- Stampa a schermo (Colab-friendly) ----------
from IPython.display import display
print("\n=== Compact summary (per band) ===")
display(df_cert.style.hide(axis="index").format({
    "SN":"{:.2f}","z":"{:.2f}","EW":"{:.3e}","tau":"{:.3e}",
    "LSF_ratio":"{:.2f}","InRec_depth":"{:.3f}","InRec_best":"{:.2f}"
}).apply(lambda s: ["background-color:#d9f2d9" if (v in ["YES (robust)","YES (detection)"]) else
                    ("background-color:#fce8b2" if v=="MARGINAL/REVIEW" else
                     ("background-color:#f4d6d6" if v=="NO" else "")) for v in s] if s.name=="Final" else [""]*len(s), axis=0))

# ---------- Export tabella ----------
try:
    if SAVE_CSV:
        df_cert.to_csv(f"{OUT_BASENAME}.csv", index=False)
        print(f"[OK] CSV → {OUT_BASENAME}.csv")
    if SAVE_TEX:
        with open(f"{OUT_BASENAME}.tex","w") as f:
            f.write(df_cert.to_latex(index=False, escape=True))
        print(f"[OK] LaTeX → {OUT_BASENAME}.tex")
except Exception as e:
    print("Export tabella (CSV/LaTeX) fallito:", e)

# ---------- Figure: tabella impaginata + miniplot per banda ----------
def _table_figure(df, title):
    # spezza se troppe righe
    figs = []
    n = len(df)
    for start in range(0, n, MAX_ROWS_PER_TABLE_FIG):
        chunk = df.iloc[start:start+MAX_ROWS_PER_TABLE_FIG]
        fig, ax = plt.subplots(figsize=(12, 0.6*len(chunk) + 2.2))
        ax.axis('off')
        ax.set_title(title + (f" (rows {start+1}-{start+len(chunk)})" if n>MAX_ROWS_PER_TABLE_FIG else ""), loc='left', fontsize=12, pad=8)
        tbl = ax.table(cellText=chunk.values,
                       colLabels=chunk.columns,
                       cellLoc='center', loc='upper left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.35)
        figs.append(fig)
    return figs

def _bar_pair(val1, val2, title, ylabel):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar([0,1], [val1, val2], width=0.6)
    ax.set_xticks([0,1], ["S/N", "z"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', lw=0.4, alpha=0.6)
    # Nota in inglese (utile per il referee/archivio)
    ax.text(0.02, -0.22, "Useful for: quick band strength vs significance comparison",
            transform=ax.transAxes, fontsize=8, va='top')
    plt.tight_layout()
    return fig

# Crea figure
figs = []
# Tabella “compatta” (solo colonne chiave)
df_short = df_cert[["Band","Core","SN","z","Base","SAFE_pair","LSF_ratio","InRec_best","Final"]].copy()
df_short["SN"] = df_short["SN"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "nan")
df_short["z"]  = df_short["z"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "nan")
df_short["LSF_ratio"]  = df_short["LSF_ratio"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "nan")
df_short["InRec_best"] = df_short["InRec_best"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "nan")
figs += _table_figure(df_short, title=f"{OBJECT_ID} — CNT/HNT per-band certification" if 'OBJECT_ID' in globals() else "CNT/HNT per-band certification")

# Mini-plot per ogni banda: barre S/N e z
for _, r in df_cert.iterrows():
    try:
        sn = float(r["SN"]); zz = float(r["z"])
        title = f"{r['Band']} ({r['Core']}) — {r['Final']}"
        figs.append(_bar_pair(sn, zz, title=title, ylabel="σ"))
    except Exception:
        pass

# Mostra figure
for fig in figs:
    plt.show(fig)

# Salvatore
try:
    if SAVE_PNG and len(figs) > 0:
        figs[0].savefig(f"{OUT_BASENAME}_table.png", dpi=200, bbox_inches='tight')
        print(f"[OK] PNG → {OUT_BASENAME}_table.png")
    if SAVE_PDF and len(figs) > 0:
        with PdfPages(f"{OUT_BASENAME}.pdf") as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
        print(f"[OK] PDF → {OUT_BASENAME}.pdf")
except Exception as e:
    print("Export figure (PNG/PDF) fallito:", e)



# ===============================================================
# CNT/HNT: (2) Template matching, (3) Slope check, (4) Coherence
# Append-only cell for Colab. Safe with your existing pipeline.
# Requires: lam, flux, sig, BANDS, MASKS_GLOBAL,
#           fit_continuum, metrics_core, in_windows,
#           DET_THRESH, MARG_THRESH, OBJECT_ID
# Optional (if present): results
# ===============================================================

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from scipy.stats import norm

# ------------------------ user options -------------------------
SAVE = True              # set False to skip all file outputs
OUT = (OBJECT_ID.replace(" ", "_") if 'OBJECT_ID' in globals() else "object")

TPL_FWHM = 0.030         # µm, Gaussian width for theory templates (Chen & Li smoothing ~30 cm^-1)
LEFT_PAD  = 0.50         # µm, slope window left of core
RIGHT_PAD = 0.50         # µm, slope window right of core
GAP       = 0.10         # µm, gap between core edge and slope window

# Where to save
PATH_TPL_CSV = f"{OUT}_tplmatch.csv"
PATH_TPL_TEX = f"{OUT}_tplmatch.tex"
PATH_TPL_PDF = f"{OUT}_tplmatch.pdf"
PATH_SLP_CSV = f"{OUT}_slopecheck.csv"
PATH_SLP_TEX = f"{OUT}_slopecheck.tex"
PATH_SLP_PDF = f"{OUT}_slopecheck.pdf"
PATH_CO_CSV  = f"{OUT}_coherence.csv"
PATH_CO_TEX  = f"{OUT}_coherence.tex"
PATH_CO_PDF  = f"{OUT}_coherence.pdf"

# ---------------- small helpers that do NOT override yours ----------------
def _fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

def _safe_minmax(a):
    return float(np.nanmin(a)), float(np.nanmax(a))

def _continuum_for_band(band, lam, flux, sig, extra_masks=None):
    """Fit continuum on the band's anchors (re-using your fit_continuum)."""
    anchors = band["anchors"]; pivot = band["pivot"]
    m_lines = in_windows(lam, MASKS_GLOBAL)
    m_anch  = in_windows(lam, anchors) & (~m_lines)
    if extra_masks:
        for w in extra_masks:
            m_anch &= ~in_windows(lam, [w])
    return fit_continuum(lam, flux, sig, m_anch, pivot, model="linear")

def _core_used_mask(band, lam):
    c0, c1 = band["core"]
    m_core = (lam >= c0) & (lam <= c1)
    m_lines = in_windows(lam, MASKS_GLOBAL)
    return m_core & (~m_lines)

def _residuals(lam, flux, sig, C, used):
    """Return residual y=(C-F)/C and its sigma, y_sigma=σ/C, on 'used' mask."""
    L = lam[used]; F = flux[used]; S = sig[used]
    Cval = C(L)
    Cval = np.where(np.isfinite(Cval) & (Cval>0), Cval, np.nanmax([1e-30]))
    y = (Cval - F)/Cval
    ysig = S/np.maximum(Cval, 1e-30)
    return L, y, ysig

def _wls_alpha(y, t, ysig, alpha_nonneg=True):
    """Weighted LS amplitude for model alpha*t approximating y (weights=1/ysig^2)."""
    w = 1.0/np.maximum(ysig**2, 1e-30)
    num = np.nansum(w * y * t)
    den = np.nansum(w * t * t)
    if den <= 0 or not np.isfinite(den):
        return np.nan
    a = num/den
    if alpha_nonneg:
        a = max(0.0, a)
    return a

def _chi2(y, ysig):
    w = 1.0/np.maximum(ysig**2, 1e-30)
    r = np.nan_to_num(y, nan=0.0)
    return float(np.nansum(w * r*r))

def _linfit_weighted(x, y, ysig):
    """Weighted linear fit y = a + b x; returns (b, b_err)."""
    w = 1.0/np.maximum(ysig**2, 1e-30)
    W = np.sum(w)
    if W <= 0:
        return np.nan, np.nan
    xw = np.sum(w*x)/W
    yw = np.sum(w*y)/W
    Sxx = np.sum(w*(x-xw)**2)
    if Sxx <= 0:
        return np.nan, np.nan
    b = np.sum(w*(x-xw)*(y-yw))/Sxx
    # conservative variance of slope:
    resid = y - (yw + b*(x-xw))
    s2 = np.sum(w*resid**2)/max((len(x)-2), 1)
    b_err = math.sqrt(s2 / Sxx)
    return float(b), float(b_err)

def _corr_pearson(x, y):
    if len(x)<3 or len(y)<3:
        return np.nan, np.nan
    r = np.corrcoef(x, y)[0,1]
    # p-value (approx, two-sided) using normal approximation atanh(r)*sqrt(n-3)
    n = min(len(x), len(y))
    if not np.isfinite(r) or abs(r) >= 1 or n<5:
        return float(r), np.nan
    z = 0.5*np.log((1+r)/(1-r)) * math.sqrt(n-3)
    p = 2*norm.sf(abs(z))
    return float(r), float(p)

# ----------------------- 2) TEMPLATE LIBRARY -------------------------------
# Chen & Li (2022) dominant bands; we synthesize multi-Gaussian optical-depth shapes.
def _make_template(name, peaks_um, fwhm_um=TPL_FWHM, weights=None, lam_min=None, lam_max=None):
    lam0 = float(np.nanmin(lam)) if lam_min is None else float(lam_min)
    lam1 = float(np.nanmax(lam)) if lam_max is None else float(lam_max)
    grid = np.linspace(lam0, lam1, 6000)
    tau = np.zeros_like(grid)
    sig = _fwhm_to_sigma(fwhm_um)
    if weights is None:
        weights = np.ones(len(peaks_um), float)
    for p, w in zip(peaks_um, weights):
        tau += w*np.exp(-0.5*((grid-p)/sig)**2)
    m = np.nanmax(tau)
    if m>0 and np.isfinite(m):
        tau = tau/m
    return dict(name=name, lam=grid, tau=tau)

TPL = [
    _make_template("CNT (neutral) — Chen & Li 2022", [5.3, 7.0, 8.8, 9.7, 10.9, 14.2, 16.8]),
    _make_template("CNT+ (cation) — Chen & Li 2022", [5.2, 7.1, 8.3, 9.2, 13.4, 16.7], weights=[1,1,1,1,1.2,1.1]),
    _make_template("HNT — Chen & Li 2022",         [3.3, 6.5, 8.2, 10.8, 13.0, 17.2]),
    _make_template("HNT+ (cation) — Chen & Li 2022",[3.3, 6.6, 8.3, 10.6, 12.8, 17.2], weights=[1,1,1,1,1.2,1.1]),
]

# ---------------- run per-band measurements (re-use your windows) ----------
tpl_rows, slope_rows = [], []

# For coherence we will store normalized residual profiles for key bands
CO_BANDS = {
    "10.7": "CNT_10p60_10p90",
    "12.9": "CNT_12p86_13p40",
    "17.0": "CNT_16p80_17p20",
}
co_prof = {}   # co_prof["10.7"] = (x_norm, y_norm)

for b in BANDS:
    try:
        # 2a) Continuum & used mask in the band's core
        fit = _continuum_for_band(b, lam, flux, sig)
        used = _core_used_mask(b, lam)
        M, used2 = metrics_core(lam, flux, sig, fit["C"], used)  # to respect your internal clipping
        used = used2
        if used.sum() < 3:
            # Not enough data → skip cleanly
            tpl_rows.append(dict(band=b["name"], best_template="n/a", z= np.nan,
                                 alpha=np.nan, delta_chi2=np.nan, N=int(used.sum())))
            slope_rows.append(dict(band=b["name"], left_slope=np.nan, right_slope=np.nan,
                                   delta=np.nan, t=np.nan, p=np.nan, verdict="n/a"))
            continue

        L, y, ysig = _residuals(lam, flux, sig, fit["C"], used)

        # 2b) TEMPLATE MATCHING (weighted least-squares on residuals)
        best = dict(name="(none)", delta=-np.inf, alpha=np.nan, z=np.nan)
        for t in TPL:
            f = interp1d(t["lam"], t["tau"], kind="linear", bounds_error=False, fill_value=0.0)
            T = f(L)  # template sampled at data wavelengths
            if np.allclose(T, 0.0) or not np.any(np.isfinite(T)):
                continue
            a = _wls_alpha(y, T, ysig, alpha_nonneg=True)
            if not np.isfinite(a):
                continue
            y_model = a*T
            chi_null = _chi2(y, ysig)
            chi_fit  = _chi2(y - y_model, ysig)
            dchi = max(chi_null - chi_fit, 0.0)
            zval = math.sqrt(dchi)
            if dchi > best["delta"]:
                best.update(name=t["name"], delta=float(dchi), alpha=float(a), z=float(zval))

        tpl_rows.append(dict(
            band=b["name"], core=f"{b['core'][0]:.2f}-{b['core'][1]:.2f}",
            best_template=best["name"], z=best["z"], delta_chi2=best["delta"],
            alpha=best["alpha"], N=int(used.sum())
        ))

        # 3) SLOPE CHECK (weighted linear fits left/right of the core)
        c0, c1 = b["core"]
        Lmin, Lmax = _safe_minmax(lam)

        L_left0  = max(Lmin, c0 - LEFT_PAD); L_left1  = max(Lmin, c0 - GAP)
        L_right0 = min(Lmax, c1 + GAP);     L_right1 = min(Lmax, c1 + RIGHT_PAD)

        mL = (lam >= L_left0)  & (lam <= L_left1)
        mR = (lam >= L_right0) & (lam <= L_right1)
        # exclude global masked lines
        mL &= ~in_windows(lam, MASKS_GLOBAL)
        mR &= ~in_windows(lam, MASKS_GLOBAL)

        verdict = "n/a"; delta=np.nan; tval=np.nan; pval=np.nan
        if mL.sum()>=3 and mR.sum()>=3:
            CL = fit["C"](lam[mL]); CR = fit["C"](lam[mR])
            # work on continuum-normalized flux (so slopes are comparable)
            yL = flux[mL]/np.maximum(CL, 1e-30)
            yR = flux[mR]/np.maximum(CR, 1e-30)
            sL = sig[mL]/np.maximum(CL, 1e-30)
            sR = sig[mR]/np.maximum(CR, 1e-30)
            bL, eL = _linfit_weighted(lam[mL], yL, sL)
            bR, eR = _linfit_weighted(lam[mR], yR, sR)
            if np.isfinite(bL) and np.isfinite(bR):
                delta = bL - bR
                se = math.sqrt((eL if np.isfinite(eL) else 0.0)**2 + (eR if np.isfinite(eR) else 0.0)**2)
                if se>0 and np.isfinite(se):
                    tval = delta/se
                    pval = 2*norm.sf(abs(tval))
                    verdict = "OK" if abs(tval) <= 2.0 else "CHECK"
        slope_rows.append(dict(
            band=b["name"], left_slope=bL if 'bL' in locals() else np.nan,
            right_slope=bR if 'bR' in locals() else np.nan,
            delta=delta, t=tval, p=pval, verdict=verdict
        ))

        # 4) store normalized residual profile for coherence (selected bands only)
        for key, bname in CO_BANDS.items():
            if b["name"] == bname:
                # map λ to [0,1] inside the core and z-score normalize y
                xnorm = (L - c0)/max((c1-c0), 1e-6)
                y_norm = (y - np.nanmean(y)) / (np.nanstd(y) if np.nanstd(y)>0 else 1.0)
                co_prof[key] = (xnorm, y_norm)
    except Exception as e:
        # never crash the notebook due to one band
        tpl_rows.append(dict(band=b.get("name","?"), best_template="error", z=np.nan,
                             delta_chi2=np.nan, alpha=np.nan, N=0))
        slope_rows.append(dict(band=b.get("name","?"), left_slope=np.nan, right_slope=np.nan,
                               delta=np.nan, t=np.nan, p=np.nan, verdict="error"))
        print(f"[WARN] Skipped band {b.get('name','?')}: {e}")

tpl_df = pd.DataFrame(tpl_rows, columns=["band","core","best_template","z","delta_chi2","alpha","N"])
slp_df = pd.DataFrame(slope_rows, columns=["band","left_slope","right_slope","delta","t","p","verdict"])

# 4) SPECTRAL COHERENCE (10.7–12.9–17 µm)
co_rows = []
pairs = [("10.7","12.9"), ("12.9","17.0"), ("10.7","17.0")]
for a,bk in pairs:
    if a in co_prof and bk in co_prof:
        xa, ya = co_prof[a]
        xb, yb = co_prof[bk]
        # resample both on a common 0..1 grid for fair Pearson r
        grid = np.linspace(0,1,100)
        fa = interp1d(xa, ya, kind="linear", bounds_error=False, fill_value="extrapolate")
        fb = interp1d(xb, yb, kind="linear", bounds_error=False, fill_value="extrapolate")
        ya2, yb2 = fa(grid), fb(grid)
        r, p = _corr_pearson(ya2, yb2)
        co_rows.append(dict(pair=f"{a} vs {bk}", r=r, p=p))
    else:
        co_rows.append(dict(pair=f"{a} vs {bk}", r=np.nan, p=np.nan))
co_df = pd.DataFrame(co_rows, columns=["pair","r","p"])

# -------------------------- pretty plots & saves ---------------------------
# (English axis titles, as requested)
def _nice_template_page(ax, L, y, ysig, Lmodel, ymodel, title, zbest):
    ax.errorbar(L, y, ysig, fmt='o', ms=3, alpha=0.85)
    ax.plot(Lmodel, ymodel, lw=1.6)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Residual (C−F)/C")
    ax.grid(True, alpha=0.25)
    if np.isfinite(zbest):
        ax.text(0.02, 0.95, f"Template Δχ²→ z = {zbest:.2f}", transform=ax.transAxes,
                va="top", ha="left", bbox=dict(fc="white", ec="0.7", alpha=0.8, boxstyle="round,pad=0.25"))

# build one PDF with per-band template overlay if possible
figs_tpl = []
for row in tpl_rows:
    try:
        bname = row["band"]
        # find the original band spec
        bdef = next((bb for bb in BANDS if bb["name"]==bname), None)
        if bdef is None:
            continue
        fit = _continuum_for_band(bdef, lam, flux, sig)
        used = _core_used_mask(bdef, lam)
        _, used = metrics_core(lam, flux, sig, fit["C"], used)
        if used.sum()<3:
            continue
        L, y, ysig = _residuals(lam, flux, sig, fit["C"], used)
        if row["best_template"] not in [t["name"] for t in TPL]:
            # nothing to draw
            continue
        t = next(t for t in TPL if t["name"]==row["best_template"])
        f = interp1d(t["lam"], t["tau"], kind="linear", bounds_error=False, fill_value=0.0)
        T = f(L)
        a = row["alpha"] if np.isfinite(row["alpha"]) else 0.0
        ymodel = a*T
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        _nice_template_page(ax, L, y, ysig, L, ymodel,
                            title=f"{OBJECT_ID} — {bname} (template: {row['best_template']})",
                            zbest=row.get("z", np.nan))
        plt.tight_layout()
        figs_tpl.append(fig)
    except Exception:
        pass

if SAVE and len(figs_tpl)>0:
    with PdfPages(PATH_TPL_PDF) as pdf:
        for f in figs_tpl: pdf.savefig(f, bbox_inches="tight")
    print(f"[OK] PDF (template match) → {PATH_TPL_PDF}")

if SAVE:
    try:
        tpl_df.to_csv(PATH_TPL_CSV, index=False)
        tpl_df.to_latex(PATH_TPL_TEX, index=False, escape=True)
        print(f"[OK] CSV/LaTeX (template) → {PATH_TPL_CSV}, {PATH_TPL_TEX}")
    except Exception as e:
        print("[WARN] template table save skipped:", e)

# Slope-check PDF: one page per band with left/right points & lines
figs_slp = []
for row in slope_rows:
    try:
        bdef = next((bb for bb in BANDS if bb["name"]==row["band"]), None)
        if bdef is None: continue
        c0, c1 = bdef["core"]
        Lmin, Lmax = _safe_minmax(lam)
        L_left0  = max(Lmin, c0 - LEFT_PAD); L_left1  = max(Lmin, c0 - GAP)
        L_right0 = min(Lmax, c1 + GAP);     L_right1 = min(Lmax, c1 + RIGHT_PAD)
        mL = (lam >= L_left0) & (lam <= L_left1) & (~in_windows(lam, MASKS_GLOBAL))
        mR = (lam >= L_right0) & (lam <= L_right1) & (~in_windows(lam, MASKS_GLOBAL))
        if mL.sum()<3 or mR.sum()<3:
            continue
        fit = _continuum_for_band(bdef, lam, flux, sig)
        CL = fit["C"](lam[mL]); CR = fit["C"](lam[mR])
        yL = flux[mL]/np.maximum(CL,1e-30); sL = sig[mL]/np.maximum(CL,1e-30)
        yR = flux[mR]/np.maximum(CR,1e-30); sR = sig[mR]/np.maximum(CR,1e-30)
        bL,_ = _linfit_weighted(lam[mL], yL, sL)
        bR,_ = _linfit_weighted(lam[mR], yR, sR)
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.errorbar(lam[mL], yL, sL, fmt='o', ms=3, label="left window")
        ax.errorbar(lam[mR], yR, sR, fmt='s', ms=3, label="right window")
        if np.isfinite(bL):
            xg = np.linspace(lam[mL].min(), lam[mL].max(), 50)
            ax.plot(xg, (yL.mean() + bL*(xg - lam[mL].mean())), lw=1.2)
        if np.isfinite(bR):
            xg = np.linspace(lam[mR].min(), lam[mR].max(), 50)
            ax.plot(xg, (yR.mean() + bR*(xg - lam[mR].mean())), lw=1.2)
        ax.set_title(f"{OBJECT_ID} — {row['band']} slope check  (verdict: {row['verdict']})")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Flux / Continuum")
        ax.legend(); ax.grid(True, alpha=0.25)
        plt.tight_layout()
        figs_slp.append(fig)
    except Exception:
        pass

if SAVE and len(figs_slp)>0:
    with PdfPages(PATH_SLP_PDF) as pdf:
        for f in figs_slp: pdf.savefig(f, bbox_inches="tight")
    print(f"[OK] PDF (slope check) → {PATH_SLP_PDF}")

if SAVE:
    try:
        slp_df.to_csv(PATH_SLP_CSV, index=False)
        slp_df.to_latex(PATH_SLP_TEX, index=False, escape=True)
        print(f"[OK] CSV/LaTeX (slope) → {PATH_SLP_CSV}, {PATH_SLP_TEX}")
    except Exception as e:
        print("[WARN] slope table save skipped:", e)

# Coherence: print table + simple figure with r values
if not co_df.empty:
    if SAVE:
        try:
            co_df.to_csv(PATH_CO_CSV, index=False)
            co_df.to_latex(PATH_CO_TEX, index=False, escape=True)
            print(f"[OK] CSV/LaTeX (coherence) → {PATH_CO_CSV}, {PATH_CO_TEX}")
        except Exception as e:
            print("[WARN] coherence table save skipped:", e)

    # small plot of correlations
    try:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar(np.arange(len(co_df)), co_df["r"].values)
        ax.set_xticks(np.arange(len(co_df)), co_df["pair"].values)
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{OBJECT_ID} — shape coherence (10.7/12.9/17 µm)")
        ax.grid(True, axis='y', alpha=0.25)
        if SAVE:
            with PdfPages(PATH_CO_PDF) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
            print(f"[OK] PDF (coherence) → {PATH_CO_PDF}")
        plt.show()
    except Exception:
        pass

# quick console summaries
print("\n=== Template matching (best per band) ===")
display(tpl_df)

print("\n=== Slope check (left vs right) ===")
display(slp_df)

print("\n=== Shape coherence (10.7–12.9–17 µm) ===")
display(co_df)
# ===========================
# CARBONET — EXTRA APPEND-ONLY BLOCK
# Matched-filter, PAH-leak, Benchmark vs Literature, Calibration
# Safe to paste at the very end of your notebook
# ===========================

import numpy as np, pandas as pd, math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Options ----------
EXTRA_SAVE_CSV   = True
EXTRA_SAVE_TEX   = True
EXTRA_SAVE_PDF   = True
EXTRA_BASENAME   = (OBJECT_ID.replace(" ", "_") if 'OBJECT_ID' in globals() else "object") + "_extra"

# ---------- Helper: safe gets ----------
def _g(name, default=None):
    return globals()[name] if name in globals() else default

LAM = _g("lam"); FLUX = _g("flux"); SIG = _g("sig")
BANDS_ = _g("BANDS", [])
RESULTS = _g("results", [])
MASKS = _g("MASKS_GLOBAL", [])
DET = _g("DET_THRESH", 3.0); MARG = _g("MARG_THRESH", 2.0)
OBJ = _g("OBJECT_ID", "target")
INJ = _g("inj_df", None)

if LAM is None or FLUX is None or SIG is None or len(BANDS_)==0 or len(RESULTS)==0:
    print("[EXTRA] Missing core variables (lam/flux/sig/BANDS/results). Skipping extra block.")
else:
    LAM = np.asarray(LAM, float); FLUX = np.asarray(FLUX, float); SIG = np.asarray(SIG, float)

    # ---------- Local masks (from literature) ----------
    # PAH 11.2–11.3, PAH 12.7 etc. (kept minimal; your main code already has MASKS_GLOBAL)
    PAH_112 = [(11.18, 11.36)]
    PAH_127 = [(12.62, 12.78)]
    SIL_9_18 = [(9.0, 10.2), (10.2, 10.4), (17.8, 18.6)]

    def _in_windows(x, windows):
        if not windows: return np.zeros_like(x, bool)
        m = np.zeros_like(x, bool)
        for a,b in windows: m |= (x>=a)&(x<=b)
        return m

    # ---------- Continuum + residuals re-derivation (read-only, independent of your functions) ----------
    # We refit each band locally (linear continuum) on its anchors, excluding global masks.
    def _fit_linear_continuum(lam, flux, sig, anchors, pivot):
        ok = np.isfinite(lam)&np.isfinite(flux)&np.isfinite(sig)&(sig>0)
        lam,flux,sig = lam[ok],flux[ok],sig[ok]
        if lam.size<2: 
            return lambda x: np.nan*np.ones_like(np.asarray(x,float)), 0, 0
        ma = _in_windows(lam, anchors) & (~_in_windows(lam, MASKS))
        if ma.sum()<2:  # fallback: weighted mean
            w = 1/np.maximum(sig**2,1e-30)
            a = float(np.sum(w*flux)/np.sum(w))
            return (lambda x: a+0*(np.asarray(x)-pivot)), a, 0
        x = lam - pivot
        X = np.column_stack([np.ones(ma.sum()), x[ma]])
        y = flux[ma]; s = sig[ma]; w = 1/np.maximum(s**2,1e-30)
        Xw = X*np.sqrt(w[:,None]); yw = y*np.sqrt(w)
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        a, b = map(float, beta)
        C = lambda xx: a + b*(np.asarray(xx)-pivot)
        return C, a, b

    def _band_defs_by_name():
        return {b["name"]: b for b in BANDS_}

    BMAP = _band_defs_by_name()
    RES_BY = {r["band"]: r for r in RESULTS}

    # ---------- Matched-filter SNR per band ----------
    # Two templates: top-hat on the core, or Gaussian with FWHM = core width
    def _fwhm_to_sigma(fwhm):
        return fwhm / (2.0*math.sqrt(2.0*math.log(2.0)))

    def _template_on_grid(lam, band, kind="tophat"):
        c0,c1 = band["core"]; pivot = band["pivot"]
        if kind=="tophat":
            return ((lam>=c0)&(lam<=c1)).astype(float)
        # gaussian centered at core center, sigma from core width
        mu = 0.5*(c0+c1); fwhm = max(c1-c0, 1e-6); sig = _fwhm_to_sigma(fwhm)
        return np.exp(-0.5*((lam-mu)/sig)**2)

    def _matched_filter_snr(lam, flux, sig, band, anchors):
        # continuum on anchors, residuals y=(C-F)/C, weights=1/sigma_y^2
        C,_,_ = _fit_linear_continuum(lam, flux, sig, anchors, band["pivot"])
        Cval = C(lam)
        good = np.isfinite(Cval) & (Cval>0) & np.isfinite(sig) & (sig>0)
        if good.sum()<3: return dict(SNR_MF=np.nan, SNR_MF_gauss=np.nan, N=0)
        y = (Cval - flux)/Cval
        ysig = sig/Cval
        w = 1/np.maximum(ysig**2, 1e-30)

        def _snr_for(kind):
            t = _template_on_grid(lam, band, kind=kind)
            t = t * (~_in_windows(lam, MASKS))  # exclude masked lines
            num = np.sum(w * y * t)
            den = np.sqrt(np.sum(w * t * t))
            return float(num/den) if den>0 else np.nan

        return dict(SNR_MF=_snr_for("tophat"),
                    SNR_MF_gauss=_snr_for("gauss"),
                    N=int(good.sum()))

    mf_rows = []
    for b in BANDS_:
        name = b["name"]
        anchors = b.get("anchors", [])
        out = _matched_filter_snr(LAM, FLUX, SIG, b, anchors)
        base = RES_BY.get(name, {})
        baseSN = float(base.get("SN", np.nan))
        # simple consistency flags
        agree_sign = np.sign(baseSN) == np.sign(out["SNR_MF"]) if (np.isfinite(baseSN) and np.isfinite(out["SNR_MF"])) else False
        strong_both = (abs(baseSN)>=MARG) and (abs(out["SNR_MF"])>=MARG)
        mf_rows.append(dict(
            band=name, core=f"{b['core'][0]:.2f}-{b['core'][1]:.2f}",
            SN_base=baseSN, z_base=float(base.get("zsig", np.nan)),
            SNR_MF=out["SNR_MF"], SNR_MF_gauss=out["SNR_MF_gauss"], N_used=out["N"],
            MF_agrees=bool(agree_sign), MF_strong_agreement=bool(agree_sign and strong_both)
        ))
    mf_df = pd.DataFrame(mf_rows, columns=["band","core","SN_base","z_base","SNR_MF","SNR_MF_gauss","N_used","MF_agrees","MF_strong_agreement"])
    print("\n[EXTRA] Matched-filter summary")
    display(mf_df)

    # ---------- PAH-leak check around 11.2–11.3 µm (for 10.6–10.9 band) ----------
    def _pah_index_112(lam, flux, sig, ref_band):
        # continuum from ref_band anchors; compute mean residual inside PAH_112
        C,_,_ = _fit_linear_continuum(lam, flux, sig, ref_band["anchors"], ref_band["pivot"])
        Cval = C(lam); good = np.isfinite(Cval)&(Cval>0)&np.isfinite(sig)&(sig>0)
        if good.sum()<5: return np.nan
        y = (Cval - flux)/Cval
        m = _in_windows(lam, PAH_112)
        if (m & good).sum()<1: return np.nan
        return float(np.nanmean(y[m & good]))

    leak_rows = []
    for b in BANDS_:
        if b["name"] == "CNT_10p60_10p90":
            idx = _pah_index_112(LAM, FLUX, SIG, b)
            # simple leak flag: PAH residual magnitude comparable to core residual level
            base = RES_BY.get(b["name"], {})
            core_SN = float(base.get("SN", np.nan))
            leak_flag = bool(np.isfinite(idx) and abs(idx) >= 0.01 and (not np.isfinite(core_SN) or abs(core_SN) < DET))
            leak_rows.append(dict(band=b["name"], PAH112_index=idx, PAH_leak_flag=leak_flag))
    leak_df = pd.DataFrame(leak_rows, columns=["band","PAH112_index","PAH_leak_flag"]) if leak_rows else pd.DataFrame(columns=["band","PAH112_index","PAH_leak_flag"])
    if not leak_df.empty:
        print("\n[EXTRA] PAH-leak check (10.6–10.9 µm)")
        display(leak_df)

    # ---------- Benchmark vs literature (optional)
    # Provide a DataFrame `lit_df` with columns:
    #  - 'band' (matching your band names) and either:
    #    a) 'label' in {'YES','NO'}  OR  b) 'verdict' containing strings with 'YES'/'NO'
    lit_df = _g("lit_df", None)
    bench_df = None
    if isinstance(lit_df, pd.DataFrame) and not lit_df.empty:
        def _label_from_text(v):
            if isinstance(v, str):
                v = v.upper()
                if "YES" in v: return 1
                if "NO" in v:  return 0
            return np.nan

        y_true_map = {}
        for _, row in lit_df.iterrows():
            name = str(row.get("band", "")).strip()
            lab = row.get("label", None)
            if lab is None: lab = row.get("verdict", None)
            y_true_map[name] = _label_from_text(lab if lab is not None else "")

        rows = []
        for r in RESULTS:
            name = r["band"]; yt = y_true_map.get(name, np.nan)
            vb = str(r.get("verdict","")).upper()
            yp = 1 if "YES" in vb else (0 if "NO" in vb else np.nan)  # ignore "MARGINAL"
            rows.append(dict(band=name, y_true=yt, y_pred=yp))
        bench_df = pd.DataFrame(rows).dropna()
        if not bench_df.empty:
            tp = int(((bench_df.y_true==1)&(bench_df.y_pred==1)).sum())
            tn = int(((bench_df.y_true==0)&(bench_df.y_pred==0)).sum())
            fp = int(((bench_df.y_true==0)&(bench_df.y_pred==1)).sum())
            fn = int(((bench_df.y_true==1)&(bench_df.y_pred==0)).sum())
            prec = tp/(tp+fp) if (tp+fp)>0 else np.nan
            rec  = tp/(tp+fn) if (tp+fn)>0 else np.nan
            acc  = (tp+tn)/max(len(bench_df),1)
            # Cohen's kappa
            p0 = acc
            py = (bench_df.y_true.mean())
            pp = (bench_df.y_pred.mean())
            pe = py*pp + (1-py)*(1-pp)
            kapp = (p0-pe)/(1-pe) if (1-pe)>0 else np.nan
            print("\n[EXTRA] Benchmark vs literature (binary YES/NO; MARGINAL excluded)")
            print(f"  n={len(bench_df)}  TP={tp} FP={fp} TN={tn} FN={fn}")
            print(f"  precision={prec:.3f}  recall={rec:.3f}  accuracy={acc:.3f}  kappa={kapp:.3f}")
        else:
            print("\n[EXTRA] Benchmark: all rows were NaN due to missing/ambiguous labels.")
    else:
        print("\n[EXTRA] No 'lit_df' found → skipping literature benchmark (you can provide it later).")

    # ---------- Calibration from injection–recovery (if available) ----------
    calib_df = None
    if isinstance(INJ, pd.DataFrame) and not INJ.empty:
        # INJ already has rec_rate by (band, depth, rebin). Summarize near thresholds.
        grp = INJ.groupby(["band","depth","rebin"], as_index=False)["rec_rate"].mean().rename(columns={"rec_rate":"P_yes"})
        # also compute a per-band curve collapsing rebin by max
        curv = grp.groupby(["band","depth"], as_index=False)["P_yes"].max().rename(columns={"P_yes":"P_yes_maxR"})
        calib_df = curv
        print("\n[EXTRA] Calibration P(YES | depth) [max over rebin]")
        display(curv.pivot(index="band", columns="depth", values="P_yes_maxR"))
    else:
        print("\n[EXTRA] No 'inj_df' found → skipping calibration table.")

    # ---------- Collect & Export ----------
    # Merge key summaries per band (base+MF+PAH) for a one-shot CSV
    base_df = pd.DataFrame([{k: RES_BY[name].get(k, np.nan) for k in ["band","SN","zsig","EW","tau_bar","verdict"]}
                            | {"core": BMAP[name]["core"]} for name in RES_BY.keys() if name in BMAP])
    base_df = base_df.rename(columns={"SN":"SN_base","zsig":"z_base","tau_bar":"tau_base"})
    out_df = base_df.merge(mf_df[["band","SNR_MF","SNR_MF_gauss","MF_agrees","MF_strong_agreement"]], on="band", how="left")
    if not leak_df.empty:
        out_df = out_df.merge(leak_df, on="band", how="left")

    print("\n[EXTRA] Per-band compact summary (base + matched-filter + PAH-leak)")
    display(out_df)

    try:
        if EXTRA_SAVE_CSV:
            out_df.to_csv(f"{EXTRA_BASENAME}_perband.csv", index=False)
            mf_df.to_csv(f"{EXTRA_BASENAME}_mf.csv", index=False)
            if not leak_df.empty: leak_df.to_csv(f"{EXTRA_BASENAME}_pah.csv", index=False)
            if calib_df is not None: calib_df.to_csv(f"{EXTRA_BASENAME}_calib.csv", index=False)
            if bench_df is not None: bench_df.to_csv(f"{EXTRA_BASENAME}_benchmark_rows.csv", index=False)
            print(f"[OK] CSV exports → {EXTRA_BASENAME}_*.csv")
        if EXTRA_SAVE_TEX:
            try:
                out_df.to_latex(f"{EXTRA_BASENAME}_perband.tex", index=False, escape=True)
                print(f"[OK] LaTeX → {EXTRA_BASENAME}_perband.tex")
            except Exception:
                pass
        if EXTRA_SAVE_PDF:
            figs = []
            # 1) Table page
            def _table_page(df, title):
                fig, ax = plt.subplots(figsize=(11, 0.5*max(len(df),1)+1.8))
                ax.axis('off'); ax.set_title(title, loc='left')
                tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='upper left')
                tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.3)
                figs.append(fig)
            _table_page(out_df[["band","core","SN_base","z_base","SNR_MF","MF_agrees","verdict"]].copy(),
                        title=f"{OBJ} — Extra summary")

            # 2) Simple bar compare SN_base vs SNR_MF per band
            for _, r in out_df.iterrows():
                try:
                    fig, ax = plt.subplots(figsize=(4.6,3.0))
                    ax.bar([0,1],[float(r["SN_base"]), float(r["SNR_MF"])])
                    ax.set_xticks([0,1], ["S/N (base)", "SNR (MF)"]); ax.set_ylabel("σ"); 
                    ax.set_title(str(r["band"])); ax.grid(True, axis='y', lw=0.4, alpha=0.6)
                    figs.append(fig)
                except Exception:
                    pass
            with PdfPages(f"{EXTRA_BASENAME}.pdf") as pdf:
                for f in figs: pdf.savefig(f, bbox_inches='tight')
            print(f"[OK] PDF → {EXTRA_BASENAME}.pdf")
    except Exception as e:
        print("[EXTRA] Export skipped:", e)

    print("\n[EXTRA] Done.")

# — Manifest / versioning —
import sys, json, datetime
_manifest = {
    "object": OBJ,
    "timestamp_utc": datetime.datetime.utcnow().isoformat()+"Z",
    "numpy": np.__version__, "pandas": pd.__version__,
    "DET": float(DET), "MARG": float(MARG),
    "masked_windows_literature": {
        "PAH_11.2-11.3": [11.18, 11.36],
        "PAH_12.7": [12.62, 12.78],
        "silicate_9.7/18": [[9.0,10.2],[10.2,10.4],[17.8,18.6]]
    },
    "bands": {b["name"]: {"core": b["core"], "anchors": b["anchors"]} for b in BANDS_}
}
with open(EXTRA_BASENAME+"_manifest.json","w") as f: json.dump(_manifest, f, indent=2)
print(f"[OK] Manifest → {EXTRA_BASENAME}_manifest.json")

# — Benjamini–Hochberg su p-value derivati da z_base (≈N(0,1)) —
from math import erf, sqrt
def z_to_p(z):  # two-sided
    return 2.0*(1.0 - 0.5*(1+erf(abs(z)/sqrt(2))))
if not base_df.empty:
    df_fdr = base_df[["band","z_base"]].copy()
    df_fdr["p_two"] = df_fdr["z_base"].apply(lambda z: z_to_p(z) if np.isfinite(z) else np.nan)
    df_fdr = df_fdr.dropna().sort_values("p_two")
    m = len(df_fdr); alpha = 0.05
    thresh = None
    for i, (_, row) in enumerate(df_fdr.iterrows(), start=1):
        if row["p_two"] <= alpha*i/m: thresh = row["p_two"]
    print("[FDR] BH α=0.05 → p* =", thresh if thresh is not None else "none (no discoveries)")

# ===============================================================
# === LEGEND / INTERPRETATION PAGE (append-only, explanatory) ===
# ===============================================================

import matplotlib.pyplot as plt
from textwrap import fill

legend_text = """
CNT/HNT PIPELINE — INTERPRETATION GUIDE
=======================================
CNT/HNT pipeline v1.0.0
Author: M. Vengher
Purpose: Automated spectral analysis for CNT/HNT feature detection (5–20 µm)
Methods: quick-look, SAFE masks, LSF injection-recovery, core–wide coherence,
template matching (Chen & Li 2019/2020), slope & shape coherence tests.

Tests made to build this pipeline: All objects analyzed with this pipeline are
drawn from the IRSA Spitzer Enhanced Products archive.

This section provides a quick legend explaining the meaning of each diagnostic
table and metric in the CNT/HNT pipeline. It is intended for reviewers,
students, or collaborators to interpret results correctly and responsibly.

------------------------------------------------------------
1. Quick-look table
------------------------------------------------------------
• S/N (signal-to-noise)  — Integrated significance of the deficit (negative) or
  emission (positive) within the defined band.
• √Δχ² (zsig)  — Equivalent Gaussian significance derived from Δχ².
• EW (equivalent width)  — Band strength in μm. Negative = absorption, positive = emission.
• τ̄ (mean optical depth) — Average optical depth over the band.
• verdict  — "YES" indicates a statistically significant feature; "NO" = none.

Interpretation:
    YES (detection) → statistically significant spectral feature.
    YES (robust)    → confirmed by multiple independent checks.
    NO              → no significant feature.
    MARGINAL        → weak, possibly real but below threshold.

------------------------------------------------------------
2. SAFE / water-safe / PAH-safe comparison
------------------------------------------------------------
Tests whether the same feature persists when nearby PAH or water bands are
masked.  "YES (robust)" means the signal survives all safety masks.

<h3>📊 Table A — Dust, Molecules, Nanostructures (CNT, PAH, Fullerenes, Ices, etc.)</h3>

<table>
  <thead>
    <tr><th>Paper (year, authors)</th><th>Species / Feature</th><th>Band (µm / Å / GHz / THz)</th><th>Type</th><th>Notes</th></tr>
  </thead>
  <tbody>
    <tr><td><b>Megías et al. (2025)</b></td><td>H<sub>2</sub>O ice</td><td>≈3, ≈6 µm</td><td>Absorption</td><td>Stretch &amp; bending</td></tr>
    <tr><td></td><td>CO<sub>2</sub> ice</td><td>4.27, 15.2 µm</td><td>Absorption</td><td></td></tr>
    <tr><td></td><td>CO ice</td><td>4.67 µm</td><td>Absorption</td><td></td></tr>
    <tr><td></td><td>CH<sub>4</sub> ice</td><td>7.7 µm</td><td>Absorption</td><td></td></tr>
    <tr><td></td><td>NH<sub>3</sub> ice</td><td>2.96 µm</td><td>Absorption</td><td></td></tr>
    <tr><td></td><td>CH<sub>3</sub>OH ice</td><td>3.53, 9.7 µm</td><td>Absorption</td><td></td></tr>

    <tr><td><b>Yoon et al. (2024)</b> – PRIMA/FIRESS PAH</td><td>PAH (qPAH models)</td><td>3.3, 6.2, 7.7, 11.2 µm</td><td>Emission</td><td>qPAH = 0.5%, 1.8%, 3.8%</td></tr>

    <tr><td><b>Chen &amp; Li (2020)</b> – CNTs in Space</td><td>CNT neutral (NT)</td><td>5.3, 7.0, 8.8, 9.7, 10.9, 14.2, 16.8 µm</td><td>Absorption</td><td>Predicted IR bands</td></tr>
    <tr><td></td><td>CNT<sup>+</sup> (NT<sup>+</sup>)</td><td>5.2, 7.1, 8.3, 9.2, 13.4, 16.7 µm</td><td>Absorption</td><td></td></tr>
    <tr><td></td><td>HNT</td><td>3.3, 6.5, 8.2, 10.8, 13.0, 17.2 µm</td><td>Absorption</td><td>Hydrogenated CNTs</td></tr>
    <tr><td></td><td>HNT<sup>+</sup></td><td>3.3, 6.6, 8.3, 10.6, 12.8, 17.2 µm</td><td>Absorption</td><td>Hydrogenated cationic CNTs</td></tr>
    <tr><td></td><td>C<sub>60</sub></td><td>7.0, 8.45, 17.3, 18.9 µm</td><td>Emission</td><td>Fullerene</td></tr>
    <tr><td></td><td>C<sub>60</sub><sup>+</sup></td><td>6.4, 7.1, 8.2, 10.5 µm</td><td>Emission</td><td>Fullerene cation</td></tr>
    <tr><td></td><td>Graphene C<sub>24</sub></td><td>6.6, 9.8, 20 µm</td><td>Absorption</td><td>Graphene sheet</td></tr>
    <tr><td></td><td>PAH (classic)</td><td>3.3, 6.2, 7.7, 8.6, 11.3, 12.7 µm</td><td>Emission</td><td>Aromatic bands</td></tr>

    <tr><td><b>Shuba et al. (2008)</b> – CNT absorption</td><td>Metallic CNT (zigzag (9,0))</td><td>12.7 µm</td><td>Absorption</td><td>Geometric resonance</td></tr>
    <tr><td></td><td>Metallic CNT</td><td>4.9 µm</td><td>Absorption</td><td></td></tr>
    <tr><td></td><td>Metallic CNT</td><td>3.2 µm</td><td>Absorption</td><td>Antenna-like</td></tr>

    <tr><td><b>Rai &amp; Rastogi (2009)</b> – Nanodiamonds</td><td>Nanodiamond H–C stretch</td><td>3.43, 3.53 µm</td><td>Emission</td><td>Hydrogenated</td></tr>
    <tr><td></td><td>Nanodiamond</td><td>3.47 µm</td><td>Absorption</td><td>Tertiary C–H</td></tr>
    <tr><td></td><td>Nanodiamond/graphite</td><td>2175 Å (0.2175 µm)</td><td>Absorption</td><td>UV bump</td></tr>

    <tr><td><b>Gavdush et al. (2025)</b> – Ice Analogues III</td><td>CO ice</td><td>≈1.5 THz (200 µm)</td><td>Absorption</td><td>Vibrational</td></tr>
    <tr><td></td><td>CO<sub>2</sub> ice</td><td>≈3.5 THz (85 µm)</td><td>Absorption</td><td>Vibrational</td></tr>
    <tr><td></td><td>CO<sub>2</sub> ice</td><td>15–18 THz (~20 µm)</td><td>Absorption</td><td>Sidebands, porosity</td></tr>
  </tbody>
</table>

<h3 style="margin-top:2em">📊 Table B — Atomic and Ionic Lines (JWST/MIRI, PRIMA/FIRESS, etc.)</h3>

<table>
  <thead>
    <tr><th>Paper (year, authors)</th><th>Species / Feature</th><th>Band (µm)</th><th>Type</th><th>Notes</th></tr>
  </thead>
  <tbody>
    <tr><td><b>Kastner et al. (2025)</b> – JWST/NIRCam</td><td>[Fe II]</td><td>1.64 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>H<sub>2</sub></td><td>2.12 µm</td><td>Emission</td><td>Molecular</td></tr>
    <tr><td></td><td>H I (Brα)</td><td>4.05 µm</td><td>Emission</td><td>Hydrogen</td></tr>

    <tr><td><b>Fernández-Ontiveros et al. (2025)</b> – PRIMA/FIRESS</td><td>[S IV]</td><td>10.5 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[Ne III]</td><td>15.6 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[O IV]</td><td>25.9 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[O III]</td><td>52, 88 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[N III]</td><td>57 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[N II]</td><td>122, 205 µm</td><td>Emission</td><td>Ionized</td></tr>

    <tr><td><b>Hermosa Muñoz et al. (2025)</b> – JWST/MIRI (GATOS)</td><td>H<sub>2</sub> 0–0 S(2)</td><td>12.28 µm</td><td>Emission</td><td>Molecular</td></tr>
    <tr><td></td><td>H I (7–6)</td><td>12.37 µm</td><td>Emission</td><td>Hydrogen</td></tr>
    <tr><td></td><td>[Ne II]</td><td>12.81 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[Ar V]</td><td>13.10 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[Mg V]</td><td>13.52 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[Ne V]</td><td>14.32 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[Cl II]</td><td>14.37 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>[Ne III]</td><td>15.56 µm</td><td>Emission</td><td>Ionized</td></tr>
    <tr><td></td><td>H<sub>2</sub> 0–0 S(1)</td><td>17.03 µm</td><td>Emission</td><td>Molecular</td></tr>
    <tr><td></td><td>[Ar VI]</td><td>11.6–11.76 µm</td><td>Emission</td><td>High excitation</td></tr>

    <tr><td><b>Setton et al. (2025)</b> – ALMA SQuIGGŁE</td><td>CO(2–1)</td><td>230.538 GHz (~1.30 mm)</td><td>Emission</td><td>Molecular (ALMA)</td></tr>
  </tbody>
</table>

------------------------------------------------------------
3. Core vs Wide cross-confirmation
------------------------------------------------------------
Each band is measured in two versions (core and wide) to check consistency.
Verdict:
    IT FITS    → same sign and amplitude trend (consistent matching).
    MARGINAL   → slightly inconsistent.
    NO         → opposite sign or incoherent shape.

------------------------------------------------------------
4. LSF and Injection–Recovery tests
------------------------------------------------------------
• width/LSF ≥ 1 → the band is resolved relative to the instrument line spread.
• rec_rate ≥ 0.8 → artificial injections of similar depth are recovered ≥80% of times.
These verify that the signal is neither noise nor instrumental.

------------------------------------------------------------
5. Extra Verification
------------------------------------------------------------
Additional robustness tests:
    – Rebin(1,2): stability across spectral resolutions.
    – Edge ±0.05: shift of integration limits.
    – Jackknife: stability when excluding subsets of points.
    – Poly2 vs Linear: stability against continuum model choice.

All must pass (OK) for a "CONFIRM_STRONG" classification.

------------------------------------------------------------
6. Template Matching (Chen & Li 2022)
------------------------------------------------------------
Observed residuals are compared with synthetic CNT/HNT spectra predictions from
Tao Chen & Aigen Li (2019, A&A, "Synthesizing carbon nanotubes in space").

• z (√Δχ²) > 5 → strong morphological match.
• best_template shows which CNT/HNT family fits best (neutral, cation, HNT...).
• alpha is the scaling factor; positive means absorption, negative emission.

------------------------------------------------------------
7. Slope Check (polarization/slope test)
------------------------------------------------------------
Left vs right continuum slopes (±0.5 μm from band center) are compared.
    CHECK → slope changes significantly (real continuum distortion).
    OK    → consistent with noise.
Slope distortion is characteristic of CNT/HNT absorption geometry.

------------------------------------------------------------
8. Shape Coherence (spectral correlation)
------------------------------------------------------------
Measures normalized correlation r between bands (10.7–12.9–17 µm):
    r ≈ +1 → similar profile shape (same carrier species).
    r ≈ −1 → mirrored/opposite shape (e.g., emission vs absorption).
    |r| < 0.5 → unrelated.

p-value close to zero → correlation statistically significant.

------------------------------------------------------------
9. Interpretation and Caution
------------------------------------------------------------
These diagnostics indicate *consistency with* CNT/HNT-like features,
not direct proof of carbon nanotubes or nanostructures.

Results should be interpreted as *candidates* pending independent verification
through:
    – higher-resolution spectroscopy (e.g., JWST/MIRI),
    – polarization and imaging studies,
    – comparison with laboratory spectra.

No discovery claims are made in this analysis; detections are
presented as statistically guided hypotheses encouraging further study.
"""

# --- Render legend as a scrollable figure (optional save) ---
plt.figure(figsize=(9, 11))
plt.axis("off")
plt.text(0.02, 0.98, legend_text, fontsize=9.5, va="top", family="monospace")
plt.tight_layout()
plt.show()

# Optional export (toggle True if you want files)
SAVE_LEGEND = True
OBJ = OBJECT_ID.replace(" ", "_") if "OBJECT_ID" in globals() else "Unknown_Object"
if SAVE_LEGEND:
    with open(f"{OBJ}_legend.txt", "w") as f:
        f.write(legend_text)
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    pdf = canvas.Canvas(f"{OBJ}_legend.pdf", pagesize=letter)
    textobject = pdf.beginText(40, 760)
    for line in legend_text.splitlines():
        textobject.textLine(line)
    pdf.drawText(textobject)
    pdf.save()
    print(f"[OK] Legend saved → {OBJ}_legend.txt and {OBJ}_legend.pdf")
