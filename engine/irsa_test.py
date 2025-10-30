from astroquery.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
import requests, pandas as pd, numpy as np

# 1) parametri
target = "NGC 7023"
candidates = [
    "spitzer.sings_irs_spec",
    "spitzer.goals_irs_spec",
    "spitzer.s5_spectra",
    "spitzer.feps_spectra_v5",
    "spitzer.fivemuses_spectra",
]
radii = [30*u.arcsec, 2*u.arcmin, 5*u.arcmin]

# 2) risolvi coordinate e cerca
coord = SkyCoord.from_name(target)
Irsa.ROW_LIMIT = 200

row = None
cat_used = None
for cat in candidates:
    for rad in radii:
        try:
            t = Irsa.query_region(coord, catalog=cat, spatial="Cone", radius=rad)
            if len(t) == 0: 
                continue
            if "spectrum_download_u" in t.colnames:
                row = t[0]           # prendi la prima riga disponibile
                cat_used = cat
                break
        except Exception as e:
            pass
    if row is not None:
        break

if row is None:
    print("Nessuno spettro trovato vicino a", target, "nei cataloghi provati.")
else:
    print("Trovato in", cat_used, " — riga esempio:")
    print(row)
    url = row["spectrum_download_u"]
    name = row["spectrum_filename"]
    Path("run").mkdir(exist_ok=True)
    local = Path("run")/str(name)
    print("Scarico:", url)
    r = requests.get(url, timeout=60)
    local.write_bytes(r.content)
    print("Salvato:", local, "| Estensione:", local.suffix.lower())

    # 3) prova lettura ASCII; se fallisce, prova FITS
    made = False
    if local.suffix.lower() in [".tbl",".txt",".dat",".csv"]:
        try:
            try:
                df = pd.read_csv(local, comment="#", delim_whitespace=True)
            except Exception:
                df = pd.read_csv(local, comment="#")
            cols = [c.lower() for c in df.columns]
            # heuristica nomi colonne
            def find_one(pats):
                for c in df.columns:
                    cl = c.lower()
                    if any(p in cl for p in pats):
                        return c
                return None
            W = find_one(["wave","lambda","lam","wl","wavelength"])
            F = find_one(["flux","fnu","f_lambda","flambda"])
            E = find_one(["err","unc","sigma","e_flux"])
            if not (W and F and E):
                raise RuntimeError(f"Colonne non riconosciute: {df.columns.tolist()}")
            out = pd.DataFrame({
                "lam": df[W].astype(float),
                "F":   df[F].astype(float),
                "S":   df[E].replace(0, np.nan).astype(float),
                "flag": 0
            }).replace([np.inf,-np.inf], np.nan).dropna()
            out.to_csv("run/input.tbl", sep=" ", header=False, index=False)
            print("Scritto run/input.tbl (ASCII) — righe:", len(out))
            made = True
        except Exception as e:
            print("ASCII parse fallito:", e)

    if not made:
        try:
            from astropy.io import fits
            with fits.open(local) as hdul:
                hdul.info()
                idx = None
                for i,h in enumerate(hdul):
                    if isinstance(h, fits.BinTableHDU):
                        idx = i; break
                if idx is None:
                    raise RuntimeError("Nessuna BINTABLE trovata nel FITS.")
                tab = hdul[idx].data
                # prova alcuni nomi comuni
                candW = [k for k in tab.columns.names if any(s in k.lower() for s in ["lam","wave","wavelength"])]
                candF = [k for k in tab.columns.names if any(s in k.lower() for s in ["flux","fnu","flambda","f_lambda"])]
                candE = [k for k in tab.columns.names if any(s in k.lower() for s in ["err","unc","sigma","e_flux"])]
                if not (candW and candF and candE):
                    raise RuntimeError(f"Nomi colonne non trovati: {tab.columns.names}")
                W,F,E = candW[0], candF[0], candE[0]
                lam = np.array(tab[W], float)
                flux = np.array(tab[F], float)
                err = np.array(tab[E], float)
                df = pd.DataFrame({"lam":lam,"F":flux,"S":err,"flag":0}).replace([np.inf,-np.inf], np.nan).dropna()
                df.to_csv("run/input.tbl", sep=" ", header=False, index=False)
                print("Scritto run/input.tbl (FITS) — righe:", len(df))
                made = True
        except Exception as e:
            print("FITS parse fallito:", e)

    if not made:
        print("Non sono riuscito a costruire run/input.tbl da questo file.")
