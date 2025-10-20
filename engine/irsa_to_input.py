# irsa_to_input.py
from astropy.table import Table
import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(r"C:\Users\Alessio\Desktop\carbonette\run\SPITZER_S5_18940416_01_merge.tbl")
OUT = Path(r"C:\Users\Alessio\Desktop\carbonette\run\input.tbl")

print("Leggo:", SRC)
t = Table.read(str(SRC), format="ascii.ipac")

def pick(keys):
    for n in t.colnames:
        ln = n.lower().replace("(", " ").replace(")", " ")
        if any(k in ln for k in keys):
            return n
    return None

W = pick(["wave", "wavelength", "lam"])
F = pick(["flux_density", "flux", "fnu", "flambda", "f_lambda"])
E = pick(["error", "err", "unc", "sigma"])
FLAG = pick(["flag"])

if not (W and F and E):
    raise RuntimeError(f"Colonne non trovate. Presenti: {t.colnames}")

lam  = np.array(t[W], float)
flux = np.array(t[F], float)
err  = np.array(t[E], float)
flag = np.array(t[FLAG], int) if FLAG else np.zeros(len(t), dtype=int)

df = pd.DataFrame({"lam": lam, "F": flux, "S": err, "flag": flag})
df = df.replace([np.inf, -np.inf], np.nan).dropna()

OUT.write_text("", encoding="utf-8")  # crea/azzera file
df.to_csv(OUT, sep=" ", header=False, index=False, float_format="%.8g")
print(f"Scritto {OUT}  righe={len(df)}  colonne={list(df.columns)}")
