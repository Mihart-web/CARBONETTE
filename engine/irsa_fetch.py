# engine/irsa_fetch.py
from __future__ import annotations

from pathlib import Path
import os, io, re, time, hashlib, requests, math, shutil
from typing import Optional, List, Tuple
from urllib.parse import urlencode
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io.votable import parse_single_table
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

IRSA_SSA = "https://irsa.ipac.caltech.edu/SSA"

BASE_DIR = Path(__file__).resolve().parents[1]   # .../carbonette
RUN_DIR  = (BASE_DIR / "run")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Output:
# - run/input_001.tbl, input_002.tbl, ... (PULITI, 4 colonne: λ F σ flag)
# - run/input_raw_001.tbl, ... (GREZZI originali)
# - run/input_manifest.txt (mapping indice → URL, range λ, righe)
MANIFEST = RUN_DIR / "input_manifest.txt"

def _make_session(total_retries: int = 5, backoff: float = 1.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries, read=total_retries, connect=total_retries,
        backoff_factor=backoff, status_forcelist=(429,500,502,503,504),
        allowed_methods=frozenset(["GET"]), raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter); s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "CarbonetteFetcher/3.0",
        "Cache-Control": "no-cache", "Pragma": "no-cache",
    })
    return s

HTTP = _make_session()

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush()
        try: os.fsync(f.fileno())
        except Exception: pass
    os.replace(tmp, path)

def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))

def _download(url: str, connect_timeout: int = 20, read_timeout: int = 120) -> bytes:
    r = HTTP.get(url, timeout=(connect_timeout, read_timeout))
    r.raise_for_status()
    return r.content

def _ssa_query(ra_deg: float, dec_deg: float, size_deg: float = 0.02) -> bytes:
    params = {
        "COLLECTION": "spitzer_irsenh",
        "POS": f"{ra_deg:.7f},{dec_deg:.7f}",
        "SIZE": f"{size_deg:.6f}",
        "FORMAT": "ALL",
    }
    url = IRSA_SSA + "?" + urlencode(params)
    return _download(url)

def _list_tbl_urls_from_votable(votable_bytes: bytes) -> List[str]:
    with io.BytesIO(votable_bytes) as bio:
        table = parse_single_table(bio).to_table()

    cand_cols = [c for c in table.colnames if c.lower() in (
        "access_url","accessurl","acref","access.reference"
    )]
    urls: List[str] = []
    for col in cand_cols:
        try: urls.extend([str(x) for x in table[col]])
        except Exception: pass

    if not urls:
        url_re = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
        for col in table.colnames:
            try:
                for val in table[col]:
                    urls.extend(url_re.findall(str(val)))
            except Exception: continue

    seen = set()
    urls = [u for u in urls if u.lower().endswith(".tbl") and not (u in seen or seen.add(u))]
    return urls

def _parse_merge_tbl_to_four_cols(text: str) -> List[Tuple[float,float,float,int]]:
    rows: List[Tuple[float,float,float,int]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("\\") or line.startswith("|"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _order = int(parts[0])   # ignorato
            w = float(parts[1]); f = float(parts[2]); e = float(parts[3]); flag = int(parts[4])
        except Exception:
            continue
        if math.isfinite(w) and math.isfinite(f) and math.isfinite(e):
            rows.append((w,f,e,flag))
    # ordina per λ crescente (così ogni spettro è già in ordine)
    rows.sort(key=lambda t: t[0])
    return rows

def _write_clean_table(idx: int, rows: List[Tuple[float,float,float,int]]) -> Path:
    out = RUN_DIR / f"input_{idx:03d}.tbl"
    text = "\n".join(f"{w:.6f} {f:.10g} {e:.10g} {flag:d}" for (w,f,e,flag) in rows) + "\n"
    _atomic_write_text(out, text)
    return out

def _clear_previous_inputs() -> None:
    # pulizia “gentile” dei vecchi input_*.tbl: non tocchiamo altri file
    for p in RUN_DIR.glob("input_*.tbl"):
        try: p.unlink()
        except Exception: pass
    # manifest vecchio
    try: (RUN_DIR / "input_manifest.txt").unlink()
    except Exception: pass

def fetch_irsa_split_tbls_by_pos(ra_deg: float, dec_deg: float, size_arcmin: float = 1.2) -> List[Path]:
    _clear_previous_inputs()

    size_deg = (size_arcmin * u.arcmin).to(u.deg).value
    vot = _ssa_query(float(ra_deg), float(dec_deg), size_deg)
    urls = _list_tbl_urls_from_votable(vot)
    if not urls:
        raise RuntimeError("Nessun .tbl disponibile su IRSA per la query.")

        # Ordina URL per λ_min crescente → tipicamente SL prima, poi LL
    url_info = []
    for url in urls:
        try:
            raw = _download(url)
            text = raw.decode("utf-8", errors="replace")
            rows = _parse_merge_tbl_to_four_cols(text)
            if not rows:
                continue
            wmin, wmax = rows[0][0], rows[-1][0]
            url_info.append((url, raw, rows, wmin, wmax))
        except Exception:
            continue

    if not url_info:
        raise RuntimeError("Tutti i .tbl scaricati sono vuoti/non parsabili.")

    url_info.sort(key=lambda t: t[3])  # wmin

    manifest_lines = []
    outputs: List[Path] = []

    for i, (url, raw, rows, wmin, wmax) in enumerate(url_info, start=1):
        raw_path = RUN_DIR / f"input_raw_{i:03d}.tbl"
        _atomic_write_bytes(raw_path, raw)
        out_clean = _write_clean_table(i, rows)
        md5_raw = hashlib.md5(raw).hexdigest()
        manifest_lines.append(
            f"{out_clean.name}\t{url}\trows={len(rows)}\trange={wmin:.2f}-{wmax:.2f} µm\tmd5raw={md5_raw}"
        )
        print(f"[FETCHER] WROTE {out_clean.name}: rows={len(rows)} λ={wmin:.2f}–{wmax:.2f} µm  (src: {raw_path.name})")
        outputs.append(out_clean)
    (RUN_DIR / "input_manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return outputs
    
def fetch_irsa_split_tbls_by_name(object_name: str, size_arcmin: float = 1.2) -> List[Path]:
    if not object_name or not str(object_name).strip():
        raise ValueError("object_name mancante o vuoto.")
    last_err = None
    for _ in range(3):
        try:
            c = SkyCoord.from_name(object_name)
            break
        except Exception as e:
            last_err = e; time.sleep(1.5)
    else:
        raise RuntimeError(f"Risoluzione nome fallita per '{object_name}': {last_err}")
    return fetch_irsa_split_tbls_by_pos(c.ra.deg, c.dec.deg, size_arcmin=size_arcmin)

# Compat: main.py può continuare a importare questa
def fetch_irsa_to_tbl(
    run_dir: Path,
    object_name: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    radius_arcsec: float = 72.0,
    source: Optional[dict] = None,
) -> List[Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    size_arcmin = float(radius_arcsec)/60.0 if radius_arcsec is not None else 1.2
    if object_name and str(object_name).strip():
        return fetch_irsa_split_tbls_by_name(object_name, size_arcmin=size_arcmin)
    if ra is not None and dec is not None:
        return fetch_irsa_split_tbls_by_pos(float(ra), float(dec), size_arcmin=size_arcmin)
    raise ValueError("Serve 'object_name' oppure 'ra' e 'dec'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m engine.irsa_fetch \"OBJECT NAME\" [size_arcmin]")
        sys.exit(1)
    name = sys.argv[1]
    size = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.2
    fetch_irsa_split_tbls_by_name(name, size_arcmin=size)
