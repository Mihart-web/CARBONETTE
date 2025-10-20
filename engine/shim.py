import os, re, tempfile, json, subprocess, sys, shutil
from pathlib import Path
from typing import Any, Dict

ENGINE_DIR = Path(__file__).resolve().parent             # .../engine
PROJECT_DIR = ENGINE_DIR.parent                          # .../carbonette
RUN_DIR = (PROJECT_DIR / "run").resolve()                # .../carbonette/run
RAW_DIR = (ENGINE_DIR / "raw").resolve()                 # .../engine/raw

# --- util: rimuovi/neutralizza DATA hardcoded nell'engine ---
def _neutralize_hardcoded_DATA(src_text: str) -> str:
    """
    Tenta di neutralizzare assegnazioni a DATA (soprattutto con triple-quoted).
    Sostituisce con un placeholder che poi rimpiazziamo con l'injection.
    """
    # pattern per DATA = """ ... """  oppure ''' ... '''
    pat_triple = re.compile(
        r'(?ms)^\s*DATA\s*=\s*(?:[rRuUbBfF]+)?\s*(?P<q>"""|\'\'\')(?P<body>.*?)(?P=q)\s*',
        re.UNICODE,
    )
    src_text = pat_triple.sub('DATA = __CARBONETTE_DATA__\n', src_text)

    # pattern per DATA = "..." o '...' (singola riga)
    pat_single = re.compile(
        r'(?m)^\s*DATA\s*=\s*(?:[rRuUbBfF]+)?\s*(?P<q>"|\')(?P<body>.*?)(?P=q)\s*$'
    )
    src_text = pat_single.sub('DATA = __CARBONETTE_DATA__', src_text)

    return src_text

def _rename_outputs(obj_id: str):
    """Rinomina i file generati dal motore da 'object_name_*' / 'object name_*' a '<obj_id>_*'."""
    import os
    base = RUN_DIR
    patt = [
        "object_name_*.*",
        "object name_*.*",
    ]
    changed = []
    for pattern in patt:
        for p in base.glob(pattern):
            new = base / p.name.replace("object_name_", f"{obj_id}_").replace("object name_", f"{obj_id}_")
            try:
                if new != p:
                    # sovrascrivi se esiste
                    if new.exists():
                        new.unlink()
                    p.rename(new)
                    changed.append((p.name, new.name))
            except Exception:
                # se rename fallisce su Windows (file lock), prova copy+unlink
                try:
                    import shutil
                    shutil.copy2(p, new)
                    p.unlink(missing_ok=True)
                    changed.append((p.name, new.name))
                except Exception:
                    pass
    return changed

def run_CNT_engine_blackbox(payload: Dict[str, Any]) -> Dict[str, Any]:
    RUN_DIR.mkdir(exist_ok=True)
    (RUN_DIR / "input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # trova il file .py del motore
    src_path = next((p for p in RAW_DIR.glob("*.py")), None)
    if not src_path:
        return {"status": "error", "message": "Nessuno script .py trovato in engine/raw"}

    # leggi input.tbl (se esiste) per prepararci all’iniezione
    injected_data = ""
    tbl_path = RUN_DIR / "input.tbl"
    if tbl_path.exists() and tbl_path.stat().st_size > 0:
        try:
            injected_data = tbl_path.read_text(encoding="utf-8")
        except Exception:
            injected_data = tbl_path.read_text(encoding="latin-1", errors="ignore")

    # ambiente: UTF-8 + backend headless + info per capture/injection
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["SHIM_RUN_DIR"] = str(RUN_DIR)
    env["SHIM_OBJECT_ID"] = (payload.get("object") or "RUN_LOCAL").replace(" ", "_")

    # === header: cattura figure/tabelle + injection (forte) di DATA/OBJECT_ID ===
    capture_inject_header = r'''
# === Carbonette capture + strong DATA inject ===
import os, atexit
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as _pd

_run_dir = os.environ.get("SHIM_RUN_DIR",".")
_obj_id  = os.environ.get("SHIM_OBJECT_ID","RUN_LOCAL").replace(" ","_")

# ---- PDF "lazy": crea solo quando c'è almeno una figura reale ----
_pdf = None
_pdf_path = os.path.join(_run_dir, f"{_obj_id}_extra_plots.pdf")
_seen = set()
def _ensure_pdf():
    global _pdf
    if _pdf is None:
        from matplotlib.backends.backend_pdf import PdfPages
        _pdf = PdfPages(_pdf_path)
def _fig_has_content(fig):
    try:
        return bool(fig.axes)
    except Exception:
        return True
def _save_fig(fig):
    try:
        if not _fig_has_content(fig):
            return
        num = getattr(fig, 'number', None)
        tag = num if num is not None else len(_seen)+1
        if num in _seen:
            return
        _ensure_pdf()
        png = os.path.join(_run_dir, f"{_obj_id}_extra_fig_{tag}.png")
        fig.savefig(png, bbox_inches="tight", dpi=120)
        _pdf.savefig(fig, bbox_inches="tight")
        _seen.add(num)
    except Exception:
        pass
def _save_all_open_figs():
    try:
        for num in plt.get_fignums():
            if num in _seen:
                continue
            fig = plt.figure(num)
            _save_fig(fig)
    except Exception:
        pass
def _patched_show(*a, **k):
    _save_all_open_figs()
    plt.close("all")
plt.show = _patched_show
_orig_close = plt.close
def _patched_close(*a, **k):
    _save_all_open_figs()
    return _orig_close(*a, **k)
plt.close = _patched_close
try:
    import IPython.display as _ipd
    _orig_display = _ipd.display
    def _display(obj=None, *a, **k):
        try:
            if isinstance(obj, _pd.DataFrame):
                obj.to_csv(os.path.join(_run_dir, f"{_obj_id}_extra_table.csv"), index=False)
                with open(os.path.join(_run_dir, f"{_obj_id}_extra_table.html"), "w", encoding="utf-8") as f:
                    f.write(obj.to_html())
            import matplotlib.figure as _mfig, matplotlib.axes as _maxes, matplotlib.artist as _mart
            if isinstance(obj, _mfig.Figure): _save_fig(obj)
            elif isinstance(obj, _maxes.Axes): _save_fig(obj.figure)
            elif isinstance(obj, _mart.Artist) and getattr(obj,'figure',None) is not None: _save_fig(obj.figure)
        except Exception:
            pass
        return _orig_display(obj, *a, **k)
    _ipd.display = _display
except Exception:
    pass
@atexit.register
def _finish_pdf():
    try:
        _save_all_open_figs()
        if _pdf is not None: _pdf.close()
    except Exception:
        pass

# ---- Placeholder per iniezione forte (lo shim lo rimpiazza prima dell'esecuzione) ----
__CARBONETTE_DATA__ = None
# Se l'engine ridefinisce DATA più avanti, noi lo rimetteremo con un hook semplice:
def __carbonette_force_DATA():
    global DATA
    if __CARBONETTE_DATA__:
        DATA = __CARBONETTE_DATA__
        try:
            globals()["OBJECT_ID"] = _obj_id
        except Exception:
            pass
# chiamiamo subito (moduli a livello top)
__carbonette_force_DATA()
'''
    # carica il sorgente e neutralizza DATA hardcoded
    src_code = src_path.read_text(encoding="utf-8", errors="ignore")
    src_code = _neutralize_hardcoded_DATA(src_code)

    # inserisci l'header + sostituisci placeholder con il contenuto del tbl (se c'è)
    if injected_data:
        injected_literal = json.dumps(injected_data)  # stringa python valida (escape sicuri)
    else:
        injected_literal = "None"

    capture_inject_header = capture_inject_header.replace(
        "__CARBONETTE_DATA__ = None",
        f"__CARBONETTE_DATA__ = {injected_literal}"
    )

    # Aggiungi anche un piccolo hook alla fine del file per ri-forzare DATA,
    # nel caso il motore lo abbia riassegnato in fondo.
    footer_hook = r'''
# === Carbonette: ensure external DATA sticks even if engine reassigns ===
try:
    __carbonette_force_DATA()
except Exception:
    pass
'''

    # scrivi file temporaneo con [header + sorgente + footer]
    tmpdir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmpdir.name) / "CNT_engine_run.py"
    tmp_path.write_text(capture_inject_header + src_code + footer_hook, encoding="utf-8")

    # esegui il motore (CWD = engine/raw)
    proc = subprocess.run(
        [sys.executable, str(tmp_path)],
        cwd=RAW_DIR,
        capture_output=True,
        text=True,
        env=env,
    )

    tmpdir.cleanup()

    result: Dict[str, Any] = {"status": "ok", "files": [], "stdout": (proc.stdout or "")[-2000:]}
    if proc.returncode != 0:
        result["status"] = "error"
        result["message"] = "Engine failed"
        result["stderr"] = (proc.stderr or "")[-2000:]
        return result

    # raccogli file prodotti in engine/raw → run/
    for ext in ("*.csv", "*.json", "*.pdf", "*.png", "*.html"):
        for f in RAW_DIR.glob(ext):
            dest = RUN_DIR / f.name
            try:
                shutil.copy2(f, dest)
                result["files"].append(str(dest))
            except Exception as e:
                result.setdefault("warnings", []).append(f"Non copio {f.name}: {e}")

    # includi extra già in run/
    for pattern in ("*_extra_fig_*.png", "*_extra_plots.pdf", "*_extra_table.html", "*_extra_table.csv"):
        for f in RUN_DIR.glob(pattern):
            p = str(f)
            if p not in result["files"]:
                result["files"].append(p)

    # include output.json se presente
    out_json = RUN_DIR / "output.json"
    if out_json.exists():
        try:
            result.update(json.loads(out_json.read_text(encoding="utf-8", errors="ignore")))
        except Exception as e:
            result.setdefault("warnings", []).append(f"output.json non leggibile: {e}")

    return result
