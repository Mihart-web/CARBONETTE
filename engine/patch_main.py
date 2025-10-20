from pathlib import Path
import re

MAIN = Path(r"C:\Users\Alessio\Desktop\carbonette\main.py")
src = MAIN.read_text(encoding="utf-8")

changed = False

# --- Patch 1: evita len(None) dopo fetch ---
pat1 = "created_files = _fetch_all_spectra(run_input)"
rep1 = "created_files = _fetch_all_spectra(run_input) or []"
if pat1 in src and rep1 not in src:
    src = src.replace(pat1, rep1, 1)
    print("[OK] fetch -> or []")
    changed = True
else:
    print("[=] fetch già patchato o non trovato")

# --- Patch 2: usa shutil.copy2 al posto di _atomic_copy(...) ---
r_copy = re.compile(r"_atomic_copy\s*\(\s*p_clean\s*,\s*RUN_INPUT\s*\)")
if r_copy.search(src):
    src = r_copy.sub("import shutil; shutil.copy2(p_clean, RUN_INPUT)", src, count=1)
    print("[OK] _atomic_copy -> shutil.copy2")
    changed = True
else:
    print("[=] niente _atomic_copy (già patchato?)")

# --- Patch 3: ritorna SEMPRE results[] anche in single ---
r_single = re.compile(
    r"result\s*=\s*run_CNT_engine_blackbox\(payload\)\s*return\s*\{\s*\"mode\"\s*:\s*\"single\".*?payload\[\"warnings\"\]\s*\}",
    re.S,
)
single_block = (
    'result = run_CNT_engine_blackbox(payload)\n'
    'return {"mode":"single","count":1,"results":[{"input_file":"input.tbl","range":None,"rows":None,"src_url":None,"result": result}],"warnings": payload["warnings"]}'
)
if r_single.search(src):
    src = r_single.sub(single_block, src, count=1)
    print("[OK] single -> results[]")
    changed = True
else:
    print("[=] blocco single già patchato o diverso")

# --- Patch 4: ignora i RAW nel ramo multi ---
r_multi = re.compile(r'multi_inputs\s*=\s*sorted\(RUN_DIR\.glob\("input_\*\.tbl"\)\)')
if r_multi.search(src):
    src = r_multi.sub('multi_inputs = sorted([p for p in RUN_DIR.glob("input_*.tbl") if "raw" not in p.name.lower()])', src, count=1)
    print("[OK] filtro RAW nel multi")
    changed = True
else:
    print("[=] filtro RAW già presente o pattern non trovato")

# --- Patch 5: aggiungi /list per avere un listing via glob ---
if "/list" not in src:
    add = '''
from fastapi import Query

@app.get("/list")
def list_files(glob: str = Query("*.pdf")) -> dict:
    import glob as _g, os
    base = str(RUN_DIR)
    files = sorted(_g.glob(os.path.join(base, glob)))
    from pathlib import Path as _P
    return {"base": base, "glob": glob, "count": len(files), "files": [_P(f).name for f in files]}
'''
    src = src + add
    print("[OK] aggiunta route /list")
    changed = True
else:
    print("[=] /list già presente")

if changed:
    MAIN.write_text(src, encoding="utf-8")
    print("[DONE] main.py aggiornato")
else:
    print("[=] Nessuna modifica necessaria")
