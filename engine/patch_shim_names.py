from pathlib import Path
import re

SHIM = Path(r"C:\Users\Alessio\Desktop\carbonette\engine\shim.py")
src = SHIM.read_text(encoding="utf-8")

# Inserisce una funzione _rename_outputs(obj) e la chiama dentro run_CNT_engine_blackbox
if "_rename_outputs(" not in src:
    add_fun = r'''
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
'''
    # inserisci la funzione prima di run_CNT_engine_blackbox
    src = re.sub(r'(\ndef run_CNT_engine_blackbox\()',
                 add_fun + r'\1', src, count=1, flags=re.M)

# Chiama il rename alla fine di run_CNT_engine_blackbox, prima del return
if "_rename_outputs(" not in src.split("def run_CNT_engine_blackbox",1)[1]:
    src = re.sub(
        r'(return\s*\{\s*\"status\"\s*:\s*\"ok\".*?\"files\"\s*:\s*)(files)(.*?\})',
        r'_ = _rename_outputs(os.environ.get("SHIM_OBJECT_ID","RUN_LOCAL").replace(" ","_"))\n    \1 files \3',
        src,
        count=1,
        flags=re.S
    )

SHIM.write_text(src, encoding="utf-8")
print("[DONE] shim patch applicata: rinomina 'object_name_*' → '<OGGETTO>_*'")
