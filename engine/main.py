from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import json, time, platform, subprocess, os, shutil, re
from datetime import datetime
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.shim import run_CNT_engine_blackbox
try:
    from engine.irsa_fetch import fetch_irsa_to_tbl  # optional
except Exception:
    fetch_irsa_to_tbl = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
BASE_DIR     = Path(__file__).parent.resolve()
RUN_DIR      = (BASE_DIR / "run").resolve();       RUN_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR  = (BASE_DIR / "archive").resolve();   ARCHIVE_DIR.mkdir(exist_ok=True)
RAW_DIR      = (BASE_DIR / "engine" / "raw").resolve()   # dove a volte l'engine butta roba
FRONTEND_DIR = (BASE_DIR / "frontend").resolve();  FRONTEND_DIR.mkdir(exist_ok=True)
RUN_INPUT    = RUN_DIR / "input.tbl"

APP_VERSION = "1.0.2"

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI APP
app = FastAPI(title="Carbonette API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servi i file statici del frontend su /app/
# html=True fa servire automaticamente index.html quando navighi /app/
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="app")

# Root → redirect alla UI
@app.get("/")
def root():
    return RedirectResponse(url="/app/")

# /app senza slash → aggiungi slash, così i link assoluti /app/ funzionano sempre
@app.get("/app")
def app_root():
    return RedirectResponse(url="/app/")

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
class SourceCfg(BaseModel):
    catalog: Optional[str] = None
    wave_col: Optional[str] = None
    flux_col: Optional[str] = None
    err_col: Optional[str] = None
    unit_wave: Optional[str] = "micron"
    unit_flux: Optional[str] = "Jy"

class RunInput(BaseModel):
    object: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = 20.0
    options: Optional[Dict[str, Any]] = None  # options.clean=True mantiene input_*.tbl
    spectrum: Optional[Dict[str, list]] = None
    source: Optional[SourceCfg] = None

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(BASE_DIR), timeout=1
        ).decode("utf-8").strip()
    except Exception:
        return None

def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]", "", s)
    return s or "unknown"

def _format_ts(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(ts or time.time())
    return dt.strftime("%Y%m%d_%H%M%S")

def _validate_payload(p: Dict[str, Any]) -> Optional[str]:
    ra, dec = p.get("ra"), p.get("dec")
    rad = p.get("radius_arcsec")
    name = (p.get("object") or "").strip()

    if rad is not None:
        try:
            r = float(rad)
            if r <= 0 or r > 120:
                return "Radius must be in (0, 120] arcsec."
        except Exception:
            return "Radius is not a number."

    if (ra is not None) ^ (dec is not None):
        return "Both RA and DEC are required (or provide an object name)."

    if ra is not None and dec is not None:
        try:
            ra = float(ra); dec = float(dec)
        except Exception:
            return "RA/DEC must be numbers."
        if not (0.0 <= ra < 360.0):
            return "RA out of range (0 ≤ RA < 360)."
        if not (-90.0 <= dec <= 90.0):
            return "DEC out of range (-90 ≤ DEC ≤ +90)."

    if not name and (ra is None or dec is None):
        return "Provide an object name or RA/DEC."
    return None

def _provenance_line(manifest: Dict[str, Any] | None, source_name: str | None) -> str:
    if manifest and source_name and source_name in manifest:
        m = manifest.get(source_name, {})
        aor = m.get("aor", "n/a")
        module = m.get("module", "n/a")
        extr = m.get("extraction", "n/a")
        qual = m.get("quality", "n/a")
        return f"Source: Spitzer IRSA Enhanced Products • AOR={aor} • Module={module} • Extract={extr} • Quality={qual}"
    return "Source: Spitzer IRSA Enhanced Products • details: n/a"

def _load_manifest() -> Optional[Dict[str, Any]]:
    mf = RUN_DIR / "input_manifest.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

ALLOWED_EXT = {".pdf", ".png", ".csv", ".json", ".html", ".tex", ".txt"}

def _clean_before_run(keep_inputs: bool = True) -> Dict[str, Any]:
    """Pulisce RUN_DIR dagli output (PDF/PNG/CSV/HTML/TEX/TXT/JSON) di run precedenti."""
    kept, removed, errors = [], [], []
    keep_names = {"input_manifest.json"}

    def _keep(p: Path) -> bool:
        if not keep_inputs:
            return False
        if p.name in keep_names: return True
        if p.name.startswith("input") and p.suffix.lower() == ".tbl": return True
        return False

    for p in RUN_DIR.iterdir():
        try:
            if p.is_dir(): kept.append(p.name); continue
            if _keep(p):   kept.append(p.name); continue
            p.unlink(missing_ok=True); removed.append(p.name)
        except Exception as e:
            errors.append(f"{p.name}: {e}")

    print(f"[CLEAN] kept={len(kept)} removed={len(removed)} errors={len(errors)}")
    return {"kept": kept, "removed": removed, "errors": errors}

def _snapshot_outputs() -> Dict[str, float]:
    snap = {}
    for p in RUN_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            try: snap[p.name] = p.stat().st_mtime
            except Exception: pass
    return snap

def _delta_outputs(before: Dict[str, float]) -> List[str]:
    after = _snapshot_outputs()
    newly = []
    for name, mt in after.items():
        if name not in before or after[name] > before.get(name, 0):
            newly.append(name)
    return newly

def _infer_object_label(payload: Dict[str, Any]) -> str:
    name = (payload.get("object") or "").strip()
    if name: return name
    ra, dec = payload.get("ra"), payload.get("dec")
    if ra is not None and dec is not None:
        try: return f"RA{float(ra):.5f}_DEC{float(dec):+.5f}"
        except Exception: pass
    return "unknown"

def _archive_run_subdir(payload: Dict[str, Any]) -> Path:
    label = _infer_object_label(payload)
    slug  = _slugify(label)
    ts    = _format_ts()
    dest  = ARCHIVE_DIR / slug / ts
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def _archive_copy_files(dest: Path, files: List[str]) -> List[str]:
    copied = []
    for name in files:
        src = RUN_DIR / name
        if src.exists() and src.is_file():
            try: shutil.copy2(src, dest / name); copied.append(name)
            except Exception: pass
    return copied

@contextmanager
def _chdir(where: Path):
    """Cambia CWD temporaneamente (per engine che usano percorsi relativi)."""
    cur = Path.cwd()
    try:
        os.chdir(str(where)); yield
    finally:
        os.chdir(str(cur))

def _purge_raw_outputs() -> Dict[str, List[str]]:
    """Pulisce engine/raw/ lasciando solo *.tbl (e directory); rimuove ALLOWED_EXT."""
    if not RAW_DIR.exists():
        return {"removed": [], "kept": []}
    removed, kept = [], []
    for p in RAW_DIR.iterdir():
        try:
            if p.is_dir(): kept.append(p.name); continue
            if p.suffix.lower() == ".tbl": kept.append(p.name); continue
            if p.suffix.lower() in ALLOWED_EXT:
                p.unlink(missing_ok=True); removed.append(p.name)
            else:
                kept.append(p.name)
        except Exception:
            pass
    return {"removed": removed, "kept": kept}

def _collect_from_raw(tag_prefix: str) -> List[str]:
    """Sposta in RUN_DIR eventuali outputs creati per sbaglio in RAW_DIR; li prefissa."""
    if not RAW_DIR.exists(): return []
    moved = []
    for p in RAW_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            dst = RUN_DIR / f"{tag_prefix}__{p.name}"
            try:
                if dst.exists():
                    stem, suf = dst.stem, dst.suffix
                    dst = RUN_DIR / f"{stem}__{int(time.time())}{suf}"
                shutil.move(str(p), str(dst))
                moved.append(dst.name)
            except Exception:
                pass
    return moved

def _safe_in_dir(base: Path, target: Path) -> bool:
    try:
        base = base.resolve(); target = target.resolve()
        return (target == base) or (base in target.parents)
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────
# RUN
@app.post("/run")
def run(run_input: RunInput):
    """
    Processa tutti i candidati input_*.tbl in run/ (escludendo *raw*).
    Ogni candidato viene promosso a input.tbl; gli output vengono taggati come 'input_XXX__*'.
    Copia anche in archive/<object>/<timestamp>/.
    """
    t0 = time.time()
    payload: Dict[str, Any] = run_input.model_dump()
    payload.setdefault("warnings", [])
    opts = payload.get("options") or {}
    notes: list[str] = []

    # Validazione
    err = _validate_payload(payload)
    if err:
        return {"result": {"status": "error", "code": "VALIDATION",
                           "message": err, "elapsed_s": round(time.time() - t0, 3)}}

    # Clean run/ (mantieni input*.tbl)
    if bool(opts.get("clean", True)):
        _ = _clean_before_run(keep_inputs=True); notes.append("clean:ok")
    else:
        notes.append("clean:skip")

    # Pulisci engine/raw
    raw_clean = _purge_raw_outputs()
    notes.append(f"raw_clean:{len(raw_clean.get('removed', []))}")

    # Fetch IRSA (opzionale)
    try:
        has_target = bool(payload.get("object")) or (payload.get("ra") is not None and payload.get("dec") is not None)
        if has_target and fetch_irsa_to_tbl:
            fetch_irsa_to_tbl(
                run_dir=RUN_DIR,
                object_name=(payload.get("object") or None),
                ra=payload.get("ra"), dec=payload.get("dec"),
                radius_arcsec=payload.get("radius_arcsec") or 20.0,
                source=(payload.get("source") or {}),
            ); notes.append("fetch:ok")
        else:
            notes.append("fetch:skip")
    except Exception as e:
        payload["warnings"].append(f"IRSA fetch failed: {e}"); notes.append("fetch:fail")

    # Candidati
    candidates = [p for p in RUN_DIR.glob("input_*.tbl") if "raw" not in p.name.lower()]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)

    manifest = _load_manifest()
    results: List[Dict[str, Any]] = []

    # Esecuzione per ogni candidato
    for cand in candidates:
        tag = cand.stem  # es. input_023

        # Promuovi a run/input.tbl in modo atomico
        tmp = RUN_DIR / (RUN_INPUT.name + ".tmp")
        shutil.copy2(cand, tmp); os.replace(tmp, RUN_INPUT)

        provenance = _provenance_line(manifest, cand.name)
        snap_before = _snapshot_outputs()

        # Esegui engine (cwd=RUN_DIR per evitare scritture altrove)
        try:
            with _chdir(RUN_DIR):
                r = run_CNT_engine_blackbox(dict(payload)) or {}
        except Exception as e:
            r = {"status": "error", "stdout": f"{provenance}\nengine failed: {e}", "files": []}

        # File nuovi in RUN_DIR
        new_files = _delta_outputs(snap_before)
        # E se l'engine ha sputato roba in engine/raw, spostala qui e prefissala
        strays = _collect_from_raw(tag)
        new_files.extend(strays)

        # Prefissa i file creati (se non già prefissati)
        tagged_files: List[str] = []
        for name in new_files:
            if name.startswith(f"{tag}__"):  # già prefissato
                tagged_files.append(name); continue
            src = RUN_DIR / name
            tagged = RUN_DIR / f"{tag}__{name}"
            try:
                os.replace(src, tagged)
                tagged_files.append(tagged.name)
            except Exception:
                try:
                    shutil.copy2(src, tagged); tagged_files.append(tagged.name)
                except Exception:
                    pass

        one_meta = {"input": cand.name, "provenance": provenance, "elapsed_s": None}
        out_item: Dict[str, Any] = {
            "input_file": cand.name,
            "result": {
                "status": (r or {}).get("status", "ok"),
                "files": tagged_files,
                "warnings": (r or {}).get("warnings", []),
                "stdout": f"{provenance}\n{(r or {}).get('stdout','')}".strip(),
            },
            "meta": one_meta,
        }

        # Archivio
        try:
            dest = _archive_run_subdir(payload)
            copied = _archive_copy_files(dest, tagged_files)
            run_json = {
                "object_label": _infer_object_label(payload),
                "slug": _slugify(_infer_object_label(payload)),
                "timestamp": dest.name,
                "created_epoch": time.time(),
                "payload": {
                    "object": payload.get("object"),
                    "ra": payload.get("ra"),
                    "dec": payload.get("dec"),
                    "radius_arcsec": payload.get("radius_arcsec"),
                    "source": payload.get("source"),
                },
                "source_name": cand.name,
                "result_meta": {"version": APP_VERSION, "git": _git_hash()},
                "files": copied,
            }
            (dest / "run.json").write_text(json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            out_item["result"]["warnings"] = list(out_item["result"].get("warnings") or []) + [f"archive failed: {e}"]

        results.append(out_item)

    meta = {
        "version": APP_VERSION, "git": _git_hash(),
        "python": platform.python_version(), "os": f"{platform.system()} {platform.release()}",
        "elapsed_s": round(time.time() - t0, 3), "notes": list(dict.fromkeys(notes)), "run_dir": str(RUN_DIR), "count": len(results),
    }

    return {"result": {"status": "ok", "mode": "multi", "count": len(results),
                       "results": results, "warnings": payload.get("warnings", []), "meta": meta}}

# ──────────────────────────────────────────────────────────────────────────────
# HISTORY
@app.get("/history")
def history(limit: int = Query(100, ge=1, le=1000)) -> Dict[str, Any]:
    objects, recent = [], []
    if not ARCHIVE_DIR.exists():
        return {"objects": [], "recent": []}

    for obj_dir in sorted(ARCHIVE_DIR.iterdir()):
        if not obj_dir.is_dir(): continue
        slug = obj_dir.name
        runs = [d for d in obj_dir.iterdir() if d.is_dir()]
        if not runs: continue
        runs_sorted = sorted(runs, key=lambda d: d.name, reverse=True)
        latest = runs_sorted[0]
        label = slug
        meta_path = latest / "run.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                label = meta.get("object_label", slug)
            except Exception:
                pass
        objects.append({"slug": slug, "label": label, "count": len(runs_sorted), "latest_ts": latest.name})

        for d in runs_sorted[:limit]:
            try:
                meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
                recent.append({"slug": slug, "label": meta.get("object_label", slug),
                               "timestamp": d.name, "files": meta.get("files", [])})
            except Exception:
                recent.append({"slug": slug, "label": label, "timestamp": d.name, "files": []})

    recent = sorted(recent, key=lambda r: r["timestamp"], reverse=True)[:limit]
    return {"objects": objects, "recent": recent}

@app.get("/history/{slug}")
def history_object(slug: str) -> Dict[str, Any]:
    obj_dir = (ARCHIVE_DIR / _slugify(slug))
    if not obj_dir.exists() or not obj_dir.is_dir():
        raise HTTPException(status_code=404, detail="Object not found in history")

    runs = []
    for d in sorted([p for p in obj_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
        try:
            meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {"object_label": slug, "timestamp": d.name}
        runs.append({"timestamp": d.name, "meta": meta, "files": meta.get("files", [])})

    label = runs[0]["meta"].get("object_label", slug) if runs else slug
    return {"slug": slug, "label": label, "runs": runs}

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
@app.get("/download_archive")
def download_archive(path: str = Query(..., description="Relative path under archive/, e.g. slug/timestamp/file")):
    rel = Path(path)
    p = (ARCHIVE_DIR / rel).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _safe_in_dir(ARCHIVE_DIR, p):
        raise HTTPException(status_code=403, detail="Access denied")

    suf = p.suffix.lower()
    media_type = None
    headers = {"X-Content-Type-Options": "nosniff"}
    if suf == ".pdf":
        media_type = "application/pdf"
        headers["Content-Disposition"] = f'inline; filename="{p.name}"'
    elif suf == ".html":
        media_type = "text/html; charset=utf-8"
    return FileResponse(str(p), filename=p.name, media_type=media_type, headers=headers)

@app.get("/download")
def download(file: str = Query(..., description="Filename inside run/")):
    safe = Path(file).name
    p = (RUN_DIR / safe).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _safe_in_dir(RUN_DIR, p):
        raise HTTPException(status_code=403, detail="Access denied")

    suf = p.suffix.lower()
    media_type = None
    headers = {"X-Content-Type-Options": "nosniff"}
    if suf == ".pdf":
        media_type = "application/pdf"
        headers["Content-Disposition"] = f'inline; filename="{p.name}"'
    elif suf == ".html":
        media_type = "text/html; charset=utf-8"
    return FileResponse(str(p), filename=p.name, media_type=media_type, headers=headers)
