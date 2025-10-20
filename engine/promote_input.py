from pathlib import Path
import shutil, os, tempfile, hashlib, sys

RUN = Path(r"C:\Users\Alessio\Desktop\carbonette\run")

def newest_clean_input(run_dir: Path) -> Path | None:
    cands = [p for p in run_dir.glob("input_*.tbl") if "raw" not in p.name.lower()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def atomic_replace(src: Path, dst: Path):
    # copia in un file temp nella STESSA cartella e poi replace (atomico)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent)) as tf:
        with src.open("rb") as f:
            shutil.copyfileobj(f, tf)
        tf.flush()
        os.fsync(tf.fileno())
        tempname = Path(tf.name)
    os.replace(str(tempname), str(dst))

def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    run = RUN
    run.mkdir(parents=True, exist_ok=True)
    dst = run / "input.tbl"

    # Se l'utente passa un nome specifico: python promote_input.py input_002.tbl
    src = None
    if len(sys.argv) > 1:
        cand = run / sys.argv[1]
        if cand.exists() and "raw" not in cand.name.lower():
            src = cand
        else:
            print(f"[ERRORE] File specificato non valido: {cand}")
            sys.exit(2)
    else:
        src = newest_clean_input(run)

    if not src:
        print("[STOP] Nessun input_*.tbl (clean) trovato in run/. Fai prima il fetch IRSA.")
        sys.exit(1)

    src_md5 = md5sum(src)
    print(f"[PROMOTE] {src.name}  ->  {dst.name}")
    print(f"[INFO] sorgente size={src.stat().st_size}  md5={src_md5}")

    atomic_replace(src, dst)

    dst_md5 = md5sum(dst)
    print(f"[OK]    scritto {dst}  size={dst.stat().st_size}  md5={dst_md5}")
    if dst_md5 != src_md5:
        print("[WARN] md5 differente tra sorgente e destinazione! (controlla filesystem)")
        sys.exit(3)

if __name__ == "__main__":
    main()
