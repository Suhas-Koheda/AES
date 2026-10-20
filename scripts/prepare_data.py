#!/usr/bin/env python
"""Prepare the official ASAP 2.0 data for the pipeline.

Steps
-----
1. (optional) clone https://github.com/scrosseye/ASAP_2.0
2. extract the training archive
3. extract the password-protected test archive (password: ``asap2_test``)
4. extract the source-text archive and convert PDFs to text when possible
5. extract the official holistic rubric (.docx) to a plain-text file
6. verify row counts and report what was produced

The script never downloads a substitute dataset. If the official repository is
unavailable, point ``--repo-dir`` at a local copy of the same files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/scrosseye/ASAP_2.0.git"
TEST_ZIP_PASSWORD = "asap2_test"

TRAIN_ZIP = "ASAP_2_Final_github_train.zip"
TEST_ZIP = "ASAP_2_Final_github_test.zip"
SOURCE_ZIP = "ASAP_2_source_texts.zip"
RUBRIC_DOCX = "asap_scoring_rubric.docx"

EXPECTED_TRAIN_ROWS = 17307
EXPECTED_TEST_ROWS = 7421


def log(message: str) -> None:
    print(f"[prepare_data] {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def clone_repo(repo_dir: Path, force: bool = False) -> None:
    if repo_dir.exists() and (repo_dir / RUBRIC_DOCX).exists():
        log(f"repository already present at {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists() and force:
        shutil.rmtree(repo_dir)
    if not shutil.which("git"):
        raise RuntimeError("git is required to clone the official repository")
    run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])


def unzip(archive: Path, dest: Path, password: str | None = None) -> None:
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    dest.mkdir(parents=True, exist_ok=True)

    if password:
        # Prefer unzip -P; fall back to 7z.
        if shutil.which("unzip"):
            result = subprocess.run(
                ["unzip", "-o", "-P", password, "-q", str(archive), "-d", str(dest)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return
            log(f"unzip failed ({result.stderr.strip()}); trying 7z")
        if shutil.which("7z"):
            run(["7z", "x", f"-p{password}", "-y", f"-o{dest}", str(archive)],
                stdout=subprocess.DEVNULL)
            return
        raise RuntimeError(
            "could not decrypt the official test archive; install `unzip` or `7z`"
        )

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def convert_pdfs(pdf_dir: Path, txt_dir: Path) -> dict:
    """Convert source PDFs to text; report which ones yielded useful text."""
    txt_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    if not pdf_dir.exists():
        return results
    if not shutil.which("pdftotext"):
        log("pdftotext not found; skipping PDF conversion (install poppler-utils)")
        return results
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        out = txt_dir / f"{pdf.stem}.txt"
        subprocess.run(["pdftotext", "-q", str(pdf), str(out)], check=False)
        text = out.read_text(encoding="utf-8", errors="ignore").strip() if out.exists() else ""
        results[pdf.name] = {
            "characters": len(text),
            "useful": len(text) > 500,
        }
    return results


def extract_rubric(docx_path: Path, out_path: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.rubric_loader import extract_docx_text

    text = extract_docx_text(docx_path)
    if len(text) < 200:
        raise ValueError("extracted rubric is suspiciously short; aborting")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    return len(text)


def verify(csv_path: Path, name: str, expected: int) -> int:
    import pandas as pd

    df = pd.read_csv(csv_path)
    n = len(df)
    log(f"{name}: {n} rows, columns={list(df.columns)}")
    if expected and n != expected:
        log(f"WARNING: expected {expected} rows for {name}, found {n}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default="data/asap2_raw", help="where the official repo lives")
    parser.add_argument("--download", action="store_true", help="clone the official repository if missing")
    parser.add_argument("--force", action="store_true", help="re-extract even if outputs exist")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    extracted = repo_dir / "extracted"
    train_csv = extracted / "train" / "ASAP_2_Final_github_train.csv"
    test_csv = extracted / "test" / "ASAP_2_Final_github_test.csv"
    rubric_txt = extracted / "rubric" / "asap_scoring_rubric.txt"
    source_txt = extracted / "source_texts_txt"

    if args.download or not repo_dir.exists():
        try:
            clone_repo(repo_dir, force=args.force)
        except Exception as exc:  # noqa: BLE001
            log(f"could not clone official repository: {exc}")
            log("Provide the official files locally with --repo-dir and re-run.")
            raise SystemExit(1)

    if args.force or not train_csv.exists():
        unzip(repo_dir / TRAIN_ZIP, extracted / "train")
    else:
        log(f"training CSV already present: {train_csv}")

    if args.force or not test_csv.exists():
        unzip(repo_dir / TEST_ZIP, extracted / "test", password=TEST_ZIP_PASSWORD)
    else:
        log(f"test CSV already present: {test_csv}")

    if args.force or not any(source_txt.glob("*.txt")):
        unzip(repo_dir / SOURCE_ZIP, extracted / "source_texts")
        pdf_dir = extracted / "source_texts" / "ASAP2_source-texts"
        results = convert_pdfs(pdf_dir, source_txt)
        for name, info in results.items():
            flag = "OK " if info["useful"] else "EMPTY/IMAGE-ONLY"
            log(f"  source PDF {name}: {info['characters']} chars [{flag}]")

    if args.force or not rubric_txt.exists():
        n_chars = extract_rubric(repo_dir / RUBRIC_DOCX, rubric_txt)
        log(f"rubric text: {n_chars} characters -> {rubric_txt}")
    else:
        log(f"rubric already present: {rubric_txt}")

    log("=" * 70)
    verify(train_csv, "train", EXPECTED_TRAIN_ROWS)
    verify(test_csv, "test", EXPECTED_TEST_ROWS)
    log("=" * 70)
    log("Done. Next: python scripts/inspect_dataset.py")


if __name__ == "__main__":
    main()
