#!/usr/bin/env python
"""Prepare local ASAP 1.0 files from the Kaggle download ZIP.

The source archive is kept local and ignored by Git. This utility extracts
only the labeled training TSV and source prompt/rubric descriptions needed by
the adapter; it does not publish or copy competition data into tracked files.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _docx_text(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as docx:
        root = ET.fromstring(docx.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras = []
    for paragraph in root.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
        if text.strip():
            paras.append(text.strip())
    return "\n".join(paras)


def prepare(archive: Path, output_dir: Path) -> tuple[Path, Path]:
    if not archive.is_file():
        raise FileNotFoundError(f"ASAP 1.0 Kaggle ZIP not found: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.tsv"
    contexts_path = output_dir / "prompt_contexts.json"

    with zipfile.ZipFile(archive) as outer:
        train_path.write_bytes(outer.read("training_set_rel3.tsv"))
        descriptions_payload = outer.read("Essay_Set_Descriptions.zip")

    contexts = {}
    with zipfile.ZipFile(io.BytesIO(descriptions_payload)) as descriptions:
        for prompt_id in range(1, 9):
            matches = [
                name for name in descriptions.namelist()
                if re.search(rf"Essay Set #{prompt_id}--ReadMeFirst\.docx$", name)
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one original description for prompt {prompt_id}")
            contexts[str(prompt_id)] = _docx_text(descriptions.read(matches[0]))

    contexts_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared local labeled essays: {train_path}")
    print(f"Prepared source prompt/rubric text: {contexts_path}")
    print("The validation/test TSVs contain no gold labels and are not used as labeled evaluation data.")
    return train_path, contexts_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "asap1_kaggle.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/asap1")
    args = parser.parse_args()
    prepare(args.archive, args.output_dir)


if __name__ == "__main__":
    main()
