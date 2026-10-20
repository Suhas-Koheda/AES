"""Loader for the official ASAP 2.0 holistic scoring rubric.

The rubric ships as ``asap_scoring_rubric.docx`` in the official repository
(https://github.com/scrosseye/ASAP_2.0).  This module reads the docx directly
(by unzipping ``word/document.xml``) so no extra dependency is required, and
also accepts a plain-text fallback that ``scripts/prepare_data.py`` writes.

Important: ASAP 2.0 provides **one common holistic rubric** used by human
raters for all seven prompts.  There are no per-trait / per-criterion labels
and no rubric variation across essays.  Prompt-specific *assignment* text
exists, and optional source passages are shipped as PDFs.  This module makes
that explicit so downstream code never fabricates a "rubric column".
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

from .utils import PROJECT_ROOT, get_logger

logger = get_logger()

# The six holistic score bands, in descending order.
SCORE_BANDS = (6, 5, 4, 3, 2, 1)

# A human-readable header prepended to the rubric when it is fed to a model.
RUBRIC_HEADER = (
    "Official ASAP 2.0 holistic scoring rubric (scale 1-6). "
    "The distance between adjacent scores is considered equal."
)

# Marker used to separate fields in the serialised model input.
FIELD_MARKERS = {
    "question": "[QUESTION]",
    "rubric": "[RUBRIC]",
    "answer": "[ANSWER]",
    "source": "[SOURCE]",
}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_docx_text(docx_path: str | Path) -> str:
    """Extract paragraph text from a .docx file without python-docx."""
    import xml.etree.ElementTree as ET

    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"rubric docx not found: {docx_path}")

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx_path) as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError(f"{docx_path} is not a valid .docx (missing word/document.xml)")
        xml_bytes = zf.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for para in root.iter(f"{ns}p"):
        text = "".join(node.text or "" for node in para.iter(f"{ns}t"))
        paragraphs.append(text)
    return "\n".join(paragraphs).strip()


def read_rubric_text(rubric_path: str | Path) -> str:
    """Read rubric text from .docx, .txt or .md."""
    rubric_path = Path(rubric_path)
    if not rubric_path.exists():
        raise FileNotFoundError(
            f"rubric not found: {rubric_path}. Run `python scripts/prepare_data.py` "
            "or point data.rubric_path at the official docx/txt."
        )
    suffix = rubric_path.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(rubric_path)
    if suffix in {".txt", ".md", ".text"}:
        return rubric_path.read_text(encoding="utf-8").strip()
    raise ValueError(f"unsupported rubric format: {suffix} (expected .docx/.txt/.md)")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_SCORE_HEADER_RE = re.compile(
    r"SCORE\s+OF\s+([1-6])\s*:\s*", re.IGNORECASE
)


def parse_score_descriptions(rubric_text: str) -> Dict[int, str]:
    """Split the rubric into ``{score: description}`` for scores 1..6.

    The official rubric encodes each band as ``SCORE OF k: ...``.  A missing
    band is an error -- we never invent descriptions.
    """
    normalized = re.sub(r"\s+", " ", rubric_text)
    matches = list(_SCORE_HEADER_RE.finditer(normalized))
    if not matches:
        raise ValueError(
            "could not find any 'SCORE OF k:' headers in the rubric text; "
            "refusing to synthesise rubric descriptions"
        )

    descriptions: Dict[int, str] = {}
    for i, match in enumerate(matches):
        score = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
        descriptions[score] = normalized[start:end].strip()

    missing = [s for s in SCORE_BANDS if s not in descriptions]
    if missing:
        raise ValueError(f"rubric is missing score bands: {missing}")
    return descriptions


def build_rubric_text(descriptions: Dict[int, str]) -> str:
    """Canonical serialisation used as model input (score 6 first)."""
    lines = [RUBRIC_HEADER]
    for score in SCORE_BANDS:
        lines.append(f"SCORE OF {score}: {descriptions[score]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class RubricLoader:
    """Loads, parses and caches the official holistic rubric."""

    def __init__(self, rubric_path: str | Path):
        self.rubric_path = Path(rubric_path)
        self._raw_text: Optional[str] = None
        self._descriptions: Optional[Dict[int, str]] = None

    @property
    def raw_text(self) -> str:
        if self._raw_text is None:
            self._raw_text = read_rubric_text(self.rubric_path)
        return self._raw_text

    @property
    def descriptions(self) -> Dict[int, str]:
        if self._descriptions is None:
            self._descriptions = parse_score_descriptions(self.raw_text)
        return self._descriptions

    @property
    def full_text(self) -> str:
        """The complete rubric as a single string, suitable for model input."""
        return build_rubric_text(self.descriptions)

    def for_prompt(self, prompt_name: str) -> str:
        """ASAP 2.0 uses one common holistic rubric for every prompt.

        ``prompt_name`` is accepted for interface symmetry and logged so that
        any future prompt-specific rubric can be slotted in without changing
        call sites.  We never invent prompt-specific criteria.
        """
        if not prompt_name:
            raise ValueError("prompt_name must be a non-empty string")
        return self.full_text

    def describe(self) -> Dict[str, object]:
        return {
            "path": str(self.rubric_path),
            "source": self.rubric_path.suffix.lower(),
            "score_bands": sorted(self.descriptions),
            "shared_across_prompts": True,
            "num_characters": len(self.full_text),
            "num_words": len(self.full_text.split()),
        }


def default_rubric_path() -> Path:
    """Conventional location written by ``scripts/prepare_data.py``."""
    return PROJECT_ROOT / "data" / "asap2_raw" / "extracted" / "rubric" / "asap_scoring_rubric.txt"


def load_rubric(rubric_path: Optional[str | Path] = None) -> RubricLoader:
    path = Path(rubric_path) if rubric_path else default_rubric_path()
    loader = RubricLoader(path)
    # Force parsing so errors surface early with a clear message.
    _ = loader.full_text
    return loader


def assemble_model_text(
    question: str,
    rubric: str,
    answer: str,
    source: Optional[str] = None,
    include_source: bool = False,
) -> str:
    """Build the ``[QUESTION] ... [RUBRIC] ... [ANSWER] ...`` string.

    Only used for the concatenation baseline / inspection.  The cross-attention
    model receives question+rubric and answer as separate encoder inputs.
    """
    parts = [
        FIELD_MARKERS["question"], question.strip(),
        FIELD_MARKERS["rubric"], rubric.strip(),
    ]
    if include_source and source:
        parts += [FIELD_MARKERS["source"], source.strip()]
    parts += [FIELD_MARKERS["answer"], answer.strip()]
    return " ".join(parts)
