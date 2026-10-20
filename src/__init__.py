"""Rubric-conditioned Automated Essay Scoring (ASAP 2.0).

Package layout:

- ``labels``        : ASAP 1-6 <-> internal 0-5 conversion and score decoding.
- ``metrics``       : QWK, MAE, RMSE, accuracy, Spearman, confusion matrix.
- ``preprocessing`` : text cleaning, truncation helpers, rubric/question assembly.
- ``rubric_loader`` : load the official holistic rubric (docx or txt).
- ``data``          : dataset loading, official/held-out splits, leakage checks.
- ``collator``      : dynamic padding / chunking / input-format collation.
- ``models``        : TF-IDF baseline, Transformer baseline, cross-attention model.
- ``losses``        : CORAL ordinal loss and combined classification+ordinal loss.
- ``train``         : training loop with AMP, early stopping, checkpoint/resume.
- ``evaluate``      : evaluation, prediction CSVs, metric tables.
- ``generalization``: held-out-prompt protocol.
- ``fairness``      : subgroup error analysis.
- ``inspect``       : dataset/schema inspection utilities.
- ``runner``        : config-driven experiment dispatcher.
"""

__version__ = "0.1.0"

from .labels import (
    NUM_CLASSES,
    class_to_score,
    score_to_class,
    decode_softmax,
    decode_expected_score,
)
from .metrics import compute_metrics, quadratic_weighted_kappa

__all__ = [
    "NUM_CLASSES",
    "class_to_score",
    "score_to_class",
    "decode_softmax",
    "decode_expected_score",
    "compute_metrics",
    "quadratic_weighted_kappa",
]
