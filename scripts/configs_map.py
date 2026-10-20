"""Central map of experiment keys to result directories (shared by scripts)."""

OUTPUT_DIRS = {
    "tfidf": "01_tfidf_baseline",
    "transformer": "02_transformer_qra_concat",
    "transformer_long": "02b_transformer_qra_concat_long",
    "cross_attention": "03_cross_attention",
    "cross_attention_matched": "03b_cross_attention_matched",
    "cross_attention_coral": "04_cross_attention_coral",
    "cross_attention_expected": "04b_cross_attention_expected",
    "cross_attention_source": "05_cross_attention_plus_source",
    "essay_only": "abl_01_essay_only",
    "qa_concat": "abl_02_qa_concat",
    "qra_concat": "abl_03_qra_concat",
    "qra_cross": "abl_04_qra_cross",
    "qra_cross_coral": "abl_05_qra_cross_coral",
}

ABLATION_ROWS = [
    ("essay_only", "Essay only"),
    ("qa_concat", "Question + answer"),
    ("qra_concat", "Question + rubric + answer (concat)"),
    ("qra_cross", "Question + rubric + answer (cross-attention)"),
    ("qra_cross_coral", "Cross-attention + ordinal (CORAL)"),
    ("cross_attention_matched", "Cross-attention, matched token budget"),
]
