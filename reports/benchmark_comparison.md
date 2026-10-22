# Benchmark comparison status

## Results from this repository

| Benchmark | Protocol | QWK | Other metrics | Status |
|---|---|---:|---|---|
| ASAP 1.0 | Prompt-specific score-stratified 5-fold CV, 60/20/20 within each prompt; seed 42 | Not run | Per-prompt results unavailable | Kaggle ZIP is now available locally and prep path is implemented; full 40-run sweep not run because this environment has no GPU |
| ASAP 2.0 | Official 17,307 train / 7,421 test; 15% seeded validation from train; seeds 42, 43, 44 | 0.7843 ± 0.0016 | Pearson 0.7882 ± 0.0012; Spearman 0.7997 ± 0.0008; MAE 0.3774 ± 0.0071; RMSE 0.6491 ± 0.0076; exact accuracy 0.6435 ± 0.0059 | User-supplied completed Kaggle run logs; not rerun locally |

ASAP 2.0 per-seed test QWKs were 0.7843 (seed 42), 0.7862 (seed 43), and
0.7824 (seed 44). These are overall test results, not prompt-wise values.
The log supplied in the task did not include per-prompt metrics or predictions,
and the Kaggle result files were not present in this checkout; therefore no
per-prompt ASAP 2.0 QWK is claimed here. The three runs did finish and report
7421 test predictions each. The notebook's final papermill failure occurred
afterward in a later cell containing a bare `|` (`SyntaxError`), so it did not
invalidate those completed run logs.

The supplied ASAP 2.0 run emitted PyTorch's scheduler-before-optimizer
warning. The code now advances the learning-rate schedule only when GradScaler
actually updates weights, and handles partial gradient-accumulation groups.
Since that changes training-step scheduling, the supplied scores are retained
as historical run results and must be rerun with the corrected loop for a
strict reproduction of the current repository. Only CPU is available in the
local environment, so full fine-tuning was not repeated here.

## ASAP 1.0 data and protocol

The adapter accepts the standard `essay_set`, `essay`, and `domain1_score`
fields and covers prompt IDs 1–8. Native score ranges are P1 2–12, P2 1–6,
P3/P4 0–3, P5/P6 0–4, P7 0–30, and P8 0–60. These bounds are recorded by
the ASAP 1.0 literature and encoded in `src/asap1.py`. To retain the current
six-class scoring head, scores map to six evenly spaced ordinal bins and
predictions are decoded to native score bounds before calculating metrics.
This is lossy for prompts whose native score scale has more than six values;
QWK remains informative but direct comparison to native-scale systems is
qualified, and exact-score accuracy is particularly constrained.

The implemented protocol is an explicit, prompt-specific five-fold
score-stratified CV: each prompt is split into 60% train, 20% validation, 20%
test, with deterministic per-score shuffling from the configured seed. Fold
`i` is test, `(i+1) mod 5` validation, and the remaining folds train. This
mirrors the five-fold 60/20/20 proportions reported by R²BERT and the
multi-scale BERT paper, but does not claim to reproduce their exact fold
indices. It is not equivalent to the historical Kaggle random train/validation/test
split. The context adapter requires source prompt/rubric text from the
original materials and rejects absent context rather than fabricating it.

The original Kaggle ZIP was subsequently found in the workspace. It contains
12,976 labeled training essays across all eight prompts and original prompt /
rubric descriptions. Kaggle's `valid_set.tsv` (4,218 rows) and `test_set.tsv`
(4,254 rows) contain essays and prediction IDs, but no gold scores; they are
not evaluation splits. The local preparation command is
`python scripts/prepare_asap1.py`. The full 8-prompt × 5-fold ASAP 1.0 run has
not been performed because only CPU is available locally; its command is
`python scripts/run_asap1.py --config configs/asap1_cross_attention.yaml --seed 42`.
The ZIP and extracted corpus stay untracked: Kaggle labels the data “Subject
to Competition Rules,” so this repository does not redistribute it to users
who may not have accepted those rules.

## Published-method comparison

| Method | Benchmark / evaluation | QWK reported | Comparability |
|---|---|---|---|
| R²BERT (Yang et al., 2020) | ASAP 1.0, all eight prompts; prompt-specific five-fold 60/20/20; holistic | Per-prompt: .817, .719, .698, .845, .841, .847, .839, .744; macro .794 | Closest reference. Same benchmark, task, prompts and fold proportions; exact fold assignment and our six-bin mapping differ. See [paper, Table 2](https://aclanthology.org/2020.findings-emnlp.141.pdf). |
| BERT-DOC-TOK-SEG (Wang et al., 2022) | ASAP 1.0, all eight prompts; prompt-specific five-fold 60/20/20; holistic | Per-prompt: .836, .695, .700, .815, .812, .816, .838, .744; macro .782 | Close reference with the same caveats on exact folds and score mapping. See [paper, Table 2](https://aclanthology.org/2022.naacl-main.249.pdf). |
| GAPS (Do et al., 2025) | Combined ASAP/ASAP++; cross-prompt leave-one-prompt-out; trait-level plus “Overall” label | “Overall” mean .670; full model mean trait QWK .591 | Not directly comparable: different corpus composition, held-out-prompt protocol, and mixed trait/overall reporting. See [paper](https://aclanthology.org/2025.findings-naacl.153/). |
| TRATES (Eltanbouly et al., 2025) | Combined ASAP/ASAP++; cross-prompt; analytic traits | ASAP mean across traits: Starling .595, Llama .566, Gemma .586; holistic scoring not studied | Not comparable to holistic prompt-specific results. See [paper](https://aclanthology.org/2025.findings-acl.1054/). |
| ASAP 2.0 corpus paper (Crossley et al., 2025) | ASAP 2.0, official corpus partition; holistic labels | No model QWK reported | Dataset source, not a model result. No independently verified published model result with a matching official split was identified in this task. See [corpus paper](https://doi.org/10.1016/j.asw.2025.100954). |

Published ASAP 1.0 model scores are from the cited papers' tables. The R²BERT
paper describes its split as the established five-fold 60/20/20 setup; the
NAACL BERT paper gives the same proportions. The ASAP 1.0 original competition
also used prompt-specific random partitions, but those should not be
conflated with this implementation's generated folds. Cross-dataset values
are not ranked against one another. Full source and protocol fields are in
[`literature_comparison.csv`](literature_comparison.csv).
