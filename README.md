# Rubric-Conditioned Neural Automated Essay Scoring (ASAP 2.0)

A reproducible research codebase for testing whether conditioning an essay
representation on the **question** and the **official scoring rubric** improves
holistic essay scoring and cross-prompt generalization, compared with simpler
pretrained-Transformer baselines.

> **Status:** the full pipeline is implemented, unit-tested and ready to train.
> No full neural training run was executed in the development environment
> (CPU-only). Train it on a GPU (e.g. Google Colab) with the commands below;
> result tables and figures are generated automatically. Nothing in this
> repository reports fabricated numbers.

---

## 1. Research question

Given `(question, answer, rubric)`, predict the human holistic score
`1..6`. The central hypothesis is:

> **H1.** Explicitly conditioning essay representations on the question and
> rubric improves scoring accuracy and generalization to unseen prompts,
> compared with (a) essay-only encoders and (b) plain concatenation of
> question + rubric + answer.

This is treated as a hypothesis, not a claim. Five controlled ablations
(essay-only → question+answer → concat → cross-attention → ordinal) are
provided so that question information, rubric information, fusion mechanism and
ordinal learning can each be isolated.

---

## 2. Verified dataset facts (from the released files)

These were checked directly against the official repository, not assumed:

| Property | Official train | Official test |
|---|---|---|
| Rows | **17,307** | **7,421** |
| Prompts | **7** | **5** |
| `set` | `train` (all) | `test` (all) |
| `pubpriv` | `0` (all) | `public` 2,257 / `private` 5,164 |
| Score range | 1–6 | 1–6 |

* Score distribution (train): `1`: 1,252 · `2`: 4,723 · `3`: 6,280 ·
  `4`: 3,926 · `5`: 970 · `6`: 156.
* Test distribution: `1`: 499 · `2`: 2,124 · `3`: 2,741 · `4`: 1,627 ·
  `5`: 386 · `6`: 44.
* **Two prompts appear only in training**: `Car-free cities` (1,959) and
  `Does the electoral college work?` (2,046). The released test file does not
  contain them.
* Essay length (train): median 346 words, p75 454, max 8,944.
* Essay-ID and full-text overlap between train and test: **0**.
* 18 train rows and 10 test rows have an empty `assignment`. Each prompt has
  exactly one assignment text, so those cells are back-filled from the prompt
  before validation (`fill_missing_assignments`); no rows are dropped.
* The rubric is **one common holistic 1–6 rubric** for all seven prompts.
  There are no per-trait labels and no rubric variation across essays.
* Source passages: `FL1`/`FL2` PDFs have a text layer; `IN1`–`IN6` are
  image-only for `pdftotext`, so using them requires OCR.

Because ASAP 2.0 releases no validation partition, the official protocol carves
a stratified validation split out of the **training** partition. The official
test file is never used for model selection.

See [`DATASET.md`](DATASET.md) for the full schema and preprocessing contract.

---

## 3. Models

| ID | Model | Input | Head / objective |
|---|---|---|---|
| **A** | TF-IDF + logistic regression (grid over `C`) | essay | 6-way softmax |
| **B** | Pretrained encoder (`deberta-v3-small`/`base`) | `[QUESTION]…[RUBRIC]…[ANSWER]…` | 6-way softmax |
| **C** | Pretrained encoder + **rubric cross-attention** | answer ⟂ question+rubric | 6-way softmax |
| **D1** | Model C + **CORAL** ordinal loss | as C | 5 ordered logits |
| **D2** | Model C + **CE + expected-score regression** | as C | 6-way softmax |

### Model C

```
Ha = Encoder(answer_chunks)            # (B*C, La, H)
Hc = Encoder([QUESTION] q [RUBRIC] r)  # (B,    Lc, H)   shared weights
C  = MultiHeadCrossAttention(Q=Ha, K=Hc, V=Hc, key_padding_mask=~mask_c)
h  = Ha + sigmoid(W[Ha; C]) * C        # gated residual ("combine: gate")
p  = MaskedAttentionPool(h, mask_a)    # padded tokens excluded
z  = mean over essay chunks (masked)
logits = Dropout -> Linear(z)          # 6 (or 5 for CORAL)
```

Correct masking is enforced in three places: the encoder self-attention mask,
the cross-attention key-padding mask (so padded question/rubric tokens can
never be evidence), and masked pooling over answer tokens. Unit tests assert
that appending padded tokens leaves the logits unchanged.

`combine` can be `residual`, `gate`, or `concat_project`; `pooling` can be
`mean`, `cls`, or `attention`. The encoder may be shared across the two passes
(default) or separate (`share_encoder: false`), documenting the
compute/parameter trade-off.

The context is constant for every essay of a prompt, so cross-attention runs
use a prompt-grouped batch sampler and encode each unique prompt context once
per batch, broadcasting its hidden states to the essays in that batch. This is
equivalent to encoding it per essay (unit-tested) but avoids roughly 8x
redundant context compute. Together with gradient checkpointing and batch 8 it
keeps the full-budget model inside a 16 GB GPU.

---

## 4. Evaluation protocols (kept separate, never merged)

**Evaluation 1 — Official benchmark.** Train on the training partition, select
on a stratified validation split of it, evaluate once on the untouched
official test file. Overall + per-prompt + public/private breakdowns.

**Evaluation 2 — Held-out prompt.** Each prompt is removed from *both*
training and validation, then used only for testing. Rotated over all seven
prompts. Pool = official training partition (because the official test file
lacks two prompts). This experiment never reads official test labels.

**Evaluation 3 — Held-out rubric.** **Not possible with ASAP 2.0**: every essay
uses the same holistic rubric. The code exposes a rubric *loader* but makes no
unseen-rubric claim. This would require a different dataset.

---

## 5. Quickstart on Google Colab (GPU)

```bash
# 1. Clone the repository
!git clone https://github.com/Suhas-Koheda/AES.git
%cd AES

# 2. Install, download & verify the official corpus, inspect it
!bash scripts/colab_setup.sh
```

Then run any experiment:

```bash
# Model A + Model B
!python scripts/run_baselines.py

# Proposed model C
!python scripts/run_from_config.py --config configs/cross_attention.yaml

# Ordinal variants
!python scripts/run_from_config.py --config configs/ordinal_coral.yaml
!python scripts/run_from_config.py --config configs/ordinal_expected.yaml

# Ablations + held-out prompts
!python scripts/run_ablation.py
!python scripts/run_cross_prompt.py --config configs/cross_attention.yaml

# Tables + figures
!python scripts/make_report.py --ablation
```

A ready-to-run notebook is in [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb).
See [`COLAB.md`](COLAB.md) for GPU-specific notes and a timed recipe.

### Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --download
python scripts/inspect_dataset.py --tokenizer microsoft/deberta-v3-small
pytest -q
```

---

## 6. Exact commands per experiment

| Goal | Command |
|---|---|
| TF-IDF baseline | `python scripts/run_baselines.py --only tfidf` |
| Transformer baseline | `python scripts/run_baselines.py --only transformer` |
| Cross-attention | `python scripts/run_from_config.py --config configs/cross_attention.yaml` |
| CORAL ordinal | `python scripts/run_from_config.py --config configs/ordinal_coral.yaml` |
| Expected-score ordinal | `python scripts/run_from_config.py --config configs/ordinal_expected.yaml` |
| Ablation matrix | `python scripts/run_ablation.py` |
| Held-out prompts | `python scripts/run_cross_prompt.py --config configs/cross_attention.yaml` |
| Multi-seed | `python scripts/run_from_config.py --config configs/cross_attention.yaml --seeds 42 43 44` |
| Everything | `python scripts/run_all.py --prepare --seeds 42 43 44` |
| Smoke test (synthetic, no data) | `python scripts/smoke_test.py` |
| Unit tests | `pytest -q` |
| Tables/figures | `python scripts/make_report.py --ablation` |

---

## 7. Repository layout

```
configs/            Experiment configs (inherit from _base.yaml)
src/
  labels.py         score<->class conversion, softmax/CORAL/expected decoding
  metrics.py        QWK (self-contained), MAE, RMSE, accuracy, Spearman
  preprocessing.py  cleaning, head/tail truncation, token-length stats
  rubric_loader.py  official holistic rubric (docx/txt), band parsing
  data.py           loading, schema checks, official & held-out splits, leakage
  collator.py       dynamic padding, chunking, concat/cross input formats
  models.py         TF-IDF, Transformer baseline, cross-attention, CORAL head
  losses.py         CORAL loss, CE + expected-score regression
  train.py          AMP, grad accumulation, early stopping, checkpoint/resume
  evaluate.py       prediction, metrics, per-prompt/group breakdowns
  generalization.py held-out-prompt sweep
  fairness.py       subgroup metrics, distributions, worst errors
  inspect.py        dataset/rubric inspection
  runner.py         config-driven orchestration
scripts/            CLI entry points (see table above)
tests/              unit tests (labels, metrics, data, splits, collator, models, losses)
results/            metrics, predictions, configs, figures (git-ignored contents)
notebooks/          Colab notebook
```

---

## 8. Outputs produced by a run

Each experiment writes to `results/<name>/`:

```
config.json  environment.json  model.json  rubric.json
split_manifest.csv  split_summary.json
train_summary.json  history.csv  history.json
checkpoint_best.pt  checkpoint_epoch_*.pt
metrics_val.json  metrics_test.json
predictions_val.csv  predictions_test.csv
fairness/fairness_report.json
fairness/group_metrics_*.csv
fairness/worst_errors.csv  largest_over_scores.csv  largest_under_scores.csv
summary.json
```

`scripts/make_report.py` aggregates these into
`results/tables/model_comparison.{csv,md}`, `results/tables/ablation.{csv,md}`,
`results/tables/per_prompt_test.csv` and `results/figures/*.png`.

---

## 9. Configuration reference (key fields)

| Key | Meaning |
|---|---|
| `experiment.protocol` | `official` or `heldout_prompt` |
| `experiment.heldout_prompt` | prompt to hold out when `protocol=heldout_prompt` |
| `data.input_mode` | `essay_only`, `qa_concat`, `qra_concat`, `qra_cross` |
| `data.max_answer_length` / `max_context_length` | separate token budgets |
| `data.max_total_length` | total budget for concat modes |
| `data.truncation` | `head`, `tail`, `head_tail` (default preserves conclusions) |
| `data.chunk` / `chunk_size` / `max_chunks` | long-essay chunking (keeps first + last window) |
| `data.include_source` | experimental source-passage conditioning (off by default) |
| `model.name` | `tfidf`, `transformer`, `cross_attention` |
| `model.head_type` | `softmax` or `coral` |
| `model.combine` | `residual`, `gate`, `concat_project` |
| `train.objective` | `ce`, `ce_expected`, `coral` |
| `train.fp16` | mixed precision (auto-disabled on CPU) |
| `train.early_stopping_patience` | epochs without val-QWK improvement |
| `train.max_train_steps` | optional cap (used by the smoke test) |
| `eval.decode_mode` | `argmax` or `expected` (softmax head only) |

---

## 10. Known limitations

1. **No official validation split.** Validation is a stratified 15% carve-out
   of the training partition; results may vary slightly with the seed.
2. **Train/test prompt mismatch.** Two prompts are train-only, and the test
   file has a public/private split. Per-prompt test metrics are reported
   separately; do not average them as if they were a uniform sample.
3. **Concat baseline token budget.** At 512 tokens, concatenation truncates the
   rubric to 192 tokens. `configs/transformer_long.yaml` and
   `configs/cross_attention_matched.yaml` isolate this confound.
4. **One rubric.** ASAP 2.0 supports holistic 1–6 scoring only; no
   criterion-level supervision and no cross-rubric generalization.
5. **Source passages.** Only two of seven have an extractable text layer.
6. **Demographics** are used only for post-hoc subgroup analysis, never as
   model inputs. Small groups are flagged, not hidden.

---

## 11. Related work and novelty

See [`RELATED_WORK.md`](RELATED_WORK.md) for verified summaries and a
comparison table. Short version: rubric-assisted and cross-prompt AES are
active areas (ASAP 2.0; GAPS; TRATES; cross-rubric generalization work), and
cross-attention fusion of prompt/rubric context into an essay encoder is a
composition of familiar components. This project therefore claims **a
controlled empirical comparison**, not a novel architecture. The specific gap
it can address is the lack of a same-split, same-budget ablation isolating
(a) rubric presence, (b) fusion mechanism and (c) ordinal objective on ASAP 2.0.

---

## 12. Licence and citation

Code: MIT. Data: CC BY 4.0, © the ASAP 2.0 authors; download from the official
repository and comply with its terms.

```bibtex
@article{crossley2025asap2,
  title   = {A large-scale corpus for assessing source-based writing quality: ASAP 2.0},
  author  = {Crossley, Scott A. and Baffour, Papa and Burleigh, L. and King, J.},
  journal = {Assessing Writing},
  volume  = {65},
  pages   = {100954},
  year    = {2025},
  doi     = {10.1016/j.asw.2025.100954}
}
```
