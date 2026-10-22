# Rubric conditioned cross attention essay scoring

This repository trains and evaluates one model: an essay encoder whose token
representations query question and scoring rubric representations through
cross attention, then use the existing gated residual, masked pooling and
holistic score head.

## Dataset and research protocol

The currently implemented adapter supports official ASAP 2.0 holistic scores
(1–6). The official files contain 17,307 training and 7,421 test essays across
seven prompts (the test split contains five; the files sum to 24,728 rows,
though corpus prose says 24,278). The corpus uses one shared
holistic rubric, so this experiment does not test unseen rubric generalization.
See [DATASET.md](DATASET.md) for the schema and [PROTOCOL_INVENTORY.md](PROTOCOL_INVENTORY.md)
for the datasets and protocols used by GAPS and TRATES and their compatibility
limits. Legacy ASAP/ASAP++ and ELLIPSE trait results are not direct comparisons
to ASAP 2.0 holistic scoring.

ASAP 2.0 is distributed under CC BY 4.0. `scripts/prepare_data.py` extracts
the official archive already placed in `data/asap2_raw/`; for a fresh checkout,
run it with `--download` to obtain the official source repository. The
password-protected test archive is handled by the script.

The repository also includes an ASAP 1.0 adapter and prompt-specific
five-fold evaluation path. ASAP 1.0 requires the original labeled essays and
genuine prompt/rubric context text supplied by the user. The adapter maps each
prompt's published native score range to the existing six-class head, then
decodes predictions to the native scale for metrics. This quantization means
the ASAP 1.0 results are not strictly head-to-head with papers training and
predicting directly on each native score scale. Details and current data
availability are in [DATASET.md](DATASET.md) and
[reports/benchmark_comparison.md](reports/benchmark_comparison.md).

## Setup and run

```bash
source /home/ssp/projects/Python/myenv/bin/activate
python -m pip install -e '.[test]'
python scripts/prepare_data.py --download
python scripts/inspect_dataset.py --tokenizer microsoft/deberta-v3-small
pytest -q
python scripts/smoke_test.py
python scripts/run_from_config.py --config configs/cross_attention.yaml
```

For repeat runs with distinct seeds:

```bash
python scripts/run_from_config.py --config configs/cross_attention.yaml --seeds 42 43 44
```

Run a held-out-prompt protocol separately (it does not read official test
labels):

```bash
python scripts/run_cross_prompt.py --config configs/cross_attention.yaml
```

The official protocol reserves a seeded, stratified 15% of the official
training file for validation, selects checkpoints on validation QWK, and
evaluates the selected checkpoint once on official test. See
[configs/cross_attention.yaml](configs/cross_attention.yaml) for the full
hyperparameters. GPU is strongly recommended; the config enables safe CUDA
mixed precision and gradient checkpointing. No test data is used for model
selection.

Each run writes its configuration, environment, split manifest, checkpoint,
training history, metrics and per-example predictions under `results/<run>/`.
The result includes QWK, Pearson, Spearman, MAE, RMSE and per-prompt and
public/private breakdowns. Dataset files, checkpoints, and outputs are kept
out of Git.

## ASAP 1.0 reproduction

The private repository includes `asap1_kaggle.zip`. After cloning it in an
environment authorized to access the repository, prepare the local,
Git-ignored files:

```bash
python scripts/prepare_asap1.py
```

The script extracts the labeled training TSV and the eight original prompt
and rubric descriptions. It does not extract or use the unlabeled Kaggle
validation/test essays as if they had gold scores. The adapter accepts the
standard `essay_set`, `essay`, and `domain1_score` columns (or equivalent
`prompt_id`, `full_text`, `score`). It applies prompt-specific score bounds
and uses seeded score-stratified folds: for each of five folds, 60% train,
20% validation, 20% test, within a single prompt. Fold `i` is test and fold
`(i+1) mod 5` is validation. Run all prompts and folds with:

```bash
python scripts/run_asap1.py --config configs/asap1_cross_attention.yaml --seed 42
```

This is a documented five-fold cross-validation protocol, not the historical
Kaggle random split. Outputs include per-fold native-scale predictions,
metrics, checkpoints, and prompt-level means/std under `results/asap1/`.

## ASAP 2.0 reproduction

The official split has 17,307 training and 7,421 test essays. The 15%
stratified validation carve-out is drawn only from the official training
partition; the official test partition is not used for model selection.
Reproduce the three-seed run with:

```bash
python scripts/run_from_config.py --config configs/cross_attention.yaml --seeds 42 43 44
```

## Result status

No full experiment is reported as completed until its training and evaluation
commands finish and the resulting artifacts are checked. Current run status
and paper-reference scores are in [REPORT.md](REPORT.md). The published GAPS
and TRATES studies use different trait-level datasets and protocols, so their
numbers are reference-only.

## Citation and license

Code is MIT licensed. Cite the corpus paper when using ASAP 2.0:

Crossley, S. A., Baffour, P., Burleigh, L., & King, L. (2025). *A large-scale
corpus for assessing source-based writing quality: ASAP 2.0*. Assessing
Writing, 65, 100954. https://doi.org/10.1016/j.asw.2025.100954
