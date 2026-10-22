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
