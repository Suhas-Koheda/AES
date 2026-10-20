# Running on Google Colab (GPU)

Everything in this repo is designed to run in a single Colab GPU session.
The default model is `microsoft/deberta-v3-small` (~44M parameters) with
`fp16`, batch size 16 and 4 epochs, which fits comfortably on a T4 (16 GB).

---

## 1. One-time setup

Open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) in Colab,
or paste this into a cell:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Then:

```bash
!git clone https://github.com/Suhas-Koheda/AES.git
%cd AES
!bash scripts/colab_setup.sh
```

`colab_setup.sh` installs `requirements.txt`, clones and decrypts the official
ASAP 2.0 corpus, converts the rubric to text, and prints schema/length
statistics. Re-running is safe.

> **GPU check:** if `torch.cuda.is_available()` is `False`, go to
> *Runtime → Change runtime type → Hardware accelerator → GPU* and re-run.

---

## 2. Verify before spending GPU time

```bash
!pytest -q                     # ~30 unit tests, offline, seconds
!python scripts/smoke_test.py  # synthetic end-to-end run, ~1-2 min
```

The smoke test uses a randomly initialised tiny DeBERTa and synthetic essays,
so it never depends on the official corpus or a large checkpoint. Its metrics
are meaningless; success means the pipeline runs start to finish.

---

## 3. Training recipe

Sequential, in this order (each stage writes its own `results/` directory):

```bash
# 1. Cheap reference (no GPU needed, ~2-5 min)
!python scripts/run_baselines.py --only tfidf

# 2. Transformer baseline (concatenation)
!python scripts/run_baselines.py --only transformer

# 3. Proposed cross-attention model
!python scripts/run_from_config.py --config configs/cross_attention.yaml

# 4. Ordinal variants
!python scripts/run_from_config.py --config configs/ordinal_coral.yaml
!python scripts/run_from_config.py --config configs/ordinal_expected.yaml

# 5. Ablations (6 runs)
!python scripts/run_ablation.py

# 6. Held-out-prompt sweep (7 runs)
!python scripts/run_cross_prompt.py --config configs/cross_attention.yaml

# 7. Tables + figures
!python scripts/make_report.py --ablation
```

### Multi-seed

```bash
!python scripts/run_from_config.py --config configs/cross_attention.yaml --seeds 42 43 44
```

Each seed writes `results/03_cross_attention/seed_<n>/`; the aggregate is
`results/03_cross_attention/aggregate.json`.

### Everything at once

```bash
!python scripts/run_all.py --seeds 42 43 44
```

---

## 4. Expected runtime (single T4, deberta-v3-small)

| Stage | Approx. time |
|---|---|
| TF-IDF grid (5 values of C) | 3–6 min (CPU) |
| Transformer baseline, 4 epochs | 60–90 min |
| Cross-attention, 4 epochs (shared encoder) | 75–110 min |
| CORAL / expected-score variant | same as cross-attention |
| Ablations (6 runs) | 5–8 h |
| Held-out-prompt sweep (7 runs) | 7–10 h |

Colab sessions are time-limited. Run stages in separate sessions and rely on
checkpoint resume:

```bash
!python scripts/run_from_config.py --config configs/cross_attention.yaml \
    --override train.resume_from=results/03_cross_attention/checkpoint_epoch_2.pt
```

To cut cost while debugging, subsample:

```bash
!python scripts/run_from_config.py --config configs/cross_attention.yaml \
    --override data.max_train_samples=4000 data.max_eval_samples=1000 train.epochs=2
```

---

## 5. Saving results

Colab discs are ephemeral. Either download the artefacts:

```python
from google.colab import files
!zip -r results.zip results
files.download("results.zip")
```

or mount Drive and point outputs there:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!python scripts/run_from_config.py --config configs/cross_attention.yaml \
    --output-dir /content/drive/MyDrive/aes_results/03_cross_attention
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: sentencepiece` | `!pip install sentencepiece` (needed by the DeBERTa-v3 tokenizer) |
| CUDA out of memory | reduce `train.batch_size`, increase `train.grad_accumulation`, or lower `data.max_context_length` |
| DataLoader worker errors | set `--override data.num_workers=0` |
| `pdftotext not found` during setup | expected on Colab; source-text experiments are optional. `!apt-get -qq install -y poppler-utils` if you need them |
| Test zip will not open | install `unzip`/`7z`; password is `asap2_test` |
| `torch.cuda.is_available() == False` | switch the Colab runtime to GPU |
| Figures missing | `!pip install matplotlib` then re-run `scripts/make_report.py` |
| Slow first step | model download + tokenizer warm-up; subsequent epochs are faster |

---

## 7. What "done" looks like

After the full recipe:

```
results/
  tables/model_comparison.csv       # A vs B vs C vs D
  tables/ablation.csv               # 6 controlled rows
  tables/per_prompt_test.csv
  figures/qwk_comparison.png
  figures/confusion_*.png
  figures/distribution_*.png
  cross_prompt/<model>/heldout_prompt_results.csv
```

Paste `results/tables/*.md` into `REPORT.md` section 5, and record hardware,
seeds and wall-clock time from each `summary.json`.
