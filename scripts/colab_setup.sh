#!/usr/bin/env bash
# One-shot environment setup for Google Colab (or any fresh Linux box).
#
#   bash scripts/colab_setup.sh
#
# Installs Python dependencies, clones and extracts the official ASAP 2.0
# corpus, and prints dataset/rubric statistics. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Installing Python dependencies ==="
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

echo "=== Checking GPU ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
PY

echo "=== Preparing official ASAP 2.0 data ==="
python scripts/prepare_data.py --download

echo "=== Dataset / rubric inspection ==="
python scripts/inspect_dataset.py --tokenizer microsoft/deberta-v3-small

echo
echo "Setup complete. Suggested next steps:"
echo "  python scripts/run_baselines.py"
echo "  python scripts/run_from_config.py --config configs/cross_attention.yaml"
echo "  python scripts/run_ablation.py"
echo "  python scripts/run_cross_prompt.py --config configs/cross_attention.yaml"
echo "  python scripts/make_report.py --ablation"
