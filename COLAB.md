# Running the single cross-attention AES model on Colab

Select a GPU runtime, clone this repository, and run:

```bash
!git clone https://github.com/Suhas-Koheda/AES.git
%cd AES
!bash scripts/colab_setup.sh
!pytest -q
!python scripts/smoke_test.py
!python scripts/run_from_config.py --config configs/cross_attention.yaml --seeds 42 43 44
```

`scripts/colab_setup.sh` installs dependencies, downloads/extracts the official
ASAP 2.0 dataset, and inspects its schema. The smoke test uses synthetic data;
its scores have no research meaning. The full command uses only the official
training set for fitting and validation selection, then evaluates on the
untouched official test set. Each seed writes a separate directory under
`results/`, including checkpoint, configuration, environment, split manifest,
metrics and per-example predictions.

To run the held-out-prompt protocol separately:

```bash
!python scripts/run_cross_prompt.py --config configs/cross_attention.yaml
```

See [PROTOCOL_INVENTORY.md](PROTOCOL_INVENTORY.md) and [REPORT.md](REPORT.md)
for the published dataset protocols, reference-only paper results and current
experiment status. A full run is compute intensive; no scores should be
reported until its outputs have been checked.
