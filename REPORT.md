# Results and experiment status

The official ASAP 2.0 dataset was inspected and its rubric extracted. The
cross-attention pipeline passed the unit suite and a synthetic end-to-end smoke
run in `/home/ssp/projects/Python/myenv`. The smoke run used generated essays
and a tiny randomly initialized encoder; its temporary predictions were
cleaned up, so it is not an AES result. No full training checkpoint or
benchmark predictions were produced. PyTorch is CPU-only and no GPU is visible.
The `outputs/` directory contains unrelated artifacts and is not an AES result source.

## Results table

| Dataset/task | Evaluation protocol | Our cross-attention score | Published paper score | Directly comparable? |
|---|---|---:|---:|---|
| ASAP 2.0 holistic 1–6 | Official train/test; seeded validation carved from train; report QWK, Pearson, Spearman, MAE, RMSE | Not run | No model score in the corpus paper | No published model result for this benchmark |
| Legacy ASAP + ASAP++ holistic “Overall” label | GAPS leave-one-prompt-out; mean QWK over prompts, five runs | Not run | GAPS: 0.670 QWK | No: legacy corpus/version, cross-prompt split, and paper aggregation differ from ASAP 2.0 official test |
| Legacy ASAP + ASAP++ analytic traits | GAPS leave-one-prompt-out; mean across trait/prompt results | Not run | GAPS full model: 0.591 mean trait QWK; 0.597 mean across prompts | No: different trait labels and task. Reference only |
| Legacy ASAP + ASAP++ analytic traits | TRATES leave-one-prompt-out; mean QWK over traits | Not run | TRATES Starling: 0.595; Llama: 0.566; Gemma: 0.586 | No: different trait task and feature-generation/regression architecture. Reference only |
| ELLIPSE analytic traits | TRATES 11-fold prompt-group cross-validation, four prompts per fold | Not run | TRATES Starling: 0.53 average QWK across six traits | No: different dataset, traits, score scale, and fold protocol. Reference only |
| ASAP 2.0 held-out prompt | Each prompt isolated from train/validation and tested separately | Not run | No matching published result identified | No published matched result identified |

The GAPS result rows come from Tables 2–3 of Do et al. (2025); the TRATES
rows come from Tables 1 and 3 of Eltanbouly et al. (2025). These are
published-paper results, not outputs of this repository. GAPS Table 2 reports
five-run trait QWK, including a distinct `Overall` label; it is not an ASAP
2.0 holistic score. TRATES explicitly states that it did not study holistic
scoring.

## Status

| Experiment | Status | Reason / output |
|---|---|---|
| Official ASAP 2.0 data extraction and inspection | Completed | Official CSVs validated at 17,307/7,421 rows; rubric extracted; schema report saved under ignored `results/`. |
| Cross-attention synthetic smoke run | Completed (pipeline check only) | Tiny-model synthetic training, validation selection, checkpoint reload, and test evaluation completed; temporary outputs were cleaned. Not an AES score. |
| Tests and source compilation | Completed | `python -m pytest -q` passed; `python -m compileall -q src scripts tests` passed using `/home/ssp/projects/Python/myenv`. |
| ASAP 2.0 official full training | Blocked in current compute environment | No GPU is visible; CPU full fine-tuning is not a feasible paper-quality substitute. Exact command is below. |
| ASAP 2.0 held-out-prompt sweep | Not run | Same hardware constraint; separate protocol remains available. |
| GAPS/ASAP++ trait run | Not run | Public data exists, but its task-specific trait/rubric adapter is not implemented; labels cannot be substituted for holistic scores. |
| TRATES/ELLIPSE trait run | Not run | ELLIPSE data is public under CC BY-NC-SA 4.0, but exact paper subset/fold mapping needs reconciliation and a trait adapter is not implemented. |

## Commands

```bash
source /home/ssp/projects/Python/myenv/bin/activate
python scripts/prepare_data.py --repo-dir data/asap2_raw
python -m pytest -q
python scripts/smoke_test.py
python scripts/run_from_config.py --config configs/cross_attention.yaml --seeds 42 43 44
python scripts/run_cross_prompt.py --config configs/cross_attention.yaml
```

The experiment uses only the official test partition after checkpoint
selection on validation QWK. Every run should retain its checkpoint, exact
configuration, package/hardware metadata, split manifest, and per-example
predictions in `results/`.

## Published sources and limitations

* Crossley et al. (2025), [ASAP 2.0](https://doi.org/10.1016/j.asw.2025.100954).
  The released CSV files total 24,728 rows, while the paper/repository prose
  states 24,278; split row counts sum to 24,728 and are used here.
* Do et al. (2025), [GAPS, Findings of NAACL](https://aclanthology.org/2025.findings-naacl.153/).
  It combines ASAP and ASAP++, uses leave-one-prompt-out, batch 10, up to 50
  epochs, best validation checkpoint and five runs. The paper reports QWK and
  states its official code is absent. It does not specify a tokenization
  recipe or learning rate in its experiment section.
* Eltanbouly et al. (2025), [TRATES, Findings of ACL](https://aclanthology.org/2025.findings-acl.1054/).
  It combines ASAP and ASAP++ for trait scoring and separately evaluates
  ELLIPSE. For ASAP it uses leave-one-prompt-out, sequential hyperparameter
  tuning, early stopping patience 10, learning-rate scheduler patience 5,
  factor 0.1, batch 128, and QWK. ELLIPSE uses 11 folds holding out four
  prompts per fold. TRATES notes holistic scoring was not explored.

The ASAP++ release is [here](https://lwsam.github.io/ASAP%2B%2B/lrec2018.html).
ELLIPSE's official repository documents its protected test archive and
CC BY-NC-SA 4.0 terms [here](https://github.com/scrosseye/ELLIPSE-Corpus).
The complete inventory is [PROTOCOL_INVENTORY.md](PROTOCOL_INVENTORY.md).
