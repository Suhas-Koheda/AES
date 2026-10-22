# Dataset and protocol inventory

This inventory was checked against the original experimental sections and the
dataset releases on 2026-10-22. Published trait scores below are reference
results only: they are not holistic results and are not directly comparable to
our ASAP 2.0 run.

| Dataset/task | Used by paper | Labels and subset | Paper split and evaluation | Reproduction status here |
|---|---|---|---|---|
| ASAP 2.0 holistic | Crossley et al. (2025) introduces this source-based writing corpus; our documented benchmark uses it | Seven prompts, grades 6/8/9/10, one common holistic integer rubric 1–6. Official files contain 17,307 train + 7,421 test = 24,728 rows; the paper and GitHub README prose say 24,278, an internal arithmetic mismatch. | Official corpus train/test files; this paper describes a dataset, not a model experiment or model score | Raw official archives already present and extracted locally. No official validation split; reserve validation from train, never select on test. |
| ASAP + ASAP++ analytic trait scoring (GAPS) | Do et al. (2025) combines the original ASAP and ASAP++ | 12,978 essays across 8 prompts; multiple prompt-specific human trait labels. ASAP provides traits for P7–P8 and ASAP++ adds P1–P6. Trait ranges differ by prompt. | Leave one prompt out: seven prompts train, target prompt test. GPU reported: RTX 2080 Ti; batch 10; up to 50 epochs; best validation checkpoint. Five seeds in reported tables. Primary metric QWK. | ASAP++ essay text and trait annotations are available via the authors' release page; original ASAP has separate access/download. The task needs trait-specific training and task/rubric context. No official held-out split files are specified. Not yet adapted/run. |
| ASAP + ASAP++ analytic trait scoring (TRATES) | Eltanbouly et al. (2025), main experiments | Same combined 12,978-essay / 8-prompt trait dataset. Prompt trait membership and score ranges vary; some traits exist in only one prompt and cannot support cross-prompt scoring. | Leave-one-prompt-out; sequential one-hyperparameter-at-a-time tuning; early stop patience 10, ReduceLROnPlateau patience 5/factor 0.1, batch 128. Report QWK on unseen prompts and trait averages. Reported baseline results are copied from prior studies. | ASAP++ source is public. Feature files in TRATES are public but encode LLM-generated features and are not model inputs for this experiment. Same trait adaptation needed; not yet adapted/run. |
| ELLIPSE trait scoring (TRATES) | TRATES uses it as a second-dataset generalization experiment | Paper's selected corpus has 6,482 essays / 44 prompts / grades 8–12; six trait labels, 1–5 in 0.5 increments. The corpus release has a larger raw file and a protected test archive; paper's exact 6,482 subset must be verified against its preprocessing before reuse. | 11 folds, each test fold holds out 4 prompts; QWK. TRATES uses Starling-derived trait features and evaluates mean results over prompts. This is not the same split as the dataset's protected train/test split. | Official ELLIPSE GitHub repo is public under CC BY-NC-SA 4.0. Full paper-specific subset/split mapping has not been verified locally; not yet adapted/run. |

## Protocol details and limits

* ASAP 2.0 files and rubric are under CC BY 4.0. Current repository loader
  backfills only empty assignment cells from the unique assignment for that
  prompt, uses light whitespace normalization, and maps score `s` to class
  `s-1`. Official test remains untouched. Because ASAP 2.0 has one rubric,
  it cannot test cross-rubric generalization.
* GAPS and TRATES do **not** evaluate on ASAP 2.0. They evaluate trait scores
  on the legacy ASAP+ASAP++ dataset, whose eight prompts, labels and score
  ranges differ. Their QWK values cannot be compared numerically with holistic
  ASAP 2.0 QWK.
* GAPS describes its main experiments and five-run QWK tables but does not
  report a tokenization contract or learning rate in the paper text. The
  released paper also notes that official GAPS code is absent; its GEC model
  is a separately available T5-based system.
* TRATES is not a neural essay encoder: LLMs generate rubric-specific feature
  questions and answers, then classical regression combines these with
  writing and prompt features. Its training configuration and data splits do
  not transfer to our architecture. It reports QWK, not Pearson, Spearman,
  MAE or RMSE for the main results.
* For our supported ASAP 2.0 experiment, the validation split is a seeded,
  stratified 15% carve-out of official training data by prompt and score;
  checkpoint selection is by validation QWK. Test metrics include QWK,
  Pearson, Spearman, MAE and RMSE, plus per-prompt and public/private results.

## Sources

* Crossley et al., [ASAP 2.0](https://doi.org/10.1016/j.asw.2025.100954); data
  release [scrosseye/ASAP_2.0](https://github.com/scrosseye/ASAP_2.0).
* Do et al., [GAPS paper](https://aclanthology.org/2025.findings-naacl.153/),
  [ASAP++ release](https://lwsam.github.io/ASAP%2B%2B/lrec2018.html).
* Eltanbouly et al., [TRATES paper](https://aclanthology.org/2025.findings-acl.1054/),
  [TRATES repository](https://github.com/Sohaila-se/TRATES).
* Crossley et al., [ELLIPSE corpus repository](https://github.com/scrosseye/ELLIPSE-Corpus),
  licensed CC BY-NC-SA 4.0.
