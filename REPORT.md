# Benchmark report

Current benchmark status and literature comparison are maintained in
[`reports/benchmark_comparison.md`](reports/benchmark_comparison.md), with
machine-readable paper references in
[`reports/literature_comparison.csv`](reports/literature_comparison.csv).

The ASAP 1.0 path is implemented but cannot be executed until the original
labeled table and authentic prompt/rubric context texts are supplied. The
ASAP 2.0 three-seed official-split run completed in the user's Kaggle
notebook; the provided aggregate test QWK was 0.7843 ± 0.0016. Those logs
predate the local scheduler/AMP correction and should be treated as historical
results until rerun on GPU with current code. The local environment is
CPU-only. See the benchmark report for per-seed metrics and limitations.
