# Related work and novelty audit

All summaries below were checked against the published abstracts/pages on
2026-10-21. Anything marked **proposed** is an idea in this repository, not a
published result.

---

## 1. Verified references

### ASAP 2.0 corpus
Crossley, S. A., Baffour, P., Burleigh, L., & King, J. (2025).
*A large-scale corpus for assessing source-based writing quality: ASAP 2.0.*
Assessing Writing, 65, 100954.
<https://doi.org/10.1016/j.asw.2025.100954>

* 24,278 persuasive source-based essays, seven prompts, grades 6/8/9/10.
* 12,871 essays from PERSUADE 2.0; 11,829 previously unreleased.
* Ships one holistic 1–6 rubric, human holistic ratings and demographic
  metadata, plus the original source passages.
* Relevance: the dataset used here. It provides **no trait labels and no
  rubric variation**.

### GAPS — grammar-aware cross-prompt AES
Do, H., Park, T., Ryu, S., & Lee, G. (2025). *Towards Prompt Generalization:
Grammar-aware Cross-Prompt Automated Essay Scoring.* Findings of NAACL 2025,
pp. 2818–2824. <https://aclanthology.org/2025.findings-naacl.153/>

* Argument: prompt-specific essay–score pairs make it hard to learn a
  prompt-general essay representation.
* Method: grammar-error-corrected essays are used alongside originals so the
  model focuses on prompt-independent syntactic features; trait scoring.
* Evaluation: cross-prompt trait scoring; reports QWK gains in the hardest
  cross-prompt setting.
* Relevance: establishes that **cross-prompt** generalization is the hard
  setting, and that auxiliary prompt-independent signals help. It does not
  condition on the rubric text itself.

### TRATES — trait-specific rubric-assisted cross-prompt scoring
Eltanbouly, S., Albatarni, S., & Elsayed, T. (2025). *TRATES: Trait-Specific
Rubric-Assisted Cross-Prompt Essay Scoring.* Findings of ACL 2025,
pp. 20528–20543. <https://aclanthology.org/2025.findings-acl.1054/>

* Method: an LLM uses **trait grading rubrics** to generate trait-specific
  assessment questions, answers them for a given essay, then combines those
  features with generic writing-quality and prompt-specific features in a
  classical regression model.
* Evaluation: trait scores on an unseen prompt; reports state-of-the-art
  across traits on a widely used dataset, with LLM-generated features the most
  important.
* Relevance: closest to "rubric-assisted" scoring, but rubric use is mediated
  through **LLM-generated features**, not through attention over rubric tokens.
  This is the main architectural contrast with Model C.

### When Rubrics Change — cross-rubric generalization
Kumar, N. A., Wittawatolarn, P., Kang, S., Peczuh, M. C., Lehman, B., Baker,
R., Mills, C., Lachman, S., Sun, R., & Lan, A. (2026). *When Rubrics Change:
Cross-Rubric Generalization for Critical Thinking Essay Scoring.*
arXiv:2607.13433 (AI for Education Day at SIGKDD 2026).
<https://arxiv.org/abs/2607.13433>

* Problem: prior AES work holds scoring criteria constant while varying the
  prompt; in practice rubrics change too.
* Method: LLM fine-tuning with rubric-agnostic intermediate **traits** plus
  target-essay supervision under seen rubrics.
* Results (abstract): +5.0% macro-F1 over a no-traits baseline in the hardest
  setting (unseen target rubrics *and* unseen target essays); best open-source
  Llama-based model beats GPT-5-mini prompting by 2.1% macro-F1 and trails
  GPT-5 by 1.9%.
* Relevance: defines **cross-rubric** generalization as a distinct target.
  ASAP 2.0 cannot support it, so this repository explicitly does not claim it.

### Adjacent, well-established components (not re-verified here)
* **CORAL** ordinal regression (Cao, Mirjalili & Raschka, 2020) — the
  extended-binary target and shared-bias formulation used by `src/losses.py`.
* **Quadratic Weighted Kappa** as the standard AES metric.
* **DeBERTa-v3** (He et al., 2021) as the encoder.

---

## 2. Comparison table

| Work | Inputs | Architecture | Supervision | Evaluation protocol | Generalization target |
|---|---|---|---|---|---|
| **ASAP 2.0** (2025) | essay + prompt + metadata | corpus, not a model | holistic 1–6 | fixed train/test | none (benchmark) |
| **GAPS** (2025) | essay + grammar-corrected essay | encoder + GEC integration, trait heads | trait scores | cross-prompt | unseen prompt |
| **TRATES** (2025) | essay + **trait rubric** (via LLM) | LLM feature generation + classical regressor | trait scores | cross-prompt | unseen prompt |
| **When Rubrics Change** (2026) | essay + rubric (LLM fine-tune) | LLM + rubric-agnostic traits | rubric-defined labels | cross-rubric | unseen rubric (± unseen essay) |
| **This repo — Model B** | `[QUESTION][RUBRIC][ANSWER]` concat | encoder + 6-way head | holistic | official + held-out prompt | unseen prompt |
| **This repo — Model C** | answer ⟂ (question+rubric) | encoder + **cross-attention + gated residual** | holistic | official + held-out prompt | unseen prompt |
| **This repo — Model D** | as C | as C | **CORAL** or CE+expected-score | official + held-out prompt | unseen prompt |

---

## 3. Novelty assessment

**Not novel individually:** cross-attention over prompt context, gated
residual fusion, masked pooling, CORAL heads, and rubric text as input are all
established components. Combining them is engineering, not a contribution.

**The specific gap this repository targets** is methodological rather than
architectural:

> There is no public, same-split, same-token-budget ablation on ASAP 2.0 that
> separates the effect of (i) *question* information, (ii) *rubric*
> information, (iii) *fusion mechanism* (concatenation vs cross-attention) and
> (iv) *ordinal objective*, while also reporting held-out-prompt results and
> subgroup errors on the official 1–6 scale.

Whether that gap is worth a paper depends entirely on the measured effect
sizes. If cross-attention is within noise of concatenation once token budgets
are matched (`configs/cross_attention_matched.yaml`), the honest conclusion is
that rubric conditioning per se — not the attention mechanism — is what
matters, or that neither helps.

**Experiments that could substantiate a contribution**

1. Matched-budget concat vs cross-attention (isolates fusion).
2. Rubric-only vs question-only vs both (isolates which context matters).
3. Frozen vs fine-tuned encoder (isolates whether gains come from
   representation learning or from context access).
4. Held-out-prompt sweep with per-prompt effect sizes and confidence intervals.
5. Multiple seeds with paired tests across prompts, not just mean QWK.
6. Attention-weight inspection: does the model actually attend to rubric
   bands near the predicted score? (Qualitative, exploratory.)

**Do not claim:** novelty of cross-attention, superiority over TRATES/GAPS
(different supervision and datasets), or unseen-rubric generalization.

---

## 4. References

```bibtex
@article{crossley2025asap2,
  title={A large-scale corpus for assessing source-based writing quality: ASAP 2.0},
  author={Crossley, Scott A. and Baffour, Papa and Burleigh, L. and King, J.},
  journal={Assessing Writing}, volume={65}, pages={100954}, year={2025},
  doi={10.1016/j.asw.2025.100954}}

@inproceedings{do2025gaps,
  title={Towards Prompt Generalization: Grammar-aware Cross-Prompt Automated Essay Scoring},
  author={Do, Heejin and Park, Taehee and Ryu, Sangwon and Lee, Gary},
  booktitle={Findings of NAACL 2025}, pages={2818--2824}, year={2025},
  doi={10.18653/v1/2025.findings-naacl.153}}

@inproceedings{eltanbouly2025trates,
  title={TRATES: Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring},
  author={Eltanbouly, Sohaila and Albatarni, Salam and Elsayed, Tamer},
  booktitle={Findings of ACL 2025}, pages={20528--20543}, year={2025},
  doi={10.18653/v1/2025.findings-acl.1054}}

@article{kumar2026rubrics,
  title={When Rubrics Change: Cross-Rubric Generalization for Critical Thinking Essay Scoring},
  author={Kumar, Nischal Ashok and Wittawatolarn, Payu and Kang, Sana and Peczuh, Marisa C. and Lehman, Blair and Baker, Ryan and Mills, Caitlin and Lachman, Sherry and Sun, Ruochen and Lan, Andrew},
  journal={arXiv preprint arXiv:2607.13433}, year={2026}}

@article{cao2020coral,
  title={Rank consistent ordinal regression for neural networks with application to age estimation},
  author={Cao, Wenzhi and Mirjalili, Vahid and Raschka, Sebastian},
  journal={Pattern Recognition Letters}, volume={140}, pages={325--331}, year={2020}}
```
