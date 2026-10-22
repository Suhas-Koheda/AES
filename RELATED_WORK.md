# Related papers and protocol comparison

The repository retains one proposed system: essay representations query
question and rubric representations through cross-attention, followed by a
gated residual, masked attention pooling, and a six-class holistic score head.
This composition is not claimed as a novel architecture.

## GAPS

Do, Park, Ryu, and Lee (2025), “Towards Prompt Generalization:
Grammar-aware Cross-Prompt Automated Essay Scoring,” Findings of NAACL 2025,
2818–2824. [Paper](https://aclanthology.org/2025.findings-naacl.153/)

GAPS uses legacy ASAP and ASAP++ combined: eight prompts and 12,978 essays,
with analytic trait labels (ASAP has traits for prompts 7–8; ASAP++ adds
prompts 1–6). It leaves one prompt out for test and uses the other seven for
training, reports five-run QWK, and says it uses an RTX 2080 Ti, batch 10, up
to 50 epochs and validation-best checkpoint. Its model uses GEC-corrected
essays, POS features and a hierarchical essay scorer. The paper reports
0.670 QWK on its separate Overall label; full GAPS has 0.591 mean QWK across
traits and 0.597 mean across prompts. These are reference values only: the
labels and evaluation protocol do not match ASAP 2.0's holistic official test.
The paper reports no learning rate or tokenization recipe and notes that its
official code is absent.

## TRATES

Eltanbouly, Albatarni, and Elsayed (2025), “TRATES: Trait-Specific
Rubric-Assisted Cross-Prompt Essay Scoring,” Findings of ACL 2025,
20528–20543. [Paper](https://aclanthology.org/2025.findings-acl.1054/)

TRATES uses the same combined legacy ASAP/ASAP++ analytic trait dataset and
leave-one-prompt-out protocol. It generates questions from each trait rubric,
uses open LLMs to extract trait-specific essay features, and fits classical
regression with generic and prompt-specific features. It reports QWK; the
Starling, Llama and Gemma mean trait scores are 0.595, 0.566 and 0.586.
The paper additionally evaluates ELLIPSE (6,482 essays in its selected
subset, 44 prompts, six analytic traits, scores 1–5 in half-point increments)
using 11 folds with four prompts held out per fold; TRATES Starling reports
0.53 mean trait QWK. The paper expressly notes holistic scoring was not
explored. These numbers are reference-only and must not be compared as if they
were holistic ASAP 2.0 scores.

## Dataset compatibility

The ASAP 2.0 corpus paper introduces a different corpus: seven source-based
prompts, grades 6/8/9/10, a single common holistic 1–6 rubric, and official
17,307/7,421 train/test rows. It has no trait labels or rubric variation.
Thus ASAP 2.0 supports holistic scoring and prompt-held-out evaluation, but
cannot reproduce GAPS/TRATES trait tasks or establish unseen-rubric
generalization.

The legacy ASAP++ score release and ELLIPSE corpus are accessible; see the
[dataset/protocol inventory](PROTOCOL_INVENTORY.md) for licences, split
definitions, preprocessing caveats, and reproduction status. We have not
converted traits into holistic scores. A trait experiment would require a
task-specific label/rubric adapter and must be reported separately.

## References

* Crossley, S. A., Baffour, P., Burleigh, L., & King, L. (2025). *A large-scale
  corpus for assessing source-based writing quality: ASAP 2.0*. Assessing
  Writing, 65, 100954. https://doi.org/10.1016/j.asw.2025.100954
* Do, H., Park, T., Ryu, S., & Lee, G. (2025). *Towards Prompt Generalization:
  Grammar-aware Cross-Prompt Automated Essay Scoring*. Findings of NAACL 2025.
  https://doi.org/10.18653/v1/2025.findings-naacl.153
* Eltanbouly, S., Albatarni, S., & Elsayed, T. (2025). *TRATES: Trait-Specific
  Rubric-Assisted Cross-Prompt Essay Scoring*. Findings of ACL 2025.
  https://doi.org/10.18653/v1/2025.findings-acl.1054
* Mathias, S. & Bhattacharyya, P. (2018). *ASAP++: Enriching the ASAP
  Automated Essay Grading Dataset with Essay Attribute Scores*. LREC 2018.
