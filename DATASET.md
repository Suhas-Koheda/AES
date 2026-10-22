# Dataset documentation — ASAP 2.0

Source: <https://github.com/scrosseye/ASAP_2.0> (CC BY 4.0).
Paper: Crossley, Baffour, Burleigh & King (2025), *Assessing Writing* 65,
100954, <https://doi.org/10.1016/j.asw.2025.100954>.

Everything below was verified against the released files (not inferred from
documentation).

---

## 1. Files

| File | Content |
|---|---|
| `ASAP_2_Final_github_train.zip` | training CSV (17,307 rows) |
| `ASAP_2_Final_github_test.zip` | password-protected test CSV (**password `asap2_test`**) |
| `ASAP_2_source_texts.zip` | seven source-passage PDFs |
| `asap_scoring_rubric.docx` | the holistic rating form + 1–6 rubric |

`scripts/prepare_data.py` clones the repo, decrypts the test archive, extracts
the rubric to plain text, and converts the PDFs to text when a text layer
exists. It never downloads a substitute dataset.

---

## 2. CSV schema

Both CSVs have the **same 15 columns**:

| Column | Type | Notes |
|---|---|---|
| `essay_id` | string | unique within a split; **never used as a feature** |
| `score` | int | human holistic score, 1–6 |
| `full_text` | string | the essay |
| `set` | string | `train` / `test` |
| `pubpriv` | int in train, string in test | train = `0`; test = `public`/`private` |
| `assignment` | string | the writing prompt (one unique text per prompt) |
| `prompt_name` | string | one of seven prompt names |
| `economically_disadvantaged` | string / NaN | analysis only |
| `student_disability_status` | string / NaN | analysis only |
| `ell_status` | string / NaN | analysis only |
| `race_ethnicity` | string / NaN | analysis only |
| `gender` | string / NaN | analysis only |
| `grade_level` | int | 6, 8, 9, 10 |
| `essay_word_count` | float | supplied by the corpus |
| `task` | string | `Text dependent` for every row |

The test CSV's `pubpriv` is a **string** while the train CSV's is `0`; the
loader tolerates both and never treats this column as a feature.

---

## 3. Prompt inventory and split coverage

| Prompt | Train n | Test n | Grade |
|---|---|---|---|
| Driverless cars | 3,500 | 2,670 | 10 |
| Facial action coding system | 3,043 | 1,840 | 10 |
| Exploring Venus | 3,017 | 1,463 | 10 |
| The Face on Mars | 2,094 | 921 | 8 |
| Does the electoral college work? | 2,046 | **0** | 9 |
| Car-free cities | 1,959 | **0** | 10 |
| A Cowboy Who Rode the Waves | 1,648 | 527 | 6 |

Two prompts are train-only. The official test file is additionally split into
`public` (2,257) and `private` (5,164) essays; `src/evaluate.py` reports that
breakdown when `pubpriv` is present.

### Missing `assignment` values

The released files contain a handful of rows with an empty `assignment`:
**18 in train** (The Face on Mars 5, Exploring Venus 4, Car-free cities 4,
Facial action coding system 3, Driverless cars 2) and **10 in test**
(The Face on Mars 5, Facial action coding system 3, Driverless cars 2). Each
prompt has exactly **one** distinct assignment text elsewhere in the same file,
so `src/data.py::fill_missing_assignments` copies it into the empty cells
*before* schema validation. No rows are dropped and no text is invented. The
number filled is reported by `scripts/inspect_dataset.py` as
`n_filled_assignment`.

---

## 4. Labels

Human holistic scores `{1,2,3,4,5,6}` map to internal classes `{0,…,5}` via
`label = score - 1`, implemented and tested in `src/labels.py`. Every metric is
computed on the original 1–6 scale by mapping predictions back with
`score = class + 1`.

Class imbalance is severe (score 6 = 156/17,307 ≈ 0.9% in train). Configs can
enable `train.class_weight: balanced`; the default is `none`.

---

## 5. Rubric

`asap_scoring_rubric.docx` defines:

* a 1–6 holistic scale with equal spacing between adjacent scores;
* six band descriptions (`SCORE OF 6:` … `SCORE OF 1:`) covering point of
  view / critical thinking, use of source evidence, organisation, language and
  mechanics.

`src/rubric_loader.py` extracts `word/document.xml`, parses the six bands, and
refuses to operate if any band is missing — descriptions are never invented.
The rubric is **shared by all prompts**; `RubricLoader.for_prompt()` returns the
same text and documents this explicitly.

---

## 6. Preprocessing contract

1. **Load** both CSVs, validate the schema (required columns, unique non-null
   `essay_id`, non-null text, scores in 1–6).
2. **Clean lightly**: normalise CRLF, NBSP and repeated whitespace. No
   lowercasing, punctuation stripping or spelling correction — mechanics is
   part of the rubric.
3. **Add** `label = score - 1`.
4. **Split**:
   * *official*: stratified 85/15 train/val carve-out of the training
     partition, stratified jointly on `(prompt_name, score)`; the official test
     file is untouched.
   * *held-out prompt*: one prompt removed from train **and** val, used only
     for test; remaining prompts split 85/15.
5. **Leakage checks**: essay IDs must be disjoint across splits; the held-out
   prompt must not appear in train or val.
6. **Tokenise at collation time** (dynamic padding). Question+rubric and answer
   have **separate budgets**, so a long essay can never truncate the rubric to
   nothing.
7. **Long essays**: default `truncation: head_tail` keeps the first and last
   halves (introduction + conclusion) with a separator token. Chunking
   (`data.chunk: true`) encodes overlapping windows and keeps the first and
   last window. Truncation counts are recorded by
   `scripts/inspect_dataset.py`.

---

## 7. What this dataset cannot support

* **Criterion / trait-level supervision** — only one holistic score exists.
* **Cross-rubric generalization** — there is one rubric.
* **Prompt-specific rubric instructions** — only prompt-specific *assignment*
  text exists.
* **Demographic fairness ground truth** — demographics are self-reported
  metadata with substantial missingness (e.g. 4,019 train rows lack economic
  and disability status); subgroup results are descriptive only.

Any experiment requiring the above needs an additional dataset.
