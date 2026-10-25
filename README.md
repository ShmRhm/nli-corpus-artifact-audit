# Native Language Identification: cross-corpus audit of SVM and LLM systems

Code for the paper *How Much of Native Language Identification Is Corpus
Artifact? A Cross-Corpus Audit of SVMs and LLMs on TOEFL11 and ICLE*.

The pipeline trains a TF-IDF + LinearSVC classifier on TOEFL11 only, evaluates
it together with a zero-shot LLM (gpt-4o-mini) on both TOEFL11 and ICLEv3, and
runs three audits for corpus artifacts that can inflate native language
identification (NLI) accuracy: document-ID headers, self-disclosure of
nationality, and topic exclusivity.

## Data is not included

This repository contains **code only**. TOEFL11 and ICLEv3 are licensed
corpora and cannot be redistributed:

- **TOEFL11** must be obtained from ETS (LDC catalogue LDC2014T06).
- **ICLEv3** must be obtained from UCLouvain (Centre for English Corpus
  Linguistics).

No corpus text, derived corpus file, feature matrix, or model artifact is
included here, and `.gitignore` is set up to keep them out. To reproduce the
results you need your own licensed copies.

## Expected layout

Place your licensed data as follows, then run the notebooks in order from
`notebooks/`:

```
project-root/
  data/
    raw/
      TOEFL11/
        toefl_train.csv        # columns: text, Language
        toefl_test.csv         # columns: text, Language
      ICLEV3/
        metadata.csv           # includes: File name, Native language, Title
        texts/                 # one .txt file per essay
    processed/
      results/                 # written by the notebooks
  models/                      # written by the notebooks
  notebooks/
```

## Pipeline

| Notebook | Purpose |
| --- | --- |
| `01_data_preprocessing` | Build the combined corpus from TOEFL11 and ICLEv3. Strips the ICLE document-ID header (see below). |
| `02_pos_masking` | Replace content words (NOUN, PROPN, VERB, ADJ, ADV) with single-letter POS codes using spaCy. |
| `02b_masking_fidelity_audit.py` | Independent token-level check that POS masking replaced every maskable token and left every other token unchanged. |
| `03_feature_extraction` | Fit TF-IDF vectorizers on TOEFL11 training data only. |
| `04_model_training` | Train LinearSVC on masked text. |
| `05_llm_evaluation` | Zero-shot LLM evaluation, masked text, forced-JSON prompt. |
| `07_unmasked_svm_baseline` | Full-text SVM baseline. |
| `08_llm_full_text_baseline` | Zero-shot LLM evaluation, full text. |
| `09_llm_reasoning_ablation` | Free-reasoning prompt format, both text conditions. |
| `10_full_comparison_2x2` | System x text-condition x prompt-format comparison across raw, clean and doubly-clean tiers. |
| `11_leakage_audit_country_names` | Flags essays that state the writer's own nationality or L1. |
| `12_icle_topic_bias_audit` | Flags essays whose ICLE title is exclusive to one L1 at an 80% threshold. |
| `13_paper_figures_and_tables` | Builds the tables reported in the paper. |
| `14_icle_residual_signal_diagnostic` | Diagnostics on the signal that survives both filters. |
| `15_institution_leak_audit` | Finds vocabulary disproportionately associated with one contributing institution. |
| `16_doc_header_leak_check` | Detects the ICLE document-ID header. |
| `17_strip_icle_header_fix` | One-time repair for corpora built before the header strip was added to notebook 01. |
| `18_significance_tests.py` | McNemar and chi-square tests reported in the paper. |

Notebooks 16 and 17 are retained for transparency. Notebook 01 now strips the
document-ID header at source, so a pipeline built from scratch will not need
notebook 17.

## The document-ID header

Raw ICLEv3 text files begin with a corpus-ID header such as
`<ICLE-FR-UCL-0027.2>`, which encodes the country and institution of each
essay. It is present in all 2,619 essays used here and absent from TOEFL11.
Any system that reads the raw text can therefore read each ICLE essay's
country of origin directly. Notebook 01 removes it; notebook 16 checks for it.
Anyone working with ICLEv3 should verify that this header is not reaching
their model.

## Requirements

Python >= 3.10.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install the versions pinned in `requirements.txt` rather than the latest:
`scikit-learn` affects the SVM results and the spaCy model version affects
POS masking, so both change the reported numbers.

The LLM notebooks read an OpenAI API key from the `OPENAI_API_KEY`
environment variable. No key is stored in this repository.

## Reproducibility notes

- The SVM vectorizers and classifier are fit only on the 8,000 TOEFL11
  training essays. ICLE text is only ever transformed, never fit on.
- Evaluation uses a fixed, balanced 800-document set: 50 essays per language
  per partition, shared by every system and condition.
- LLM results depend on the model version served by the API and will not
  reproduce exactly.
- POS masking uses spaCy's tokenization. Note that `<P>` and `<S>` boundary
  markers are split by the tokenizer rather than preserved intact; this is
  described in the paper.

## Licence

Code is released under the MIT Licence; see [`LICENSE`](LICENSE). The corpora
are not covered by this licence and remain subject to their own terms.

## Citing this code

**DOI: TODO -- insert the Zenodo DOI here before making this repository public.**

If you use this code, please cite the paper:

> Shima Rahimi (2026). *How Much of Native Language Identification Is Corpus
> Artifact? A Cross-Corpus Audit of SVMs and LLMs on TOEFL11 and ICLEv3.*
> Proceedings of the 15th Workshop on Natural Language Processing for
> Computer-Assisted Language Learning (NLP4CALL 2026), Gothenburg, Sweden.

`CITATION.cff` carries the same information in machine-readable form.
