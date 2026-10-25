# ============================================================================
# 02b. MASKING FIDELITY AUDIT
# ============================================================================
# Verifies that POS masking did what it is specified to do. The check re-tags
# the pre-masking `text` column with the same spaCy model used for masking
# (en_core_web_sm), then walks token-for-token through the aligned
# original/masked pair, confirming that:
#   (a) every token spaCy tags as NOUN/PROPN/VERB/ADJ/ADV was replaced by its
#       MASK_MAP code in masked_text, and
#   (b) every other token (function words, punctuation, the <P>/<S> boundary
#       markers) was left unchanged.
#
# Precondition. The token-level comparison requires masked_text to be a 1:1
# token-aligned transform of `text`: spaCy's tokenization of `text`, with
# maskable tokens replaced by a single-letter code and the result joined on
# single spaces, so that masked_text.split() recovers the same token count
# and order as the spaCy Doc built from `text`. Notebook 02 constructs
# masked_text this way. Essays whose token counts do not align are reported
# separately as `token_count_mismatch` and excluded from the fidelity rate
# rather than counted as imperfections, so a large mismatch count indicates
# a change in the masking code's tokenization rather than a drop in masking
# fidelity.

import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

MASK_MAP = {
    "NOUN": "N",
    "PROPN": "P",
    "VERB": "V",
    "ADJ": "A",
    "ADV": "R",
}
BOUNDARY_MARKERS = {"<P>", "<S>"}

nlp = spacy.load("en_core_web_sm")

imperfect_rows = []
essays_checked = 0
essays_misaligned = 0
essays_with_imperfection = 0

# Expects `df` (the masked corpus, with `text` and `masked_text` columns) in
# scope; the masking cell in notebook 02 leaves it there. To run standalone,
# uncomment the next two lines and set the path:
# INPUT_PATH = BASE_DIR / "data/processed/combined_nli_masked.csv"
# df = pd.read_csv(INPUT_PATH)

for row in tqdm(df.itertuples(), total=len(df)):
    doc_id = row.doc_id
    original_text = row.text
    masked_text = row.masked_text

    if not isinstance(original_text, str) or not isinstance(masked_text, str):
        continue

    essays_checked += 1
    doc = nlp(original_text)
    orig_tokens = list(doc)
    masked_tokens = masked_text.split()

    if len(orig_tokens) != len(masked_tokens):
        essays_misaligned += 1
        imperfect_rows.append({
            "doc_id": doc_id,
            "issue": "token_count_mismatch",
            "n_original_tokens": len(orig_tokens),
            "n_masked_tokens": len(masked_tokens),
        })
        continue

    essay_flagged = False
    for orig_tok, masked_tok in zip(orig_tokens, masked_tokens):
        if orig_tok.text in BOUNDARY_MARKERS:
            if masked_tok != orig_tok.text:
                essay_flagged = True
                imperfect_rows.append({
                    "doc_id": doc_id,
                    "issue": "boundary_marker_altered",
                    "original_token": orig_tok.text,
                    "original_pos": orig_tok.pos_,
                    "actual_masked_token": masked_tok,
                })
            continue

        expected_code = MASK_MAP.get(orig_tok.pos_)
        if expected_code is not None:
            if masked_tok != expected_code:
                essay_flagged = True
                imperfect_rows.append({
                    "doc_id": doc_id,
                    "issue": "content_word_not_masked",
                    "original_token": orig_tok.text,
                    "original_pos": orig_tok.pos_,
                    "expected_code": expected_code,
                    "actual_masked_token": masked_tok,
                })
        else:
            if masked_tok != orig_tok.text:
                essay_flagged = True
                imperfect_rows.append({
                    "doc_id": doc_id,
                    "issue": "non_content_word_altered",
                    "original_token": orig_tok.text,
                    "original_pos": orig_tok.pos_,
                    "actual_masked_token": masked_tok,
                })

    if essay_flagged:
        essays_with_imperfection += 1

imperfect_df = pd.DataFrame(imperfect_rows)

print(f"Essays checked: {essays_checked}")
print(f"Essays excluded from token-level check due to length mismatch: {essays_misaligned}")
print(f"Essays with >=1 masking imperfection (of the {essays_checked - essays_misaligned} alignable essays): {essays_with_imperfection}")
if (essays_checked - essays_misaligned) > 0:
    rate = essays_with_imperfection / (essays_checked - essays_misaligned)
    print(f"Masking-imperfection rate: {essays_with_imperfection}/{essays_checked - essays_misaligned} = {rate:.4%}")

AUDIT_OUTPUT_PATH = Path("..").resolve() / "data/processed/results/masking_fidelity_audit.csv"
AUDIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
imperfect_df.to_csv(AUDIT_OUTPUT_PATH, index=False)
print(f"Saved flagged imperfections: {AUDIT_OUTPUT_PATH}")
