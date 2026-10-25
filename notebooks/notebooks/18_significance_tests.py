"""
18. Significance tests for the accuracy gaps and confound splits.

Reproduces the McNemar and chi-square results reported in the paper:
  - full-text vs masked accuracy gaps on TOEFL11 (raw and clean tiers)
  - self-disclosure split (LLM and SVM)
  - topic-exclusivity split (LLM and SVM)
  - label-composition control for the topic-exclusivity split

Run after notebooks 10, 11 and 12, which produce the prediction and
flag files this script reads.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, chi2_contingency, fisher_exact

BASE_DIR = Path("..").resolve()
RESULTS_DIR = BASE_DIR / "data/processed/results"


def mcnemar_exact(n_only_a: int, n_only_b: int) -> float:
    """Exact binomial McNemar test on the discordant pairs."""
    n = n_only_a + n_only_b
    if n == 0:
        return float("nan")
    return binomtest(min(n_only_a, n_only_b), n, 0.5).pvalue


def load_llm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["error"].fillna("").eq("") & df["predicted_l1"].notna()].copy()


eval_df = pd.read_csv(RESULTS_DIR / "llm_eval_subset.csv")
toefl = eval_df[eval_df["eval_partition"] == "toefl11_test"][["doc_id", "label"]].copy()

flagged = pd.read_csv(RESULTS_DIR / "leakage_flagged_essays.csv")
self_ref_ids = set(flagged.loc[flagged["self_referential_in_text"] == True, "doc_id"])

topic_flags = pd.read_csv(RESULTS_DIR / "icle_topic_flags.csv")
topic_excl_ids = set(topic_flags.loc[topic_flags["topic_exclusive_to_own_label"] == True, "doc_id"])

arms = {
    ("LLM", "forced-JSON", "fulltext"): load_llm(RESULTS_DIR / "llm_predictions_fulltext.csv").set_index("doc_id")["predicted_l1"],
    ("LLM", "forced-JSON", "masked"): load_llm(RESULTS_DIR / "llm_predictions.csv").set_index("doc_id")["predicted_l1"],
    ("LLM", "reasoning", "fulltext"): load_llm(RESULTS_DIR / "llm_predictions_fulltext_reasoning.csv").set_index("doc_id")["predicted_l1"],
    ("LLM", "reasoning", "masked"): load_llm(RESULTS_DIR / "llm_predictions_masked_reasoning.csv").set_index("doc_id")["predicted_l1"],
    ("SVM", "n/a", "fulltext"): pd.read_csv(RESULTS_DIR / "svm_predictions_fulltext.csv").set_index("doc_id")["svm_predicted_l1"],
    ("SVM", "n/a", "masked"): pd.read_csv(RESULTS_DIR / "svm_predictions_full.csv").set_index("doc_id")["svm_predicted_l1"],
}

SYSTEMS = [("LLM", "forced-JSON"), ("LLM", "reasoning"), ("SVM", "n/a")]


def gap_test(subset: pd.DataFrame, tier_name: str) -> None:
    print(f"\n=== Full-text vs masked gap on TOEFL11, {tier_name} tier (n={len(subset)}) ===")
    for system, fmt in SYSTEMS:
        preds_full = arms[(system, fmt, "fulltext")]
        preds_mask = arms[(system, fmt, "masked")]
        full_correct, mask_correct = {}, {}
        for doc_id, true_l1 in zip(subset["doc_id"], subset["label"]):
            if doc_id in preds_full.index and doc_id in preds_mask.index:
                full_correct[doc_id] = preds_full.loc[doc_id] == true_l1
                mask_correct[doc_id] = preds_mask.loc[doc_id] == true_l1
        common = set(full_correct) & set(mask_correct)
        only_full = sum(1 for d in common if full_correct[d] and not mask_correct[d])
        only_mask = sum(1 for d in common if mask_correct[d] and not full_correct[d])
        acc_full = sum(full_correct.values()) / len(full_correct)
        acc_mask = sum(mask_correct.values()) / len(mask_correct)
        print(
            f"  {system} {fmt}: full={acc_full:.1%} masked={acc_mask:.1%} "
            f"gap={100 * (acc_full - acc_mask):.1f}pp | "
            f"discordant {only_full} vs {only_mask}, p={mcnemar_exact(only_full, only_mask):.2e}"
        )


gap_test(toefl, "raw")
gap_test(toefl[~toefl["doc_id"].isin(self_ref_ids)], "clean")


def split_test(df: pd.DataFrame, flag_col: str, name: str) -> None:
    table = pd.crosstab(df[flag_col], df["correct"])
    chi2, p, _, _ = chi2_contingency(table, correction=True)
    print(f"\n=== {name} (n={len(df)}) ===")
    print(table)
    print(f"  chi2(1)={chi2:.2f}, p={p:.3e}")
    for value, group in df.groupby(flag_col):
        print(f"  {flag_col}={value}: accuracy={group['correct'].mean():.1%} (n={len(group)})")


eval_ids = set(eval_df["doc_id"])

llm_full = load_llm(RESULTS_DIR / "llm_predictions_fulltext.csv")
llm_full = llm_full[llm_full["doc_id"].isin(eval_ids)].copy()
llm_full["correct"] = llm_full["predicted_l1"] == llm_full["true_l1"]
llm_full["self_disclosed"] = llm_full["doc_id"].isin(self_ref_ids)
split_test(llm_full, "self_disclosed", "Self-disclosure split, LLM full-text forced-JSON")

svm_full = pd.read_csv(RESULTS_DIR / "svm_predictions_fulltext.csv")
svm_full["correct"] = svm_full["svm_predicted_l1"] == svm_full["true_l1"]
svm_full["self_disclosed"] = svm_full["doc_id"].isin(self_ref_ids)
split_test(svm_full, "self_disclosed", "Self-disclosure split, SVM full-text (full corpus)")

icle_clean = llm_full[
    (llm_full["eval_partition"] == "iclev3_external_test") & (~llm_full["self_disclosed"])
].copy()
icle_clean["topic_exclusive"] = icle_clean["doc_id"].isin(topic_excl_ids)
split_test(icle_clean, "topic_exclusive", "Topic-exclusivity split, LLM full-text (self-disclosure-clean ICLE)")

table = pd.crosstab(icle_clean["topic_exclusive"], icle_clean["correct"])
print(f"  Fisher exact p={fisher_exact(table.values)[1]:.4f}")

# Label-composition control: restrict to labels present in both groups.
shared = [
    label for label in icle_clean["true_l1"].unique()
    if icle_clean.loc[icle_clean["true_l1"] == label, "topic_exclusive"].nunique() == 2
]
controlled = icle_clean[icle_clean["true_l1"].isin(shared)]
print(f"\n=== Topic-exclusivity split restricted to labels in both groups {sorted(shared)} ===")
for value, group in controlled.groupby("topic_exclusive"):
    print(f"  topic_exclusive={value}: accuracy={group['correct'].mean():.1%} (n={len(group)})")

for condition, path in [
    ("masked", "svm_predictions_full.csv"),
    ("fulltext", "svm_predictions_fulltext.csv"),
]:
    svm = pd.read_csv(RESULTS_DIR / path)
    svm = svm[svm["partition"] == "iclev3_external_test"].copy()
    svm["correct"] = svm["svm_predicted_l1"] == svm["true_l1"]
    svm = svm[~svm["doc_id"].isin(self_ref_ids)].copy()
    svm["topic_exclusive"] = svm["doc_id"].isin(topic_excl_ids)
    split_test(svm, "topic_exclusive", f"Topic-exclusivity split, SVM {condition} (self-disclosure-clean ICLE)")
