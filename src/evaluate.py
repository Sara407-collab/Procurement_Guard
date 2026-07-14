"""
Evaluation utilities — used from Day 3 onward, extended but never replaced.

Kept deliberately simple. If a number in a demo is ever questioned, the
person asking should be able to recompute it by hand from these functions in
under a minute.
"""

import pandas as pd


def precision_recall_f1(predicted_ids: set, true_ids: set) -> dict:
    tp = len(predicted_ids & true_ids)
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def precision_at_k(scored: pd.DataFrame, score_col: str, truth_col: str, k: int,
                   tiebreak_col: str = None) -> float:
    """Of the top-k highest-scored tenders, what fraction are actually fraud?
    This is the number that matters most: an auditor only has time to check
    a handful of alerts, not the whole list.

    tiebreak_col is not optional in practice. rule_score is an integer 0-6, so
    2,700 tenders tie at 0 and 44 tie at 2. Without a tiebreaker, "top 20" means
    "whichever 20 tied rows pandas happened to sort first" — the number moves if
    you re-sort the CSV. Break ties by amount: among equally-flagged tenders,
    the biggest money first. That is what an auditor does anyway."""
    by = [score_col] + ([tiebreak_col] if tiebreak_col else [])
    top_k = scored.sort_values(by, ascending=False).head(k)
    return top_k[truth_col].sum() / min(k, len(top_k))


def recall_by_fraud_type(predicted_ids: set, truth: pd.DataFrame) -> pd.DataFrame:
    """For each fraud scheme, what fraction did this rule/model actually catch?
    The gaps here are as informative as the hits — they're the argument for
    why the next layer (graph, anomaly model) needs to exist."""
    fraud = truth[truth["is_fraud"] == 1].copy()
    fraud["types"] = fraud["fraud_type"].str.split("|")
    exploded = fraud.explode("types")
    exploded["caught"] = exploded["tender_id"].isin(predicted_ids)
    result = (exploded.groupby("types")["caught"]
              .agg(caught="sum", total="count", recall="mean")
              .sort_values("recall", ascending=False))
    return result
