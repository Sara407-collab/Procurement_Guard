"""
ProcurementGuard — Day 3: rules engine.

    python -m src.run_rules

Reads the CSVs Day 1-2 produced, runs all six rules, validates every one of
them against ground_truth.csv, and writes data/rule_flags.csv for Day 4 to
build on.
"""

import pandas as pd

from . import config as C
from . import rules, evaluate as ev


def load():
    tenders = pd.read_csv(C.FILES["tenders"])
    vendors = pd.read_csv(C.FILES["vendors"])
    truth = pd.read_csv(C.FILES["labels"])
    return tenders, vendors, truth


def report(flags: pd.DataFrame, truth: pd.DataFrame, tenders: pd.DataFrame):
    truth_ids = set(truth[truth["is_fraud"] == 1]["tender_id"])
    n_fraud = len(truth_ids)
    n_total = len(truth)

    print("\n" + "═" * 68)
    print("  RULES ENGINE — Day 3")
    print("═" * 68)
    print(f"  {n_fraud:,} fraudulent tenders out of {n_total:,} "
          f"({n_fraud/n_total:.2%}) — this is what the rules are hunting for\n")

    rule_cols = [c for c in flags.columns if c.startswith("rule_") and flags[c].dtype == bool]

    print(f"  {'rule':<28}{'flagged':>9}{'precision':>12}{'recall':>10}{'f1':>8}")
    print("  " + "─" * 66)
    for col in rule_cols:
        pred_ids = set(flags[flags[col]]["tender_id"])
        m = ev.precision_recall_f1(pred_ids, truth_ids)
        print(f"  {col.replace('rule_',''):<28}{len(pred_ids):>9,}"
              f"{m['precision']:>12.1%}{m['recall']:>10.1%}{m['f1']:>8.2f}")

    print("  " + "─" * 66)

    # combined: "any rule fired" vs "at least 2 rules fired"
    for thresh, label in [(1, "any rule fires"), (2, "2+ rules fire")]:
        pred_ids = set(flags[flags["n_rules_triggered"] >= thresh]["tender_id"])
        m = ev.precision_recall_f1(pred_ids, truth_ids)
        print(f"  {label:<28}{len(pred_ids):>9,}"
              f"{m['precision']:>12.1%}{m['recall']:>10.1%}{m['f1']:>8.2f}")

    print("═" * 68)

    # the number that matters most: can an auditor trust the top of the list?
    merged = (flags.merge(truth[["tender_id", "is_fraud"]], on="tender_id")
                   .merge(tenders[["tender_id", "amount"]], on="tender_id"))
    for k in (20, 50, 100):
        p = ev.precision_at_k(merged, "rule_score", "is_fraud", k, tiebreak_col="amount")
        print(f"  Precision@{k:<5} {p:>6.1%}   "
              f"(of the top {k} by rule_score, this fraction are real)")

    print("═" * 68)
    print("  Recall by fraud type — where rules alone fall short:")
    pred_any = set(flags[flags["n_rules_triggered"] >= 1]["tender_id"])
    breakdown = ev.recall_by_fraud_type(pred_any, truth)
    for scheme, row in breakdown.iterrows():
        bar = "█" * int(row["recall"] * 20)
        print(f"  {scheme:<22} {row['recall']:>6.1%}  {bar:<20}"
              f" ({int(row['caught'])}/{int(row['total'])})")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    tenders, vendors, truth = load()

    flags = rules.run_all_rules(tenders, vendors)
    flags.to_csv(C.FILES["rule_flags"], index=False)

    report(flags, truth, tenders)
    print(f"  written → {C.FILES['rule_flags']}\n")
