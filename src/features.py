"""
Feature engineering — Day 4.

One row per tender. Everything a model is allowed to see.

Two rules govern this file, and they are not negotiable:

  1. NOTHING from ground_truth.csv enters here. Not is_fraud, not fraud_type,
     not ring_id. The labels are for scoring, never for building.

  2. NOTHING that only the generator could know enters here. `is_ghost` sits in
     vendors.csv looking exactly like a feature. It is not one — it is ground
     truth wearing a feature's clothes. No real SAP vendor master has a column
     that says "this supplier is fake". C.LEAKY_COLUMNS names it, and
     audit_features() below refuses to let it (or anything like it) through.

The second rule exists because we already got this wrong once. The first model
hit 0.95 PR-AUC by exploiting four generator bugs. It was not detecting fraud;
it was detecting our own mistakes. audit_features() is the tripwire that stops
it happening twice.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C


# ══════════════════════════════════════════════════════════════ the feature table
def build_features(tenders: pd.DataFrame,
                   bids: pd.DataFrame,
                   vendors: pd.DataFrame,
                   employees: pd.DataFrame,
                   rule_flags: pd.DataFrame,
                   graph_feats: pd.DataFrame = None) -> pd.DataFrame:
    """Returns one row per tender: tender_id + every feature the model may see."""
    df = tenders.copy()
    for col in ("award_date", "invoice_date", "payment_date"):
        df[col] = pd.to_datetime(df[col])

    # ── the transaction itself ───────────────────────────────────────────────
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_over_limit"] = df["amount"] / df["approval_limit"]
    df["headroom_to_limit"] = (df["approval_limit"] - df["amount"]) / df["approval_limit"]
    df["is_round"] = (df["amount"] % C.RULE_THRESHOLDS["round_to"] == 0).astype(int)
    df["n_bidders"] = df["n_bidders"].astype(int)
    df["days_to_invoice"] = (df["invoice_date"] - df["award_date"]).dt.days
    df["days_to_payment"] = (df["payment_date"] - df["invoice_date"]).dt.days

    for col in ("award_method", "category", "department"):
        df[f"{col}_code"] = df[col].astype("category").cat.codes

    # ── price against its peers ──────────────────────────────────────────────
    # An inflated award is only visible relative to what the same category
    # normally costs. This is the honest way to see collusion's overcharge:
    # not by comparing the award to its own bid (that was the leak), but by
    # comparing it to the market.
    g = df.groupby("category")["amount"]
    df["amount_z_in_category"] = (df["amount"] - g.transform("mean")) / g.transform("std")
    df["amount_vs_cat_median"] = df["amount"] / g.transform("median")

    # ── the vendor ───────────────────────────────────────────────────────────
    reg = pd.to_datetime(vendors.set_index("vendor_id")["registration_date"])
    df["vendor_age_days"] = (df["award_date"] - df["vendor_id"].map(reg)).dt.days
    df["vendor_n_tenders"] = df.groupby("vendor_id")["tender_id"].transform("count")
    df["vendor_mean_amount"] = df.groupby("vendor_id")["amount"].transform("mean")
    df["vendor_n_buyers"] = df.groupby("vendor_id")["employee_id"].transform("nunique")
    first_award = df.groupby("vendor_id")["award_date"].transform("min")
    df["is_vendors_first_award"] = (df["award_date"] == first_award).astype(int)
    df["days_reg_to_first_award"] = (first_award - df["vendor_id"].map(reg)).dt.days

    # ── the buyer ────────────────────────────────────────────────────────────
    df["buyer_n_tenders"] = df.groupby("employee_id")["tender_id"].transform("count")
    df["buyer_mean_amount"] = df.groupby("employee_id")["amount"].transform("mean")
    df["buyer_n_vendors"] = df.groupby("employee_id")["vendor_id"].transform("nunique")

    # ── the relationship between them ────────────────────────────────────────
    # buyer_exclusivity is the single most useful number here. A ghost vendor
    # gets 1.00 of its work from the buyer who invented it. A colluding vendor
    # ~0.55. An honest vendor spreads across buyers and lands near 0.11.
    pair_n = df.groupby(["vendor_id", "employee_id"])["tender_id"].transform("count")
    pair_amt = df.groupby(["vendor_id", "employee_id"])["amount"].transform("sum")
    df["pair_n_tenders"] = pair_n
    df["buyer_exclusivity"] = pair_n / df["vendor_n_tenders"]
    df["pair_share_of_buyer"] = pair_amt / df.groupby("employee_id")["amount"].transform("sum")

    # ── the bidding ──────────────────────────────────────────────────────────
    # n_bidders now equals the bid-row count by invariant, so counting rows adds
    # nothing. What the bid table still knows, and the tender table does not, is
    # the SHAPE of the competition — and that is where a cartel might show.
    b = bids.sort_values("bid_amount")
    stats = b.groupby("tender_id")["bid_amount"].agg(["min", "max", "mean", "std"])
    stats.columns = ["bid_min", "bid_max", "bid_mean", "bid_std"]
    second = (b[~b["is_winner"]].groupby("tender_id")["bid_amount"].min()
              .rename("second_lowest_bid"))
    df = df.merge(stats, on="tender_id", how="left").merge(second, on="tender_id", how="left")

    df["bid_cv"] = df["bid_std"] / df["bid_mean"]
    df["bid_spread"] = (df["bid_max"] - df["bid_min"]) / df["bid_mean"]
    # In an honest tender the winner just edges out the runner-up. A cartel's
    # cover bids are deliberately padded, so the winner clears the field by a
    # comfortable, suspiciously consistent margin.
    df["winner_margin"] = (df["second_lowest_bid"] - df["amount"]) / df["amount"]
    df["winner_margin"] = df["winner_margin"].fillna(0.0)   # single-source: no runner-up

    # ── what the rules already said ──────────────────────────────────────────
    rule_cols = [c for c in rule_flags.columns
                 if c.startswith("rule_") and c != "rule_reasons"]
    df = df.merge(rule_flags[["tender_id", "n_rules_triggered"] + rule_cols],
                  on="tender_id", how="left")
    for c in rule_cols:
        df[c] = df[c].astype(int)

    # ── the graph (Day 5) ────────────────────────────────────────────────────
    # Everything above this line is a property of a row, or of rows grouped by a
    # key. Everything below only exists BETWEEN records. Passing graph_feats=None
    # reproduces the Day 4 model exactly — same features, same folds, same seed.
    # That is what makes the ablation honest: one thing changes, and one only.
    if graph_feats is not None:
        df = df.merge(graph_feats, on="tender_id", how="left")

    # ── assemble ─────────────────────────────────────────────────────────────
    ids = ["tender_id", "vendor_id", "employee_id", "amount"]
    drop = ["po_id", "award_date", "invoice_date", "payment_date", "invoice_id",
            "department", "category", "award_method", "approval_limit",
            "bid_min", "bid_max", "bid_mean", "bid_std", "second_lowest_bid",
            "rule_score"]
    feats = [c for c in df.columns if c not in ids + drop]

    out = df[ids + feats].copy()
    out[feats] = out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def feature_columns(features: pd.DataFrame) -> list:
    """Everything the model may train on. IDs and the raw amount stay out —
    amount reappears at the end as expected_loss, not as a feature."""
    return [c for c in features.columns
            if c not in ("tender_id", "vendor_id", "employee_id", "amount")]


# ═══════════════════════════════════════════════════════════════ the tripwire
def audit_features(features: pd.DataFrame, truth: pd.DataFrame,
                   raise_on_fail: bool = True) -> pd.DataFrame:
    """
    Ask every feature, one at a time: could YOU alone separate fraud from clean?

    A single column with an AUC near 1.0 is not a brilliant feature. It is a
    bug. Real fraud signals are weak and overlapping — that is what makes fraud
    hard and models necessary. Anything that separates perfectly is separating
    on something we accidentally built, not on something a fraudster did.

    This is the Day 4 counterpart of assert_no_leakage() in main.py, and it
    exists for the same reason: we already shipped four leaks once.
    """
    for col in C.LEAKY_COLUMNS:
        assert col not in features.columns, (
            f"'{col}' is in the feature table. It is generator ground truth, "
            f"not an observable — no real vendor master carries it.")
    for col in ("is_fraud", "fraud_type", "ring_id"):
        assert col not in features.columns, f"'{col}' is a LABEL. It cannot be a feature."

    y = features[["tender_id"]].merge(truth[["tender_id", "is_fraud"]], on="tender_id")["is_fraud"]

    rows = []
    for col in feature_columns(features):
        x = features[col]
        if x.nunique() < 2:
            continue
        auc = roc_auc_score(y, x)
        rows.append({"feature": col, "auc": max(auc, 1 - auc)})   # direction-agnostic

    audit = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    suspects = audit[audit["auc"] >= C.LEAK_AUC_CEILING]

    if len(suspects) and raise_on_fail:
        raise AssertionError(
            "LEAK. These features separate fraud almost perfectly on their own:\n"
            + suspects.to_string(index=False)
            + f"\n\nNo honest procurement signal has AUC >= {C.LEAK_AUC_CEILING}. "
              "Something in the generator is telling the model the answer. "
              "Find it before you trust a single number downstream.")
    return audit
