"""
Explanations — Day 7.

SHAP tells you which features moved a score, and by how much. That is necessary
and it is not sufficient, because this is what SHAP actually says:

    buyer_exclusivity = 0.78   ->   +0.31

An auditor does not know what buyer_exclusivity is. Nobody outside this repo
does. Handed that line, they cannot act, cannot challenge it, and cannot repeat
it to the person they are about to accuse. A number they don't understand is not
an explanation — it is a second thing to take on faith.

So this file does two jobs:

  1. SHAP, honestly. Each tender is explained by the fold model that never saw
     it. Explaining a row with a model that memorised it would produce a
     confident, beautiful, meaningless story.

  2. TRANSLATION. Every feature carries a sentence, and the sentence carries the
     comparison that makes the number mean something:

        "This vendor gets 78% of all its work from this one buyer.
         A typical vendor gets 11%."

     That is a sentence an auditor can put in an email. That is the deliverable.

The baselines are computed from the clean population, not invented. When the
data changes, the sentences change with it.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from . import config as C
from . import features as F


# ══════════════════════════════════════════════════════════ honest SHAP values
def shap_by_fold(feats: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """
    One SHAP row per tender, from the fold model that did not train on it.

    Same folds, same seed, same ring-awareness as the score it explains. If the
    explanation came from a model that had seen the row, it would be explaining
    a memory, not a prediction.
    """
    cols = F.feature_columns(feats)
    X = feats[cols].astype(float).to_numpy()
    out = np.zeros((len(feats), len(cols)))
    base = np.zeros(len(feats))

    for tr, te in GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups):
        m = XGBClassifier(eval_metric="logloss", random_state=C.RANDOM_SEED)
        m.fit(X[tr], y[tr])
        ex = shap.TreeExplainer(m)
        out[te] = ex.shap_values(X[te])
        base[te] = ex.expected_value

    return pd.DataFrame(out, columns=cols, index=feats.index), base


# ═══════════════════════════════════════════════════ feature -> plain English
# Each entry: (short label, sentence builder). The builder gets the raw value and
# the population baselines, and returns a sentence — OR None.
#
# None is the important part. SHAP will happily tell you that a feature pushed
# the risk up when its value is LOW, because the model learned some interaction
# you cannot see. Rendered naively that produces:
#
#     "These firms turn up on the same tenders 0% of the time."   (+0.54 risk)
#
# which is not evidence, it is nonsense wearing evidence's clothes. An auditor
# who reads one sentence like that stops believing the other four. So every
# builder GUARDS itself: if the value is not actually in the incriminating
# direction, it returns None and the reason is dropped. A short honest
# explanation beats a long one with a lie in it.
def _pct(x):
    return f"{x:.0%}"


TRANSLATIONS = {
    "buyer_exclusivity": (
        "Vendor depends on one buyer",
        lambda v, b: None if v <= b["buyer_exclusivity"] else
                     f"This vendor takes {_pct(v)} of all its work from this one "
                     f"buyer. A typical vendor gets "
                     f"{_pct(b['buyer_exclusivity'])}."),
    "pair_n_tenders": (
        "This pair keeps trading",
        lambda v, b: None if v <= b["pair_n_tenders"] else
                     f"This buyer has awarded this vendor {int(v)} tenders. The "
                     f"usual buyer-vendor pair has done "
                     f"{b['pair_n_tenders']:.0f}."),
    "n_bidders": (
        "Thin competition",
        lambda v, b: None if v > 2 else
                     (f"Only {int(v)} company bid. Nobody else was asked."
                      if v <= 1 else
                      f"Just {int(v)} companies bid. The usual tender draws "
                      f"{b['n_bidders']:.0f}.")),
    "amount_over_limit": (
        "Sitting under the ceiling",
        lambda v, b: None if not (0.75 <= v < 1.0) else
                     f"The award is {_pct(v)} of the approver's limit — just "
                     f"under the ceiling, where a splitter would put it."),
    "headroom_to_limit": (
        "Barely under the limit",
        lambda v, b: None if not (0 < v <= 0.25) else
                     f"It clears the approval limit by only {_pct(v)}. A little "
                     f"more and a director would have had to sign."),
    "is_round": (
        "A suspiciously round number",
        lambda v, b: None if v < 1 else
                     "The amount is a round figure. Real invoices come from "
                     "quantities times prices, and they come out messy. Invented "
                     "ones come out round."),
    "vendor_age_days": (
        "A brand-new supplier",
        lambda v, b: None if v > 365 else
                     f"This vendor was registered only {int(v)} days before "
                     f"winning this. Legitimate suppliers build a track record "
                     f"first."),
    "days_reg_to_first_award": (
        "Registered, then paid, almost at once",
        lambda v, b: None if v > 90 else
                     f"Only {int(v)} days between this vendor appearing in the "
                     f"master file and its first award."),
    "is_vendors_first_award": (
        "Their very first order",
        lambda v, b: None if v < 1 else
                     "This is the first thing this vendor has ever won."),
    "vendor_n_buyers": (
        "Serves almost nobody else",
        lambda v, b: None if v >= b["vendor_n_buyers"] else
                     (f"This vendor works with only {int(v)} buyer in the entire "
                      f"company. Real suppliers spread across "
                      f"{b['vendor_n_buyers']:.0f}." if v <= 1 else
                      f"This vendor works with just {int(v)} buyers. Most work "
                      f"with {b['vendor_n_buyers']:.0f}.")),
    "amount_z_in_category": (
        "Priced above its peers",
        lambda v, b: None if v < 1.0 else
                     f"The price sits {v:.1f} standard deviations above what this "
                     f"category normally costs."),
    "amount_vs_cat_median": (
        "Expensive for what it is",
        lambda v, b: None if v < 1.5 else
                     f"It cost {v:.1f}x the median price in its category."),
    "winner_margin": (
        "The winner cleared the field too easily",
        lambda v, b: None if v < 0.03 else
                     f"The runner-up bid {_pct(v)} more than the winner. In an "
                     f"honest tender the top two are close; a padded cover bid "
                     f"is not."),
    "bid_cv": (
        "The bids were suspiciously alike",
        lambda v, b: None if v >= b["bid_cv"] or v <= 0 else
                     f"The spread across bids is unusually tight ({v:.2f} against "
                     f"a normal {b['bid_cv']:.2f}). Independent firms pricing "
                     f"independently do not land this close together."),
    "days_to_payment": (
        "Paid unusually fast",
        lambda v, b: None if v >= b["days_to_payment"] else
                     f"Paid {int(v)} days after invoice. The company normally "
                     f"takes {b['days_to_payment']:.0f}."),
    # ── the graph ────────────────────────────────────────────────────────────
    "g_direct_shared_ids": (
        "The vendor and the buyer share a bank account",
        lambda v, b: None if v < 1 else
                     f"This vendor and this buyer share {int(v)} identifier"
                     f"{'s' if v > 1 else ''} — a bank account, an address or a "
                     f"phone number. That is not a coincidence."),
    "g_hops_to_buyer": (
        "Linked to the buyer through a shell",
        lambda v, b: None if v < 1 else
                     ("The vendor shares an identifier with this buyer directly."
                      if v == 1 else
                      f"The vendor is {int(v)} steps from this buyer in the "
                      f"network. They share nothing directly — a third company "
                      f"sits in between. A database query looking for a shared "
                      f"bank account would come back empty.")),
    "g_reachable_within_2_hops": (
        "Two steps from the buyer",
        lambda v, b: None if v < 1 else
                     "A shell company links this vendor to the buyer who awarded "
                     "it the work."),
    "g_identity_component_size": (
        "Part of a tangle of shared identities",
        lambda v, b: None if v <= 1 else
                     f"This vendor sits in a cluster of {int(v)} entities all "
                     f"sharing bank accounts, addresses or phone numbers."),
    "g_ring_score": (
        "The vendor belongs to a suspected ring",
        lambda v, b: None if v <= 0 else
                     f"The winner is a member of a suspected bid-rigging ring "
                     f"(ring score {v:.2f}). See the Cartels tab."),
    "g_bidders_in_ring": (
        "Most of the bidders were in on it",
        lambda v, b: None if v <= 0 else
                     f"{_pct(v)} of the companies that bid on this tender belong "
                     f"to the same suspected ring."),
    "g_ring_win_evenness": (
        "They take turns winning",
        lambda v, b: None if v < 0.5 else
                     f"Inside this group the wins are split almost perfectly "
                     f"evenly ({v:.2f} of a possible 1.00). Honest competition "
                     f"has a cheapest supplier, and it wins more than its share."),
    "g_cobid_mean_affinity": (
        "These firms always bid together",
        lambda v, b: None if v <= b["g_cobid_mean_affinity"] else
                     f"The bidders here meet each other far more often than "
                     f"chance allows (affinity {v:.2f}; honest rivals sit near "
                     f"{b['g_cobid_mean_affinity']:.2f})."),
    "g_cobid_min_affinity": (
        "Every pair of bidders is tied to every other",
        lambda v, b: None if v <= b["g_cobid_min_affinity"] else
                     f"Not one loose pair — every bidder here co-bids with every "
                     f"other at {v:.2f} or more. That is a closed group, not a "
                     f"market."),
    "g_jaccard_mean": (
        "The same names, tender after tender",
        lambda v, b: None if v <= b["g_jaccard_mean"] else
                     f"These firms turn up on the same tenders far more than "
                     f"rivals should ({v:.0%} overlap against a normal "
                     f"{b['g_jaccard_mean']:.0%})."),
    # ── the rules ────────────────────────────────────────────────────────────
    "rule_po_splitting":         ("PO splitting rule fired", None),
    "rule_duplicate_invoice":    ("Duplicate invoice rule fired", None),
    "rule_thin_competition":     ("Thin competition rule fired", None),
    "rule_round_near_threshold": ("Round-number rule fired", None),
    "rule_vendor_concentration": ("Vendor concentration rule fired", None),
    "rule_new_vendor_large_first": ("New vendor, large first order rule fired", None),
    "n_rules_triggered": (
        "Several rules fired at once",
        # Only speak when there is actually a "several". One rule firing is
        # common and unremarkable; the individual rule below will say so itself.
        lambda v, b: None if v < 2 else
                     f"{int(v)} separate audit rules flagged this tender "
                     f"independently. Any one of them alone is common. "
                     f"{int(v)} together is not."),
}


def baselines(feats: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """
    What normal looks like. Measured on the CLEAN tenders only — a comparison
    against 'the average tender' is muddied by the fraud sitting inside it.
    """
    clean = feats.merge(truth[["tender_id", "is_fraud"]], on="tender_id")
    clean = clean[clean["is_fraud"] == 0]
    return {c: float(clean[c].median())
            for c in F.feature_columns(feats) if c in clean.columns}


def explain_tender(sv: pd.Series, row: pd.Series, base: dict, top_n: int = 5):
    """The reasons this tender rose, biggest push first, in English."""
    # Only the reasons that pushed the risk UP. A feature that pulled it DOWN is
    # not part of the case against this tender — it is part of the case for it,
    # and listing it next to the evidence just muddies what the auditor is being
    # asked to look at.
    ranked = sv[sv > 0].sort_values(ascending=False)
    out = []
    for feat, contrib in ranked.items():
        if len(out) >= top_n or contrib < 0.005:
            break
        if feat not in TRANSLATIONS:
            continue
        label, builder = TRANSLATIONS[feat]
        val = row[feat]
        if builder is None:                     # a rule flag: only speak if it fired
            if val < 1:
                continue
            text = "An audit rule written by a human, not a model, flagged this."
        else:
            try:
                text = builder(val, base)
            except Exception:
                continue
            if text is None:        # the guard fired: this value is not evidence
                continue
        out.append({"feature": feat, "label": label, "text": text,
                    "contribution": float(contrib), "value": float(val),
                    "direction": "up" if contrib > 0 else "down"})
    return out


# ══════════════════════════════════════════════════════════════════════ public
def build_explanations(feats, truth, y, groups) -> pd.DataFrame:
    """One row per tender: the top reasons, ready for the dashboard to render."""
    sv, base_val = shap_by_fold(feats, y, groups)
    base = baselines(feats, truth)

    rows = []
    for i, tid in enumerate(feats["tender_id"]):
        reasons = explain_tender(sv.iloc[i], feats.iloc[i], base)
        for r in reasons:
            rows.append({"tender_id": tid, **r})
    return pd.DataFrame(rows)
