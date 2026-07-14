"""
Models — Day 4.

Two models, on purpose, doing different jobs:

  ISOLATION FOREST (unsupervised)
      Never sees a label. Asks only: "how weird is this row compared to the
      other 3,000?" This is the layer that would actually run on TMC's real
      SAP data tomorrow, because TMC has no labelled fraud and never will.
      Everything else here is a story about what becomes possible *if* labels
      ever exist. This one works without them.

  XGBOOST (supervised)
      Trained on ground truth. On synthetic data this is the upper bound: the
      best a model can do when it knows the answers. It is not deployable at
      TMC on day one. It is the argument for why labelling a few hundred
      confirmed cases would be worth someone's time.

Both are run with DEFAULT hyperparameters. Not laziness — a deliberate choice.
Tuning a model against labels you invented yourself measures nothing except how
well you tuned. The time is better spent on the graph.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C
from . import features as F


# ═══════════════════════════════════════════════ honest train/test splitting
def ring_components(truth: pd.DataFrame) -> pd.Series:
    """
    A group ID per tender, such that no fraud ring is ever split across folds.

    Why this matters: a cartel plants up to 14 tenders sharing the same four
    vendors. Split them randomly and the model sees vendor V0231 win in the
    training fold, then "predicts" V0231 in the test fold. That is not
    detection, it is memorisation, and it inflates every number.

    Three tenders belong to two rings at once (COLL004|SPLIT007 and friends),
    so rings can overlap. Union-find merges any rings that share a tender into
    one component. Clean tenders each get their own singleton group.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in truth["ring_id"].dropna().unique():
        parts = str(r).split("|")
        for p in parts[1:]:
            union(parts[0], p)

    def group_of(row):
        if pd.isna(row["ring_id"]):
            return f"solo::{row['tender_id']}"
        return f"ring::{find(str(row['ring_id']).split('|')[0])}"

    return truth.apply(group_of, axis=1)


# ═══════════════════════════════════════════════════ 1. unsupervised anomaly
def isolation_forest_score(features: pd.DataFrame) -> np.ndarray:
    """
    Higher = weirder. No labels anywhere in this function — check for yourself.

    Rule flags are deliberately excluded. The Isolation Forest is meant to be
    the layer that knows nothing about procurement: if it is handed the rules'
    conclusions, it stops being an independent signal and the ablation below
    stops meaning anything.
    """
    cols = [c for c in F.feature_columns(features)
            if not c.startswith("rule_") and c != "n_rules_triggered"]
    X = StandardScaler().fit_transform(features[cols].astype(float))

    iso = IsolationForest(
        n_estimators=C.IFOREST_N_ESTIMATORS,
        contamination=C.IFOREST_CONTAMINATION,
        random_state=C.RANDOM_SEED,
    )
    iso.fit(X)                                    # y is not passed. It cannot be.
    return -iso.score_samples(X)                  # flip: bigger number = more anomalous


# ═════════════════════════════════════════════════════ 2. supervised, honest
def xgboost_oof_score(features: pd.DataFrame, y: np.ndarray,
                      groups: np.ndarray, extra_cols: list = None):
    """
    Out-of-fold probabilities: every tender is scored by a model that never saw
    it, nor any other tender from its ring. That makes the score comparable to
    the unsupervised ones — no row is scored by a model that memorised it.

    Returns (scores, gain_importance).
    """
    cols = F.feature_columns(features) + (extra_cols or [])
    X = features[cols].astype(float).to_numpy()

    oof = np.zeros(len(features))
    gain = {}

    for tr, te in GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups):
        m = XGBClassifier(eval_metric="logloss", random_state=C.RANDOM_SEED)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        for k, v in m.get_booster().get_score(importance_type="gain").items():
            gain[cols[int(k[1:])]] = gain.get(cols[int(k[1:])], 0.0) + v

    imp = (pd.Series(gain).sort_values(ascending=False) / C.N_FOLDS)
    return oof, imp


# ══════════════════════════════════════════════════════════════════ blending
def _rank01(x: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0, 1]. Scores from three different models live on
    three different scales; ranks are the only fair way to add them up."""
    return pd.Series(x).rank(pct=True).to_numpy()


def blend(rule_score, iforest_score, xgb_score) -> np.ndarray:
    """
    One number an auditor can sort by.

    The weights are round numbers, chosen and then left alone. Fitting them
    would be fitting to labels we planted, which teaches us nothing — and any
    weight we picked would be indefensible in a room. These are defensible:
    the supervised model carries the most, the rules carry real domain
    knowledge, the anomaly score is a tiebreaker that works without labels.
    """
    return (C.BLEND_WEIGHTS["rules"]    * _rank01(rule_score) +
            C.BLEND_WEIGHTS["iforest"]  * _rank01(iforest_score) +
            C.BLEND_WEIGHTS["xgboost"]  * _rank01(xgb_score))
