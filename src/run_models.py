"""
ProcurementGuard — Day 4: features, anomaly detection, supervised model.

    python -m src.run_models

Reads what Days 1-3 produced. Builds the feature table, audits it for leaks,
scores every tender three ways, blends them, and writes data/risk_scores.csv
for the dashboard to serve.

The report it prints is the demo. Read it top to bottom and you have the whole
argument: what the rules alone can do, what a model with no labels adds, what a
model with labels adds — and the one scheme none of them can touch.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config as C
from . import features as F
from . import models as M
from . import graph as GR
from . import explain as EX
from . import evaluate as ev


def load():
    return (pd.read_csv(C.FILES["tenders"]),
            pd.read_csv(C.FILES["bids"]),
            pd.read_csv(C.FILES["vendors"]),
            pd.read_csv(C.FILES["employees"]),
            pd.read_csv(C.FILES["labels"]),
            pd.read_csv(C.FILES["rule_flags"]))


def scheme_recall_at_k(scored: pd.DataFrame, truth: pd.DataFrame,
                       score_col: str, k: int) -> pd.DataFrame:
    """Of each scheme's tenders, how many make it into the top k alerts?
    An auditor works a queue. This is the only recall number that matters."""
    top = set(scored.nlargest(k, score_col)["tender_id"])
    fraud = truth[truth["is_fraud"] == 1].copy()
    fraud["types"] = fraud["fraud_type"].str.split("|")
    ex = fraud.explode("types")
    ex["caught"] = ex["tender_id"].isin(top)
    return (ex.groupby("types")["caught"]
              .agg(caught="sum", total="count", recall="mean")
              .sort_values("recall", ascending=False))


def main():
    tenders, bids, vendors, employees, truth, flags = load()

    # ── 1. features, and the tripwire ───────────────────────────────────────
    gfeats, Gi, Gc = GR.build_graph_features(tenders, bids, vendors, employees)
    feats_ng = F.build_features(tenders, bids, vendors, employees, flags)
    feats = F.build_features(tenders, bids, vendors, employees, flags, gfeats)
    audit = F.audit_features(feats, truth)          # raises if anything leaks

    y = feats[["tender_id"]].merge(truth, on="tender_id")["is_fraud"].to_numpy()
    groups = M.ring_components(
        feats[["tender_id"]].merge(truth, on="tender_id")).to_numpy()

    print("\n" + "═" * 72)
    print("  PROCUREMENTGUARD — Day 4: anomaly detection + supervised model")
    print("═" * 72)
    print(f"  {len(feats):,} tenders   {len(F.feature_columns(feats))} features   "
          f"{y.sum()} fraudulent ({y.mean():.2%})")
    print(f"  {pd.Series(groups).nunique():,} ring-components — no fraud ring is "
          f"split across folds")
    print(f"  ✓ leak audit passed: no single feature reaches AUC "
          f"{C.LEAK_AUC_CEILING} on its own")
    print(f"    (strongest lone feature: {audit.iloc[0]['feature']} @ "
          f"{audit.iloc[0]['auc']:.3f})")

    # ── 2. score it three ways ──────────────────────────────────────────────
    s_rules = flags.set_index("tender_id").loc[feats["tender_id"], "n_rules_triggered"].to_numpy()
    s_iso = M.isolation_forest_score(feats_ng)

    # The ablation the whole project exists to produce: the SAME model, the SAME
    # folds, the SAME seed — once without the graph, once with. Any difference
    # is the graph, and nothing else.
    s_xgb_ng, _ = M.xgboost_oof_score(feats_ng, y, groups)
    s_xgb, importance = M.xgboost_oof_score(feats, y, groups)
    s_blend_ng = M.blend(s_rules, s_iso, s_xgb_ng)
    s_blend = M.blend(s_rules, s_iso, s_xgb)

    scored = feats[["tender_id", "vendor_id", "employee_id", "amount"]].copy()
    scored["rule_score"] = s_rules
    scored["iforest_score"] = s_iso
    scored["xgb_score"] = s_xgb
    scored["risk_score_nograph"] = s_blend_ng
    scored["risk_score"] = s_blend
    scored["is_fraud"] = y

    # Expected loss = probability x money. Use XGBoost's calibrated probability,
    # NOT the blended rank: risk_score is a rank in [0,1], and multiplying a
    # rank by an amount that spans three orders of magnitude just re-sorts by
    # amount. A probability times money is a currency figure and means something.
    scored["expected_loss"] = scored["xgb_score"] * scored["amount"]

    # ── 3. the ablation ─────────────────────────────────────────────────────
    layers = [
        ("Rules only",              "rule_score",    "6 pandas checks, no ML"),
        ("Isolation Forest only",   "iforest_score", "NO LABELS USED"),
        ("Rules + Isolation Forest","_ri",           "still no labels for the IF"),
        ("Full blend (+ XGBoost)",  "risk_score_nograph", "Day 4 — tables only"),
        ("+ GRAPH (Day 5)",         "risk_score",    "vendor<->employee, vendor<->vendor"),
    ]
    scored["_ri"] = (0.6 * pd.Series(s_rules).rank(pct=True).to_numpy()
                     + 0.4 * pd.Series(s_iso).rank(pct=True).to_numpy())

    print("\n" + "─" * 72)
    print(f"  {'layer':<28}{'PR-AUC':>9}{'P@20':>9}{'P@50':>9}{'P@100':>9}")
    print("─" * 72)
    for name, col, _ in layers:
        ap = average_precision_score(y, scored[col])
        row = "".join(
            f"{ev.precision_at_k(scored, col, 'is_fraud', k, tiebreak_col='amount'):>8.0%} "
            for k in (20, 50, 100))
        print(f"  {name:<28}{ap:>9.3f} {row}")
    print("─" * 72)
    for name, _, note in layers:
        print(f"  {name:<28} {note}")

    # ── 4. where each layer still fails ─────────────────────────────────────
    print("\n" + "═" * 72)
    print("  Recall by scheme, inside the top 100 alerts an auditor would open")
    print("═" * 72)
    before = scheme_recall_at_k(scored, truth, "risk_score_nograph", 100)
    after = scheme_recall_at_k(scored, truth, "risk_score", 100)
    print(f"  {'scheme':<22}{'Day 4':>9}{'Day 5':>9}{'change':>11}")
    print("  " + "─" * 53)
    for scheme in after.index:
        b, a = before.loc[scheme, "recall"], after.loc[scheme, "recall"]
        mark = "   <<<" if (a - b) > 0.10 else ""
        print(f"  {scheme:<22}{b:>8.1%}{a:>9.1%}{a - b:>+10.1%}{mark}")

    # ── 5. the slide that lands ─────────────────────────────────────────────
    base = y.mean()
    hits = scored.nlargest(20, "risk_score")["is_fraud"].sum()
    print("\n" + "═" * 72)
    print("  An auditor has time for 20 tenders this week.")
    print("═" * 72)
    print(f"  Picking at random          →  {20 * base:>4.1f} real cases "
          f"(the {base:.1%} base rate)")
    print(f"  ProcurementGuard's top 20  →  {hits:>4.0f} real cases")
    print(f"  {'':26}    {hits / (20 * base):>4.1f}x more fraud found, "
          f"same 20 hours of work")
    # ── 5a. the ring detector: the graph's own answer, no labels anywhere ───
    rings = GR.find_rings(Gc, bids)
    cart = truth[truth["fraud_type"].str.contains("bid_rigging")]
    true_rings = {}
    for r in cart.merge(tenders[["tender_id", "vendor_id"]], on="tender_id").itertuples():
        for part in str(r.ring_id).split("|"):
            if part.startswith("CARTEL"):
                true_rings.setdefault(part, set()).add(r.vendor_id)

    print("\n" + "═" * 72)
    print("  SUSPECTED CARTELS — what the graph found on its own")
    print("═" * 72)
    print("  Not one label was used to produce this table. Only who bids against")
    print("  whom, and who takes their turn winning.\n")
    print(f"  {'#':<3}{'vendors':<34}{'tenders':>8}{'affinity':>10}{'evenness':>10}{'score':>8}  verdict")
    print("  " + "─" * 84)
    for i, r in rings.head(6).iterrows():
        got = set(r.vendors.split(","))
        hit = next((k for k, v in true_rings.items() if got == v), None)
        near = next((k for k, v in true_rings.items() if len(got & v) >= 3), None)
        verdict = (f"** {hit} — EXACT **" if hit
                   else f"partial {near}" if near else "false positive")
        print(f"  {i+1:<3}{r.vendors:<34}{r.n_tenders:>8}{r.mean_affinity:>10.3f}"
              f"{r.win_evenness:>10.3f}{r.ring_score:>8.2f}  {verdict}")
    print("  " + "─" * 84)
    exact = sum(1 for _, r in rings.head(2).iterrows()
                if set(r.vendors.split(",")) in true_rings.values())
    print(f"  {len(true_rings)} cartels were planted. The top {len(true_rings)} "
          f"suspects match {exact} of them EXACTLY — every member, no strays.")
    print()
    print("  This is why the graph exists. XGBoost could not do it and never will:")
    print("  with only 2 rings in the data, leave-one-ring-out means training on ONE")
    print("  example of a cartel. No company on earth has 100 labelled cartels to")
    print("  learn from. The graph needs none — it needs no labels at all.")

    # ── 5b. two different queues, and they are not the same queue ───────────
    exposure = scored.loc[scored.is_fraud == 1, "amount"].sum()
    print("\n" + "═" * 72)
    print("  Two ways to sort the queue. They disagree, and the disagreement matters.")
    print("═" * 72)
    print(f"  Total fraud exposure in the book: ${exposure:,.0f}\n")
    print(f"  {'sort by':<24}{'cases found':>13}{'fraud $ recovered':>20}{'% of exposure':>15}")
    print("  " + "─" * 70)
    for col, label in [("risk_score", "risk_score"), ("expected_loss", "expected_loss")]:
        for k in (20, 50):
            top = scored.nlargest(k, col)
            f = top[top.is_fraud == 1]
            print(f"  {label + f' (top {k})':<24}{len(f):>7} / {k:<3}"
                  f"${f['amount'].sum():>19,.0f}{f['amount'].sum() / exposure:>14.1%}")
    print("  " + "─" * 70)
    print("  risk_score finds MORE CASES. expected_loss finds BIGGER ONES.")
    print("  Neither is wrong. The auditor picks, and the dashboard offers both.")

    # ── 6. what the model leaned on ─────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  What XGBoost actually used (gain)")
    print("═" * 72)
    for k, v in importance.head(10).items():
        print(f"  {k:<28}{v:>10.1f}")

    # ── 7. why. Day 7: SHAP, from the fold model that never saw the row, then
    #        translated out of feature names and into English.
    print("\n" + "═" * 72)
    print("  Explaining every alert (SHAP, per fold)")
    print("═" * 72)
    expl = EX.build_explanations(feats, truth, y, groups)
    n_with = expl["tender_id"].nunique()
    print(f"  {len(expl):,} reasons across {n_with:,} tenders "
          f"({len(expl) / max(n_with, 1):.1f} per tender)")
    print("\n  A worked example — the highest-risk tender in the book:\n")
    top1 = scored.nlargest(1, "risk_score")["tender_id"].iloc[0]
    for r in expl[expl["tender_id"] == top1].itertuples():
        arrow = "▲" if r.direction == "up" else "▼"
        print(f"  {arrow} {r.contribution:+.3f}  {r.label}")
        print(f"           {r.text}")

    rings.to_csv(C.FILES["rings"], index=False)
    expl.to_csv(C.FILES["explanations"], index=False)
    scored.drop(columns=["_ri"]).to_csv(C.FILES["risk_scores"], index=False)
    feats.to_csv(C.FILES["features"], index=False)
    print("\n" + "═" * 72)
    print(f"  written → {C.FILES['risk_scores']}")
    print(f"  written → {C.FILES['features']}")
    print(f"  written → {C.FILES['rings']}")
    print(f"  written → {C.FILES['explanations']}\n")


if __name__ == "__main__":
    main()
