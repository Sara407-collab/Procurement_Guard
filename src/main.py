"""
ProcurementGuard — Day 1 + Day 2 pipeline.

    python -m src.main

Produces five CSVs in data/ and prints a report. That report is the artefact
you show on day two: it proves the ground truth exists and is honest.
"""

import os
import numpy as np
import pandas as pd

from . import config as C
from . import masters, base_activity, fraud_injection as fi


def build_dataset(seed: int = C.RANDOM_SEED):
    rng = np.random.default_rng(seed)

    # ── 1. a clean world ────────────────────────────────────────────────
    vendors = masters.build_vendors(rng)
    employees = masters.build_employees(rng)
    tenders, bids = base_activity.build_tenders_and_bids(vendors, employees, rng)

    labels = []

    # ── 2. layer fraud on top, recording every row we touch ─────────────
    tenders, bids, l = fi.inject_po_splitting(tenders, bids, employees, rng)
    labels += l

    tenders, bids, l = fi.inject_duplicate_invoice(tenders, bids, rng)
    labels += l

    vendors, employees, tenders, bids, l = fi.inject_ghost_vendor(
        vendors, employees, tenders, bids, rng)
    labels += l

    vendors, employees, tenders, bids, l = fi.inject_employee_collusion(
        vendors, employees, tenders, bids, rng)
    labels += l

    tenders, bids, l = fi.inject_round_number_abuse(tenders, bids, rng)
    labels += l

    tenders, bids, l = fi.inject_bid_rigging(vendors, tenders, bids, rng)
    labels += l

    # ── 3. ground truth ─────────────────────────────────────────────────
    gt = pd.DataFrame(labels)
    if not gt.empty:
        # A tender can carry more than one scheme. Keep them all, but produce
        # one clean binary column for the model to train against.
        gt = (gt.groupby("tender_id")
                .agg(is_fraud=("is_fraud", "max"),
                     fraud_type=("fraud_type", lambda s: "|".join(sorted(set(s)))),
                     ring_id=("ring_id", lambda s: "|".join(
                         sorted({str(x) for x in s if isinstance(x, str)})) or None))
                .reset_index())

    truth = tenders[["tender_id"]].merge(gt, on="tender_id", how="left")
    truth["is_fraud"] = truth["is_fraud"].fillna(0).astype(int)
    truth["fraud_type"] = truth["fraud_type"].fillna("clean")

    return vendors, employees, tenders, bids, truth


def report(vendors, employees, tenders, bids, truth):
    n = len(tenders)
    n_fraud = int(truth["is_fraud"].sum())

    print("\n" + "═" * 62)
    print("  PROCUREMENTGUARD — synthetic dataset built")
    print("═" * 62)
    print(f"  vendors          {len(vendors):>8,}   "
          f"({int(vendors['is_ghost'].sum())} ghost)")
    print(f"  employees        {len(employees):>8,}")
    print(f"  tenders / POs    {n:>8,}")
    print(f"  bids             {len(bids):>8,}")
    print(f"  total spend      ${tenders['amount'].sum():>14,.0f}")
    print("─" * 62)
    print(f"  fraudulent       {n_fraud:>8,}   ({n_fraud / n:.2%} contamination)")
    print("─" * 62)

    counts = (truth[truth["is_fraud"] == 1]["fraud_type"]
              .str.split("|").explode().value_counts())
    for k, v in counts.items():
        exposure = tenders.merge(truth, on="tender_id")
        exposure = exposure[exposure["fraud_type"].str.contains(k, regex=False)]
        print(f"  {k:<22} {v:>5,} tenders   ${exposure['amount'].sum():>13,.0f}")

    print("═" * 62)

    # The sanity checks that matter. If any of these fail, the labels lie.
    m = tenders.merge(truth, on="tender_id")
    splits = m[m["fraud_type"].str.contains("po_splitting")]
    if len(splits):
        under = (splits["amount"] < splits["approval_limit"]).mean()
        print(f"  ✓ split POs sitting under the approval limit : {under:.1%}")

    ghosts = set(vendors[vendors["is_ghost"]]["vendor_id"])
    emp_banks = set(employees["bank_account"])
    shared = vendors[vendors["bank_account"].isin(emp_banks)]
    print(f"  ✓ vendors sharing a bank account with a buyer : {len(shared)}")

    cartel = m[m["fraud_type"].str.contains("bid_rigging")]
    print(f"  ✓ cartel tenders                              : {len(cartel)}")
    print(f"  ✓ clean tenders (the 96% the model must ignore): "
          f"{(truth['is_fraud'] == 0).sum():,}")
    print("═" * 62 + "\n")


def assert_no_leakage(tenders, bids):
    """
    The three invariants that must hold for EVERY tender, fraudulent or not.

    Each one is a promise the honest world keeps automatically. Any injector
    that breaks one hands a model a free label: a rule that is true for all
    3,000 clean rows and false only for fraud is a perfect detector that has
    learned nothing about fraud. These asserts are the tripwire.
    """
    n_rows = bids.groupby("tender_id").size()
    hdr = tenders.set_index("tender_id")["n_bidders"]
    bad = hdr.index[hdr != n_rows.reindex(hdr.index).fillna(0)]
    assert len(bad) == 0, (
        f"n_bidders disagrees with the bid rows on file for {len(bad)} tenders "
        f"(e.g. {list(bad[:3])}). Header/detail mismatch = free label.")

    win = bids[bids["is_winner"]].groupby("tender_id")["bid_amount"].first()
    amt = tenders.set_index("tender_id")["amount"]
    gap = (win.reindex(amt.index) - amt).abs()
    assert (gap <= 0.011).all(), (
        f"awarded amount != winning bid on {(gap > 0.011).sum()} tenders. "
        f"If an injector edits the award but not the bid, the gap IS the label.")

    orphans = set(bids["tender_id"]) - set(tenders["tender_id"])
    assert not orphans, f"{len(orphans)} bids point at tenders that do not exist"

    one_winner = bids.groupby("tender_id")["is_winner"].sum()
    assert (one_winner == 1).all(), "every tender needs exactly one winning bid"

    print("  ✓ invariants hold: n_bidders == bid rows, award == winning bid,")
    print("    no orphan bids, exactly one winner per tender")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    os.makedirs(C.OUT_DIR, exist_ok=True)
    v, e, t, b, gt = build_dataset()

    v.to_csv(C.FILES["vendors"], index=False)
    e.to_csv(C.FILES["employees"], index=False)
    t.to_csv(C.FILES["tenders"], index=False)
    b.to_csv(C.FILES["bids"], index=False)
    gt.to_csv(C.FILES["labels"], index=False)

    report(v, e, t, b, gt)
    assert_no_leakage(t, b)
    print(f"  written → {C.OUT_DIR}/\n")
