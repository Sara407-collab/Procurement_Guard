"""
Clean procurement activity.

This is the "normal world" — no fraud in it at all. Everything the model has to
find gets layered on top in fraud_injection.py. Keeping the two apart is what
makes the ground truth trustworthy: if a row is labelled fraudulent, it is
because we put fraud in it, not because the generator got sloppy.

A tender produces several bids and exactly one winning PO.
"""

import numpy as np
import pandas as pd

from . import config as C


def _lognormal_amount(rng, category: str) -> float:
    """Procurement spend is heavy-tailed: many small POs, a few enormous ones."""
    base = {
        "Construction":        (11.8, 1.0),
        "Consulting Services": (10.9, 0.9),
        "Raw Materials":       (10.6, 1.0),
        "Hardware":            (10.2, 0.9),
        "Software Licenses":   (10.0, 0.9),
        "Transport":           (9.8, 0.8),
        "Spare Parts":         (9.5, 0.9),
        "Security Services":   (9.4, 0.7),
        "Cleaning Services":   (9.2, 0.7),
        "Office Supplies":     (8.6, 0.8),
    }
    mu, sigma = base.get(category, (10.0, 0.9))
    return float(np.round(rng.lognormal(mu, sigma), 2))


def build_tenders_and_bids(vendors: pd.DataFrame,
                           employees: pd.DataFrame,
                           rng: np.random.Generator):
    """Returns (tenders, bids). One winning PO per tender."""
    end = pd.Timestamp("2026-06-30")
    start = end - pd.DateOffset(years=C.YEARS_OF_HISTORY)
    span_days = (end - start).days

    vendor_ids = vendors["vendor_id"].to_numpy()
    vendor_cat = dict(zip(vendors["vendor_id"], vendors["primary_category"]))

    tenders, bids = [], []

    for i in range(C.N_TENDERS):
        emp = employees.iloc[int(rng.integers(0, len(employees)))]
        category = rng.choice(C.CATEGORIES)
        method = rng.choice(C.AWARD_METHODS, p=C.AWARD_METHOD_WEIGHTS)

        # Bidders: prefer vendors whose primary category matches the tender.
        eligible = [v for v in vendor_ids if vendor_cat[v] == category]
        if len(eligible) < 4:
            eligible = list(vendor_ids)

        n_bidders = {
            "open_tender":    int(rng.integers(3, 8)),
            "limited_tender": int(rng.integers(2, 4)),
            "single_source":  1,
        }[method]
        n_bidders = min(n_bidders, len(eligible))

        bidders = rng.choice(eligible, size=n_bidders, replace=False)
        award_date = start + pd.Timedelta(days=int(rng.integers(0, span_days)))
        true_value = _lognormal_amount(rng, category)

        # Honest competitive bidding: spread around the true value, lowest wins.
        bid_amounts = np.round(true_value * rng.normal(1.0, 0.10, n_bidders), 2)
        bid_amounts = np.maximum(bid_amounts, 500.0)
        win_idx = int(np.argmin(bid_amounts))

        # Real POs land on round numbers all the time — negotiated prices, unit
        # counts, budget ceilings. If the honest world never produced a round
        # amount, "is it round?" would be a perfect fraud oracle, which is a
        # statement about the generator and not about fraud. Floor (never round
        # up) so the winner stays the winner.
        if rng.random() < C.CLEAN_ROUND_RATE and bid_amounts[win_idx] >= 2000:
            bid_amounts[win_idx] = float(np.floor(bid_amounts[win_idx] / 1000) * 1000)

        tender_id = f"T{i:05d}"
        for j, vid in enumerate(bidders):
            bids.append({
                "bid_id":     f"{tender_id}-B{j}",
                "tender_id":  tender_id,
                "vendor_id":  str(vid),
                "bid_amount": float(bid_amounts[j]),
                "is_winner":  j == win_idx,
            })

        winner = str(bidders[win_idx])
        amount = float(bid_amounts[win_idx])
        invoice_date = award_date + pd.Timedelta(days=int(rng.integers(5, 45)))

        tenders.append({
            "tender_id":       tender_id,
            "po_id":           f"PO{i:05d}",
            "award_date":      award_date,
            "vendor_id":       winner,
            "employee_id":     emp["employee_id"],
            "department":      emp["department"],
            "category":        category,
            "award_method":    method,
            "amount":          amount,
            "n_bidders":       n_bidders,
            "approval_limit":  int(emp["approval_limit"]),
            "invoice_id":      f"INV{i:05d}",
            "invoice_date":    invoice_date,
            "payment_date":    invoice_date + pd.Timedelta(days=int(rng.integers(10, 60))),
        })

    return pd.DataFrame(tenders), pd.DataFrame(bids)
