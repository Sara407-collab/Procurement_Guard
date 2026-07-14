"""
Fraud injection — six patterns, each with ground truth.

Design rule: every function here does two things and only two things.
  1. Mutate the data the way a real fraudster would.
  2. Record exactly which rows it touched, and why.

That second part is the whole project. Without honest labels you cannot report
precision, you cannot run an ablation, and you have nothing to show anyone.

Labels are emitted at TENDER level:
    tender_id | is_fraud | fraud_type | ring_id
"""

import numpy as np
import pandas as pd

from . import config as C


def _label(tender_id, fraud_type, ring_id=None):
    return {"tender_id": tender_id, "is_fraud": 1,
            "fraud_type": fraud_type, "ring_id": ring_id}


# ═══════════════════════════════════════════════════════════ 1. PO SPLITTING
def inject_po_splitting(tenders, bids, employees, rng):
    """
    One real need of $180k, an approval limit of $100k, and a buyer who does
    not want a director's signature. Result: three POs of ~$62k, same vendor,
    same buyer, same week. The classic.
    """
    labels = []
    n = int(len(tenders) * C.FRAUD_RATES["po_splitting"])
    # Only worth splitting if the amount actually exceeds the limit.
    candidates = tenders[tenders["amount"] > tenders["approval_limit"]].index
    if len(candidates) == 0:
        return tenders, bids, labels
    picks = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    other_vendors = tenders["vendor_id"].unique()

    new_tenders, new_bids = [], []
    drop = []

    for k, idx in enumerate(picks):
        row = tenders.loc[idx]
        limit = row["approval_limit"]
        total = row["amount"]
        n_parts = int(np.ceil(total / limit)) + int(rng.integers(0, 2))
        n_parts = max(2, min(n_parts, 6))
        ring = f"SPLIT{k:03d}"

        for p in range(n_parts):
            hug = rng.uniform(*C.SPLIT_HUG_RANGE)
            part_amt = float(np.round(limit * hug, 2))
            tid = f"{row['tender_id']}-S{p}"

            t = row.copy()
            t["tender_id"] = tid
            t["po_id"] = f"{row['po_id']}-S{p}"
            t["invoice_id"] = f"{row['invoice_id']}-S{p}"
            t["amount"] = part_amt
            t["award_date"] = row["award_date"] + pd.Timedelta(days=int(rng.integers(0, 12)))
            t["award_method"] = "limited_tender"
            n_bid = int(rng.integers(1, 3))
            t["n_bidders"] = n_bid
            new_tenders.append(t)

            # Write exactly n_bidders bid rows. If the header says two bidders
            # and only one bid exists on file, that mismatch is a perfect
            # fraud oracle — an artefact of the generator, not of fraud.
            new_bids.append({
                "bid_id": f"{tid}-B0", "tender_id": tid,
                "vendor_id": row["vendor_id"], "bid_amount": part_amt,
                "is_winner": True,
            })
            for j in range(1, n_bid):
                loser = str(rng.choice(other_vendors))
                new_bids.append({
                    "bid_id": f"{tid}-B{j}", "tender_id": tid,
                    "vendor_id": loser,
                    "bid_amount": float(np.round(part_amt * rng.uniform(1.02, 1.15), 2)),
                    "is_winner": False,
                })
            labels.append(_label(tid, "po_splitting", ring))

        drop.append(idx)

    # The parent tender is replaced by its parts — so its bids must go too.
    # (The previous version indexed an empty selection and silently dropped
    #  nothing, leaving orphan bids pointing at tenders that no longer exist.)
    dropped_ids = set(tenders.loc[drop, "tender_id"])
    tenders = pd.concat(
        [tenders.drop(index=drop), pd.DataFrame(new_tenders)], ignore_index=True)
    bids = pd.concat(
        [bids[~bids["tender_id"].isin(dropped_ids)], pd.DataFrame(new_bids)],
        ignore_index=True)
    return tenders, bids, labels


# ═════════════════════════════════════════════════════ 2. DUPLICATE INVOICE
def inject_duplicate_invoice(tenders, bids, rng):
    """
    Same vendor bills the same amount twice, weeks apart, under a slightly
    different invoice number. Nobody notices because nobody is looking.
    """
    labels = []
    n = int(len(tenders) * C.FRAUD_RATES["duplicate_invoice"])
    picks = rng.choice(tenders.index, size=n, replace=False)

    new_tenders, new_bids = [], []
    for k, idx in enumerate(picks):
        row = tenders.loc[idx].copy()
        tid = f"{row['tender_id']}-D"
        jitter = 1.0 if rng.random() < 0.5 else rng.uniform(0.995, 1.005)

        row["tender_id"] = tid
        row["po_id"] = f"{row['po_id']}-D"
        row["invoice_id"] = f"{row['invoice_id']}A"          # near-identical
        row["amount"] = float(np.round(row["amount"] * jitter, 2))
        row["invoice_date"] = row["invoice_date"] + pd.Timedelta(days=int(rng.integers(5, 35)))
        row["payment_date"] = row["invoice_date"] + pd.Timedelta(days=int(rng.integers(10, 40)))
        row["n_bidders"] = 1        # a re-bill, not a re-tender: one bid row below
        new_tenders.append(row)

        new_bids.append({
            "bid_id": f"{tid}-B0", "tender_id": tid,
            "vendor_id": row["vendor_id"], "bid_amount": float(row["amount"]),
            "is_winner": True,
        })
        labels.append(_label(tid, "duplicate_invoice", f"DUP{k:03d}"))

    tenders = pd.concat([tenders, pd.DataFrame(new_tenders)], ignore_index=True)
    bids = pd.concat([bids, pd.DataFrame(new_bids)], ignore_index=True)
    return tenders, bids, labels


# ═══════════════════════════════════════════════════════════ 3. GHOST VENDOR
def inject_ghost_vendor(vendors, employees, tenders, bids, rng):
    """
    A vendor that does not really exist. Registered days before its first
    award, wins single-source, bills large, disappears. And — the giveaway —
    its bank account belongs to an employee.
    """
    labels = []
    n_ghosts = max(2, int(C.N_VENDORS * 0.03))
    new_vendors, new_tenders, new_bids = [], [], []

    def _bank(): return f"PK{rng.integers(10, 99)}TMCB{rng.integers(10**10, 10**11 - 1)}"
    def _phone(): return f"+92-3{rng.integers(0, 10)}{rng.integers(0, 10)}-{rng.integers(1000000, 9999999)}"
    def _addr(): return f"{rng.integers(1, 400)} Unknown Rd, Lahore"

    for g in range(n_ghosts):
        emp = employees.iloc[int(rng.integers(0, len(employees)))]
        vid = f"V9{g:03d}"
        first_award = pd.Timestamp("2023-09-01") + pd.Timedelta(days=int(rng.integers(0, 900)))

        # Not every fraudster is careless enough to bank in his own name.
        #
        # DIRECT (~55%): the ghost shares a bank account or phone with the buyer
        #   who created it. One SQL join finds this. It is the easy case, and it
        #   should stay in the data, because the easy case is real.
        #
        # LAYERED (~45%): the money goes through a CUTOUT — a second shell that
        #   never wins a tender and never appears in tenders.csv at all. The
        #   ghost shares a bank with the cutout; the cutout shares an address
        #   with the buyer. The ghost has NO direct link to any employee.
        #
        #       ghost ──same bank── cutout ──same address── buyer
        #
        #   A join sees nothing: no employee shares an identifier with this
        #   vendor. Two hops is not a harder join, it is a DIFFERENT KIND of
        #   query. This is the reason the graph exists, and until now the
        #   generator never produced a case that required one.
        layered = rng.random() < C.GHOST_LAYERED_RATE

        if layered:
            cutout_bank = _bank()
            new_vendors.append({
                "vendor_id":         f"V8{g:03d}",
                "vendor_name":       f"{['Meridian','Halcyon','Trident','Onyx','Cobalt'][g % 5]} "
                                     f"Holdings {['Ltd','LLC','& Co'][g % 3]}",
                "tax_id":            f"{rng.integers(1000000, 9999999)}-{rng.integers(0, 10)}",
                "bank_account":      cutout_bank,        # ← hop 1: shared with the ghost
                "address":           emp["address"],     # ← hop 2: shared with the buyer
                "phone":             _phone(),
                "registration_date": first_award - pd.Timedelta(days=int(rng.integers(30, 400))),
                "primary_category":  rng.choice(C.CATEGORIES),
                "is_ghost":          True,   # bookkeeping only: keeps it out of the
                                             # collusion/cartel pools. It wins nothing,
                                             # so it never reaches ground truth.
            })
            ghost_bank, ghost_phone, ghost_addr = cutout_bank, _phone(), _addr()
        else:
            ghost_bank = emp["bank_account"]
            ghost_phone = emp["phone"] if rng.random() < 0.7 else _phone()
            ghost_addr = emp["address"] if rng.random() < 0.5 else _addr()

        new_vendors.append({
            "vendor_id":         vid,
            "vendor_name":       f"{['Zenith','Vanguard','Orbit','Nexa','Solara'][g % 5]} "
                                 f"{['Trading','Supplies','Services'][g % 3]} Ltd",
            "tax_id":            f"{rng.integers(1000000, 9999999)}-{rng.integers(0, 10)}",
            "bank_account":      ghost_bank,
            "address":           ghost_addr,
            "phone":             ghost_phone,
            "registration_date": first_award - pd.Timedelta(days=int(rng.integers(3, 30))),
            "primary_category":  rng.choice(C.CATEGORIES),
            "is_ghost":          True,
        })

        # The vendor's FIRST award must actually land on first_award, because
        # registration_date is anchored to it. Scattering every tender forward
        # by up to 240 days meant the earliest award drifted ~120 days away —
        # so "registered days before its first order", the whole tell, was
        # false for most ghosts and rule_new_vendor_large_first fired 3 times.
        for p in range(int(rng.integers(2, 6))):
            tid = f"TG{g:03d}{p:02d}"
            amt = float(np.round(rng.uniform(40_000, 220_000), 2))
            award = (first_award if p == 0 else
                     first_award + pd.Timedelta(days=int(rng.integers(10, 240))))
            inv = award + pd.Timedelta(days=int(rng.integers(3, 20)))

            new_tenders.append({
                "tender_id": tid, "po_id": f"POG{g:03d}{p:02d}",
                "award_date": award, "vendor_id": vid,
                "employee_id": emp["employee_id"], "department": emp["department"],
                "category": rng.choice(C.CATEGORIES), "award_method": "single_source",
                "amount": amt, "n_bidders": 1,
                "approval_limit": int(emp["approval_limit"]),
                "invoice_id": f"INVG{g:03d}{p:02d}",
                "invoice_date": inv,
                "payment_date": inv + pd.Timedelta(days=int(rng.integers(5, 25))),
            })
            new_bids.append({
                "bid_id": f"{tid}-B0", "tender_id": tid, "vendor_id": vid,
                "bid_amount": amt, "is_winner": True,
            })
            labels.append(_label(tid, "ghost_vendor", f"GHOST{g:03d}"))

    vendors = pd.concat([vendors, pd.DataFrame(new_vendors)], ignore_index=True)
    tenders = pd.concat([tenders, pd.DataFrame(new_tenders)], ignore_index=True)
    bids = pd.concat([bids, pd.DataFrame(new_bids)], ignore_index=True)
    return vendors, employees, tenders, bids, labels


# ═════════════════════════════════════════════════════ 4. EMPLOYEE COLLUSION
def inject_employee_collusion(vendors, employees, tenders, bids, rng):
    """
    A *real* vendor, quietly connected to a buyer. Shared address, shared
    phone, sometimes a shared bank account. That buyer then steers work to
    them: high award share, thin competition, amounts that creep upward.

    No tabular model finds this. The signal is not inside any single row —
    it is the edge between two records. This is why we build a graph.
    """
    labels = []
    n_pairs = max(3, int(C.N_EMPLOYEES * 0.10))
    real = vendors[~vendors["is_ghost"]].index

    for k in range(n_pairs):
        v_idx = int(rng.choice(real))
        emp = employees.iloc[int(rng.integers(0, len(employees)))]
        vid = vendors.at[v_idx, "vendor_id"]

        # Plant the link. Sometimes one shared field, sometimes several.
        link = rng.choice(["bank", "address", "phone", "all"], p=[.25, .3, .25, .2])
        if link in ("bank", "all"):
            vendors.at[v_idx, "bank_account"] = emp["bank_account"]
        if link in ("address", "all"):
            vendors.at[v_idx, "address"] = emp["address"]
        if link in ("phone", "all"):
            vendors.at[v_idx, "phone"] = emp["phone"]

        # Steer business toward them.
        mask = tenders["vendor_id"] == vid
        steer = tenders[mask].index
        if len(steer) == 0:
            continue
        take = rng.choice(steer, size=max(1, int(len(steer) * 0.6)), replace=False)

        pool = vendors["vendor_id"].to_numpy()

        for tid_idx in take:
            tid = tenders.at[tid_idx, "tender_id"]
            n_bid = int(rng.integers(1, 3))

            # The vendor bids high and the buyer waves it through. The
            # inflation lives in the BID, not in a post-hoc edit of the award
            # — otherwise awarded_amount != winning_bid on every colluded
            # tender and nowhere else, which is a free label, not a signal.
            inflated = float(np.round(
                tenders.at[tid_idx, "amount"] * rng.uniform(1.15, 1.6), 2))

            tenders.at[tid_idx, "employee_id"] = emp["employee_id"]
            tenders.at[tid_idx, "department"] = emp["department"]
            tenders.at[tid_idx, "n_bidders"] = n_bid
            tenders.at[tid_idx, "award_method"] = rng.choice(
                ["limited_tender", "single_source"], p=[0.6, 0.4])
            tenders.at[tid_idx, "amount"] = inflated

            rows = [{"bid_id": f"{tid}-K0", "tender_id": tid,
                     "vendor_id": vid, "bid_amount": inflated, "is_winner": True}]
            for j in range(1, n_bid):
                rows.append({"bid_id": f"{tid}-K{j}", "tender_id": tid,
                             "vendor_id": str(rng.choice(pool)),
                             "bid_amount": float(np.round(
                                 inflated * rng.uniform(1.02, 1.20), 2)),
                             "is_winner": False})
            bids = bids[bids["tender_id"] != tid]
            bids = pd.concat([bids, pd.DataFrame(rows)], ignore_index=True)

            labels.append(_label(tid, "employee_collusion", f"COLL{k:03d}"))

    return vendors, employees, tenders, bids, labels


# ═════════════════════════════════════════════════════ 5. ROUND NUMBER ABUSE
def inject_round_number_abuse(tenders, bids, rng):
    """
    Invented numbers are round numbers. Nobody fabricating an invoice writes
    $47,283.61 — they write $50,000. Benford's law made concrete.
    """
    labels = []
    n = int(len(tenders) * C.FRAUD_RATES["round_number_abuse"])
    picks = rng.choice(tenders.index, size=n, replace=False)

    for idx in picks:
        limit = tenders.at[idx, "approval_limit"]
        # Round *and* suspiciously close to the ceiling. Granularity scales
        # with the limit: a flat 5,000 step made every $25k-tier fraud land on
        # exactly $20,000, which is a fingerprint, not a pattern.
        gran = max(1000, limit // 25)
        amt = float(np.floor(limit * rng.uniform(0.88, 0.98) / gran) * gran)
        if amt <= 0:
            continue
        tenders.at[idx, "amount"] = amt
        tid = tenders.at[idx, "tender_id"]
        bids.loc[(bids["tender_id"] == tid) & bids["is_winner"], "bid_amount"] = amt
        labels.append(_label(tid, "round_number_abuse", None))

    return tenders, bids, labels


# ═══════════════════════════════════════════════════════ 6. BID-RIGGING CARTEL
def inject_bid_rigging(vendors, tenders, bids, rng):
    """
    The hardest one, and the one worth showing.

    Three to five vendors agree not to compete. They bid on the same tenders,
    take turns winning, and the losers submit deliberately high "cover bids" so
    the winner looks like a bargain. Every individual tender looks perfectly
    normal. The pattern only exists across the whole history — which is exactly
    what a network view reveals and a row-by-row model cannot.
    """
    labels = []
    real = vendors[~vendors["is_ghost"]]["vendor_id"].to_numpy()
    n_rings = C.CARTEL_N_RINGS

    # Two rings in the same category would be two rings in the same conspiracy.
    # Give each its own market.
    cats = list(rng.choice(C.CATEGORIES, size=min(n_rings, len(C.CATEGORIES)),
                           replace=False))

    for r in range(n_rings):
        size = int(rng.integers(*C.CARTEL_RING_SIZE))
        category = cats[r]
        ring_id = f"CARTEL{r:03d}"

        # A cartel is not a handful of rigged tenders scattered across a market.
        # It is an ARRANGEMENT, and it covers a segment.
        #
        # The previous version had a ring rig ~8 tenders out of ~190 in its
        # category — 5% — and bid honestly on the other 95%. That is not a
        # cartel, it is noise, and it is why co-bidding affinity for real ring
        # pairs (0.277) sat right on top of ordinary pairs (0.132): the signal
        # was diluted by all the honest bidding the members did elsewhere.
        #
        # Real cartels dominate. The EU truck cartel held ~90% of the market for
        # fourteen years. Japan's dango construction rings carved up public
        # works between them. So: draw the ring from vendors who actually
        # SPECIALISE in this category (they are each other's natural rivals —
        # that is the whole point of agreeing to stop), and have them take most
        # of the bigger tenders in it.
        specialists = vendors[(~vendors["is_ghost"]) &
                              (vendors["primary_category"] == category)]["vendor_id"].to_numpy()
        if len(specialists) < size:
            continue
        ring = rng.choice(specialists, size=size, replace=False)

        cat_pool = tenders[(tenders["category"] == category) &
                           (tenders["award_method"] == "open_tender")]
        if len(cat_pool) < 12:
            continue
        # the segment they carve up: the larger contracts, where the money is
        cutoff = cat_pool["amount"].quantile(1.0 - C.CARTEL_SEGMENT_TOP_SHARE)
        pool = cat_pool[cat_pool["amount"] >= cutoff].index
        if len(pool) < 8:
            continue
        n_take = int(len(pool) * rng.uniform(*C.CARTEL_SEGMENT_CAPTURE))
        take = rng.choice(pool, size=max(6, min(n_take, len(pool))), replace=False)

        for turn, idx in enumerate(sorted(take)):
            tid = tenders.at[idx, "tender_id"]
            winner = ring[turn % size]                       # ← rotation
            base = float(tenders.at[idx, "amount"])

            # A cartel cannot stop outsiders from bidding — it can only make
            # sure they lose. Keep some of the honest bidders who were already
            # on this tender. Their bids are all ABOVE `base` by construction
            # (base was the winning bid, i.e. the minimum), so the ring still
            # wins, and nothing is fabricated.
            #
            # Without this, EVERY cartel tender has exactly the ring as its
            # bidder set, and "these same four names, again" becomes a perfect
            # oracle — a fingerprint of the injector, not of a conspiracy. Real
            # cartel detection reads the CO-BIDDING STATISTICS, not an exact set
            # match, and now the model is forced to do the same.
            original = bids[bids["tender_id"] == tid]
            outsiders = original[(~original["vendor_id"].isin([str(x) for x in ring]))
                                 & (~original["is_winner"])]
            n_keep = min(int(rng.integers(0, 3)), len(outsiders))
            keep = outsiders.head(n_keep)

            bids = bids[bids["tender_id"] != tid]
            rows = [{"bid_id": f"{tid}-C0", "tender_id": tid,
                     "vendor_id": str(winner), "bid_amount": base,
                     "is_winner": True}]
            for j, v in enumerate([v for v in ring if v != winner]):
                cover = float(np.round(base * rng.uniform(*C.CARTEL_COVER_BID_MARKUP), 2))
                rows.append({"bid_id": f"{tid}-C{j+1}", "tender_id": tid,
                             "vendor_id": str(v), "bid_amount": cover,
                             "is_winner": False})
            for j, o in enumerate(keep.itertuples()):
                rows.append({"bid_id": f"{tid}-O{j}", "tender_id": tid,
                             "vendor_id": str(o.vendor_id),
                             "bid_amount": float(o.bid_amount), "is_winner": False})
            bids = pd.concat([bids, pd.DataFrame(rows)], ignore_index=True)

            tenders.at[idx, "vendor_id"] = str(winner)
            tenders.at[idx, "n_bidders"] = size + n_keep
            labels.append(_label(tid, "bid_rigging_cartel", ring_id))

    return tenders, bids, labels
