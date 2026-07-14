"""
Graph layer — Day 5.

Two graphs, because there are two kinds of fact a table cannot hold.

  IDENTITY GRAPH — vendors and employees, joined wherever they share a bank
  account, an address, or a phone number. Vendors are also joined to each other
  the same way.
      A vendor that banks where a buyer banks is not a coincidence. Nothing in
      tenders.csv can say this: the vendor's bank lives in vendors.csv, the
      buyer's in employees.csv, and the tender table only ever holds their IDs.
      The fact exists only BETWEEN the two records.

      And the careful fraudster does not bank in his own name. He puts a cutout
      in between:

          ghost ──same bank── cutout ──same address── buyer

      The ghost shares NOTHING with any employee. A join looking for
      `vendor.bank = employee.bank` returns an empty set. Two hops is not a
      harder join — it is a different question, and only a graph can ask it.
      → ghost_vendor, employee_collusion

  CO-BIDDING GRAPH — vendors joined to the vendors they bid against.
      Every cartel tender, on its own, is spotless: several bidders, lowest
      price wins, nothing to see. The conspiracy is not inside any tender. It
      is in who keeps turning up together, and in who takes their turn.
      Day 4 threw 34 features and two models at the cartel and caught 0 of 43.
      Not because the models were weak — because the signal was not in the rows.
      → bid_rigging_cartel

No labels enter this file. It is built from vendors.csv, employees.csv and
bids.csv only.
"""

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C

IDENTIFIERS = ("bank_account", "address", "phone")


# ═══════════════════════════════════════════════════════ 1. the identity graph
def build_identity_graph(vendors: pd.DataFrame, employees: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from((f"V:{v}" for v in vendors["vendor_id"]), kind="vendor")
    G.add_nodes_from((f"E:{e}" for e in employees["employee_id"]), kind="employee")

    for field in IDENTIFIERS:
        v_by = vendors.groupby(field)["vendor_id"].apply(list)
        e_by = employees.groupby(field)["employee_id"].apply(list)

        for key in set(v_by.index) & set(e_by.index):          # vendor ↔ employee
            for v in v_by[key]:
                for e in e_by[key]:
                    G.add_edge(f"V:{v}", f"E:{e}", shared=field)

        for key, vs in v_by.items():                            # vendor ↔ vendor
            if len(vs) > 1:                                     # the cutout lives here
                for a, b in combinations(vs, 2):
                    G.add_edge(f"V:{a}", f"V:{b}", shared=field)
    return G


def identity_features(tenders, vendors, employees, G) -> pd.DataFrame:
    v_idx = vendors.set_index("vendor_id")
    e_idx = employees.set_index("employee_id")

    # Shortest path length, capped. 1 = they share an identifier directly (a join
    # finds this). 2 = there is a cutout between them (a join finds NOTHING).
    # 0 = unreachable, which is what an honest vendor looks like.
    hop = dict(nx.all_pairs_shortest_path_length(G, cutoff=C.IDENTITY_MAX_HOPS))
    comp = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
    comp_size = Counter(comp.values())

    rows = []
    for t in tenders.itertuples():
        vn, en = f"V:{t.vendor_id}", f"E:{t.employee_id}"
        v, e = v_idx.loc[t.vendor_id], e_idx.loc[t.employee_id]

        direct = sum(v[f] == e[f] for f in IDENTIFIERS)         # the join's answer
        d = hop.get(vn, {}).get(en, 0)                          # the graph's answer

        rows.append({
            "tender_id":                t.tender_id,
            "g_direct_shared_ids":       direct,
            "g_hops_to_buyer":           d,
            "g_reachable_within_2_hops": int(0 < d <= 2),
            "g_vendor_emp_degree":       sum(1 for n in G.neighbors(vn) if n.startswith("E:"))
                                         if vn in G else 0,
            "g_identity_component_size": comp_size.get(comp.get(vn, -1), 1),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════ 2. the co-bidding graph
def build_cobid_graph(bids: pd.DataFrame) -> nx.Graph:
    """
    Vendors joined to the vendors they bid against.

    Raw co-bid counts are not enough: a busy vendor co-bids with everyone. What
    matters is AFFINITY — of the tenders these two could have met on, how often
    did they actually meet? Two vendors who show up together on a third of each
    other's bids are not competing. They are travelling together.
    """
    pairs = Counter()
    for _, g in bids.groupby("tender_id"):
        for a, b in combinations(sorted(g["vendor_id"].unique()), 2):
            pairs[(a, b)] += 1

    n_bids = bids.groupby("vendor_id")["tender_id"].nunique().to_dict()

    G = nx.Graph()
    G.add_nodes_from(bids["vendor_id"].unique())
    for (a, b), w in pairs.items():
        na, nb = n_bids.get(a, 0), n_bids.get(b, 0)
        # A vendor that bid twice, both times alongside the same firm, scores an
        # affinity of 1.0 and means nothing. Two data points are not a pattern.
        if min(na, nb) < C.COBID_MIN_BIDS:
            G.add_edge(a, b, weight=w, affinity=0.0)
            continue
        G.add_edge(a, b, weight=w, affinity=w / min(na, nb))
    return G


def find_rings(G: nx.Graph, bids: pd.DataFrame) -> pd.DataFrame:
    """
    The output that actually matters.

    A cartel is not a property of a tender. It is a property of a GROUP OF
    VENDORS. You do not tell an auditor "tender T00412 is rigged" — you tell
    them "these four firms are an arrangement", and every tender they touched
    comes with it.

    So: keep only abnormally tight co-bidding edges, find the MAXIMAL CLIQUES
    (not connected components — components chain A-B-C-D together even when A
    and D never met; a ring is a clique, everyone bids with everyone), and score
    each one by how evenly its members share the wins.

    Honest competition has a cheapest supplier, and it wins more than its share.
    A cartel gives everyone a turn, so the win distribution comes out flat.
    """
    aff = np.array([d["affinity"] for _, _, d in G.edges(data=True)])
    cut = np.quantile(aff[aff > 0], C.COBID_AFFINITY_QUANTILE) if (aff > 0).any() else 1.0

    tight = nx.Graph()
    tight.add_edges_from((a, b) for a, b, d in G.edges(data=True) if d["affinity"] >= cut)

    bidders = bids.groupby("tender_id")["vendor_id"].apply(set).to_dict()
    winners = bids[bids["is_winner"]].set_index("tender_id")["vendor_id"].to_dict()

    rows = []
    for clique in nx.find_cliques(tight):
        if len(clique) < C.COBID_MIN_CLUSTER:
            continue
        members = set(clique)
        ts = [t for t, B in bidders.items() if len(B & members) >= 2]
        wins = Counter(winners[t] for t in ts if winners.get(t) in members)
        if len(wins) < 2:
            continue

        p = np.array(list(wins.values()), float)
        p /= p.sum()
        evenness = float(-(p * np.log(p)).sum() / np.log(len(members)))
        mean_aff = float(np.mean([G[a][b]["affinity"]
                                  for a, b in combinations(sorted(members), 2)
                                  if G.has_edge(a, b)]))
        rows.append({
            "vendors":       ",".join(sorted(members)),
            "n_vendors":     len(members),
            "n_tenders":     len(ts),
            "mean_affinity": round(mean_aff, 3),
            "win_evenness":  round(evenness, 3),
            "ring_score":    round(mean_aff * evenness * len(members), 3),
        })
    if not rows:
        return pd.DataFrame(columns=["vendors", "n_vendors", "n_tenders",
                                     "mean_affinity", "win_evenness", "ring_score"])
    return pd.DataFrame(rows).sort_values("ring_score", ascending=False).reset_index(drop=True)


def cobid_features(tenders, bids, G) -> pd.DataFrame:
    """Per-tender features. The ones that matter carry the ring detector's
    verdict down onto every tender the ring touched."""
    bidders = bids.groupby("tender_id")["vendor_id"].apply(set).to_dict()
    bid_sets = bids.groupby("vendor_id")["tender_id"].apply(set).to_dict()
    winners = bids[bids["is_winner"]].set_index("tender_id")["vendor_id"].to_dict()

    rings = find_rings(G, bids)
    ring_members = [(set(r.vendors.split(",")), r.ring_score, r.win_evenness)
                    for r in rings.itertuples()]

    rows = []
    for t in tenders.itertuples():
        B = bidders.get(t.tender_id, set())
        w = winners.get(t.tender_id)

        # Which suspected ring, if any, does this tender belong to? The winner
        # must be a member, and at least one of its co-conspirators must have
        # been in the room — otherwise it is not a rigged tender, it is just a
        # tender a cartel member happened to bid on honestly.
        best, best_even = 0.0, 0.0
        for mem, score, even in ring_members:
            if w in mem and len(B & mem) >= 2:
                if score > best:
                    best, best_even = score, even

        if len(B) < 2:
            rows.append({"tender_id": t.tender_id, "g_cobid_min_affinity": 0.0,
                         "g_cobid_mean_affinity": 0.0, "g_jaccard_mean": 0.0,
                         "g_ring_score": 0.0, "g_ring_win_evenness": 0.0,
                         "g_bidders_in_ring": 0.0})
            continue

        affs, jac = [], []
        for a, b in combinations(sorted(B), 2):
            affs.append(G[a][b]["affinity"] if G.has_edge(a, b) else 0.0)
            sa, sb = bid_sets.get(a, set()), bid_sets.get(b, set())
            jac.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0)

        in_ring = 0.0
        for mem, score, _ in ring_members:
            if score == best and best > 0:
                in_ring = len(B & mem) / len(B)
                break

        rows.append({
            "tender_id":             t.tender_id,
            # min, not mean: a ring needs EVERY pair tight. One tight pair
            # inside an otherwise normal tender is a coincidence.
            "g_cobid_min_affinity":  float(min(affs)),
            "g_cobid_mean_affinity": float(np.mean(affs)),
            "g_jaccard_mean":        float(np.mean(jac)),
            "g_ring_score":          float(best),
            "g_ring_win_evenness":   float(best_even),
            "g_bidders_in_ring":     float(in_ring),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════ public
def build_graph_features(tenders, bids, vendors, employees):
    """One row per tender. Everything the tables could not say."""
    Gi = build_identity_graph(vendors, employees)
    Gc = build_cobid_graph(bids)
    fi = identity_features(tenders, vendors, employees, Gi)
    fc = cobid_features(tenders, bids, Gc)
    return fi.merge(fc, on="tender_id"), Gi, Gc
