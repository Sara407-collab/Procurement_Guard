"""
ProcurementGuard — configuration.

Every magic number lives here. When your lead asks "what if the threshold
were different?", you change one line, not twenty.
"""

RANDOM_SEED = 42

# ---------------------------------------------------------------- dataset size
N_VENDORS = 300
N_EMPLOYEES = 60
N_TENDERS = 3000          # each tender -> 1 winning PO (+ losing bids)
YEARS_OF_HISTORY = 3

# ------------------------------------------------------------------ org design
DEPARTMENTS = [
    "IT", "Facilities", "Logistics", "Production",
    "Marketing", "R&D", "Maintenance", "Admin",
]

CATEGORIES = [
    "Hardware", "Software Licenses", "Consulting Services", "Raw Materials",
    "Office Supplies", "Transport", "Construction", "Spare Parts",
    "Cleaning Services", "Security Services",
]

# Approval thresholds (USD). The single most exploited control in procurement:
# split one big need into several small POs, each just under a threshold.
APPROVAL_THRESHOLDS = [25_000, 100_000, 500_000]

AWARD_METHODS = ["open_tender", "limited_tender", "single_source"]
AWARD_METHOD_WEIGHTS = [0.60, 0.30, 0.10]

# ---------------------------------------------------------------- fraud volume
# Deliberately low. Real fraud is rare — a model that only works at 30%
# contamination is a model that only works in a slide deck.
FRAUD_RATES = {
    "po_splitting":        0.012,   # ~1.2% of tenders become split groups
    "duplicate_invoice":   0.010,
    "ghost_vendor":        0.008,
    "employee_collusion":  0.010,
    "round_number_abuse":  0.008,
    "bid_rigging_cartel":  0.012,
}

# How strongly a splitter hugs the threshold: 0.85–0.98 x limit
SPLIT_HUG_RANGE = (0.85, 0.98)

# Share of *honest* POs that land on a round 1,000. Without this, "round"
# alone perfectly separates fraud from clean — a generator artefact, not a
# signal. Real procurement is full of round numbers.
CLEAN_ROUND_RATE = 0.10

# Never let these reach a model. They are ground truth wearing a feature's
# clothes; no real ERP stamps them on a record.
LEAKY_COLUMNS = {"is_ghost"}

# Cartel ring shape
# Share of ghost vendors that hide behind a cutout shell instead of banking in
# the buyer's own name. These are the cases a SQL join cannot find — the ghost
# shares no identifier with ANY employee. Only a two-hop graph walk reaches the
# buyer. If this is 0.0, the graph layer is just a join with extra steps.
GHOST_LAYERED_RATE = 0.45

CARTEL_RING_SIZE = (3, 5)

# The slice of a category a ring goes after: the biggest contracts, where the
# money is. And how much of that slice they take. A ring that wins 5% of its
# market is not a cartel — it is a coincidence, and no method will ever find it
# because there is nothing there to find.
CARTEL_N_RINGS = 2
CARTEL_SEGMENT_TOP_SHARE = 0.22   # they operate in the top 22% by value
CARTEL_SEGMENT_CAPTURE = (0.65, 0.85)   # and take 65-85% of it
CARTEL_COVER_BID_MARKUP = (1.03, 1.18)   # losers bid 3–18% above the winner

# ------------------------------------------------------------------- filepaths
OUT_DIR = "data"
FILES = {
    "vendors":      f"{OUT_DIR}/vendors.csv",
    "employees":    f"{OUT_DIR}/employees.csv",
    "tenders":      f"{OUT_DIR}/tenders.csv",
    "bids":         f"{OUT_DIR}/bids.csv",
    "labels":       f"{OUT_DIR}/ground_truth.csv",
    "rule_flags":   f"{OUT_DIR}/rule_flags.csv",
}

# ------------------------------------------------------------- Day 3: rules
# Every threshold an auditor would actually defend in a meeting. Loose on
# purpose — a rules engine is meant to be high-recall. Precision comes later,
# from the anomaly model (Day 4) and the graph (Day 5) narrowing the list down.
RULE_THRESHOLDS = {
    "split_window_days":        14,     # how close together POs must sit to count as one burst
    "split_hug_ratio":          0.75,   # amount must be >= 75% of the limit to look deliberate
    "dup_window_days":          60,     # same vendor, same amount, within this many days
    "dup_amount_tolerance":     0.01,   # 1% — catches the "off by a rounding jitter" duplicates too
    "thin_competition_min_amt": 50_000, # single/limited-bidder only matters above this size
    "thin_competition_max_bid": 2,      # n_bidders <= this counts as thin
    "round_to":                 1000,   # amount must be a multiple of this to count as "round"
    "round_proximity":          0.15,   # ...and within 15% below the limit
    "concentration_min_tenders": 5,     # buyer needs this many awards before concentration means anything
    "concentration_share":      0.40,   # one vendor taking >40% of a buyer's spend is worth a look
    "new_vendor_min_amount":    40_000, # a "large" first order
    "new_vendor_max_days":      30,     # registered this close to their first award
}


# ─────────────────────────────────────────────────────────── Day 4: models
# No honest procurement signal separates fraud from clean on its own. If any
# single feature reaches this AUC by itself, it is not a feature — it is a bug
# in the generator leaking the answer. features.audit_features() halts the
# build. We shipped four such leaks once; we are not doing it twice.
LEAK_AUC_CEILING = 0.98

N_FOLDS = 5

# Isolation Forest: unsupervised, so it needs to be TOLD roughly how much of
# the data is anomalous. Note this is a guess about the world, not a peek at
# the labels — on TMC's real data you would set it from the audit team's prior,
# not from ground truth.
IFOREST_N_ESTIMATORS = 200
IFOREST_CONTAMINATION = 0.10

# Fixed, not fitted. Fitting blend weights against labels we planted ourselves
# would measure nothing. These are defensible in a meeting; a tuned number
# would not be.
BLEND_WEIGHTS = {
    "rules":   0.30,   # real domain knowledge, but blunt
    "iforest": 0.20,   # the only layer that needs no labels
    "xgboost": 0.50,   # strongest — when labels exist
}

FILES["features"]     = f"{OUT_DIR}/features.csv"
FILES["risk_scores"]  = f"{OUT_DIR}/risk_scores.csv"
FILES["rings"]        = f"{OUT_DIR}/rings.csv"
FILES["explanations"] = f"{OUT_DIR}/explanations.csv"


# ─────────────────────────────────────────────────────────── Day 5: the graph
# How far the identity graph will walk. 1 hop is what a SQL join can already
# do. 2 hops is the cutout case — the ghost shares nothing with any employee,
# but its bank belongs to a shell whose address belongs to the buyer. Beyond 2,
# real vendor masters are so densely tangled by coincidence that the paths stop
# meaning anything.
IDENTITY_MAX_HOPS = 3

# Of all co-bidding pairs, keep only the tightest few percent, then see who
# still clusters. In a working market this graph disintegrates: everyone
# competes with everyone a little. A ring survives the cut.
COBID_AFFINITY_QUANTILE = 0.97
COBID_MIN_CLUSTER = 3
COBID_MIN_BIDS = 8   # a vendor with 3 bids has no co-bidding pattern, only noise
