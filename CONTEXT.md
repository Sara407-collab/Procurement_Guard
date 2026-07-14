# ProcurementGuard — Project Context

Read this first. It is written to be the only thing you need. If a chat dies, a
laptop dies, or three weeks pass, start here and lose nothing.

**Live:** `https://procurementguard.streamlit.app`
**Repo:** `https://github.com/Sara407-collab/Procurement_Guard`

## One-line pitch
Procurement fraud & leakage detection for ERP data — rules engine + anomaly
detection + graph-based cartel discovery + explained alerts, deployed as a live
dashboard.

## Who this is for
The demo goes to my manager. Not TMC production, no real company data required.
So: free to build the best version of the idea. No access fights, no security
review.

---

## The two questions you WILL be asked. Memorise the answers.

### 1. *"SAP already does this."*
**It does.** *SAP Business Integrity Screening*, part of the GRC suite. **Never
claim SAP cannot do this** — a SAP consultant will call it in the first minute.

- Base SAP has **prevention**: approval limits, segregation of duties, three-way
  match. It **blocks** a PO over the limit.
- Base SAP has no **detection**. It checks one row at a time. It cannot see that
  three POs under the limit are one purchase in disguise.
- **PO splitting exists *because* the approval limit exists. The control creates
  the fraud.** ← *this sentence is the pitch*
- GRC is priced for the Fortune 500 — six figures plus 6–18 months of
  consultants. Mid-market SAP shops don't buy it, **not because they don't want
  detection, but because the only option costs more than the fraud does.**

> **A distribution gap, not a technology gap.**

### 2. *"A SQL join finds a shared bank account. Why a graph?"*
Because the careful fraudster does not bank in his own name:

```
V9002  --same bank-->  V8002 (shell, wins nothing)  --same address-->  E0049
  ^
  shares NOTHING with any employee
```

`SELECT * FROM vendors v JOIN employees e ON v.bank = e.bank` returns **zero
rows** for V9002. Two of the nine ghosts in this build are layered like this.

**Two hops is not a harder join. It is a different question.**

And a cartel is not in any row at all — see below.

---

## The domain — know this cold. It matters more than the code.

A procurement manager will never ask about PR-AUC. They will ask what PO
splitting is.

| scheme | what happens | where the signal lives |
|---|---|---|
| **PO splitting** | Need $135k, limit is $25k → raise six POs of ~$22k instead. Each obeys the rule. Nobody senior ever sees the real total. | several rows, one table |
| **Duplicate invoice** | Vendor bills $50k as INV-1023, gets paid, re-sends it 3 weeks later as INV-1023**A**. 5,000 invoices a month — nobody remembers. | two rows, one table |
| **Ghost vendor** | Employee invents a supplier that does not exist. Raises POs to it, "receives" goods that never came, company pays — **into the employee's own bank account.** | vendor ↔ employee, **1 or 2 hops** |
| **Employee collusion** | The vendor is **real** and delivers. But a buyer steers work to them: no proper tender, 1–2 bidders, inflated price. **Every record is legal.** | vendor ↔ employee + buyer-exclusivity |
| **Round-number abuse** | Invented amounts are round. Nobody fabricating an invoice writes $47,283.61 — they write $50,000. | one row |
| **Bid-rigging cartel** | 3–4 vendors secretly stop competing and **take turns winning.** Losers submit deliberately high **cover bids.** **Every single tender looks spotless.** | **the network. Nothing else.** |

**The last one is the whole project.** A real example from this build — tender
`T01508`:

```
V0108   $9,532   <- won        Five bidders. Cheapest wins.
V0069  $10,488                 Perfectly clean.
V0125  $10,546
V0092  $11,792
V0234  $13,021   <- 37% above the winner. A COVER BID.
                    V0108 and V0234 are in a ring together.
```

And the thing invisible in any single row — who won, in order:
```
V0148 → V0060 → V0234 → V0108 → V0148 → V0108 → V0060 → V0234 → ...
win share:   32%  ·  24%  ·  22%  ·  22%
```
**That is not a market. That is a rota.**

---

## Status: Days 1–7 COMPLETE, verified, leak-free, DEPLOYED

```bash
python -m src.main && python -m src.run_rules && python -m src.run_models
streamlit run app.py
```

**Current build: 3,130 tenders · 11,610 bids · 311 vendors (11 ghost, incl. 2
cutouts) · 60 employees · 274 fraudulent (8.75%) · $164M spend · $17.9M exposed.**

### ⚠️ The most important thing in this file

**This dataset leaked. Six times.** The first model hit **PR-AUC 0.950** by
exploiting bugs instead of learning anything.

| leak | precision as a lone detector |
|---|---|
| `vendors.is_ghost` — ground truth sitting in the vendor master | **1.000** |
| no *clean* amount was ever round → `amount % 5000 == 0` was a perfect tell | **1.000** |
| collusion inflated `tenders.amount` but never updated `bids` | **1.000** |
| splitting/duplicate/collusion rewrote the PO header, not the bid rows | **1.000** |
| *(Day 5)* cartel injection wiped every outside bidder → the exact same bidder set recurred 8–16 times | **1.000** |
| *(Day 5)* cartel rigged only ~5% of its category → co-bidding affinity diluted into noise | *unrealism, not a bug* |

The first four together: **precision 1.000, recall 0.609.** A four-line detector
with zero intelligence found **61% of all fraud.**

**Blacklisting features does NOT fix this.** `n_bidders` and `n_bid_rows` are
both obvious features; XGBoost just rebuilds the difference. **The leak was in
the *data*.**

### Two tripwires. NEVER delete them. NEVER raise the ceiling.

1. **`main.py :: assert_no_leakage()`** — every tender must satisfy
   `n_bidders == bid rows`, `award == winning bid`, no orphan bids, exactly one
   winner. **Build dies otherwise.**
2. **`features.py :: audit_features()`** — no single feature may reach AUC 0.98
   **against any ONE scheme**, not just against all fraud together.

**That per-scheme check is not a detail.** The first version only checked
globally, and `g_group_repeat` sailed straight through: it nailed **every cartel
tender perfectly** while scoring a bland global AUC of **0.57**, because the
other 200+ fraud rows drowned it out.

**Honest PR-AUC after the fixes: 0.83.** Lower than 0.950 and *real*.
**Lead the demo with this story. It is a headline, not an embarrassment.**

---

## Files

```
src/config.py          every tunable number, incl. LEAKY_COLUMNS, LEAK_AUC_CEILING
src/masters.py         vendor + employee master (bank/address/phone = graph edges)
src/base_activity.py   clean tenders/bids, NO fraud (the honest baseline)
src/fraud_injection.py 6 schemes, each recording its own ground truth
src/main.py            Day 1-2 orchestrator + assert_no_leakage()
src/rules.py           Day 3: 6 pandas rules
src/evaluate.py        precision/recall, precision@k (tie-broken by amount)
src/run_rules.py       Day 3 orchestrator
src/features.py        Day 4: 45 features + audit_features() tripwire
src/models.py          Day 4: Isolation Forest, XGBoost (ring-aware CV), blend
src/graph.py           Day 5: identity graph + co-bidding graph + find_rings()
src/explain.py         Day 7: SHAP, translated out of feature names into English
src/run_models.py      Days 4-7 orchestrator — prints the whole demo
app.py                 Day 6: Streamlit dashboard (root, NOT src/)
Dockerfile             Day 6: backup deploy route. Runs the pipeline at BUILD time.
PRD.md / TRD.md        product + technical docs, every number verified against data
data/*.csv             GENERATED. Never edit by hand. Rerun main.py.
```

Work in a `.venv`. It is the same empty box Docker will be — **if it runs there,
it runs in production.**

---

## The numbers (all verified against live data)

### Day 3 — rules
```
rule                     flagged  precision  recall
po_splitting                 114      99.1%   41.2%
duplicate_invoice             58      79.3%   16.8%
thin_competition             235      21.3%   18.2%   <- the false-positive factory
round_near_threshold          30      83.3%    9.1%
vendor_concentration          56      89.3%   18.2%
new_vendor_large_first         9     100.0%    3.3%
────────────────────────────────────────────────────
any rule fires               433      52.0%   82.1%
2+ rules fire                 64      98.4%   23.0%
```
Rules solve 5 of 6 schemes at ~100% recall. **Cartel: 2.0% (1/49).**

### Days 4–5 — the ablation (this is the thesis)
```
layer                       PR-AUC   P@20   P@50   P@100
Rules only                   0.550   100%    98%    70%
Isolation Forest only        0.328    70%    66%    53%   <- NO LABELS USED
Rules + Isolation Forest     0.628   100%    90%    78%
+ XGBoost                    0.804   100%   100%    99%
+ GRAPH (Day 5)              0.832   100%   100%   100%
```
Every layer earns its place. **The Isolation Forest is the only one that needs no
labels — it is the only layer that could run on TMC's real data tomorrow.** With
zero ground truth, 14 of its top 20 are real fraud: an **8× lift from nothing.**

### The cartel — and why this is the whole project
```
SUSPECTED CARTELS — found by the graph, with ZERO labels
#  vendors                    tenders  affinity  evenness  score  verdict
1  V0060,V0108,V0148,V0234         62     0.504     0.989   2.00  ** CARTEL001 — EXACT **
2  V0002,V0147,V0153               35     0.500     0.999   1.50  ** CARTEL000 — EXACT **
3  V0037,V0060,V0108,V0148         60     0.402     0.904   1.45  partial (dup of #1)
4  V0080,V0116,V0211               36     0.309     0.987   0.91  false positive
```
**2 cartels planted. Top 2 suspects match both EXACTLY — every member, no strays.**

Separation is clean and honest: core ring pairs **min affinity 0.463**; every
other pair **p99 = 0.286**. The *weakest* ring pair beats the 99th percentile of
everything else.

**But cartel stays at 1/49 in the tender queue, and that is the finding, not a
failure:**
```
fold 1:  TRAIN has 24 cartel  |  TEST has 25
fold 2:  TRAIN has 25 cartel  |  TEST has 24
fold 3:  TRAIN has 49 cartel  |  TEST has  0
```
**Only 2 rings.** Leave one out and the model trains on **ONE example of a
cartel.** No model learns from one example — **and no company on earth has a
hundred labelled cartels to learn from.**

> **Supervised learning cannot solve cartels. Not here, not anywhere.**
> **The graph needs no labels at all. That is why it exists.**

**So the product has two outputs, not one:**
- **Alerts** — a ranked tender queue (rules + anomaly + XGBoost + graph)
- **Cartels** — suspected rings (graph). *This is what no tool gives you.*

### Day 7 — explanations
SHAP, computed **per fold** by the model that never saw the row, then
**translated out of feature names and into English**:

> *"This vendor takes **100%** of all its work from this one buyer. A typical
> vendor gets **10%**."*

**Every sentence guards itself.** SHAP will happily say a feature pushed risk up
while its value is *low*, which produced things like *"These firms turn up on the
same tenders **0%** of the time"* → +0.54 risk. **That is nonsense wearing
evidence's clothes**, and an auditor who reads one line like that stops believing
the other four. Guards dropped the reasons from **15,650 → 5,217.** The survivors
are all true.

**Bar length is the SHAP value. The number is never shown.** `+0.31` tells an
auditor nothing they can act on.

### The slides that land
- **"An auditor has 20 hours this week."** Random 20 → **1.8** real cases.
  ProcurementGuard's top 20 → **20 real cases. 11.4×.**
- **Two queues that disagree.** Total exposure **$17.9M**. Top 20 by `risk_score`
  recovers $4.3M. Top 20 by **`expected_loss`** (probability × amount) recovers
  **$7.9M — nearly 2× the money, same 100% precision.** Top 50 by expected loss:
  **63.4% of all exposure.**
- **Never quote "accuracy."** 91% of tenders are clean, so a model that flags
  **nothing** scores 91%. Quote **Precision@20, PR-AUC, and lift.**

---

## Roadmap — Day 8 to 14

- **Day 8**: LLM narration (Claude API) — SHAP + rule hits → a paragraph an
  auditor can paste into an email. Cache responses. **Key goes in `.env`. `.env`
  is git-ignored from day one.** **First thing to cut if behind.**
- **Day 9**: 🎯 **THE ONE THAT MATTERS.** Re-run at **1% and 3% contamination**
  (real fraud is rarer than 8.75%; if precision collapses, that is an honest
  finding and reporting it makes the work *more* credible). **Then: World Bank
  data.** Public contract awards + their public **debarred-firms list**. Run the
  same pipeline on real procurement. **If even one firm in the top 20 was really
  debarred for corruption, that single slide outweighs every synthetic number in
  this file.**
- **Day 10–11**: Dashboard polish, README, architecture diagram, 5 slides, demo
  script + rehearsal
- **Day 12–14**: Buffer. Something always breaks here.

## Non-negotiables
1. **Never delete a tripwire. Never raise the ceiling.** If one fires, something
   is wrong — go and find it.
2. **Never tune hyperparameters against labels you planted yourself.** It
   measures how well you tuned. Nothing else.
3. **Never quote accuracy.** 91% for doing nothing.
4. **Never train at page load.** Batch pipeline, serving dashboard.
5. **Never commit a secret.** `.env` git-ignored; Streamlit secrets in prod.
6. Streamlit, not React. No Kubernetes, no Kafka. Procurement audit is a batch
   process by nature.
7. **When the generator changes, re-verify every number in PRD.md, TRD.md and
   this file.** They are derived, not written. We have already shipped stale
   figures once, and one fabricated identifier.
8. If behind, cut in this order: LLM narration → dashboard polish → XGBoost.
   (Rules + Isolation Forest + the ring detector still tell the whole story.)

## Known limitations — say these BEFORE anyone else does
- **Everything is measured on data we generated.** It proves the *code* works. It
  does **not** prove the *idea* works. **That is what Day 9 is for. Do not
  oversell until it lands.**
- **8.75% contamination is generous.** Real procurement fraud is 1–3%, often
  under 1%.
- **The rules may be flattering themselves** — tested against a generator written
  by the same person who wrote the rules. Real data will be muddier.
- **n = 2 cartel rings.** The detector found both exactly. **A demonstration, not
  a statistic.**

## How to run
```bash
python -m venv .venv && .\.venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -m src.main        # → vendors, employees, tenders, bids, ground_truth
python -m src.run_rules   # → rule_flags.csv
python -m src.run_models  # → features, risk_scores, rings, explanations
streamlit run app.py      # → localhost:8501
```
`main.py` must end with **`✓ invariants hold`**. `run_models.py` must print
**`✓ leak audit passed`**. **If either line is missing, STOP. Do not trust a
single number downstream.**

## To continue
Tell Claude: *"Read CONTEXT.md. Days 1–7 are done, verified, and deployed at
procurementguard.streamlit.app. Let's build Day 8 — LLM narration."*

Or, better: *"…Let's skip to Day 9 — the World Bank real-data test. That is the
one that decides whether any of this is real."*