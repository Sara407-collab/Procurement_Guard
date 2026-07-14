# ProcurementGuard — Project Context

Read this first. It is written to be the only thing you need. If a chat dies, a
laptop dies, or three weeks pass, start here and lose nothing.

## One-line pitch
Procurement fraud & leakage detection for ERP data — rules engine + anomaly
detection + graph-based cartel detection + LLM-explained alerts, deployed as a
live dashboard.

## Who this is for
The demo goes to my manager. Not to TMC production, no real company data
required. That means: free to build the best version of the idea, no access
fights, no security review.

## The competitor answer — memorise this
SAP **does** sell fraud detection: *SAP Business Integrity Screening*, part of
the GRC suite. **Never claim SAP cannot do this** — a SAP consultant will call
it instantly. The real argument:

- Base SAP has **prevention**: approval limits, segregation of duties,
  three-way match. It **blocks** a PO over the limit.
- Base SAP has no **detection**. It checks one row at a time. It cannot see
  that three POs under the limit are one purchase in disguise.
- **PO splitting exists *because* the approval limit exists.** The control
  creates the fraud. ← *this sentence is the pitch*
- GRC is priced for the Fortune 500. Mid-market SAP shops don't buy it — not
  because they don't want detection, but because the only option costs more
  than the fraud. **A distribution gap, not a technology gap.**

And when someone says *"a SQL join finds a shared bank account, why a graph?"*:
- The careful fraudster does not bank in his own name. He puts a cutout in
  between: `ghost ──same bank── cutout ──same address── buyer`. The ghost shares
  nothing with any employee. **A join returns an empty set.**
- And a cartel is not in any row at all. See below.

## The domain — know this cold, it matters more than the code
A procurement manager will never ask about PR-AUC. They will ask what PO
splitting is.

| scheme | what happens | where the signal lives |
|---|---|---|
| **PO splitting** | Need 60k, limit is 25k → raise three 20k POs instead, same vendor, same week, self-approved. Boss never sees it. | several rows, one table |
| **Duplicate invoice** | Vendor bills 50k, gets paid, re-sends the same bill 3 weeks later as INV-1023**A**. Nobody remembers. Paid twice. | two rows, one table |
| **Ghost vendor** | Employee invents a supplier that does not exist. Raises POs to it, "receives" goods that never came, company pays — into the employee's own bank account. | vendor ↔ employee, **1 or 2 hops** |
| **Employee collusion** | The vendor is **real** and delivers. But a buyer steers work to them: no proper tender, 1–2 bidders, inflated price. Every record is legal. | vendor ↔ employee + buyer-exclusivity |
| **Round-number abuse** | Invented amounts are round. Nobody fabricating an invoice writes 47,283.61 — they write 50,000. Worse when it hugs a ceiling. | one row |
| **Bid-rigging cartel** | 3–4 vendors secretly stop competing and **take turns winning**. Losers submit deliberately high **cover bids**. **Every single tender looks clean.** The pattern only exists across the whole history. | **the network. Nothing else.** |

---

## Status: Days 1–5 COMPLETE, verified, leak-free

```bash
python -m src.main && python -m src.run_rules && python -m src.run_models
```
Runs clean end to end. **Current build: 3,130 tenders, 274 fraudulent (8.75%),
11,612 bids, 311 vendors (11 ghost incl. 2 cutouts).**

### ⚠️ The most important thing in this file

**This dataset leaked. Four times. Then twice more.** The first model hit
PR-AUC 0.950 by exploiting generator bugs instead of learning anything:

| leak | precision as a lone detector |
|---|---|
| `vendors.is_ghost` — ground truth sitting in the vendor master | **1.000** |
| no *clean* amount was ever round → `amount % 5000 == 0` was a perfect tell | **1.000** |
| collusion inflated `tenders.amount` but never updated `bids` | **1.000** |
| splitting/duplicate/collusion rewrote the PO header, not the bid rows | **1.000** |

Together: **precision 1.000, recall 0.609.** A four-line detector with zero
intelligence found 61% of all fraud.

**Blacklisting features does NOT fix this.** `n_bidders` and `n_bid_rows` are
both obvious features; XGBoost just rebuilds the difference. The leak was in the
*data*. It had to be fixed in the generator.

Then on Day 5 the graph leaked too: `g_group_repeat` (the exact same bidder set,
over and over) separated cartel **perfectly** — because the injector wiped every
outside bidder. That is a fingerprint of the injector, not a conspiracy.

**Two tripwires now exist. Never delete them:**
1. `main.py :: assert_no_leakage()` — every tender must satisfy
   `n_bidders == bid rows`, `award == winning bid`, no orphan bids, exactly one
   winner. **The build dies otherwise.**
2. `features.py :: audit_features()` — no single feature may reach AUC 0.98
   **against any one scheme**, not just against all fraud together. (The first
   version only checked globally, and `g_group_repeat` sailed straight through:
   it nailed 43/43 cartels while scoring a bland 0.57 overall.)

**Honest PR-AUC: 0.82.** Lower than 0.950 and real.
*Lead the demo with this story. It is a headline, not an embarrassment.*

### Files
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
src/run_models.py      Day 4+5 orchestrator, prints the whole demo
requirements.txt       pinned — Day 6 deploys from this
data/*.csv             GENERATED. Never edit by hand. Rerun main.py.
```
Work in a `.venv`. It is the same empty box Docker will be on Day 6 — if it runs
there, it will run in Docker.

---

## The numbers

### Day 3 — rules
```
rule                     flagged  precision  recall
po_splitting                 114      99.1%   41.2%
duplicate_invoice             58      79.3%   16.8%
thin_competition             235      21.3%   18.2%   ← the false-positive factory
round_near_threshold          30      83.3%    9.1%
vendor_concentration          56      89.3%   18.2%
new_vendor_large_first         9     100.0%    3.3%
────────────────────────────────────────────────────
any rule fires               433      52.0%   82.1%
2+ rules fire                 64      98.4%   23.0%
```
Rules solve 5 of 6 schemes at ~100% recall. **Cartel: 2.0% (1/49).**

### Day 4 + 5 — the ablation (this is the thesis)
```
layer                       PR-AUC   P@20   P@50   P@100
Rules only                   0.550   100%    98%    70%
Isolation Forest only        0.333    65%    66%    55%   ← NO LABELS USED
Rules + Isolation Forest     0.634   100%    92%    79%
Full blend (+ XGBoost)       0.798   100%   100%   100%
+ GRAPH (Day 5)              0.822   100%   100%   100%
```
Every layer earns its place. The Isolation Forest is the only layer that needs
no labels — **it is the only one that could run on TMC's real data tomorrow.**

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

**But cartel stays at 2% in the tender queue, and that is the finding, not a
failure.** There are only 2 rings. Ring-aware CV means that when one ring is in
the test fold, the model trained on ONE example of a cartel. **No model learns
from one example — and no company on earth has 100 labelled cartels to learn
from.** Supervised learning cannot solve cartels, ever.

**The graph needs no labels at all.** That is why it exists.

**So the product has two outputs, not one:**
- **Alerts** — a ranked tender queue (rules + anomaly + XGBoost)
- **Rings** — suspected cartels (graph). *This is what no tool gives you.*

### The slides that land
- **"An auditor has 20 hours this week."** Random 20 → 1.8 real cases.
  ProcurementGuard's top 20 → **20 real cases. 11.4x.**
- **Two queues, and they disagree.** Total fraud exposure **$17.9M**.
  Top 20 by `risk_score` → $4.3M recovered. Top 20 by `expected_loss`
  (probability × amount) → **$7.9M — nearly 2x the money, same 100% precision.**
  Top 50 by expected loss → **63.6% of all exposure.**
- **Never quote "accuracy."** 91% of tenders are clean, so a model that flags
  nothing scores 91% accuracy. Quote Precision@20, PR-AUC, and lift.

---

## Roadmap — Day 6 to 14

- **Day 6**: 🚨 **DEPLOY. CANNOT SLIP.** Streamlit dashboard (two tabs: Alerts,
  Rings) + Dockerfile + free-tier hosting. Ugly-but-working first. Deployment
  always breaks; that is why it has its own day.
- **Day 7**: SHAP on XGBoost, wired into the dashboard (click alert → see why)
- **Day 8**: LLM narration (Claude API) — SHAP + rule hits → plain-English alert.
  Cache responses. **First thing to cut if behind.**
- **Day 9**: Evaluation. **Re-run at 1% and 3% contamination** — real fraud is
  rarer than 8.75%, and if precision collapses that is an honest finding worth
  reporting. **Then: World Bank data.** Public contract awards + the public
  debarred-firms list. Run the same pipeline on real procurement. **If even one
  firm in the top 20 was really debarred, that single slide outweighs every
  synthetic number in this file.**
- **Day 10**: Dashboard polish — filters, CSV export, embed a pyvis ring graph
- **Day 11**: README, architecture diagram, 5 slides, demo script + rehearsal
- **Day 12–14**: Buffer. Something always breaks here.

## Non-negotiables
1. **Day 6 deploy cannot slip.** Ugly-but-working first, polish after.
2. **Never delete a tripwire, never raise the ceiling.** If one fires, something
   is wrong. Fix the cause.
3. Streamlit, not React. This is a 2-week project.
4. No Kubernetes, no Kafka. Procurement audit is a batch process by nature.
5. **Don't hand-tune hyperparameters.** Tuning against labels you planted
   yourself measures nothing except how well you tuned.
6. If behind, cut in this order: LLM narration → dashboard polish → XGBoost.
   (Rules + Isolation Forest + the ring detector still tell the whole story.)

## Known limitations — say these before anyone else does
- **Everything is measured on data we generated.** The numbers prove the code
  works; they do not prove the *idea* works. That is what Day 9's World Bank
  test is for. Don't oversell until it's done.
- **8.75% contamination is generous.** Real procurement fraud is 1–3%, often
  under 1%.
- **The rules may be flattering themselves** — tested against a generator
  written by the same person. Real data will be muddier.
- **Only 2 cartel rings.** The ring detector found both exactly, but n=2 is a
  demonstration, not a statistic.

## How to run
```bash
python -m venv .venv && .\.venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -m src.main        # → vendors, employees, tenders, bids, ground_truth
python -m src.run_rules   # → rule_flags.csv
python -m src.run_models  # → features.csv, risk_scores.csv
```
`main.py` must end with `✓ invariants hold`. `run_models.py` must print
`✓ leak audit passed`. **If either line is missing, stop. Do not trust a single
number downstream.**

## To continue
Tell Claude: *"Read CONTEXT.md. Days 1–5 are done and verified — the graph found
both cartels exactly. Let's build Day 6: deploy. Streamlit dashboard with two
tabs (Alerts, Rings), Dockerfile, free-tier hosting. This day cannot slip."*