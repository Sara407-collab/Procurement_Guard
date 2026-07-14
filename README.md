# ProcurementGuard

Fraud and leakage detection for ERP procurement data.

**Day 1 + Day 2 complete.** This repo currently builds a fully labelled
procurement dataset with six fraud schemes planted in it. Everything downstream
— rules engine, anomaly model, graph layer, dashboard — trains and is measured
against these labels.

## Run it

```bash
python -m src.main
```

Writes five CSVs to `data/` and prints a build report.

## What comes out

| file | rows | what it is |
|---|---|---|
| `vendors.csv` | ~309 | vendor master — bank account, address, phone, tax ID |
| `employees.csv` | 60 | buyers and approvers, each with an approval limit |
| `tenders.csv` | ~3,164 | one winning PO per tender |
| `bids.csv` | ~11,700 | every bid submitted, winners and losers |
| `ground_truth.csv` | ~3,164 | `is_fraud`, `fraud_type`, `ring_id` |

## The six schemes

| scheme | what the fraudster does | how it shows up |
|---|---|---|
| **PO splitting** | breaks one large need into several POs, each just under an approval limit | cluster of same-vendor, same-buyer POs hugging a threshold |
| **Duplicate invoice** | bills the same amount twice under a near-identical invoice number | repeated amount, same vendor, weeks apart |
| **Ghost vendor** | invents a supplier that does not exist | registered days before first award, single-source, shares a bank account with a buyer |
| **Employee collusion** | a real vendor quietly linked to a buyer who steers work to them | shared bank / address / phone, thin competition, inflated amounts |
| **Round-number abuse** | fabricates amounts, and fabricated amounts are round | suspiciously round values sitting just below a ceiling |
| **Bid-rigging cartel** | several vendors stop competing, take turns winning, submit cover bids | rotation across tenders, losers clustered just above the winner |

## Why the labels are trustworthy

The clean world is generated first, in `base_activity.py`, with no fraud in it
at all. Fraud is layered on afterwards in `fraud_injection.py`, and every
function there records exactly which tenders it touched. A row is labelled
fraudulent because we put fraud in it — never because the generator drifted.

`src/main.py` ends with assertions that prove it: split POs really do sit under
their approval limit, ghost vendors really do share a bank account with a buyer,
cartel rings really do rotate their winners.

## Tuning

Everything lives in `src/config.py`. The number worth arguing about is
`FRAUD_RATES` — the current build lands around 9% contamination, which is
generous. Real procurement fraud is rarer. Lower the rates and re-run to see
how far precision holds up; that experiment is worth a slide on its own.

## Next

- **Day 3** — rules engine (6 rules, pure pandas). Highest value per hour in the
  whole project.
- **Day 4** — features + Isolation Forest + XGBoost → blended risk score
- **Day 5** — graph layer (NetworkX): vendor↔employee edges, cartel rings
- **Day 6** — deploy. FastAPI + Streamlit + Docker. Non-negotiable.
