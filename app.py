"""
ProcurementGuard — the dashboard.

    streamlit run app.py

This is built for an auditor, not for a data scientist. An auditor has about
twenty hours this week and four questions:

    What should I look at?      → a ranked queue
    Why this one?               → the reasons, in English
    How much money is at risk?  → expected loss, in dollars
    Is anyone working together? → the ring detector

So there is no PR-AUC on this page. No ROC curve, no confusion matrix, no
feature importances. Those belong in the deck, where they are the evidence that
the queue can be trusted — not on the screen of the person working the queue.

It reads CSVs. It does not train anything. Procurement audit is a batch process:
the pipeline runs, the dashboard serves. A dashboard that retrains on page load
is a dashboard that times out on page load.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from src import narrate as NARR
except Exception:
    import narrate as NARR

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="ProcurementGuard", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

# ── styling ──────────────────────────────────────────────────────────────────
# A ledger, not a fintech dashboard. This tool points a finger at people; it
# should look sober. Figures are monospaced so they line up in a column, the way
# they do on a page of accounts. One accent, used only for money at risk.
st.markdown("""
<style>
  :root {
    --ink:    #17191C;
    --muted:  #6E7681;
    --rule:   #E3E1DC;
    --paper:  #FBFAF8;
    --flag:   #9E2A20;
    --ring:   #1F4E5F;
  }
  .stApp { background: var(--paper); }
  html, body, [class*="css"] { color: var(--ink); }

  .masthead {
    border-bottom: 2px solid var(--ink);
    padding-bottom: .5rem; margin-bottom: 1.4rem;
  }
  .masthead h1 {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.65rem; font-weight: 600; letter-spacing: -.01em;
    margin: 0; color: var(--ink);
  }
  .masthead p {
    margin: .25rem 0 0; color: var(--muted);
    font-size: .82rem; letter-spacing: .02em;
  }
  .stat { border-left: 2px solid var(--rule); padding: .1rem 0 .1rem .85rem; }
  .stat .n {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 1.45rem; font-weight: 600; line-height: 1.2;
  }
  .stat .k {
    font-size: .68rem; text-transform: uppercase; letter-spacing: .09em;
    color: var(--muted); margin-top: .2rem;
  }
  .money .n { color: var(--flag); }
  .ringstat .n { color: var(--ring); }

  .why {
    border-left: 2px solid var(--flag); padding: .55rem 0 .55rem .85rem;
    font-size: .87rem; line-height: 1.55; color: #2E3238; margin: .3rem 0;
  }
  .why.quiet { border-left-color: var(--rule); color: var(--muted); }

  /* The evidence. Each reason gets a bar whose length is how hard it pushed
     this tender up the queue — SHAP, but never shown as a number, because
     "+0.31" tells an auditor nothing. The bar is the number. */
  .ev { margin: 0 0 1.05rem; }
  .ev-top {
    display: flex; align-items: baseline; gap: .6rem; margin-bottom: .28rem;
  }
  .ev-rank {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: .68rem; color: var(--muted); min-width: 1.1rem;
  }
  .ev-label {
    font-weight: 600; font-size: .9rem; letter-spacing: -.005em;
    color: var(--ink);
  }
  .ev-track {
    height: 5px; background: #EFEDE8; border-radius: 3px;
    margin: .1rem 0 .4rem; overflow: hidden;
  }
  .ev-fill {
    height: 100%; background: var(--flag); border-radius: 3px;
    animation: grow .55s cubic-bezier(.2,.8,.3,1) both;
  }
  .ev-fill.g { background: var(--ring); }        /* the graph found this one */
  .ev-fill.r { background: #8A6D3B; }            /* a human-written rule did  */
  @keyframes grow { from { width: 0 } }
  @media (prefers-reduced-motion: reduce) { .ev-fill { animation: none } }
  .ev-text {
    font-size: .86rem; line-height: 1.6; color: #3A3F45;
    padding-left: 1.7rem;
  }
  .ev-src {
    font-size: .62rem; text-transform: uppercase; letter-spacing: .09em;
    color: var(--muted); margin-left: .35rem;
  }
  .ev-legend {
    font-size: .68rem; color: var(--muted); letter-spacing: .04em;
    margin: -.3rem 0 1rem;
  }
  .chip {
    display: inline-block; width: 8px; height: 8px; border-radius: 2px;
    margin-right: .3rem; vertical-align: middle;
  }

  /* The narration: the whole case in one short paragraph, ready to paste into
     an email. Set slightly apart, like a summary at the top of a memo. */
  .brief {
    background: #F7F5F1; border: 1px solid var(--rule); border-radius: 6px;
    padding: 1rem 1.15rem; margin: .2rem 0 1.3rem;
    font-size: .92rem; line-height: 1.62; color: #2A2E33;
  }
  .brief-tag {
    display: inline-block; font-size: .6rem; text-transform: uppercase;
    letter-spacing: .1em; color: var(--muted); margin-bottom: .5rem;
  }
  .rot {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: .95rem; letter-spacing: .06em; line-height: 2.1;
    color: var(--ring);
  }
  .note { color: var(--muted); font-size: .82rem; line-height: 1.6; }
  [data-testid="stMetricValue"] { font-family: ui-monospace, monospace; }
</style>
""", unsafe_allow_html=True)


# ── data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    missing = [f for f in ("risk_scores.csv", "rule_flags.csv", "tenders.csv",
                           "bids.csv", "rings.csv", "explanations.csv")
               if not (DATA / f).exists()]
    if missing:
        return None, missing
    d = {n: pd.read_csv(DATA / f"{n}.csv") for n in
         ("risk_scores", "rule_flags", "tenders", "bids", "rings",
          "explanations")}
    alerts = (d["risk_scores"]
              .merge(d["rule_flags"][["tender_id", "rule_reasons", "n_rules_triggered"]],
                     on="tender_id")
              .merge(d["tenders"][["tender_id", "po_id", "award_date", "category",
                                   "department", "award_method", "n_bidders",
                                   "approval_limit"]], on="tender_id"))
    alerts["rule_reasons"] = alerts["rule_reasons"].fillna("")
    return {"alerts": alerts, "bids": d["bids"], "tenders": d["tenders"],
            "rings": d["rings"], "why": d["explanations"]}, []


data, missing = load()

st.markdown("""
<div class="masthead">
  <h1>ProcurementGuard</h1>
  <p>Fraud and leakage detection for ERP procurement data</p>
</div>
""", unsafe_allow_html=True)

if data is None:
    st.error("No scored data yet.")
    st.markdown(f"""<div class="note">
    Missing: <code>{'</code>, <code>'.join(missing)}</code><br><br>
    Build it first — the dashboard serves results, it does not compute them:
    <pre>python -m src.main
python -m src.run_rules
python -m src.run_models</pre></div>""", unsafe_allow_html=True)
    st.stop()

alerts, bids, tenders, rings, why = (data["alerts"], data["bids"],
                                     data["tenders"], data["rings"],
                                     data["why"])

# ── controls ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### The week")
    capacity = st.slider("Tenders you can review", 5, 100, 20, step=5,
                         help="An auditor's real constraint. Everything below "
                              "the line goes unchecked.")
    order = st.radio(
        "Rank by",
        ["Money at risk", "Likelihood"],
        help="They disagree, and the disagreement is the point. Likelihood "
             "finds MORE cases. Money at risk finds BIGGER ones.")
    sort_col = "expected_loss" if order == "Money at risk" else "risk_score"

    st.markdown("---")
    _has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    USE_LLM = st.toggle(
        "Write briefs with Claude", value=_has_key, disabled=not _has_key,
        help="On: each brief is written by Claude from the findings. "
             "Off (or no API key set): a deterministic template writes it. "
             "Either way, only the findings on screen are ever used — the brief "
             "cannot introduce a fact that is not in the evidence below.")
    if not _has_key:
        st.caption("Set ANTHROPIC_API_KEY to enable Claude-written briefs.")

    st.markdown("---")
    st.markdown("#### Narrow it down")
    cats = st.multiselect("Category", sorted(alerts["category"].unique()))
    depts = st.multiselect("Department", sorted(alerts["department"].unique()))
    min_amt = st.number_input("Minimum amount ($)", 0, 2_000_000, 0, step=10_000)

q = alerts.copy()
if cats:
    q = q[q["category"].isin(cats)]
if depts:
    q = q[q["department"].isin(depts)]
q = q[q["amount"] >= min_amt]
queue = q.nlargest(capacity, sort_col).reset_index(drop=True)
queue.insert(0, "rank", range(1, len(queue) + 1))

tab_alerts, tab_rings, tab_about = st.tabs(
    ["  Alerts  ", "  Cartels  ", "  How it works  "])

# ══════════════════════════════════════════════════════════════════ ALERTS
with tab_alerts:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat"><div class="n">{len(queue)}</div>'
                    f'<div class="k">In your queue</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat money"><div class="n">'
                    f'${queue["amount"].sum():,.0f}</div>'
                    f'<div class="k">Spend under review</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat"><div class="n">'
                    f'{queue["n_rules_triggered"].gt(0).sum()}</div>'
                    f'<div class="k">Trip a written rule</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat"><div class="n">{len(q):,}</div>'
                    f'<div class="k">Tenders in scope</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    show = queue[["rank", "tender_id", "po_id", "award_date", "vendor_id",
                  "employee_id", "category", "amount", "expected_loss",
                  "n_bidders"]].copy()
    show.columns = ["#", "Tender", "PO", "Awarded", "Vendor", "Buyer",
                    "Category", "Amount", "At risk", "Bidders"]
    st.dataframe(
        show, hide_index=True, width='stretch', height=430,
        column_config={
            "Amount":  st.column_config.NumberColumn(format="$%d"),
            "At risk": st.column_config.NumberColumn(
                format="$%d",
                help="Probability x amount. A 90%-likely fraud worth $500 is "
                     "not worth your morning. A 40%-likely one worth $400,000 is."),
        })

    st.download_button(
        "Export this queue (CSV)",
        queue.to_csv(index=False).encode(),
        f"procurementguard_queue_{sort_col}_{capacity}.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### Why these were flagged")

    pick = st.selectbox(
        "Open an alert",
        queue["tender_id"],
        format_func=lambda t: (
            f"#{queue.loc[queue.tender_id == t, 'rank'].iloc[0]}  ·  {t}  ·  "
            f"${queue.loc[queue.tender_id == t, 'amount'].iloc[0]:,.0f}  ·  "
            f"{queue.loc[queue.tender_id == t, 'vendor_id'].iloc[0]}"))

    row = queue[queue["tender_id"] == pick].iloc[0]
    left, right = st.columns([3, 2])

    with left:
        ev = why[why["tender_id"] == pick].sort_values(
            "contribution", ascending=False)

        # ── the one-paragraph brief (Day 8) ─────────────────────────────────
        reasons = ev.to_dict("records")
        brief, engine = NARR.narrate(pick, row["amount"], reasons,
                                     prefer_llm=USE_LLM)
        tag = ("written by Claude" if engine == "claude"
               else "auto-generated summary")
        st.markdown(f'<div class="brief"><span class="brief-tag">Auditor brief · '
                    f'{tag}</span>{brief}</div>', unsafe_allow_html=True)

        if len(ev):
            st.markdown(
                '<div class="ev-legend">'
                '<span class="chip" style="background:#9E2A20"></span>the transaction'
                '&nbsp;&nbsp;&nbsp;'
                '<span class="chip" style="background:#1F4E5F"></span>the network'
                '&nbsp;&nbsp;&nbsp;'
                '<span class="chip" style="background:#8A6D3B"></span>a written rule'
                '</div>', unsafe_allow_html=True)

            top = ev["contribution"].max()
            for i, r in enumerate(ev.itertuples(), 1):
                if r.feature.startswith("g_"):
                    cls, src = "g", "network"
                elif r.feature.startswith("rule_") or r.feature == "n_rules_triggered":
                    cls, src = "r", "rule"
                else:
                    cls, src = "", "transaction"
                w = max(4, 100 * r.contribution / top)
                st.markdown(f"""
<div class="ev">
  <div class="ev-top">
    <span class="ev-rank">{i}</span>
    <span class="ev-label">{r.label}<span class="ev-src">{src}</span></span>
  </div>
  <div class="ev-track"><div class="ev-fill {cls}" style="width:{w:.0f}%"></div></div>
  <div class="ev-text">{r.text}</div>
</div>""", unsafe_allow_html=True)

            st.markdown(
                '<div class="note">Bar length is how hard each fact pushed this '
                'tender up the queue, measured with SHAP against the model that '
                'never saw it. It is deliberately not shown as a number: '
                '"+0.31" tells an auditor nothing they can act on.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="why quiet">Nothing here rose above the noise. This '
                'tender sits in the queue on the weight of many small signals, '
                'none of them decisive on its own. Treat it as the weakest kind '
                'of lead.</div>', unsafe_allow_html=True)

        tb = bids[bids["tender_id"] == pick].sort_values("bid_amount")
        if len(tb):
            st.markdown("**Bids on this tender**")
            b = tb[["vendor_id", "bid_amount", "is_winner"]].copy()
            b.columns = ["Vendor", "Bid", "Won"]
            st.dataframe(b, hide_index=True, width='stretch',
                         column_config={"Bid": st.column_config.NumberColumn(
                             format="$%d")})

    with right:
        st.markdown(f"""
| | |
|---|---|
| **Awarded** | {row['award_date']} |
| **Vendor** | `{row['vendor_id']}` |
| **Buyer** | `{row['employee_id']}` |
| **Amount** | ${row['amount']:,.0f} |
| **Approval limit** | ${row['approval_limit']:,.0f} |
| **Method** | {row['award_method']} |
| **Bidders** | {row['n_bidders']} |
| **Money at risk** | **${row['expected_loss']:,.0f}** |
""")

# ══════════════════════════════════════════════════════════════════ CARTELS
with tab_rings:
    st.markdown(
        '<div class="note">A cartel is not a property of a tender — it is a '
        'property of a <b>group of vendors</b>. Every rigged tender, on its own, '
        'looks spotless: several bidders, lowest price wins. The arrangement '
        'only exists across the whole history.<br><br>'
        'Nothing below used a single label. Only who bids against whom, and who '
        'takes their turn winning.</div><br>',
        unsafe_allow_html=True)

    if not len(rings):
        st.info("No suspected rings. Run `python -m src.run_models` first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat ringstat"><div class="n">'
                        f'{len(rings)}</div><div class="k">Suspected rings'
                        f'</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat ringstat"><div class="n">'
                        f'{int(rings["n_tenders"].sum())}</div>'
                        f'<div class="k">Tenders they touched</div></div>',
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        r = rings.copy()
        r.insert(0, "#", range(1, len(r) + 1))
        disp = r[["#", "vendors", "n_vendors", "n_tenders", "mean_affinity",
                  "win_evenness", "ring_score"]]
        disp.columns = ["#", "Vendors", "Members", "Tenders", "Co-bidding",
                        "Win evenness", "Score"]
        st.dataframe(
            disp, hide_index=True, width='stretch',
            column_config={
                "Co-bidding": st.column_config.NumberColumn(
                    format="%.2f",
                    help="Of the tenders these firms could have met on, how "
                         "often did they actually meet? Honest rivals: ~0.13."),
                "Win evenness": st.column_config.NumberColumn(
                    format="%.2f",
                    help="1.00 = everyone gets an equal turn. Honest "
                         "competition has a cheapest supplier, and it wins more "
                         "than its share."),
            })

        st.markdown("---")
        sel = st.selectbox("Open a ring", r["#"],
                           format_func=lambda i: (
                               f"Ring {i}  ·  "
                               f"{r.loc[r['#'] == i, 'n_vendors'].iloc[0]} vendors  ·  "
                               f"score {r.loc[r['#'] == i, 'ring_score'].iloc[0]:.2f}"))
        ring = r[r["#"] == sel].iloc[0]
        members = ring["vendors"].split(",")

        st.markdown(f"**Members** — `{'`  `'.join(members)}`")

        # The rotation. This is the evidence: not a score, a sequence.
        #
        # A ring's tenders are the ones it RIGGED — where two or more members
        # showed up together and one of them won. Not every tender a member ever
        # touched: they also bid honestly elsewhere, and mixing those in buries
        # the rota under noise.
        ring_bids = bids[bids["vendor_id"].isin(members)]
        rigged = (ring_bids.groupby("tender_id")["vendor_id"].nunique()
                  .loc[lambda x: x >= 2].index)
        won = tenders[tenders["tender_id"].isin(rigged)
                      & tenders["vendor_id"].isin(members)].copy()
        won["award_date"] = pd.to_datetime(won["award_date"])
        won = won.sort_values("award_date")

        st.markdown("**Who won, in order**")
        st.markdown(
            f'<div class="rot">{"  →  ".join(won["vendor_id"].head(24))}</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="note">Read it left to right. In an honest market the '
            'cheapest supplier wins more often than the rest, and the order is '
            'noise. Here they are taking turns.</div>',
            unsafe_allow_html=True)

        share = (won["vendor_id"].value_counts(normalize=True)
                 .rename("share of wins").to_frame())
        st.bar_chart(share, height=200)

        w = won[["tender_id", "award_date", "vendor_id", "employee_id",
                 "category", "amount", "n_bidders"]].copy()
        w.columns = ["Tender", "Awarded", "Won by", "Buyer", "Category",
                     "Amount", "Bidders"]
        st.dataframe(w, hide_index=True, width='stretch', height=280,
                     column_config={"Amount": st.column_config.NumberColumn(
                         format="$%d")})
        st.download_button(
            "Export this ring's tenders (CSV)",
            won.to_csv(index=False).encode(),
            f"procurementguard_ring_{sel}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════ ABOUT
with tab_about:
    st.markdown("""
#### Four layers, and each one exists because the last one is blind to something

| layer | what it catches | what it misses |
|---|---|---|
| **Rules** — 6 pandas checks | PO splitting, duplicate invoices, round numbers, thin competition | anything relational |
| **Isolation Forest** — unsupervised | unusual transactions, **using no labels at all** | anything relational |
| **XGBoost** — supervised | ranks the queue, cuts false positives | the cartel — see below |
| **Graph** — NetworkX | **ghost vendors two hops away, and cartels** | — |

#### Why the graph is not optional

**A SQL join finds a vendor who shares a bank account with a buyer.** It does not
find the careful one, who puts a shell company in between:

`ghost ──same bank── cutout ──same address── buyer`

That ghost shares nothing with any employee. The join comes back empty. Two hops
is not a harder join — it is a different question.

**And a cartel cannot be learned at all.** There are two rings in this data. Hold
one out to test on, and the model trains on exactly one example of a cartel. No
model learns from one example — and no company on earth has a hundred labelled
cartels to learn from. **The graph needs none.** It found both rings exactly,
ranked first and second, from nothing but who bids with whom and who wins.

#### What this tool does not do

It does not block payments. It does not accuse anyone. It produces a queue and a
reason, and a person decides. That is deliberate: the moment a model can stop a
payment on its own, someone has to own its mistakes.

#### The honest caveat

Every number here is measured on synthetic data. It proves the code works. It
does not yet prove the idea works — that needs real procurement data, and it is
the next thing on the list.
""")
