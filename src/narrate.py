"""
Narration — Day 8.

Day 7 already produces true sentences. This turns the bullet list into a short
paragraph an auditor can paste into an email, and names the scheme the pattern
resembles — the one thing SHAP cannot do, because SHAP sees features, not
stories.

Two engines, and the choice between them is invisible to the user:

  TEMPLATE (always available)
      Deterministic. No API, no key, no cost, and — this is the point — no way
      to hallucinate, because it only ever concatenates sentences that were
      already computed and already guarded in explain.py.

  CLAUDE (used only if a key is present)
      Better prose. But on a leash so short it cannot lie: it is handed the
      SHAP reasons as the ONLY facts it may use, and told in the strongest
      terms to invent nothing. If it has no key, no network, or errors for any
      reason, the code silently falls back to the template. The demo never
      breaks.

Why the leash matters more here than anywhere else in the codebase: this is a
tool that points a finger at people. One fabricated sentence — "this vendor has
been flagged before", when no such record exists — and every true sentence
next to it becomes suspect. We already shipped one fabricated identifier in a
document. We are not going to let a model do it live, to an auditor, about a
named employee.
"""

import os
from collections import Counter

from . import config as C


# ═══════════════════════════════════════════════════════ which scheme is this?
# SHAP names features. A human names schemes. This map is the bridge: which
# features, when they fire together, look like which known fraud. It is a hint
# for the auditor, phrased as a resemblance ("consistent with…"), never as a
# verdict — the tool does not get to conclude, it gets to point.
# Two weights per scheme:
#   "strong"  — features that point at THIS scheme and few others. A hit here is
#               worth 2. A brand-new vendor reachable only through a shell is a
#               ghost; nothing else looks like that.
#   "weak"    — features the scheme shares with its neighbours (buyer_exclusivity
#               shows up for both ghosts and colluders). Worth 1, tie-breakers
#               only, never enough to name a scheme on their own.
# This is what stops a ghost being mislabelled "collusion" just because both
# involve one dominant buyer.
SCHEME_SIGNATURES = {
    "a ghost-vendor pattern": {
        "strong": {"g_hops_to_buyer", "g_reachable_within_2_hops",
                   "vendor_age_days", "days_reg_to_first_award",
                   "is_vendors_first_award", "rule_new_vendor_large_first",
                   "vendor_n_buyers"},
        "weak": {"g_direct_shared_ids", "buyer_exclusivity"},
    },
    "employee–vendor collusion": {
        "strong": {"pair_n_tenders", "pair_share_of_buyer",
                   "rule_vendor_concentration"},
        "weak": {"buyer_exclusivity", "g_direct_shared_ids", "n_bidders"},
    },
    "PO splitting": {
        "strong": {"rule_po_splitting", "headroom_to_limit"},
        "weak": {"amount_over_limit", "pair_n_tenders"},
    },
    "a bid-rigging ring": {
        "strong": {"g_ring_score", "g_bidders_in_ring", "g_ring_win_evenness",
                   "g_cobid_min_affinity", "g_jaccard_mean"},
        "weak": {"g_cobid_mean_affinity"},
    },
    "duplicate billing": {
        "strong": {"rule_duplicate_invoice"},
        "weak": {"days_to_payment"},
    },
    "a fabricated (round-number) amount": {
        "strong": {"is_round", "rule_round_near_threshold"},
        "weak": {"amount_over_limit"},
    },
}


def _resembles(features_fired: set) -> str | None:
    """Best-matching scheme, weighting distinguishing features over shared ones.
    Returns None unless a scheme is backed by at least one STRONG signal — a
    resemblance built only from shared features is not specific enough to name,
    and naming the wrong scheme is its own small hallucination."""
    best, score = None, 0
    for scheme, sig in SCHEME_SIGNATURES.items():
        strong = len(features_fired & sig["strong"])
        weak = len(features_fired & sig["weak"])
        total = 2 * strong + weak
        if strong >= 1 and total > score:
            best, score = scheme, total
    return best if score >= 2 else None


# ═══════════════════════════════════════════════════════════ the template engine
def narrate_template(tender_id, amount, reasons: list) -> str:
    """
    Deterministic prose from the guarded SHAP reasons. Cannot hallucinate — it
    has nothing to hallucinate WITH. Every clause below traces to a reason that
    explain.py already checked against the data.
    """
    if not reasons:
        return (f"Tender {tender_id} (${amount:,.0f}) sits in the queue on the "
                f"weight of many small signals, none decisive on its own. Treat "
                f"it as a weak lead.")

    fired = {r["feature"] for r in reasons}
    scheme = _resembles(fired)
    top = reasons[:3]

    lead = f"Tender {tender_id} (${amount:,.0f}) was flagged"
    if scheme:
        lead += f", and the combination is consistent with {scheme}."
    else:
        lead += "."

    body = " ".join(r["text"] for r in top)

    tail = ""
    if len(reasons) > 3:
        tail = f" A further {len(reasons) - 3} weaker signal" \
               f"{'s' if len(reasons) - 3 > 1 else ''} point the same way."

    close = " Recommend a manual review before any further action."
    return f"{lead} {body}{tail}{close}"


# ═══════════════════════════════════════════════════════════════ the Claude engine
_SYSTEM = """You are a fraud-analysis assistant writing one short paragraph for a \
procurement auditor. You will be given a list of findings about a single tender. \

ABSOLUTE RULES — breaking any of them makes the output worse than useless:
- Use ONLY the findings provided. Invent NOTHING.
- Do NOT add facts, history, amounts, dates, names, or prior incidents that are \
not in the findings. If it is not in the list, it does not exist.
- Do NOT state a conclusion of guilt. These are leads for a human to review, not \
verdicts. Phrase patterns as "consistent with" or "worth reviewing", never as \
"this is fraud".
- 3–4 sentences. Plain English. No jargon, no feature names, no numbers the \
findings did not give you.
- End by recommending a manual review.

You are writing about a named employee and vendor. A single invented sentence \
could defame a real person. When in doubt, say less."""


def narrate_claude(tender_id, amount, reasons: list):
    """Returns a string on success, or None on any failure (→ caller falls back)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not reasons:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)

        findings = "\n".join(f"- {r['label']}: {r['text']}" for r in reasons)
        fired = {r["feature"] for r in reasons}
        scheme = _resembles(fired)
        hint = f"\n\nThe feature pattern resembles {scheme}, but say so only as a " \
               f"possibility." if scheme else ""

        msg = client.messages.create(
            model=C.NARRATION_MODEL,
            max_tokens=300,
            system=_SYSTEM,
            messages=[{"role": "user", "content":
                       f"Tender {tender_id}, ${amount:,.0f}. Findings:\n"
                       f"{findings}{hint}"}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        return text or None
    except Exception:
        # No key, no network, rate-limited, API change — does not matter.
        # The template is right there. The demo does not break.
        return None


# ══════════════════════════════════════════════════════════════════════ public
def narrate(tender_id, amount, reasons: list, prefer_llm: bool = True) -> tuple:
    """
    Returns (text, engine). engine is "claude" or "template" — surfaced in the
    UI so it is always honest about which one wrote the words.
    """
    if prefer_llm:
        text = narrate_claude(tender_id, amount, reasons)
        if text:
            return text, "claude"
    return narrate_template(tender_id, amount, reasons), "template"
