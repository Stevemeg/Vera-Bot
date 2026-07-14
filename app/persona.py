"""
Merchant persona classification.

Not ML — a small set of deterministic rules over already-grounded facts
(subscription, performance, offers, conversation history). The persona
label is used in two places:

  1. opportunities.py — as a scoring modifier (a `discount_heavy` merchant
     gets an offer/recall/winback boost; a `dormant`/`inactive` merchant
     gets a reengagement boost; a `growth_focused` merchant with strong
     recent performance gets a knowledge/performance_positive boost).
  2. composer.py — as a light framing hook (e.g. leading with footfall
     framing for a growth-focused restaurant vs. retention framing for a
     gym), without inventing any new facts — the persona only chooses
     *which already-true fact to lead with*, never a new one.

Every merchant can get more than one tag; `primary(merchant)` picks the
single most decision-relevant one deterministically.
"""
from __future__ import annotations

from . import facts

PERSONAS = (
    "new",
    "established",
    "growth_focused",
    "inactive",
    "discount_heavy",
    "premium",
    "price_sensitive",
    "busy",
)


def tags(merchant: dict) -> set[str]:
    out: set[str] = set()
    if merchant is None:
        return out

    established_year = facts.dig(merchant, "identity", "established_year")
    if isinstance(established_year, int) and established_year >= 2025:
        out.add("new")
    else:
        out.add("established")

    perf = facts.dig(merchant, "performance", default={}) or {}
    views_pct = facts.dig(perf, "delta_7d", "views_pct")
    calls_pct = facts.dig(perf, "delta_7d", "calls_pct")
    if (views_pct is not None and views_pct > 0.1) or (calls_pct is not None and calls_pct > 0.1):
        out.add("growth_focused")

    if facts.has_signal(merchant, "dormant") or facts.has_signal(merchant, "stale_posts"):
        out.add("inactive")

    offers = facts.dig(merchant, "offers", default=[]) or []
    active = [o for o in offers if isinstance(o, dict) and o.get("status") == "active"]
    cheap_offers = [o for o in active if _offer_value(o) is not None and _offer_value(o) < 500]
    if len(cheap_offers) >= 1 and len(active) <= 2:
        out.add("discount_heavy")
    if any(_offer_value(o) is not None and _offer_value(o) >= 1500 for o in active):
        out.add("premium")

    plan = facts.dig(merchant, "subscription", "plan")
    if isinstance(plan, str) and plan.lower() in ("basic", "free", "lite"):
        out.add("price_sensitive")

    history = facts.dig(merchant, "conversation_history", default=[]) or []
    if len(history) >= 6:
        out.add("busy")

    return out


def _offer_value(offer: dict) -> float | None:
    try:
        return float(str(offer.get("value")).replace(",", ""))
    except (TypeError, ValueError):
        return None


# Deterministic priority order when a merchant has multiple tags — pick the
# single tag most useful for scoring/framing purposes.
_PRIORITY = ["inactive", "discount_heavy", "premium", "growth_focused", "price_sensitive", "new", "busy", "established"]


def primary(merchant: dict) -> str:
    t = tags(merchant)
    for p in _PRIORITY:
        if p in t:
            return p
    return "established"


# Category-level psychology hook (per-vertical primary motivator), used as
# a tie-break framing signal only — never invents a fact, just chooses the
# angle to lead with when more than one true fact is available.
CATEGORY_MOTIVATOR = {
    "restaurants": "footfall",
    "gyms": "retention",
    "dentists": "appointments",
    "salons": "visual_transformation",
    "pharmacies": "availability",
}


def motivator_for(category_slug: str | None) -> str:
    return CATEGORY_MOTIVATOR.get(category_slug or "", "generic_growth")


# Deterministic scoring nudges by (persona, opportunity_family). Kept small
# and explainable — added directly into the score breakdown in
# opportunities.py, never silently baked into an opaque number.
PERSONA_FAMILY_BONUS = {
    ("inactive", "reengagement"): 8,
    ("inactive", "winback"): 6,
    ("discount_heavy", "customer_recall"): 6,
    ("discount_heavy", "winback"): 6,
    ("discount_heavy", "seasonal"): 4,
    ("premium", "knowledge"): 5,
    ("premium", "reputation"): 4,
    ("growth_focused", "performance_positive"): 6,
    ("growth_focused", "knowledge"): 3,
    ("price_sensitive", "subscription"): -4,  # renewal pushes land worse with price-sensitive merchants
    ("busy", "engagement_cadence"): -5,  # busy merchants have less patience for pure curiosity nudges
}


def persona_bonus(persona: str, family: str) -> int:
    return PERSONA_FAMILY_BONUS.get((persona, family), 0)
