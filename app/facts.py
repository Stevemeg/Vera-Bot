"""
Grounded-fact helpers.

Everything the composer says must trace back to a value that actually
exists in one of the four pushed contexts. This module is the *only*
place allowed to reach into raw payload dicts, so "never hallucinate" is
enforced structurally: if a fact isn't returned by one of these getters,
the composer has no way to say it.

All getters are defensive against missing/null/malformed data (the
"EDGE CASES" list in challenge-brief.md: missing merchant, null fields,
partial context, schema evolution, etc.) — they return None rather than
raising, and callers must treat None as "don't mention this".
"""
from __future__ import annotations

from typing import Any, Optional


def dig(d: Optional[dict], *path: str, default: Any = None) -> Any:
    """Safe nested get: dig(merchant, 'identity', 'name').""" 
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def dig_any(d: Optional[dict], *paths: tuple, default: Any = None) -> Any:
    """Schema-tolerant get: tries several possible field paths in order and
    returns the first that resolves. Use this for fields that might be
    renamed across schema versions, e.g.:

        dig_any(merchant, ("performance", "ctr"), ("performance", "engagement"))

    so that a future rename (challenge-brief.md "future schema fields") of
    `performance.ctr` -> `performance.engagement` degrades gracefully
    instead of silently returning None everywhere."""
    for path in paths:
        val = dig(d, *path)
        if val is not None:
            return val
    return default


# Generic fallback CategoryContext for verticals the judge injects that we
# were never pushed a matching category for (e.g. "opticians", "pet_clinics").
# Deliberately conservative: peer/professional tone, broad taboo list, no
# vertical-specific vocabulary claimed. Used only when no real category
# context is available — never overrides a real one.
GENERIC_CATEGORY_FALLBACK: dict = {
    "slug": "_generic_fallback",
    "display_name": "General Business",
    "voice": {
        "tone": "peer_professional",
        "register": "respectful_collegial",
        "code_mix": "hindi_english_natural",
        "vocab_allowed": [],
        "vocab_taboo": ["guaranteed", "100% safe", "completely cure", "miracle", "best in city"],
        "salutation_examples": [],
    },
    "offer_catalog": [],
    "peer_stats": {},
    "digest": [],
    "patient_content_library": [],
    "seasonal_beats": [],
    "trend_signals": [],
}


def with_category_fallback(category: Optional[dict]) -> dict:
    """Never return None to the composer — an unseen category still gets a
    safe, conservative voice instead of crashing or defaulting to no
    guardrails at all."""
    return category if isinstance(category, dict) and category.get("voice") else GENERIC_CATEGORY_FALLBACK


def first_name(merchant: Optional[dict]) -> Optional[str]:
    return dig(merchant, "identity", "owner_first_name") or dig(merchant, "identity", "name")


def merchant_name(merchant: Optional[dict]) -> str:
    return dig(merchant, "identity", "name", default="there")


def locality(merchant: Optional[dict]) -> Optional[str]:
    return dig(merchant, "identity", "locality")


def languages(merchant: Optional[dict]) -> list[str]:
    langs = dig(merchant, "identity", "languages", default=[])
    return langs if isinstance(langs, list) else []


def prefers_hindi_mix(merchant: Optional[dict], customer: Optional[dict] = None) -> bool:
    if customer is not None:
        pref = (dig(customer, "identity", "language_pref") or "").lower()
        if pref:
            return "hi" in pref
    langs = languages(merchant)
    return "hi" in [str(l).lower() for l in langs]


def active_offers(merchant: Optional[dict], trigger: Optional[dict] = None) -> list[dict]:
    offers = dig(merchant, "offers", default=[]) or []
    active = [o for o in offers if isinstance(o, dict) and o.get("status") == "active"]
    if trigger:
        # Trigger > merchant precedence: if the trigger itself flags an
        # offer as expired/ended/discontinued (fresher signal than a
        # possibly-stale merchant.offers snapshot), don't use it even if
        # the merchant context still lists it as active.
        payload = trigger.get("payload") or {}
        blocked_ids = set()
        for key in ("expired_offer_id", "discontinued_offer_id", "ended_offer_id"):
            v = payload.get(key)
            if v:
                blocked_ids.add(v)
        if str(trigger.get("kind", "")).endswith("offer_expired") and payload.get("offer_id"):
            blocked_ids.add(payload["offer_id"])
        if blocked_ids:
            active = [o for o in active if o.get("id") not in blocked_ids]
    return active


def signals(merchant: Optional[dict]) -> list[str]:
    sig = dig(merchant, "signals", default=[]) or []
    return [str(s) for s in sig]


def has_signal(merchant: Optional[dict], prefix: str) -> bool:
    return any(str(s).startswith(prefix) for s in signals(merchant))


def digest_item(category: Optional[dict], item_id: Optional[str]) -> Optional[dict]:
    if not item_id:
        return None
    for item in dig(category, "digest", default=[]) or []:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    return None


def peer_stat(category: Optional[dict], key: str) -> Optional[Any]:
    return dig(category, "peer_stats", key)


def merchant_ctr(merchant: Optional[dict]) -> Optional[float]:
    """Schema-tolerant CTR lookup: some future payload versions may rename
    performance.ctr -> performance.engagement (challenge-brief.md's
    'future schema fields' edge case)."""
    return dig_any(merchant, ("performance", "ctr"), ("performance", "engagement"))


def peer_ctr(category: Optional[dict]) -> Optional[float]:
    return dig_any(category, ("peer_stats", "avg_ctr"), ("peer_stats", "avg_engagement"))


def voice(category: Optional[dict]) -> dict:
    return dig(category, "voice", default={}) or {}


def taboo_words(category: Optional[dict]) -> list[str]:
    return [str(w).lower() for w in dig(category, "voice", "vocab_taboo", default=[]) or []]


def salutation(category: Optional[dict], merchant: Optional[dict]) -> str:
    """Pick a category-appropriate salutation grounded in the merchant's own name."""
    name = first_name(merchant)
    examples = dig(category, "voice", "salutation_examples", default=[]) or []
    if examples and name:
        template = examples[0]
        if "{first_name}" in template:
            return template.replace("{first_name}", str(name))
    return str(name) if name else merchant_name(merchant)


def fmt_pct(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    try:
        return f"{abs(float(x)) * 100:.0f}%"
    except (TypeError, ValueError):
        return None


def fmt_num(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    try:
        n = float(x)
        return f"{n:,.0f}" if n == int(n) else f"{n:,.2f}"
    except (TypeError, ValueError):
        return None


def consent_scope(customer: Optional[dict]) -> list[str]:
    return [str(s) for s in dig(customer, "consent", "scope", default=[]) or []]


def has_reminder_style_consent(customer: Optional[dict]) -> bool:
    """True if any consent scope token looks like an operational reminder
    grant rather than pure marketing. Deliberately vocabulary-tolerant
    across verticals: dentists say 'recall_reminders', pharmacies say
    'refill_reminders' / 'recall_alerts', etc — schema evolution across
    category-specific consent taxonomies shouldn't break this check."""
    tokens = ("remind", "recall", "refill", "followup", "follow_up", "alert")
    return any(any(tok in s.lower() for tok in tokens) for s in consent_scope(customer))


def strip_taboo(text: str, category: Optional[dict]) -> str:
    """Defensive post-filter: category-fit guardrail against taboo vocabulary
    accidentally introduced by a template default. Removes exact matches,
    case-insensitively, leaving surrounding punctuation clean."""
    out = text
    for w in taboo_words(category):
        if not w:
            continue
        for candidate in (w, w.capitalize(), w.upper()):
            out = out.replace(candidate, "")
    # collapse doubled whitespace left behind by removals
    return " ".join(out.split())
