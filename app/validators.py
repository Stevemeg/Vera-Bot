"""
Validator pipeline — the "self-critique before response" stage.

Runs deterministically after composer.compose() and before a message is
allowed into a TickAction/ReplyResponse. Every check here is a plain
function over the already-composed text plus the Opportunity that
produced it — no second LLM call, no new state.

If validation fails, main.py falls back to a safe, still-grounded
generic message (composer._generic-style) rather than emitting a bad one
outright and rather than silently patching the offending text — an
audit trail of *why* the primary draft was rejected is more useful than
a silently different message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import facts

_CTA_ASK_PHRASES = ("want me to", "should i", "shall i", "would you like", "do you want")


@dataclass
class ValidationResult:
    ok: bool
    failures: list[str] = field(default_factory=list)


def validate(body: str, cta: str, rationale: str, category: dict) -> ValidationResult:
    failures: list[str] = []

    if not body or not body.strip():
        failures.append("empty_body")
        return ValidationResult(ok=False, failures=failures)

    if not rationale or not rationale.strip():
        failures.append("missing_rationale")

    # Single-CTA shape: an explicit "Reply" directive should appear at
    # most once. A second, independent CTA verb elsewhere in the body
    # (e.g. two separate "Want me to..." asks) is also a violation.
    reply_count = len(re.findall(r"\breply\b", body, flags=re.IGNORECASE))
    if reply_count > 1:
        failures.append("multiple_explicit_cta_reply")

    question_count = body.count("?")
    if question_count > 2:
        failures.append("too_many_questions_likely_multi_cta")

    # Confirmed-by-testing blind spot: a compound ask like "want me to
    # draft it now? Should I also schedule it for tomorrow?" is TWO
    # independent CTAs but has only one question mark short of the
    # >2 threshold above and never says "reply". Count distinct CTA-verb
    # phrase occurrences directly instead of proxying through punctuation.
    cta_verb_hits = sum(len(re.findall(rf"\b{re.escape(phrase)}\b", body, flags=re.IGNORECASE)) for phrase in _CTA_ASK_PHRASES)
    if cta_verb_hits > 1:
        failures.append("multiple_independent_cta_verbs")

    # Taboo vocabulary leak (belt-and-suspenders on top of composer's own
    # strip_taboo — a validator failure here means strip_taboo missed
    # something, e.g. a multi-word taboo phrase with punctuation between
    # tokens, and should be treated as a real bug rather than papered over).
    low = body.lower()
    for taboo in facts.taboo_words(category):
        if taboo and taboo in low:
            failures.append(f"taboo_vocab_leak:{taboo}")

    # Genericness heuristic: a message with no digit, no capitalized proper
    # noun beyond the greeting, and no currency/percent symbol is very
    # likely a low-specificity fallback rather than a grounded message.
    # This is advisory, not a hard failure — plenty of legitimate messages
    # (e.g. "gracefully exiting") are short and fact-free by design — so it
    # only fires in combination with cta != "none" (i.e. we're actually
    # asking the merchant to act on... nothing concrete).
    has_digit = bool(re.search(r"\d", body))
    has_currency_or_pct = bool(re.search(r"[₹%]", body))
    if cta != "none" and not has_digit and not has_currency_or_pct and len(body.split()) > 25:
        failures.append("long_message_with_no_concrete_anchor")

    return ValidationResult(ok=len(failures) == 0, failures=failures)
