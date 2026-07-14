"""
Property-based invariant tests (Hypothesis).

Example-based tests check specific scenarios; these check that certain
properties hold across a wide, randomized space of inputs — the kind of
edge case a hand-written example is unlikely to hit by accident.
"""
from __future__ import annotations

import copy

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.composer import compose
from app.opportunities import evaluate_trigger
from app.validators import validate

URGENCY = st.integers(min_value=1, max_value=5)
KINDS = st.sampled_from(
    [
        "research_digest", "regulation_change", "recall_due", "chronic_refill_due",
        "perf_dip", "perf_spike", "milestone_reached", "competitor_opened",
        "review_theme_emerged", "dormant_with_vera", "renewal_due", "festival_upcoming",
        "totally_novel_future_kind_1", "totally_novel_future_kind_2",
    ]
)
SCOPES = st.sampled_from(["merchant", "customer"])


def _trigger_strategy():
    return st.builds(
        lambda kind, scope, urgency, has_slots, placeholder: {
            "id": f"trg_prop_{abs(hash((kind, scope, urgency, has_slots, placeholder)))}",
            "kind": kind,
            "scope": scope,
            "source": "internal",
            "merchant_id": "m_prop_1",
            "customer_id": "c_prop_1" if scope == "customer" else None,
            "urgency": urgency,
            "suppression_key": f"prop:{kind}:{urgency}",
            "expires_at": "2099-01-01T00:00:00Z",
            "payload": (
                {"placeholder": True, "metric_or_topic": kind}
                if placeholder
                else ({"available_slots": [{"label": "Mon 9am"}]} if has_slots else {})
            ),
        },
        kind=KINDS,
        scope=SCOPES,
        urgency=URGENCY,
        has_slots=st.booleans(),
        placeholder=st.booleans(),
    )


BASE_CATEGORY = {
    "slug": "dentists",
    "voice": {"vocab_taboo": ["guaranteed", "miracle"], "salutation_examples": ["Dr. {first_name}"]},
    "offer_catalog": [],
    "peer_stats": {"avg_ctr": 0.03},
    "digest": [],
    "seasonal_beats": [],
    "patient_content_library": [],
}

BASE_MERCHANT = {
    "merchant_id": "m_prop_1",
    "category_slug": "dentists",
    "identity": {"name": "Prop Test Clinic", "owner_first_name": "Prop", "languages": ["en"]},
    "subscription": {"status": "active", "plan": "Pro", "days_remaining": 30},
    "performance": {"ctr": 0.02, "delta_7d": {"views_pct": 0.1, "calls_pct": -0.1}},
    "offers": [{"id": "o_1", "title": "Cleaning @ ₹299", "status": "active", "value": "299"}],
    "conversation_history": [],
    "signals": [],
}

BASE_CUSTOMER = {
    "customer_id": "c_prop_1",
    "merchant_id": "m_prop_1",
    "identity": {"name": "PropCustomer", "language_pref": "en"},
    "relationship": {"last_visit": "2026-01-01"},
    "state": "active",
    "preferences": {},
    "consent": {"opted_in_at": "2026-01-01", "scope": ["recall_reminders", "appointment_reminders"]},
}


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(trigger=_trigger_strategy())
def test_evaluate_and_compose_never_crash_and_stay_grounded(trigger):
    merchant = copy.deepcopy(BASE_MERCHANT)
    category = copy.deepcopy(BASE_CATEGORY)
    customer = copy.deepcopy(BASE_CUSTOMER) if trigger["scope"] == "customer" else None

    op = evaluate_trigger(trigger, category, merchant, customer, "2026-04-26T10:00:00Z", suppressed=False)
    if not op.eligible:
        return  # ineligibility is a valid, tested-elsewhere outcome

    msg = compose(category, merchant, trigger, customer, op)

    # Invariant: at most one explicit "Reply" CTA verb.
    assert msg.body.lower().count("reply") <= 1

    # Invariant: no taboo vocabulary leak regardless of which random kind fired.
    low = msg.body.lower()
    for taboo in category["voice"]["vocab_taboo"]:
        assert taboo not in low

    # Invariant: send_as always matches trigger scope.
    expected_send_as = "merchant_on_behalf" if trigger["scope"] == "customer" else "vera"
    assert msg.send_as == expected_send_as

    # Invariant: rationale is never empty.
    assert msg.rationale.strip() != ""

    # Invariant: body is never empty.
    assert msg.body.strip() != ""


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(trigger=_trigger_strategy())
def test_identical_inputs_always_produce_identical_outputs(trigger):
    merchant = copy.deepcopy(BASE_MERCHANT)
    category = copy.deepcopy(BASE_CATEGORY)
    customer = copy.deepcopy(BASE_CUSTOMER) if trigger["scope"] == "customer" else None

    op_a = evaluate_trigger(trigger, category, merchant, customer, "2026-04-26T10:00:00Z", suppressed=False)
    op_b = evaluate_trigger(copy.deepcopy(trigger), copy.deepcopy(category), copy.deepcopy(merchant), copy.deepcopy(customer), "2026-04-26T10:00:00Z", suppressed=False)
    assert op_a.score == op_b.score
    assert op_a.eligible == op_b.eligible

    if not op_a.eligible:
        return
    msg_a = compose(category, merchant, trigger, customer, op_a)
    msg_b = compose(category, merchant, trigger, customer, op_b)
    assert msg_a.body == msg_b.body
    assert msg_a.cta == msg_b.cta
    assert msg_a.send_as == msg_b.send_as


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(trigger=_trigger_strategy())
def test_scores_are_finite_and_within_sane_bounds(trigger):
    merchant = copy.deepcopy(BASE_MERCHANT)
    category = copy.deepcopy(BASE_CATEGORY)
    customer = copy.deepcopy(BASE_CUSTOMER) if trigger["scope"] == "customer" else None
    op = evaluate_trigger(trigger, category, merchant, customer, "2026-04-26T10:00:00Z", suppressed=False)
    assert -50 <= op.score <= 250  # generous bound; catches runaway/NaN-style bugs, not a tuning assertion


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(trigger=_trigger_strategy())
def test_suppression_key_inputs_are_stable_strings(trigger):
    # The suppression key builder is pure string formatting — verify it
    # never raises and always returns a non-empty deterministic string for
    # any trigger shape the strategy can produce.
    from app.suppression import SuppressionEngine

    supp = SuppressionEngine()
    k1 = supp.build_key(trigger["merchant_id"], trigger["kind"], trigger["suppression_key"], trigger.get("customer_id"))
    k2 = supp.build_key(trigger["merchant_id"], trigger["kind"], trigger["suppression_key"], trigger.get("customer_id"))
    assert k1 == k2
    assert isinstance(k1, str) and k1 != ""


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(trigger=_trigger_strategy())
def test_validator_never_raises_on_composer_output(trigger):
    merchant = copy.deepcopy(BASE_MERCHANT)
    category = copy.deepcopy(BASE_CATEGORY)
    customer = copy.deepcopy(BASE_CUSTOMER) if trigger["scope"] == "customer" else None
    op = evaluate_trigger(trigger, category, merchant, customer, "2026-04-26T10:00:00Z", suppressed=False)
    if not op.eligible:
        return
    msg = compose(category, merchant, trigger, customer, op)
    result = validate(msg.body, msg.cta, msg.rationale, category)
    assert isinstance(result.ok, bool)
