import re

from app.composer import compose
from app.opportunities import evaluate_trigger


def _compose_for(category, merchant, trigger, customer=None):
    op = evaluate_trigger(trigger, category, merchant, customer, "2026-04-26T10:00:00Z", suppressed=False)
    assert op.eligible, op.ineligible_reason
    return compose(category, merchant, trigger, customer, op), op


def test_research_digest_message_is_specific_and_grounded(dentists_category, drmeera, triggers):
    t = triggers["trg_001_research_digest_dentists"]
    msg, op = _compose_for(dentists_category, drmeera, t)
    # Specificity: must contain a real, high-value number from the digest
    # item, not an invented one. Under the 320-char budget the quantified
    # finding (38%) is prioritized over the trial-size anchor / source
    # citation, so it must survive; the message must stay grounded and
    # within the limit rather than being truncated.
    assert "38%" in msg.body
    assert "3-month" in msg.body and "6-month" in msg.body  # the grounded comparison
    assert len(msg.body) <= 320
    assert msg.send_as == "vera"
    assert msg.cta in ("open_ended", "binary_yes_stop", "none")


def test_no_composed_body_exceeds_320_chars_across_dataset(all_categories, merchants, triggers, customers):
    from app.composer import compose
    from app.opportunities import evaluate_trigger

    for tid, t in triggers.items():
        m = merchants.get(t.get("merchant_id"))
        if not m:
            continue
        cat = all_categories.get(m.get("category_slug"))
        cust = customers.get(t.get("customer_id")) if t.get("customer_id") else None
        op = evaluate_trigger(t, cat, m, cust, "2026-04-26T10:00:00Z", suppressed=False)
        if not op.eligible:
            continue
        msg = compose(cat or {}, m or {}, t, cust, op)
        assert len(msg.body) <= 320, f"{tid} ({t.get('kind')}) produced {len(msg.body)} chars"


def test_message_never_contains_taboo_vocabulary(dentists_category, drmeera, triggers):
    for t in triggers.values():
        if t.get("merchant_id") != drmeera["merchant_id"]:
            continue
        try:
            msg, _ = _compose_for(dentists_category, drmeera, t)
        except AssertionError:
            continue
        low = msg.body.lower()
        for taboo in ("guaranteed", "100% safe", "completely cure", "miracle", "best in city"):
            assert taboo not in low


def test_customer_facing_message_uses_send_as_merchant_on_behalf(dentists_category, drmeera, triggers, priya):
    t = triggers["trg_003_recall_due_priya"]
    msg, op = _compose_for(dentists_category, drmeera, t, priya)
    assert msg.send_as == "merchant_on_behalf"
    assert priya["identity"]["name"] in msg.body
    # real offer price from merchant's own catalog, not invented
    assert "299" in msg.body


def test_customer_language_preference_is_honored_with_hindi_mix(dentists_category, drmeera, triggers, priya):
    t = triggers["trg_003_recall_due_priya"]
    msg, _ = _compose_for(dentists_category, drmeera, t, priya)
    # priya's language_pref is "hi-en mix" in the seed data
    assert any(tok in msg.body for tok in ("Aapka", "yahaan", "Apke", "ya "))


def test_composer_is_deterministic_for_identical_inputs(dentists_category, drmeera, triggers):
    t = triggers["trg_001_research_digest_dentists"]
    msg1, op1 = _compose_for(dentists_category, drmeera, t)
    msg2, op2 = _compose_for(dentists_category, drmeera, t)
    assert msg1.body == msg2.body
    assert msg1.cta == msg2.cta
    assert msg1.send_as == msg2.send_as


def test_single_cta_shape_no_multi_choice_leak(dentists_category, drmeera, triggers, priya):
    for t in triggers.values():
        if t.get("merchant_id") != drmeera["merchant_id"]:
            continue
        cust = priya if t.get("customer_id") == priya["customer_id"] else None
        try:
            msg, _ = _compose_for(dentists_category, drmeera, t, cust)
        except AssertionError:
            continue
        # anti-pattern: multiple explicit CTA verbs like "Reply YES for X, NO for Y, MAYBE for Z"
        assert msg.body.count("Reply") <= 1


def test_unknown_trigger_kind_does_not_crash_and_stays_grounded(dentists_category, drmeera):
    future_trigger = {
        "id": "trg_future_1",
        "scope": "merchant",
        "kind": "brand_new_kind_from_2099",
        "source": "external",
        "merchant_id": drmeera["merchant_id"],
        "customer_id": None,
        "payload": {"metric_or_topic": "something_new"},
        "urgency": 3,
        "suppression_key": "future:1",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    msg, op = _compose_for(dentists_category, drmeera, future_trigger)
    assert op.family == "generic_signal"
    assert drmeera["identity"]["owner_first_name"] in msg.body or drmeera["identity"]["name"] in msg.body
    assert "something new" in msg.body  # only echoes provided payload facts, nothing invented


def test_across_all_five_categories_voice_stays_category_specific(all_categories, merchants, triggers):
    """Never mix strategies — a dentist trigger composed against a salon merchant
    should still only surface facts from the merchant's *own* category context."""
    for merchant_id, m in merchants.items():
        cat = all_categories.get(m.get("category_slug"))
        if cat is None:
            continue
        candidate_triggers = [t for t in triggers.values() if t.get("merchant_id") == merchant_id]
        for t in candidate_triggers[:2]:
            op = evaluate_trigger(t, cat, m, None, "2026-04-26T10:00:00Z", suppressed=False)
            if not op.eligible:
                continue
            msg = compose(cat, m, t, None, op)
            low = msg.body.lower()
            for taboo in [w.lower() for w in cat.get("voice", {}).get("vocab_taboo", [])]:
                assert taboo not in low, f"{merchant_id}: taboo '{taboo}' leaked into message"
