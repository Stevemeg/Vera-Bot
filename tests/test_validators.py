from app.validators import validate
from app.composer import safe_fallback
from app.opportunities import Opportunity


def test_valid_grounded_message_passes(dentists_category):
    result = validate(
        "Dr. Meera, JIDA's Oct issue landed — a 2,100-patient trial found 38% lower recurrence. Want the abstract?",
        "open_ended",
        "grounded in digest item",
        dentists_category,
    )
    assert result.ok


def test_empty_body_fails(dentists_category):
    result = validate("", "open_ended", "some rationale", dentists_category)
    assert not result.ok
    assert "empty_body" in result.failures


def test_missing_rationale_flagged(dentists_category):
    result = validate("Dr. Meera, here's an update. Want details?", "open_ended", "", dentists_category)
    assert not result.ok
    assert "missing_rationale" in result.failures


def test_multiple_reply_ctas_flagged(dentists_category):
    body = "Reply YES for the cleaning offer, or Reply NO if not interested, or Reply LATER for another time."
    result = validate(body, "binary_yes_stop", "test", dentists_category)
    assert not result.ok
    assert "multiple_explicit_cta_reply" in result.failures


def test_taboo_vocabulary_leak_flagged(dentists_category):
    body = "Dr. Meera, this treatment is guaranteed to work for every patient. Want details?"
    result = validate(body, "open_ended", "test", dentists_category)
    assert not result.ok
    assert any(f.startswith("taboo_vocab_leak") for f in result.failures)


def test_long_generic_message_with_action_cta_flagged(dentists_category):
    body = (
        "Dr. Meera, there is an update on your account that I think you should really look at "
        "because it might be quite important for your business going forward this month. "
        "Want me to tell you more about it in detail?"
    )
    result = validate(body, "open_ended", "test", dentists_category)
    assert not result.ok
    assert "long_message_with_no_concrete_anchor" in result.failures


def test_safe_fallback_is_always_valid(dentists_category, drmeera):
    trigger = {"id": "trg_x", "kind": "unknown_kind", "scope": "merchant", "payload": {}}
    opp = Opportunity(trigger_id="trg_x", trigger=trigger, family="generic_signal", score=0.0, reasoning="")
    msg = safe_fallback(dentists_category, drmeera, trigger, opp)
    result = validate(msg.body, msg.cta, msg.rationale, dentists_category)
    assert result.ok
