from app.opportunities import decision_hash, evaluate_trigger, explain, fatigue_penalty
from app import facts


def test_explain_marks_winner_and_why_losers_lost(dentists_category, drmeera, triggers):
    t1 = triggers["trg_001_research_digest_dentists"]
    t2 = triggers["trg_002_compliance_dci_radiograph"]
    op1 = evaluate_trigger(t1, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    op2 = evaluate_trigger(t2, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    explanation = explain([op1, op2])
    winner = [e for e in explanation if "why_lost" not in e]
    losers = [e for e in explanation if "why_lost" in e]
    assert len(winner) == 1
    assert winner[0]["trigger_id"] == op2.trigger_id  # compliance outranks knowledge here
    assert losers[0]["why_lost"].startswith("outscored_by_")


def test_explain_reports_ineligible_reason(dentists_category, drmeera, triggers):
    t = dict(triggers["trg_001_research_digest_dentists"])
    t["expires_at"] = "2020-01-01T00:00:00Z"
    op = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    explanation = explain([op])
    assert explanation[0]["why_lost"] == "trigger_expired"
    assert explanation[0]["eligible"] is False


def test_decision_hash_is_stable_for_identical_inputs(dentists_category, drmeera, triggers):
    t = triggers["trg_001_research_digest_dentists"]
    op_a = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    op_b = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert decision_hash(1, 1, op_a) == decision_hash(1, 1, op_b)


def test_decision_hash_changes_with_context_version(dentists_category, drmeera, triggers):
    t = triggers["trg_001_research_digest_dentists"]
    op = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert decision_hash(1, 1, op) != decision_hash(2, 1, op)


def test_fatigue_penalty_zero_with_no_history():
    penalty, reason = fatigue_penalty({"conversation_history": []})
    assert penalty == 0.0


def test_fatigue_penalty_positive_for_repeated_ignores():
    history = [
        {"from": "vera", "body": "a", "engagement": "ignored"},
        {"from": "vera", "body": "b", "engagement": "ignored"},
        {"from": "vera", "body": "c", "engagement": "ignored"},
    ]
    penalty, reason = fatigue_penalty({"conversation_history": history})
    assert penalty > 0
    assert "trailing_ignored=3" in reason


def test_fatigue_bonus_for_engaged_merchant():
    history = [
        {"from": "vera", "body": "a", "engagement": "merchant_replied"},
        {"from": "merchant", "body": "ok", "engagement": None},
    ]
    penalty, reason = fatigue_penalty({"conversation_history": history})
    assert penalty < 0  # bonus, not penalty


def test_trigger_precedence_excludes_offer_flagged_expired_by_trigger(drmeera):
    merchant = dict(drmeera)
    merchant["offers"] = [{"id": "o_meera_001", "title": "Dental Cleaning @ ₹299", "status": "active"}]
    trigger = {"kind": "some_offer_expired", "payload": {"offer_id": "o_meera_001"}}
    offers = facts.active_offers(merchant, trigger)
    assert offers == []


def test_trigger_precedence_leaves_unrelated_offers_alone(drmeera):
    merchant = dict(drmeera)
    merchant["offers"] = [{"id": "o_meera_001", "title": "Dental Cleaning @ ₹299", "status": "active"}]
    trigger = {"kind": "recall_due", "payload": {}}
    offers = facts.active_offers(merchant, trigger)
    assert len(offers) == 1


def test_generic_category_fallback_used_for_unseen_vertical():
    fallback = facts.with_category_fallback(None)
    assert fallback["slug"] == "_generic_fallback"
    assert "guaranteed" in fallback["voice"]["vocab_taboo"]


def test_real_category_never_replaced_by_fallback(dentists_category):
    assert facts.with_category_fallback(dentists_category) is dentists_category


def test_schema_tolerant_ctr_alias(drmeera, dentists_category):
    m = dict(drmeera)
    m["performance"] = {"engagement": 0.045}  # renamed field, no "ctr" key at all
    assert facts.merchant_ctr(m) == 0.045
