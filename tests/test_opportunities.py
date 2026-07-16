from app.opportunities import evaluate_trigger, best_eligible, rank


def test_missing_merchant_is_ineligible(dentists_category, triggers):
    t = triggers["trg_001_research_digest_dentists"]
    op = evaluate_trigger(t, dentists_category, None, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert not op.eligible
    assert op.ineligible_reason == "missing_merchant_context"


def test_missing_category_is_ineligible(triggers, drmeera):
    t = triggers["trg_001_research_digest_dentists"]
    op = evaluate_trigger(t, None, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert not op.eligible
    assert op.ineligible_reason == "missing_category_context"


def test_expired_trigger_is_ineligible(dentists_category, drmeera, triggers):
    t = dict(triggers["trg_001_research_digest_dentists"])
    t["expires_at"] = "2020-01-01T00:00:00Z"
    op = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert not op.eligible
    assert op.ineligible_reason == "trigger_expired"


def test_suppressed_trigger_is_ineligible(dentists_category, drmeera, triggers):
    t = triggers["trg_001_research_digest_dentists"]
    op = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=True)
    assert not op.eligible
    assert op.ineligible_reason == "suppressed_duplicate"


def test_customer_scoped_trigger_requires_customer_context(dentists_category, drmeera, triggers):
    t = triggers["trg_003_recall_due_priya"]
    op = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert not op.eligible
    assert op.ineligible_reason == "missing_customer_context_for_customer_scoped_trigger"


def test_customer_scoped_trigger_respects_consent_scope(dentists_category, drmeera, triggers, priya):
    t = triggers["trg_003_recall_due_priya"]
    stripped_customer = dict(priya)
    stripped_customer["consent"] = {"opted_in_at": "2025-11-04", "scope": []}
    op = evaluate_trigger(t, dentists_category, drmeera, stripped_customer, "2026-04-26T10:00:00Z", suppressed=False)
    assert not op.eligible
    assert op.ineligible_reason == "customer_consent_scope_missing"


def test_eligible_customer_scoped_trigger_scores_and_is_customer_recall_family(dentists_category, drmeera, triggers, priya):
    t = triggers["trg_003_recall_due_priya"]
    op = evaluate_trigger(t, dentists_category, drmeera, priya, "2026-04-26T10:00:00Z", suppressed=False)
    assert op.eligible
    assert op.family == "customer_recall"
    assert op.score > 0


def test_unknown_future_trigger_kind_falls_back_to_generic_family(dentists_category, drmeera):
    future_trigger = {
        "id": "trg_future_999",
        "scope": "merchant",
        "kind": "quantum_offer_singularity",  # deliberately unseen kind, no inferable keyword
        "source": "external",
        "merchant_id": drmeera["merchant_id"],
        "customer_id": None,
        "payload": {"metric_or_topic": "quantum_offer"},
        "urgency": 3,
        "suppression_key": "future:test",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    op = evaluate_trigger(future_trigger, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert op.eligible
    assert op.family == "generic_signal"


def test_unseen_kind_with_inferable_keyword_reaches_specific_family():
    from app.opportunities import _family_for
    # Future kinds the judge might inject should route by name, not dump to generic.
    assert _family_for("vaccine_recall_due") == "customer_recall"
    assert _family_for("gst_regulation_update") == "compliance"
    assert _family_for("new_competitor_nearby") == "competitive"
    assert _family_for("membership_renewal_soon") == "subscription"
    assert _family_for("google_review_dip") == "reputation"  # review beats dip
    assert _family_for("something_with_no_known_keyword") == "generic_signal"


def test_ranking_is_deterministic_for_identical_inputs(dentists_category, drmeera, triggers):
    t1 = triggers["trg_001_research_digest_dentists"]
    t2 = dict(triggers["trg_002_compliance_dci_radiograph"])
    op1a = evaluate_trigger(t1, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    op1b = evaluate_trigger(t1, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    op2 = evaluate_trigger(t2, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert op1a.score == op1b.score  # same inputs -> same score, always
    ranked = rank([op1a, op2])
    ranked_again = rank([op1b, op2])
    assert [o.trigger_id for o in ranked] == [o.trigger_id for o in ranked_again]


def test_compliance_outranks_pure_knowledge_when_urgency_similar(dentists_category, drmeera, triggers):
    research = evaluate_trigger(triggers["trg_001_research_digest_dentists"], dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    compliance = evaluate_trigger(triggers["trg_002_compliance_dci_radiograph"], dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    best = best_eligible([research, compliance])
    assert best.family == "compliance"


def test_best_eligible_returns_none_when_all_ineligible(dentists_category, drmeera, triggers):
    t = dict(triggers["trg_001_research_digest_dentists"])
    t["expires_at"] = "2020-01-01T00:00:00Z"
    op = evaluate_trigger(t, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert best_eligible([op]) is None
