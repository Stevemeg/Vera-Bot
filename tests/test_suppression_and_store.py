import pytest

from app.store import ContextStore, StaleVersionError
from app.suppression import SuppressionEngine


def test_store_put_get_roundtrip():
    store = ContextStore()
    store.put("merchant", "m1", 1, {"a": 1})
    assert store.get("merchant", "m1") == {"a": 1}


def test_store_rejects_stale_version():
    store = ContextStore()
    store.put("merchant", "m1", 3, {"v": 3})
    with pytest.raises(StaleVersionError) as exc:
        store.put("merchant", "m1", 3, {"v": 3})  # exact replay
    assert exc.value.current_version == 3
    with pytest.raises(StaleVersionError):
        store.put("merchant", "m1", 2, {"v": 2})  # lower version


def test_store_accepts_higher_version_atomically():
    store = ContextStore()
    store.put("merchant", "m1", 1, {"v": 1})
    store.put("merchant", "m1", 2, {"v": 2})
    assert store.get("merchant", "m1") == {"v": 2}
    assert store.get_version("merchant", "m1") == 2


def test_store_counts_by_scope():
    store = ContextStore()
    store.put("merchant", "m1", 1, {})
    store.put("merchant", "m2", 1, {})
    store.put("category", "dentists", 1, {})
    counts = store.counts()
    assert counts["merchant"] == 2
    assert counts["category"] == 1
    assert counts["customer"] == 0


def test_store_teardown_wipes_everything():
    store = ContextStore()
    store.put("merchant", "m1", 1, {})
    store.teardown()
    assert store.get("merchant", "m1") is None
    assert store.counts()["merchant"] == 0


def test_suppression_dedup_key_prevents_resend():
    supp = SuppressionEngine()
    key = supp.build_key("m1", "research_digest", "research:dentists:2026-W17")
    assert not supp.already_fired(key)
    supp.mark_fired(key)
    assert supp.already_fired(key)


def test_suppression_expires():
    supp = SuppressionEngine()
    key = supp.build_key("m1", "recall_due", "recall:c1:6mo")
    supp.mark_fired(key, expires_at_iso="2020-01-01T00:00:00Z")  # already expired
    assert not supp.already_fired(key)


def test_suppression_different_customers_do_not_cross_suppress():
    supp = SuppressionEngine()
    k1 = supp.build_key("m1", "recall_due", "recall:shared_key", customer_id="c1")
    k2 = supp.build_key("m1", "recall_due", "recall:shared_key", customer_id="c2")
    assert k1 != k2
    supp.mark_fired(k1)
    assert supp.already_fired(k1)
    assert not supp.already_fired(k2)


def test_anti_repetition_blocks_identical_body_in_same_conversation():
    supp = SuppressionEngine()
    assert not supp.is_repeat_body("conv1", "hello")
    supp.record_body("conv1", "hello")
    assert supp.is_repeat_body("conv1", "hello")
    assert not supp.is_repeat_body("conv2", "hello")  # scoped per conversation
