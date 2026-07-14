"""
Fault injection tests.

The review's point: fallback paths deserve as much testing as the happy
path. These deliberately break things — concurrent writes, malformed
payloads, partial/missing context, a store that raises mid-operation —
and assert the system degrades the way it's documented to, rather than
crashing or silently corrupting state.
"""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app, store as app_store, conversations as app_conversations, suppression as app_suppression, decisions as app_decisions
from app.store import ContextStore, StaleVersionError
from app.opportunities import evaluate_trigger
from app.composer import compose


@pytest.fixture()
def client():
    c = TestClient(app)
    yield c
    app_store.teardown()
    app_conversations.teardown()
    app_suppression.teardown()
    app_decisions.teardown()


def test_concurrent_context_pushes_never_corrupt_the_store():
    store = ContextStore()
    errors = []

    def push(version):
        try:
            store.put("merchant", "m_concurrent", version, {"v": version})
        except StaleVersionError:
            pass
        except Exception as e:  # any other exception is the real failure mode under test
            errors.append(e)

    threads = [threading.Thread(target=push, args=(v,)) for v in range(1, 21)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Whatever the final state is, it must be internally consistent: the
    # stored version must be one of the versions we actually pushed.
    assert store.get_version("merchant", "m_concurrent") in range(1, 21)


def test_concurrent_suppression_marks_are_consistent():
    from app.suppression import SuppressionEngine

    supp = SuppressionEngine()
    key = supp.build_key("m1", "recall_due", "recall:test")
    errors = []

    def fire():
        try:
            supp.mark_fired(key)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=fire) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert supp.already_fired(key)


def test_partial_merchant_context_does_not_crash_composer(dentists_category, triggers):
    # Merchant missing almost everything except merchant_id/identity.
    sparse_merchant = {"merchant_id": "m_sparse", "identity": {"name": "Sparse Clinic"}}
    t = triggers["trg_001_research_digest_dentists"]
    op = evaluate_trigger(t, dentists_category, sparse_merchant, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert op.eligible  # missing fields shouldn't make an otherwise-valid trigger ineligible
    msg = compose(dentists_category, sparse_merchant, t, None, op)
    assert msg.body.strip() != ""


def test_null_fields_throughout_merchant_context_do_not_crash(dentists_category, triggers):
    merchant = {
        "merchant_id": "m_nulls",
        "identity": None,
        "subscription": None,
        "performance": None,
        "offers": None,
        "conversation_history": None,
        "signals": None,
    }
    t = triggers["trg_001_research_digest_dentists"]
    op = evaluate_trigger(t, dentists_category, merchant, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert op.eligible
    msg = compose(dentists_category, merchant, t, None, op)
    assert msg.body.strip() != ""


def test_malformed_trigger_missing_required_fields_does_not_crash(dentists_category, drmeera):
    malformed = {"id": "trg_broken"}  # no kind, no scope, no payload, no urgency
    op = evaluate_trigger(malformed, dentists_category, drmeera, None, "2026-04-26T10:00:00Z", suppressed=False)
    assert op.eligible  # degrades to generic_signal family rather than raising
    msg = compose(dentists_category, drmeera, malformed, None, op)
    assert msg.body.strip() != ""


def test_context_push_with_wrong_payload_type_returns_400_not_500(client):
    r = client.post(
        "/v1/context",
        json={"scope": "merchant", "context_id": "m1", "version": 1, "payload": "not_an_object", "delivered_at": "2026-04-26T10:00:00Z"},
    )
    assert r.status_code == 400
    assert r.status_code != 500


def test_context_push_missing_required_field_returns_400_not_500(client):
    r = client.post("/v1/context", json={"scope": "merchant", "version": 1, "payload": {}})  # no context_id
    assert r.status_code == 400
    assert r.status_code != 500


def test_tick_with_malformed_body_returns_422_not_500(client):
    r = client.post("/v1/tick", json={"available_triggers": "not_a_list"})
    assert r.status_code in (400, 422)
    assert r.status_code != 500


def test_reply_missing_optional_fields_still_works(client):
    r = client.post("/v1/reply", json={"conversation_id": "c1", "message": "hello", "turn_number": 1})
    assert r.status_code == 200


def test_double_teardown_is_idempotent_and_safe(client):
    r1 = client.post("/v1/teardown")
    r2 = client.post("/v1/teardown")
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_storage_failure_during_context_push_does_not_corrupt_state(monkeypatch):
    """Simulate a storage failure mid-write and confirm the endpoint
    degrades to a 500 with a structured body rather than leaving the
    store in a half-written state or crashing the process."""
    from app import main as main_module

    def broken_put(*args, **kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(main_module.store, "put", broken_put)
    # raise_server_exceptions=False so the app's own exception handler
    # (which real deployments always hit) runs instead of TestClient
    # re-raising for local debugging.
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post(
        "/v1/context",
        json={"scope": "merchant", "context_id": "m_fail", "version": 1, "payload": {}, "delivered_at": "2026-04-26T10:00:00Z"},
    )
    assert r.status_code == 500
    body = r.json()
    assert "error" in body
    # subsequent, unrelated requests still work — the failure didn't take the process down
    monkeypatch.undo()
    r2 = c.get("/v1/healthz")
    assert r2.status_code == 200
    app_store.teardown()
