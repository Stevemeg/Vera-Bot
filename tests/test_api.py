import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, store as app_store, suppression as app_suppression, conversations as app_conversations


@pytest.fixture()
def client():
    c = TestClient(app)
    yield c
    # isolate tests from each other
    app_store.teardown()
    app_conversations.teardown()
    app_suppression.teardown()


def test_healthz_shape(client):
    r = client.get("/v1/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["contexts_loaded"].keys()) == {"category", "merchant", "customer", "trigger"}


def test_metadata_shape(client):
    r = client.get("/v1/metadata")
    assert r.status_code == 200
    body = r.json()
    for key in ("team_name", "team_members", "model", "approach", "contact_email", "version", "submitted_at"):
        assert key in body


def test_context_push_idempotent_and_versioned(client, dentists_category):
    payload = {"scope": "category", "context_id": "dentists", "version": 1, "payload": dentists_category, "delivered_at": "2026-04-26T10:00:00Z"}
    r1 = client.post("/v1/context", json=payload)
    assert r1.status_code == 200
    assert r1.json()["accepted"] is True

    r2 = client.post("/v1/context", json=payload)  # exact replay
    assert r2.status_code == 409
    assert r2.json()["reason"] == "stale_version"

    payload2 = dict(payload, version=2)
    r3 = client.post("/v1/context", json=payload2)
    assert r3.status_code == 200


def test_context_push_rejects_invalid_scope(client):
    r = client.post("/v1/context", json={"scope": "not_a_scope", "context_id": "x", "version": 1, "payload": {}, "delivered_at": "2026-04-26T10:00:00Z"})
    assert r.status_code == 400
    assert r.json()["reason"] == "invalid_scope"


def test_context_push_handles_malformed_json(client):
    r = client.post("/v1/context", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_full_lifecycle_tick_then_reply(client, dentists_category, drmeera, triggers):
    client.post("/v1/context", json={"scope": "category", "context_id": "dentists", "version": 1, "payload": dentists_category, "delivered_at": "2026-04-26T10:00:00Z"})
    client.post("/v1/context", json={"scope": "merchant", "context_id": drmeera["merchant_id"], "version": 1, "payload": drmeera, "delivered_at": "2026-04-26T10:00:00Z"})
    t = triggers["trg_001_research_digest_dentists"]
    client.post("/v1/context", json={"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t, "delivered_at": "2026-04-26T10:00:00Z"})

    r = client.post("/v1/tick", json={"now": "2026-04-26T10:05:00Z", "available_triggers": [t["id"]]})
    assert r.status_code == 200
    actions = r.json()["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["merchant_id"] == drmeera["merchant_id"]
    assert action["cta"] in ("open_ended", "binary_yes_stop", "none")

    # replaying the same tick should not resend (suppression)
    r2 = client.post("/v1/tick", json={"now": "2026-04-26T10:05:00Z", "available_triggers": [t["id"]]})
    assert r2.json()["actions"] == []

    rr = client.post(
        "/v1/reply",
        json={
            "conversation_id": action["conversation_id"],
            "merchant_id": action["merchant_id"],
            "customer_id": None,
            "from_role": "merchant",
            "message": "Yes, send me the abstract",
            "received_at": "2026-04-26T10:10:00Z",
            "turn_number": 1,
        },
    )
    assert rr.status_code == 200
    assert rr.json()["action"] in ("send", "wait", "end")


def test_tick_with_unknown_trigger_id_is_silently_skipped(client):
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:05:00Z", "available_triggers": ["trg_does_not_exist"]})
    assert r.status_code == 200
    assert r.json()["actions"] == []


def test_tick_with_no_context_at_all_returns_empty_actions_not_error(client):
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:05:00Z", "available_triggers": []})
    assert r.status_code == 200
    assert r.json() == {"actions": []}


def test_reply_with_unseen_conversation_id_creates_new_state(client):
    r = client.post(
        "/v1/reply",
        json={"conversation_id": "brand_new_conv", "merchant_id": "m_unknown", "customer_id": None, "from_role": "merchant", "message": "hello?", "received_at": "2026-04-26T10:10:00Z", "turn_number": 1},
    )
    assert r.status_code == 200


def test_teardown_wipes_state(client, dentists_category):
    client.post("/v1/context", json={"scope": "category", "context_id": "dentists", "version": 1, "payload": dentists_category, "delivered_at": "2026-04-26T10:00:00Z"})
    assert client.get("/v1/healthz").json()["contexts_loaded"]["category"] == 1
    client.post("/v1/teardown")
    assert client.get("/v1/healthz").json()["contexts_loaded"]["category"] == 0
