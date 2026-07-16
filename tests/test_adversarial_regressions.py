"""
Adversarial regression tests.

Each test here reproduces a CONFIRMED exploit found during an adversarial
pass against the running system (not a hypothetical) and asserts it stays
fixed. See the accompanying report for the full attack narrative.
"""
from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app, store as app_store, conversations as app_conversations, suppression as app_suppression, decisions as app_decisions
from app.conversation import ConversationStore, decide_reply
from app.suppression import SuppressionEngine
from app.validators import validate


@pytest.fixture()
def client():
    c = TestClient(app)
    yield c
    app_store.teardown()
    app_conversations.teardown()
    app_suppression.teardown()
    app_decisions.teardown()


# --- Exploit 1: conversation_id reused across two different merchants ---

def test_conversation_id_reused_across_merchants_does_not_splice_state(client):
    r1 = client.post("/v1/reply", json={"conversation_id": "conv_x", "merchant_id": "m_AAA", "customer_id": None,
                                          "from_role": "merchant", "message": "yes let's do it",
                                          "received_at": "2026-04-26T10:00:00Z", "turn_number": 1})
    assert r1.json()["action"] == "send"

    # A totally different merchant reusing the same conversation_id must
    # NOT inherit merchant A's committed_action / turn history.
    r2 = client.post("/v1/reply", json={"conversation_id": "conv_x", "merchant_id": "m_BBB", "customer_id": None,
                                          "from_role": "merchant", "message": "actually never mind",
                                          "received_at": "2026-04-26T10:01:00Z", "turn_number": 1})
    # Without a prior commitment in the (correctly fresh) state for m_BBB,
    # "never mind" is a bare retraction -> backs off, does NOT read as
    # rolling back a commitment that belongs to a different merchant.
    assert r2.json()["action"] == "wait"


def test_conversation_store_isolates_state_at_unit_level():
    store = ConversationStore()
    st_a = store.get_or_create("shared", "m_A", None)
    decide_reply(st_a, "yes", 1, "merchant")
    assert st_a.committed_action is True

    st_b = store.get_or_create("shared", "m_B", None)
    assert st_b is not st_a
    assert st_b.committed_action is False
    assert st_b.turns == []


# --- Exploit 2: turn-budget bypass via caller-controlled turn_number ---

def test_turn_budget_cannot_be_bypassed_by_stuck_turn_number(client):
    last_action = None
    for i in range(10):
        r = client.post("/v1/reply", json={"conversation_id": "conv_stuck", "merchant_id": "m_stuck",
                                             "customer_id": None, "from_role": "merchant",
                                             "message": f"hmm not sure about {i}", "received_at": "2026-04-26T10:00:00Z",
                                             "turn_number": 1})  # attacker/broken-client always sends 1
        last_action = r.json()["action"]
        if last_action == "end":
            break
    assert last_action == "end"


# --- Exploit 3: duplicate /v1/reply retry mistaken for auto-reply ---

def test_identical_retry_does_not_count_toward_auto_reply_detection(client):
    msg = "Can you tell me more about this offer?"
    actions = []
    for _ in range(3):
        r = client.post("/v1/reply", json={"conversation_id": "conv_retry", "merchant_id": "m_retry",
                                             "customer_id": None, "from_role": "merchant", "message": msg,
                                             "received_at": "2026-04-26T10:00:00Z", "turn_number": 1})
        actions.append(r.json()["action"])
    assert actions == ["send", "send", "send"]  # never misread as a 3x-repeat auto-reply


def test_genuine_repeated_body_from_distinct_decisions_is_still_caught(client):
    r1 = client.post("/v1/reply", json={"conversation_id": "conv_genuine_repeat", "merchant_id": "m_gr",
                                          "customer_id": None, "from_role": "merchant", "message": "hmm what do you mean",
                                          "received_at": "2026-04-26T10:00:00Z", "turn_number": 1})
    assert r1.json()["action"] == "send"
    r2 = client.post("/v1/reply", json={"conversation_id": "conv_genuine_repeat", "merchant_id": "m_gr",
                                          "customer_id": None, "from_role": "merchant", "message": "i dont understand you",
                                          "received_at": "2026-04-26T10:05:00Z", "turn_number": 2})
    # Two genuinely different unclassifiable messages that both land on the
    # same generic fallback text must still trip anti-repetition.
    assert r2.json()["action"] == "end"


# --- Exploit 4: suppression expiry used real wall-clock instead of simulated 'now' ---

def test_suppression_expiry_uses_simulated_now_not_real_clock():
    import datetime
    supp = SuppressionEngine()
    key = "k1"
    supp.mark_fired(key, expires_at_iso="2026-04-27T00:00:00Z")  # expires "tomorrow" in a simulated 2026-04-26 timeline
    simulated_now = datetime.datetime(2026, 4, 26, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
    assert supp.already_fired(key, now=simulated_now) is True  # still valid in the simulated timeline


def test_duplicate_send_no_longer_possible_across_ticks_with_updated_context(client):
    cat = json.load(open("../dataset/categories/dentists.json"))
    client.post("/v1/context", json={"scope": "category", "context_id": "dentists", "version": 1, "payload": cat, "delivered_at": "2026-04-26T10:00:00Z"})
    merchants = json.load(open("../dataset/merchants_seed.json"))["merchants"]
    m = dict(merchants[0])
    m["performance"]["delta_7d"]["calls_pct"] = -0.30
    client.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"], "version": 1, "payload": m, "delivered_at": "2026-04-26T10:00:00Z"})

    t = {
        "id": "trg_regression_perfdip", "scope": "merchant", "kind": "perf_dip", "source": "internal",
        "merchant_id": m["merchant_id"], "customer_id": None, "payload": {}, "urgency": 4,
        "suppression_key": "perfdip:regression", "expires_at": "2026-04-27T00:00:00Z",
    }
    client.post("/v1/context", json={"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t, "delivered_at": "2026-04-26T10:00:00Z"})

    r1 = client.post("/v1/tick", json={"now": "2026-04-26T10:05:00Z", "available_triggers": [t["id"]]})
    assert len(r1.json()["actions"]) == 1

    m2 = dict(m)
    m2["performance"]["delta_7d"]["calls_pct"] = -0.55
    client.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"], "version": 2, "payload": m2, "delivered_at": "2026-04-26T10:10:00Z"})
    r2 = client.post("/v1/tick", json={"now": "2026-04-26T10:15:00Z", "available_triggers": [t["id"]]})
    assert len(r2.json()["actions"]) == 0  # correctly suppressed now


# --- Exploit 5: Devanagari-script replies not recognized ---

def test_devanagari_stop_is_recognized(client):
    r = client.post("/v1/reply", json={"conversation_id": "conv_dev1", "merchant_id": "m_dev", "customer_id": None,
                                         "from_role": "merchant", "message": "रोको", "received_at": "2026-04-26T10:00:00Z", "turn_number": 1})
    assert r.json()["action"] == "end"


def test_devanagari_later_is_recognized(client):
    r = client.post("/v1/reply", json={"conversation_id": "conv_dev2", "merchant_id": "m_dev", "customer_id": None,
                                         "from_role": "merchant", "message": "अभी नहीं", "received_at": "2026-04-26T10:00:00Z", "turn_number": 1})
    assert r.json()["action"] == "wait"


def test_devanagari_yes_is_recognized(client):
    r = client.post("/v1/reply", json={"conversation_id": "conv_dev3", "merchant_id": "m_dev", "customer_id": None,
                                         "from_role": "merchant", "message": "हाँ ठीक है", "received_at": "2026-04-26T10:00:00Z", "turn_number": 1})
    assert r.json()["action"] == "send"
    assert r.json()["cta"] == "none"  # routed to action mode, same as a romanized "yes"


# --- Exploit 6: validator blind spot for two-question multi-CTA ---

def test_validator_catches_two_independent_cta_asks_without_the_word_reply(dentists_category):
    body = "Dr. Meera, want me to draft the post now? Should I also schedule it for tomorrow morning?"
    result = validate(body, "open_ended", "test", dentists_category)
    assert not result.ok
    assert "multiple_independent_cta_verbs" in result.failures


def test_validator_does_not_false_positive_on_normal_single_cta(dentists_category):
    body = "Dr. Meera, want me to check your current setup against this?"
    result = validate(body, "binary_yes_stop", "test", dentists_category)
    assert result.ok


def test_validator_does_not_false_positive_on_reply_plus_interested(dentists_category):
    body = "First Month @ ₹499 is available. Reply YES if interested."
    result = validate(body, "binary_yes_stop", "test", dentists_category)
    assert result.ok


# --- Exploit 7 (TOCTOU): suppression check-then-act race — now actually fixed ---

def test_try_reserve_is_atomic_under_concurrent_access():
    import threading
    import time as time_module

    supp = SuppressionEngine()
    key = "toctou_regression"
    wins = []

    def worker():
        if supp.try_reserve(key):
            time_module.sleep(0.01)  # simulate compose()+validate() happening after the reservation
            wins.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1  # exactly one winner, every time, regardless of scheduling


def test_try_reserve_respects_expiry():
    supp = SuppressionEngine()
    key = "expiring_reservation"
    assert supp.try_reserve(key, expires_at_iso="2020-01-01T00:00:00Z") is True  # reserved, but already-expired
    assert supp.try_reserve(key) is True  # expired reservation doesn't block a fresh one


def test_release_unblocks_a_reservation_that_was_never_actually_sent():
    supp = SuppressionEngine()
    key = "released_reservation"
    assert supp.try_reserve(key) is True
    assert supp.try_reserve(key) is False  # still reserved
    supp.release(key)
    assert supp.try_reserve(key) is True  # released, so a fresh reservation succeeds


def test_concurrent_tick_calls_for_the_same_trigger_never_double_send(client):
    import threading

    cat = json.load(open("../dataset/categories/dentists.json"))
    client.post("/v1/context", json={"scope": "category", "context_id": "dentists", "version": 1, "payload": cat, "delivered_at": "2026-04-26T10:00:00Z"})
    merchants = json.load(open("../dataset/merchants_seed.json"))["merchants"]
    m = merchants[0]
    client.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"], "version": 1, "payload": m, "delivered_at": "2026-04-26T10:00:00Z"})
    triggers = json.load(open("../dataset/triggers_seed.json"))["triggers"]
    t = [x for x in triggers if x["id"] == "trg_001_research_digest_dentists"][0]
    client.post("/v1/context", json={"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t, "delivered_at": "2026-04-26T10:00:00Z"})

    totals = []

    def fire():
        r = client.post("/v1/tick", json={"now": "2026-04-26T10:05:00Z", "available_triggers": [t["id"]]})
        totals.append(len(r.json()["actions"]))

    threads = [threading.Thread(target=fire) for _ in range(15)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sum(totals) == 1  # exactly one send across all concurrent calls, never zero, never more than one
