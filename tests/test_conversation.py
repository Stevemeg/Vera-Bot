from app.conversation import ConversationStore, decide_reply


def _store():
    return ConversationStore()


def test_stop_ends_conversation_immediately():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "please STOP messaging me", 1, "merchant")
    assert result["action"] == "end"
    assert st.opted_out is True


def test_yes_routes_to_action_not_requalification():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "Yes, let's do it", 1, "merchant")
    assert result["action"] == "send"
    assert "?" not in result["body"]  # must not ask another qualifying question


def test_auto_reply_detected_after_three_verbatim_repeats():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    canned = "Thank you for contacting us. Our team will get back to you shortly."
    r1 = decide_reply(st, canned, 1, "merchant")
    r2 = decide_reply(st, canned, 2, "merchant")
    r3 = decide_reply(st, canned, 3, "merchant")
    assert r1["action"] == "send"
    assert r2["action"] == "send"  # probes once
    assert r3["action"] == "end"  # gives up gracefully, no wasted 4th turn
    assert st.status == "ended"


def test_already_done_ends_gracefully():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "already done, thanks", 1, "merchant")
    assert result["action"] == "end"


def test_later_backs_off_with_wait():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "not now, call later", 1, "merchant")
    assert result["action"] == "wait"
    assert result["wait_seconds"] and result["wait_seconds"] > 0


def test_hostile_message_does_not_escalate():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "you are a stupid bot", 1, "merchant")
    assert result["action"] == "send"
    low = result["body"].lower()
    assert "stupid" not in low and "idiot" not in low  # never mirrors abuse


def test_off_topic_question_stays_on_mission():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "can you also help me file my GST returns", 1, "merchant")
    assert result["action"] == "send"


def test_turn_budget_forces_graceful_exit():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = None
    for turn in range(1, 7):
        result = decide_reply(st, f"hmm not sure {turn}", turn, "merchant")
    assert result["action"] == "end"


def test_ended_conversation_never_reopens():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    decide_reply(st, "stop", 1, "merchant")
    result = decide_reply(st, "actually wait, yes let's go", 2, "merchant")
    assert result["action"] == "end"


def test_positive_emoji_reads_as_affirmative():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "👍", 1, "merchant")
    assert result["action"] == "send"


def test_negative_emoji_reads_as_disengagement():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "😡", 1, "merchant")
    assert result["action"] == "end"


def test_retraction_after_commitment_rolls_back_gracefully():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    yes_result = decide_reply(st, "yes let's do it", 1, "merchant")
    assert yes_result["action"] == "send"
    assert st.committed_action is True
    retract_result = decide_reply(st, "actually never mind", 2, "merchant")
    assert retract_result["action"] == "end"
    assert st.status == "ended"


def test_retraction_without_prior_commitment_just_backs_off():
    store = _store()
    st = store.get_or_create("c1", "m1", None)
    result = decide_reply(st, "never mind", 1, "merchant")
    assert result["action"] == "wait"
