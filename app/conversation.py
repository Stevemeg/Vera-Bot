"""
Conversation engine for POST /v1/reply.

Handles the full reply taxonomy called out in challenge-brief.md
("yes / no / later / already done / thanks / stop / unsubscribe / emoji
replies / irrelevant questions / merchant intent switching / conversation
recovery / continuation / auto replies / unknown replies / timeout") with
a small deterministic state machine rather than an LLM decision — the
LLM-never-decides-the-action rule applies here too.

State lives per `conversation_id` in ConversationStore (in-memory, wiped
on teardown). Every inbound message is recorded verbatim so auto-reply
detection (same text 3+ times) and anti-repetition checks have a full
history to work from.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger("vera_bot.conversation")

MAX_TURNS = 5
AUTO_REPLY_REPEAT_THRESHOLD = 3  # same verbatim text this many times = auto-reply

_STOP_WORDS = {
    "stop", "unsubscribe", "band karo", "mat bhejo", "no thanks", "not interested",
    "बंद करो", "बंद कर दो", "रोको", "मुझे मत भेजो",  # Devanagari: "stop it" / "don't send me"
}
_YES_WORDS = {
    "yes", "y", "ok", "okay", "go ahead", "let's do it", "lets do it",
    "haan", "ha", "theek hai", "kar do", "chalega", "proceed", "confirm",
    "i want to join", "join karna hai", "mujhe join karna hai", "yes please",
    "sure thing", "for sure",
    "हाँ", "हां", "ठीक है", "कर दो", "चलेगा",  # Devanagari: "yes" / "okay" / "go ahead" / "works for me"
}
# NOTE: bare "sure" is deliberately excluded — "not sure" is uncertainty,
# not affirmation, and a substring/word match on "sure" alone would
# misclassify it as a yes.
_LATER_WORDS = {
    "later", "not now", "abhi nahi", "baad me", "baad mein", "call me later", "busy", "not sure",
    "अभी नहीं", "बाद में",  # Devanagari: "not now" / "later"
}
_DONE_WORDS = {"already done", "already did", "kar liya", "done already", "handled already", "कर लिया"}
_THANKS_WORDS = {"thanks", "thank you", "shukriya", "dhanyavad", "ty", "शुक्रिया", "धन्यवाद"}
_RETRACTION_WORDS = {
    "never mind", "actually no", "actually never mind", "cancel that", "scratch that",
    "wait no", "chodo", "chhodo", "cancel it", "don't bother", "forget it",
}
_EMOJI_ONLY_RE = re.compile(r"^[\W\s]+$", re.UNICODE)
_POSITIVE_EMOJI = {"👍", "🙂", "😊", "✅", "👌"}
_NEGATIVE_EMOJI = {"👎", "🙁", "😡"}


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_any(text: str, phrases: set[str]) -> bool:
    """Word-boundary containment check — deliberately NOT plain substring
    matching, because short tokens like 'y' or 'ok' would otherwise match
    inside unrelated words ('you', 'book'). Multi-word phrases match as a
    contiguous, boundary-delimited span."""
    t = _norm(text)
    for p in phrases:
        pattern = r"(?<!\w)" + re.escape(p) + r"(?!\w)"
        if re.search(pattern, t):
            return True
    return False


@dataclass
class Turn:
    from_role: str
    message: str
    turn_number: int


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    turns: list[Turn] = field(default_factory=list)
    status: Literal["active", "ended"] = "active"
    opted_out: bool = False
    auto_reply_warned: bool = False
    committed_action: bool = False
    last_response: Optional[dict] = None


class ConversationStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: dict[str, ConversationState] = {}

    def get_or_create(self, conversation_id: str, merchant_id: Optional[str], customer_id: Optional[str]) -> ConversationState:
        with self._lock:
            st = self._states.get(conversation_id)
            if st is None:
                st = ConversationState(conversation_id=conversation_id, merchant_id=merchant_id, customer_id=customer_id)
                self._states[conversation_id] = st
                return st

            # Confirmed-by-adversarial-testing bug: conversation_id is the
            # only correlation key in the API contract, but if the caller
            # reuses one across two different merchants (or customers),
            # naively returning the existing state would splice merchant
            # A's turn history — including committed_action / auto-reply
            # counters — into merchant B's thread. Detect the mismatch and
            # start a fresh, isolated state instead of silently reusing.
            mismatch = (
                (merchant_id is not None and st.merchant_id is not None and merchant_id != st.merchant_id)
                or (customer_id is not None and st.customer_id is not None and customer_id != st.customer_id)
            )
            if mismatch:
                logger.warning(
                    "conversation_id_reused_across_identity conversation_id=%s "
                    "prior_merchant=%s new_merchant=%s prior_customer=%s new_customer=%s "
                    "— starting a fresh isolated state instead of reusing history",
                    conversation_id, st.merchant_id, merchant_id, st.customer_id, customer_id,
                )
                st = ConversationState(conversation_id=conversation_id, merchant_id=merchant_id, customer_id=customer_id)
                self._states[conversation_id] = st
            return st

    def teardown(self) -> None:
        with self._lock:
            self._states.clear()


def _repeat_count(state: ConversationState, message: str) -> int:
    norm = _norm(message)
    return sum(1 for t in state.turns if t.from_role in ("merchant", "customer") and _norm(t.message) == norm and norm != "")


def decide_reply(state: ConversationState, incoming_message: str, turn_number: int, from_role: str) -> dict:
    """Returns a dict shaped like ReplyResponse (action/body/cta/wait_seconds/rationale).

    Retry-idempotency wrapper: if this is byte-for-byte the same
    (from_role, turn_number, message) as the immediately preceding turn —
    the signature of a network retry / at-least-once delivery, not a
    human repeating themselves — replay the cached response instead of
    reprocessing. Confirmed-by-testing bug this closes: without it, a
    single retried message could get double-counted toward auto-reply
    detection and end a conversation with a real, engaged human."""
    if state.turns and state.last_response is not None:
        last_turn = state.turns[-1]
        if (
            last_turn.from_role == from_role
            and last_turn.turn_number == turn_number
            and _norm(last_turn.message) == _norm(incoming_message)
        ):
            return state.last_response

    result = _decide_reply_uncached(state, incoming_message, turn_number, from_role)
    state.last_response = result
    return result


def _decide_reply_uncached(state: ConversationState, incoming_message: str, turn_number: int, from_role: str) -> dict:
    state.turns.append(Turn(from_role=from_role, message=incoming_message, turn_number=turn_number))

    if state.status == "ended":
        return {"action": "end", "rationale": "Conversation already ended; not re-engaging."}

    text = incoming_message or ""
    repeats = _repeat_count(state, text) if text.strip() else 0

    # --- Hard stop / consent withdrawal — always wins, no further checks ---
    if _contains_any(text, _STOP_WORDS):
        state.status = "ended"
        state.opted_out = True
        return {"action": "end", "rationale": "Explicit stop/unsubscribe signal; honoring immediately, no further messages."}

    # --- Turn budget exhausted — uses the INTERNAL turn count (len of
    # our own recorded turns), not the caller-supplied `turn_number`.
    # Confirmed-by-testing bug this closes: trusting the request's own
    # turn_number let a caller that always sends turn_number=1 keep a
    # conversation alive forever, since the budget check never saw
    # anything but "1". Our own count can't be manipulated by the caller.
    if len(state.turns) >= MAX_TURNS:
        state.status = "ended"
        return {"action": "end", "rationale": f"Reached max turn budget ({MAX_TURNS}) without resolution; exiting rather than looping."}

    # --- Auto-reply detection: same verbatim canned text repeating ---
    if repeats >= AUTO_REPLY_REPEAT_THRESHOLD:
        state.status = "ended"
        return {
            "action": "end",
            "rationale": "Same message repeated 3+ times verbatim — classified as WhatsApp Business auto-reply; exiting gracefully instead of burning further turns.",
        }
    if repeats == 2 and not state.auto_reply_warned:
        state.auto_reply_warned = True
        return {
            "action": "send",
            "body": "Samajh gayi — just checking, is this an automated reply, or would you like to look at this yourself? 2-minute thing either way.",
            "cta": "binary_yes_stop",
            "rationale": "Second identical message received; probing once before assuming auto-reply, per graceful-exit pattern.",
        }

    # --- Explicit intent / commitment — route to action immediately, no re-qualification ---
    if _contains_any(text, _YES_WORDS) or (text.strip() in _POSITIVE_EMOJI):
        state.committed_action = True
        return {
            "action": "send",
            "body": "Great — starting now. I'll confirm here as soon as it's done.",
            "cta": "none",
            "rationale": "Merchant gave explicit affirmative/intent; routing directly to action instead of re-asking a qualifying question (anti-pattern from brief: don't lose momentum).",
        }

    # --- Retraction: merchant walks back a commitment we already acted on ---
    if state.committed_action and _contains_any(text, _RETRACTION_WORDS):
        state.status = "ended"
        return {
            "action": "end",
            "rationale": "Merchant retracted a prior commitment ('never mind'/'cancel that') after we'd already confirmed starting the action; rolling back and closing the thread rather than continuing to push it.",
        }
    if _contains_any(text, _RETRACTION_WORDS):
        return {
            "action": "wait",
            "wait_seconds": 3600,
            "rationale": "Retraction signal received; backing off an hour rather than proceeding or re-pitching immediately.",
        }

    # --- Already satisfied ---
    if _contains_any(text, _DONE_WORDS):
        state.status = "ended"
        return {"action": "end", "rationale": "Merchant indicates this is already handled; exiting without redundant follow-up."}

    # --- Defer ---
    if _contains_any(text, _LATER_WORDS):
        return {"action": "wait", "wait_seconds": 86400, "rationale": "Merchant asked for time; backing off 24h before re-approaching."}

    # --- Pure gratitude / soft close ---
    if _contains_any(text, _THANKS_WORDS) and len(text.split()) <= 4:
        state.status = "ended"
        return {"action": "end", "rationale": "Polite closing signal; ending the thread cleanly rather than forcing another turn."}

    # --- Negative-only emoji ---
    if text.strip() in _NEGATIVE_EMOJI:
        state.status = "ended"
        return {"action": "end", "rationale": "Negative emoji-only reply read as disengagement; exiting gracefully."}

    # --- Unknown / off-topic / hostile: stay on-mission, don't escalate ---
    if _looks_hostile(text):
        return {
            "action": "send",
            "body": "Understood — happy to step back if this isn't useful right now. If you'd still like the detail I mentioned, just say so.",
            "cta": "binary_yes_stop",
            "rationale": "Hostile/abusive tone detected; de-escalating politely without matching tone, staying on the original mission.",
        }

    return {
        "action": "send",
        "body": "Got it. Let me know if you'd like me to go ahead with what I mentioned, or if something else is more useful right now.",
        "cta": "open_ended",
        "rationale": "Unrecognized/off-topic reply; keeping the thread open with a low-friction option rather than guessing intent.",
    }


_HOSTILE_MARKERS = {"stupid", "useless", "shut up", "idiot", "nonsense", "bakwas", "bewakoof"}


def _looks_hostile(text: str) -> bool:
    return _contains_any(text, _HOSTILE_MARKERS)
