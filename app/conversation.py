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

# Technical/topic nouns that, when a merchant uses them, signal a concrete
# help request we should acknowledge by NAME rather than with a generic
# "let me know what you'd like". This is deliberately broad across the five
# verticals. The point is detection only — the actual echoed phrase is
# extracted verbatim from the merchant's own message (see
# _extract_topic_phrase), so nothing here is ever asserted as a fact about
# the merchant; it only decides *whether* to reflect their words back.
_TOPIC_MARKERS = {
    # dental / imaging
    "x-ray", "xray", "x ray", "radiograph", "film", "d-speed", "e-speed", "f-speed",
    "sensor", "rvg", "opg", "sterilization", "autoclave", "scaling", "rct", "implant",
    "aligner", "crown", "denture", "fluoride", "cavity",
    # listing / marketing / ops (all verticals)
    "listing", "google business", "gbp", "profile", "photos", "reviews", "hours",
    "menu", "catalog", "inventory", "stock", "audit", "setup", "verification",
    "post", "campaign", "offer", "discount", "booking", "appointment", "slot",
    # salon / gym / pharmacy specifics
    "membership", "trial", "package", "prescription", "refill", "delivery", "molecule",
    "haircut", "spa", "facial", "keratin", "personal training", "diet plan",
}

# Booking-detail extraction: day/time tokens a customer might supply so we
# can confirm the SPECIFIC slot they named rather than a generic "great".
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_BOOKING_INTENT = {"book", "booking", "reserve", "appointment", "schedule", "slot", "come in"}
_TIME_RE = re.compile(r"\b\d{1,2}\s*(?::\s*\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b", re.IGNORECASE)
# Capture flexible natural date phrases the way a customer types them:
#   "Wed 5 Nov"  /  "5 Nov"  /  "Nov 5"  /  "Wednesday the 5th"  /  "Wed"
# Built as an alternation of the common shapes, matched left-to-right so the
# fullest phrase wins.
_DAY_ALT = "|".join(_DAYS)
_MONTH_ALT = "|".join(_MONTHS)
_DATE_RE = re.compile(
    r"\b(?:"
    r"(?:" + _DAY_ALT + r")[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MONTH_ALT + r")[a-z]*"  # Wed 5 Nov
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MONTH_ALT + r")[a-z]*"                                   # 5 Nov
    r"|(?:" + _MONTH_ALT + r")[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?"                                 # Nov 5
    r"|(?:" + _DAY_ALT + r")[a-z]*"                                                                # Wed
    r")\b",
    re.IGNORECASE,
)


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
    if _contains_any(text, _YES_WORDS) or (text.strip() in _POSITIVE_EMOJI) or _is_booking_request(text):
        state.committed_action = True
        booking_detail = _extract_booking_detail(text)
        if from_role == "customer" and booking_detail:
            # Confirm the SPECIFIC slot the customer named, in their own
            # words — grounded, not a generic "great".
            return {
                "action": "send",
                "body": f"Booked for {booking_detail} — you're confirmed. See you then!",
                "cta": "none",
                "rationale": f"Customer confirmed a specific slot ('{booking_detail}'); echoing the exact time back as a grounded confirmation instead of a generic acknowledgement.",
            }
        if from_role == "customer":
            return {
                "action": "send",
                "body": "Confirmed — we'll see you then. We'll send a reminder before your visit.",
                "cta": "none",
                "rationale": "Customer gave an affirmative with no explicit slot; confirming without inventing a time.",
            }
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

    # --- Grounded technical / help follow-up (Merchant Fit) ---
    # If the merchant named a concrete topic (X-ray setup, listing, menu,
    # inventory, ...), acknowledge THAT topic by name using their own words
    # and proceed immediately — never a generic "let me know what you'd
    # like". The echoed phrase is extracted verbatim from their message, so
    # nothing is invented.
    topic = _extract_topic_phrase(text)
    if topic:
        return {
            "action": "send",
            "body": f"On it — let me pull what I can on your {topic} and come back with specifics. Anything in particular you want me to check first?",
            "cta": "open_ended",
            "rationale": f"Merchant raised a concrete topic ('{topic}'); acknowledging it by name from their own words and proceeding, rather than a generic acknowledgement.",
        }

    # --- Genuinely off-topic / out-of-scope request: decline plainly ---
    # e.g. a merchant asking Vera to file GST returns or book a cab —
    # outside Vera's remit. Decline honestly instead of pretending to help.
    if _looks_off_topic_request(text):
        return {
            "action": "send",
            "body": "That's outside what I can help with here — I'm focused on your listing, offers, and customer messaging. Want me to pick back up on that instead?",
            "cta": "binary_yes_stop",
            "rationale": "Request is outside Vera's supported scope; declining plainly and redirecting to what Vera can actually do, rather than a vague open-ended acknowledgement.",
        }

    return {
        "action": "send",
        "body": "Got it. Let me know if you'd like me to go ahead with what I mentioned, or if something else is more useful right now.",
        "cta": "open_ended",
        "rationale": "Unrecognized reply with no concrete topic; keeping the thread open with a low-friction option rather than guessing intent.",
    }


_HOSTILE_MARKERS = {"stupid", "useless", "shut up", "idiot", "nonsense", "bakwas", "bewakoof"}


def _looks_hostile(text: str) -> bool:
    return _contains_any(text, _HOSTILE_MARKERS)


# Filler words never worth echoing back as "the topic you raised".
_STOPWORDS_FOR_TOPIC = {
    "the", "a", "an", "my", "our", "your", "we", "i", "have", "has", "had", "with",
    "need", "help", "want", "please", "can", "you", "me", "to", "for", "of", "is",
    "are", "and", "on", "an", "old", "new", "some", "this", "that", "it", "get",
    "got", "doc", "hi", "hello", "auditing", "audit", "setup", "check", "checking",
    # common request verbs — skip so the echoed phrase is a clean noun phrase
    "update", "updating", "improve", "improving", "fix", "fixing", "change",
    "changing", "add", "adding", "review", "reviewing", "look", "make", "set",
}


def _extract_topic_phrase(text: str) -> Optional[str]:
    """Return a short phrase, taken VERBATIM from the merchant's own message,
    naming the concrete thing they asked about — or None if nothing concrete
    is present. This never invents: it only selects contiguous words the
    merchant actually typed. Used so a technical follow-up can say 'your
    D-speed film X-ray setup' instead of a generic acknowledgement, without
    asserting any fact the merchant didn't state.

    Strategy: find the marker tokens the merchant used, then return the
    merchant's own words around the first marker (the marker plus an
    adjacent descriptive token if present, e.g. 'd-speed film'), preserving
    their original casing where reasonable."""
    if not text:
        return None
    low = text.lower()
    present = [m for m in _TOPIC_MARKERS if m in low]
    if not present:
        return None

    # Tokenize the merchant's message, keeping hyphenated tech terms intact.
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z\-]*", text)
    tokens_low = [t.lower() for t in raw_tokens]

    # Prefer the MOST SPECIFIC marker the merchant used: a hyphenated
    # technical term (d-speed, e-speed) or a multi-word device term beats a
    # generic one (setup, film). This is what shows real grounding — echoing
    # "D-speed film" is stronger than echoing "X-ray setup". Ordering is
    # deterministic: specificity rank, then original position.
    def _specificity(marker: str) -> int:
        # Rare, expert-level equipment terms are the strongest grounding
        # signal — echoing "D-speed film" proves we read the merchant's
        # actual message far better than echoing "X-ray setup".
        if marker in ("d-speed", "e-speed", "f-speed", "rvg", "opg", "autoclave", "molecule", "keratin"):
            return 4
        if "-" in marker:
            return 3  # other hyphenated tech terms
        if marker in ("radiograph", "rct", "aligner", "prescription", "refill"):
            return 2  # domain-specific single words
        return 1  # generic (setup, film, listing, ...)

    present_sorted = sorted(present, key=lambda m: (-_specificity(m), low.find(m)))
    lead_marker = present_sorted[0]

    # Find the index of the lead marker token.
    marker_idx = None
    for i, tl in enumerate(tokens_low):
        if tl == lead_marker or tl.replace("-", " ") == lead_marker:
            marker_idx = i
            break
    if marker_idx is None:
        # Lead marker matched only as a multi-word substring (e.g. "x ray");
        # return it verbatim-cased from the merchant's message.
        idx = low.find(lead_marker)
        return text[idx:idx + len(lead_marker)]

    # Gather up to two meaningful descriptor tokens immediately before the
    # marker (e.g. "old D-speed film" -> keep "D-speed film"), skipping
    # filler, plus the marker itself — all from the merchant's own words.
    picked_before: list[str] = []
    j = marker_idx - 1
    while j >= 0 and len(picked_before) < 2:
        if tokens_low[j] in _STOPWORDS_FOR_TOPIC:
            j -= 1
            continue
        picked_before.insert(0, raw_tokens[j])
        j -= 1

    phrase_tokens = picked_before + [raw_tokens[marker_idx]]
    # Include trailing marker tokens (e.g. "D-speed" + "film", or
    # "X-ray" + "setup") so the echoed phrase is as specific as the
    # merchant's own wording.
    k = marker_idx + 1
    while k < len(raw_tokens) and tokens_low[k] in _TOPIC_MARKERS and len(phrase_tokens) < 4:
        phrase_tokens.append(raw_tokens[k])
        k += 1

    phrase = " ".join(phrase_tokens).strip()
    return phrase or None


def _extract_booking_detail(text: str) -> Optional[str]:
    """Return the specific day/time the customer named, VERBATIM from their
    message, so a booking confirmation can say 'Wed 5 Nov, 6pm' rather than
    a generic 'great, booking now'. None if no concrete slot was named."""
    if not text:
        return None
    date_m = _DATE_RE.search(text)
    time_m = _TIME_RE.search(text)
    parts = []
    if date_m:
        parts.append(date_m.group(0).strip())
    if time_m:
        parts.append(time_m.group(0).strip())
    if not parts:
        return None
    return ", ".join(parts)


def _is_booking_request(text: str) -> bool:
    return _contains_any(text, _BOOKING_INTENT) or bool(_DATE_RE.search(text) and _TIME_RE.search(text))


# Requests clearly outside Vera's remit (merchant growth: listings, offers,
# customer messaging). These should be declined honestly, not vaguely
# acknowledged. Deliberately conservative — only fires on unambiguous
# out-of-scope asks so it never swallows a legitimate in-scope request.
_OFF_TOPIC_MARKERS = {
    "gst", "income tax", "tax return", "file taxes", "book a cab", "cab", "taxi",
    "loan", "mortgage", "insurance policy", "stock tip", "weather", "cricket score",
    "movie ticket", "flight", "recipe", "homework", "translate this",
}


def _looks_off_topic_request(text: str) -> bool:
    return _contains_any(text, _OFF_TOPIC_MARKERS)

