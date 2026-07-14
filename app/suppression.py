"""
SuppressionEngine.

Key shape (per challenge brief §"SUPPRESSION" and the trigger's own
`suppression_key`):

    {merchant_id}:{trigger_kind}:{campaign_or_suppression_key}:{customer_scope}:{time_bucket}

We trust the trigger's own `suppression_key` as the primary dedup token
(it's what the judge/production system uses to correlate across
resends), but we *compose* our own full key on top of it so that the
same suppression_key reused for a different merchant/customer scope
does not cross-suppress unrelated conversations — a deliberate defense
against "duplicate context" and "identical scores" edge cases.

Two independent protections:
  1. `already_sent(key)` — true if this exact suppression key has fired
     and not expired. Prevents duplicate sends / loops / replay spam.
  2. `is_repeat_body(conversation_id, body)` — true if we already sent
     this exact text in this conversation. Prevents verbatim repetition
     inside a live multi-turn thread even when the suppression key
     legitimately changes turn to turn.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


def time_bucket(now_iso: Optional[str], granularity_seconds: int = 3600) -> str:
    """Deterministic coarse time bucket, used only to make suppression keys
    resendable after a cooldown window rather than permanent."""
    try:
        ts = datetime.fromisoformat((now_iso or "").replace("Z", "+00:00"))
    except (ValueError, TypeError):
        ts = datetime.now(timezone.utc)
    bucket = int(ts.timestamp() // granularity_seconds)
    return str(bucket)


def now_epoch(now_iso: Optional[str]) -> float:
    """Parse the judge's simulated `now` into epoch seconds, falling back
    to real wall-clock only if the input is missing/malformed. This is
    the reference time suppression expiry must use — see
    SuppressionEngine.already_fired's docstring for why."""
    try:
        return datetime.fromisoformat((now_iso or "").replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return time.time()


@dataclass
class _Entry:
    fired_at: float
    expires_at: Optional[float]


class SuppressionEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._fired: dict[str, _Entry] = {}
        self._sent_bodies: dict[str, set[str]] = {}

    def build_key(
        self,
        merchant_id: Optional[str],
        trigger_kind: str,
        suppression_key: str,
        customer_id: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> str:
        scope = customer_id or "merchant_scope"
        parts = [merchant_id or "unknown_merchant", trigger_kind or "unknown_kind", suppression_key or "no_key", scope]
        if bucket:
            parts.append(bucket)
        return ":".join(parts)

    def already_fired(self, key: str, now: Optional[float] = None) -> bool:
        """`now` is the *caller-supplied* reference time (epoch seconds),
        defaulting to real wall-clock only if the caller has no simulated
        clock to offer. Passing the judge's own simulated `now` here (see
        main.py) is required — comparing a trigger's simulated
        `expires_at` against the real server clock caused every trigger
        whose expiry predates the real deployment date to bypass
        suppression entirely (confirmed by adversarial testing)."""
        reference = now if now is not None else time.time()
        with self._lock:
            entry = self._fired.get(key)
            if entry is None:
                return False
            if entry.expires_at is not None and reference > entry.expires_at:
                del self._fired[key]
                return False
            return True

    def mark_fired(self, key: str, expires_at_iso: Optional[str] = None) -> None:
        expires_at = None
        if expires_at_iso:
            try:
                expires_at = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                expires_at = None
        with self._lock:
            self._fired[key] = _Entry(fired_at=time.time(), expires_at=expires_at)

    def try_reserve(self, key: str, now: Optional[float] = None, expires_at_iso: Optional[str] = None) -> bool:
        """Atomic check-and-mark: returns True (and marks the key fired)
        only if it was NOT already fired — under a single lock
        acquisition, closing the TOCTOU gap that existed when callers did
        a separate `already_fired()` check followed by real work
        (compose/validate) and only then called `mark_fired()`. Two
        concurrent callers racing on the same key: exactly one gets
        `True`; the other gets `False` and must not send."""
        reference = now if now is not None else time.time()
        expires_at = None
        if expires_at_iso:
            try:
                expires_at = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                expires_at = None
        with self._lock:
            entry = self._fired.get(key)
            if entry is not None and (entry.expires_at is None or reference <= entry.expires_at):
                return False  # already reserved and not yet expired
            self._fired[key] = _Entry(fired_at=time.time(), expires_at=expires_at)
            return True

    def release(self, key: str) -> None:
        """Undo a try_reserve() when the reserved send is subsequently
        abandoned (e.g. anti-repetition decided not to send after all),
        so a legitimately different future attempt isn't permanently
        blocked by a reservation nothing was ever sent for."""
        with self._lock:
            self._fired.pop(key, None)

    def is_repeat_body(self, conversation_id: str, body: str) -> bool:
        with self._lock:
            seen = self._sent_bodies.get(conversation_id, set())
            return body in seen

    def record_body(self, conversation_id: str, body: str) -> None:
        with self._lock:
            self._sent_bodies.setdefault(conversation_id, set()).add(body)

    def teardown(self) -> None:
        with self._lock:
            self._fired.clear()
            self._sent_bodies.clear()
