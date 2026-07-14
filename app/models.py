"""
API contract models — mirrors challenge-testing-brief.md section 2/3 exactly.

Kept intentionally permissive on `payload` (raw dict) because the four
context shapes (category / merchant / customer / trigger) are versioned by
magicpin independently of this bot, and schema evolution is an explicit
test dimension (see challenge-brief.md "future schema fields"). Strict
per-scope payload validation happens one layer down, in validation.py,
where we can fail soft instead of rejecting the whole request.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


class ContextPush(BaseModel):
    # Kept as plain str (not a Literal) so an invalid scope produces our own
    # 400 {"accepted": false, "reason": "invalid_scope"} response instead of
    # FastAPI's generic 422 — see main.py:push_context.
    scope: str
    context_id: str = Field(min_length=1)
    version: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    delivered_at: Optional[str] = None

    @field_validator("payload")
    @classmethod
    def _payload_is_object(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("payload must be a JSON object")
        return v


class ContextAck(BaseModel):
    accepted: bool
    ack_id: Optional[str] = None
    stored_at: Optional[str] = None
    reason: Optional[str] = None
    current_version: Optional[int] = None
    details: Optional[str] = None


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class TickAction(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    send_as: Literal["vera", "merchant_on_behalf"]
    trigger_id: str
    template_name: str
    template_params: list[str] = Field(default_factory=list)
    body: str
    cta: Literal["binary_yes_stop", "open_ended", "none"]
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: list[TickAction] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: Literal["merchant", "customer"] = "merchant"
    message: str = ""
    received_at: Optional[str] = None
    turn_number: int = 1


class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: Optional[str] = None
    cta: Optional[Literal["binary_yes_stop", "open_ended", "none"]] = None
    wait_seconds: Optional[int] = None
    rationale: str = ""


class HealthzResponse(BaseModel):
    status: Literal["ok", "degraded"]
    uptime_seconds: int
    contexts_loaded: dict[str, int]


class MetadataResponse(BaseModel):
    team_name: str
    team_members: list[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str
    avg_latency_ms: dict[str, float] = Field(default_factory=dict)
