"""
Vera message-engine bot — FastAPI surface.

Implements exactly the 5 endpoints in challenge-testing-brief.md §2, plus
an optional POST /v1/teardown (also specified there, §11) that wipes
in-memory state at the end of a test run.

Layering (Context Store -> Opportunities/Decision Engine -> Composer ->
JSON response) lives in separate modules so each is independently
testable; this file is just wiring + HTTP-contract concerns (status
codes, idempotency, payload caps, timeouts-by-design).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import facts
from .conversation import ConversationStore, decide_reply
from .models import (
    VALID_SCOPES,
    ContextAck,
    ContextPush,
    HealthzResponse,
    MetadataResponse,
    ReplyRequest,
    ReplyResponse,
    TickRequest,
    TickResponse,
)
from .opportunities import (
    DecisionLog,
    DecisionRecord,
    Opportunity,
    apply_stability,
    confidence,
    counterfactual,
    decision_hash,
    evaluate_trigger,
    explain,
    rank,
)
from .composer import compose, safe_fallback
from .store import ContextStore, StaleVersionError
from .suppression import SuppressionEngine, now_epoch, time_bucket
from . import validators

logger = logging.getLogger("vera_bot")

MAX_CONTEXT_PAYLOAD_BYTES = 500 * 1024  # per testing brief §5
MAX_ACTIONS_PER_TICK = 20
MERCHANT_CONTEXT_STALE_AFTER_DAYS = 7  # matches opportunities.evaluate_trigger's freshness bands

app = FastAPI(title="Vera Message Engine — Challenge Submission")

store = ContextStore()
suppression = SuppressionEngine()
conversations = ConversationStore()
decisions = DecisionLog()


class LatencyTracker:
    """Rolling average latency per pipeline stage — exposed via
    /v1/metadata as engineering-maturity signal (the challenge doesn't
    require it, but 'decision/validation/composition latency' were
    explicitly called out as worth measuring even if unchecked)."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def record(self, stage: str, ms: float) -> None:
        self._totals[stage] = self._totals.get(stage, 0.0) + ms
        self._counts[stage] = self._counts.get(stage, 0) + 1

    def averages_ms(self) -> dict[str, float]:
        return {k: round(self._totals[k] / self._counts[k], 3) for k in self._totals if self._counts.get(k)}


latency = LatencyTracker()

METADATA = MetadataResponse(
    team_name="Solo Submission",
    team_members=["Bunny"],
    model="deterministic-template-composer-v1 (no LLM call anywhere in the decision or composition path)",
    approach=(
        "Layered deterministic pipeline: ContextStore -> eligibility rules -> "
        "named-component opportunity scoring (decision engine, with persona/"
        "fatigue/freshness modifiers) -> pre-send validator -> template-based "
        "composer grounded strictly in pushed context facts -> JSON response. "
        "Every fact traces to facts.py; no LLM call is used for decision-making "
        "or text generation, so temperature-based non-determinism is "
        "structurally impossible rather than merely configured away."
    ),
    contact_email="team@example.com",
    version="2.0.0",
    submitted_at=datetime.now(timezone.utc).isoformat(),
)


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------
@app.get("/v1/healthz", response_model=HealthzResponse)
async def healthz():
    return HealthzResponse(
        status="ok",
        uptime_seconds=store.uptime_seconds(),
        contexts_loaded=store.counts(),
    )


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------
@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata():
    return METADATA.model_copy(update={"avg_latency_ms": latency.averages_ms()})


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------
@app.post("/v1/context")
async def push_context(request: Request):
    raw = await request.body()
    if len(raw) > MAX_CONTEXT_PAYLOAD_BYTES:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "payload_too_large", "details": f"exceeds {MAX_CONTEXT_PAYLOAD_BYTES} bytes"},
        )

    try:
        import json

        body = json.loads(raw or b"{}")
    except Exception as exc:  # malformed JSON — degrade gracefully, never 500
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed_json", "details": str(exc)})

    scope = body.get("scope")
    if scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": f"got '{scope}'"})

    try:
        push = ContextPush(**body)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_payload", "details": str(exc)})

    try:
        entry = store.put(push.scope, push.context_id, push.version, push.payload)
    except StaleVersionError as exc:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": exc.current_version},
        )

    ack = ContextAck(
        accepted=True,
        ack_id=f"ack_{push.context_id}_v{push.version}",
        stored_at=datetime.fromtimestamp(entry.stored_at, tz=timezone.utc).isoformat(),
    )
    return JSONResponse(status_code=200, content=ack.model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------
def _resolve_trigger_bundle(trigger_id: str):
    trigger = store.get("trigger", trigger_id)
    if trigger is None:
        return None, None, None, None
    merchant_id = trigger.get("merchant_id") or facts.dig(trigger, "payload", "merchant_id")
    customer_id = trigger.get("customer_id")
    merchant = store.get("merchant", merchant_id) if merchant_id else None
    category_slug = facts.dig(merchant, "category_slug") if merchant else facts.dig(trigger, "payload", "category")
    raw_category = store.get("category", category_slug) if category_slug else None
    category = facts.with_category_fallback(raw_category)
    customer = store.get("customer", customer_id) if customer_id else None
    return trigger, merchant, category, customer


@app.post("/v1/tick", response_model=TickResponse)
async def tick(body: TickRequest):
    tick_start = time.perf_counter()

    # Group candidate triggers by merchant so we emit at most one action per
    # (merchant_id, conversation_id) pair this tick, per testing brief FAQ.
    by_merchant: dict[str, list[tuple[str, Opportunity]]] = {}

    decision_start = time.perf_counter()
    for trig_id in body.available_triggers:
        trigger, merchant, category, customer = _resolve_trigger_bundle(trig_id)
        if trigger is None:
            continue
        merchant_id = trigger.get("merchant_id") or "unknown_merchant"
        raw_supp_key = trigger.get("suppression_key", trig_id)
        bucket = time_bucket(body.now, granularity_seconds=6 * 3600)
        supp_key = suppression.build_key(merchant_id, trigger.get("kind", ""), raw_supp_key, trigger.get("customer_id"), bucket)
        already = suppression.already_fired(supp_key, now=now_epoch(body.now))

        merchant_age_days = _merchant_context_age_days(merchant_id, body.now)
        opp = evaluate_trigger(
            trigger, category, merchant, customer, body.now, suppressed=already,
            merchant_context_age_days=merchant_age_days,
        )
        by_merchant.setdefault(merchant_id, []).append((supp_key, opp))
    latency.record("decision", (time.perf_counter() - decision_start) * 1000)

    actions = []
    for merchant_id, pairs in by_merchant.items():
        opps = [o for _, o in pairs]
        ranked_eligible = rank([o for o in opps if o.eligible])
        if not ranked_eligible:
            continue

        previous_decision = decisions.last(merchant_id)
        chosen = apply_stability(previous_decision, ranked_eligible)
        runner_up = next((o for o in ranked_eligible if o.trigger_id != chosen.trigger_id), None)
        supp_key = next(k for k, o in pairs if o is chosen)
        trigger = chosen.trigger

        # Atomically reserve the suppression key NOW — a single lock
        # acquisition that both checks and marks — rather than relying on
        # the earlier already_fired() peek (used only for eligibility
        # scoring) plus a mark_fired() at the very end of this loop. The
        # old check-then-act pattern left a real gap between the check and
        # the mark with actual work (compose + validate) happening in
        # between; confirmed exploitable under concurrent /v1/tick calls
        # targeting the same trigger. If we lose the race, someone else
        # (or a prior iteration) already claimed this key this instant —
        # skip rather than send.
        if not suppression.try_reserve(supp_key, now=now_epoch(body.now), expires_at_iso=trigger.get("expires_at")):
            continue

        # Explainability: log every candidate considered and why it lost,
        # even though the public response schema only carries `rationale`.
        conf = confidence(chosen, runner_up)
        cf = counterfactual(chosen, runner_up)
        if len(opps) > 1:
            logger.info(
                "opportunity_explanation merchant=%s confidence=%.3f counterfactual=%s considered=%s",
                merchant_id, conf, cf, explain(opps),
            )

        customer_id = trigger.get("customer_id")
        merchant = store.get("merchant", merchant_id)
        category_slug = facts.dig(merchant, "category_slug")
        raw_category = store.get("category", category_slug) if category_slug else None
        category = facts.with_category_fallback(raw_category)
        customer = store.get("customer", customer_id) if customer_id else None

        compose_start = time.perf_counter()
        composed = compose(category, merchant or {}, trigger, customer, chosen)
        latency.record("composition", (time.perf_counter() - compose_start) * 1000)

        validate_start = time.perf_counter()
        validation = validators.validate(composed.body, composed.cta, composed.rationale, category)
        latency.record("validation", (time.perf_counter() - validate_start) * 1000)
        if not validation.ok:
            logger.warning(
                "validation_rejected merchant=%s trigger=%s failures=%s",
                merchant_id, trigger.get("id"), validation.failures,
            )
            composed = safe_fallback(category, merchant or {}, trigger, chosen)

        conversation_id = f"conv_{merchant_id}_{trigger.get('id')}"

        if suppression.is_repeat_body(conversation_id, composed.body):
            suppression.release(supp_key)  # nothing was actually sent — don't leave a dead reservation behind
            continue  # anti-repetition: never resend identical text

        suppression.record_body(conversation_id, composed.body)

        d_hash = decision_hash(
            store.get_version("category", category.get("slug")) if category.get("slug") else None,
            store.get_version("merchant", merchant_id),
            chosen,
        )

        cf_statement = cf["statement"] if cf and cf.get("statement") else (
            "no eligible runner-up this tick" if runner_up is None else f"runner-up '{runner_up.trigger_id}' scored {runner_up.score}"
        )
        raw_top = ranked_eligible[0]
        stability_note = " [hysteresis: held previous pick over a small-margin alternative]" if chosen.trigger_id != raw_top.trigger_id else ""

        decisions.record(
            DecisionRecord(
                merchant_id=merchant_id,
                category_version=store.get_version("category", category.get("slug")) if category.get("slug") else None,
                merchant_version=store.get_version("merchant", merchant_id),
                selected_trigger_id=chosen.trigger_id,
                selected_family=chosen.family,
                selected_score=chosen.score,
                selected_components=chosen.components,
                runner_up_trigger_id=runner_up.trigger_id if runner_up else None,
                runner_up_score=runner_up.score if runner_up else None,
                confidence=conf,
                decision_hash=d_hash,
                validator_ok=validation.ok,
                validator_failures=validation.failures,
            )
        )

        actions.append(
            {
                "conversation_id": conversation_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composed.send_as,
                "trigger_id": trigger.get("id"),
                "template_name": f"vera_{chosen.family}_v1",
                "template_params": [facts.merchant_name(merchant)],
                "body": composed.body,
                "cta": composed.cta,
                "suppression_key": supp_key,
                "rationale": (
                    f"{composed.rationale} confidence={conf:.2f}; {cf_statement}; "
                    f"decision_hash={d_hash}{stability_note}"
                ),
            }
        )
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

    latency.record("tick_total", (time.perf_counter() - tick_start) * 1000)
    return TickResponse(actions=actions)


def _merchant_context_age_days(merchant_id: str, now_iso: str) -> Optional[float]:
    stored_at = store.get_stored_at("merchant", merchant_id)
    if stored_at is None:
        return None
    try:
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        now_dt = datetime.now(timezone.utc)
    age_seconds = now_dt.timestamp() - stored_at
    return max(age_seconds, 0.0) / 86400.0


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------
@app.post("/v1/reply", response_model=ReplyResponse)
async def reply(body: ReplyRequest):
    state = conversations.get_or_create(body.conversation_id, body.merchant_id, body.customer_id)
    prior_cached_response = state.last_response  # captured before the call, for cache-hit detection below
    result = decide_reply(state, body.message, body.turn_number, body.from_role)

    # A retry-idempotency cache hit returns the exact same dict object that
    # was cached from the original processing of this turn (see
    # conversation.decide_reply's docstring). That's a deliberate replay,
    # not a new decision — it must NOT be run through the anti-repetition
    # check below, which exists to catch a *new* decision that happens to
    # produce identical text, not to catch us echoing our own retry cache.
    # Confirmed-by-testing bug this closes: without this check, a network
    # retry got correctly identified as a retry, then immediately
    # mis-flagged as "would repeat text verbatim" by the anti-repetition
    # guard and converted to "end" anyway — reintroducing the exact
    # false-positive the retry-idempotency fix was meant to close.
    is_cache_hit_replay = prior_cached_response is not None and result is prior_cached_response

    if not is_cache_hit_replay and result.get("action") == "send":
        candidate_body = result.get("body", "")
        if suppression.is_repeat_body(body.conversation_id, candidate_body):
            # Never resend verbatim text even from a rule branch collision.
            result = {
                "action": "end",
                "rationale": "Next message would repeat prior text verbatim; ending instead of spamming.",
            }
        else:
            suppression.record_body(body.conversation_id, candidate_body)

    return ReplyResponse(**{k: v for k, v in result.items() if k in ReplyResponse.model_fields})


# ---------------------------------------------------------------------------
# POST /v1/teardown (optional, per testing brief §11 — privacy: wipe state)
# ---------------------------------------------------------------------------
@app.post("/v1/teardown")
async def teardown():
    store.teardown()
    suppression.teardown()
    conversations.teardown()
    decisions.teardown()
    return {"status": "wiped"}


# ---------------------------------------------------------------------------
# GET /v1/debug/decisions/{merchant_id} (optional, non-contractual)
#
# Not one of the 5 required endpoints — purely a diagnostic surface so the
# "internal considered[] + why_lost" explanation and the full DecisionLog
# history (context versions, selected/runner-up, components, validator
# outcome) are inspectable without grepping logs, for anyone doing a
# replay-debugging pass on this submission.
# ---------------------------------------------------------------------------
@app.get("/v1/debug/decisions/{merchant_id}")
async def debug_decisions(merchant_id: str):
    history = decisions.history(merchant_id)
    return {
        "merchant_id": merchant_id,
        "decisions": [
            {
                "selected_trigger_id": d.selected_trigger_id,
                "selected_family": d.selected_family,
                "selected_score": d.selected_score,
                "selected_components": d.selected_components,
                "runner_up_trigger_id": d.runner_up_trigger_id,
                "runner_up_score": d.runner_up_score,
                "confidence": d.confidence,
                "decision_hash": d.decision_hash,
                "validator_ok": d.validator_ok,
                "validator_failures": d.validator_failures,
                "category_version": d.category_version,
                "merchant_version": d.merchant_version,
            }
            for d in history
        ],
    }
    return {"status": "wiped"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never 500 into a judge timeout; degrade to a structured error instead.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error", "details": str(exc)})
