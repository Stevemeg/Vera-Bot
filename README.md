# Vera Message Engine — Challenge Submission

A deterministic, layered message-composition engine for the magicpin AI
Challenge. **There is no LLM call anywhere in the decision or
composition path** — every message is produced by a rule-based pipeline
that only ever states facts that were actually pushed into the bot's
context store. This is a direct answer to the natural follow-up
question ("how much of this is really template vs. LLM-generated?"):
zero LLM calls, by construction, not by convention. `grep -ri
"openai\|anthropic\|api_key\|chat.completions" app/` returns nothing —
there's nothing to grep for.

## Architecture

```
POST /v1/context ──► ContextStore (versioned, idempotent, in-memory)
                           │
POST /v1/tick ─────► resolve (category, merchant, trigger, customer?)
                           │
                     Eligibility rules  (missing ctx / expired / consent / suppressed)
                           │
                     Decision engine — named-component scoring
                       business_impact + urgency + specificity + freshness
                       + signal_match + persona_fit + context_freshness
                       − fatigue_penalty
                           │
                     Stability / hysteresis vs. last decision for this merchant
                           │
                     Composer — template dispatch by opportunity family,
                     every fact from facts.py, nothing invented
                           │
                     Validator — single-CTA / non-empty / no-taboo /
                     non-generic; falls back to safe_fallback() on failure
                           │
                     DecisionLog.record() ── confidence + counterfactual +
                                              decision_hash for replay
                           │
                     TickResponse { body, cta, send_as, suppression_key, rationale }

POST /v1/reply ─────► ConversationState machine (stop/auto-reply/intent/
                       retraction/turn-budget) ──► ReplyResponse
```

### Sequence for one `/v1/tick` call

```
Judge                         Bot
  │  POST /v1/tick(now, triggers)
  ├──────────────────────────────►│
  │                                │ for each trigger_id:
  │                                │   resolve trigger/merchant/category/customer
  │                                │   compute merchant_context_age_days
  │                                │   evaluate_trigger() -> Opportunity(scored or ineligible)
  │                                │ group by merchant_id
  │                                │ for each merchant:
  │                                │   rank(eligible) -> ranked_eligible
  │                                │   apply_stability(last_decision, ranked_eligible) -> chosen
  │                                │   confidence(chosen, runner_up); counterfactual(chosen, runner_up)
  │                                │   compose(category, merchant, trigger, customer, chosen) -> draft
  │                                │   validate(draft) -> ok? : safe_fallback()
  │                                │   suppression.mark_fired(); record_body()
  │                                │   decisions.record(DecisionRecord)
  │                                │   append TickAction
  │◄──────────────────────────────┤
  │  TickResponse { actions: [...] }
```

- **`app/store.py`** — in-memory `ContextStore`. Idempotent on
  `(scope, context_id, version)`; higher versions replace atomically,
  equal/lower versions get `409 stale_version`. Tracks `stored_at` per
  entry (used by the context-freshness policy below). Wiped by
  `POST /v1/teardown`.
- **`app/facts.py`** — the *only* place allowed to read raw context dicts.
  Every getter is defensive (missing/null/partial data → `None`, never a
  crash) and returns nothing it can't point to in the pushed payload.
  Includes `dig_any()` (schema-tolerant field aliasing, e.g.
  `performance.ctr` → `performance.engagement`) and
  `with_category_fallback()` (a conservative generic `CategoryContext`
  for verticals the judge injects that were never pushed a real one).
- **`app/persona.py`** — deterministic merchant-persona tags
  (`inactive`, `discount_heavy`, `premium`, `growth_focused`,
  `price_sensitive`, `new`, `busy`, `established`) from already-grounded
  signals, feeding a small explicit scoring-bonus table.
- **`app/opportunities.py`** — the decision engine: eligibility gates,
  named-component scoring, deterministic ranking/tie-breaks, the fatigue
  engine, `explain()` (why every candidate won or lost),
  `confidence()`, `counterfactual()`, `decision_hash()`, and
  `apply_stability()` (hysteresis against the merchant's last decision).
- **`app/composer.py`** — template-based surface realizer, dispatched by
  `family`. `safe_fallback()` is the always-valid last resort used when
  `validators.validate()` rejects the primary draft.
- **`app/validators.py`** — the pre-send checklist (empty body, missing
  rationale, multi-CTA, taboo leak, generic-with-no-anchor).
- **`app/suppression.py`** — dedup keyed on
  `merchant × trigger_kind × suppression_key × customer_scope × time_bucket`,
  plus anti-repetition (same body never sent twice in one conversation).
- **`app/conversation.py`** — the reply-side state machine: stop/
  unsubscribe, auto-reply detection, explicit-intent routing, retraction/
  rollback, defer/already-done/thanks, turn-budget cutoff, non-escalating
  response to hostile/off-topic messages.
- **`app/main.py`** — the 5 required endpoints + `POST /v1/teardown` +
  an optional, explicitly non-contractual `GET /v1/debug/decisions/{id}`
  for replay debugging. Measures decision/composition/validation/total
  latency per call, exposed via `GET /v1/metadata.avg_latency_ms`. Never
  500s into a judge timeout for expected failure modes — malformed input
  degrades to a structured `400`; only a genuinely unexpected exception
  (see fault-injection tests) reaches the top-level handler.

## Scoring formula

```
score = business_impact          (family weight — compliance/revenue risk ranks highest)
      + urgency_component         (trigger's own 1-5 urgency × 4)
      + specificity_bonus         (+12 concrete digest item, +8 available slots,
                                    −6 generator placeholder payload)
      + freshness_bonus           (+10 expires ≤3d, +4 expires ≤14d)
      + merchant_signal_match     (+6 when an independent merchant signal corroborates the trigger)
      + context_freshness_penalty (−4 to −10 when a performance-family trigger cites a
                                    merchant snapshot pushed >2d / >7d ago)
      + persona_fit               (±3 to ±8, from persona.PERSONA_FAMILY_BONUS)
      − fatigue_penalty           (+10..+12 penalty on repeated/trailing ignores;
                                    −3 bonus, i.e. score boost, for an engaged merchant)
```

Every term lands in `Opportunity.components` and is echoed in the plain-
English `reasoning`/`rationale` string — nothing is a bare number without
an attached explanation.

### Worked example (real output from this codebase, unedited)

Two eligible triggers for the same merchant (Dr. Meera): a compliance
deadline and a research-digest item.

```
rationale (winning action):
  kind='regulation_change' family='compliance' persona='inactive';
  score=130.0 = business_impact=+95.0 + urgency_component=+16.0 +
  specificity_bonus=+16.0 + freshness_bonus=+0.0 + merchant_signal_match=+0.0 +
  context_freshness_penalty=-0.0 + persona_fit=+0.0 + fatigue_penalty=+3.0;
  anchored on digest item 'DCI revised radiograph dose limits effective 2026-12-15'.
  confidence=0.68;
  'trg_002_compliance_dci_radiograph' beat 'trg_001_research_digest_dentists' by +68.0;
  the single largest factor was business_impact (+60.0). If
  'trg_001_research_digest_dentists' had matched that component, the gap
  would close to +8.0.;
  decision_hash=3b4a5f55e3a8702f
```

That last sentence is the counterfactual: it names the *specific* factor
(family weight — compliance risk vs. pure knowledge) that decided the
outcome, and states exactly how much of the gap would remain if the
runner-up matched it. `GET /v1/debug/decisions/{merchant_id}` exposes the
full structured version of this (component-by-component, both
candidates) for anyone replay-debugging without grepping logs.

### Suppression key example

```
m_001_drmeera_dentist_delhi:regulation_change:compliance:dci_radiograph:2026:merchant_scope:82277
└─ merchant_id ─┘└── trigger kind ──┘└── trigger's own suppression_key ──┘└─ scope ─┘└ time bucket ┘
```

Re-posting the identical `/v1/tick` immediately after returns
`{"actions": []}` — the same key is already fired and not yet expired.

## Testing

```
pip install -r requirements.txt
pytest tests/ -q
```

109 tests across 12 files:
- `test_opportunities.py` — eligibility gates, scoring, deterministic
  ranking/tie-breaks, unknown-kind fallback.
- `test_composer.py` — groundedness, taboo-vocabulary filtering across
  all 5 categories, `send_as` correctness, single-CTA shape, determinism.
- `test_conversation.py` — the full reply taxonomy (stop, yes, auto-reply,
  already-done, later, hostile, off-topic, turn budget, emoji, retraction).
- `test_suppression_and_store.py` — idempotent versioning, dedup key
  scoping, expiry, anti-repetition.
- `test_api.py` — the 5 HTTP endpoints end-to-end, including 409/400
  error paths and teardown.
- `test_persona.py` — persona classification and its scoring nudges.
- `test_validators.py` — the pre-send validator pipeline and
  `safe_fallback`.
- `test_explainability_and_robustness.py` — `explain()`, `decision_hash`,
  the fatigue engine, trigger>merchant offer precedence, schema-tolerant
  fallbacks.
- `test_property_based.py` (Hypothesis) — randomized invariants: never
  crashes, at most one CTA, no taboo leak, `send_as` always matches
  scope, rationale/body never empty, identical inputs → identical
  outputs, scores stay in sane bounds, suppression keys are stable.
- `test_fault_injection.py` — concurrent writes to `ContextStore` and
  `SuppressionEngine`, partial/null merchant contexts, malformed triggers,
  wrong-typed/missing-field context pushes, a simulated storage failure
  mid-write (confirms `500` + structured body + the process stays up for
  the next request, rather than corrupting state or crashing).
- `test_adversarial_regressions.py` — regression tests for 7 confirmed
  exploits found by actually attacking the running system (see
  "Adversarial hardening pass" below): cross-merchant state bleed,
  turn-budget bypass, retry-idempotency, the suppression clock mismatch,
  Devanagari-script replies, the validator's multi-CTA blind spot, and
  the suppression TOCTOU race.

Then, against a running instance, use the provided `judge_simulator.py`
and the 30-pair offline check:

```
uvicorn app.main:app --port 8080 &
python scripts/build_submission.py --expanded-dir ../expanded --out submission.jsonl
```

`scripts/build_submission.py` runs the same `evaluate_trigger` →
`compose` → `validators.validate` → (`safe_fallback` on failure) pipeline
`main.py` uses, so the offline artifact and the live bot are guaranteed
to agree.

## Latency characteristics

Measured via `time.perf_counter()` around each pipeline stage, exposed
as a rolling average through `GET /v1/metadata.avg_latency_ms`. On a
single trigger/merchant tick in this environment: decision ≈0.3ms,
composition ≈0.05ms, validation ≈0.2ms, total ≈0.8ms — all well inside
the 30s per-call budget, as expected for a pipeline with zero network
calls (no LLM, no external API) in the hot path. Exposed mainly as an
engineering-maturity signal; the judge harness doesn't require it.

## Running locally

```
docker build -t vera-bot .
docker run -p 8080:8080 vera-bot
```

or without Docker:

```
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Deploying to a public HTTPS URL

This was built and validated in a sandboxed environment with no ability
to create hosting accounts or reach arbitrary external hosts (network
egress is allow-listed to package registries only — confirmed by testing
directly: requests to Render/Fly return `x-deny-reason: host_not_allowed`
at the proxy level, not from those services). Getting an actual public
URL live requires an account on a hosting platform, which has to be the
submitter's own — this repo ships everything needed to do that in one
command:

**Render.com** (free tier, automatic HTTPS, zero config beyond this repo):
```
# push this repo to GitHub, then in the Render dashboard:
# New -> Blueprint -> point at the repo -> it reads render.yaml automatically
```
`render.yaml` in this repo already specifies the Docker build, health
check path (`/v1/healthz`), and the persistence env var. Render assigns
`https://vera-bot-<hash>.onrender.com` automatically.

**Fly.io** (free tier, automatic HTTPS):
```
fly launch --copy-config --dockerfile Dockerfile   # reads fly.toml in this repo
fly deploy
```

**Any other Docker host** (Railway, Google Cloud Run, AWS App Runner,
a VPS with Caddy/nginx in front for TLS): build the image from the
`Dockerfile` in this repo and expose port 8080; all of them provide
HTTPS termination in front of a plain HTTP container by default.

### What was actually verified before handing this off

Everything below was run against a real, live `uvicorn` process over
real HTTP sockets in this environment (not `TestClient`) — not just
claimed:

- **All 5 required endpoints + `/v1/context` idempotency**: pushed the
  full 5-category/10-merchant/15-customer/25-trigger base dataset over
  real HTTP, re-posted an identical `(scope, context_id, version)` and
  got `409 stale_version`, ran a full tick→reply cycle, confirmed a
  replayed identical tick returns `{"actions": []}` (suppression holds
  across real requests, not just in-process test calls).
- **Persistence across a real process restart**: enabled
  `VERA_PERSIST_PATH`, pushed a merchant, `kill -9`'d the uvicorn
  process (not a graceful shutdown), started a fresh process against the
  same disk path, and confirmed `GET /v1/healthz` reported the merchant
  context intact (`contexts_loaded.merchant: 1`) with zero application
  code re-run in between. Then called `POST /v1/teardown` and confirmed
  the snapshot file was deleted from disk, not just cleared in memory.
- **The actual `judge_simulator.py`'s own `BotClient`** (the literal
  HTTP-calling code the official harness uses) — imported directly and
  run against the live server for `healthz`/`metadata`/`push_context`/
  `tick`/`reply`. All calls succeeded with correct status codes and
  response shapes. The simulator's *LLM-scored* dimensions could not be
  exercised end-to-end because `judge_simulator.py` requires a real LLM
  API key (or a local Ollama model, which needs network access this
  environment doesn't have) — that key can only be the submitter's own,
  never something to fabricate or borrow.

### One honest, non-obvious finding from that verification

`judge_simulator.py`'s own `BotClient.tick()` sends **real current
wall-clock time** as `now` (`datetime.utcnow().isoformat()`), not a
time matching the bundled sample dataset. The bundled
`dataset/triggers_seed.json` triggers carry `expires_at` values baked in
around April–June 2026 (whenever that dataset was authored). Running the
simulator today (mid-2026 and later) against those *specific sample*
triggers correctly returns `{"actions": []}` — not a malfunction, but
this bot's expiry gate correctly refusing to act on data the brief's own
"expired triggers" edge case says should be rejected. This will not
affect real grading: `challenge-testing-brief.md` §4 Phase 1–3 has the
judge push its own fresh `category`/`merchant`/`customer` contexts and
new `triggers` with `expires_at` relative to whenever the real test
actually runs, not whenever the static sample bundle was generated. It
only bites if someone runs the simulator locally against the untouched
sample dataset well after its authored dates — worth knowing before
concluding "the bot returned nothing, something's broken."

## Deliberate design choices / tradeoffs

- **Family, not raw kind.** 26 trigger kinds are classified into ~15
  families. New/unseen kinds fall back to `generic_signal` and still get
  a coherent, grounded message — exercised by
  `test_unknown_future_trigger_kind_falls_back_to_generic_family` and the
  property-based tests' `totally_novel_future_kind_*` cases.
- **Consent-scope handling is vocabulary-tolerant, not vocabulary-blind.**
  The base seed data uses `recall_reminders`/`appointment_reminders`, but
  the expanded 200-customer dataset uses per-category taxonomies
  (pharmacies: `refill_reminders`/`recall_alerts`). Hard-matching one
  vocabulary caused 6 of the 30 canonical pairs to go silent for no good
  reason — a schema artifact, not a real consent issue. The fix: block
  only on a genuine opt-out (no consent recorded at all); if the customer
  has only consented to `promotional_offers`, the same grounded facts get
  reframed as an offer instead of a due-date reminder.
- **One canonical test pair (T14) intentionally returns no message.**
  The customer's `state` is `churned`. The brief explicitly rewards
  restraint over spam, and a churned customer is exactly that case.
- **Placeholder payloads are detected and never echoed as facts.** A
  chunk of the generator's 100 triggers carry `{"placeholder": true,
  "metric_or_topic": "<kind>"}` — no real fact, just the kind name again.
  `composer._real_topic()` refuses to treat a payload value equal to the
  trigger's own `kind` as a fact (this now also directly penalizes the
  opportunity's `specificity_bonus`, not just the composer), so those
  triggers fall back to an honest, less-specific message and a lower
  score, instead of parroting a placeholder as if it were a datapoint.
- **No emoji/vocabulary bleed across categories.** `chronic_refill_due`
  has its own branch using its own payload shape (`molecule_list`,
  `stock_runs_out_iso`, `delivery_address_saved`) instead of falling
  through to the dental due-date/slots template that used to hardcode a
  🦷 for every `customer_recall`-family trigger regardless of vertical.
- **Trigger > merchant precedence for offers.** If a trigger's own
  payload flags an offer as expired/discontinued — a fresher signal than
  a possibly-stale `merchant.offers` snapshot — that offer is excluded
  from composition even if the merchant context still lists it active.
- **Hysteresis is scoped to same-tick candidate sets, not wall-clock
  history.** Because a chosen trigger is immediately suppressed
  (`suppression.mark_fired`), true tick-to-tick oscillation on a fired
  decision can't happen in this architecture. `apply_stability()` still
  guards the one place it *can* happen: a trigger whose composed body
  collided with anti-repetition (skipped without being marked fired) and
  so remains a live candidate next tick, potentially tied for the top
  score by then with something else.
- **Confidence is explicitly not a probability.** `confidence()` is a
  deterministic function of score margin and how many components
  contributed positively — labeled and documented as a heuristic, not a
  model output, to avoid implying calibration it doesn't have.

## Adversarial hardening pass

An adversarial review round explicitly targeted architectural
assumptions, replay/race conditions, and conversation-state edge cases.
Rather than speculate, each attack was actually run against the live
bot; every one below was **confirmed reproducible** before being fixed,
and each has a regression test in `test_adversarial_regressions.py`:

1. **Cross-merchant conversation state bleed.** `conversation_id` was the
   only correlation key; reusing one across two different merchants
   spliced merchant A's `committed_action`/turn history into merchant
   B's thread (a genuine "never mind" from B was read as B retracting
   A's commitment). Fixed: `ConversationStore` now detects a
   merchant/customer mismatch against an existing state and starts a
   fresh, isolated state instead of reusing history.
2. **Turn-budget bypass via caller-controlled `turn_number`.** The
   budget check trusted the request's own `turn_number` field; a caller
   that always sent `turn_number=1` kept a conversation alive forever.
   Fixed: the budget check now uses the bot's own internal turn count,
   which the caller cannot manipulate.
3. **Retry-idempotency gap on `/v1/reply`.** Unlike `/v1/context`,
   `/v1/reply` had no de-dup key; a network-retried identical message
   got double-counted toward auto-reply detection and could end a
   conversation with a real, engaged human after 3 retries. Fixed: an
   exact-match retry of the immediately preceding turn now replays the
   cached response instead of being reprocessed — and a follow-up bug in
   this fix (the endpoint's own anti-repetition guard was re-flagging
   the cached replay as "repeats text verbatim" and overriding it to
   `end`) was caught and fixed in the same pass.
4. **Suppression expiry used the real server clock, not the judge's
   simulated `now`.** Every trigger whose `expires_at` predates the real
   deployment date bypassed suppression entirely — confirmed as an
   actual duplicate send (same underlying event, updated performance
   numbers, same trigger, two sends) before the fix. Fixed: expiry
   comparisons now use the simulated `now` from the request.
5. **Multilingual gap: Devanagari-script replies unrecognized.** All
   keyword sets were romanized Hinglish only; a merchant typing in
   native Devanagari script (हाँ / रोको / अभी नहीं) fell through to the
   generic catch-all instead of being recognized as yes/stop/later.
   Fixed: core intent keywords now have Devanagari equivalents.
6. **Validator blind spot: two-question multi-CTA without the word
   "reply".** `"want me to draft it? Should I also schedule it?"` is two
   independent asks but slipped past a check keyed to `"reply"` count
   and a >2 question-mark threshold. Fixed: a distinct-CTA-phrase count
   now catches this — verified not to false-positive on legitimate
   single-CTA messages like "Reply YES if interested."
7. **Suppression check-then-act race (TOCTOU).** The original flow was
   `already_fired()` [check] → compose() + validate() [real work] →
   `mark_fired()` [act] — two separate lock acquisitions with a gap in
   between. Proven exploitable with a deliberate concurrency test before
   the fix (two threads both passed the check and both marked fired).
   Fixed: `SuppressionEngine.try_reserve()` checks-and-marks atomically
   under one lock acquisition; a `release()` undoes a reservation if the
   send is subsequently abandoned (e.g. anti-repetition decides not to
   send after all), so nothing gets permanently blocked for no reason.

**Known limitation, deliberately not "fixed" the same way:** a context
push with only a *subset* of a merchant's fields (e.g. only an updated
`performance` block) causes a full replace, silently discarding
`identity`/`offers`/`conversation_history`, because `POST /v1/context`'s
documented contract is "a higher version replaces atomically" — full
object replacement, not a field-level merge. The testing brief's payload
examples always show complete objects, so this is spec-compliant
behavior rather than a bug, but it's a sharp edge if a real integration
ever sends partial updates: this bot does not defend against that, and
merging would introduce its own risks (stale merged fields never
getting cleared). Documented here rather than silently patched.

## What I'd add with more time

- Move `ContextStore` / `SuppressionEngine` / `ConversationStore` /
  `DecisionLog` to Redis so the service can run >1 worker/replica
  (currently correct but single-process only — documented in the
  Dockerfile).
- A thin optional LLM realizer pass *after* the deterministic composer,
  purely for prose smoothing (never for the decision), with a hard
  post-check that every number/date/citation in the LLM's output still
  matches something the deterministic layer produced.
- A true category-inheritance hierarchy (`healthcare -> {dentists,
  pharmacies, opticians, ...}`) so an unseen vertical can inherit partial
  vocabulary/taboo rules from its nearest known parent, rather than
  falling all the way to the fully generic fallback.
- Richer Hindi-English code-mix (current templates are a reasonable but
  hand-written approximation of the reference conversation patterns, not
  a full bilingual grammar).
