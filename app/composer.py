"""
Message planner + surface realizer.

Composition is entirely template-based and deterministic: given identical
(category, merchant, trigger, customer, opportunity) inputs it produces
byte-identical output, every time, with zero LLM calls. This satisfies
"the LLM must never decide what action to take" *and* removes LLM
non-determinism/latency/cost from the composition path entirely.

Rendering is grouped by opportunity `family` (see opportunities.py) rather
than by raw trigger `kind`, so a brand-new/unrecognized kind that maps to
"generic_signal" still gets a safe, grounded, non-crashing message instead
of a KeyError.

Every f-string slot below is filled only from `facts.py` getters — never
from a literal invented number, offer, date, or citation. If a fact isn't
available, the sentence that would have used it is simply omitted, not
padded with a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import facts
from .opportunities import Opportunity

# Hard cap on a composed message body. The judge enforces a 320-char limit;
# we assemble structurally to fit it (dropping the lowest-priority *optional*
# grounded clause) rather than blind-truncating mid-word, so a shortened
# message is still a complete, grounded sentence.
MAX_BODY_CHARS = 320
REQUIRED = 1_000_000  # sentinel priority for clauses that must never be dropped


@dataclass
class ComposedMessage:
    body: str
    cta: str
    send_as: str
    rationale: str


def _assemble_within_budget(clauses: list[tuple], max_chars: int = MAX_BODY_CHARS) -> str:
    """Assemble a message from ordered `clauses`, each a (text, priority)
    pair. `priority` is an int: higher = more important. Clauses are kept in
    their given reading order; when the joined result exceeds `max_chars`,
    the LOWEST-priority clause is dropped (ties broken by later position),
    and we retry — so the message stays in natural reading order, stays a
    set of complete grounded fragments (never truncated mid-clause), and
    sheds only as much as needed to fit.

    A priority of `REQUIRED` marks a clause that must never be dropped (the
    lead fact and the single CTA). If required clauses alone still exceed the
    budget, they're returned as-is: we never drop a required grounded fact or
    truncate one to hit a character count."""
    kept = [(text, prio) for (text, prio) in clauses if text]
    while True:
        candidate = " ".join(t for (t, _p) in kept).strip()
        if len(candidate) <= max_chars:
            return candidate
        # Find the lowest-priority droppable clause (highest index wins ties).
        droppable = [(i, p) for i, (_t, p) in enumerate(kept) if p != REQUIRED]
        if not droppable:
            return candidate  # only required clauses remain; return as-is
        drop_i = min(droppable, key=lambda ip: (ip[1], -ip[0]))[0]
        kept.pop(drop_i)


def _real_topic(trigger: dict, *keys: str) -> Optional[str]:
    """Return the first present payload value for `keys`, unless it's a
    generator placeholder (payload.get(key) == trigger kind, meaning no
    real fact was actually supplied) or the payload is explicitly flagged
    `placeholder: true`. Prevents literal kind-name leakage like
    'milestone hit — milestone_reached'."""
    payload = trigger.get("payload") or {}
    if payload.get("placeholder") is True:
        return None
    kind = trigger.get("kind")
    for k in keys:
        v = payload.get(k)
        if v and v != kind:
            return v
    return None


def _greeting(category: dict, merchant: dict, customer: Optional[dict]) -> str:
    if customer is not None:
        name = facts.dig(customer, "identity", "name")
        return f"Hi {name}" if name else "Hi"
    return facts.salutation(category, merchant)


def _mix(hi_variant: str, en_variant: str, use_hindi: bool) -> str:
    return hi_variant if use_hindi else en_variant


def _knowledge(category, merchant, trigger, opp: Opportunity) -> str:
    payload = trigger.get("payload") or {}
    item = facts.digest_item(category, payload.get("top_item_id"))
    greet = _greeting(category, merchant, None)
    if item:
        title = item.get("title")
        source = item.get("source")
        trial_n = item.get("trial_n")
        summary = item.get("summary")
        actionable = item.get("actionable")
        cohort = facts.has_signal(merchant, "high_risk_adult_cohort")
        ask = actionable or "Want me to pull the full item and draft something you can share?"

        # Title and the quantified finding overlap (the title is the
        # headline; the summary restates it WITH the numbers). Under budget
        # pressure the number-bearing finding is more valuable than the
        # headline, so rank it higher. The greeting is the only required
        # lead; the CTA is the only other required clause.
        title_clause = f"{title}." if title else None
        anchor_clause = None
        if trial_n and item.get("kind") == "research":
            anchor_clause = f"{facts.fmt_num(trial_n)}-patient trial."
        cohort_preamble = "Likely relevant to your high-risk adult patients." if (cohort and summary and item.get("kind") == "research") else None
        finding_clause = summary if summary else None

        return _assemble_within_budget([
            (f"{greet},", REQUIRED),
            (title_clause, 30),
            (cohort_preamble, 25),
            (finding_clause, 50),
            (anchor_clause, 20),
            (ask, REQUIRED),
            (f"— {source}" if source else None, 10),
        ])
    # No resolvable digest item — fall back to trend signal if present.
    return f"{greet}, there's a category trend worth a look this week. Want the details?"


def _compliance(category, merchant, trigger, opp: Opportunity) -> str:
    payload = trigger.get("payload") or {}
    item = facts.digest_item(category, payload.get("top_item_id"))
    greet = _greeting(category, merchant, None)
    deadline = payload.get("deadline_iso")
    if item:
        title = item.get("title")
        summary = item.get("summary")
        actionable = item.get("actionable")
        lead = f"{greet}, heads up — {title}." if title else f"{greet}, a compliance update just landed."
        deadline_clause = f"Deadline: {deadline}." if deadline else None
        actionable_clause = (actionable + ".") if actionable else None
        # lead + CTA required; summary highest-value optional, then the
        # actionable step, then the deadline restatement (the deadline often
        # already appears inside the summary/actionable, so it drops first).
        return _assemble_within_budget([
            (lead, REQUIRED),
            (summary, 50),
            (actionable_clause, 40),
            (deadline_clause, 20),
            ("Want me to check your current setup against this?", REQUIRED),
        ])
    if trigger.get("kind") == "gbp_unverified":
        return f"{greet}, your Google Business Profile is still unverified — this caps how much Vera can update automatically. Want the 2-minute verification steps?"
    return f"{greet}, there's a regulatory update relevant to your category. Want the details?"


def _customer_recall(category, merchant, trigger, customer, opp: Opportunity) -> str:
    kind = trigger.get("kind")
    if kind == "chronic_refill_due":
        return _chronic_refill(category, merchant, trigger, customer, opp)
    return _recall_or_followup(category, merchant, trigger, customer, opp)


def _chronic_refill(category, merchant, trigger, customer, opp: Opportunity) -> str:
    payload = trigger.get("payload") or {}
    use_hindi = facts.prefers_hindi_mix(merchant, customer)
    m_name = facts.merchant_name(merchant)
    c_name = facts.dig(customer, "identity", "name") or "there"
    molecules = payload.get("molecule_list") or []
    runs_out = payload.get("stock_runs_out_iso")
    delivery_saved = payload.get("delivery_address_saved")

    med_text = None
    if molecules:
        med_text = ", ".join(molecules[:3])

    runs_out_date = runs_out.split("T")[0] if isinstance(runs_out, str) else None

    if use_hindi:
        parts = [f"Hi {c_name}, {m_name} yahaan se"]
        if med_text:
            parts.append(f"aapki {med_text} ki stock")
            parts.append(f"{runs_out_date} tak khatam ho jayegi." if runs_out_date else "khatam hone wali hai.")
        else:
            parts.append("aapka refill due hone wala hai.")
        if delivery_saved:
            parts.append("Saved address par deliver kar doon?")
        else:
            parts.append("Refill schedule karoon?")
        return " ".join(parts)

    parts = [f"Hi {c_name}, {m_name} here"]
    if med_text:
        parts.append(f"— your {med_text} supply runs out" + (f" around {runs_out_date}." if runs_out_date else " soon."))
    else:
        parts.append("— your regular refill is coming due.")
    if delivery_saved:
        parts.append("Want it delivered to your saved address?")
    else:
        parts.append("Want me to schedule the refill?")
    return " ".join(parts)


def _recall_or_followup(category, merchant, trigger, customer, opp: Opportunity) -> str:
    payload = trigger.get("payload") or {}
    use_hindi = facts.prefers_hindi_mix(merchant, customer)
    m_name = facts.merchant_name(merchant)
    c_name = facts.dig(customer, "identity", "name") or "there"
    last_visit = facts.dig(customer, "relationship", "last_visit")
    service_due = payload.get("service_due")
    slots = payload.get("available_slots") or []
    offers = facts.active_offers(merchant, trigger)
    offer_title = offers[0].get("title") if offers else None
    promo_only = bool(facts.consent_scope(customer)) and not facts.has_reminder_style_consent(customer)

    slot_labels = [s.get("label") for s in slots if isinstance(s, dict) and s.get("label")]
    slots_text = None
    if slot_labels:
        if len(slot_labels) == 1:
            slots_text = slot_labels[0]
        else:
            slots_text = " or ".join([slot_labels[0], slot_labels[1]])

    due_note = service_due.replace("_", " ") if isinstance(service_due, str) else "your recall visit"

    if promo_only:
        # Customer only opted in to promotional contact, not clinical
        # reminders — reframe the same grounded facts as an offer rather
        # than a due-date nudge, staying inside the consent they gave.
        if use_hindi:
            parts = [f"Hi {c_name}, {m_name} se ek offer hai."]
            if offer_title:
                parts.append(f"{offer_title}.")
            if slots_text:
                parts.append(f"Slot available: {slots_text}.")
            parts.append("Interested ho toh reply karein.")
            return " ".join(parts)
        parts = [f"Hi {c_name}, {m_name} has an offer for you."]
        if offer_title:
            parts.append(f"{offer_title}.")
        if slots_text:
            parts.append(f"Slot available: {slots_text}.")
        parts.append("Reply YES if you're interested, or STOP to opt out.")
        return " ".join(parts)

    if use_hindi:
        parts = [f"Hi {c_name}, {m_name} yahaan se."]
        if last_visit:
            parts.append(f"Aapka {due_note} due hai (last visit {last_visit}).")
        else:
            parts.append(f"Aapka {due_note} due hai.")
        if slots_text:
            parts.append(f"Apke liye slot ready hai: {slots_text}.")
        if offer_title:
            parts.append(f"{offer_title}.")
        parts.append("Reply 1 for the first slot, 2 for the next, ya apna time bata dein.")
        return " ".join(parts)

    parts = [f"Hi {c_name}, {m_name} here."]
    if last_visit:
        parts.append(f"It's been a while since your last visit ({last_visit}) — your {due_note} is due.")
    else:
        parts.append(f"Your {due_note} is due.")
    if slots_text:
        parts.append(f"Slot options: {slots_text}.")
    if offer_title:
        parts.append(f"{offer_title}.")
    parts.append("Reply 1 or 2 for a slot, or tell us a time that works.")
    return " ".join(parts)


def _performance_negative(category, merchant, trigger, opp: Opportunity) -> str:
    perf = facts.dig(merchant, "performance", default={}) or {}
    greet = _greeting(category, merchant, None)
    calls_pct = facts.dig(perf, "delta_7d", "calls_pct")
    views_pct = facts.dig(perf, "delta_7d", "views_pct")
    ctr = facts.merchant_ctr(merchant)
    peer_ctr = facts.peer_ctr(category)
    bits = []
    if calls_pct is not None and calls_pct < 0:
        bits.append(f"calls are down {facts.fmt_pct(calls_pct)} week-over-week")
    if views_pct is not None and views_pct < 0:
        bits.append(f"views are down {facts.fmt_pct(views_pct)} week-over-week")
    if ctr is not None and peer_ctr is not None and ctr < peer_ctr:
        bits.append(f"your CTR ({facts.fmt_pct(ctr)}) is below the category median ({facts.fmt_pct(peer_ctr)})")
    if not bits:
        return f"{greet}, your account shows a performance dip this week. Want me to pull the specifics?"
    signal_text = "; ".join(bits)
    return f"{greet}, quick flag: {signal_text}. Want me to show what's driving it and one fix to try this week?"


def _performance_positive(category, merchant, trigger, opp: Opportunity) -> str:
    perf = facts.dig(merchant, "performance", default={}) or {}
    greet = _greeting(category, merchant, None)
    views_pct = facts.dig(perf, "delta_7d", "views_pct")
    milestone = _real_topic(trigger, "milestone", "metric_or_topic")
    if trigger.get("kind") == "milestone_reached" and milestone:
        return f"{greet}, milestone hit — {milestone}. Want a Google post drafted to mark it, while momentum's fresh?"
    if views_pct is not None and views_pct > 0:
        return f"{greet}, good news: views are up {facts.fmt_pct(views_pct)} this week. Want me to draft a post while the momentum's fresh, so it doesn't fade next week?"
    return f"{greet}, your numbers ticked up this week. Want the breakdown?"


def _competitive(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    payload = trigger.get("payload") or {}
    distance = payload.get("distance_km") or payload.get("distance")
    if distance:
        return f"{greet}, a new competitor opened {distance}km from you and is live on Google. Want me to check how your listing compares on the basics — photos, hours, reviews?"
    return f"{greet}, a new competitor just opened nearby and is live on Google. Want a quick side-by-side on the basics?"


def _reputation(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    theme = _real_topic(trigger, "theme", "metric_or_topic")
    if theme:
        theme_text = str(theme).replace("_", " ")
        return f"{greet}, a theme is emerging in this week's reviews: {theme_text}. Want me to draft a response template you can reuse?"
    return f"{greet}, a review theme is emerging this week. Want the summary?"


def _reengagement(category, merchant, trigger, opp: Opportunity) -> str:
    use_hindi = facts.prefers_hindi_mix(merchant)
    greet = _greeting(category, merchant, None)
    stale_days = None
    for s in facts.signals(merchant):
        if s.startswith("stale_posts"):
            stale_days = s.split(":")[-1] if ":" in s else None
    if stale_days:
        if use_hindi:
            return f"{greet}, aapka last Google post {stale_days} pehle gaya tha. 2-minute mein ek naya draft kar doon?"
        return f"{greet}, your last Google post went out {stale_days} ago. Want me to draft a fresh one — 2 minutes, your review before it's live?"
    if use_hindi:
        return f"{greet}, kuch din se baat nahi hui — sab theek chal raha hai? Kuch help chahiye toh batayein."
    return f"{greet}, it's been a bit since we last talked — everything running fine on your end? Happy to help with anything."


def _subscription(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    days_remaining = facts.dig(merchant, "subscription", "days_remaining")
    plan = facts.dig(merchant, "subscription", "plan")
    if days_remaining is not None:
        return f"{greet}, your {plan or 'plan'} has {days_remaining} days remaining. Want me to walk you through renewal so there's no gap in visibility?"
    return f"{greet}, your subscription is coming up for renewal. Want the details?"


def _operational(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    topic = _real_topic(trigger, "metric_or_topic", "item")
    if topic:
        return f"{greet}, flagging a supply/inventory item: {str(topic).replace('_', ' ')}. Want me to check current status?"
    return f"{greet}, there's an operational item worth a quick check. Want details?"


def _seasonal(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    beats = facts.dig(category, "seasonal_beats", default=[]) or []
    note = None
    for b in beats:
        if isinstance(b, dict) and b.get("note"):
            note = b["note"]
            break
    offers = facts.active_offers(merchant, trigger)
    offer_title = offers[0].get("title") if offers else None
    kind = trigger.get("kind")
    lead = {
        "festival_upcoming": "a festival is coming up",
        "ipl_match_today": "there's an IPL match today",
        "weather_heatwave": "a heatwave is forecast",
        "local_news_event": "there's a local event affecting footfall",
    }.get(kind, "a seasonal moment is coming up")
    parts = [f"{greet}, {lead}."]
    if note:
        parts.append(f"Category pattern: {note}.")
    if offer_title:
        parts.append(f"Want me to push {offer_title} for the window?")
    else:
        parts.append("Want me to draft something timely for it?")
    return " ".join(parts)


def _appointment_reminder(category, merchant, trigger, customer, opp: Opportunity) -> str:
    payload = trigger.get("payload") or {}
    m_name = facts.merchant_name(merchant)
    if customer is not None:
        use_hindi = facts.prefers_hindi_mix(merchant, customer)
        c_name = facts.dig(customer, "identity", "name") or "there"
        time_label = payload.get("time_label") or payload.get("slot_label") or payload.get("appointment_time")
        if use_hindi:
            base = f"Hi {c_name}, {m_name} yahaan se — kal ka appointment reminder"
            if time_label:
                base += f", {time_label} par"
            base += ". Confirm karenge?"
            return base
        base = f"Hi {c_name}, {m_name} here — reminder for tomorrow's appointment"
        if time_label:
            base += f" at {time_label}"
        base += ". Reply YES to confirm or STOP to cancel."
        return base
    greet = _greeting(category, merchant, None)
    return f"{greet}, you have an appointment scheduled for tomorrow. Want me to send the customer a confirmation reminder?"


def _winback(category, merchant, trigger, customer, opp: Opportunity) -> str:
    if customer is None:
        greet = _greeting(category, merchant, None)
        return f"{greet}, a set of lapsed customers is eligible for winback outreach. Want the list?"
    use_hindi = facts.prefers_hindi_mix(merchant, customer)
    c_name = facts.dig(customer, "identity", "name") or "there"
    m_name = facts.merchant_name(merchant)
    last_visit = facts.dig(customer, "relationship", "last_visit")
    offers = facts.active_offers(merchant, trigger)
    offer_title = offers[0].get("title") if offers else None
    if use_hindi:
        base = f"Hi {c_name}, {m_name} se bahut din ho gaye"
        if last_visit:
            base += f" (last visit {last_visit})"
        base += ". Aapko dekhna hai kaise miss kar rahe hain?"
        if offer_title:
            base += f" {offer_title} abhi available hai."
        return base
    base = f"Hi {c_name}, it's been a while since we've seen you at {m_name}"
    if last_visit:
        base += f" (last visit {last_visit})"
    base += "."
    if offer_title:
        base += f" {offer_title} is available if you'd like to come back."
    base += " Reply YES to book, or STOP if you'd rather not hear from us."
    return base


def _engagement_cadence(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    library = facts.dig(category, "patient_content_library", default=[]) or []
    if library:
        return f"{greet}, quick one for this week — what's the most-asked question from your customers lately? Might turn it into a shareable post for you."
    return f"{greet}, quick one — what's trending in your bookings this week? Curious what's working."


def _intent(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    intent_note = _real_topic(trigger, "metric_or_topic") or "what you asked for"
    return f"{greet}, got it — starting on {str(intent_note).replace('_', ' ')} right now. I'll confirm as soon as it's done, no further questions needed from you."


def _generic(category, merchant, trigger, opp: Opportunity) -> str:
    greet = _greeting(category, merchant, None)
    topic = _real_topic(trigger, "metric_or_topic")
    if topic:
        return f"{greet}, flagging {str(topic).replace('_', ' ')} on your account. Want the details?"
    return f"{greet}, there's a new item on your account worth a look. Want details?"


_BUILDERS = {
    "knowledge": lambda cat, m, t, c, o: _knowledge(cat, m, t, o),
    "compliance": lambda cat, m, t, c, o: _compliance(cat, m, t, o),
    "customer_recall": lambda cat, m, t, c, o: _customer_recall(cat, m, t, c, o),
    "performance_negative": lambda cat, m, t, c, o: _performance_negative(cat, m, t, o),
    "performance_positive": lambda cat, m, t, c, o: _performance_positive(cat, m, t, o),
    "competitive": lambda cat, m, t, c, o: _competitive(cat, m, t, o),
    "reputation": lambda cat, m, t, c, o: _reputation(cat, m, t, o),
    "reengagement": lambda cat, m, t, c, o: _reengagement(cat, m, t, o),
    "subscription": lambda cat, m, t, c, o: _subscription(cat, m, t, o),
    "operational": lambda cat, m, t, c, o: _operational(cat, m, t, o),
    "seasonal": lambda cat, m, t, c, o: _seasonal(cat, m, t, o),
    "winback": lambda cat, m, t, c, o: _winback(cat, m, t, c, o),
    "appointment_reminder": lambda cat, m, t, c, o: _appointment_reminder(cat, m, t, c, o),
    "engagement_cadence": lambda cat, m, t, c, o: _engagement_cadence(cat, m, t, o),
    "intent": lambda cat, m, t, c, o: _intent(cat, m, t, o),
}


def safe_fallback(category: dict, merchant: dict, trigger: dict, opp) -> ComposedMessage:
    """Last-resort composer used only when the primary draft fails
    validators.validate(). Deliberately minimal — a single sentence, no
    CTA multiplicity, nothing but the merchant's own name and the trigger
    kind, so it is *always* valid by construction."""
    greet = _greeting(category, merchant, None)
    body = f"{greet}, there's an update on your account worth a look. Want the details?"
    scope = trigger.get("scope", "merchant")
    send_as = "merchant_on_behalf" if scope == "customer" else "vera"
    rationale = (
        f"Fallback template used: primary draft for trigger '{opp.trigger_id}' "
        f"(family={opp.family}) failed validation and was discarded rather than sent."
    )
    return ComposedMessage(body=body, cta="open_ended", send_as=send_as, rationale=rationale)


def _enforce_budget_at_sentence_boundary(body: str, max_chars: int = MAX_BODY_CHARS) -> str:
    """Last-resort guarantee: if a builder somehow returns a body over the
    char budget (e.g. a future builder that doesn't use
    _assemble_within_budget), trim it back to the last COMPLETE sentence
    that fits, never mid-word. Builders that already assemble within budget
    are unaffected. This is a structural safety net, not the primary
    mechanism — the primary mechanism is clause-level assembly."""
    if len(body) <= max_chars:
        return body
    # Prefer trimming at a sentence boundary (., !, ?) at or before the cap.
    cut = body[:max_chars]
    for sep in (". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx > 0:
            return cut[: idx + 1].strip()
    # No sentence boundary found — fall back to the last whole word.
    idx = cut.rfind(" ")
    return (cut[:idx] if idx > 0 else cut).strip()


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict], opp: Opportunity) -> ComposedMessage:
    builder = _BUILDERS.get(opp.family, lambda cat, m, t, c, o: _generic(cat, m, t, o))
    body = builder(category, merchant, trigger, customer, opp)
    body = facts.strip_taboo(body, category)
    body = _enforce_budget_at_sentence_boundary(body)

    scope = trigger.get("scope", "merchant")
    send_as = "merchant_on_behalf" if scope == "customer" else "vera"

    cta = opp.expected_cta
    rationale = (
        f"{opp.reasoning}. Selected as top-ranked eligible opportunity "
        f"(score={opp.score}) among available triggers for this tick."
    )
    return ComposedMessage(body=body, cta=cta, send_as=send_as, rationale=rationale)
