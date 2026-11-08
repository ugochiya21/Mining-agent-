"""v1 Agent 2 — upgraded reasoning + clamp + cross-provider redundancy.

One web-search call (targeted query built from the event's full metadata),
then an LLM synthesis call using a stronger default model and a stronger,
calibration-focused prompt.

NOTE ON THE CAPABILITY LADDER: this file started as "BASIC search + BASIC
LLM" (Agent 2) in a ladder meant to isolate the effect of evidence quantity
alone, with every rung sharing the same LLM tier. Upgrading the default model
here breaks that apples-to-apples comparison with the other ladder agents. If
you still want the ladder intact for analysis, keep an unmodified copy of the
original basic version elsewhere.

v1.4.0 additions (on top of v1.3.0's clamp + redundancy):
- Upgraded default reasoning model: GOOD_LLM (claude-sonnet-4-6) is now tried
  FIRST instead of BASIC_LLM. This costs more credits per successful
  prediction than gpt-4o-mini did — worth watching against your balance,
  especially since credit exhaustion was already flagged as a likely cause
  of intermittent rank drops.
- Stronger, calibration-focused system prompt: explicitly walks the model
  through base rate -> evidence for -> evidence against -> final probability,
  and explicitly instructs it to name when the research context is thin or
  generic rather than inventing specifics. This directly targets the
  no-citation, possibly-ungrounded reasoning pattern seen in earlier traces
  (e.g. the BSW/Grüne markets, where predictions diverged sharply from
  market price with no visible sourcing).

Unchanged from v1.3.0:
- Extreme-probability clamp: final `prediction` floored at 0.02 / capped at
  0.98 — a confident 0/1 miss craters a Brier score far worse than a hedged
  wrong answer.
- Search redundancy (within OpenRouter only — web_search() has no
  provider_id hook, and neither Engy nor SayGM exposes a search-capable
  model).
- Forecast redundancy across three independent providers, so a single
  rate-limited model or a whole-provider outage doesn't stop the agent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType
from src.agent.examples._v1_common import (
    AMAZING_SEARCH,
    BASIC_LLM,
    BASIC_SEARCH,
    GOOD_LLM,
    JSON_SYSTEM_PROMPT,
    Forecast,
    belief_path_from_forecast,
    belief_path_single_final,
    build_json_prompt,
    request_json,
    web_search,
)

# Never claim certainty. A single confident 0/1 miss craters a Brier score far
# more than a hedged-but-wrong prediction does — this is the cheapest
# protection an agent can add.
PROBABILITY_FLOOR = 0.02
PROBABILITY_CAP = 0.98

# Search stays within OpenRouter (see module docstring for why).
SEARCH_MODEL_CHAIN = (BASIC_SEARCH, AMAZING_SEARCH)

# Stronger calibration prompt, layered on top of the shared JSON contract
# instructions so parsing/validation (Forecast, parse_forecast) is unaffected —
# only the reasoning QUALITY guidance changes.
CALIBRATION_ADDENDUM = """

Before answering, work through this checklist silently, then compress your
conclusion into the required 2-4 sentence reasoning:
1. Base rate: what typically happens in situations like this (same category,
   similar structure), before looking at any specific evidence?
2. Evidence FOR yes: what SPECIFIC, dated facts from the research context
   support it? If the research context is thin, generic, or empty, say so
   explicitly rather than inventing specifics that sound plausible.
3. Evidence FOR no: same standard — specific and dated, or explicitly flagged
   as absent.
4. Combine 1-3 into a probability. Resist defaulting to a round, intuitive
   number (0.5, 0.7, 0.9) just because it feels right — a more precise value
   (e.g. 0.58, 0.63, 0.71) is often the more honest answer.
5. Do not imply near-certainty (a probability close to 0 or 1) unless the
   evidence is genuinely overwhelming and directly on-point. Absence of
   strong evidence means the answer should stay closer to the base rate, not
   drift to an extreme."""

STRONG_JSON_SYSTEM_PROMPT = JSON_SYSTEM_PROMPT + CALIBRATION_ADDENDUM

# Forecast redundancy: (model, provider_id, max_tokens) tried in order.
# GOOD_LLM first (upgraded default) -> Engy -> SayGM (independent providers,
# so an OpenRouter-wide outage doesn't block the agent) -> BASIC_LLM last as
# a cheap final fallback so the agent still answers even if every stronger
# option is unavailable.
ENGY_LLM = "qwen3.6-35b-a3b"
ENGY_PROVIDER = "engy"
SAYGM_LLM = "glm-5.2"
SAYGM_PROVIDER = "saygm"

FORECAST_MODEL_CHAIN = (
    (GOOD_LLM, "openrouter", 1200),
    (ENGY_LLM, ENGY_PROVIDER, 2000),
    (SAYGM_LLM, SAYGM_PROVIDER, 2000),
    (BASIC_LLM, "openrouter", 900),
)


def _clamp_extreme(p: float, floor: float = PROBABILITY_FLOOR, cap: float = PROBABILITY_CAP) -> float:
    """Clamp a probability away from the extremes. Applied to the final
    prediction only — never let the agent output a raw 0 or 1."""
    return max(floor, min(cap, p))


def _search_with_fallback(ctx: ForecastingContext, query: str, *, max_tokens: int = 900) -> tuple[str, str | None]:
    """Try each model in SEARCH_MODEL_CHAIN until one returns usable text."""
    for model in SEARCH_MODEL_CHAIN:
        try:
            result = web_search(ctx, model, query, max_tokens=max_tokens)
        except Exception:
            result = ""
        if result and result.strip():
            return result, model
    return "", None


def _request_forecast_strong(
    ctx: ForecastingContext, model: str, title: str, description: str,
    context: str | None, *, provider_id: str, max_tokens: int,
) -> Forecast:
    """Like request_forecast(), but uses STRONG_JSON_SYSTEM_PROMPT instead of
    the plain JSON_SYSTEM_PROMPT. build_json_prompt() (the user-message
    formatting) is reused unchanged — only the system-prompt guidance differs.
    """
    messages = [
        {"role": "system", "content": STRONG_JSON_SYSTEM_PROMPT},
        {"role": "user", "content": build_json_prompt(title, description, context)},
    ]
    return request_json(
        ctx, model, messages, max_retries=1, max_tokens=max_tokens, provider_id=provider_id,
    )


def _forecast_with_fallback(
    ctx: ForecastingContext, title: str, description: str, context: str | None,
) -> tuple[Forecast, str, str]:
    """Try each (model, provider_id, max_tokens) in FORECAST_MODEL_CHAIN, using
    the stronger calibration prompt, until one produces a valid Forecast.

    Returns (forecast, model_used, provider_used). Raises ValueError only if
    every provider in the chain fails.
    """
    last_exc: Exception | None = None
    for model, provider_id, max_tokens in FORECAST_MODEL_CHAIN:
        try:
            fc = _request_forecast_strong(
                ctx, model, title, description, context,
                provider_id=provider_id, max_tokens=max_tokens,
            )
            return fc, model, provider_id
        except ValueError as exc:
            last_exc = exc
            continue
    raise ValueError(f"all forecast providers failed; last error: {last_exc}")


class V1Agent2Basic(BaseAgent):
    agent_id = UUID("a0000002-0000-4000-8000-000000000002")
    agent_version = "1.4.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        event = ctx.event
        now = datetime.now(timezone.utc)
        deadline = event.resolution_deadline
        # Guard against a naive resolution_deadline (no tzinfo) so this never
        # raises TypeError when subtracted from an aware `now`.
        if deadline is not None and deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        days_left = (deadline - now).days if deadline else None

        # --- Phase 1: web search, with within-OpenRouter fallback -------
        criteria_bit = f" Resolution criteria: {event.resolution_criteria}." if event.resolution_criteria else ""
        deadline_bit = (
            f" This resolves by {deadline.date().isoformat()} "
            f"({days_left} days from today, {now.date().isoformat()})."
            if deadline else ""
        )
        tags_bit = f" Tags: {', '.join(event.tags)}." if event.tags else ""

        query = (
            f"Current status and most recent developments (last 30 days if possible) "
            f"relevant to this forecasting question: {ctx.event_title}. "
            f"{ctx.event_description}{criteria_bit}{deadline_bit}{tags_bit} "
            f"Category: {event.category.value}. "
            f"Include concrete dated facts, official statements, polls, or odds "
            f"if available, with sources."
        )
        research, search_model_used = _search_with_fallback(ctx, query, max_tokens=900)

        ctx.record_reasoning_step(
            ReasoningStepType.GAP_QUERY,
            reasoning_text=(
                "Ran a web search with within-OpenRouter model fallback "
                f"(deadline-aware, {days_left if days_left is not None else 'unknown'} days out). "
                f"Model used: {search_model_used or 'none — all search models failed/empty'}.\n"
                f"Findings: {research[:1200] if research else '(no results — search returned empty)'}"
            ),
            provider_id="openrouter",
            inference_model_used=search_model_used or SEARCH_MODEL_CHAIN[0],
        )

        # --- Phase 2: forecast synthesis, upgraded model + stronger prompt,
        #              with cross-provider fallback --------------------
        enriched_description = (
            f"{ctx.event_description}\n\n"
            f"Category: {event.category.value}"
            + (f"\nResolution criteria: {event.resolution_criteria}" if event.resolution_criteria else "")
            + (f"\nResolution deadline: {deadline.date().isoformat()} "
               f"({days_left} days from today, {now.date().isoformat()})" if deadline else "")
            + (f"\nTags: {', '.join(event.tags)}" if event.tags else "")
        )

        try:
            fc, forecast_model_used, forecast_provider_used = _forecast_with_fallback(
                ctx, ctx.event_title, enriched_description, context=research,
            )
        except ValueError as exc:
            reason = f"fail-closed neutral forecast (all forecast providers failed): {exc}"
            return AgentResult(
                prediction=0.5, confidence=None, reasoning=reason,
                beliefPath=belief_path_single_final(0.5, reason),
            )

        # Never output a raw 0 or 1 — clamp before building the belief path so
        # AgentResult's "final belief-path probability must equal prediction"
        # check stays satisfied.
        clamped_prediction = _clamp_extreme(fc.prediction)
        fc = fc.model_copy(update={"prediction": clamped_prediction})

        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=fc.reasoning,
            intermediate_probability=fc.prediction,
            provider_id=forecast_provider_used,
            inference_model_used=forecast_model_used,
        )
        return AgentResult(
            prediction=fc.prediction, confidence=fc.confidence,
            reasoning=fc.reasoning, beliefPath=belief_path_from_forecast(fc),
        )
