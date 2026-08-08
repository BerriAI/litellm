"""
Shadow Eval Logger: pre-adoption evaluation of an auto-router against live traffic.

For each successful request on a key with an active shadow-eval job, a sampled
slice of requests is duplicated through the auto-router (the user never sees the
shadow response), an LLM judge compares the two responses blind, and the verdict
is stored stratified by the router's own tier classification.

Flow per sampled request (all in a detached background task, zero added latency):
1. Re-send the same messages through ``router.acompletion(model=<router_name>)``.
   The auto-router's pre-routing hook classifies the prompt and picks a model;
   the routing decision (tier, routed model) is read back from the request's
   metadata bucket.
2. Ask the judge model which response is better, with A/B labels randomized so
   the judge cannot learn a position bias.
3. Write a ``LiteLLM_ShadowEvalVerdict`` row and bump the job's counters.

Shadow and judge calls carry ``shadow_eval_internal`` metadata so this logger
ignores its own traffic and cannot recurse.
"""

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Optional, cast

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.types.utils import StandardLoggingPayload

_JSON_FENCE_RE: Final = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Metadata marker that tags shadow/judge calls made by this logger, so the
# success hook skips them instead of shadowing the shadow.
SHADOW_EVAL_INTERNAL_MARKER: Final = "shadow_eval_internal"

# How long the per-key active-job lookup is cached. A shadow-eval job starting
# or stopping takes up to this long to be noticed by running pods.
_JOB_CACHE_TTL_SECONDS: Final = 30.0

# Upper bound on concurrent shadow+judge pipelines per pod, so a traffic spike
# turns into skipped samples rather than an unbounded task pileup.
_MAX_CONCURRENT_SHADOW_TASKS: Final = 16

# Truncation bound for text handed to the judge, to keep judge calls affordable.
_MAX_JUDGE_CHARS: Final = 16_000

_SEEN_FLUSH_INTERVAL_SECONDS: Final = 10.0

PAIRWISE_JUDGE_SYSTEM_PROMPT: Final = """You are an impartial quality judge comparing two responses to the same conversation.

The responses are labeled A and B in random order. You do not know which system produced which.

Criteria: correctness, completeness, clarity, conciseness.

Return ONLY valid JSON in this exact format, no other text:
{
  "preference": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}"""


def _parse_pairwise_verdict(raw: str) -> dict[str, Any]:
    """Parse the judge's JSON verdict, tolerating markdown fences and surrounding prose."""
    text = raw.strip()
    fenced: Final = _JSON_FENCE_RE.search(text)
    if fenced is not None:
        text = fenced.group(1).strip()
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start: Final = text.find("{")
        end: Final = text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    return cast(dict[str, Any], parsed)  # cast-ok: narrowed to dict by the isinstance guard above


def _extract_text_from_content(content: Any) -> str:
    """Return plain text from a message content field (str or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: Final = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return ""


def _sample_hits(request_id: str, job_id: str, percentage: float) -> bool:
    """Deterministically decide whether a request falls in the shadowed slice.

    Hash-based rather than random so retries of the same request sample the
    same way and multiple pods agree without coordination.
    """
    digest: Final = hashlib.sha256(f"{job_id}:{request_id}".encode()).digest()
    bucket: Final = int.from_bytes(digest[:8], "big") / float(2**64)  # uniform [0, 1)
    return bucket * 100.0 < percentage


def _unmask_preference(raw_preference: str, real_is_a: bool) -> str:
    """Map the judge's blind A/B/tie verdict back to real/shadow/tie."""
    normalized: Final = raw_preference.strip().lower()
    if normalized == "a":
        return "real" if real_is_a else "shadow"
    if normalized == "b":
        return "shadow" if real_is_a else "real"
    return "tie"


class ShadowEvalLogger(CustomLogger):
    """Fires blind pairwise shadow evaluations for keys with an active shadow-eval job."""

    def __init__(
        self,
        router_provider: Optional[Callable[[], "Router | None"]] = None,
        prisma_provider: Optional[Callable[[], "PrismaClient | None"]] = None,
    ):
        """Providers are callables so the proxy's lazily-initialized globals are
        resolved at call time, not at logger construction."""
        self._router_provider = router_provider or _default_router_provider
        self._prisma_provider = prisma_provider or _default_prisma_provider
        # api_key_hash -> (fetched_at_monotonic, job_record_or_None)
        self._job_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self._inflight_shadow_tasks: int = 0
        self._pending_seen: dict[str, int] = {}
        self._last_seen_flush: float = 0.0

    #### hook ####

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            payload: Final[Optional["StandardLoggingPayload"]] = kwargs.get("standard_logging_object")
            if payload is None:
                return
            metadata: Final = payload.get("metadata") or {}
            request_metadata: Final = (kwargs.get("litellm_params") or {}).get("metadata") or {}
            if request_metadata.get(SHADOW_EVAL_INTERNAL_MARKER):
                return  # our own shadow/judge traffic
            api_key_hash: Final = metadata.get("user_api_key_hash")
            if not api_key_hash:
                return
            job: Final = await self._get_active_job(api_key_hash)
            if job is None:
                return
            request_id: Final = payload.get("id") or ""
            if not request_id:
                return
            # The job tracks every request it saw, sampled or not, so the UI can
            # show "N of M requests shadowed".
            self._pending_seen[job["id"]] = self._pending_seen.get(job["id"], 0) + 1
            now: Final = asyncio.get_event_loop().time()
            if now - self._last_seen_flush >= _SEEN_FLUSH_INTERVAL_SECONDS:
                self._last_seen_flush = now
                asyncio.create_task(self._flush_seen_counts())
            if not _sample_hits(request_id, job["id"], float(job["shadow_percentage"])):
                return
            if payload.get("call_type") not in (None, "completion", "acompletion", "chat_completion"):
                return  # only chat-shaped traffic is comparable
            if self._inflight_shadow_tasks >= _MAX_CONCURRENT_SHADOW_TASKS:
                return
            self._inflight_shadow_tasks += 1
            task = asyncio.create_task(
                self._run_shadow_eval(
                    job=dict(job),
                    request_id=request_id,
                    messages=list(kwargs.get("messages") or []),
                    response_obj=response_obj,
                    real_model=payload.get("model") or "",
                    model_parameters=dict(payload.get("model_parameters") or {}),
                )
            )
            task.add_done_callback(lambda _: setattr(self, "_inflight_shadow_tasks", self._inflight_shadow_tasks - 1))
        except Exception as e:  # noqa: BLE001  # logging hooks must never fail the request
            verbose_logger.debug("shadow_eval: failed to schedule task: %s", e)

    #### job lookup ####

    async def _get_active_job(self, api_key_hash: str) -> dict[str, Any] | None:
        cached: Final = self._job_cache.get(api_key_hash)
        now: Final = asyncio.get_event_loop().time()
        if cached is not None and now - cached[0] < _JOB_CACHE_TTL_SECONDS:
            return cached[1]
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return None
        job: dict[str, Any] | None = None
        try:
            record: Final = await prisma.db.litellm_shadowevaljob.find_first(
                where={"api_key_id": api_key_hash, "status": {"in": ["pending", "running"]}},
                order={"created_at": "desc"},
            )
            if record is not None:
                job = {
                    "id": record.id,
                    "router_name": record.router_name,
                    "shadow_percentage": record.shadow_percentage,
                    "judge_model": record.judge_model,
                    "status": record.status,
                }
        except Exception as e:  # noqa: BLE001  # a DB blip must not break request logging
            verbose_logger.debug("shadow_eval: job lookup failed: %s", e)
            return cached[1] if cached is not None else None
        self._job_cache[api_key_hash] = (now, job)
        return job

    async def _flush_seen_counts(self) -> None:
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        pending: Final = self._pending_seen
        self._pending_seen = {}
        for job_id, count in pending.items():
            try:
                await prisma.db.litellm_shadowevaljob.update(
                    where={"id": job_id},
                    data={"request_count": {"increment": count}},
                )
            except Exception as e:  # noqa: BLE001  # counter drift is acceptable; failing the loop is not
                verbose_logger.debug("shadow_eval: request_count flush failed: %s", e)

    #### the shadow pipeline ####

    async def _run_shadow_eval(
        self,
        job: dict[str, Any],
        request_id: str,
        messages: list[dict[str, Any]],
        response_obj: Any,
        real_model: str,
        model_parameters: dict[str, Any],
    ) -> None:
        """Detached background task: shadow call -> blind judge -> verdict row."""
        prisma: Final = self._prisma_provider()
        try:
            real_text: Final = self._extract_response_text(response_obj)
            if not real_text or not messages:
                return

            shadow = await self._call_router_shadow(job["router_name"], messages, model_parameters)
            if shadow is None:
                await self._bump_failed(job["id"])
                return
            shadow_text, shadow_model, tier, shadow_tokens = shadow

            verdict = await self._call_judge(
                judge_model=job["judge_model"],
                messages=messages,
                real_text=real_text,
                shadow_text=shadow_text,
            )
            if verdict is None:
                await self._bump_failed(job["id"])
                return
            preference, confidence, reasoning, judge_cost = verdict

            if prisma is None:
                return
            await prisma.db.litellm_shadowevalverdict.create(
                data={
                    "job_id": job["id"],
                    "request_id": request_id,
                    "tier_classification": tier,
                    "real_model": real_model,
                    "shadow_model": shadow_model,
                    "shadow_response_tokens": shadow_tokens,
                    "judge_preference": preference,
                    "judge_confidence": confidence,
                    "judge_reasoning": reasoning[:1000] if reasoning else None,
                    "judge_model": job["judge_model"],
                }
            )
            await prisma.db.litellm_shadowevaljob.update_many(
                where={"id": job["id"], "status": {"in": ["pending", "running"]}},
                data={
                    "completed_count": {"increment": 1},
                    "cost_actual": {"increment": judge_cost},
                    "status": "running",
                },
            )
        except Exception as e:  # noqa: BLE001  # detached task: log, count, never raise
            verbose_logger.debug("shadow_eval: pipeline failed for %s: %s", request_id, e)
            await self._bump_failed(job["id"])

    async def _bump_failed(self, job_id: str) -> None:
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        try:
            await prisma.db.litellm_shadowevaljob.update(
                where={"id": job_id},
                data={"failed_count": {"increment": 1}},
            )
        except Exception as e:  # noqa: BLE001  # counter drift is acceptable
            verbose_logger.debug("shadow_eval: failed_count increment failed: %s", e)

    async def _call_router_shadow(
        self, router_name: str, messages: list[dict[str, Any]], model_parameters: dict[str, Any]
    ) -> tuple[str, str, str | None, int | None] | None:
        """Send the prompt through the auto-router; return (text, model, tier, completion_tokens)."""
        router: Final = self._router_provider()
        if router is None:
            verbose_logger.debug("shadow_eval: no router available")
            return None
        # The router's pre-routing hook writes its routing decision into this
        # metadata dict; read it back after the call for tier attribution.
        shadow_metadata: dict[str, Any] = {SHADOW_EVAL_INTERNAL_MARKER: True}
        shadow_params: Final = {k: v for k, v in model_parameters.items() if k not in ("stream", "metadata")}
        try:
            response = await router.acompletion(
                model=router_name,
                messages=messages,
                metadata=shadow_metadata,
                **shadow_params,
            )
        except Exception as e:  # noqa: BLE001  # provider errors are a counted failure, not a crash
            verbose_logger.debug("shadow_eval: router call failed: %s", e)
            return None
        text: Final = self._extract_response_text(response)
        if not text:
            return None
        routing_decision: Final = shadow_metadata.get("routing_decision") or {}
        tier: Final = routing_decision.get("tier_label") or routing_decision.get("tier")
        model: Final = getattr(response, "model", None) or routing_decision.get("routed_model") or ""
        usage: Final = getattr(response, "usage", None)
        completion_tokens: Final = getattr(usage, "completion_tokens", None) if usage is not None else None
        return text, model, tier, completion_tokens

    async def _call_judge(
        self,
        judge_model: str,
        messages: list[dict[str, Any]],
        real_text: str,
        shadow_text: str,
    ) -> tuple[str, float, str, float] | None:
        """Blind pairwise judge. Returns (preference, confidence, reasoning, cost)."""
        real_is_a: Final = random.random() < 0.5
        response_a: Final = real_text if real_is_a else shadow_text
        response_b: Final = shadow_text if real_is_a else real_text

        conversation: Final = "\n".join(
            f"{m.get('role', 'user').upper()}: {_extract_text_from_content(m.get('content'))}"
            for m in messages
            if m.get("content") is not None
        )
        user_prompt: Final = (
            f"Conversation:\n{conversation[-_MAX_JUDGE_CHARS:]}\n\n"
            f"Response A:\n{response_a[:_MAX_JUDGE_CHARS]}\n\n"
            f"Response B:\n{response_b[:_MAX_JUDGE_CHARS]}\n\n"
            "Which response is better?"
        )
        try:
            response = await litellm.acompletion(
                model=judge_model,
                messages=[
                    {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=200,
                metadata={SHADOW_EVAL_INTERNAL_MARKER: True},
            )
        except Exception as e:  # noqa: BLE001  # judge outages are a counted failure, not a crash
            verbose_logger.debug("shadow_eval: judge call failed: %s", e)
            return None
        try:
            raw: Final = response["choices"][0]["message"]["content"] or ""
            verdict: Final = _parse_pairwise_verdict(raw)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as e:
            verbose_logger.debug("shadow_eval: unparseable judge verdict: %s", e)
            return None
        preference: Final = _unmask_preference(str(verdict.get("preference", "tie")), real_is_a)
        try:
            confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        reasoning: Final = str(verdict.get("reasoning", ""))
        try:
            cost = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:  # noqa: BLE001  # unmapped judge model: verdict still counts, cost stays 0
            cost = 0.0
        return preference, confidence, reasoning, cost

    @staticmethod
    def _extract_response_text(response_obj: Any) -> str:
        """Extract the assistant's text from a ModelResponse-shaped object or dict."""
        try:
            if isinstance(response_obj, dict):
                content = response_obj["choices"][0]["message"]["content"]
            else:
                content = response_obj.choices[0].message.content
        except (AttributeError, KeyError, IndexError, TypeError):
            return ""
        return _extract_text_from_content(content) if not isinstance(content, str) else content


def _default_router_provider() -> "Router | None":
    try:
        from litellm.proxy.proxy_server import llm_router
    except ImportError:
        return None
    return llm_router


def _default_prisma_provider() -> "PrismaClient | None":
    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        return None
    return prisma_client
