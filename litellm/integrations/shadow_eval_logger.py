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
ignores its own traffic and cannot recurse. They also carry the shadowed key's
identity metadata, so the provider spend they incur is attributed to that key
(and its team/org/user) and counts against every budget that key is subject to,
including the global proxy budget. A job additionally stops itself once its
sampling window (``ends_at``) closes or its judge spend reaches a multiple of
the estimate quoted when it started.
"""

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.internal_call_metadata import sanitized_forwardable_call_metadata
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN

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

# Output budget for a judge call. The judge answers with a small JSON object
# (preference + confidence + a one-sentence reasoning), which fits well inside
# this; a tighter budget truncates the JSON mid-object and the verdict is lost
# to failed_count, while a larger one buys nothing but cost exposure.
JUDGE_MAX_OUTPUT_TOKENS: Final = 500

_SEEN_FLUSH_INTERVAL_SECONDS: Final = 10.0

# A job stops sampling once its judge spend reaches this multiple of the estimate shown
# at start. The headroom absorbs an estimate that undershot the real traffic mix without
# letting the job run away; the floor keeps a cent-sized estimate from stopping a job on
# its first verdict. cost_actual is read from the job cache, so overshoot is bounded by
# _JOB_CACHE_TTL_SECONDS worth of judge calls.
_SPEND_CAP_MULTIPLIER: Final = 1.5
_SPEND_CAP_FLOOR_USD: Final = 1.0

_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})

PAIRWISE_JUDGE_SYSTEM_PROMPT: Final = """You are an impartial quality judge comparing two responses to the same conversation.

The responses are labeled A and B in random order. You do not know which system produced which.

Criteria: correctness, completeness, clarity, conciseness.

Return ONLY valid JSON in this exact format, no other text:
{
  "preference": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}"""


class PairwiseVerdict(BaseModel):
    """The judge's blind A/B verdict, validated at the parse boundary."""

    preference: str = "tie"
    confidence: float = 0.0
    reasoning: str = ""


_VERDICT_ADAPTER: Final = TypeAdapter(PairwiseVerdict)


def _parse_pairwise_verdict(raw: str) -> PairwiseVerdict:
    """Parse the judge's JSON verdict, tolerating markdown fences and surrounding prose."""
    stripped: Final = raw.strip()
    fenced: Final = _JSON_FENCE_RE.search(stripped)
    text: Final = fenced.group(1).strip() if fenced is not None else stripped
    try:
        return _VERDICT_ADAPTER.validate_json(text)
    except ValueError:
        start: Final = text.find("{")
        end: Final = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return _VERDICT_ADAPTER.validate_json(text[start : end + 1])


def _extract_text_from_content(content: object) -> str:
    """Return plain text from a message content field (str or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return ""


def _sample_hits(request_id: str, job_id: str, percentage: float) -> bool:
    """Deterministically decide whether a request falls in the shadowed slice.

    Hash-based rather than random so retries of the same request sample the
    same way and multiple pods agree without coordination.
    """
    digest: Final = hashlib.sha256(f"{job_id}:{request_id}".encode()).digest()
    bucket: Final = int.from_bytes(digest[:8], "big") / float(2**64)  # uniform [0, 1)
    return bucket * 100.0 < percentage


def _judge_call_cost(response: object) -> float:
    """Price a judge call, treating an unmapped judge model as free rather than fatal."""
    try:
        return litellm.completion_cost(completion_response=response) or 0.0
    except Exception:  # noqa: BLE001  # unmapped judge model: verdict still counts, cost stays 0
        return 0.0


def _unmask_preference(raw_preference: str, real_is_a: bool) -> str:
    """Map the judge's blind A/B/tie verdict back to real/shadow/tie."""
    normalized: Final = raw_preference.strip().lower()
    if normalized == "a":
        return "real" if real_is_a else "shadow"
    if normalized == "b":
        return "shadow" if real_is_a else "real"
    return "tie"


@dataclass(frozen=True, slots=True)
class ActiveShadowEvalJob:
    """The subset of a shadow-eval job row the request path actually needs."""

    id: str
    router_name: str
    shadow_percentage: float
    judge_model: str
    status: str
    cost_estimate: float | None = None
    cost_actual: float = 0.0
    ends_at: datetime | None = None


def _as_utc(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _job_is_past_its_end(job: ActiveShadowEvalJob) -> bool:
    """Whether the job's sampling window has closed.

    An eval is a measurement, not a service: it prices a fixed window at start
    time, so sampling past ends_at would bill traffic the estimate never covered.
    Jobs created before durations existed have no ends_at and only stop by hand
    or via the spend cap.
    """
    return job.ends_at is not None and datetime.now(timezone.utc) >= job.ends_at


def _job_is_over_spend_cap(job: ActiveShadowEvalJob) -> bool:
    """Whether the job has spent past what its start-time estimate justifies.

    Budgets bound what the *key* may spend; this bounds what a single eval may spend
    even under a generous budget, so a bad estimate or a traffic spike cannot quietly
    turn a "$3 eval" into a much larger bill. A job with no estimate (older rows,
    unpriced judge model) has nothing to be a multiple of, so it is uncapped.
    """
    if job.cost_estimate is None or job.cost_estimate <= 0.0:
        return False
    return job.cost_actual >= max(job.cost_estimate * _SPEND_CAP_MULTIPLIER, _SPEND_CAP_FLOOR_USD)


_JobCache: TypeAlias = "dict[str, tuple[float, ActiveShadowEvalJob | None]]"  # mutable-ok: TTL cache


class ShadowEvalLogger(CustomLogger):
    """Fires blind pairwise shadow evaluations for keys with an active shadow-eval job."""

    def __init__(
        self,
        router_provider: Callable[[], "Router | None"] | None = None,
        prisma_provider: Callable[[], "PrismaClient | None"] | None = None,
    ) -> None:
        """Providers are callables so the proxy's lazily-initialized globals are
        resolved at call time, not at logger construction."""
        self._router_provider = router_provider or _default_router_provider
        self._prisma_provider = prisma_provider or _default_prisma_provider
        # api_key_hash -> (fetched_at_monotonic, job_record_or_None)
        self._job_cache: _JobCache = {}  # mutable-ok: a TTL cache is mutable state by definition
        self._inflight_shadow_tasks: int = 0
        self._pending_seen: dict[str, int] = {}  # mutable-ok: flush buffer
        self._last_seen_flush: float = 0.0

    #### hook ####

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        try:
            payload: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")  # pyright: ignore[reportAssignmentType]  # untyped callback kwargs
            if payload is None:
                return
            metadata: Final = payload.get("metadata") or _EMPTY_METADATA
            litellm_params: Final = kwargs.get("litellm_params")
            raw_request_metadata: Final = (
                litellm_params.get("metadata") if isinstance(litellm_params, Mapping) else None
            )
            request_metadata: Final = (
                raw_request_metadata if isinstance(raw_request_metadata, Mapping) else _EMPTY_METADATA
            )
            if request_metadata.get(SHADOW_EVAL_INTERNAL_MARKER):
                return  # our own shadow/judge traffic
            api_key_hash: Final = metadata.get("user_api_key_hash")
            if not api_key_hash:
                return
            job: Final = await self._get_active_job(api_key_hash)
            if job is None:
                return
            if _job_is_past_its_end(job):
                await self._finalize_job(job, "reached its scheduled end")
                return
            if _job_is_over_spend_cap(job):
                await self._finalize_job(
                    job,
                    f"spend ${job.cost_actual:.4f} reached the cap for its ${job.cost_estimate or 0.0:.4f} estimate",
                )
                return
            request_id: Final = payload.get("id") or ""
            if not request_id:
                return
            # The job tracks every request it saw, sampled or not, so the UI can
            # show "N of M requests shadowed".
            self._pending_seen[job.id] = self._pending_seen.get(job.id, 0) + 1
            now: Final = asyncio.get_event_loop().time()
            if now - self._last_seen_flush >= _SEEN_FLUSH_INTERVAL_SECONDS:
                self._last_seen_flush = now
                asyncio.create_task(self._flush_seen_counts())
            if not _sample_hits(request_id, job.id, job.shadow_percentage):
                return
            if payload.get("call_type") not in (None, "completion", "acompletion", "chat_completion"):
                return  # only chat-shaped traffic is comparable
            if self._inflight_shadow_tasks >= _MAX_CONCURRENT_SHADOW_TASKS:
                return
            raw_messages: Final = kwargs.get("messages")
            self._inflight_shadow_tasks += 1
            task: Final = asyncio.create_task(
                self._run_shadow_eval(
                    job=job,
                    request_id=request_id,
                    messages=tuple(m for m in raw_messages if isinstance(m, Mapping))
                    if isinstance(raw_messages, Sequence)
                    else (),
                    response_obj=response_obj,
                    real_model=payload.get("model") or "",
                    model_parameters=MappingProxyType(
                        dict(payload.get("model_parameters") or {})  # mutable-ok: frozen snapshot
                    ),
                    parent_metadata=MappingProxyType(dict(request_metadata)),  # mutable-ok: frozen snapshot
                )
            )
            task.add_done_callback(lambda _: setattr(self, "_inflight_shadow_tasks", self._inflight_shadow_tasks - 1))
        except Exception as e:  # noqa: BLE001  # logging hooks must never fail the request
            verbose_logger.debug("shadow_eval: failed to schedule task: %s", e)

    #### job lookup ####

    async def _get_active_job(self, api_key_hash: str) -> ActiveShadowEvalJob | None:
        cached: Final = self._job_cache.get(api_key_hash)
        now: Final = asyncio.get_event_loop().time()
        if cached is not None and now - cached[0] < _JOB_CACHE_TTL_SECONDS:
            return cached[1]
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return None
        try:
            record: Final = await prisma.db.litellm_shadowevaljob.find_first(
                where={  # mutable-ok: Prisma filter
                    "api_key_id": api_key_hash,
                    "status": {"in": ["pending", "running"]},  # mutable-ok: Prisma filter
                },
                order={"created_at": "desc"},  # mutable-ok: Prisma order
            )
            job: Final = (
                ActiveShadowEvalJob(
                    id=str(record.id),
                    router_name=str(record.router_name),
                    shadow_percentage=float(record.shadow_percentage),
                    judge_model=str(record.judge_model),
                    status=str(record.status),
                    cost_estimate=float(record.cost_estimate) if record.cost_estimate is not None else None,
                    cost_actual=float(record.cost_actual or 0.0),
                    ends_at=_as_utc(getattr(record, "ends_at", None)),
                )
                if record is not None
                else None
            )
        except Exception as e:  # noqa: BLE001  # a DB blip must not break request logging
            verbose_logger.debug("shadow_eval: job lookup failed: %s", e)
            return cached[1] if cached is not None else None
        self._job_cache[api_key_hash] = (now, job)
        return job

    async def _finalize_job(self, job: ActiveShadowEvalJob, reason: str) -> None:
        """Flip a finished job to completed, keeping the verdicts it already produced.

        Guarded on the job still being pending/running so two pods finishing the
        same job cannot resurrect one an admin stopped in between.
        """
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        self._job_cache = {  # mutable-ok: TTL cache
            k: v for k, v in self._job_cache.items() if v[1] is None or v[1].id != job.id
        }
        verbose_logger.info("shadow_eval: stopping job %s: %s", job.id, reason)
        try:
            await prisma.db.litellm_shadowevaljob.update_many(
                where={  # mutable-ok: Prisma filter
                    "id": job.id,
                    "status": {"in": ["pending", "running"]},  # mutable-ok: Prisma filter
                },
                data={  # mutable-ok: Prisma payload
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc),
                },
            )
        except Exception as e:  # noqa: BLE001  # logging hooks must never fail the request
            verbose_logger.debug("shadow_eval: failed to stop job %s: %s", job.id, e)

    async def _flush_seen_counts(self) -> None:
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        pending: Final = self._pending_seen
        self._pending_seen = {}  # mutable-ok: fresh flush buffer
        for job_id, count in pending.items():
            try:
                await prisma.db.litellm_shadowevaljob.update(
                    where={"id": job_id},  # mutable-ok: Prisma filter
                    data={"request_count": {"increment": count}},  # mutable-ok: Prisma payload
                )
            except Exception as e:  # noqa: BLE001  # counter drift is acceptable; failing the loop is not
                verbose_logger.debug("shadow_eval: request_count flush failed: %s", e)

    #### the shadow pipeline ####

    async def _run_shadow_eval(
        self,
        job: ActiveShadowEvalJob,
        request_id: str,
        messages: Sequence[Mapping[str, object]],
        response_obj: object,
        real_model: str,
        model_parameters: Mapping[str, object],
        parent_metadata: Mapping[str, object],
    ) -> None:
        """Detached background task: shadow call -> blind judge -> verdict row."""
        prisma: Final = self._prisma_provider()
        try:
            real_text: Final = self._extract_response_text(response_obj)
            if not real_text or not messages:
                return

            shadow: Final = await self._call_router_shadow(job.router_name, messages, model_parameters, parent_metadata)
            if shadow is None:
                await self._bump_failed(job.id)
                return
            shadow_text, shadow_model, tier, shadow_tokens = shadow

            verdict: Final = await self._call_judge(
                judge_model=job.judge_model,
                messages=messages,
                real_text=real_text,
                shadow_text=shadow_text,
                parent_metadata=parent_metadata,
            )
            if verdict is None:
                await self._bump_failed(job.id)
                return
            preference, confidence, reasoning, judge_cost = verdict

            if prisma is None:
                return
            await prisma.db.litellm_shadowevalverdict.create(
                data={  # mutable-ok: Prisma payload
                    "job_id": job.id,
                    "request_id": request_id,
                    "tier_classification": tier,
                    "real_model": real_model,
                    "shadow_model": shadow_model,
                    "shadow_response_tokens": shadow_tokens,
                    "judge_preference": preference,
                    "judge_confidence": confidence,
                    "judge_reasoning": reasoning[:1000] if reasoning else None,
                    "judge_model": job.judge_model,
                }
            )
            await prisma.db.litellm_shadowevaljob.update_many(
                where={  # mutable-ok: Prisma filter
                    "id": job.id,
                    "status": {"in": ["pending", "running"]},  # mutable-ok: Prisma filter
                },
                data={  # mutable-ok: Prisma payload
                    "completed_count": {"increment": 1},  # mutable-ok: Prisma operator
                    "cost_actual": {"increment": judge_cost},  # mutable-ok: Prisma operator
                    "status": "running",
                },
            )
        except Exception as e:  # noqa: BLE001  # detached task: log, count, never raise
            verbose_logger.debug("shadow_eval: pipeline failed for %s: %s", request_id, e)
            await self._bump_failed(job.id)

    async def _bump_failed(self, job_id: str) -> None:
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        try:
            await prisma.db.litellm_shadowevaljob.update(
                where={"id": job_id},  # mutable-ok: Prisma filter
                data={"failed_count": {"increment": 1}},  # mutable-ok: Prisma payload
            )
        except Exception as e:  # noqa: BLE001  # counter drift is acceptable
            verbose_logger.debug("shadow_eval: failed_count increment failed: %s", e)

    async def _call_router_shadow(
        self,
        router_name: str,
        messages: Sequence[Mapping[str, object]],
        model_parameters: Mapping[str, object],
        parent_metadata: Mapping[str, object],
    ) -> tuple[str, str, str | None, int | None] | None:
        """Send the prompt through the auto-router; return (text, model, tier, completion_tokens)."""
        router: Final = self._router_provider()
        if router is None:
            verbose_logger.debug("shadow_eval: no router available")
            return None
        # Carries the shadowed key's identity so this call's real provider spend is
        # attributed to, and budget-checked against, the key whose traffic it copies.
        # The router's pre-routing hook also writes its routing decision into this dict;
        # read it back after the call for tier attribution.
        attribution: Final = sanitized_forwardable_call_metadata(parent_metadata, SHADOW_EVAL_ROUTER_CALL_ORIGIN)
        shadow_metadata: Final[dict[str, object]] = attribution | {  # mutable-ok: router writes back
            SHADOW_EVAL_INTERNAL_MARKER: True
        }
        shadow_params: Final = {  # mutable-ok: splatted as kwargs
            k: v for k, v in model_parameters.items() if k not in ("stream", "metadata")
        }
        try:
            response: Final = await router.acompletion(
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
        raw_decision: Final = shadow_metadata.get("routing_decision")
        routing_decision: Final = raw_decision if isinstance(raw_decision, Mapping) else _EMPTY_METADATA
        raw_tier: Final = routing_decision.get("tier_label") or routing_decision.get("tier")
        tier: Final = str(raw_tier) if raw_tier is not None else None
        model: Final = str(getattr(response, "model", None) or routing_decision.get("routed_model") or "")
        usage: Final = getattr(response, "usage", None)
        raw_tokens: Final = getattr(usage, "completion_tokens", None) if usage is not None else None
        completion_tokens: Final = int(raw_tokens) if isinstance(raw_tokens, int) else None
        return text, model, tier, completion_tokens

    async def _call_judge(
        self,
        judge_model: str,
        messages: Sequence[Mapping[str, object]],
        real_text: str,
        shadow_text: str,
        parent_metadata: Mapping[str, object],
    ) -> tuple[str, float, str, float] | None:
        """Blind pairwise judge. Returns (preference, confidence, reasoning, cost)."""
        real_is_a: Final = random.random() < 0.5
        response_a: Final = real_text if real_is_a else shadow_text
        response_b: Final = shadow_text if real_is_a else real_text

        conversation: Final = "\n".join(
            f"{str(m.get('role', 'user')).upper()}: {_extract_text_from_content(m.get('content'))}"
            for m in messages
            if m.get("content") is not None
        )
        user_prompt: Final = (
            f"Conversation:\n{conversation[-_MAX_JUDGE_CHARS:]}\n\n"
            f"Response A:\n{response_a[:_MAX_JUDGE_CHARS]}\n\n"
            f"Response B:\n{response_b[:_MAX_JUDGE_CHARS]}\n\n"
            "Which response is better?"
        )
        judge_metadata: Final = sanitized_forwardable_call_metadata(parent_metadata, SHADOW_EVAL_JUDGE_CALL_ORIGIN) | {
            SHADOW_EVAL_INTERNAL_MARKER: True
        }
        try:
            response: Final = await litellm.acompletion(
                model=judge_model,
                messages=[  # mutable-ok: SDK takes a list
                    {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM_PROMPT},  # mutable-ok: SDK message
                    {"role": "user", "content": user_prompt},  # mutable-ok: SDK message
                ],
                temperature=0,
                max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                metadata=judge_metadata,
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
        preference: Final = _unmask_preference(verdict.preference, real_is_a)
        confidence: Final = max(0.0, min(1.0, verdict.confidence))
        return preference, confidence, verdict.reasoning, _judge_call_cost(response)

    @staticmethod
    def _extract_response_text(response_obj: object) -> str:
        """Extract the assistant's text from a ModelResponse-shaped object or dict."""
        try:
            content: Final = (
                response_obj["choices"][0]["message"]["content"]
                if isinstance(response_obj, Mapping)
                else response_obj.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]  # duck-typed ModelResponse
            )
        except (AttributeError, KeyError, IndexError, TypeError):
            return ""
        return _extract_text_from_content(content)


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
