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

Shadow and judge calls carry an ``internal_call_origin`` stamp, the same field
every internal sub-call (including the auto-router's own classifier) is marked
with, and the logger skips any request carrying it: its own traffic cannot
recurse, and internal sub-calls billed to the shadowed key are never judged as
if a user sent them. The calls also carry the shadowed key's identity metadata,
so the provider spend they incur is attributed to that key (and its team/org/
user) and counts against every budget that key is subject to, including the
global proxy budget. A shadow/judge pair is skipped outright if the shadowed
key or its team is already at or over budget, read from the same cross-pod
spend counters the normal auth path reserves against.

Job lifecycle (counter flushes, snapshot refresh, stopping a job at its
``ends_at`` or judge-spend cap) runs on a periodic loop started at proxy
startup, never on the request path: an idle key's job still ends on schedule,
and the final counter batch lands without needing another request to arrive.
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
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.internal_call_metadata import sanitized_forwardable_call_metadata
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.types.utils import StandardLoggingPayload

_JSON_FENCE_RE: Final = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Cadence of the lifecycle loop: snapshot refresh, counter flush, and job
# finalization all run on this tick. A job starting or stopping takes up to
# one tick to be noticed by running pods.
_LIFECYCLE_TICK_SECONDS: Final = 10.0

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

# Persisted diagnosis bound: the most recent shadow/judge failure, truncated.
_MAX_LAST_ERROR_CHARS: Final = 500

# A job stops sampling once its judge spend reaches this multiple of the estimate shown
# at start. The headroom absorbs an estimate that undershot the real traffic mix without
# letting the job run away; the floor keeps a cent-sized estimate from stopping a job on
# its first verdict. cost_actual is read from the job snapshot, so overshoot is bounded
# by _LIFECYCLE_TICK_SECONDS worth of judge calls.
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
class _CallFailure:
    """A shadow or judge call that produced no usable response, with why."""

    error: str


@dataclass(frozen=True, slots=True)
class _ShadowResponse:
    """A successful shadow call: (text, routed model, tier, completion tokens)."""

    text: str
    model: str
    tier: str | None
    completion_tokens: int | None


@dataclass(frozen=True, slots=True)
class _JudgeVerdict:
    """A parsed judge verdict, unmasked back to real/shadow/tie."""

    preference: str
    confidence: float
    reasoning: str
    cost: float


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


async def _key_or_team_is_over_budget(metadata: Mapping[str, object]) -> bool:
    """Whether the shadowed key or its team has no budget left for the shadow/judge calls.

    These calls run outside the normal auth path (no /key/generate-issued request
    reaches user_api_key_auth for them), so they never go through
    reserve_budget_for_request. Reading the same cross-pod counters that path
    reserves against, at read time, stops a key or team that is already out of
    budget from having a shadow eval push it further over: not a hard reservation
    (a burst of concurrent requests could still all pass this read), but it closes
    the gap for the common case a background quality-measurement task should
    never make worse.
    """
    try:
        from litellm.proxy.proxy_server import get_current_spend
    except ImportError:
        return False

    api_key_hash: Final = metadata.get("user_api_key_hash")
    max_budget: Final = metadata.get("user_api_key_max_budget")
    if isinstance(api_key_hash, str) and isinstance(max_budget, (int, float)) and max_budget > 0:
        spend: Final = await get_current_spend(
            counter_key=f"spend:key:{api_key_hash}",
            fallback_spend=float(metadata.get("user_api_key_spend") or 0.0),
            max_budget=float(max_budget),
        )
        if spend >= max_budget:
            return True

    team_id: Final = metadata.get("user_api_key_team_id")
    team_max_budget: Final = metadata.get("user_api_key_team_max_budget")
    if isinstance(team_id, str) and isinstance(team_max_budget, (int, float)) and team_max_budget > 0:
        team_spend: Final = await get_current_spend(
            counter_key=f"spend:team:{team_id}",
            fallback_spend=float(metadata.get("user_api_key_team_spend") or 0.0),
            max_budget=float(team_max_budget),
        )
        if team_spend >= team_max_budget:
            return True

    return False


def _request_was_routed_by(request_metadata: Mapping[str, object], router_name: str) -> bool:
    """Whether this request was already served by the router being shadowed.

    Duplicating such a request compares the router to itself: guaranteed ties, judge
    spend for zero information. Requests routed by a *different* auto-router still
    sample, which is a meaningful router-vs-router comparison.
    """
    decision: Final = request_metadata.get("routing_decision")
    if not isinstance(decision, Mapping):
        return False
    return decision.get("router_model_name") == router_name


def _job_is_over_spend_cap(job: ActiveShadowEvalJob) -> bool:
    """Whether the job has spent past what its start-time estimate justifies.

    Budgets bound what the *key* may spend; this bounds what a single eval may spend
    even under a generous budget, so a bad estimate or a traffic spike cannot quietly
    turn a "$3 eval" into a much larger bill. Only a job with no estimate at all
    (rows created before estimates existed) is uncapped; a $0 estimate — a key that
    was quiet during the lookback — still gets the floor, since a later traffic spike
    on exactly such a job is the scenario the cap exists for.
    """
    if job.cost_estimate is None:
        return False
    return job.cost_actual >= max(job.cost_estimate * _SPEND_CAP_MULTIPLIER, _SPEND_CAP_FLOOR_USD)


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
        # Snapshot of every active job, keyed by shadowed api_key_id. Refreshed as a
        # whole by the lifecycle loop: active jobs are admin-started and capped at one
        # per key, so the set is tiny, and one find_many per pod per tick keeps DB load
        # flat no matter how many distinct keys the proxy serves.
        self._jobs_by_key: dict[str, ActiveShadowEvalJob] = {}  # mutable-ok: loop-refreshed snapshot
        self._inflight_shadow_tasks: int = 0
        self._pending_seen: dict[str, int] = {}  # mutable-ok: flush buffer
        self._lifecycle_task: asyncio.Task[None] | None = None

    def start_lifecycle_loop(self) -> None:
        """Idempotently start the periodic loop that owns job lifecycle off the request path."""
        if self._lifecycle_task is not None and not self._lifecycle_task.done():
            return
        self._lifecycle_task = asyncio.create_task(self._lifecycle_loop())

    async def _lifecycle_loop(self) -> None:
        while True:
            try:
                await self._lifecycle_tick()
            except Exception as e:  # noqa: BLE001  # the loop must survive any single tick failing
                verbose_logger.debug("shadow_eval: lifecycle tick failed: %s", e)
            await asyncio.sleep(_LIFECYCLE_TICK_SECONDS)

    async def _lifecycle_tick(self) -> None:
        """One tick: flush counters while jobs are still active, refresh, then finalize.

        Flushing before finalizing lets an expiring job's last counter batch land while
        its row still passes the active-status guard.
        """
        await self._flush_seen_counts()
        await self._refresh_active_jobs()
        for job in tuple(self._jobs_by_key.values()):
            if _job_is_past_its_end(job):
                await self._finalize_job(job, "reached its scheduled end")
            elif _job_is_over_spend_cap(job):
                await self._finalize_job(
                    job,
                    f"spend ${job.cost_actual:.4f} reached the cap for its ${job.cost_estimate or 0.0:.4f} estimate",
                )

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
            if request_metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
                return  # internal sub-call (our own shadow/judge, a classifier, ...), not user traffic
            api_key_hash: Final = metadata.get("user_api_key_hash")
            if not api_key_hash:
                return
            job: Final = self._jobs_by_key.get(str(api_key_hash))
            if job is None:
                return
            if _job_is_past_its_end(job) or _job_is_over_spend_cap(job):
                return  # stop sampling now; the lifecycle loop finalizes the row
            request_id: Final = payload.get("id") or ""
            if not request_id:
                return
            # The job tracks every request it saw, sampled or not, so the UI can
            # show "N of M requests shadowed". Flushed by the lifecycle loop.
            self._pending_seen[job.id] = self._pending_seen.get(job.id, 0) + 1
            if not _sample_hits(request_id, job.id, job.shadow_percentage):
                return
            if payload.get("call_type") not in (None, "completion", "acompletion", "chat_completion"):
                return  # only chat-shaped traffic is comparable
            if _request_was_routed_by(request_metadata, job.router_name):
                return
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
                    budget_metadata=MappingProxyType(dict(metadata)),  # mutable-ok: frozen snapshot
                )
            )
            task.add_done_callback(lambda _: setattr(self, "_inflight_shadow_tasks", self._inflight_shadow_tasks - 1))
        except Exception as e:  # noqa: BLE001  # logging hooks must never fail the request
            verbose_logger.debug("shadow_eval: failed to schedule task: %s", e)

    #### job lookup ####

    async def _refresh_active_jobs(self) -> None:
        """Reload the active-job set. On a DB blip the stale snapshot is kept and the
        next tick retries, so a blip degrades freshness rather than turning the feature off."""
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        try:
            records: Final = await prisma.db.litellm_shadowevaljob.find_many(
                where={"status": {"in": ["pending", "running"]}},  # mutable-ok: Prisma filter
                order={"created_at": "desc"},  # mutable-ok: Prisma order
            )
        except Exception as e:  # noqa: BLE001  # a DB blip must not break request logging
            verbose_logger.debug("shadow_eval: active-job refresh failed: %s", e)
            return
        jobs_by_key: Final[dict[str, ActiveShadowEvalJob]] = {}  # mutable-ok: building the new snapshot
        for record in reversed(records or []):
            jobs_by_key[str(record.api_key_id)] = ActiveShadowEvalJob(
                id=str(record.id),
                router_name=str(record.router_name),
                shadow_percentage=float(record.shadow_percentage),
                judge_model=str(record.judge_model),
                status=str(record.status),
                cost_estimate=float(record.cost_estimate) if record.cost_estimate is not None else None,
                cost_actual=float(record.cost_actual or 0.0),
                ends_at=_as_utc(getattr(record, "ends_at", None)),
            )
        self._jobs_by_key = jobs_by_key  # mutable-ok: atomic snapshot swap

    async def _finalize_job(self, job: ActiveShadowEvalJob, reason: str) -> None:
        """Flip a finished job to completed, keeping the verdicts it already produced.

        Guarded on the job still being pending/running so two pods finishing the
        same job cannot resurrect one an admin stopped in between.
        """
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        self._jobs_by_key = {  # mutable-ok: atomic snapshot swap
            k: v for k, v in self._jobs_by_key.items() if v.id != job.id
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
        """Write the buffered request counts, guarded on the job still being active.

        The guard makes stopping a job freeze its counter: a pod serving a stale
        snapshot keeps buffering for up to one tick, and this write drops those
        increments instead of growing a stopped job's request_count.
        """
        prisma: Final = self._prisma_provider()
        if prisma is None or not self._pending_seen:
            return
        pending: Final = self._pending_seen
        self._pending_seen = {}  # mutable-ok: fresh flush buffer
        for job_id, count in pending.items():
            try:
                await prisma.db.litellm_shadowevaljob.update_many(
                    where={  # mutable-ok: Prisma filter
                        "id": job_id,
                        "status": {"in": ["pending", "running"]},  # mutable-ok: Prisma filter
                    },
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
        budget_metadata: Mapping[str, object] = _EMPTY_METADATA,
    ) -> None:
        """Detached background task: budget gate -> shadow call -> blind judge -> verdict row.

        The budget read lives here, not in the success hook: get_current_spend can fall
        back to an authoritative DB read, which a detached task absorbs and the
        production callback must not.
        """
        prisma: Final = self._prisma_provider()
        try:
            real_text: Final = self._extract_response_text(response_obj)
            if not real_text or not messages:
                return
            if await _key_or_team_is_over_budget(budget_metadata):
                return

            shadow: Final = await self._call_router_shadow(job.router_name, messages, model_parameters, parent_metadata)
            if isinstance(shadow, _CallFailure):
                await self._bump_failed(job.id, shadow.error)
                return

            verdict: Final = await self._call_judge(
                judge_model=job.judge_model,
                messages=messages,
                real_text=real_text,
                shadow_text=shadow.text,
                parent_metadata=parent_metadata,
            )
            if isinstance(verdict, _CallFailure):
                await self._bump_failed(job.id, verdict.error)
                return

            if prisma is None:
                return
            await prisma.db.litellm_shadowevalverdict.create(
                data={  # mutable-ok: Prisma payload
                    "job_id": job.id,
                    "request_id": request_id,
                    "tier_classification": shadow.tier,
                    "real_model": real_model,
                    "shadow_model": shadow.model,
                    "shadow_response_tokens": shadow.completion_tokens,
                    "judge_preference": verdict.preference,
                    "judge_confidence": verdict.confidence,
                    "judge_reasoning": verdict.reasoning[:1000] if verdict.reasoning else None,
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
                    "cost_actual": {"increment": verdict.cost},  # mutable-ok: Prisma operator
                    "status": "running",
                },
            )
        except Exception as e:  # noqa: BLE001  # detached task: log, count, never raise
            verbose_logger.debug("shadow_eval: pipeline failed for %s: %s", request_id, e)
            await self._bump_failed(job.id, f"pipeline error: {e}")

    async def _bump_failed(self, job_id: str, error: str) -> None:
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return
        try:
            await prisma.db.litellm_shadowevaljob.update_many(
                where={  # mutable-ok: Prisma filter
                    "id": job_id,
                    "status": {"in": ["pending", "running"]},  # mutable-ok: Prisma filter
                },
                data={  # mutable-ok: Prisma payload
                    "failed_count": {"increment": 1},  # mutable-ok: Prisma operator
                    "last_error": error[:_MAX_LAST_ERROR_CHARS],
                },
            )
        except Exception as e:  # noqa: BLE001  # counter drift is acceptable
            verbose_logger.debug("shadow_eval: failed_count increment failed: %s", e)

    async def _call_router_shadow(
        self,
        router_name: str,
        messages: Sequence[Mapping[str, object]],
        model_parameters: Mapping[str, object],
        parent_metadata: Mapping[str, object],
    ) -> "_ShadowResponse | _CallFailure":
        """Send the prompt through the auto-router being evaluated."""
        router: Final = self._router_provider()
        if router is None:
            return _CallFailure("no router configured on this pod")
        # Carries the shadowed key's identity so this call's real provider spend is
        # attributed to, and budget-checked against, the key whose traffic it copies.
        # The router's pre-routing hook also writes its routing decision into this dict;
        # read it back after the call for tier attribution.
        shadow_metadata: Final[dict[str, object]] = sanitized_forwardable_call_metadata(  # mutable-ok: router writes back
            parent_metadata, SHADOW_EVAL_ROUTER_CALL_ORIGIN
        )
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
            return _CallFailure(f"shadow router call failed: {e}")
        text: Final = self._extract_response_text(response)
        if not text:
            return _CallFailure("shadow router returned an empty response")
        raw_decision: Final = shadow_metadata.get("routing_decision")
        routing_decision: Final = raw_decision if isinstance(raw_decision, Mapping) else _EMPTY_METADATA
        raw_tier: Final = routing_decision.get("tier_label") or routing_decision.get("tier")
        usage: Final = getattr(response, "usage", None)
        raw_tokens: Final = getattr(usage, "completion_tokens", None) if usage is not None else None
        return _ShadowResponse(
            text=text,
            model=str(getattr(response, "model", None) or routing_decision.get("routed_model") or ""),
            tier=str(raw_tier) if raw_tier is not None else None,
            completion_tokens=int(raw_tokens) if isinstance(raw_tokens, int) else None,
        )

    async def _call_judge(
        self,
        judge_model: str,
        messages: Sequence[Mapping[str, object]],
        real_text: str,
        shadow_text: str,
        parent_metadata: Mapping[str, object],
    ) -> "_JudgeVerdict | _CallFailure":
        """Blind pairwise judge, dispatched through the proxy's router when the judge
        model is a configured deployment (so DB-stored credentials work), the SDK
        otherwise (provider-qualified public names with credentials in the env)."""
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
        judge_metadata: Final = sanitized_forwardable_call_metadata(parent_metadata, SHADOW_EVAL_JUDGE_CALL_ORIGIN)
        judge_messages: Final = [  # mutable-ok: SDK takes a list
            {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM_PROMPT},  # mutable-ok: SDK message
            {"role": "user", "content": user_prompt},  # mutable-ok: SDK message
        ]
        router: Final = self._router_provider()
        use_router: Final = router is not None and (
            judge_model in router.model_group_alias or router.get_model_list(model_name=judge_model)
        )
        try:
            response: Final = (
                await router.acompletion(  # pyright: ignore[reportOptionalMemberAccess]  # use_router implies router is not None
                    model=judge_model,
                    messages=judge_messages,
                    temperature=0,
                    max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                    metadata=judge_metadata,
                    num_retries=0,
                    fallbacks=[],
                )
                if use_router
                else await litellm.acompletion(
                    model=judge_model,
                    messages=judge_messages,
                    temperature=0,
                    max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                    metadata=judge_metadata,
                )
            )
        except Exception as e:  # noqa: BLE001  # judge outages are a counted failure, not a crash
            verbose_logger.debug("shadow_eval: judge call failed: %s", e)
            return _CallFailure(f"judge call failed: {e}")
        try:
            raw: Final = response["choices"][0]["message"]["content"] or ""
            verdict: Final = _parse_pairwise_verdict(raw)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as e:
            verbose_logger.debug("shadow_eval: unparseable judge verdict: %s", e)
            return _CallFailure(f"unparseable judge verdict: {e}")
        return _JudgeVerdict(
            preference=_unmask_preference(verdict.preference, real_is_a),
            confidence=max(0.0, min(1.0, verdict.confidence)),
            reasoning=verdict.reasoning,
            cost=_judge_call_cost(response),
        )

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
