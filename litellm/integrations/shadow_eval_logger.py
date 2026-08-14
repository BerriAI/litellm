"""Shadow Eval Logger: samples a shadowed key's successful chat requests, duplicates each
through the auto-router in a detached task, blind-judges real vs shadow, and appends one
``LiteLLM_ShadowEvalAttempt`` row (verdict or error) as the feature's only hot-path write.
Counts, status, and spend derive from those rows at read time, so nothing can disagree
across pods or stop races; the hook reads active jobs through a short-TTL cache."""

import asyncio
import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.litellm_core_utils.internal_call_metadata import sanitized_forwardable_call_metadata
from litellm.litellm_core_utils.llm_judge import (
    default_router_provider,
    extract_text_from_content,
    judge_acompletion,
    parse_json_verdict,
)
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.types.utils import StandardLoggingPayload

# A job starting, stopping, or hitting its turn budget propagates to sampling within one
# TTL; the turn budget can overshoot by at most one TTL of in-flight samples per pod.
_JOBS_CACHE_TTL_SECONDS: Final = 10

# Concurrent shadow+judge pipelines per pod: a traffic spike turns into skipped samples
# rather than an unbounded task pileup.
_MAX_CONCURRENT_SHADOW_TASKS: Final = 16

# Total character budget for the judge's user prompt, however long the conversation and
# the two responses are, so the prompt can never overflow a judge model's context window.
_MAX_JUDGE_RESPONSE_CHARS: Final = 8_000
_MAX_JUDGE_PROMPT_CHARS: Final = 24_000

# The judge answers with a small JSON object; a tighter budget truncates the JSON
# mid-object and the attempt is lost to an error row.
JUDGE_MAX_OUTPUT_TOKENS: Final = 500

_MAX_ERROR_CHARS: Final = 500

_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})

_SAMPLED_CALL_TYPES: Final = frozenset({"completion", "acompletion"})

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


def _sample_hits(request_id: str, job_id: str, percentage: float) -> bool:
    """Deterministically decide whether a request falls in the shadowed slice: hash-based
    rather than random so retries sample the same way and pods agree without coordination."""
    digest: Final = hashlib.sha256(f"{job_id}:{request_id}".encode()).digest()
    bucket: Final = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket * 100.0 < percentage


def _judge_call_cost(response: object) -> float:
    """Price a judge call, treating an unmapped judge model as free rather than fatal."""
    import litellm

    try:
        return litellm.completion_cost(completion_response=response) or 0.0
    except Exception:  # noqa: BLE001  # unmapped judge model: the verdict still counts, cost stays 0
        return 0.0


def _unmask_preference(raw_preference: str, real_is_a: bool) -> str:
    """Map the judge's blind A/B/tie verdict back to real/shadow/tie."""
    normalized: Final = raw_preference.strip().lower()
    if normalized == "a":
        return "real" if real_is_a else "shadow"
    if normalized == "b":
        return "shadow" if real_is_a else "real"
    return "tie"


def _judge_user_prompt(conversation: str, response_a: str, response_b: str) -> str:
    """The judge prompt under one total character budget: each response is capped, and
    the conversation tail gets whatever budget the responses left over."""
    a: Final = response_a[:_MAX_JUDGE_RESPONSE_CHARS]
    b: Final = response_b[:_MAX_JUDGE_RESPONSE_CHARS]
    conversation_budget: Final = _MAX_JUDGE_PROMPT_CHARS - len(a) - len(b)
    return (
        f"Conversation:\n{conversation[-conversation_budget:]}\n\n"
        f"Response A:\n{a}\n\n"
        f"Response B:\n{b}\n\n"
        "Which response is better?"
    )


async def _key_or_team_is_over_budget(metadata: Mapping[str, object]) -> bool:
    """Whether the shadowed key or its team is over budget, decided by the same owners
    the request path uses, so counter keys and thresholds can never drift from auth's.

    Advisory and fail-open: real traffic on an over-budget key is already rejected at
    auth (so nothing reaches the success hook), and this gate only closes the race
    where the key crosses its budget while a request is in flight.
    """
    try:
        from litellm.exceptions import BudgetExceededError
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.auth_checks import (
            _team_max_budget_check,
            _virtual_key_max_budget_check,
            get_team_object,
        )
        from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache
    except ImportError:
        return False

    auth: Final = metadata.get("user_api_key_auth")
    if not isinstance(auth, UserAPIKeyAuth):
        return False
    try:
        await _virtual_key_max_budget_check(valid_token=auth, proxy_logging_obj=proxy_logging_obj)
        if auth.team_id:
            team: Final = await get_team_object(
                team_id=auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                check_cache_only=True,
            )
            await _team_max_budget_check(team_object=team, valid_token=auth, proxy_logging_obj=proxy_logging_obj)
    except BudgetExceededError:
        return True
    except Exception as e:  # noqa: BLE001  # advisory gate: a failed read must not block sampling
        verbose_logger.debug("shadow_eval: budget read failed: %s", e)
    return False


def _request_was_routed_by(request_metadata: Mapping[str, object], router_name: str) -> bool:
    """Duplicating a request the shadowed router already served compares the router to
    itself: guaranteed ties, judge spend for zero information."""
    decision: Final = request_metadata.get("routing_decision")
    if not isinstance(decision, Mapping):
        return False
    return decision.get("router_model_name") == router_name


@dataclass(frozen=True, slots=True)
class _CallFailure:
    """A shadow or judge call that produced no usable response. cost carries any judge
    spend the failed attempt still billed, so job-level judge_spend never undercounts."""

    error: str
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class _ShadowResponse:
    """A successful shadow call, with what the attempt row records."""

    text: str
    model: str
    tier: str | None


@dataclass(frozen=True, slots=True)
class _JudgeVerdict:
    """A parsed judge verdict, unmasked back to real/shadow/tie."""

    preference: str
    confidence: float
    cost: float


@dataclass(frozen=True, slots=True)
class ActiveShadowEvalJob:
    """One active job as the sampling path needs it: immutable config plus the attempt
    count as of the cache fill (the turn budget's staleness is bounded by the cache TTL)."""

    id: str
    router_name: str
    shadow_percentage: float
    judge_model: str
    max_turns: int
    ends_at: datetime
    attempts: int


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


_jobs_cache: Final = InMemoryCache(max_size_in_memory=4, default_ttl=_JOBS_CACHE_TTL_SECONDS)
_JOBS_CACHE_KEY: Final = "shadow_eval:active_jobs"


class ShadowEvalLogger(CustomLogger):
    """Fires blind pairwise shadow evaluations for keys with an active shadow-eval job."""

    def __init__(
        self,
        router_provider: Callable[[], "Router | None"] | None = None,
        prisma_provider: Callable[[], "PrismaClient | None"] | None = None,
        jobs_cache: InMemoryCache | None = None,
    ) -> None:
        """Providers are callables so the proxy's lazily-initialized globals are resolved
        at call time, not at logger construction."""
        self._router_provider = router_provider or default_router_provider
        self._prisma_provider = prisma_provider or _default_prisma_provider
        self._jobs_cache = jobs_cache or _jobs_cache
        self._inflight_shadow_tasks: int = 0
        # Starts per job since the last cache fill, never decremented within a
        # generation; the refill absorbs written rows and resets.
        self._job_starts: dict[str, int] = {}  # mutable-ok: per-generation counter

    async def _active_jobs(self) -> Mapping[str, ActiveShadowEvalJob]:
        """Active jobs by api_key_id, cache-first. A DB fault returns empty without
        caching, so sampling pauses for that request and the next one retries."""
        cached: Final = await self._jobs_cache.async_get_cache(_JOBS_CACHE_KEY)
        if cached is not None:
            return cached  # pyright: ignore[reportReturnType]  # cache stores exactly this mapping shape
        prisma: Final = self._prisma_provider()
        if prisma is None:
            return _EMPTY_JOBS
        try:
            records: Final = await prisma.db.litellm_shadowevaljob.find_many(
                where={  # mutable-ok: Prisma filter
                    "stopped_at": None,
                    "ends_at": {"gt": datetime.now(timezone.utc)},  # mutable-ok: Prisma filter
                },
            )
            grouped: Final = (
                await prisma.db.litellm_shadowevalattempt.group_by(
                    by=["job_id"],
                    count=True,
                    where={"job_id": {"in": [str(record.id) for record in records]}},  # mutable-ok: Prisma filter
                )
                if records
                else ()
            )
            attempt_counts: Final = {str(row["job_id"]): int(row["_count"]["_all"]) for row in grouped or []}
            jobs: Final = {
                str(record.api_key_id): ActiveShadowEvalJob(
                    id=str(record.id),
                    router_name=str(record.router_name),
                    shadow_percentage=float(record.shadow_percentage),
                    judge_model=str(record.judge_model),
                    max_turns=int(record.max_turns),
                    ends_at=_as_utc(record.ends_at),
                    attempts=attempt_counts.get(str(record.id), 0),
                )
                for record in records or []
            }
            await self._jobs_cache.async_set_cache(_JOBS_CACHE_KEY, jobs)
            self._job_starts = {}  # rebind-ok: new generation, counts absorbed into the fill
            return jobs
        except Exception as e:  # noqa: BLE001  # a DB blip must never break request logging
            verbose_logger.debug("shadow_eval: active-job read failed: %s", e)
            return _EMPTY_JOBS

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
            raw_meta: Final = get_litellm_metadata_from_kwargs(dict(kwargs))  # mutable-ok: helper needs dict
            request_metadata: Final = raw_meta if isinstance(raw_meta, Mapping) else _EMPTY_METADATA
            if request_metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
                return  # internal sub-call (our own shadow/judge, a classifier), not user traffic
            # redaction rewrites logged content before callbacks run, so this hook
            # only ever sees placeholders for a redacted request
            if should_redact_message_logging(dict(kwargs)):  # mutable-ok: predicate takes a plain dict
                return
            metadata: Final = payload.get("metadata") or _EMPTY_METADATA
            api_key_hash: Final = metadata.get("user_api_key_hash")
            if not api_key_hash:
                return
            job: Final = (await self._active_jobs()).get(str(api_key_hash))
            if job is None:
                return
            if datetime.now(timezone.utc) >= job.ends_at:
                return
            if job.attempts + self._job_starts.get(job.id, 0) >= job.max_turns:
                return
            request_id: Final = payload.get("id") or ""
            if not request_id:
                return
            if not _sample_hits(request_id, job.id, job.shadow_percentage):
                return
            if payload.get("call_type") not in _SAMPLED_CALL_TYPES:
                return  # only known chat-shaped traffic is comparable; unknown or missing types fail closed
            if _request_was_routed_by(request_metadata, job.router_name):
                return
            if self._inflight_shadow_tasks >= _MAX_CONCURRENT_SHADOW_TASKS:
                return
            raw_messages: Final = kwargs.get("messages")
            self._job_starts[job.id] = self._job_starts.get(job.id, 0) + 1
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
            task.add_done_callback(self._release_shadow_slot)
        except Exception as e:  # noqa: BLE001  # logging hooks must never fail the request
            verbose_logger.debug("shadow_eval: failed to schedule task: %s", e)

    def _release_shadow_slot(self, _task: "asyncio.Task[None]") -> None:
        self._inflight_shadow_tasks -= 1

    #### the detached pipeline: one attempt row per sampled request, verdict or error ####

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
        """Budget gate -> shadow call -> blind judge -> one attempt row. The prisma gate
        sits above the dispatch so no provider spend happens without a place to record
        the outcome, and the budget read lives here rather than in the success hook."""
        prisma: Final = self._prisma_provider()
        try:
            if prisma is None:
                return
            real_text: Final = self._extract_response_text(response_obj)
            if not real_text or not messages:
                return
            if await _key_or_team_is_over_budget(parent_metadata):
                return

            shadow: Final = await self._call_router_shadow(job.router_name, messages, model_parameters, parent_metadata)
            if isinstance(shadow, _CallFailure):
                await self._record_attempt(prisma, job, request_id, outcome="error", error=shadow.error)
                return

            verdict: Final = await self._call_judge(
                judge_model=job.judge_model,
                messages=messages,
                real_text=real_text,
                shadow_text=shadow.text,
                parent_metadata=parent_metadata,
            )
            if isinstance(verdict, _CallFailure):
                await self._record_attempt(
                    prisma,
                    job,
                    request_id,
                    outcome="error",
                    error=verdict.error,
                    shadow=shadow,
                    judge_cost=verdict.cost,
                )
                return
            await self._record_attempt(
                prisma,
                job,
                request_id,
                outcome=verdict.preference,
                shadow=shadow,
                real_model=real_model,
                confidence=verdict.confidence,
                judge_cost=verdict.cost,
            )
        except Exception as e:  # noqa: BLE001  # detached task: record what happened, never raise
            verbose_logger.debug("shadow_eval: pipeline failed for %s: %s", request_id, e)
            await self._record_attempt(prisma, job, request_id, outcome="error", error=f"pipeline error: {e}")

    @staticmethod
    async def _record_attempt(
        prisma: "PrismaClient | None",
        job: ActiveShadowEvalJob,
        request_id: str,
        *,
        outcome: str,
        shadow: _ShadowResponse | None = None,
        real_model: str = "",
        confidence: float | None = None,
        judge_cost: float = 0.0,
        error: str | None = None,
    ) -> None:
        if prisma is None:
            return
        try:
            await prisma.db.litellm_shadowevalattempt.create(
                data={  # mutable-ok: Prisma payload
                    "job_id": job.id,
                    "request_id": request_id,
                    "outcome": outcome,
                    "tier": shadow.tier if shadow else None,
                    "real_model": real_model or None,
                    "shadow_model": shadow.model if shadow else None,
                    "confidence": confidence,
                    "judge_cost": judge_cost,
                    "error": error[:_MAX_ERROR_CHARS] if error else None,
                }
            )
        except Exception as e:  # noqa: BLE001  # a lost row degrades sample size, nothing can disagree with it
            verbose_logger.debug("shadow_eval: attempt write failed for %s: %s", request_id, e)

    async def _call_router_shadow(
        self,
        router_name: str,
        messages: Sequence[Mapping[str, object]],
        model_parameters: Mapping[str, object],
        parent_metadata: Mapping[str, object],
    ) -> "_ShadowResponse | _CallFailure":
        """Send the prompt through the auto-router being evaluated. The metadata carries
        the shadowed key's identity (spend attribution) and receives the router's routing
        decision write-back, read back for tier attribution."""
        router: Final = self._router_provider()
        if router is None:
            return _CallFailure("no router configured on this pod")
        shadow_metadata: Final[dict[str, object]] = (  # mutable-ok: router writes its routing decision back
            sanitized_forwardable_call_metadata(parent_metadata, SHADOW_EVAL_ROUTER_CALL_ORIGIN)
        )
        shadow_params: Final = {  # mutable-ok: splatted as kwargs
            k: v for k, v in model_parameters.items() if k not in ("stream", "metadata")
        }
        try:
            response: Final = await router.acompletion(
                model=router_name,
                messages=messages,  # pyright: ignore[reportArgumentType]  # snapshot of the SDK's own message dicts
                metadata=shadow_metadata,
                num_retries=0,
                fallbacks=[],  # mutable-ok: SDK kwarg; a failed shadow is a recorded error, never a spend multiplier
                **shadow_params,
            )
        except Exception as e:  # noqa: BLE001  # provider errors become error rows, not crashes
            verbose_logger.debug("shadow_eval: router call failed: %s", e)
            return _CallFailure(f"shadow router call failed: {e}")
        text: Final = self._extract_response_text(response)
        if not text:
            return _CallFailure("shadow router returned an empty response")
        raw_decision: Final = shadow_metadata.get("routing_decision")
        routing_decision: Final = raw_decision if isinstance(raw_decision, Mapping) else _EMPTY_METADATA
        raw_tier: Final = routing_decision.get("tier_label") or routing_decision.get("tier")
        return _ShadowResponse(
            text=text,
            model=str(getattr(response, "model", None) or routing_decision.get("routed_model") or ""),
            tier=str(raw_tier) if raw_tier is not None else None,
        )

    async def _call_judge(
        self,
        judge_model: str,
        messages: Sequence[Mapping[str, object]],
        real_text: str,
        shadow_text: str,
        parent_metadata: Mapping[str, object],
    ) -> "_JudgeVerdict | _CallFailure":
        """Blind pairwise judge with A/B labels randomized to cancel position bias."""
        real_is_a: Final = random.random() < 0.5
        response_a: Final = real_text if real_is_a else shadow_text
        response_b: Final = shadow_text if real_is_a else real_text

        conversation: Final = "\n".join(
            f"{str(m.get('role', 'user')).upper()}: {extract_text_from_content(m.get('content'))}"
            for m in messages
            if m.get("content") is not None
        )
        judge_metadata: Final = sanitized_forwardable_call_metadata(parent_metadata, SHADOW_EVAL_JUDGE_CALL_ORIGIN)
        judge_messages: Final = [  # mutable-ok: SDK takes a list
            {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM_PROMPT},  # mutable-ok: SDK message
            {
                "role": "user",
                "content": _judge_user_prompt(conversation, response_a, response_b),
            },  # mutable-ok: SDK message
        ]
        try:
            response: Final = await judge_acompletion(
                self._router_provider(),
                judge_model,
                judge_messages,  # pyright: ignore[reportArgumentType]  # plain SDK message dicts
                temperature=0,
                max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                metadata=judge_metadata,
            )
        except Exception as e:  # noqa: BLE001  # judge outages become error rows, not crashes
            verbose_logger.debug("shadow_eval: judge call failed: %s", e)
            return _CallFailure(f"judge call failed: {e}")
        try:
            raw: Final = response["choices"][0]["message"]["content"] or ""
            verdict: Final = PairwiseVerdict.model_validate(parse_json_verdict(raw))
        except Exception as e:  # noqa: BLE001  # malformed verdicts become error rows
            verbose_logger.debug("shadow_eval: unparseable judge verdict: %s", e)
            return _CallFailure(f"unparseable judge verdict: {e}", cost=_judge_call_cost(response))
        return _JudgeVerdict(
            preference=_unmask_preference(verdict.preference, real_is_a),
            confidence=max(0.0, min(1.0, verdict.confidence)),
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
        return extract_text_from_content(content)


_EMPTY_JOBS: Final[Mapping[str, ActiveShadowEvalJob]] = MappingProxyType({})


def _default_prisma_provider() -> "PrismaClient | None":
    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        return None
    return prisma_client
