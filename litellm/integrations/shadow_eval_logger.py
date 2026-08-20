"""Shadow Eval Logger: samples a shadowed key's successful LLM requests (chat completions,
Anthropic Messages, and Responses API surfaces, each normalized to chat shape), duplicates
each against the job's other arm in a detached task (the auto-router for a forward job, the
fixed baseline model for a reverse one), blind-judges real vs shadow, and appends one
``LiteLLM_ShadowEvalAttempt`` row (verdict or error) as the feature's only hot-path write.
Counts, status, and spend derive from those rows at read time, so nothing can disagree
across pods or stop races; the hook reads active jobs through a short-TTL cache."""

import asyncio
import hashlib
import random
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator, model_validator

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
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.types.management_endpoints.auto_router_endpoints import ShadowEvalDirection
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.types.utils import StandardLoggingPayload

# A job starting, stopping, or hitting a budget propagates to sampling within one TTL;
# the spend gate re-checks the cross-pod counter at pipeline entry, so it overshoots
# only by the samples already in flight when the cap is crossed.
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
JUDGE_MAX_OUTPUT_TOKENS: Final = 1500

_MAX_ERROR_CHARS: Final = 500

_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})

# Typed boundaries around the owner transformations, which declare untyped returns:
# a request or message that fails this lenient shape check is skipped, never sampled.
_CHAT_REQUEST_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_CHAT_MESSAGES_ADAPTER: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_MESSAGE_ITEMS_ADAPTER: Final = TypeAdapter(tuple[object, ...])


def _chat_messages(kwargs: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw: Final = kwargs.get("messages")
    return tuple(m for m in raw if isinstance(m, Mapping)) if isinstance(raw, Sequence) else ()


def _proxy_wire_body(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    litellm_params: Final = kwargs.get("litellm_params")
    request: Final = litellm_params.get("proxy_server_request") if isinstance(litellm_params, Mapping) else None
    body: Final = request.get("body") if isinstance(request, Mapping) else None
    return body if isinstance(body, Mapping) else _EMPTY_METADATA


def _chat_request_from_chat(
    kwargs: Mapping[str, object], model_parameters: Mapping[str, object]
) -> Mapping[str, object]:
    """Chat requests are already chat-shaped: the logged model_parameters forward as-is."""
    return MappingProxyType({**model_parameters, "messages": _chat_messages(kwargs)})


# Anthropic params the adapter copies through untranslated; the translatable set comes
# from the adapter itself at call time.
_ANTHROPIC_SAMPLING_PARAM_KEYS: Final = frozenset(("max_tokens", "temperature", "top_p", "top_k", "reasoning_effort"))


def _chat_request_from_anthropic_messages(
    kwargs: Mapping[str, object], _model_parameters: Mapping[str, object]
) -> Mapping[str, object]:
    """/v1/messages logs surface-native block messages with ``system`` top-level: the
    native provider path carries it in kwargs, the openai-compatible bridge path only in
    the proxy's snapshot of the client's wire body. Params come from the wire body alone,
    because the logged optional_params switch dialect per provider path (the bridge's
    inner completion rewrites them to chat shape mid-flight); the adapter translates
    them alongside the messages, and sampling params copy through untranslated."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    adapter: Final = LiteLLMAnthropicMessagesAdapter()
    wire_body: Final = _proxy_wire_body(kwargs)
    system: Final = kwargs.get("system") or wire_body.get("system")
    param_keys: Final = (
        frozenset(adapter.translatable_anthropic_params()) | _ANTHROPIC_SAMPLING_PARAM_KEYS
    ) - frozenset(("messages", "system"))
    request: Final = MappingProxyType(
        dict(
            (
                *((k, v) for k, v in wire_body.items() if k in param_keys),
                ("model", str(kwargs.get("model") or "")),
                ("messages", _CHAT_MESSAGES_ADAPTER.validate_python(kwargs.get("messages") or ())),
                *((("system", system),) if system is not None else ()),
            )
        )
    )
    translated, _ = adapter.translate_anthropic_to_openai(request)  # pyright: ignore[reportArgumentType]  # wire-body mapping is the surface's native request shape; the adapter is duck-typed and read-only here
    return translated


def _chat_request_from_responses(
    kwargs: Mapping[str, object], _model_parameters: Mapping[str, object]
) -> Mapping[str, object]:
    """/v1/responses logs the raw ``input`` under ``kwargs["messages"]``, an alias
    function_setup creates for responses call types: a bare string, chat-shaped dicts,
    or item dicts; ``instructions`` is the system prompt. Params come from the wire body
    for the same reason as the messages surface; the transformer translates them with
    the input (max_output_tokens to max_tokens, Responses tools to chat tools, reasoning
    to reasoning_effort) and never reads surface-only keys like previous_response_id."""
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )
    from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

    wire_body: Final = _proxy_wire_body(kwargs)
    instructions: Final = kwargs.get("instructions") or wire_body.get("instructions")
    responses_request: Final = MappingProxyType(
        dict(
            (
                *((k, v) for k, v in wire_body.items() if k in ResponsesAPIOptionalRequestParams.__annotations__),
                *((("instructions", instructions),) if instructions is not None else ()),
            )
        )
    )
    return _CHAT_REQUEST_ADAPTER.validate_python(
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(  # pyright: ignore[reportUnknownMemberType]  # transformer declares a bare dict return
            model=str(kwargs.get("model") or ""),
            input=kwargs.get("messages"),  # pyright: ignore[reportArgumentType]  # untyped callback kwargs; transformer validates shapes
            responses_api_request=responses_request,  # pyright: ignore[reportArgumentType]  # wire-body dict filtered to the surface's own request keys; the transformer is duck-typed
        )
    )


def _chat_final_text(response_obj: object) -> str:
    """The assistant's text, or empty when the turn carries tool calls: only text-final
    turns produce a judgeable A/B comparison."""
    try:
        message: Final = (
            response_obj["choices"][0]["message"]
            if isinstance(response_obj, Mapping)
            else response_obj.choices[0].message  # pyright: ignore[reportAttributeAccessIssue]  # duck-typed ModelResponse
        )
    except (AttributeError, KeyError, IndexError, TypeError):
        return ""
    read: Final = message.get if isinstance(message, Mapping) else lambda key: getattr(message, key, None)
    if read("tool_calls") or read("function_call"):
        return ""
    return extract_text_from_content(read("content"))


def _responses_final_text(response_obj: object) -> str:
    """The turn's aggregated output text, or empty when the turn carries tool calls. A
    dict-shaped payload is validated into the owner type first, because ``output_text``
    is a derived property rather than a serialized field, so it never exists on a dict;
    a dict the owner type rejects is unjudgeable and skipped."""
    from litellm.types.llms.openai import ResponsesAPIResponse

    try:
        response: Final = (
            ResponsesAPIResponse.model_validate(response_obj) if isinstance(response_obj, Mapping) else response_obj
        )
    except ValidationError:
        return ""
    output: Final = getattr(response, "output", None)
    if not isinstance(output, Sequence):
        return ""
    items: Final = tuple(item.model_dump() if isinstance(item, BaseModel) else item for item in output)
    if any(
        not isinstance(item, Mapping) or item.get("type") in ("function_call", "custom_tool_call") for item in items
    ):
        return ""
    return str(getattr(response, "output_text", "") or "")


class _SurfaceOps:
    """One row per sampled call_type: how its logged request becomes a chat-shaped
    request (messages plus translated generation params) and how its response yields
    the judgeable final text. Membership in this table IS the sampling allowlist;
    unknown call types fail closed. ``wire_params`` marks the surfaces whose params
    come from the proxy's wire-body snapshot, which is taken before the guardrail
    pre-call hook: those rows must not sample a request a pre-call guardrail rewrote,
    or the shadow call would replay content (tools, unmasked entities) the guardrail
    removed."""

    __slots__ = ("chat_request", "final_text", "wire_params")

    def __init__(
        self,
        chat_request: Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]],
        final_text: Callable[[object], str],
        wire_params: bool,
    ) -> None:
        self.chat_request = chat_request
        self.final_text = final_text
        self.wire_params = wire_params


_CHAT_OPS: Final = _SurfaceOps(_chat_request_from_chat, _chat_final_text, wire_params=False)
_ANTHROPIC_OPS: Final = _SurfaceOps(_chat_request_from_anthropic_messages, _chat_final_text, wire_params=True)
_RESPONSES_OPS: Final = _SurfaceOps(_chat_request_from_responses, _responses_final_text, wire_params=True)

# Guardrail hooks that never rewrite the outbound request: they run in parallel with
# the call, on the response, or on logged copies. Anything else (pre_call, pre_mcp_call,
# a future mode) counts as request-mutating, failing closed.
_NON_MUTATING_GUARDRAIL_MODES: Final = frozenset(
    ("during_call", "post_call", "logging_only", "during_mcp_call", "post_mcp_call", "realtime_input_transcription")
)


def _request_mutating_guardrail_ran(request_metadata: Mapping[str, object]) -> bool:
    """Whether a guardrail that can rewrite the outbound request ran on this one, read
    from the same guardrail-information entries spend logging uses. str-enum modes
    compare equal to their plain-string values, and an entry whose mode is missing or
    unrecognized counts as mutating."""
    raw: Final = request_metadata.get("standard_logging_guardrail_information")
    entries: Final = raw if isinstance(raw, Sequence) else ()
    modes_per_entry: Final = tuple(entry.get("guardrail_mode") for entry in entries if isinstance(entry, Mapping))
    return any(
        not all(
            mode in _NON_MUTATING_GUARDRAIL_MODES for mode in (modes if isinstance(modes, list | tuple) else (modes,))
        )
        for modes in modes_per_entry
    )


# Translated-request keys that never forward to the shadow call: identity and transport,
# not generation. Empty-list values (e.g. tools) carry nothing and are dropped with them.
_UNFORWARDED_REQUEST_KEYS: Final = frozenset(("model", "messages", "stream", "stream_options", "metadata"))


def _forwards_nothing(value: object) -> bool:
    return value is None or (isinstance(value, list) and len(value) == 0)


def _judgeable_sample(
    ops: _SurfaceOps,
    kwargs: Mapping[str, object],
    model_parameters: Mapping[str, object],
    response_obj: object,
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object], str] | None:
    """The normalized chat conversation, the forwardable generation params, and the
    judgeable final text; None when this request's shapes cannot be sampled (tool-final
    turn, empty text, or a shape the owner transformations reject)."""
    try:
        request: Final = ops.chat_request(kwargs, model_parameters)
        items: Final = _MESSAGE_ITEMS_ADAPTER.validate_python(request.get("messages"))
        messages: Final = _CHAT_MESSAGES_ADAPTER.validate_python(
            tuple(m.model_dump(exclude_none=True) if isinstance(m, BaseModel) else m for m in items)
        )
    except Exception as e:  # noqa: BLE001  # a rejected shape is skipped, never sampled
        verbose_logger.debug("shadow_eval: request normalization failed, skipping: %s", e)
        return None
    real_text: Final = ops.final_text(response_obj)
    if not messages or not real_text:
        return None
    params: Final = MappingProxyType(
        {k: v for k, v in request.items() if k not in _UNFORWARDED_REQUEST_KEYS and not _forwards_nothing(v)}
    )
    return messages, params, real_text


_SURFACE_OPS: Final[Mapping[str, _SurfaceOps]] = MappingProxyType(
    {
        "completion": _CHAT_OPS,
        "acompletion": _CHAT_OPS,
        "anthropic_messages": _ANTHROPIC_OPS,
        "aresponses": _RESPONSES_OPS,
        "responses": _RESPONSES_OPS,
    }
)

PAIRWISE_JUDGE_SYSTEM_PROMPT: Final = """You are an impartial quality judge comparing two responses to the same conversation.

The responses are labeled A and B in random order. You do not know which system produced which.

Criteria: correctness, completeness, clarity, conciseness.

Return ONLY valid JSON in this exact format, no other text:
{
  "preference": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>
}"""


class PairwiseVerdict(BaseModel):
    """The judge's blind A/B verdict: the response_format schema sent with the judge call
    and the validation contract on its reply. Both fields are required and preference is
    closed over the prompt's labels, so a malformed or truncated reply is an
    unparseable-verdict error row, never a defaulted or fabricated verdict."""

    preference: Literal["A", "B", "tie"]
    confidence: float


PAIRWISE_JUDGE_RESPONSE_FORMAT: Final = type_to_response_format_param(PairwiseVerdict)


def _sample_hits(request_id: str, job_id: str, percentage: float) -> bool:
    """Deterministically decide whether a request falls in the shadowed slice: hash-based
    rather than random so retries sample the same way and pods agree without coordination."""
    digest: Final = hashlib.sha256(f"{job_id}:{request_id}".encode()).digest()
    bucket: Final = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket * 100.0 < percentage


def _failure_detail(e: BaseException) -> str:
    """Exception class, message, and the raising frame, so an attempt's error row names
    the faulty code path without needing debug logs on the pod."""
    frames: Final = traceback.extract_tb(e.__traceback__)
    location: Final = f" at {frames[-1].filename.rsplit('/', 1)[-1]}:{frames[-1].lineno}" if frames else ""
    return f"{type(e).__name__}{location}: {e}"


def _call_cost(response: object) -> float:
    """Price one eval-arm call with the figure the spend pipeline bills: the router client
    stamps _hidden_params.response_cost from the deployment's own pricing, which the public
    price map lookup below cannot see (it reads 0 for deployment-priced models)."""
    getter: Final = getattr(getattr(response, "_hidden_params", None), "get", None)
    stamped: Final = getter("response_cost") if callable(getter) else None
    if isinstance(stamped, (int, float)):
        return float(stamped)
    return _price_map_cost(response)


def _price_map_cost(response: object) -> float:
    """Public price map fallback, treating an unmapped model as free rather than fatal."""
    import litellm

    try:
        return litellm.completion_cost(completion_response=response) or 0.0
    except Exception:  # noqa: BLE001  # unmapped model: the attempt still counts, cost stays 0
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


def _job_spend_counter_key(job_id: str) -> str:
    return f"spend:shadow_eval:{job_id}"


async def _job_spend_from_counter(counter_key: str, fallback_spend: float, max_budget: float) -> float:
    """The leg's spend through the cross-pod counter the key budget gates read. The owner
    degrades internally to the fill-time DB floor and raises only under fail-closed
    enforcement, which the caller honors by skipping the sample."""
    from litellm.proxy.proxy_server import get_current_spend

    return await get_current_spend(counter_key=counter_key, fallback_spend=fallback_spend, max_budget=max_budget)


async def _add_job_spend_to_counter(counter_key: str, cost: float) -> None:
    """Advance the counter the moment a cost is known, so even a lost row closes the gate.
    Known failure mode: a Redis outage freezes the counter (the owner invalidates it), the
    gate degrades to the fill floor, and overshoot grows to in-flight plus one TTL of
    samples, the same degradation the key budget counters accept."""
    try:
        from litellm.proxy.proxy_server import increment_spend_counter

        await increment_spend_counter(counter_key=counter_key, increment=cost)
    except Exception as e:  # noqa: BLE001  # attempt recording must proceed; the row stays truth and the fill floor gates
        verbose_logger.warning("shadow_eval: spend counter increment failed for %s: %s", counter_key, e)


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


def _routing_decision(metadata: Mapping[str, object]) -> Mapping[str, object]:
    """The routing decision a pre-routing strategy wrote to a call's metadata, empty when
    a plain model served it. Read off the sampled request for the control arm, and off the
    shadow call's own write-back for the shadow arm."""
    decision: Final = metadata.get("routing_decision")
    return decision if isinstance(decision, Mapping) else _EMPTY_METADATA


def _routed_tier(metadata: Mapping[str, object]) -> str | None:
    decision: Final = _routing_decision(metadata)
    raw: Final = decision.get("tier_label") or decision.get("tier")
    return str(raw) if raw is not None else None


def _request_was_routed_by(request_metadata: Mapping[str, object], router_name: str) -> bool:
    """Whether the router under evaluation served this request, which is what decides
    the direction it belongs to. A forward job skips its own router's traffic, since
    duplicating it would compare the router to itself: guaranteed ties, judge spend for
    zero information. A reverse job samples exactly that traffic and nothing else."""
    return _routing_decision(request_metadata).get("router_model_name") == router_name


@dataclass(frozen=True, slots=True)
class _CallFailure:
    """A shadow or judge call that produced no usable response. cost carries any spend
    the failed call still billed, so job-level spend figures never undercount."""

    error: str
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class _ShadowResponse:
    """A successful shadow call, with what the attempt row records."""

    text: str
    model: str
    tier: str | None
    cost: float


@dataclass(frozen=True, slots=True)
class _JudgeVerdict:
    """A parsed judge verdict, unmasked back to real/shadow/tie."""

    preference: str
    confidence: float
    cost: float


class ActiveShadowEvalJob(BaseModel):
    """One active job as the sampling path needs it, validated straight off the untyped
    job row: immutable config plus the attempt count as of the cache fill (the turn
    budget's staleness is bounded by the cache TTL). Every way a row can be unsamplable
    is a validation error here, so a bad row is skipped rather than sampled wrongly."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: str
    router_name: str
    direction: ShadowEvalDirection = "forward"
    baseline_model: str | None = None
    shadow_percentage: float
    judge_model: str
    max_turns: int
    max_budget: float | None = None
    ends_at: datetime
    attempts: int = 0
    spend: float = 0.0

    @field_validator("ends_at")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    @model_validator(mode="after")
    def _baseline_model_matches_direction(self) -> "ActiveShadowEvalJob":
        if (self.baseline_model is not None) != (self.direction == "reverse"):
            raise ValueError("baseline_model is set for exactly the reverse jobs")
        return self

    @property
    def shadow_target(self) -> str:
        """The model the duplicated arm calls: the router itself for a forward job, the
        fixed baseline for a reverse one. Total because the validator above pins
        baseline_model to reverse jobs and only those."""
        return self.baseline_model or self.router_name


def _as_active_job(record: object, attempts: int, spend: float) -> ActiveShadowEvalJob | None:
    """The sampling path's view of one job row, or None for a row it cannot sample: an
    unknown direction, or a reverse job with no baseline model to duplicate against.
    Failing closed here is what keeps the dispatch path total."""
    try:
        job: Final = ActiveShadowEvalJob.model_validate(record)
    except ValidationError as e:
        verbose_logger.debug("shadow_eval: skipping unsamplable job row: %s", e)
        return None
    return job.model_copy(update={"attempts": attempts, "spend": spend})  # mutable-ok: pydantic update payload


_jobs_cache: Final = InMemoryCache(max_size_in_memory=4, default_ttl=_JOBS_CACHE_TTL_SECONDS)
_JOBS_CACHE_KEY: Final = "shadow_eval:active_jobs"


class ShadowEvalLogger(CustomLogger):
    """Fires blind pairwise shadow evaluations for keys with an active shadow-eval job."""

    def __init__(
        self,
        router_provider: Callable[[], "Router | None"] | None = None,
        prisma_provider: Callable[[], "PrismaClient | None"] | None = None,
        jobs_cache: InMemoryCache | None = None,
        job_spend_reader: Callable[[str, float, float], Awaitable[float]] | None = None,
        job_spend_writer: Callable[[str, float], Awaitable[None]] | None = None,
    ) -> None:
        """Providers are callables so the proxy's lazily-initialized globals are resolved
        at call time, not at logger construction. The spend reader and writer wrap the
        proxy's cross-pod spend counter; tests inject a plain in-memory pair."""
        self._router_provider = router_provider or default_router_provider
        self._prisma_provider = prisma_provider or _default_prisma_provider
        self._jobs_cache = jobs_cache or _jobs_cache
        self._read_job_spend = job_spend_reader or _job_spend_from_counter
        self._write_job_spend = job_spend_writer or _add_job_spend_to_counter
        self._inflight_shadow_tasks: int = 0
        # Starts per job since the last cache fill, never decremented within a
        # generation; the refill absorbs written rows and resets.
        self._job_starts: dict[str, int] = {}  # mutable-ok: per-generation counter

    async def _active_jobs(self) -> Mapping[str, tuple[ActiveShadowEvalJob, ...]]:
        """Active jobs by api_key_id, cache-first. A key holds at most one job per
        direction, so the value is a collection. A DB fault returns empty without
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
                    sum={"judge_cost": True, "shadow_cost": True},  # mutable-ok: Prisma aggregate spec
                    where={"job_id": {"in": [str(record.id) for record in records]}},  # mutable-ok: Prisma filter
                )
                if records
                else ()
            )
            attempt_stats: Final = {  # mutable-ok: frozen snapshot of the grouped read
                str(row["job_id"]): (
                    int(row["_count"]["_all"]),
                    float((row["_sum"] or {}).get("judge_cost") or 0.0)
                    + float((row["_sum"] or {}).get("shadow_cost") or 0.0),
                )
                for row in grouped or []
            }
            by_key: Final = tuple(
                sorted(
                    (
                        (str(record.api_key_id), job)
                        for record in records or []
                        if (job := _as_active_job(record, *attempt_stats.get(str(record.id), (0, 0.0)))) is not None
                    ),
                    key=itemgetter(0),
                )
            )
            jobs: Final = MappingProxyType(
                {key: tuple(job for _, job in group) for key, group in groupby(by_key, key=itemgetter(0))}
            )
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
            request_id: Final = payload.get("id") or ""
            if not request_id:
                return
            ops: Final = _SURFACE_OPS.get(str(payload.get("call_type") or ""))
            if ops is None:
                return  # only surfaces this table can normalize are comparable; unknown types fail closed
            if ops.wire_params and _request_mutating_guardrail_ran(request_metadata):
                return  # the wire-body snapshot predates the rewrite; replaying it would resurrect stripped content
            # A key can hold one job per direction, and a request routed by one job's
            # router while bypassing the other's qualifies for both. Each is separately
            # budgeted, so both fire; the request is normalized once, and only when at
            # least one job sampled it.
            eligible: Final = tuple(
                job
                for job in (await self._active_jobs()).get(str(api_key_hash), ())
                if datetime.now(timezone.utc) < job.ends_at
                and job.attempts + self._job_starts.get(job.id, 0) < job.max_turns
                and (job.max_budget is None or job.spend < job.max_budget)
                and _sample_hits(request_id, job.id, job.shadow_percentage)
                and _request_was_routed_by(request_metadata, job.router_name) == (job.direction == "reverse")
            )
            if not eligible:
                return
            sample: Final = _judgeable_sample(
                ops,
                kwargs,
                MappingProxyType(dict(payload.get("model_parameters") or {})),  # mutable-ok: frozen snapshot
                response_obj,
            )
            if sample is None:
                return
            messages, shadow_params, real_text = sample
            control_tier: Final = _routed_tier(request_metadata)
            for job in eligible:
                if self._inflight_shadow_tasks >= _MAX_CONCURRENT_SHADOW_TASKS:
                    return
                self._job_starts[job.id] = self._job_starts.get(job.id, 0) + 1
                self._inflight_shadow_tasks += 1
                asyncio.create_task(
                    self._run_shadow_eval(
                        job=job,
                        request_id=request_id,
                        messages=messages,
                        real_text=real_text,
                        real_model=payload.get("model") or "",
                        control_tier=control_tier,
                        shadow_params=shadow_params,
                        parent_metadata=MappingProxyType(dict(request_metadata)),  # mutable-ok: frozen snapshot
                    )
                ).add_done_callback(self._release_shadow_slot)
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
        real_text: str,
        real_model: str,
        control_tier: str | None,
        shadow_params: Mapping[str, object],
        parent_metadata: Mapping[str, object],
    ) -> None:
        """Budget gate -> shadow call -> blind judge -> one attempt row. The prisma gate
        sits above the dispatch so no provider spend happens without a place to record
        the outcome, and the budget read lives here rather than in the success hook."""
        prisma: Final = self._prisma_provider()
        try:
            if prisma is None:
                return
            if await _key_or_team_is_over_budget(parent_metadata):
                return
            if job.max_budget is not None:
                try:
                    spend: Final = await self._read_job_spend(_job_spend_counter_key(job.id), job.spend, job.max_budget)
                except Exception as e:  # noqa: BLE001  # unverifiable budget: skip the sample rather than spend on it
                    verbose_logger.warning("shadow_eval: budget unverifiable for %s, sample skipped: %s", job.id, e)
                    return
                if spend >= job.max_budget:
                    return
            shadow: Final = await self._call_router_shadow(job.shadow_target, messages, shadow_params, parent_metadata)
        except Exception as e:  # noqa: BLE001  # detached task: nothing billed yet, record and never raise
            verbose_logger.debug("shadow_eval: pipeline failed for %s: %s", request_id, e)
            await self._record_attempt(
                prisma, job, request_id, control_tier, outcome="error", error=f"pipeline error: {e}"
            )
            return
        if isinstance(shadow, _CallFailure):
            await self._record_attempt(
                prisma, job, request_id, control_tier, outcome="error", error=shadow.error, shadow_cost=shadow.cost
            )
            return
        # From here the shadow call has billed, so every exit records its cost.
        try:
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
                    control_tier,
                    outcome="error",
                    error=verdict.error,
                    shadow=shadow,
                    judge_cost=verdict.cost,
                    shadow_cost=shadow.cost,
                )
                return
            await self._record_attempt(
                prisma,
                job,
                request_id,
                control_tier,
                outcome=verdict.preference,
                shadow=shadow,
                real_model=real_model,
                confidence=verdict.confidence,
                judge_cost=verdict.cost,
                shadow_cost=shadow.cost,
            )
        except Exception as e:  # noqa: BLE001  # detached task: the shadow call billed, record its cost, never raise
            verbose_logger.debug("shadow_eval: pipeline failed for %s: %s", request_id, e)
            await self._record_attempt(
                prisma,
                job,
                request_id,
                control_tier,
                outcome="error",
                error=f"pipeline error: {e}",
                shadow=shadow,
                shadow_cost=shadow.cost,
            )

    async def _record_attempt(
        self,
        prisma: "PrismaClient | None",
        job: ActiveShadowEvalJob,
        request_id: str,
        control_tier: str | None,
        *,
        outcome: str,
        shadow: _ShadowResponse | None = None,
        real_model: str = "",
        confidence: float | None = None,
        judge_cost: float = 0.0,
        shadow_cost: float = 0.0,
        error: str | None = None,
    ) -> None:
        if judge_cost + shadow_cost > 0:
            await self._write_job_spend(_job_spend_counter_key(job.id), judge_cost + shadow_cost)
        if prisma is None:
            return
        try:
            await prisma.db.litellm_shadowevalattempt.create(
                data={  # mutable-ok: Prisma payload
                    "job_id": job.id,
                    "request_id": request_id,
                    "outcome": outcome,
                    "tier": control_tier if job.direction == "reverse" else (shadow.tier if shadow else None),
                    "real_model": real_model or None,
                    "shadow_model": shadow.model if shadow else None,
                    "confidence": confidence,
                    "judge_cost": judge_cost,
                    "shadow_cost": shadow_cost,
                    "error": error[:_MAX_ERROR_CHARS] if error else None,
                }
            )
        except Exception as e:  # noqa: BLE001  # a lost row degrades sample size, nothing can disagree with it
            verbose_logger.debug("shadow_eval: attempt write failed for %s: %s", request_id, e)

    async def _call_router_shadow(
        self,
        target_model: str,
        messages: Sequence[Mapping[str, object]],
        shadow_params: Mapping[str, object],
        parent_metadata: Mapping[str, object],
    ) -> "_ShadowResponse | _CallFailure":
        """Send the prompt through the arm nobody was served: the auto-router under
        evaluation, or a reverse job's fixed baseline model. The metadata carries the
        shadowed key's identity (spend attribution) and receives a routing decision
        write-back, which a plain baseline model simply never makes."""
        router: Final = self._router_provider()
        if router is None:
            return _CallFailure("no router configured on this pod")
        shadow_metadata: Final[dict[str, object]] = (  # mutable-ok: router writes its routing decision back
            sanitized_forwardable_call_metadata(parent_metadata, SHADOW_EVAL_ROUTER_CALL_ORIGIN)
        )
        try:
            response: Final = await router.acompletion(
                model=target_model,
                messages=[  # mutable-ok: provider transforms rewrite messages in place, so the router gets its own copy
                    dict(m) for m in messages
                ],  # pyright: ignore[reportArgumentType]  # snapshot of the SDK's own message dicts
                metadata=shadow_metadata,
                num_retries=0,
                fallbacks=[],  # mutable-ok: SDK kwarg; a failed shadow is a recorded error, never a spend multiplier
                **shadow_params,
            )
        except Exception as e:  # noqa: BLE001  # provider errors become error rows, not crashes
            verbose_logger.debug("shadow_eval: router call failed: %s", e)
            return _CallFailure(f"shadow router call failed: {_failure_detail(e)}")
        text: Final = _chat_final_text(response)
        if not text:
            return _CallFailure("shadow router returned an empty response", cost=_call_cost(response))
        return _ShadowResponse(
            text=text,
            model=str(getattr(response, "model", None) or _routing_decision(shadow_metadata).get("routed_model") or ""),
            tier=_routed_tier(shadow_metadata),
            cost=_call_cost(response),
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
                response_format=PAIRWISE_JUDGE_RESPONSE_FORMAT,
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
            return _CallFailure(f"unparseable judge verdict: {e}", cost=_call_cost(response))
        return _JudgeVerdict(
            preference=_unmask_preference(verdict.preference, real_is_a),
            confidence=max(0.0, min(1.0, verdict.confidence)),
            cost=_call_cost(response),
        )


_EMPTY_JOBS: Final[Mapping[str, tuple[ActiveShadowEvalJob, ...]]] = MappingProxyType({})


def _default_prisma_provider() -> "PrismaClient | None":
    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        return None
    return prisma_client
