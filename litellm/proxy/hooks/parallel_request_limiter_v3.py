"""
This is a rate limiter implementation based on a similar one by Envoy proxy.

This is currently in development and not yet ready for production.
"""

import asyncio
import itertools
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence, Set
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
)

from fastapi import HTTPException
from pydantic import Field, TypeAdapter, ValidationError
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_503_SERVICE_UNAVAILABLE
from typing_extensions import NotRequired, ReadOnly

import litellm
from litellm import DualCache
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import Batch
from litellm.caching.redis_cache import log_redis_failure
from litellm.constants import (
    DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE,
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    NO_MAX_TOKENS_TPM_RESERVATION_FRACTION,
    RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY,
)
from litellm.cost_calculator import get_usage_object
from litellm.exceptions import ServiceUnavailableError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    ensure_response_additional_headers,
    response_has_hidden_params,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.model_latest_aliases import expand_model_latest_alias
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.litellm_core_utils.prompt_templates.tpm_media_utils import (
    split_media_for_tpm,
)
from litellm.litellm_core_utils.ptu_pricing import is_ptu_cost_attribution_enabled
from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias
from litellm.litellm_core_utils.token_counter import (
    estimate_media_block_tokens,
    offload_token_count,
    split_embedding_input_tokens,
)
from litellm.litellm_core_utils.total_deadline import total_timeout_held
from litellm.llms.azure.ptu_capacity import normalized_tokens
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    ESTIMATED_OUTPUT_TOKENS_FIELD,
    get_estimated_output_tokens,
    get_key_own_model_rate_limit,
    get_key_tag_rpm_limit,
    get_model_rate_limit_from_metadata,
)
from litellm.proxy.auth.budget_throttle import throttled_limit
from litellm.proxy.common_utils.callback_utils import get_model_group_from_litellm_kwargs
from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    PER_MODEL_RATE_LIMIT_DESCRIPTOR_KEYS,
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BATCH_ENQUEUED_REFUND_STATUSES,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenStore,
    batch_response_view,
    canonical_provider_batch_id,
)
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit, socket_session_usage
from litellm.proxy.spend_tracking.input_tokens import (
    inline_pdf_file_count,
    models_counted_for,
    openai_chat_pdf_file_tokens,
)
from litellm.redis.stores.connection import connect
from litellm.redis.stores.rate_limits import (
    RATE_LIMITS_TARGET,
    AtomicApplied,
    AtomicCounter,
    AtomicOverLimit,
    AtomicReply,
    CounterIncrement,
    GaugeOverLimit,
    LocalValue,
    RateLimitCounter,
    RateLimitScope,
    RateLimitStore,
    WindowCount,
)
from litellm.responses.litellm_completion_transformation.compaction_items import billable_input_since_newest_compaction
from litellm.types.extraction import ExtractionResponse
from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject, ResponseAPIUsage
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ModelResponse,
    RerankResponse,
    TextCompletionResponse,
    Usage,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.router import Router
    from litellm.types.agents import AgentResponse

    Span = _Span
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any


_REQUEST_RATE_LIMIT_DATA: Final = TypeAdapter(Mapping[str, object])
_MCP_KEY_RPM_LIMITS: Final[TypeAdapter[dict[str, int] | None]] = TypeAdapter(dict[str, int] | None)


@dataclass(frozen=True, slots=True)
class _ReconciledUsage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int

    @property
    def billable_input_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)


@dataclass(frozen=True, slots=True)
class RateLimitedModel:
    requested: str
    group: str

    def limit_in(self, limits: Mapping[str, int] | None) -> int | None:
        if limits is None:
            return None
        requested_limit: Final = limits.get(self.requested)
        return requested_limit if requested_limit is not None else limits.get(self.group)


def dynamic_rate_limiter_covers(model_group: str) -> bool:
    from litellm.proxy.hooks.dynamic_rate_limiter_v3 import PROXY_DynamicRateLimitHandlerV3
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return False
    group: Final = llm_router.registry().group(model_group)
    if group is None or (group.tpm is None and group.rpm is None):
        return False
    return any(
        callback == "dynamic_rate_limiter_v3" or isinstance(callback, PROXY_DynamicRateLimitHandlerV3)
        for callback in litellm.callbacks
    )


def inherited_team_model_limit(
    user_api_key_dict: UserAPIKeyAuth,
    model: RateLimitedModel,
    rate_limit_key: Literal["model_rpm_limit", "model_tpm_limit"],
) -> int | None:
    """The team's per-model limit a key inherits, or None when the key sets its own limit for that model."""
    team_limit: Final = model.limit_in(
        get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", rate_limit_key)
    )
    if (
        team_limit is None
        or model.limit_in(get_key_own_model_rate_limit(user_api_key_dict, rate_limit_key)) is not None
    ):
        return None
    return team_limit


TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE: Final = "team_model_limit_max_borrow_multiple"
MAX_BORROW_MULTIPLE_CEILING: Final = 1_000_000
_MAX_BORROW_MULTIPLE: Final[TypeAdapter[float]] = TypeAdapter(
    Annotated[float, Field(ge=1, le=MAX_BORROW_MULTIPLE_CEILING, allow_inf_nan=False)]
)


@dataclass(frozen=True, slots=True)
class BorrowingTeamModelLimits:
    """Inherited team per-model limits the dynamic rate limiter enforces at saturation instead of the regular limiter."""

    tpm: int | None
    rpm: int | None
    max_borrow_multiple: float | None = None

    def share(self, rate_limit_type: str) -> int | None:
        return self.rpm if rate_limit_type == "requests" else self.tpm

    def borrow_ceiling(self) -> "BorrowingTeamModelLimits | None":
        """The most the team may use while borrowing, or None when it may borrow up to the model's own limit."""
        if self.max_borrow_multiple is None:
            return None
        return BorrowingTeamModelLimits(
            tpm=None if self.tpm is None else int(self.tpm * self.max_borrow_multiple),
            rpm=None if self.rpm is None else int(self.rpm * self.max_borrow_multiple),
        )


NO_BORROWING: Final = BorrowingTeamModelLimits(tpm=None, rpm=None)


def team_model_limit_max_borrow_multiple(user_api_key_dict: UserAPIKeyAuth) -> float | None:
    raw: Final[object] = (user_api_key_dict.team_metadata or {}).get(TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE)
    if raw is None:
        return None
    try:
        return _MAX_BORROW_MULTIPLE.validate_python(raw)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Ignoring %s=%r on team %s: expected a number from 1 to %d, so the team borrows up to the model limit",
            TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE,
            raw,
            user_api_key_dict.team_id,
            MAX_BORROW_MULTIPLE_CEILING,
        )
        return None


def borrowing_team_model_limits(user_api_key_dict: UserAPIKeyAuth, model: RateLimitedModel) -> BorrowingTeamModelLimits:
    settings: Final = litellm.priority_reservation_settings
    if (
        settings is None
        or not settings.team_model_limit_borrows_idle_capacity_from_other_teams
        or user_api_key_dict.team_id is None
    ):
        return NO_BORROWING
    limits: Final = BorrowingTeamModelLimits(
        tpm=inherited_team_model_limit(user_api_key_dict, model, "model_tpm_limit"),
        rpm=inherited_team_model_limit(user_api_key_dict, model, "model_rpm_limit"),
    )
    if limits == NO_BORROWING or not dynamic_rate_limiter_covers(model.group):
        return NO_BORROWING
    return BorrowingTeamModelLimits(
        tpm=limits.tpm, rpm=limits.rpm, max_borrow_multiple=team_model_limit_max_borrow_multiple(user_api_key_dict)
    )


def _resolve_media_models(router: "Router | None", model: str, team_id: str | None) -> tuple[str, ...]:
    if router is None:
        return (model,)
    return tuple(dict.fromkeys(router.registry().names(model, kind="litellm_base_models", team_id=team_id))) or (model,)


def _media_models(
    resolver: Callable[[str, str | None], tuple[str, ...]] | None, model: str, team_id: str | None
) -> tuple[str, ...]:
    return resolver(model, team_id) if resolver is not None else (model,)


def _resolve_media_models_via_proxy_router(model: str, team_id: str | None) -> tuple[str, ...]:
    from litellm.proxy.proxy_server import llm_router

    return _resolve_media_models(llm_router, model, team_id)


def _resolve_model_group_alias_via_proxy_router(model: str) -> str | None:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return None
    return resolve_model_group_alias(llm_router.registry().aliases(), model)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # registry().aliases() is the untyped model_group_alias dict


def _resolve_counted_models_via_proxy_router(model: str, team_id: str | None) -> Sequence[str]:
    from litellm.proxy.proxy_server import llm_router

    return models_counted_for(model, llm_router, team_id)


def _resolve_ptu_team_ceiling_via_proxy_router(team_id: str, model_group: str) -> PTUTeamCeiling | None:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return None
    ceiling: Final = llm_router.health().team_ptu_ceiling(team_id, model_group)
    return ceiling if ceiling is not None and is_ptu_cost_attribution_enabled() else None


FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_SETTING: Final = "fail_closed_rate_limit_enforcement"
RATE_LIMIT_UNVERIFIABLE_MESSAGE: Final = (
    "Rate limit enforcement unavailable: request counters could not be verified against Redis, and "
    "fail_closed_rate_limit_enforcement is enabled, so the request was rejected to avoid exceeding the "
    "configured rate limit. Retry shortly."
)


class RateLimitUnverifiableError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": RATE_LIMIT_UNVERIFIABLE_MESSAGE},
        )


class PTUShareExceededError(HTTPException):
    def __init__(self, ceiling: PTUTeamCeiling, needed_tokens: int) -> None:
        super().__init__(
            status_code=HTTP_400_BAD_REQUEST,
            detail={
                "error": f"This request needs {needed_tokens} normalized tokens, more than your team's PTU share "
                f"of {ceiling.model_group} allows in a minute ({ceiling.tpm_limit}), so it can never be admitted. "
                "Lower max_tokens or shorten the prompt."
            },
        )


_FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_FLAG: Final = TypeAdapter(bool | None)


def fail_closed_rate_limit_enforcement_enabled(general_settings: Mapping[str, object]) -> bool:
    raw_value: Final = general_settings.get(FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_SETTING)
    try:
        return _FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_FLAG.validate_python(raw_value) is True
    except ValidationError:
        verbose_proxy_logger.warning(
            "general_settings.%s=%r is not a boolean, treating it as disabled",
            FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_SETTING,
            raw_value,
        )
        return False


def _fail_closed_rate_limit_enforcement_from_general_settings() -> bool:
    from litellm.proxy.proxy_server import general_settings

    return fail_closed_rate_limit_enforcement_enabled(general_settings)


def _force_redis_hash_tag_grouping_from_litellm_settings() -> bool:
    from litellm import force_redis_hash_tag_grouping

    return force_redis_hash_tag_grouping is True


# TPM token reservation tuning constants.
# When max_tokens is not specified in the request we still need to reserve
# *some* output budget; these define that fallback estimate.
DEFAULT_MAX_TOKENS_ESTIMATE: Final = 4096
DEFAULT_CHARS_PER_TOKEN: Final = 4
# Both embeddings and the Responses API put their prompt in data["input"],
# but only embeddings have no output tokens. Every "is this an embedding"
# check on data["input"] must exclude these call types, or a Responses call
# gets misclassified as an embedding and skips output-token reservation/caps.
RESPONSES_API_CALL_TYPES: Final = ("aresponses", "responses")
CHAT_COMPLETION_API_CALL_TYPES: Final = (CallTypes.completion.value, CallTypes.acompletion.value)
EMBEDDING_API_CALL_TYPES: Final = ("aembedding", "embedding")
TEXT_COMPLETION_API_CALL_TYPES: Final = ("atext_completion", "text_completion")
RERANK_API_CALL_TYPES: Final = (CallTypes.rerank.value, CallTypes.arerank.value)
EXTRACTION_API_CALL_TYPES: Final = (CallTypes.extraction.value, CallTypes.aextraction.value)
GOOGLE_GENAI_NATIVE_CALL_TYPES: Final = (
    CallTypes.generate_content.value,
    CallTypes.agenerate_content.value,
    CallTypes.generate_content_stream.value,
    CallTypes.agenerate_content_stream.value,
)
RESPONSES_API_MIN_OUTPUT_TOKENS: Final = 16
# litellm.token_counter has no per-type handling for "input_audio" content
# blocks (unlike images, which use use_default_image_token_count) -- it
# silently contributes 0 tokens for them. When the block carries a base64
# payload, the estimate is derived from the decoded byte count; when the
# block is a reference without a payload (or the payload is missing), this
# flat per-block floor is used instead.
DEFAULT_AUDIO_TOKEN_ESTIMATE: Final = 300
# Conservative bytes-per-token assumption for size-based audio estimation:
# equivalent to 8 kHz mono PCM-16 (16 000 bytes/s) at 10 tokens/s. Choosing
# the lowest reasonable bitrate means we never under-reserve for higher-
# quality audio recorded at the same wall-clock duration.
_AUDIO_BYTES_PER_TOKEN: Final = 1600
# Descriptor "key" values for project-scoped ITPM/OTPM. Distinct from
# "model_per_project" (the combined-TPM descriptor) so both can be enforced
# on the same project+model simultaneously without colliding on cache keys.
PROJECT_ITPM_DESCRIPTOR_KEY: Final = "model_per_project_itpm"
PROJECT_OTPM_DESCRIPTOR_KEY: Final = "model_per_project_otpm"
PTU_TEAM_DESCRIPTOR_KEY: Final = "model_per_team_ptu"
# How long an acquired slot counts toward the in-flight total before it is
# considered leaked (worker crashed without any release callback firing) and
# pruned. Also the longest request duration the gauge can track: a request
# running longer than this stops occupying its slot.
PARALLEL_REQUEST_SLOT_TTL_SECONDS: Final = 3600


CacheCounterValue: TypeAlias = int | float | str

CacheCounterValues: TypeAlias = Sequence[CacheCounterValue | None]


CounterState: TypeAlias = tuple[CacheCounterValue | None, CacheCounterValue | None]
"""A counter's window start and its count, either one absent."""

ReservationWindowIdentity: TypeAlias = tuple[RateLimitCounter, str, Literal["redis", "local"]]


def _scalar(value: LocalValue | None) -> CacheCounterValue | None:
    if isinstance(value, dict):
        raise TypeError("a rate-limit counter in memory holds a slot registry")
    return value


@dataclass(frozen=True, slots=True)
class WindowGuardedIncrement:
    """A reservation settlement that applies only while the window it was reserved in is still the open one."""

    increment: CounterIncrement
    expected_window_start: str
    reservation_backend: Literal["redis", "local"]


ReservationOperation: TypeAlias = CounterIncrement | WindowGuardedIncrement


class RateLimitDescriptorRateLimitObject(TypedDict, total=False):
    requests_per_unit: int | None
    tokens_per_unit: int | None
    max_parallel_requests: int | None
    window_size: int | None


class RateLimitDescriptor(TypedDict):
    key: str
    value: str
    rate_limit: RateLimitDescriptorRateLimitObject | None


def _minimum_descriptor_limit(
    descriptors: Sequence[RateLimitDescriptor],
    metric: Literal["requests_per_unit", "tokens_per_unit", "max_parallel_requests"],
) -> int | None:
    return min(
        (limit for descriptor in descriptors if (limit := (descriptor["rate_limit"] or {}).get(metric)) is not None),
        default=None,
    )


def _merge_rate_limit_descriptors(descriptors: Sequence[RateLimitDescriptor]) -> RateLimitDescriptor:
    first: Final = descriptors[0]
    if len(descriptors) == 1:
        return first
    return RateLimitDescriptor(
        key=first["key"],
        value=first["value"],
        rate_limit=RateLimitDescriptorRateLimitObject(
            requests_per_unit=_minimum_descriptor_limit(descriptors, "requests_per_unit"),
            tokens_per_unit=_minimum_descriptor_limit(descriptors, "tokens_per_unit"),
            max_parallel_requests=_minimum_descriptor_limit(descriptors, "max_parallel_requests"),
            window_size=(first["rate_limit"] or {}).get("window_size"),
        ),
    )


def _coalesce_rate_limit_descriptors(descriptors: Sequence[RateLimitDescriptor]) -> Sequence[RateLimitDescriptor]:
    identities: Final = tuple(dict.fromkeys((descriptor["key"], descriptor["value"]) for descriptor in descriptors))
    if len(identities) == len(descriptors):
        return descriptors
    return tuple(
        _merge_rate_limit_descriptors(
            tuple(descriptor for descriptor in descriptors if (descriptor["key"], descriptor["value"]) == identity)
        )
        for identity in identities
    )


class ParallelRequestGauge(TypedDict):
    scope: RateLimitScope
    limit: int
    descriptor_key: str


class ParallelSlotAcquisition(TypedDict):
    slot_id: str
    gauges: tuple[RateLimitScope, ...]


class RateLimitStatus(TypedDict):
    code: str
    current_limit: int
    limit_remaining: int
    rate_limit_type: Literal["requests", "tokens", "max_parallel_requests"]
    descriptor_key: str
    # Populated by the atomic_check_and_increment_by_n and windowed
    # sliding-window paths. A caller matching a status back to its
    # descriptor must key on (descriptor_key, descriptor_value) when this
    # is present, not descriptor_key alone -- e.g. a batch charging several
    # models' project ITPM/OTPM in one call, or a request carrying multiple
    # rate-limited tags, produces statuses sharing the same descriptor_key.
    descriptor_value: NotRequired[ReadOnly[str]]


class RateLimitResponse(TypedDict):
    overall_code: str
    statuses: list[RateLimitStatus]
    reservation_windows: NotRequired[ReadOnly[frozenset[ReservationWindowIdentity]]]


class RateLimitResponseWithDescriptors(TypedDict):
    descriptors: list[RateLimitDescriptor]
    response: RateLimitResponse


class _RateLimitDescriptorSink(Protocol):
    def append(self, descriptor: RateLimitDescriptor, /) -> None: ...


class WindowKeyMetadata(TypedDict):
    requests_limit: int | None
    tokens_limit: int | None
    window_size: int
    descriptor_key: str
    descriptor_value: ReadOnly[str]


class AtomicCounterMeta(TypedDict):
    descriptor_key: str
    descriptor_value: ReadOnly[str]
    current_limit: int
    rate_limit_type: Literal["requests", "tokens"]
    counter: ReadOnly[RateLimitCounter]
    increment: int
    ttl: int
    window_size: int


class AtomicCounterState(TypedDict):
    window_expired: bool
    current: int
    window_start: ReadOnly[str]


DescriptorAtomicGroup: TypeAlias = tuple[list[AtomicCounter], list[AtomicCounterMeta]]


class CallTypeRateLimiter(Protocol):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: str,
    ) -> Exception | str | dict[str, object] | None: ...


@dataclass(frozen=True, slots=True)
class BorrowingTeamModelHold:
    """A router fallback target's team share with the tokens held on it, settled to actual usage or given back.

    An idle target holds no descriptor: its usage is not charged to the share, and settlement still finds the hold so
    the admitting model's reservation is given back instead of taking the target's tokens.
    """

    descriptor: RateLimitDescriptor | None
    reserved_tokens: int = 0
    window: ReservationWindowIdentity | None = None
    cap: RateLimitDescriptor | None = None
    cap_reserved_tokens: int = 0
    cap_window: ReservationWindowIdentity | None = None
    share: RateLimitDescriptor | None = None


@dataclass(slots=True)
class RequestRateLimiterStash:
    """
    Per-request bookkeeping the pre-call hook hands to the success/failure/
    disconnect callbacks. Lives on a ContextVar instead of the request body so
    it never reaches provider-facing ``metadata`` channels.

    A single mutable instance is shared by every context forked from the
    request task (the SDK call, streaming generators, and the logging worker's
    captured context all see the same object), which is what makes the
    released flags and ``parallel_slot`` clearing effective
    across sibling callbacks: the first release wins, later callbacks observe
    the cleared state.

    Because the stash is context-inherited, nested LiteLLM calls made inside
    the request (LLM-judge guardrails, silent experiments) would also see it
    from their own logging callbacks. ``owner_litellm_call_id`` pins the stash
    to the proxy request's ``litellm_call_id`` so those callbacks can tell the
    owning request's events apart from a nested call's: router retries and
    fallbacks reuse the request's call id and keep access, while nested calls
    mint fresh ids and are ignored.
    """

    owner_litellm_call_id: str | None = None
    call_type: str | None = None
    rate_limit_response: RateLimitResponse | None = None
    parallel_slot: ParallelSlotAcquisition | None = None
    parallel_slot_reused_at: datetime | None = None
    parallel_slot_release_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    reserved_tokens: int = 0
    reserved_model: RateLimitedModel | None = None
    reserved_scopes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    fallback_models: frozenset[str] = field(default_factory=frozenset)
    ptu_ceiling: PTUTeamCeiling | None = None
    ptu_reserved_tokens: int = 0
    ptu_reserved_window_identities: frozenset[ReservationWindowIdentity] = field(default_factory=frozenset)
    itpm_reserved_tokens: int = 0
    itpm_reserved_scopes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    itpm_reserved_window_identities: frozenset[ReservationWindowIdentity] = field(default_factory=frozenset)
    otpm_reserved_tokens: int = 0
    otpm_reserved_scopes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    otpm_reserved_window_identities: frozenset[ReservationWindowIdentity] = field(default_factory=frozenset)
    batch_enqueued_reservation: BatchEnqueuedTokenReservation | None = None
    batch_tpd_refund_ops: tuple[ReservationOperation, ...] = ()
    borrowing_team_model_limit: RateLimitDescriptor | None = None
    borrowing_team_model_reserved_tokens: int = 0
    borrowing_team_model_reserved_window: ReservationWindowIdentity | None = None
    borrowing_team_model_share: RateLimitDescriptor | None = None
    borrowing_team_model_cap: RateLimitDescriptor | None = None
    borrowing_team_model_cap_reserved_tokens: int = 0
    borrowing_team_model_cap_reserved_window: ReservationWindowIdentity | None = None
    borrowing_team_model_fallback_limits: Mapping[str, BorrowingTeamModelHold] = field(
        default_factory=lambda: MappingProxyType({})
    )
    priority_class_share: RateLimitDescriptor | None = None
    dynamic_limiter_priority: str | None = None
    reservation_released: bool = False
    io_reservation_released: bool = False
    tpm_limited_tags: frozenset[str] = field(default_factory=frozenset)
    socket_agent_windows: tuple[RateLimitDescriptor, ...] = ()

    def clear_reserved_tokens(self) -> None:
        self.reserved_tokens = 0
        self.ptu_reserved_tokens = 0
        self.itpm_reserved_tokens = 0
        self.otpm_reserved_tokens = 0

    def mark_reservation_released(self) -> None:
        self.reservation_released = True
        self.io_reservation_released = True

    def discard_released_tpm_reservation(self) -> None:
        if self.reservation_released:
            self.reserved_tokens = 0
            self.ptu_reserved_tokens = 0
            self.reservation_released = False

    def discard_released_io_reservation(self) -> None:
        if self.io_reservation_released:
            self.itpm_reserved_tokens = 0
            self.otpm_reserved_tokens = 0
            self.io_reservation_released = False


@dataclass(frozen=True, slots=True)
class CounterRefund:
    counter: RateLimitCounter
    window_start: str
    increment: int


@dataclass(frozen=True, slots=True)
class TagRateLimit:
    rpm_limit: int | None
    tpm_limit: int | None


class TagRateLimitResolver(Protocol):
    def __call__(self, tag_names: Sequence[str], /) -> Awaitable[Mapping[str, TagRateLimit]]: ...


def _tag_rate_limit_descriptor(tag: str, limit: TagRateLimit, window_size: int) -> RateLimitDescriptor:
    rate_limit: Final[RateLimitDescriptorRateLimitObject] = {
        "requests_per_unit": limit.rpm_limit,
        "tokens_per_unit": limit.tpm_limit,
        "window_size": window_size,
    }
    return RateLimitDescriptor(key="tag", value=tag, rate_limit=rate_limit)


async def resolve_tag_rate_limits_from_db(tag_names: Sequence[str]) -> Mapping[str, TagRateLimit]:
    from litellm.proxy.auth.auth_checks import get_tag_objects_batch
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None or not tag_names:
        return MappingProxyType({})
    tag_objects: Final = await get_tag_objects_batch(
        tag_names=tag_names,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    return MappingProxyType(
        {
            tag_name: TagRateLimit(rpm_limit=budget.rpm_limit, tpm_limit=budget.tpm_limit)
            for tag_name, tag_object in tag_objects.items()
            if (budget := tag_object.litellm_budget_table) is not None
            and (budget.rpm_limit is not None or budget.tpm_limit is not None)
        }
    )


_request_stash: Final[ContextVar[RequestRateLimiterStash | None]] = ContextVar(
    "litellm_v3_rate_limiter_request_stash", default=None
)


def get_request_stash() -> RequestRateLimiterStash | None:
    return _request_stash.get()


def get_or_create_request_stash() -> RequestRateLimiterStash:
    stash = _request_stash.get()
    if stash is None:
        stash = RequestRateLimiterStash()
        _request_stash.set(stash)
    return stash


def claim_request_stash_for_data(data: Mapping[str, object]) -> RequestRateLimiterStash:
    stash: Final = get_or_create_request_stash()
    owner_call_id: Final = data.get("litellm_call_id")
    if isinstance(owner_call_id, str):
        stash.owner_litellm_call_id = owner_call_id
    return stash


def _ended_before_slot_reuse(ended_at: datetime | None, reused_at: datetime | None) -> bool:
    if ended_at is None or reused_at is None or (ended_at.tzinfo is None) != (reused_at.tzinfo is None):
        return False
    return ended_at <= reused_at


def _callback_kwargs_mapping(kwargs: object) -> Mapping[str, object]:
    try:
        return _REQUEST_RATE_LIMIT_DATA.validate_python(kwargs)
    except ValidationError:
        return MappingProxyType({})


def get_request_stash_for_call(litellm_call_id: str | None) -> RequestRateLimiterStash | None:
    stash: Final = _request_stash.get()
    if stash is None:
        return None
    if stash.owner_litellm_call_id is None or litellm_call_id is None:
        return stash
    return stash if litellm_call_id == stash.owner_litellm_call_id else None


def _call_id_from_callback_kwargs(kwargs: object) -> str | None:
    if not isinstance(kwargs, dict):
        return None
    call_id: Final = kwargs.get("litellm_call_id")
    return call_id if isinstance(call_id, str) else None


def _socket_connection_call_id(kwargs: Mapping[str, object]) -> object:
    litellm_params: Final = as_str_mapping(kwargs.get("litellm_params"))
    litellm_metadata: Final = None if litellm_params is None else as_str_mapping(litellm_params.get("litellm_metadata"))
    return None if litellm_metadata is None else litellm_metadata.get(RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY)


def _is_socket_turn(kwargs: Mapping[str, object]) -> bool:
    if kwargs.get("call_type") == CallTypes.aresponses_websocket.value:
        return True
    connection_stash: Final = get_request_stash()
    connection_call_id: Final = _socket_connection_call_id(kwargs)
    return (
        connection_stash is not None
        and connection_call_id is not None
        and connection_call_id == connection_stash.owner_litellm_call_id
    )


def callback_kwargs_mapping(kwargs: object) -> Mapping[str, object]:
    """A logging callback's kwargs as a typed mapping, empty when they are not a mapping."""
    return _callback_kwargs_mapping(kwargs)


def request_stash_for_callback(kwargs: Mapping[str, object]) -> RequestRateLimiterStash | None:
    """The stash of the request a logging callback settles, or the connection's for a Responses WebSocket turn.

    A nested call (an LLM judge, a silent experiment) mints its own call id and gets None.
    """
    call_id: Final = kwargs.get("litellm_call_id")
    call_stash: Final = get_request_stash_for_call(call_id if isinstance(call_id, str) else None)
    if call_stash is not None:
        return call_stash
    return get_request_stash() if _is_socket_turn(kwargs) else None


_AGENT_DESCRIPTOR_KEYS: Final = frozenset({"agent", "agent_session"})

_SOCKET_TURN_UNCOUNTED_DESCRIPTOR_KEYS: Final = frozenset(
    {PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY, PTU_TEAM_DESCRIPTOR_KEY}
)


def _socket_turn_tpm_windows(descriptors: Sequence[RateLimitDescriptor]) -> tuple[RateLimitDescriptor, ...]:
    return tuple(
        RateLimitDescriptor(
            key=descriptor["key"],
            value=descriptor["value"],
            rate_limit=RateLimitDescriptorRateLimitObject(
                tokens_per_unit=tokens_per_unit, window_size=(descriptor.get("rate_limit") or {}).get("window_size")
            ),
        )
        for descriptor in descriptors
        if descriptor["key"] not in _SOCKET_TURN_UNCOUNTED_DESCRIPTOR_KEYS
        and (tokens_per_unit := (descriptor.get("rate_limit") or {}).get("tokens_per_unit")) is not None
    )


def _unmodeled_response_usage(kwargs: Mapping[str, object], response_obj: object) -> Usage | None:
    combined_usage: Final = kwargs.get("combined_usage_object")
    if isinstance(combined_usage, Usage):
        return combined_usage
    return socket_session_usage(kwargs, response_obj)


def _tpm_limited_tags(call_stash: RequestRateLimiterStash | None, kwargs: Mapping[str, object]) -> frozenset[str]:
    if call_stash is not None:
        return call_stash.tpm_limited_tags
    connection_stash: Final = get_request_stash()
    if connection_stash is None or not _is_socket_turn(kwargs):
        return frozenset()
    return connection_stash.tpm_limited_tags


def _parse_output_cap_value(raw_value: object) -> int | None:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, str)):
        return None
    try:
        return int(float(raw_value))
    except (ValueError, OverflowError):
        return None


class _PROXY_MaxParallelRequestsHandler_v3(CustomLogger):
    def __init__(
        self,
        internal_usage_cache: InternalUsageCache,
        time_provider: Callable[[], datetime] | None = None,
        tag_rate_limit_resolver: TagRateLimitResolver = resolve_tag_rate_limits_from_db,
        model_group_resolver: Callable[[str], str | None] = _resolve_model_group_alias_via_proxy_router,
        counted_models_resolver: Callable[[str, str | None], Sequence[str]] = _resolve_counted_models_via_proxy_router,
        media_model_resolver: Callable[[str, str | None], tuple[str, ...]] = _resolve_media_models_via_proxy_router,
        ptu_team_ceiling_resolver: Callable[
            [str, str], PTUTeamCeiling | None
        ] = _resolve_ptu_team_ceiling_via_proxy_router,
        fail_closed_resolver: Callable[[], bool] = _fail_closed_rate_limit_enforcement_from_general_settings,
        force_hash_tag_grouping_resolver: Callable[[], bool] = _force_redis_hash_tag_grouping_from_litellm_settings,
        rate_limit_store: RateLimitStore | None = None,
    ):
        self.internal_usage_cache = internal_usage_cache
        self._time_provider = time_provider or datetime.now
        self._tag_rate_limit_resolver = tag_rate_limit_resolver
        self._model_group_resolver = model_group_resolver
        self._counted_models_resolver = counted_models_resolver
        self._media_model_resolver = media_model_resolver
        self._ptu_team_ceiling_resolver = ptu_team_ceiling_resolver
        self._fail_closed_resolver = fail_closed_resolver
        self._force_hash_tag_grouping_resolver = force_hash_tag_grouping_resolver
        self.rate_limits: Final = rate_limit_store or RateLimitStore(
            connect(internal_usage_cache.dual_cache.redis_cache).rate_limits(
                internal_usage_cache.dual_cache.default_in_memory_ttl
            )
        )

        self.window_size = int(os.getenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", 60))

        # When disabled, TPM is enforced post-call from actual usage (pre-v1.82
        # behavior) instead of reserving an estimated budget upfront, shedding
        # the extra per-request Redis Lua round-trip and the global-lock
        # in-memory fallback that the reservation path incurs.
        self.tpm_reservation_enabled = os.getenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "true").lower() == "true"

        # Batch rate limiter (lazy loaded)
        self._batch_rate_limiter: CallTypeRateLimiter | None = None
        self.batch_enqueued_token_store = BatchEnqueuedTokenStore(internal_usage_cache=internal_usage_cache)

        # Serializes multi-phase check+increment sequences (batch + dynamic
        # limiters) within this process to close the TOCTOU window between
        # read-only check and counter increment. Multi-replica deployments
        # additionally rely on Redis Lua atomicity for cross-process safety.
        #
        # Coarse granularity: this single lock serializes ALL atomic check+
        # increment operations across batch and dynamic limiters on this
        # instance. A slow batch input-file fetch (which happens upstream of
        # the lock) does not block here, but Redis Lua latency does. If
        # contention shows up under load (visible as p99 latency spikes
        # correlated with batch traffic), shard to a per-descriptor-key lock
        # via a `weakref.WeakValueDictionary[str, asyncio.Lock]`. Punted as a
        # follow-up because Lua dominates wall-time and the lock is held for
        # one round-trip.
        self._check_and_increment_lock = asyncio.Lock()

    def _get_batch_rate_limiter(self) -> CallTypeRateLimiter | None:
        """Get or lazy-load the batch rate limiter."""
        if self._batch_rate_limiter is None:
            try:
                from litellm.proxy.hooks.batch_rate_limiter import (
                    PROXY_BatchRateLimiter,
                )

                self._batch_rate_limiter = PROXY_BatchRateLimiter(
                    internal_usage_cache=self.internal_usage_cache,
                    parallel_request_limiter=self,
                    time_provider=self._time_provider,
                )
            except Exception as e:
                verbose_proxy_logger.debug("Could not load batch rate limiter: %s", e)
        return self._batch_rate_limiter

    def _get_current_time(self) -> datetime:
        """Return the current time for rate limiting calculations."""
        return self._time_provider()

    @staticmethod
    def no_max_tokens_output_floor(
        min_configured_tpm_limit: int | None,
    ) -> int:
        """Output-budget floor used when the request omits max_tokens.

        Capped at a fraction of the smallest configured TPM limit so a small
        per-tenant cap can't be tripped by the floor alone. Returns the
        baseline floor when no limit is provided.
        """
        baseline: Final = int(DEFAULT_MAX_TOKENS_ESTIMATE * NO_MAX_TOKENS_TPM_RESERVATION_FRACTION)
        if min_configured_tpm_limit is None:
            return baseline
        return min(baseline, max(1, int(min_configured_tpm_limit * NO_MAX_TOKENS_TPM_RESERVATION_FRACTION)))

    @staticmethod
    def _is_embedding_request(data: object, call_type: str | None) -> bool:
        if call_type in EMBEDDING_API_CALL_TYPES:
            return True
        if call_type in RESPONSES_API_CALL_TYPES:
            return False
        if call_type:
            return False
        if not isinstance(data, dict):
            return False
        return data.get("input") is not None

    @staticmethod
    def _translate_google_genai_native_request(
        data: object,
        call_type: str | None,
    ) -> Mapping[str, object] | None:
        contents: Final = data.get("contents") if isinstance(data, dict) else None
        if (
            not isinstance(data, dict)
            or call_type not in GOOGLE_GENAI_NATIVE_CALL_TYPES
            or not isinstance(contents, (dict, list))
        ):
            return None
        from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter

        config: Final = data.get("config") if "config" in data else data.get("generationConfig")
        return GoogleGenAIAdapter().translate_generate_content_to_completion(
            model=data.get("model") if isinstance(data.get("model"), str) else "",
            contents=contents,
            config=config if isinstance(config, dict) else None,
            systemInstruction=data.get("systemInstruction"),
            system_instruction=data.get("system_instruction"),
            tools=data.get("tools"),
            toolConfig=data.get("toolConfig"),
            tool_config=data.get("tool_config"),
        )

    @staticmethod
    def _get_explicit_output_cap(data: object, call_type: str | None) -> int | None:
        if not isinstance(data, dict):
            return None
        if call_type in GOOGLE_GENAI_NATIVE_CALL_TYPES:
            config: Final = data.get("config") if "config" in data else data.get("generationConfig")
            google_cap_values: Final = tuple(
                parsed
                for field in ("maxOutputTokens", "max_output_tokens")
                if isinstance(config, dict)
                for parsed in (_parse_output_cap_value(config.get(field)),)
                if parsed is not None
            )
            return max(google_cap_values, default=None)
        if call_type in RESPONSES_API_CALL_TYPES:
            responses_cap: Final = _parse_output_cap_value(data.get("max_output_tokens"))
            if responses_cap is None:
                return None
            return max(RESPONSES_API_MIN_OUTPUT_TOKENS, responses_cap)
        if call_type in EMBEDDING_API_CALL_TYPES:
            return None
        fields: Final = (
            ("max_tokens", "max_completion_tokens")
            if call_type
            else ("max_tokens", "max_completion_tokens", "max_output_tokens")
        )
        output_cap_values: Final = tuple(
            parsed for field in fields for parsed in (_parse_output_cap_value(data.get(field)),) if parsed is not None
        )
        return max(output_cap_values, default=None)

    @classmethod
    def _has_explicit_output_cap(cls, data: object, call_type: str | None) -> bool:
        """Whether the caller explicitly set an output-token cap.

        Checked via ``is not None`` (not truthiness) so an explicit 0 --
        a legitimate zero-output request -- counts as explicit.
        """
        return cls._get_explicit_output_cap(data, call_type) is not None

    @staticmethod
    def get_output_candidate_count(data: object, call_type: str | None = None) -> int:
        if not isinstance(data, Mapping):
            return 1
        config: Final = (
            (data.get("config") if "config" in data else data.get("generationConfig"))
            if call_type in GOOGLE_GENAI_NATIVE_CALL_TYPES
            else None
        )
        candidate_values: Final = (
            data.get("n"),
            data.get("best_of"),
            config.get("candidateCount") if isinstance(config, dict) else None,
            config.get("candidate_count") if isinstance(config, dict) else None,
        )
        candidate_count = 1  # rebind-ok: running maximum across candidate-count aliases
        for value in candidate_values:
            try:
                candidate_count = max(candidate_count, int(value or 1))
            except (TypeError, ValueError, OverflowError):
                continue
        return candidate_count

    @staticmethod
    def _apply_implicit_output_cap(
        data: object,
        min_configured_limit: int | None,
        call_type: str | None,
        configured_output_tokens: int | None = None,
        endpoint_type: EndpointType = EndpointType.GENERIC,
    ) -> None:
        """Hard-cap generation length when the request has no explicit cap.

        Guards against an unbounded response overshooting a small TPM/OTPM
        budget before post-call reconciliation runs. Skips requests that
        already set an explicit cap and embeddings, which have no generation
        budget. The Responses API only honors ``max_output_tokens`` (its
        underlying chat-completion transformation ignores ``max_tokens``), so
        the cap must be written to that field for Responses call types.

        ``configured_output_tokens`` is the operator-declared per-tenant
        estimate; when it exceeds the safety floor, the cap is raised to that
        value instead of clamping every tenant to the same floor.
        """
        if not isinstance(data, dict):
            return
        base_capped_floor: Final = PROXY_MaxParallelRequestsHandler_v3.no_max_tokens_output_floor(min_configured_limit)
        capped_floor: Final = (
            max(base_capped_floor, RESPONSES_API_MIN_OUTPUT_TOKENS)
            if call_type in RESPONSES_API_CALL_TYPES
            else base_capped_floor
        )
        baseline_floor: Final = int(DEFAULT_MAX_TOKENS_ESTIMATE * NO_MAX_TOKENS_TPM_RESERVATION_FRACTION)
        is_embedding: Final = PROXY_MaxParallelRequestsHandler_v3._is_embedding_request(data, call_type)
        if (
            capped_floor >= baseline_floor
            or PROXY_MaxParallelRequestsHandler_v3._has_explicit_output_cap(data, call_type)
            or is_embedding
            or endpoint_type == EndpointType.DECISIONS
            or call_type in EXTRACTION_API_CALL_TYPES
        ):
            return
        effective_cap: Final = max(capped_floor, configured_output_tokens or 0)
        if call_type in GOOGLE_GENAI_NATIVE_CALL_TYPES:
            config_field: Final = "config" if "config" in data or "generationConfig" not in data else "generationConfig"
            config: Final = data.get(config_field)
            if config is None or isinstance(config, dict):
                data[config_field] = {  # rebind-ok: routed request needs cap
                    **(config or {}),
                    "maxOutputTokens": effective_cap,
                }
            return
        cap_field: Final = "max_output_tokens" if call_type in RESPONSES_API_CALL_TYPES else "max_tokens"
        existing_cap: Final = data.get(cap_field)
        if existing_cap is None or effective_cap < existing_cap:
            data[cap_field] = effective_cap  # rebind-ok: downstream routing requires the bounded output cap

    def estimate_request_tokens(
        self,
        data: object,
        model: str,
        min_configured_tpm_limit: int,
        call_type: str | None,
        team_id: str | None,
        rate_limit_type: str = "total",
    ) -> int:
        """The reservation this limiter would hold for the request, for another limiter settling the same call."""
        if rate_limit_type == "total":
            return self._estimate_tokens_for_request(
                data=data,
                model=model,
                min_configured_tpm_limit=min_configured_tpm_limit,
                call_type=call_type,
                team_id=team_id,
            )
        estimated_input, output_budget = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_tpm_limit,
            call_type=call_type,
            team_id=team_id,
            media_model_resolver=self._media_model_resolver,
        )
        return output_budget if rate_limit_type == "output" else estimated_input

    def bound_output_for_limit(self, data: object, min_configured_limit: int, call_type: str | None) -> None:
        """Cap a request without an explicit output cap the way this limiter does for its own small TPM limits."""
        self._apply_implicit_output_cap(data=data, min_configured_limit=min_configured_limit, call_type=call_type)

    def _estimate_tokens_for_request(
        self,
        data: object,
        model: str | None = None,
        min_configured_tpm_limit: int | None = None,
        call_type: str | None = None,
        configured_output_tokens: int | None = None,
        team_id: str | None = None,
    ) -> int:
        """
        Estimate total tokens this request will consume so we can reserve them
        upfront (input + output budget):
        estimated = input_tokens + max_tokens.

        Supports chat (messages), completions (prompt), embeddings (input),
        and the Responses API (also `input`, disambiguated from embeddings
        via ``call_type``).

        ``min_configured_tpm_limit`` is the smallest ``tokens_per_unit`` among
        the TPM-bearing descriptors this request will be charged against. When
        provided, the no-``max_tokens`` output-budget floor is capped at a
        fraction of that limit so small TPM caps remain usable. Omit to
        preserve the unconstrained floor.

        ``configured_output_tokens`` is the operator-declared estimate resolved
        from key or team metadata. When provided it replaces the heuristic
        floor entirely, so the reservation reflects what this tenant's model
        actually emits rather than one constant shared by every tenant.
        """
        estimated_input_tokens, max_tokens_estimate = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_tpm_limit,
            call_type=call_type,
            configured_output_tokens=configured_output_tokens,
            team_id=team_id,
            media_model_resolver=self._media_model_resolver,
        )
        total_estimated: Final = estimated_input_tokens + max_tokens_estimate

        verbose_proxy_logger.debug(
            "TPM reservation estimate: input=%s, max_tokens=%s, total=%s",
            estimated_input_tokens,
            max_tokens_estimate,
            total_estimated,
        )

        return total_estimated

    def _estimate_ptu_tokens_for_request(
        self,
        ceiling: PTUTeamCeiling | None,
        data: Mapping[str, object],
        min_configured_tpm_limit: int | None,
        call_type: str | None,
        configured_output_tokens: int | None,
        raw_estimate: int,
        team_id: str | None = None,
    ) -> int:
        """The team PTU ceiling counts Azure normalized tokens, so its reservation weighs the
        output budget the way the ceiling does instead of the raw sum the other scopes reserve."""
        if ceiling is None:
            return raw_estimate
        estimated_input_tokens, max_tokens_estimate = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_tpm_limit,
            call_type=call_type,
            configured_output_tokens=configured_output_tokens,
            team_id=team_id,
            media_model_resolver=self._media_model_resolver,
        )
        normalized: Final = normalized_tokens(
            ceiling, prompt_tokens=estimated_input_tokens, completion_tokens=max_tokens_estimate
        )
        return max(round(normalized), 1)

    def _estimate_input_and_output_tokens(
        self,
        data: object,
        min_configured_tpm_limit: int | None = None,
        call_type: str | None = None,
        configured_output_tokens: int | None = None,
        include_media: bool = True,
        team_id: str | None = None,
        media_model_resolver: Callable[[str, str | None], tuple[str, ...]] | None = None,
    ) -> tuple[int, int]:
        """
        Estimate input tokens and output (max_tokens) budget separately, so
        callers needing independent ITPM/OTPM reservations (rather than one
        combined TPM reservation) can use each half on its own.

        ``min_configured_tpm_limit`` is the smallest ``tokens_per_unit`` among
        the TPM-bearing descriptors this request will be charged against. When
        provided, the no-``max_tokens`` output-budget floor is capped at a
        fraction of that limit so small TPM caps remain usable. Omit to
        preserve the unconstrained floor.

        ``call_type`` disambiguates embeddings from the Responses API: both
        put their prompt in ``data["input"]``, but only embeddings have no
        output tokens. Unset (the default) preserves the historical
        "any `input` means zero output" behavior for callers that don't have
        a call type to pass.

        ``configured_output_tokens`` is the operator-declared estimate resolved
        from key or team metadata. When provided it replaces the heuristic
        floor entirely, so the reservation reflects what this tenant's model
        actually emits rather than one constant shared by every tenant.
        """
        request_data: Final = as_str_mapping(data)
        if not isinstance(data, dict) or request_data is None:
            return 0, 0
        request_model: Final = request_data.get("model")
        translated_data: Final = self._translate_google_genai_native_request(data, call_type)
        estimable_data: Final = translated_data if translated_data is not None else request_data
        selected_fields: Final[tuple[object | None, object | None, object | None]] = (
            (None, None, estimable_data.get("input"))
            if call_type in RESPONSES_API_CALL_TYPES or call_type in EMBEDDING_API_CALL_TYPES
            else (None, estimable_data.get("prompt"), None)
            if call_type in TEXT_COMPLETION_API_CALL_TYPES
            else (None, self._extraction_input_text(_REQUEST_RATE_LIMIT_DATA.validate_python(estimable_data)), None)
            if call_type in EXTRACTION_API_CALL_TYPES
            else (estimable_data.get("messages"), None, None)
            if call_type
            else (
                estimable_data.get("messages"),
                estimable_data.get("prompt"),
                estimable_data.get("input"),
            )
        )
        messages, prompt, raw_input = selected_fields
        original_input: Final = (
            billable_input_since_newest_compaction(raw_input) if call_type in RESPONSES_API_CALL_TYPES else raw_input
        )

        media_source: Final = messages if messages is not None else original_input
        is_embedding_call: Final = call_type in EMBEDDING_API_CALL_TYPES
        projected, media = (
            split_media_for_tpm(media_source)
            if include_media and (isinstance(messages, list) or call_type in RESPONSES_API_CALL_TYPES)
            else (original_input, ())
        )
        fallback_model: Final = request_model if isinstance(request_model, str) else None
        resolved_models: Final = (
            _media_models(media_model_resolver, fallback_model or "", team_id) if media else (fallback_model or "",)
        )
        embedding_text, embedding_media_tokens = (
            split_embedding_input_tokens(
                original_input,
                lambda: _media_models(media_model_resolver, fallback_model or "", team_id),
                fallback_model,
            )
            if include_media and is_embedding_call
            else (original_input, 0)
        )
        input_text: Final = projected if call_type in RESPONSES_API_CALL_TYPES else embedding_text
        media_tokens: Final = embedding_media_tokens + sum(
            estimate_media_block_tokens(block, resolved_models, fallback_model) for block in media
        )

        total_chars: Final = (
            len(get_str_from_messages(messages))
            if isinstance(messages, list) and messages
            else len(prompt)
            if isinstance(prompt, str)
            else sum(len(str(item)) for item in prompt)
            if isinstance(prompt, list)
            else len(input_text)
            if isinstance(input_text, str)
            else sum(len(str(item)) for item in input_text)
            if isinstance(input_text, list)
            else 0
        )

        estimated_input_tokens: Final = (
            max(1, total_chars // DEFAULT_CHARS_PER_TOKEN) if total_chars > 0 else 0
        ) + media_tokens

        explicit_max_tokens: Final = self._get_explicit_output_cap(data, call_type)
        generates_no_output: Final = (
            self._is_embedding_request(data, call_type) or call_type in EXTRACTION_API_CALL_TYPES
        )

        base_output_floor: Final = self.no_max_tokens_output_floor(min_configured_tpm_limit)
        output_floor: Final = (
            max(base_output_floor, RESPONSES_API_MIN_OUTPUT_TOKENS)
            if call_type in RESPONSES_API_CALL_TYPES
            else base_output_floor
        )
        max_tokens_estimate: Final = (
            0
            if generates_no_output
            or (explicit_max_tokens is None and estimated_input_tokens == 0 and configured_output_tokens is None)
            else explicit_max_tokens
            if explicit_max_tokens is not None
            else configured_output_tokens
            if configured_output_tokens is not None
            else max(estimated_input_tokens, output_floor)
        )

        return estimated_input_tokens, max_tokens_estimate * self.get_output_candidate_count(data, call_type)

    @with_service_target(RATE_LIMITS_TARGET)
    async def in_memory_cache_sliding_window(
        self,
        counters: Sequence[RateLimitCounter],
        now_int: int,
        window_size: int,
    ) -> list[CounterState]:
        """
        Implement sliding window rate limiting logic using in-memory cache operations.
        This follows the same logic as the Redis Lua script but uses async cache operations.
        """
        async with self._check_and_increment_lock:
            return self._in_memory_cache_sliding_window(counters=counters, now_int=now_int, window_size=window_size)

    def _in_memory_cache_sliding_window(
        self,
        counters: Sequence[RateLimitCounter],
        now_int: int,
        window_size: int,
    ) -> list[CounterState]:
        results: Final[list[CounterState]] = []  # mutable-ok: one state appended per counter, in order
        for counter in counters:
            window_start = _scalar(self.rate_limits.local((counter.window,))[0])
            if window_start is None or (now_int - int(window_start)) >= window_size:
                for sibling in ("requests", "tokens"):
                    self.rate_limits.set_local(counter.scope.counter(sibling), 0, window_size)
                self.rate_limits.set_local(counter.window, str(now_int), window_size)
                self.rate_limits.set_local(counter, 1, window_size)
                results.append((str(now_int), 1))
                continue
            current_counter = _scalar(self.rate_limits.local((counter,))[0])
            new_counter_value = (int(current_counter) if current_counter is not None else 0) + 1
            self.rate_limits.set_local(counter, new_counter_value, window_size)
            results.append((window_start, new_counter_value))
        return results

    def rate_limit_counter(
        self, key: str, value: str, rate_limit_type: Literal["requests", "tokens"]
    ) -> RateLimitCounter:
        return RateLimitScope(key, value).counter(rate_limit_type)

    def is_cache_list_over_limit(
        self,
        counters: Sequence[RateLimitCounter],
        cache_values: Sequence[CounterState],
        key_metadata: Mapping[RateLimitScope, WindowKeyMetadata],
        read_only: bool = False,
    ) -> RateLimitResponse:
        """
        Check if the cache values are over the limit.
        """
        pending_increment: Final = 1 if read_only else 0
        statuses: Final[list[RateLimitStatus]] = []
        overall_code = "OK"

        for counter, (_window_start, counter_value) in zip(counters, cache_values):
            item_code = "OK"
            metadata = key_metadata[counter.scope]
            current_limit = metadata["requests_limit"] if counter.kind == "requests" else metadata["tokens_limit"]
            if current_limit is None:
                continue

            if counter_value is not None and int(counter_value) + pending_increment > current_limit:
                overall_code = "OVER_LIMIT"
                item_code = "OVER_LIMIT"

            limit_remaining = current_limit - int(counter_value) if counter_value is not None else current_limit

            statuses.append(
                {
                    "code": item_code,
                    "current_limit": current_limit,
                    "limit_remaining": limit_remaining,
                    "rate_limit_type": counter.kind,
                    "descriptor_key": metadata["descriptor_key"],
                    "descriptor_value": metadata["descriptor_value"],
                }
            )

        return RateLimitResponse(overall_code=overall_code, statuses=statuses)

    def _slot_groups(self, counters: Sequence[RateLimitCounter]) -> tuple[tuple[int, ...], ...]:
        """
        Indices of ``counters`` grouped by Redis Cluster slot so each Lua call's keys co-locate. Without a
        cluster every counter goes in one group unless ``force_redis_hash_tag_grouping`` asks otherwise.
        """
        return self.rate_limits.slot_groups(
            tuple(counter.scope for counter in counters), self._force_hash_tag_grouping_resolver()
        )

    def _local_counter_states(self, counters: Sequence[RateLimitCounter]) -> list[CounterState]:
        keys: Final = tuple(itertools.chain.from_iterable((counter.window, counter) for counter in counters))
        values: Final = tuple(map(_scalar, self.rate_limits.local(keys)))
        return list(zip(values[::2], values[1::2]))

    async def _read_counter_values_from_redis(self, counters: Sequence[RateLimitCounter]) -> list[CounterState]:
        groups: Final = self._slot_groups(counters)
        group_values: Final = [
            await self.rate_limits.read_counters(tuple(counters[index] for index in group), batch=Batch.NONE)
            for group in groups
        ]
        by_index: Final = dict(
            itertools.chain.from_iterable(
                ((index, (values[offset * 2], values[offset * 2 + 1])) for offset, index in enumerate(group))
                for group, values in zip(groups, group_values)
            )
        )
        return [by_index[index] for index in range(len(counters))]

    async def _read_counter_values_without_incrementing(
        self, counters: Sequence[RateLimitCounter]
    ) -> list[CounterState]:
        if not self.rate_limits.has_redis:
            return self._local_counter_states(counters)
        try:
            return await self._read_counter_values_from_redis(counters)
        except Exception as e:  # noqa: BLE001  # any Redis/Lua failure degrades to the local mirror unless fail-closed rejects
            self._reject_if_rate_limit_unverifiable("batch_counter_read_script", e)
            log_redis_failure(
                verbose_proxy_logger, logging.WARNING, "batch_counter_read_script failed, using local mirror", e
            )
            return self._local_counter_states(counters)

    def _reject_if_rate_limit_unverifiable(self, failed_operation: str, error: BaseException) -> None:
        if not self._fail_closed_resolver():
            return
        log_redis_failure(
            verbose_proxy_logger,
            logging.WARNING,
            f"fail_closed_rate_limit_enforcement: rejecting request, {failed_operation} could not verify the "
            "counters against Redis",
            error,
        )
        raise RateLimitUnverifiableError()

    async def _execute_redis_batch_rate_limiter_script(
        self,
        counters: Sequence[RateLimitCounter],
        now_int: int,
    ) -> list[CounterState]:
        """
        Run ``BATCH_RATE_LIMITER_SCRIPT`` once per cluster slot group, in caller order.
        """
        slot_groups: Final = self._slot_groups(counters)
        groups: Final = tuple(tuple(counters[index] for index in group) for group in slot_groups)
        states_by_group: Final[list[Sequence[CounterState]]] = []  # mutable-ok: one entry appended per group
        batched: Final = self.rate_limits.has_request_batch
        pipelined: Final = tuple(
            self.rate_limits.sliding_window(group, now_int, self.window_size, batch=Batch.PRE_CALL) if batched else None
            for group in groups
        )

        for index, (group, group_result) in enumerate(zip(groups, pipelined)):
            try:
                counts = (
                    await self.rate_limits.sliding_window(group, now_int, self.window_size, batch=Batch.NONE)
                    if group_result is None
                    else await group_result
                )
                states_by_group.append(tuple((count.window_start, count.count) for count in counts))
            except Exception as e:
                if self._fail_closed_resolver():
                    applied = tuple(itertools.chain.from_iterable(groups[:index]))
                    await self._refund_counter_increments(
                        self._counter_refunds_from_batch_values(
                            applied, tuple(itertools.chain.from_iterable(states_by_group))
                        )
                    )
                    await self._refund_later_pipelined_groups(groups[index + 1 :], pipelined[index + 1 :])
                self._reject_if_rate_limit_unverifiable("batch_rate_limiter_script", e)
                log_redis_failure(
                    verbose_proxy_logger, logging.WARNING, f"Redis Lua script failed for slot group {index}", e
                )
                states_by_group.append(
                    await self.in_memory_cache_sliding_window(
                        counters=group,
                        now_int=now_int,
                        window_size=self.window_size,
                    )
                )

        by_counter_index: Final = dict(
            zip(
                itertools.chain.from_iterable(slot_groups),
                itertools.chain.from_iterable(states_by_group),
            )
        )
        return [by_counter_index[index] for index in range(len(counters))]

    async def _refund_later_pipelined_groups(
        self,
        groups: Sequence[Sequence[RateLimitCounter]],
        pipelined: Sequence[Awaitable[Sequence[WindowCount]] | None],
    ) -> None:
        """Groups declared on the request batch ran in the same round trip as the one that failed, so their
        increments landed even though the loop never read them."""
        for group, group_result in zip(groups, pipelined):
            if group_result is None:
                continue
            try:
                counts = await group_result
            except Exception:  # noqa: BLE001  # a group that failed in Redis incremented nothing to refund
                continue
            await self._refund_counter_increments(
                self._counter_refunds_from_batch_values(group, tuple((c.window_start, c.count) for c in counts))
            )

    @with_service_target(RATE_LIMITS_TARGET)
    async def should_rate_limit(
        self,
        descriptors: Sequence[RateLimitDescriptor],
        parent_otel_span: Span | None = None,
        read_only: bool = False,
        skip_tpm_check: bool = False,
        parallel_slot_id: str | None = None,
        skip_parallel_check: bool = False,
    ) -> RateLimitResponse:
        """
        Check if any of the rate limit descriptors should be rate limited.
        Returns a RateLimitResponse with the overall code and status for each descriptor.
        Uses batch operations for Redis to improve performance.

        Args:
            descriptors: List of rate limit descriptors to check
            parent_otel_span: Optional OpenTelemetry span for tracing
            read_only: If True, only check limits without incrementing counters
            skip_tpm_check: If True, ignore each descriptor's ``tokens_per_unit``
                — the :tokens counter is neither read nor incremented by this
                pass. Callers that handle TPM via the atomic
                ``reserve_tpm_tokens`` reservation path should set this to
                avoid the +1-per-key Lua / in-memory increment double-charging
                the tokens counter.
            skip_parallel_check: If True, ``max_parallel_requests`` gauges are
                neither checked nor acquired, for a call that runs inside a
                request already holding its slot.

        ``max_parallel_requests`` descriptors are enforced by the dedicated
        concurrency-gauge path (``_check_parallel_request_gauges``), never by
        the windowed counters. The gauge phase must stay AFTER the windowed
        check so a windowed rejection never strands an acquired slot; the
        reverse order would leak one gauge slot per RPM/TPM rejection.
        ``parallel_slot_id`` names the slot an admission registers; callers
        that enforce (not read_only) should pass the id they will later
        release with — when omitted, a generated slot id is used and the slot
        can only be reclaimed by TTL expiry.
        """

        current_time: Final = self._get_current_time()
        now: Final = current_time.timestamp()
        now_int: Final = int(now)  # Convert to integer for Redis Lua script

        counters, key_metadata, gauges = self._collect_windowed_keys_and_gauges(
            descriptors=descriptors,
            skip_tpm_check=skip_tpm_check,
        )

        windowed_response = RateLimitResponse(overall_code="OK", statuses=[])
        if counters:
            local_response: Final = self.is_cache_list_over_limit(
                counters, self._local_counter_states(counters), key_metadata, read_only=read_only
            )
            if local_response["overall_code"] == "OVER_LIMIT":
                return local_response

            if read_only:
                cache_values = await self._read_counter_values_without_incrementing(counters)
            elif self.rate_limits.has_redis:
                cache_values = await self._execute_redis_batch_rate_limiter_script(counters=counters, now_int=now_int)
                for counter, (window_value, counter_value) in zip(counters, cache_values):
                    if counter_value is not None:
                        self.rate_limits.set_local(counter, counter_value, self.window_size)
                    if window_value is not None:
                        self.rate_limits.set_local(counter.window, window_value, self.window_size)
            else:
                cache_values = await self.in_memory_cache_sliding_window(
                    counters=counters,
                    now_int=now_int,
                    window_size=self.window_size,
                )

            windowed_response = self.is_cache_list_over_limit(counters, cache_values, key_metadata, read_only=read_only)
            if windowed_response["overall_code"] == "OVER_LIMIT":
                return windowed_response

        if not gauges or skip_parallel_check:
            return windowed_response

        gauge_response: Final = await self._check_parallel_request_gauges(
            gauges=gauges,
            slot_id=parallel_slot_id or uuid.uuid4().hex,
            parent_otel_span=parent_otel_span,
            read_only=read_only,
        )
        return RateLimitResponse(
            overall_code=gauge_response["overall_code"],
            statuses=[*windowed_response["statuses"], *gauge_response["statuses"]],
        )

    def _collect_windowed_keys_and_gauges(
        self,
        descriptors: Sequence[RateLimitDescriptor],
        skip_tpm_check: bool,
    ) -> tuple[list[RateLimitCounter], dict[RateLimitScope, WindowKeyMetadata], list[ParallelRequestGauge]]:
        """
        Split descriptors into the windowed counters with their per-window
        metadata, and the concurrency gauges for descriptors carrying a
        max_parallel_requests limit.
        """
        counters: Final[list[RateLimitCounter]] = []
        key_metadata: Final[dict[RateLimitScope, WindowKeyMetadata]] = {}
        gauges: Final[list[ParallelRequestGauge]] = []
        for descriptor in descriptors:
            descriptor_key = descriptor["key"]
            descriptor_value = descriptor["value"]
            scope = RateLimitScope(descriptor_key, descriptor_value)
            rate_limit: RateLimitDescriptorRateLimitObject = (
                descriptor.get("rate_limit") or RateLimitDescriptorRateLimitObject()
            )
            requests_limit = rate_limit.get("requests_per_unit")
            tokens_limit = None if skip_tpm_check else rate_limit.get("tokens_per_unit")
            max_parallel_requests_limit = rate_limit.get("max_parallel_requests")
            window_size = rate_limit.get("window_size") or self.window_size

            if max_parallel_requests_limit is not None:
                gauges.append(
                    ParallelRequestGauge(
                        scope=scope,
                        limit=int(max_parallel_requests_limit),
                        descriptor_key=descriptor_key,
                    )
                )

            if requests_limit is not None:
                counters.append(scope.counter("requests"))
            if tokens_limit is not None:
                counters.append(scope.counter("tokens"))
            if requests_limit is None and tokens_limit is None:
                continue

            key_metadata[scope] = {
                "requests_limit": (int(requests_limit) if requests_limit is not None else None),
                "tokens_limit": int(tokens_limit) if tokens_limit is not None else None,
                "window_size": int(window_size),
                "descriptor_key": descriptor_key,
                "descriptor_value": descriptor_value,
            }
        return counters, key_metadata, gauges

    def _gauge_status(self, gauge: ParallelRequestGauge, in_flight: int, code: str) -> RateLimitStatus:
        return RateLimitStatus(
            code=code,
            current_limit=gauge["limit"],
            limit_remaining=max(0, gauge["limit"] - in_flight),
            rate_limit_type="max_parallel_requests",
            descriptor_key=gauge["descriptor_key"],
        )

    def _gauge_in_flight_from_cache_value(self, raw_value: LocalValue | None) -> int:
        """
        In-flight count from a cached gauge value: a dict of slot_id ->
        acquire timestamp when the in-memory registry is authoritative, or
        the mirrored integer count from the last Redis script result.
        """
        if raw_value is None:
            return 0
        if isinstance(raw_value, dict):
            cutoff: Final = self._get_current_time().timestamp() - PARALLEL_REQUEST_SLOT_TTL_SECONDS
            return sum(1 for ts in raw_value.values() if ts >= cutoff)
        return max(0, int(raw_value))

    async def _check_parallel_request_gauges(
        self,
        gauges: list[ParallelRequestGauge],
        slot_id: str,
        parent_otel_span: Span | None = None,
        read_only: bool = False,
    ) -> RateLimitResponse:
        """
        Enforce max_parallel_requests as a concurrency gauge over a per-slot
        registry: each admitted request registers ``slot_id`` with its
        acquire time, and admission requires in_flight + 1 <= limit over the
        unexpired slots. Unlike the windowed RPM/TPM counters, the gauge is
        never reset while requests are in flight, a rejected request never
        occupies a slot, and a slot leaked by a crashed worker is pruned
        after PARALLEL_REQUEST_SLOT_TTL_SECONDS even under continuous
        traffic. Releases remove exactly this request's slot id, so a
        double-fired or unmatched release can never free another request's
        slot.
        """
        gauge_scopes: Final = tuple(gauge["scope"] for gauge in gauges)

        if read_only:
            if self.rate_limits.has_redis:
                try:
                    counts = [
                        max(0, value)
                        for value in await self.rate_limits.parallel_count(
                            gauge_scopes, PARALLEL_REQUEST_SLOT_TTL_SECONDS, batch=Batch.NONE
                        )
                    ]
                except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to the local mirror unless fail-closed rejects
                    self._reject_if_rate_limit_unverifiable("parallel_count_script", e)
                    log_redis_failure(
                        verbose_proxy_logger, logging.WARNING, "parallel_count_script failed, using local mirror", e
                    )
                    counts = self._read_local_gauge_counts(gauge_scopes)
            else:
                counts = self._read_local_gauge_counts(gauge_scopes)
            statuses = []
            overall_code = "OK"
            for gauge, in_flight in zip(gauges, counts):
                code = "OVER_LIMIT" if in_flight >= gauge["limit"] else "OK"
                if code == "OVER_LIMIT":
                    overall_code = "OVER_LIMIT"
                statuses.append(self._gauge_status(gauge, in_flight, code))
            return RateLimitResponse(overall_code=overall_code, statuses=statuses)

        local_counts: Final = self._read_local_gauge_counts(gauge_scopes)
        for gauge, in_flight in zip(gauges, local_counts):
            if in_flight >= gauge["limit"]:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[self._gauge_status(gauge, in_flight, "OVER_LIMIT")],
                )

        if self.rate_limits.has_redis:
            try:
                reply: Final = await self.rate_limits.parallel_acquire(
                    tuple((gauge["scope"], gauge["limit"]) for gauge in gauges),
                    PARALLEL_REQUEST_SLOT_TTL_SECONDS,
                    slot_id,
                    batch=Batch.NONE,
                )
            except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to in-memory enforcement, never a 500
                self._reject_if_rate_limit_unverifiable("parallel_acquire_script", e)
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "parallel_acquire_script failed, falling back to in-memory gauge",
                    e,
                )
                async with self._check_and_increment_lock:
                    return self._acquire_parallel_slots_in_memory(gauges, slot_id)
            if isinstance(reply, GaugeOverLimit):
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[self._gauge_status(gauges[reply.index], reply.in_flight, "OVER_LIMIT")],
                )
            statuses = []
            for gauge, in_flight in zip(gauges, reply.in_flight):
                self.rate_limits.set_local(gauge["scope"].gauge, in_flight, PARALLEL_REQUEST_SLOT_TTL_SECONDS)
                statuses.append(self._gauge_status(gauge, in_flight, "OK"))
            return RateLimitResponse(overall_code="OK", statuses=statuses)

        async with self._check_and_increment_lock:
            return self._acquire_parallel_slots_in_memory(gauges, slot_id)

    def _read_local_gauge_counts(self, gauges: Sequence[RateLimitScope]) -> list[int]:
        return [
            self._gauge_in_flight_from_cache_value(value)
            for value in self.rate_limits.local(tuple(scope.gauge for scope in gauges))
        ]

    def _acquire_parallel_slots_in_memory(
        self,
        gauges: list[ParallelRequestGauge],
        slot_id: str,
    ) -> RateLimitResponse:
        """
        All-or-nothing in-memory slot-registry acquire. Caller holds the lock.

        A cached dict is the authoritative in-memory registry. A cached
        integer is the count mirrored from the last successful Redis script
        call: when Redis fails over to this path, that mirror still counts
        the slots in flight on the Redis side, so it is carried forward as
        an integer counter (not discarded as an empty registry, which would
        briefly double the admitted concurrency during a Redis outage).
        """
        now: Final = self._get_current_time().timestamp()
        cutoff: Final = now - PARALLEL_REQUEST_SLOT_TTL_SECONDS
        states: Final[list[tuple[dict[str, float] | None, int]]] = []
        for gauge, raw_value in zip(gauges, self.rate_limits.local(tuple(g["scope"].gauge for g in gauges))):
            if isinstance(raw_value, dict):
                registry: dict[str, float] | None = {key: ts for key, ts in raw_value.items() if ts >= cutoff}
                in_flight = len(registry or {})
            elif raw_value is None:
                registry = {}
                in_flight = 0
            else:
                registry = None
                in_flight = max(0, int(raw_value))
            if in_flight + 1 > gauge["limit"]:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[self._gauge_status(gauge, in_flight, "OVER_LIMIT")],
                )
            states.append((registry, in_flight))

        statuses: Final = []
        for gauge, (registry, in_flight) in zip(gauges, states):
            new_value: dict[str, float] | int = {**registry, slot_id: now} if registry is not None else in_flight + 1
            self.rate_limits.set_local(gauge["scope"].gauge, new_value, PARALLEL_REQUEST_SLOT_TTL_SECONDS)
            statuses.append(self._gauge_status(gauge, in_flight + 1, "OK"))
        return RateLimitResponse(overall_code="OK", statuses=statuses)

    async def _release_stashed_parallel_slot(
        self,
        stash: RequestRateLimiterStash | None,
        parent_otel_span: Span | None,
        *,
        in_logging_callback: bool = False,
        round_ended_at: datetime | None = None,
    ) -> None:
        if stash is None:
            return
        async with stash.parallel_slot_release_lock:
            acquisition: Final = stash.parallel_slot
            if acquisition is None or _ended_before_slot_reuse(round_ended_at, stash.parallel_slot_reused_at):
                return
            deferred: Final = in_logging_callback and await self._defer_parallel_slot_release(
                acquisition, parent_otel_span
            )
            if not deferred:
                await self._release_parallel_request_slots(acquisition, parent_otel_span)
            stash.parallel_slot = None  # rebind-ok: marks this request's slot as released

    async def _release_parallel_request_slots(
        self,
        acquisition: ParallelSlotAcquisition,
        parent_otel_span: Span | None = None,
    ) -> None:
        """
        Release the max_parallel_requests slots acquired at pre-call by
        removing this request's slot id from every gauge it was registered
        under. Removing an absent slot id is a no-op, so a release without a
        matching acquire or a double-fired release can never free another
        request's slot. The in-memory fallback decrements integer mirror
        values (floored at 0) because the mirror carries no per-slot ids.
        """
        gauges: Final = acquisition["gauges"]
        slot_id: Final = acquisition["slot_id"]
        if not gauges or not slot_id:
            return
        if self.rate_limits.has_redis:
            try:
                remaining: Final = await self.rate_limits.parallel_release(gauges, slot_id, batch=Batch.NONE)
                self._mirror_released_parallel_slots(gauges, remaining)
                return
            except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to the in-memory release, never a 500
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "parallel_release_script failed, falling back to in-memory release",
                    e,
                )
        await self._release_parallel_request_slots_in_memory(gauges, slot_id)

    async def _defer_parallel_slot_release(
        self, acquisition: ParallelSlotAcquisition, parent_otel_span: Span | None
    ) -> bool:
        """Only for a release from the logging callbacks: the response has left and the callbacks' end flushes
        the pipeline. A release before the response goes to Redis at once, so another worker's next acquire
        never counts a finished request. The local gauge frees the slot at once, so admission on this worker
        sees the capacity before the pipeline goes out. The count Redis returns from the pipeline is not
        mirrored: by then a newer acquire on this worker may have written a fresher count, and the next
        acquire refreshes the gauge anyway."""
        gauges: Final = acquisition["gauges"]
        slot_id: Final = acquisition["slot_id"]
        if not self.rate_limits.has_request_batch or not gauges or not slot_id:
            return False
        await self._release_parallel_request_slots_in_memory(gauges, slot_id)
        return self.rate_limits.parallel_release_post_call(gauges, slot_id)

    def _mirror_released_parallel_slots(
        self, gauges: Sequence[RateLimitScope], remaining_by_gauge: Sequence[int]
    ) -> None:
        for scope, remaining in zip(gauges, remaining_by_gauge):
            self.rate_limits.set_local(scope.gauge, max(0, remaining), PARALLEL_REQUEST_SLOT_TTL_SECONDS)

    async def _release_parallel_request_slots_in_memory(self, gauges: Sequence[RateLimitScope], slot_id: str) -> None:
        async with self._check_and_increment_lock:
            for scope in gauges:
                raw_value = self.rate_limits.local((scope.gauge,))[0]
                if isinstance(raw_value, dict):
                    if slot_id not in raw_value:
                        continue
                    new_value: dict[str, float] | int = {key: ts for key, ts in raw_value.items() if key != slot_id}
                elif raw_value is None:
                    continue
                else:
                    new_value = max(0, int(raw_value) - 1)
                self.rate_limits.set_local(scope.gauge, new_value, PARALLEL_REQUEST_SLOT_TTL_SECONDS)

    @with_service_target(RATE_LIMITS_TARGET)
    async def atomic_check_and_increment_by_n(
        self,
        descriptors: list[RateLimitDescriptor],
        increments: list[dict[Literal["requests", "tokens"], int]],
        parent_otel_span: Span | None = None,
    ) -> RateLimitResponse:
        """
        Atomic check-and-increment-by-N across one or more descriptors.

        All-or-nothing: if any descriptor would exceed its limit, no counter is
        modified and the response carries `overall_code = "OVER_LIMIT"` with
        the offending descriptor's status. Closes the TOCTOU window between
        read and increment in both single-process and multi-process (Redis)
        deployments.

        Cluster-safety: each descriptor's keys all share a `{key:value}` hash
        tag, so the Redis Lua path issues one Lua call per descriptor — every
        call's keys co-locate on a single Redis Cluster slot, avoiding
        CROSSSLOT errors. Cross-descriptor atomicity is preserved via
        refund-on-rollback: if descriptor i is OVER_LIMIT, descriptors 0..i-1
        get a direct INCRBY refund (refunds need no atomicity guarantee).

        Args:
            descriptors: rate-limit descriptors to check
            increments: per-descriptor increment amounts, indexed parallel to
                `descriptors`. Each entry is `{"requests": int, "tokens": int}`
                — values default to 0 when a descriptor has no matching limit.

        Returns:
            RateLimitResponse with one status per (descriptor, rate_limit_type)
            counter, mirroring `should_rate_limit`'s shape.
        """
        if len(descriptors) != len(increments):
            raise ValueError("atomic_check_and_increment_by_n: descriptors and increments must have the same length")

        # Build per-descriptor (keys, args, meta) groups. All keys within a
        # group share the descriptor's {key:value} hash tag, so a single Lua
        # call per group never triggers CROSSSLOT on Redis Cluster.
        descriptor_groups: Final[list[DescriptorAtomicGroup]] = []
        for descriptor, increment_amounts in zip(descriptors, increments):
            counters, meta = self._build_descriptor_atomic_payload(
                descriptor=descriptor,
                increment_amounts=increment_amounts,
            )
            if counters:
                descriptor_groups.append((counters, meta))

        if not descriptor_groups:
            return RateLimitResponse(overall_code="OK", statuses=[])

        # Multi-process atomicity via Redis Lua, per descriptor for slot
        # co-location. Single-process atomicity falls back to the
        # asyncio.Lock + in-memory sliding window below — there are no
        # cluster slot concerns locally, so we keep the batched 2-phase
        # critical section for true cross-descriptor atomicity.
        if self.rate_limits.has_redis:
            return await self._atomic_lua_per_descriptor(
                descriptor_groups=descriptor_groups,
                parent_otel_span=parent_otel_span,
            )

        flat_meta: Final[list[AtomicCounterMeta]] = [
            m for _counters, group_meta in descriptor_groups for m in group_meta
        ]
        async with self._check_and_increment_lock:
            return self._atomic_check_and_increment_in_memory(per_counter_meta=flat_meta)

    def _build_descriptor_atomic_payload(
        self,
        descriptor: RateLimitDescriptor,
        increment_amounts: dict[Literal["requests", "tokens"], int],
    ) -> DescriptorAtomicGroup:
        """
        Build the counters and per-counter meta for a single descriptor's Lua
        call. All its keys share the descriptor's {key:value} hash tag.
        """
        descriptor_key: Final = descriptor["key"]
        descriptor_value: Final = descriptor["value"]
        scope: Final = RateLimitScope(descriptor_key, descriptor_value)
        rate_limit: Final[RateLimitDescriptorRateLimitObject] = (
            descriptor.get("rate_limit") or RateLimitDescriptorRateLimitObject()
        )
        window_size: Final = rate_limit.get("window_size") or self.window_size

        counters: Final[list[AtomicCounter]] = []
        meta: Final[list[AtomicCounterMeta]] = []

        rate_limit_types: Final[tuple[Literal["requests", "tokens"], ...]] = ("requests", "tokens")
        for rlt in rate_limit_types:
            if rlt == "requests":
                limit_value = rate_limit.get("requests_per_unit")
                inc_amount = int(increment_amounts.get("requests", 0) or 0)
            else:
                limit_value = rate_limit.get("tokens_per_unit")
                inc_amount = int(increment_amounts.get("tokens", 0) or 0)
            if limit_value is None or inc_amount < 0:
                continue
            counter = scope.counter(rlt)
            # Counter-key TTL and window_size are conceptually distinct
            # ("how long the counter Redis key lives" vs "how long the
            # sliding window is"). Kept as separate values so a future
            # custom-TTL descriptor doesn't reintroduce a silent expiry bug.
            ttl_seconds = int(window_size)
            window_size_seconds = int(window_size)
            counters.append(
                AtomicCounter(
                    counter=counter,
                    limit=int(limit_value),
                    increment=inc_amount,
                    ttl=ttl_seconds,
                    window_size=window_size_seconds,
                )
            )
            meta.append(
                {
                    "descriptor_key": descriptor_key,
                    "descriptor_value": descriptor_value,
                    "current_limit": int(limit_value),
                    "rate_limit_type": rlt,
                    "counter": counter,
                    "increment": inc_amount,
                    "ttl": ttl_seconds,
                    "window_size": window_size_seconds,
                }
            )
        return counters, meta

    async def _atomic_lua_per_descriptor(
        self,
        descriptor_groups: list[DescriptorAtomicGroup],
        parent_otel_span: Span | None = None,
    ) -> RateLimitResponse:
        """
        Run Lua check-and-increment one descriptor at a time so each call's
        keys co-locate on a single Redis Cluster slot. On OVER_LIMIT for
        descriptor i, refund descriptors 0..i-1's increments. On Lua failure
        mid-loop, refund applied increments and fall back to in-memory.
        """
        if not descriptor_groups:
            return RateLimitResponse(
                overall_code="OK",
                statuses=[],
            )
        applied: Final[list[tuple[CounterRefund, ...]]] = []
        statuses: Final[list[RateLimitStatus]] = []
        reservation_windows: Final[set[ReservationWindowIdentity]] = set()  # mutable-ok: filled by the group loop
        reply: AtomicReply

        if self.rate_limits.has_request_batch:
            batched: Final = tuple(
                self.rate_limits.check_and_increment(counters, batch=Batch.PRE_CALL)
                for counters, _meta in descriptor_groups
            )
            return await self._settle_pipelined_descriptor_groups(descriptor_groups, batched, parent_otel_span)

        for counters, meta in descriptor_groups:
            try:
                reply = await self.rate_limits.check_and_increment(counters, batch=Batch.NONE)
            except Exception as e:
                # Lua failure (timeout, OOM, network partition) leaves Redis
                # state ambiguous. Refund any prior groups so Redis returns
                # to its pre-call state, then fall back to in-memory for the
                # whole call (counters there are independent of Redis).
                await self._refund_applied_descriptor_groups(applied)
                self._reject_if_rate_limit_unverifiable("check_and_increment_by_n_script", e)
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.ERROR,
                    f"atomic_check_and_increment_by_n: Redis Lua execution failed ({type(e).__name__}). Refunded "
                    f"{len(applied)} prior descriptors and falling back to in-memory enforcement, counters will "
                    f"diverge from Redis until window expires (window_size={self.window_size}s)",
                    e,
                )
                flat_meta: list[AtomicCounterMeta] = [m for _c, group_meta in descriptor_groups for m in group_meta]
                async with self._check_and_increment_lock:
                    return self._atomic_check_and_increment_in_memory(per_counter_meta=flat_meta)

            response = self._build_atomic_response(reply, meta)
            if response["overall_code"] == "OVER_LIMIT":
                await self._refund_applied_descriptor_groups(applied)
                return response
            if len(descriptor_groups) == 1:
                return response
            if isinstance(reply, AtomicApplied):
                applied.append(self._counter_refunds_from_atomic_response(reply, meta))
            statuses.extend(response["statuses"])
            reservation_windows.update(response.get("reservation_windows", frozenset()))

        return RateLimitResponse(
            overall_code="OK",
            statuses=statuses,
            reservation_windows=frozenset(reservation_windows),
        )

    async def _settle_pipelined_descriptor_groups(
        self,
        descriptor_groups: list[DescriptorAtomicGroup],
        results: Sequence[Awaitable[AtomicReply]],
        parent_otel_span: Span | None,
    ) -> RateLimitResponse:
        """Every group's Lua call left in one pipeline, so each group has already checked and incremented on
        its own before any result is read. A failed or over-limit group therefore refunds every group that
        incremented, after it as well as before it, where the one-at-a-time loop only unwinds the groups it ran.
        A Redis denial stands even when another group failed: the in-memory fallback only replaces a verdict
        Redis never gave."""
        replies: Final = await asyncio.gather(*results, return_exceptions=True)
        responses: Final = tuple(
            self._pipelined_group_response(reply, meta) for reply, (_counters, meta) in zip(replies, descriptor_groups)
        )
        applied: Final[list[tuple[CounterRefund, ...]]] = []  # mutable-ok: filled by the group loop
        statuses: Final[list[RateLimitStatus]] = []  # mutable-ok: filled by the group loop
        reservation_windows: Final[set[ReservationWindowIdentity]] = set()  # mutable-ok: filled by the group loop
        for reply, response, (_counters, meta) in zip(replies, responses, descriptor_groups):
            if not isinstance(reply, AtomicApplied) or isinstance(response, BaseException):
                continue
            if response["overall_code"] != "OK":
                continue
            applied.append(self._counter_refunds_from_atomic_response(reply, meta))
            statuses.extend(response["statuses"])
            reservation_windows.update(response.get("reservation_windows", frozenset()))

        over_limit: Final = next(
            (r for r in responses if not isinstance(r, BaseException) and r["overall_code"] == "OVER_LIMIT"), None
        )
        if over_limit is not None:
            await self._refund_applied_descriptor_groups(applied)
            return over_limit
        failure: Final = next((r for r in responses if isinstance(r, BaseException)), None)
        if failure is not None:
            await self._refund_applied_descriptor_groups(applied)
            self._reject_if_rate_limit_unverifiable("check_and_increment_by_n_script", failure)
            log_redis_failure(
                verbose_proxy_logger,
                logging.ERROR,
                f"atomic_check_and_increment_by_n: Redis Lua execution failed ({type(failure).__name__}). Refunding "
                f"{len(applied)} pipelined descriptors and falling back to in-memory enforcement, counters will "
                f"diverge from Redis until window expires (window_size={self.window_size}s)",
                failure,
            )
            flat_meta: Final = tuple(itertools.chain.from_iterable(group_meta for _c, group_meta in descriptor_groups))
            async with self._check_and_increment_lock:
                return self._atomic_check_and_increment_in_memory(per_counter_meta=flat_meta)
        if len(responses) == 1 and not isinstance(responses[0], BaseException):
            return responses[0]
        return RateLimitResponse(
            overall_code="OK",
            statuses=statuses,
            reservation_windows=frozenset(reservation_windows),
        )

    def _pipelined_group_response(
        self, reply: AtomicReply | BaseException, per_counter_meta: list[AtomicCounterMeta]
    ) -> RateLimitResponse | BaseException:
        if isinstance(reply, BaseException):
            return reply
        try:
            return self._build_atomic_response(reply, per_counter_meta)
        except Exception as e:  # noqa: BLE001  # a reply this group cannot read is that group's Lua failure
            return e

    async def _refund_applied_descriptor_groups(
        self,
        applied: Sequence[Sequence[CounterRefund]],
    ) -> None:
        """
        Decrement counters for descriptor groups already applied via Lua.
        Best-effort: refund failures are logged but not raised — the original
        OVER_LIMIT / fallback decision is what matters to the caller.
        """
        await self._refund_counter_increments(tuple(itertools.chain.from_iterable(applied)))

    @staticmethod
    def _counter_refunds_from_atomic_response(
        reply: AtomicApplied,
        per_counter_meta: Sequence[AtomicCounterMeta],
    ) -> tuple[CounterRefund, ...]:
        return tuple(
            CounterRefund(
                counter=meta["counter"],
                window_start=str(window_start),
                increment=meta["increment"],
            )
            for (_count, window_start), meta in zip(reply.counters, per_counter_meta)
        )

    @staticmethod
    def _counter_refunds_from_batch_values(
        applied_counters: Sequence[RateLimitCounter],
        applied_values: Sequence[CounterState],
    ) -> tuple[CounterRefund, ...]:
        return tuple(
            CounterRefund(counter=counter, window_start=str(int(window_start)), increment=1)
            for counter, (window_start, _count) in zip(applied_counters, applied_values)
            if window_start is not None
        )

    async def _refund_counter_increments(self, refunds: Sequence[CounterRefund]) -> None:
        if not self.rate_limits.has_redis:
            return
        for refund in refunds:
            try:
                await self.rate_limits.guarded_increment(
                    refund.counter, refund.window_start, -refund.increment, 0, batch=Batch.NONE
                )
            except Exception as e:  # noqa: BLE001  # best-effort rollback, the rejection already decided the request
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    f"Failed to refund {refund.counter} on rollback",
                    e,
                )

    def _build_atomic_response(
        self,
        reply: AtomicReply,
        per_counter_meta: list[AtomicCounterMeta],
    ) -> RateLimitResponse:
        """Convert a ``CHECK_AND_INCREMENT_BY_N_SCRIPT`` reply to a RateLimitResponse.

        Indexing invariant: `per_counter_meta` and the script's counters are
        parallel-indexed at the COUNTER level, not the descriptor level. A
        descriptor with both RPM and TPM limits emits two counters and two
        meta entries, so the over-limit reply's index maps directly to
        `per_counter_meta[index]`. Keep these arrays parallel at the counter
        level when modifying this code.
        """
        if isinstance(reply, AtomicOverLimit):
            meta = per_counter_meta[reply.index]
            return RateLimitResponse(
                overall_code="OVER_LIMIT",
                statuses=[
                    RateLimitStatus(
                        code="OVER_LIMIT",
                        current_limit=reply.limit,
                        limit_remaining=max(0, reply.limit - reply.current),
                        rate_limit_type=meta["rate_limit_type"],
                        descriptor_key=meta["descriptor_key"],
                        descriptor_value=meta["descriptor_value"],
                    )
                ],
            )
        statuses: Final[list[RateLimitStatus]] = [
            RateLimitStatus(
                code="OK",
                current_limit=meta["current_limit"],
                limit_remaining=max(0, meta["current_limit"] - new_counter),
                rate_limit_type=meta["rate_limit_type"],
                descriptor_key=meta["descriptor_key"],
                descriptor_value=meta["descriptor_value"],
            )
            for (new_counter, _window_start), meta in zip(reply.counters, per_counter_meta, strict=True)
        ]
        return RateLimitResponse(
            overall_code="OK",
            statuses=statuses,
            reservation_windows=frozenset(
                (meta["counter"], str(window_start), "redis")
                for (_count, window_start), meta in zip(reply.counters, per_counter_meta, strict=True)
            ),
        )

    def _atomic_check_and_increment_in_memory(
        self,
        per_counter_meta: Sequence[AtomicCounterMeta],
    ) -> RateLimitResponse:
        """In-memory all-or-nothing check-and-increment. Caller holds lock.

        Reads/writes the store's memory tier — note this is a different
        store from Redis. When this fallback fires after a Lua failure,
        in-memory counters will diverge from Redis until each key's window
        expires (TTL bounds divergence).
        """
        # Use a single 'now' for the duration of this critical section so all
        # descriptors evaluate window expiry consistently.
        now_int: Final = int(self._get_current_time().timestamp())

        # Pass 1: read state, validate.
        descriptor_state: Final[list[AtomicCounterState]] = []
        for meta in per_counter_meta:
            window_size = meta["window_size"]
            window_start = _scalar(self.rate_limits.local((meta["counter"].window,))[0])
            window_expired = window_start is None or (now_int - int(window_start)) >= window_size
            raw_counter = None if window_expired else _scalar(self.rate_limits.local((meta["counter"],))[0])
            current_counter = 0 if window_expired else int(raw_counter or 0)
            over_limit = (
                current_counter + meta["increment"] > meta["current_limit"]
                if meta["increment"] > 0
                else current_counter >= meta["current_limit"]
            )
            if over_limit:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[
                        RateLimitStatus(
                            code="OVER_LIMIT",
                            current_limit=meta["current_limit"],
                            limit_remaining=max(0, meta["current_limit"] - current_counter),
                            rate_limit_type=meta["rate_limit_type"],
                            descriptor_key=meta["descriptor_key"],
                            descriptor_value=meta["descriptor_value"],
                        )
                    ],
                )
            descriptor_state.append(
                {
                    "window_expired": window_expired,
                    "current": current_counter,
                    "window_start": str(now_int if window_start is None or window_expired else int(window_start)),
                }
            )

        # Pass 2: apply increments.
        expired_windows: Final[Mapping[RateLimitScope, int]] = {
            meta["counter"].scope: meta["window_size"]
            for meta, state in zip(per_counter_meta, descriptor_state)
            if state["window_expired"]
        }
        for scope, window_size in expired_windows.items():
            for sibling in ("requests", "tokens"):
                self.rate_limits.set_local(scope.counter(sibling), 0, window_size)
        statuses: Final[list[RateLimitStatus]] = []
        for meta, state in zip(per_counter_meta, descriptor_state):
            new_counter = meta["increment"] if state["window_expired"] else state["current"] + meta["increment"]
            if state["window_expired"]:
                self.rate_limits.set_local(meta["counter"].window, str(now_int), meta["window_size"])
            self.rate_limits.set_local(meta["counter"], new_counter, meta["ttl"])
            statuses.append(
                RateLimitStatus(
                    code="OK",
                    current_limit=meta["current_limit"],
                    limit_remaining=max(0, meta["current_limit"] - new_counter),
                    rate_limit_type=meta["rate_limit_type"],
                    descriptor_key=meta["descriptor_key"],
                    descriptor_value=meta["descriptor_value"],
                )
            )
        return RateLimitResponse(
            overall_code="OK",
            statuses=statuses,
            reservation_windows=frozenset(
                (meta["counter"], state["window_start"], "local")
                for meta, state in zip(per_counter_meta, descriptor_state)
            ),
        )

    @with_service_target(RATE_LIMITS_TARGET)
    async def reserve_tpm_tokens(
        self,
        descriptors: list[RateLimitDescriptor],
        estimated_tokens: int,
        parent_otel_span: Span | None = None,
        scope_estimates: Mapping[str, int] = MappingProxyType({}),
    ) -> RateLimitResponse:
        """
        Reserve ``estimated_tokens`` against every TPM-bearing descriptor
        BEFORE the upstream call, so concurrent requests cannot all observe
        "under limit" before any of them increments the counter.
        ``scope_estimates`` replaces that amount per descriptor key for a
        scope counted in other units, such as the team PTU ceiling.

        Thin wrapper around ``atomic_check_and_increment_by_n``: builds a
        TPM-only descriptor/increment list and delegates the all-or-nothing
        atomicity (Lua on Redis, asyncio-locked DualCache otherwise) to the
        shared primitive.

        Excludes project ITPM/OTPM descriptors -- those are reserved
        separately (different estimate per bucket) via ``reserve_io_tokens``.
        """
        tpm_descriptors: Final[list[RateLimitDescriptor]] = [
            RateLimitDescriptor(
                key=d["key"],
                value=d["value"],
                rate_limit=RateLimitDescriptorRateLimitObject(
                    tokens_per_unit=(d.get("rate_limit") or {}).get("tokens_per_unit"),
                    window_size=(d.get("rate_limit") or {}).get("window_size"),
                ),
            )
            for d in descriptors
            if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
            and (d.get("rate_limit") or {}).get("tokens_per_unit") is not None
        ]
        if not tpm_descriptors:
            return RateLimitResponse(overall_code="OK", statuses=[])

        increments: Final[list[dict[Literal["requests", "tokens"], int]]] = [
            {"tokens": scope_estimates.get(d["key"], estimated_tokens)} for d in tpm_descriptors
        ]
        return await self.atomic_check_and_increment_by_n(
            descriptors=tpm_descriptors,
            increments=increments,
            parent_otel_span=parent_otel_span,
        )

    async def _refund_reserved_tokens(
        self,
        scopes: Sequence[tuple[str, str]],
        amount: int,
        reservation_windows: frozenset[ReservationWindowIdentity] = frozenset(),
        parent_otel_span: Span | None = None,
    ) -> None:
        """
        Directly decrement previously-reserved token counters for ``scopes``
        by ``amount``. Used to roll back a reservation that already
        succeeded once a *different* bucket in the same request turns out to
        be over its limit (e.g. ITPM reserved fine, OTPM then hits its
        limit -- the ITPM reservation must not be left inflated).
        """
        if amount <= 0 or not scopes:
            return
        if not reservation_windows:
            await self.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=self._build_reservation_aware_tpm_ops(
                    targets=scopes,
                    reserved_scopes=frozenset(scopes),
                    actual_tokens=0,
                    reserved_tokens=amount,
                ),
                parent_otel_span=parent_otel_span,
            )
            return
        pipeline_operations: Final = self._build_project_reservation_ops(
            targets=scopes,
            reserved_scopes=frozenset(scopes),
            actual_tokens=0,
            reserved_tokens=amount,
            reservation_window_identities=reservation_windows,
        )
        await self.async_increment_reservation_aware_tokens(
            pipeline_operations=pipeline_operations,
            parent_otel_span=parent_otel_span,
        )

    @with_service_target(RATE_LIMITS_TARGET)
    async def reserve_io_tokens(
        self,
        descriptors: Sequence[RateLimitDescriptor],
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        parent_otel_span: Span | None = None,
    ) -> tuple[RateLimitResponse, int, int]:
        """
        Reserve ``estimated_input_tokens`` against project ITPM descriptors
        and ``estimated_output_tokens`` against project OTPM descriptors.

        ITPM and OTPM are reserved from different-sized estimates, so unlike
        same-size TPM descriptors they can't share a single
        ``atomic_check_and_increment_by_n`` call -- each bucket gets its own
        all-or-nothing atomic call. If the OTPM reservation is over limit
        after ITPM already succeeded, the ITPM reservation this call made is
        rolled back before returning, so a partial reservation never leaks.

        Returns ``(response, itpm_reserved, otpm_reserved)`` -- the latter two
        are the amounts actually reserved (0 if that bucket wasn't
        configured, or if the reservation failed), for the caller to stash
        for post-call reconciliation.
        """
        itpm_descriptors: Final = [d for d in descriptors if d["key"] == PROJECT_ITPM_DESCRIPTOR_KEY]
        otpm_descriptors: Final = [d for d in descriptors if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY]

        if not itpm_descriptors and not otpm_descriptors:
            return RateLimitResponse(overall_code="OK", statuses=[]), 0, 0

        itpm_response: Final = (
            await self.atomic_check_and_increment_by_n(
                descriptors=itpm_descriptors,
                increments=[{"tokens": estimated_input_tokens} for _ in itpm_descriptors],
                parent_otel_span=parent_otel_span,
            )
            if itpm_descriptors
            else None
        )
        if itpm_response is not None and itpm_response["overall_code"] == "OVER_LIMIT":
            return itpm_response, 0, 0
        itpm_reserved: Final = estimated_input_tokens if itpm_response is not None else 0

        if otpm_descriptors:
            otpm_response: Final = await self.atomic_check_and_increment_by_n(
                descriptors=otpm_descriptors,
                increments=[{"tokens": estimated_output_tokens} for _ in otpm_descriptors],
                parent_otel_span=parent_otel_span,
            )
            if otpm_response["overall_code"] == "OVER_LIMIT":
                if itpm_reserved > 0:
                    await self._refund_reserved_tokens(
                        scopes=[(d["key"], d["value"]) for d in itpm_descriptors],
                        amount=itpm_reserved,
                        reservation_windows=itpm_response.get("reservation_windows", frozenset()),
                        parent_otel_span=parent_otel_span,
                    )
                return otpm_response, 0, 0
            statuses: Final = (
                [
                    *itpm_response["statuses"],
                    *otpm_response["statuses"],
                ]
                if itpm_response is not None
                else otpm_response["statuses"]
            )
            return (
                RateLimitResponse(
                    overall_code="OK",
                    statuses=statuses,
                    reservation_windows=(
                        (
                            itpm_response.get("reservation_windows", frozenset())
                            if itpm_response is not None
                            else frozenset()
                        )
                        | otpm_response.get("reservation_windows", frozenset())
                    ),
                ),
                itpm_reserved,
                estimated_output_tokens,
            )

        assert itpm_response is not None
        return itpm_response, itpm_reserved, 0

    @with_service_target(RATE_LIMITS_TARGET)
    async def enforce_project_io_token_quota_for_frame(
        self,
        user_api_key_dict: UserAPIKeyAuth | None,
        requested_model: str | None,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> None:
        """Reserve one WebSocket ``response.create`` frame's tokens against
        the caller's project ITPM/OTPM quota.

        The Responses WebSocket connection-level pre-call hook only runs once
        per connection, but a connection accepts many ``response.create``
        frames over its lifetime. Without this, a project caller could send
        unlimited high-token generations after a single minimal reservation.
        There is no per-frame post-call hook to reconcile against, so --
        like the batch rate limiter -- this charges the estimate immediately
        and never refunds it.
        """
        if user_api_key_dict is None:
            return
        descriptors: Final[list[RateLimitDescriptor]] = []  # mutable-ok: descriptor helper appends in place
        self.add_project_io_token_rate_limit_descriptors_from_metadata(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model,
            descriptors=descriptors,
        )
        if not descriptors:
            return
        response, _itpm_reserved, _otpm_reserved = await self.reserve_io_tokens(
            descriptors=descriptors,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        if response["overall_code"] == "OVER_LIMIT":
            self._handle_rate_limit_error(response, descriptors, requested_model)

    @with_service_target(RATE_LIMITS_TARGET)
    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        connection_stash: Final = get_request_stash()
        tags: Final = sorted(connection_stash.tpm_limited_tags) if connection_stash is not None else []
        agent_windows: Final = connection_stash.socket_agent_windows if connection_stash is not None else ()
        hidden_stash: Final = _request_stash.set(None)
        try:
            descriptors: Final = await self._build_request_rate_limit_descriptors(
                user_api_key_dict,
                MappingProxyType({"model": requested_model, "metadata": {"tags": tags}}),
                CallTypes.aresponses_websocket.value,
            )
        finally:
            _request_stash.reset(hidden_stash)
        built_scopes: Final = frozenset((d["key"], d["value"]) for d in descriptors)
        tpm_windows: Final = _socket_turn_tpm_windows(
            (*descriptors, *(d for d in agent_windows if (d["key"], d["value"]) not in built_scopes))
        )
        try:
            response: Final = await self.should_rate_limit(
                tpm_windows,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                read_only=True,
                skip_parallel_check=True,
            )
        except RateLimitUnverifiableError as unverifiable:
            raise ServiceUnavailableError(
                message=RATE_LIMIT_UNVERIFIABLE_MESSAGE, llm_provider="litellm_proxy", model=requested_model or ""
            ) from unverifiable
        if response["overall_code"] == "OVER_LIMIT":
            self._handle_rate_limit_error(response, tpm_windows, requested_model)

    def rate_limited_model(self, requested_model: str | None) -> RateLimitedModel | None:
        """The requested name and the model group it resolves to, so limits keyed by either name are found."""
        return self._rate_limited_model(requested_model)

    def _rate_limited_model(self, requested_model: str | None) -> RateLimitedModel | None:
        if not requested_model:
            return None
        return RateLimitedModel(
            requested=requested_model,
            group=self._model_group_resolver(requested_model) or requested_model,
        )

    def _rate_limited_models(self, model: RateLimitedModel | None) -> tuple[RateLimitedModel, ...]:
        if model is None:
            return ()
        return (
            model,
            *(
                target
                for candidate in expand_model_latest_alias(model.requested)[1:]
                if (target := self._rate_limited_model(candidate)) is not None
            ),
        )

    def create_organization_rate_limit_descriptor(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None = None
    ) -> list[RateLimitDescriptor]:
        descriptors: Final[list[RateLimitDescriptor]] = []

        # Global org rate limits
        if user_api_key_dict.org_id is not None and (
            user_api_key_dict.organization_rpm_limit is not None or user_api_key_dict.organization_tpm_limit is not None
        ):
            descriptors.append(
                RateLimitDescriptor(
                    key="organization",
                    value=user_api_key_dict.org_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.organization_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.organization_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        model_descriptors: Final = tuple(
            descriptor
            for model in self._rate_limited_models(self._rate_limited_model(requested_model))
            if (descriptor := self._organization_model_rate_limit_descriptor(user_api_key_dict, model)) is not None
        )
        return [*descriptors, *_coalesce_rate_limit_descriptors(model_descriptors)]

    def _organization_model_rate_limit_descriptor(
        self, user_api_key_dict: UserAPIKeyAuth, model: RateLimitedModel
    ) -> RateLimitDescriptor | None:
        model_specific_tpm_limit: Final = model.limit_in(
            get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_tpm_limit")
        )
        model_specific_rpm_limit: Final = model.limit_in(
            get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_rpm_limit")
        )
        if model_specific_tpm_limit is None and model_specific_rpm_limit is None:
            return None
        return RateLimitDescriptor(
            key="model_per_organization",
            value=f"{user_api_key_dict.org_id}:{model.group}",
            rate_limit={
                "requests_per_unit": model_specific_rpm_limit,
                "tokens_per_unit": model_specific_tpm_limit,
                "window_size": self.window_size,
            },
        )

    def _customer_hierarchy_rate_limit_descriptors(
        self, user_api_key_dict: UserAPIKeyAuth
    ) -> tuple[RateLimitDescriptor, ...]:
        """One descriptor per customer org, team and member level that carries a TPM or RPM limit."""
        scopes: Final = user_api_key_dict.customer_hierarchy.scopes if user_api_key_dict.customer_hierarchy else ()
        return tuple(
            RateLimitDescriptor(
                key=scope.key,
                value=scope.entity_id,
                rate_limit={
                    "requests_per_unit": scope.rpm_limit,
                    "tokens_per_unit": scope.tpm_limit,
                    "window_size": self.window_size,
                },
            )
            for scope in scopes
            if scope.rpm_limit is not None or scope.tpm_limit is not None
        )

    def _customer_hierarchy_model_rate_limit_descriptors(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> tuple[RateLimitDescriptor, ...]:
        """The ``model_per_customer_*`` descriptors for every level whose budget limits this model group."""
        model: Final = self._rate_limited_model(requested_model)
        if model is None or user_api_key_dict.customer_hierarchy is None:
            return ()
        limits: Final = tuple(
            (scope, model.limit_in(scope.model_tpm_limit), model.limit_in(scope.model_rpm_limit))
            for scope in user_api_key_dict.customer_hierarchy.scopes
        )
        return tuple(
            RateLimitDescriptor(
                key=f"model_per_{scope.key}",
                value=f"{scope.entity_id}:{model.group}",
                rate_limit={
                    "requests_per_unit": rpm_limit,
                    "tokens_per_unit": tpm_limit,
                    "window_size": self.window_size,
                },
            )
            for scope, tpm_limit, rpm_limit in limits
            if tpm_limit is not None or rpm_limit is not None
        )

    def _end_user_model_rate_limit_descriptor(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> RateLimitDescriptor | None:
        model: Final = self._rate_limited_model(requested_model)
        if model is None or not user_api_key_dict.end_user_id:
            return None
        tpm_limit: Final = model.limit_in(user_api_key_dict.end_user_model_tpm_limit)
        rpm_limit: Final = model.limit_in(user_api_key_dict.end_user_model_rpm_limit)
        if tpm_limit is None and rpm_limit is None:
            return None
        return RateLimitDescriptor(
            key="model_per_end_user",
            value=f"{user_api_key_dict.end_user_id}:{model.group}",
            rate_limit={
                "requests_per_unit": rpm_limit,
                "tokens_per_unit": tpm_limit,
                "window_size": self.window_size,
            },
        )

    def _add_model_per_key_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """
        Add model-specific rate limit descriptor for API key if applicable.

        Args:
            user_api_key_dict: User API key authentication dictionary
            requested_model: The model being requested
            descriptors: List of rate limit descriptors to append to
        """
        from litellm.proxy.auth.auth_utils import (
            get_key_model_rpm_limit,
            get_key_model_tpm_limit,
        )

        model: Final = self._rate_limited_model(requested_model)
        if model is None:
            return
        borrowing: Final = borrowing_team_model_limits(user_api_key_dict, model)
        model_specific_tpm_limit: Final = (
            None
            if borrowing.tpm is not None
            else model.limit_in(get_key_model_tpm_limit(user_api_key_dict, model_name=model.group))
        )
        model_specific_rpm_limit: Final = (
            None
            if borrowing.rpm is not None
            else model.limit_in(get_key_model_rpm_limit(user_api_key_dict, model_name=model.group))
        )
        if model_specific_tpm_limit is None and model_specific_rpm_limit is None:
            return

        descriptors.append(
            RateLimitDescriptor(
                key="model_per_key",
                value=f"{user_api_key_dict.api_key}:{model.group}",
                rate_limit={
                    "requests_per_unit": model_specific_rpm_limit,
                    "tokens_per_unit": model_specific_tpm_limit,
                    "window_size": self.window_size,
                },
            )
        )

    def _add_tag_per_key_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """
        Add per-request-tag rpm limit descriptors for the API key.

        Each tag carried on the request that has a configured limit gets its own
        ``{api_key}:{tag}`` counter, so a burst on one tag/group never consumes
        another's budget. Tags without a configured limit fall through to the
        key-level descriptor.
        """
        if not user_api_key_dict.api_key:
            return

        tag_rpm_limit: Final = get_key_tag_rpm_limit(user_api_key_dict) or {}
        if not tag_rpm_limit:
            return

        for tag in dict.fromkeys(get_tags_from_request_body(data)):
            rpm_limit = tag_rpm_limit.get(tag)
            if rpm_limit is None:
                continue
            descriptors.append(
                RateLimitDescriptor(
                    key="tag_per_key",
                    value=f"{user_api_key_dict.api_key}:{tag}",
                    rate_limit={
                        "requests_per_unit": rpm_limit,
                        "tokens_per_unit": None,
                        "window_size": self.window_size,
                    },
                )
            )

    def _add_mcp_per_key_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        mcp_server_name: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """
        Add a per-MCP-server rpm descriptor for the API key, if a limit is
        configured for the server being called.

        MCP tool calls have no token usage, so only requests_per_unit is set;
        tokens_per_unit stays None so the TPM reservation path is never engaged.
        """
        if not mcp_server_name or not user_api_key_dict.api_key:
            return

        metadata: Final = _REQUEST_RATE_LIMIT_DATA.validate_python(user_api_key_dict.metadata or {})
        mcp_rpm_limit: Final = _MCP_KEY_RPM_LIMITS.validate_python(metadata.get("mcp_rpm_limit"))
        if not mcp_rpm_limit:
            return

        server_rpm_limit: Final = mcp_rpm_limit.get(mcp_server_name)
        if server_rpm_limit is None:
            return

        descriptors.append(
            RateLimitDescriptor(
                key="mcp_per_key",
                value=f"{user_api_key_dict.api_key}:{mcp_server_name}",
                rate_limit={
                    "requests_per_unit": server_rpm_limit,
                    "tokens_per_unit": None,
                    "window_size": self.window_size,
                },
            )
        )

    def _add_mcp_per_team_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        mcp_server_name: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """
        Add a per-MCP-server rpm descriptor for the team, if a limit is
        configured for the server being called.
        """
        from litellm.proxy.auth.auth_utils import get_team_mcp_rpm_limit

        if not mcp_server_name:
            return

        # Which teams' buckets does this call charge? A key is pinned to exactly one team. A keyless
        # MCP-admitted subject reaches servers through SEVERAL teams at once and has no team_id, so
        # without the second source below its calls charged no team bucket at all and it outran every
        # team's mcp_rpm_limit. Every applicable team is charged rather than one being picked: the
        # limiter enforces all descriptors, so each team's own ceiling binds on a call made through
        # its grant, and there is no arbitrary attribution when several teams grant the same server.
        source_team_limits: Final = user_api_key_dict.mcp_source_team_rpm_limits or {}
        primary_team_ids: Final = (user_api_key_dict.team_id,) if user_api_key_dict.team_id else ()
        team_ids: Final = tuple(dict.fromkeys((*primary_team_ids, *source_team_limits)))
        team_limits: Final = tuple(
            (
                team_id,
                get_team_mcp_rpm_limit(user_api_key_dict) or source_team_limits.get(team_id)
                if team_id == user_api_key_dict.team_id
                else source_team_limits[team_id],
            )
            for team_id in team_ids
        )

        for team_id, mcp_rpm_limit in team_limits:
            if not team_id or not mcp_rpm_limit:
                continue
            server_rpm_limit = mcp_rpm_limit.get(mcp_server_name)
            if server_rpm_limit is None:
                continue
            descriptors.append(
                RateLimitDescriptor(
                    key="mcp_per_team",
                    value=f"{team_id}:{mcp_server_name}",
                    rate_limit={
                        "requests_per_unit": server_rpm_limit,
                        "tokens_per_unit": None,
                        "window_size": self.window_size,
                    },
                )
            )

    async def enforce_mcp_server_rate_limits(
        self,
        user_api_key_dict: UserAPIKeyAuth | None,
        server: MCPServer,
    ) -> None:
        descriptors: Final[list[RateLimitDescriptor]] = []  # mutable-ok: existing descriptor helpers append in place
        mcp_server_name: Final = server.alias or server.server_name or server.name
        if user_api_key_dict is not None:
            self._add_mcp_per_key_rate_limit_descriptor(
                user_api_key_dict=user_api_key_dict,
                mcp_server_name=mcp_server_name,
                descriptors=descriptors,
            )
            self._add_mcp_per_team_rate_limit_descriptor(
                user_api_key_dict=user_api_key_dict,
                mcp_server_name=mcp_server_name,
                descriptors=descriptors,
            )
        if server.rpm is not None:
            descriptors.append(
                RateLimitDescriptor(
                    key="mcp_server",
                    value=server.server_id,
                    rate_limit={
                        "requests_per_unit": server.rpm,
                        "tokens_per_unit": None,
                        "window_size": self.window_size,
                    },
                )
            )
        if not descriptors:
            return

        parent_otel_span: Final = user_api_key_dict.parent_otel_span if user_api_key_dict is not None else None
        response: Final = await self.atomic_check_and_increment_by_n(
            descriptors=descriptors,
            increments=[{"requests": 1} for _ in descriptors],
            parent_otel_span=parent_otel_span,
        )
        if response["overall_code"] == "OVER_LIMIT":
            self._handle_rate_limit_error(response, descriptors)

    def _should_enforce_rate_limit(
        self,
        limit_type: str | None,
        model_has_failures: bool,
    ) -> bool:
        """
        Determine if rate limit should be enforced based on limit type and model health.

        Args:
            limit_type: Type of rate limit ("dynamic", "guaranteed_throughput", "best_effort_throughput", or None)
            model_has_failures: Whether the model has recent failures

        Returns:
            True if rate limit should be enforced, False otherwise
        """
        if limit_type == "dynamic":
            # Dynamic mode: only enforce if model has failures
            return model_has_failures
        # All other modes (including None): always enforce
        return True

    def _get_enforced_limit(
        self,
        limit_value: int | None,
        limit_type: str | None,
        model_has_failures: bool,
    ) -> int | None:
        """
        Get the rate limit value to enforce based on limit type and model health.

        Args:
            limit_value: The configured limit value
            limit_type: Type of rate limit ("dynamic", "guaranteed_throughput", "best_effort_throughput", or None)
            model_has_failures: Whether the model has recent failures

        Returns:
            The limit value if it should be enforced, None otherwise
        """
        if limit_value is None:
            return None

        if self._should_enforce_rate_limit(
            limit_type=limit_type,
            model_has_failures=model_has_failures,
        ):
            return limit_value

        return None

    def _is_dynamic_rate_limiting_enabled(
        self,
        rpm_limit_type: str | None,
        tpm_limit_type: str | None,
    ) -> bool:
        """
        Check if dynamic rate limiting is enabled for either RPM or TPM.

        Args:
            rpm_limit_type: RPM rate limit type
            tpm_limit_type: TPM rate limit type

        Returns:
            True if dynamic mode is enabled for either limit type
        """
        return rpm_limit_type == "dynamic" or tpm_limit_type == "dynamic"

    def _get_agent_from_registry(self, agent_id: str) -> "AgentResponse | None":
        """Look up an agent from the in-memory registry by ID."""
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        return global_agent_registry.get_agent_by_id(agent_id=agent_id)

    def _get_resolved_agent_id(self, user_api_key_dict: UserAPIKeyAuth, data: dict) -> str | None:
        """
        Resolve the agent_id from either the API key or request metadata.
        Key-level agent_id takes precedence over metadata/header-supplied agent_id.
        """
        key_agent_id: Final[str | None] = getattr(user_api_key_dict, "agent_id", None)
        if key_agent_id:
            return key_agent_id
        metadata: Final = data.get("metadata") or {}
        litellm_metadata: Final = data.get("litellm_metadata") or {}
        return metadata.get("agent_id") or litellm_metadata.get("agent_id")

    def _get_session_id_from_data(self, data: dict) -> str | None:
        """Extract session_id from request metadata or litellm_session_id."""
        session_id = data.get("litellm_session_id")
        if session_id:
            return str(session_id)
        metadata: Final = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id:
            return str(session_id)
        litellm_metadata: Final = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id:
            return str(session_id)
        return None

    def _create_agent_rate_limit_descriptors(
        self,
        agent_id: str,
        data: dict,
        policy: "AgentResponse | None" = None,
    ) -> list[RateLimitDescriptor]:
        """
        Create rate limit descriptors for agent-level and session-level limits.

        Agent-level: caps total RPM/TPM across all sessions for a given agent.
        Session-level: caps RPM/TPM within a single session (identified by session_id).
        """
        descriptors: Final[list[RateLimitDescriptor]] = []

        agent: Final = policy if policy is not None else self._get_agent_from_registry(agent_id)
        if agent is None:
            return descriptors

        agent_rpm: Final = getattr(agent, "rpm_limit", None)
        agent_tpm: Final = getattr(agent, "tpm_limit", None)
        if agent_rpm is not None or agent_tpm is not None:
            descriptors.append(
                RateLimitDescriptor(
                    key="agent",
                    value=agent_id,
                    rate_limit={
                        "requests_per_unit": agent_rpm,
                        "tokens_per_unit": agent_tpm,
                        "window_size": self.window_size,
                    },
                )
            )

        session_rpm: Final = getattr(agent, "session_rpm_limit", None)
        session_tpm: Final = getattr(agent, "session_tpm_limit", None)
        if session_rpm is not None or session_tpm is not None:
            session_id: Final = self._get_session_id_from_data(data)
            if session_id is not None:
                descriptors.append(
                    RateLimitDescriptor(
                        key="agent_session",
                        value=f"{agent_id}:{session_id}",
                        rate_limit={
                            "requests_per_unit": session_rpm,
                            "tokens_per_unit": session_tpm,
                            "window_size": self.window_size,
                        },
                    )
                )

        return descriptors

    async def _create_tag_rate_limit_descriptors(self, data: Mapping[str, object]) -> tuple[RateLimitDescriptor, ...]:
        tags: Final = tuple(dict.fromkeys(get_tags_from_request_body(data)))
        if not tags:
            return ()
        tag_limits: Final = await self._tag_rate_limit_resolver(tags)
        return tuple(
            _tag_rate_limit_descriptor(tag, limit, self.window_size)
            for tag in tags
            if (limit := tag_limits.get(tag)) is not None
        )

    def create_rate_limit_descriptors(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        rpm_limit_type: str | None,
        tpm_limit_type: str | None,
        model_has_failures: bool,
        call_type: str | None = None,
    ) -> list[RateLimitDescriptor]:
        """
        Create all rate limit descriptors for the request.

        Returns list of descriptors for API key, user, team, team member, end user,
        model-specific, agent, and agent-session limits.
        """
        descriptors: Final[list[RateLimitDescriptor]] = []  # mutable-ok: existing descriptor helpers append in place

        # API Key rate limits
        if user_api_key_dict.api_key and (
            user_api_key_dict.rpm_limit is not None
            or user_api_key_dict.tpm_limit is not None
            or user_api_key_dict.max_parallel_requests is not None
        ):
            throttle_pct: Final = user_api_key_dict.budget_throttle_pct
            descriptors.append(
                RateLimitDescriptor(
                    key="api_key",
                    value=user_api_key_dict.api_key,
                    rate_limit={
                        "requests_per_unit": self._get_enforced_limit(
                            limit_value=throttled_limit(user_api_key_dict.rpm_limit, throttle_pct),
                            limit_type=rpm_limit_type,
                            model_has_failures=model_has_failures,
                        ),
                        "tokens_per_unit": self._get_enforced_limit(
                            limit_value=throttled_limit(user_api_key_dict.tpm_limit, throttle_pct),
                            limit_type=tpm_limit_type,
                            model_has_failures=model_has_failures,
                        ),
                        "max_parallel_requests": user_api_key_dict.max_parallel_requests,
                        "window_size": self.window_size,
                    },
                )
            )

        # User rate limits
        if user_api_key_dict.user_id and (
            user_api_key_dict.user_rpm_limit is not None or user_api_key_dict.user_tpm_limit is not None
        ):
            descriptors.append(
                RateLimitDescriptor(
                    key="user",
                    value=user_api_key_dict.user_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.user_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.user_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # Team rate limits
        if user_api_key_dict.team_id and (
            user_api_key_dict.team_rpm_limit is not None or user_api_key_dict.team_tpm_limit is not None
        ):
            descriptors.append(
                RateLimitDescriptor(
                    key="team",
                    value=user_api_key_dict.team_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.team_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.team_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # Team Member rate limits
        if user_api_key_dict.user_id and (
            user_api_key_dict.team_member_rpm_limit is not None or user_api_key_dict.team_member_tpm_limit is not None
        ):
            team_member_value: Final = f"{user_api_key_dict.team_id}:{user_api_key_dict.user_id}"
            descriptors.append(
                RateLimitDescriptor(
                    key="team_member",
                    value=team_member_value,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.team_member_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.team_member_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # End user rate limits
        if user_api_key_dict.end_user_id and (
            user_api_key_dict.end_user_rpm_limit is not None or user_api_key_dict.end_user_tpm_limit is not None
        ):
            descriptors.append(
                RateLimitDescriptor(
                    key="end_user",
                    value=user_api_key_dict.end_user_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.end_user_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.end_user_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        descriptors.extend(self._customer_hierarchy_rate_limit_descriptors(user_api_key_dict))

        # Model rate limits
        requested_model: Final = data.get("model", None)
        model_candidates: Final = expand_model_latest_alias(requested_model if isinstance(requested_model, str) else "")
        for candidate in model_candidates:
            self._add_model_per_key_rate_limit_descriptor(
                user_api_key_dict=user_api_key_dict,
                requested_model=candidate,
                descriptors=descriptors,
            )
            if (end_user_model := self._end_user_model_rate_limit_descriptor(user_api_key_dict, candidate)) is not None:
                descriptors.append(end_user_model)
            descriptors.extend(self._customer_hierarchy_model_rate_limit_descriptors(user_api_key_dict, candidate))

        # Per-request-tag rate limits scoped to this key
        self._add_tag_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            data=data,
            descriptors=descriptors,
        )

        for candidate in model_candidates:
            self._add_team_model_rate_limit_descriptor_from_metadata(
                user_api_key_dict=user_api_key_dict,
                requested_model=candidate,
                descriptors=descriptors,
            )
        self._add_team_ptu_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model if isinstance(requested_model, str) else None,
            descriptors=descriptors,
        )

        resolved_agent_id: Final = self._get_resolved_agent_id(user_api_key_dict, data)
        for agent_id in dict.fromkeys((resolved_agent_id, user_api_key_dict.invoked_agent_id)):
            if agent_id is None:
                continue
            descriptors.extend(
                self._create_agent_rate_limit_descriptors(
                    agent_id=agent_id,
                    data=data,
                    policy=(
                        user_api_key_dict.managed_agent_policy
                        if agent_id == user_api_key_dict.agent_id
                        else user_api_key_dict.invoked_agent_policy
                    ),
                )
            )

        return descriptors if len(model_candidates) == 1 else list(_coalesce_rate_limit_descriptors(descriptors))

    _create_rate_limit_descriptors = create_rate_limit_descriptors

    async def _check_model_has_recent_failures(
        self,
        model: str,
        parent_otel_span: Span | None = None,
    ) -> bool:
        """
        Check if any deployment for this model has recent failures by using
        the router's existing failure tracking.

        Returns True if any deployment has failures in the current minute.
        """
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return False

        try:
            # Get all deployments for this model
            model_list: Final = llm_router.registry().deployments(model)
            if not model_list:
                return False

            # Check each deployment's failure count
            for deployment in model_list:
                deployment_id = deployment.get("model_info", {}).get("id")
                if not deployment_id:
                    continue

                # Use router's existing failure tracking
                failure_count = llm_router.health().failures_this_minute(deployment_id)

                if failure_count > DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE:
                    verbose_proxy_logger.debug(
                        "[Dynamic Rate Limit] Deployment %s has %s failures in current minute - enforcing rate limits for model %s",
                        deployment_id,
                        failure_count,
                        model,
                    )
                    return True

            verbose_proxy_logger.debug(
                "[Dynamic Rate Limit] No failures detected for model %s - allowing dynamic exceeding", model
            )
            return False

        except Exception as e:
            verbose_proxy_logger.debug("Error checking model failure status: %s, defaulting to enforce limits", e)
            # Fail safe: enforce limits if we can't check
            return True

    def get_rate_limiter_for_call_type(self, call_type: str) -> CallTypeRateLimiter | None:
        """Get the rate limiter for the call type."""
        if call_type == "acreate_batch":
            batch_limiter: Final = self._get_batch_rate_limiter()
            return batch_limiter
        return None

    def _key_owns_model_limit(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: RateLimitedModel,
        rate_limit_key: Literal["model_rpm_limit", "model_tpm_limit"],
    ) -> bool:
        return model.limit_in(get_key_own_model_rate_limit(user_api_key_dict, rate_limit_key)) is not None

    def _inherited_team_model_limit(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: RateLimitedModel,
        rate_limit_key: Literal["model_rpm_limit", "model_tpm_limit"],
    ) -> int | None:
        return inherited_team_model_limit(user_api_key_dict, model, rate_limit_key)

    def _key_owns_model_tpm_limit_from_request_metadata(
        self,
        request_metadata: Mapping[str, object],
        model: RateLimitedModel | None,
    ) -> bool:
        if model is None:
            return False
        key_view: Final = UserAPIKeyAuth.model_validate(
            {
                "metadata": request_metadata.get("user_api_key_metadata") or {},
                "model_max_budget": request_metadata.get("user_api_key_model_max_budget") or {},
            }
        )
        return self._key_owns_model_limit(key_view, model, "model_tpm_limit")

    def _add_team_model_rate_limit_descriptor_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        model: Final = self._rate_limited_model(requested_model)
        if model is None:
            return
        borrowing: Final = borrowing_team_model_limits(user_api_key_dict, model)
        team_rpm_limit: Final = (
            None
            if borrowing.rpm is not None
            else inherited_team_model_limit(user_api_key_dict, model, "model_rpm_limit")
        )
        team_tpm_limit: Final = (
            None
            if borrowing.tpm is not None
            else inherited_team_model_limit(user_api_key_dict, model, "model_tpm_limit")
        )
        if team_rpm_limit is None and team_tpm_limit is None:
            return
        descriptors.append(
            RateLimitDescriptor(
                key="model_per_team",
                value=f"{user_api_key_dict.team_id}:{model.group}",
                rate_limit={
                    "requests_per_unit": team_rpm_limit,
                    "tokens_per_unit": team_tpm_limit,
                    "window_size": self.window_size,
                },
            )
        )

    def _add_team_ptu_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: _RateLimitDescriptorSink,
    ) -> None:
        model: Final = self._rate_limited_model(requested_model)
        if model is None or not user_api_key_dict.team_id:
            return
        ceiling: Final = self._ptu_team_ceiling_resolver(user_api_key_dict.team_id, model.group)
        if ceiling is None:
            return
        stash: Final = get_request_stash()
        if stash is not None:
            stash.ptu_ceiling = ceiling
        descriptors.append(
            RateLimitDescriptor(
                key=PTU_TEAM_DESCRIPTOR_KEY,
                value=f"{user_api_key_dict.team_id}:{ceiling.model_group}",
                rate_limit=RateLimitDescriptorRateLimitObject(
                    requests_per_unit=None, tokens_per_unit=ceiling.tpm_limit, window_size=self.window_size
                ),
            )
        )

    def _add_project_model_rate_limit_descriptor_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: _RateLimitDescriptorSink,
    ) -> None:
        candidates: Final = expand_model_latest_alias(requested_model) if requested_model else (None,)
        model_descriptors: Final = tuple(
            descriptor
            for candidate in candidates
            if (descriptor := self._project_model_rate_limit_descriptor(user_api_key_dict, candidate)) is not None
        )
        for descriptor in _coalesce_rate_limit_descriptors(model_descriptors):
            descriptors.append(descriptor)

    def _project_model_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
    ) -> RateLimitDescriptor | None:
        """Add project model rate limit descriptor from project_metadata if applicable."""
        model: Final = self._rate_limited_model(requested_model)
        if model is None:
            return None
        model_specific_tpm_limit: Final = model.limit_in(
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_tpm_limit")
        )
        model_specific_rpm_limit: Final = model.limit_in(
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_rpm_limit")
        )
        if model_specific_tpm_limit is None and model_specific_rpm_limit is None:
            return None
        return RateLimitDescriptor(
            key="model_per_project",
            value=f"{user_api_key_dict.project_id}:{model.group}",
            rate_limit={
                "requests_per_unit": model_specific_rpm_limit,
                "tokens_per_unit": model_specific_tpm_limit,
                "window_size": self.window_size,
            },
        )

    def add_project_io_token_rate_limit_descriptors_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: _RateLimitDescriptorSink,
    ) -> None:
        candidates: Final = expand_model_latest_alias(requested_model) if requested_model else (None,)
        model_descriptors: Final = tuple(
            itertools.chain.from_iterable(
                self._project_io_token_rate_limit_descriptors(user_api_key_dict, candidate) for candidate in candidates
            )
        )
        for descriptor in _coalesce_rate_limit_descriptors(model_descriptors):
            descriptors.append(descriptor)

    def _project_io_token_rate_limit_descriptors(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
    ) -> tuple[RateLimitDescriptor, ...]:
        """Add project-scoped ITPM/OTPM descriptors from project_metadata.

        Enforced independently of, and alongside, the combined ``model_per_project``
        TPM descriptor above -- these give Bedrock Mantle-style separate input/output
        token quotas at the project level.
        """
        model: Final = self._rate_limited_model(requested_model)
        if model is None or user_api_key_dict.project_id is None:
            return ()

        model_itpm_limit: Final = model.limit_in(
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_itpm_limit")
        )
        model_otpm_limit: Final = model.limit_in(
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_otpm_limit")
        )

        descriptor_value: Final = f"{user_api_key_dict.project_id}:{model.group}"
        return tuple(
            RateLimitDescriptor(
                key=key,
                value=descriptor_value,
                rate_limit={
                    "requests_per_unit": None,
                    "tokens_per_unit": limit,
                    "window_size": self.window_size,
                },
            )
            for key, limit in (
                (PROJECT_ITPM_DESCRIPTOR_KEY, model_itpm_limit),
                (PROJECT_OTPM_DESCRIPTOR_KEY, model_otpm_limit),
            )
            if limit is not None
        )

    def _handle_rate_limit_error(
        self,
        response: RateLimitResponse,
        descriptors: Sequence[RateLimitDescriptor],
        requested_model: str | None = None,
    ) -> None:
        """Handle rate limit exceeded by raising :class:`ProxyRateLimitError` (a 429)."""
        for status in response["statuses"]:
            if status["code"] == "OVER_LIMIT":
                descriptor_key = status["descriptor_key"]
                matching_descriptor = next(
                    (
                        desc
                        for desc in descriptors
                        if desc["key"] == descriptor_key
                        and ((status_value := status.get("descriptor_value")) is None or desc["value"] == status_value)
                    ),
                    None,
                )
                descriptor_value = matching_descriptor["value"] if matching_descriptor is not None else "unknown"

                now = self._get_current_time().timestamp()
                reset_time = now + self.window_size
                reset_time_formatted = datetime.fromtimestamp(reset_time, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )

                remaining_display = max(0, status["limit_remaining"])
                rate_limit_type = status["rate_limit_type"]
                current_limit = status["current_limit"]

                detail = (
                    f"Rate limit exceeded for {descriptor_key}: {descriptor_value}. "
                    f"Limit type: {rate_limit_type}. "
                    f"Current limit: {current_limit}, Remaining: {remaining_display}. "
                    f"Limit resets at: {reset_time_formatted}"
                )

                resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(requested_model)
                raise ProxyRateLimitError(
                    detail=detail,
                    headers={
                        "retry-after": str(self.window_size),
                        "rate_limit_type": str(status["rate_limit_type"]),
                        "rate_limited_by": descriptor_key,
                        "reset_at": reset_time_formatted,
                    },
                    rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                    model=resolved_model,
                    llm_provider=llm_provider,
                    descriptor_key=descriptor_key,
                )

    @staticmethod
    def _estimate_audio_block_tokens(block: object) -> int:
        """
        Token estimate for one ``input_audio`` content block.

        When the block carries a base64 ``data`` payload, the estimate comes
        from the decoded byte count (``len(b64) * 3 // 4 // _AUDIO_BYTES_PER_TOKEN``),
        assuming the lowest reasonable audio bitrate so we never under-reserve
        for higher-quality recordings of the same duration.

        When no payload is present (reference-only block or missing ``data``),
        falls back to ``DEFAULT_AUDIO_TOKEN_ESTIMATE``.
        """
        if not isinstance(block, dict):
            return DEFAULT_AUDIO_TOKEN_ESTIMATE
        input_audio: Final = block.get("input_audio")
        b64_data: Final = input_audio.get("data") if isinstance(input_audio, dict) else None
        if b64_data and isinstance(b64_data, str):
            decoded_bytes: Final = len(b64_data) * 3 // 4
            return max(decoded_bytes // _AUDIO_BYTES_PER_TOKEN, DEFAULT_AUDIO_TOKEN_ESTIMATE)
        return DEFAULT_AUDIO_TOKEN_ESTIMATE

    @classmethod
    def _estimate_audio_content_tokens(cls, messages: object) -> int:
        """
        Sum of per-block audio token estimates across all ``messages``.
        Returns 0 when there are no ``input_audio`` blocks, which the caller
        uses to skip the (relatively expensive) strip pass.
        """
        if not isinstance(messages, list):
            return 0
        return sum(
            cls._estimate_audio_block_tokens(block)
            for message in messages
            if isinstance(message, dict)
            for content in (message.get("content"),)
            if isinstance(content, list)
            for block in content
            if isinstance(block, dict) and block.get("type") == "input_audio"
        )

    @staticmethod
    def _strip_audio_content_blocks(messages: object) -> object:
        """
        Drop ``input_audio`` content blocks before passing ``messages`` to
        ``token_counter``, which raises ``ValueError`` on them (no per-type
        handling, unlike images). The audio contribution is added back
        separately via ``DEFAULT_AUDIO_TOKEN_ESTIMATE`` so the rest of the
        message (text/images/tools) still gets counted accurately instead of
        the whole call falling back to the cheap char-count estimate.
        """
        if not isinstance(messages, list):
            return messages
        sanitized: Final[list[object]] = []  # mutable-ok: token_counter requires a list of message dicts
        for message in messages:
            if not isinstance(message, dict):
                sanitized.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, list):
                sanitized.append(message)
                continue
            filtered_content = [
                block for block in content if not (isinstance(block, dict) and block.get("type") == "input_audio")
            ]
            sanitized.append({**message, "content": filtered_content})
        return sanitized

    @staticmethod
    def _responses_input_to_chat_messages(data: object) -> Sequence[object]:
        """
        Convert a Responses API ``input`` (string or list of input items) into
        chat-completion-style messages via the standard LiteLLM transformation
        (the same one guardrails use, e.g. ``purview_dlp.py``), so multimodal
        ``input_image``/``input_text`` content blocks get counted by
        ``token_counter``'s ``messages`` path instead of silently contributing
        zero tokens via its ``text`` path, which only joins plain strings.
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        if not isinstance(data, dict):
            return ()
        return LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=data.get("input") or "",
            responses_api_request=data,
            only_since_compaction=True,
        )

    @staticmethod
    def _count_pretokenized_embedding_input(value: object) -> int | None:
        if not isinstance(value, list):
            return None
        if all(isinstance(token, int) for token in value):
            return len(value)
        if all(
            isinstance(token_ids, list) and all(isinstance(token, int) for token in token_ids) for token_ids in value
        ):
            return sum(len(token_ids) for token_ids in value)
        return None

    @staticmethod
    def _rerank_input_to_text(data: Mapping[str, object]) -> str:
        documents: Final = data.get("documents")
        document_items: Final[Sequence[object]] = documents if isinstance(documents, list) else ()  # pyright: ignore[reportUnknownVariableType]  # rerank documents are validated runtime JSON
        input_parts: Final[tuple[object, ...]] = (  # pyright: ignore[reportUnknownVariableType]  # list narrowing preserves unknown JSON element types
            data.get("query"),
            *document_items,
        )
        return "\n".join(
            str(part)  # pyright: ignore[reportUnknownArgumentType]  # accepted document dicts have provider-defined fields
            for part in input_parts  # pyright: ignore[reportUnknownVariableType]  # runtime JSON list elements remain unknown after list narrowing
            if isinstance(part, (str, dict))
        )

    @staticmethod
    def _extraction_input_text(data: Mapping[str, object]) -> str:
        state: Final = data.get("state")
        return f"{state if isinstance(state, str) else ''}\n{json.dumps(data.get('fields'), default=str)}"

    def _estimate_precise_input_tokens(
        self, data: object, model: str | None, call_type: str | None = None, team_id: str | None = None
    ) -> int:
        """
        Model-aware input token estimate for the project ITPM reservation,
        using ``litellm.token_counter`` -- the same approach the
        deployment-level itpm/otpm check uses in
        ``io_token_rate_limit_check.py``. Unlike the cheap char-count
        estimate the combined-TPM path uses, this accounts for image/tool
        content and derives per-``input_audio``-block estimates from the
        base64 payload size (assuming the lowest reasonable bitrate so
        longer recordings always reserve proportionally more), so a burst
        of multimodal, tool-heavy, or audio-heavy requests can't each
        reserve only the one-token floor and blow past ITPM before
        post-call reconciliation catches up.

        For the Responses API, ``input`` is converted to chat messages first
        (via ``_responses_input_to_chat_messages``) so its own multimodal
        content blocks are counted the same way; ``token_counter``'s ``text``
        argument can only see plain strings in a list, not content blocks.

        Falls back to the cheap char-count estimate if ``token_counter``
        can't resolve a tokenizer for this model (e.g. an unrecognized
        custom model name) or otherwise raises -- the audio add-on still
        applies on top of the fallback.
        """
        from litellm import token_counter

        if not isinstance(data, dict):
            return 0
        is_responses_request: Final = call_type in RESPONSES_API_CALL_TYPES
        translated_request: Final = (
            None if is_responses_request else self._translate_google_genai_native_request(data, call_type)
        )
        is_embedding_request: Final = self._is_embedding_request(data, call_type)
        request: Final = as_str_mapping(data)
        raw_embedding_input: Final = request.get("input") if is_embedding_request and request is not None else None
        pretokenized_input_tokens: Final = (
            self._count_pretokenized_embedding_input(raw_embedding_input) if is_embedding_request else None
        )
        if pretokenized_input_tokens is not None:
            return pretokenized_input_tokens
        embedding_text, embedding_media_tokens = (
            split_embedding_input_tokens(
                raw_embedding_input, lambda: self._media_model_resolver(model or "", team_id), model
            )
            if is_embedding_request
            else (None, 0)
        )

        prompt: Final = data.get("prompt")
        fallback_text: Final = prompt if prompt is not None else data.get("input")
        selected_inputs: Final[tuple[object | None, object | None, object | None, object | None]] = (
            (self._responses_input_to_chat_messages(data), None, data.get("tools"), data.get("tool_choice"))
            if is_responses_request
            else (
                translated_request.get("messages"),
                None,
                translated_request.get("tools"),
                translated_request.get("tool_choice"),
            )
            if translated_request is not None
            else (None, embedding_text, data.get("tools"), data.get("tool_choice"))
            if is_embedding_request
            else (None, self._rerank_input_to_text(data), data.get("tools"), data.get("tool_choice"))
            if call_type in RERANK_API_CALL_TYPES
            else (None, prompt, data.get("tools"), data.get("tool_choice"))
            if call_type in TEXT_COMPLETION_API_CALL_TYPES
            else (None, self._extraction_input_text(_REQUEST_RATE_LIMIT_DATA.validate_python(data)), None, None)
            if call_type in EXTRACTION_API_CALL_TYPES
            else (data.get("messages"), fallback_text, data.get("tools"), data.get("tool_choice"))
        )
        messages, selected_text, countable_tools, countable_tool_choice = selected_inputs

        audio_token_estimate: Final = self._estimate_audio_content_tokens(messages)
        countable_messages: Final = self._strip_audio_content_blocks(messages) if audio_token_estimate > 0 else messages
        chat_pdf_files: Final = (
            inline_pdf_file_count(countable_messages) if call_type in CHAT_COMPLETION_API_CALL_TYPES else 0
        )

        try:
            estimate: Final = max(
                0,
                *(
                    int(
                        token_counter(
                            model=tokenizer_model,
                            messages=countable_messages,
                            text=selected_text,
                            tools=countable_tools,
                            tool_choice=countable_tool_choice,
                            use_default_image_token_count=True,
                        )
                    )
                    + openai_chat_pdf_file_tokens(tokenizer_model, chat_pdf_files)
                    for tokenizer_model in (self._counted_models_resolver(model, team_id) if model else ("",))
                ),
            )
            return estimate + audio_token_estimate + embedding_media_tokens
        except Exception:  # noqa: BLE001  # tokenizer failures degrade to the cheap estimate
            if call_type in RERANK_API_CALL_TYPES and isinstance(selected_text, str):
                return max(0, len(selected_text) // DEFAULT_CHARS_PER_TOKEN)
            estimated_input_tokens, _ = self._estimate_input_and_output_tokens(
                data=data,
                call_type=call_type,
                include_media=is_embedding_request,
                team_id=team_id,
                media_model_resolver=self._media_model_resolver,
            )
            return estimated_input_tokens + audio_token_estimate

    async def _reserve_project_io_tokens_or_raise(
        self,
        descriptors: Sequence[RateLimitDescriptor],
        data: object,
        requested_model: str | None,
        user_api_key_dict: UserAPIKeyAuth,
        tpm_reservation_scopes: Sequence[tuple[str, str]],
        tpm_reservation_amount: int,
        call_type: str | None = None,
        endpoint_type: EndpointType = EndpointType.GENERIC,
        reuses_request_slot: bool = False,
    ) -> None:
        """
        Reserve project-scoped ITPM/OTPM tokens (Bedrock Mantle-style
        separate input/output token buckets), independently of -- and, when
        both are configured, in addition to -- the combined-TPM reservation
        the caller already made. Raises (via ``_handle_rate_limit_error``) on
        an over-limit reservation, first rolling back the combined-TPM
        reservation named by ``tpm_reservation_scopes``/``tpm_reservation_amount``
        if one was made, so a partial reservation never leaks.
        """
        if not isinstance(data, dict):
            return
        stash: Final = claim_request_stash_for_data(data)
        io_token_descriptors: Final = [
            d for d in descriptors if d["key"] in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
        ]
        if not io_token_descriptors:
            return

        configured_otpm_limits: Final = [
            int(v)
            for d in io_token_descriptors
            if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
            for v in [(d.get("rate_limit") or {}).get("tokens_per_unit")]
            if v is not None
        ]
        min_configured_otpm_limit: Final = min(configured_otpm_limits) if configured_otpm_limits else None
        _, raw_estimated_output_tokens = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_otpm_limit,
            call_type=call_type,
            include_media=False,
        )
        raw_estimated_input_tokens: Final = await offload_token_count(self._estimate_precise_input_tokens)(
            data=data, model=requested_model, call_type=call_type, team_id=user_api_key_dict.team_id
        )
        estimated_input_tokens: Final = max(raw_estimated_input_tokens, 1)
        estimated_output_tokens: Final = (
            raw_estimated_output_tokens
            if self._has_explicit_output_cap(data, call_type)
            else max(raw_estimated_output_tokens, 1)
        )

        # Hard-cap generation length so an unbounded response can't overshoot
        # the OTPM budget before post-call reconciliation runs, mirroring the
        # combined-TPM floor cap in the caller.
        self._apply_implicit_output_cap(
            data=data,
            min_configured_limit=min_configured_otpm_limit,
            call_type=call_type,
            endpoint_type=endpoint_type,
        )

        io_response, itpm_reserved, otpm_reserved = await self.reserve_io_tokens(
            descriptors=io_token_descriptors,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        if io_response["overall_code"] == "OVER_LIMIT":
            # A combined-TPM reservation may have already succeeded above for
            # this same request; refund it too, or its counter stays inflated
            # until the window's TTL expires. Mark it released so the
            # ProxyRateLimitError we're about to raise doesn't get refunded
            # a second time when async_post_call_failure_hook sees the same
            # (still-stashed) reservation and refunds it again.
            if tpm_reservation_amount > 0:
                await self._refund_reserved_tokens(
                    scopes=tuple(scope for scope in tpm_reservation_scopes if scope[0] != PTU_TEAM_DESCRIPTOR_KEY),
                    amount=tpm_reservation_amount,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                await self._refund_reserved_tokens(
                    scopes=tuple(scope for scope in tpm_reservation_scopes if scope[0] == PTU_TEAM_DESCRIPTOR_KEY),
                    amount=stash.ptu_reserved_tokens,
                    reservation_windows=stash.ptu_reserved_window_identities,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                stash.reservation_released = True
            if not reuses_request_slot:
                await self._release_stashed_parallel_slot(stash, user_api_key_dict.parent_otel_span)
            self._handle_rate_limit_error(
                response=io_response,
                descriptors=descriptors,
                requested_model=requested_model,
            )

        stash.discard_released_io_reservation()
        if itpm_reserved > 0:
            itpm_scopes: Final = tuple(
                (d["key"], d["value"]) for d in io_token_descriptors if d["key"] == PROJECT_ITPM_DESCRIPTOR_KEY
            )
            stash.itpm_reserved_tokens = itpm_reserved
            stash.itpm_reserved_scopes = frozenset(itpm_scopes)
            stash.itpm_reserved_window_identities = frozenset(
                identity
                for identity in io_response.get("reservation_windows", frozenset())
                if identity[0].scope.descriptor_key == PROJECT_ITPM_DESCRIPTOR_KEY
            )
        if otpm_reserved > 0:
            otpm_scopes: Final = tuple(
                (d["key"], d["value"]) for d in io_token_descriptors if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
            )
            stash.otpm_reserved_tokens = otpm_reserved
            stash.otpm_reserved_scopes = frozenset(otpm_scopes)
            stash.otpm_reserved_window_identities = frozenset(
                identity
                for identity in io_response.get("reservation_windows", frozenset())
                if identity[0].scope.descriptor_key == PROJECT_OTPM_DESCRIPTOR_KEY
            )

        if stash.rate_limit_response is not None:
            stash.rate_limit_response["statuses"].extend(io_response["statuses"])
        elif io_response["statuses"]:
            stash.rate_limit_response = io_response

        verbose_proxy_logger.debug(
            "ITPM/OTPM tokens reserved: itpm=%s, otpm=%s for model %s",
            itpm_reserved,
            otpm_reserved,
            requested_model,
        )

    async def _build_request_rate_limit_descriptors(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Mapping[str, object],
        call_type: str | None,
    ) -> list[RateLimitDescriptor]:  # mutable-ok: the shared generation reservation helpers require a list
        metadata: Final = _REQUEST_RATE_LIMIT_DATA.validate_python(
            user_api_key_dict.metadata or MappingProxyType({})  # pyright: ignore[reportUnknownMemberType]  # validates the legacy auth metadata boundary
        )
        rpm_value: Final = metadata.get("rpm_limit_type")
        tpm_value: Final = metadata.get("tpm_limit_type")
        rpm_limit_type: Final = rpm_value if isinstance(rpm_value, str) else None
        tpm_limit_type: Final = tpm_value if isinstance(tpm_value, str) else None
        model_value: Final = data.get("model")
        requested_model: Final = model_value if isinstance(model_value, str) else None
        model_has_failures: Final = (
            await self._check_model_has_recent_failures(
                model=requested_model,
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )
            if requested_model and self._is_dynamic_rate_limiting_enabled(rpm_limit_type, tpm_limit_type)
            else False
        )
        descriptors: Final = self.create_rate_limit_descriptors(  # pyright: ignore[reportUnknownMemberType]  # legacy helper reads a dictionary with validated keys
            user_api_key_dict=user_api_key_dict,
            data=dict(data),
            rpm_limit_type=rpm_limit_type,
            tpm_limit_type=tpm_limit_type,
            model_has_failures=model_has_failures,
            call_type=call_type,
        )
        self._add_project_model_rate_limit_descriptor_from_metadata(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model,
            descriptors=descriptors,
        )
        self.add_project_io_token_rate_limit_descriptors_from_metadata(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model,
            descriptors=descriptors,
        )
        return [
            *descriptors,
            *self.create_organization_rate_limit_descriptor(user_api_key_dict, requested_model),
            *await self._create_tag_rate_limit_descriptors(data),
        ]

    async def raise_if_over_limit_before_body(self, user_api_key_dict: UserAPIKeyAuth) -> None:
        """Refuse a request from its headers alone once a limit that needs no body is exhausted.

        Counters are only read, never incremented, so an admitted request is still charged by
        ``async_pre_call_hook`` after its body is parsed. Limits that need the model are not checked here.
        """
        descriptors: Final = await self._build_request_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict, data={}, call_type=None
        )
        if not descriptors:
            return
        response: Final = await self.should_rate_limit(
            descriptors=descriptors,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            read_only=True,
            skip_tpm_check=self.tpm_reservation_enabled,
        )
        if response["overall_code"] == "OVER_LIMIT":
            self._handle_rate_limit_error(response=response, descriptors=descriptors)

    async def _release_request_capacity_when_admitted(
        self,
        admission: asyncio.Task[RateLimitResponse],
        acquisition: ParallelSlotAcquisition,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        response: Final = await admission
        if response["overall_code"] == "OK":
            await self._release_parallel_request_slots(acquisition, user_api_key_dict.parent_otel_span)

    async def async_fallback_target_allowed(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
        request_data: Mapping[str, object],
    ) -> bool:
        """Reserve the caller's per-model limits on a router fallback target. False when any is exhausted."""
        data: Final = MappingProxyType({**request_data, "model": model})
        stash: Final = _request_stash.get()
        hidden_stash: Final = _request_stash.set(None)
        try:
            built: Final = await self._build_request_rate_limit_descriptors(user_api_key_dict, data, None)
        finally:
            _request_stash.reset(hidden_stash)
        descriptors: Final = [
            descriptor for descriptor in built if descriptor["key"] in PER_MODEL_RATE_LIMIT_DESCRIPTOR_KEYS
        ]
        allowed: Final = (
            not descriptors
            or (await self.should_rate_limit(descriptors, parent_otel_span=user_api_key_dict.parent_otel_span))[
                "overall_code"
            ]
            == "OK"
        )
        if allowed and stash is not None:
            stash.fallback_models = stash.fallback_models | {model}
        return allowed

    @asynccontextmanager
    async def request_capacity(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
        *,
        request_data: Mapping[str, object] | None = None,
    ) -> AsyncGenerator[None, None]:
        """Charge one non-generation provider request to RPM and hold its concurrency slot."""
        data: Final = MappingProxyType({**(request_data or MappingProxyType({})), "model": model})
        descriptors: Final = await self._build_request_rate_limit_descriptors(user_api_key_dict, data, None)
        acquisition: Final = ParallelSlotAcquisition(
            slot_id=uuid.uuid4().hex,
            gauges=tuple(
                RateLimitScope(d["key"], d["value"])
                for d in descriptors
                if d["rate_limit"] is not None and d["rate_limit"].get("max_parallel_requests") is not None
            ),
        )
        admission: Final = asyncio.create_task(
            self.should_rate_limit(
                descriptors=descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                skip_tpm_check=True,
                parallel_slot_id=acquisition["slot_id"],
            )
        )
        try:
            response: Final = await asyncio.shield(admission)
            if response["overall_code"] == "OVER_LIMIT":
                self._handle_rate_limit_error(response, descriptors, model)
            yield
        finally:
            cleanup: Final = asyncio.create_task(
                self._release_request_capacity_when_admitted(admission, acquisition, user_api_key_dict)
            )
            cancellation: asyncio.CancelledError | None = None  # rebind-ok: retain cancellation until cleanup finishes
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            cleanup.result()
            if cancellation is not None:
                raise cancellation

    @with_service_target(RATE_LIMITS_TARGET)
    @total_timeout_held()
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
        endpoint_type: EndpointType = EndpointType.GENERIC,
    ) -> Exception | str | dict[str, object] | None:
        """
        Pre-call hook to check rate limits before making the API call.
        Supports dynamic rate limiting based on deployment health.
        """
        verbose_proxy_logger.debug("Inside Rate Limit Pre-Call Hook")

        stash: Final = claim_request_stash_for_data(data)
        stash.call_type = call_type
        reuses_request_slot: Final = call_type == CallTypes.call_mcp_tool.value and stash.parallel_slot is not None
        if reuses_request_slot:
            stash.parallel_slot_reused_at = self._get_current_time()

        #########################################################
        # Check if the call type has a specific rate limiter
        # eg. for Batch APIs we need to use the batch rate limiter to read the input file and count the tokens and requests
        #########################################################
        call_type_specific_rate_limiter: Final = self.get_rate_limiter_for_call_type(call_type=call_type)
        if call_type_specific_rate_limiter:
            return await call_type_specific_rate_limiter.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
            )

        request_data: Final = _REQUEST_RATE_LIMIT_DATA.validate_python(data)
        model_value: Final = request_data.get("model")
        requested_model: Final = model_value if isinstance(model_value, str) else None
        descriptors: Final = await self._build_request_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict,
            data=request_data,
            call_type=call_type,
        )
        stash.tpm_limited_tags = frozenset(
            d["value"]
            for d in descriptors
            if d["key"] == "tag" and d["rate_limit"] is not None and d["rate_limit"].get("tokens_per_unit") is not None
        )
        stash.socket_agent_windows = tuple(d for d in descriptors if d["key"] in _AGENT_DESCRIPTOR_KEYS)

        # Only check rate limits if we have descriptors with actual limits
        if descriptors:
            # First pass: RPM and max_parallel_requests sliding-window check.
            # When reservation is enabled, `skip_tpm_check=True` tells
            # should_rate_limit to ignore each descriptor's tokens_per_unit so
            # its +1-per-key Lua / in-memory increment never touches the
            # :tokens counters — those are owned exclusively by the atomic
            # reserve_tpm_tokens path below. Without this, every concurrent
            # in-flight request would pre-inflate the :tokens counter by 1,
            # shrinking the effective TPM budget by N and causing
            # false-positive 429s under bursts. When reservation is disabled,
            # this pass enforces TPM directly from the post-call counters --
            # except for project ITPM/OTPM descriptors, which are excluded
            # then because _reserve_project_io_tokens_or_raise below charges
            # them unconditionally and counting them here too would
            # double-charge every request.
            parallel_gauges: Final = (
                ()
                if reuses_request_slot
                else tuple(
                    RateLimitScope(d["key"], d["value"])
                    for d in descriptors
                    if (d.get("rate_limit") or {}).get("max_parallel_requests") is not None
                )
            )
            parallel_slot_id: Final = uuid.uuid4().hex if parallel_gauges else None

            first_pass_descriptors: Final = (
                descriptors
                if self.tpm_reservation_enabled
                else tuple(
                    d for d in descriptors if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                )
            )
            response: Final = await self.should_rate_limit(
                descriptors=first_pass_descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                skip_tpm_check=self.tpm_reservation_enabled,
                parallel_slot_id=parallel_slot_id,
                skip_parallel_check=reuses_request_slot,
            )

            if response["overall_code"] == "OVER_LIMIT":
                self._handle_rate_limit_error(
                    response=response,
                    descriptors=descriptors,
                    requested_model=requested_model,
                )
            else:
                if not reuses_request_slot:
                    stash.rate_limit_response = response
                if parallel_slot_id is not None:
                    stash.parallel_slot = ParallelSlotAcquisition(
                        slot_id=parallel_slot_id,
                        gauges=parallel_gauges,
                    )

            # ----------------------------------------------------------------
            # TPM token reservation
            # Atomically reserve estimated tokens upfront so concurrent
            # requests cannot all observe "under limit" before any of them
            # has incremented the counter. atomic_check_and_increment_by_n
            # uses Redis Lua when available and falls back to an asyncio-locked
            # in-memory check otherwise — single-worker protection still holds
            # even without Redis.
            # ----------------------------------------------------------------
            ptu_raw_output_limit: Final = (
                stash.ptu_ceiling.raw_output_token_limit() if stash.ptu_ceiling is not None else None
            )
            configured_tpm_limits: Final = tuple(
                ptu_raw_output_limit
                if d["key"] == PTU_TEAM_DESCRIPTOR_KEY and ptu_raw_output_limit is not None
                else int(v)
                for d in descriptors
                if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                and (v := (d.get("rate_limit") or {}).get("tokens_per_unit")) is not None
            )
            has_tpm_limits: Final = bool(configured_tpm_limits)

            # Populated on a successful combined-TPM reservation below, so the
            # project ITPM/OTPM block further down can roll it back if a
            # different bucket in the same request subsequently hits its
            # limit. Stays empty/0 whenever no combined-TPM reservation was
            # made (or it was over limit, in which case execution never
            # reaches the ITPM/OTPM block -- `_handle_rate_limit_error` raises).
            tpm_reservation_scopes: Sequence[tuple[str, str]] = ()  # rebind-ok: set after successful reservation
            tpm_reservation_amount = 0  # rebind-ok: set after successful reservation

            if has_tpm_limits and self.tpm_reservation_enabled:
                min_configured_tpm_limit: Final = min(configured_tpm_limits)

                configured_output_tokens: Final = get_estimated_output_tokens(
                    user_api_key_dict=user_api_key_dict,
                    model_name=requested_model,
                )

                # When the configured TPM cap is small enough to constrain the
                # no-max_tokens floor, also hard-cap the model output so
                # concurrent unbounded generations can't spend past the limit
                # before post-call reconciliation runs.
                self._apply_implicit_output_cap(
                    data=data,
                    min_configured_limit=min_configured_tpm_limit,
                    call_type=call_type,
                    configured_output_tokens=configured_output_tokens,
                    endpoint_type=endpoint_type,
                )

                # Floor at 1 token so contentless requests (/responses,
                # tool-call continuations, empty messages) still flow
                # through the atomic counter and get backpressure when at
                # limit. Without this floor, N concurrent contentless
                # requests would all pass pre-call with no enforcement.
                # Post-call reconciliation refunds the over-reservation
                # delta when actual usage comes in below the floor.
                estimated_tokens: Final = max(
                    await offload_token_count(self._estimate_tokens_for_request)(
                        data=data,
                        model=requested_model,
                        min_configured_tpm_limit=min_configured_tpm_limit,
                        call_type=call_type,
                        configured_output_tokens=configured_output_tokens,
                        team_id=user_api_key_dict.team_id,
                    ),
                    1,
                )

                if configured_output_tokens is not None and estimated_tokens > min_configured_tpm_limit:
                    verbose_proxy_logger.debug(
                        "Reserving %s tokens for model %s (declared %s=%s plus the input estimate) exceeds the "
                        "smallest TPM limit this request is charged against (%s), so it cannot be admitted even "
                        "against an empty window. Lower the declared estimate or raise the TPM limit.",
                        estimated_tokens,
                        requested_model,
                        ESTIMATED_OUTPUT_TOKENS_FIELD,
                        configured_output_tokens,
                        min_configured_tpm_limit,
                    )

                capped_request: Final = _REQUEST_RATE_LIMIT_DATA.validate_python(data)
                ptu_estimated_tokens: Final = await offload_token_count(self._estimate_ptu_tokens_for_request)(
                    ceiling=stash.ptu_ceiling,
                    data=capped_request,
                    min_configured_tpm_limit=min_configured_tpm_limit,
                    call_type=call_type,
                    configured_output_tokens=configured_output_tokens,
                    raw_estimate=estimated_tokens,
                    team_id=user_api_key_dict.team_id,
                )
                if stash.ptu_ceiling is not None and ptu_estimated_tokens > stash.ptu_ceiling.tpm_limit:
                    if not reuses_request_slot:
                        await self._release_stashed_parallel_slot(stash, user_api_key_dict.parent_otel_span)
                    raise PTUShareExceededError(stash.ptu_ceiling, ptu_estimated_tokens)
                tpm_response: Final = await self.reserve_tpm_tokens(
                    descriptors=descriptors,
                    estimated_tokens=estimated_tokens,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                    scope_estimates=MappingProxyType({PTU_TEAM_DESCRIPTOR_KEY: ptu_estimated_tokens}),
                )

                if tpm_response["overall_code"] == "OVER_LIMIT":
                    if not reuses_request_slot:
                        await self._release_stashed_parallel_slot(stash, user_api_key_dict.parent_otel_span)
                    self._handle_rate_limit_error(
                        response=tpm_response,
                        descriptors=descriptors,
                        requested_model=requested_model,
                    )
                else:
                    # Capture the exact (key, value) scopes the reservation
                    # incremented so post-call reconciliation only applies
                    # the (actual - reserved) delta to those — unreserved
                    # scopes get charged the full actual usage instead.
                    stash.discard_released_tpm_reservation()
                    stash.reserved_tokens = estimated_tokens
                    stash.ptu_reserved_tokens = ptu_estimated_tokens
                    stash.ptu_reserved_window_identities = frozenset(
                        identity
                        for identity in tpm_response.get("reservation_windows", frozenset())
                        if identity[0].scope.descriptor_key == PTU_TEAM_DESCRIPTOR_KEY
                    )
                    stash.reserved_model = self._rate_limited_model(requested_model)
                    stash.reserved_scopes = frozenset(
                        (d["key"], d["value"])
                        for d in descriptors
                        if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                        and (d.get("rate_limit") or {}).get("tokens_per_unit") is not None
                    )
                    tpm_reservation_scopes = tuple(  # rebind-ok: record successful reservation scopes
                        stash.reserved_scopes
                    )
                    tpm_reservation_amount = estimated_tokens  # rebind-ok: record successful reservation amount

                    # Merge TPM statuses into the stored rate-limit response
                    # so x-ratelimit-{key}-remaining-tokens / -limit-tokens
                    # headers reach the client. Without this, the RPM-only
                    # response from should_rate_limit (skip_tpm_check=True)
                    # silently drops all token headers.
                    stored_response: Final = stash.rate_limit_response
                    if stored_response is not None:
                        stored_response["statuses"].extend(tpm_response["statuses"])

                    verbose_proxy_logger.debug(
                        "TPM tokens reserved: %s for model %s", estimated_tokens, requested_model
                    )
            await self._reserve_project_io_tokens_or_raise(
                descriptors=descriptors,
                data=data,
                requested_model=requested_model,
                user_api_key_dict=user_api_key_dict,
                tpm_reservation_scopes=tpm_reservation_scopes,
                tpm_reservation_amount=tpm_reservation_amount,
                call_type=call_type,
                endpoint_type=endpoint_type,
                reuses_request_slot=reuses_request_slot,
            )

    def _get_total_tokens_from_usage(
        self, usage: object | None, rate_limit_type: Literal["output", "input", "total"]
    ) -> int:
        """
        Get total tokens from response usage for rate limiting.

        For 'input' and 'total' rate limit types, cached tokens are excluded
        because providers like AWS Bedrock don't count cached tokens toward
        rate limits. This aligns LiteLLM's TPM calculation with provider behavior.
        """
        total_tokens = 0
        cached_tokens = 0

        if usage:
            if isinstance(usage, Usage):
                if rate_limit_type == "output":
                    total_tokens = usage.completion_tokens or 0
                elif rate_limit_type == "input":
                    total_tokens = usage.prompt_tokens or 0
                elif rate_limit_type == "total":
                    total_tokens = usage.total_tokens or 0

                # Get cached tokens to exclude from input/total
                if rate_limit_type in ("input", "total"):
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details is not None:
                        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

            elif isinstance(usage, dict):
                # Responses API usage comes as a dict
                if rate_limit_type == "output":
                    total_tokens = usage.get("completion_tokens", 0) or 0
                elif rate_limit_type == "input":
                    total_tokens = usage.get("prompt_tokens", 0) or 0
                elif rate_limit_type == "total":
                    total_tokens = usage.get("total_tokens", 0) or 0

                # Get cached tokens from dict
                if rate_limit_type in ("input", "total"):
                    prompt_details: Final = usage.get("prompt_tokens_details") or {}
                    if isinstance(prompt_details, dict):
                        cached_tokens = prompt_details.get("cached_tokens", 0) or 0

        # Subtract cached tokens for input/total (providers don't count them)
        if cached_tokens > 0:
            total_tokens = max(0, total_tokens - cached_tokens)

        return total_tokens

    @staticmethod
    def _aggregate_only_total_tokens(usage: Usage | ResponseAPIUsage | Mapping[str, object] | None) -> int:
        """Total for usage that carries no input/output split, else 0.

        A source that can only report one number for the whole request (a
        pass-through target pricing its own multi-model call) charges that
        number under every ``token_rate_limit_type``. Splitting it is
        impossible, and reading 0 out of it would leave the window
        uncharged, which is how pass-through traffic slips past a TPM limit
        it is supposed to share.
        """
        if usage is None:
            return 0
        token_counts: Final = (
            (usage.prompt_tokens or 0, usage.completion_tokens or 0, usage.total_tokens or 0)
            if isinstance(usage, Usage)
            else (usage.input_tokens or 0, usage.output_tokens or 0, usage.total_tokens or 0)
            if isinstance(usage, ResponseAPIUsage)
            else (
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
                usage.get("completion_tokens") or usage.get("output_tokens") or 0,
                usage.get("total_tokens") or 0,
            )
        )
        prompt_tokens, completion_tokens, total_tokens = token_counts
        if prompt_tokens or completion_tokens or not isinstance(total_tokens, int):
            return 0
        return total_tokens

    @staticmethod
    def _response_usage(
        response_obj: object,
    ) -> Usage | ResponseAPIUsage | Mapping[str, object] | None:
        if isinstance(response_obj, (Usage, ResponseAPIUsage)):
            return response_obj
        if isinstance(response_obj, ExtractionResponse):
            return get_usage_object(response_obj)
        if isinstance(
            response_obj,
            (ModelResponse, EmbeddingResponse, TextCompletionResponse, BaseLiteLLMOpenAIResponseObject),
        ):
            usage: Final = getattr(response_obj, "usage", None)
            return usage if isinstance(usage, (Usage, ResponseAPIUsage, dict)) else None
        if isinstance(response_obj, dict):
            nested_usage: Final = response_obj.get("usage")
            if isinstance(nested_usage, (Usage, ResponseAPIUsage, dict)):
                return nested_usage
            return response_obj
        return None

    async def _execute_token_increment_script(self, pipeline_operations: Sequence[CounterIncrement]) -> None:
        """
        Execute token increment script grouped by hash tag for cluster compatibility.
        """
        for group in self._slot_groups(tuple(operation.counter for operation in pipeline_operations)):
            group_operations = tuple(pipeline_operations[index] for index in group)
            for operation in group_operations:
                verbose_proxy_logger.debug(
                    "Executing TTL-preserving increment for counter=%s, increment=%s, ttl=%s",
                    operation.counter,
                    operation.amount,
                    operation.ttl or 0,
                )
            if self.rate_limits.token_increment_post_call(group_operations):
                continue
            await self.rate_limits.token_increment(group_operations, batch=Batch.NONE)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_increment_tokens_with_ttl_preservation(
        self,
        pipeline_operations: Sequence[CounterIncrement],
        parent_otel_span: Span | None = None,
    ) -> None:
        """
        Increment token counters using Lua script to preserve existing TTL.
        This prevents TTL reset on every token increment. Inside a request scope the script rides the
        request's post-call pipeline, and a failed script falls back to the plain increment pipeline for
        its own group.
        """
        if not pipeline_operations:
            return

        if not self.rate_limits.has_redis:
            verbose_proxy_logger.debug("TTL preservation script not available, using regular pipeline")
            await self.rate_limits.increment_many(pipeline_operations)
            return

        try:
            await self._execute_token_increment_script(pipeline_operations)

            verbose_proxy_logger.debug(
                "Successfully executed TTL-preserving increment for %s keys", len(pipeline_operations)
            )

        except Exception as e:
            log_redis_failure(
                verbose_proxy_logger, logging.WARNING, "TTL preservation failed, falling back to regular pipeline", e
            )
            await self.rate_limits.increment_many(pipeline_operations)

    async def _apply_local_window_guarded_token_increments(self, operations: Sequence[WindowGuardedIncrement]) -> None:
        async with self._check_and_increment_lock:
            for operation in operations:
                counter = operation.increment.counter
                active_window_start = _scalar(self.rate_limits.local((counter.window,))[0])
                if active_window_start is None or str(active_window_start) != operation.expected_window_start:
                    continue
                current_counter = _scalar(self.rate_limits.local((counter,))[0]) or 0
                self.rate_limits.set_local(
                    counter, float(current_counter) + operation.increment.amount, operation.increment.ttl
                )

    async def _apply_redis_window_guarded_token_increments(self, operations: Sequence[WindowGuardedIncrement]) -> None:
        """Each guarded increment runs now, one after another, and falls back to the plain increment when its
        script fails, as main sends them."""
        for operation in operations:
            increment = operation.increment
            if self.rate_limits.has_redis:
                try:
                    await self.rate_limits.guarded_increment(
                        increment.counter,
                        operation.expected_window_start,
                        increment.amount,
                        increment.ttl,
                        batch=Batch.NONE,
                    )
                    continue
                except Exception as e:  # noqa: BLE001  # Redis failures use the plain increment fallback
                    log_redis_failure(
                        verbose_proxy_logger,
                        logging.WARNING,
                        f"Window-guarded token adjustment failed for {increment.counter}",
                        e,
                    )
            if increment.amount > 0:
                await self.rate_limits.increment(increment)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_increment_reservation_aware_tokens(
        self,
        pipeline_operations: Sequence[ReservationOperation],
        parent_otel_span: Span | None = None,
    ) -> None:
        for operation in pipeline_operations:
            if isinstance(operation, CounterIncrement):
                await self.rate_limits.increment(operation)
        guarded: Final = tuple(
            operation for operation in pipeline_operations if isinstance(operation, WindowGuardedIncrement)
        )
        local_guarded_operations: Final = tuple(
            operation for operation in guarded if operation.reservation_backend == "local"
        )
        redis_guarded_operations: Final = tuple(
            operation for operation in guarded if operation.reservation_backend != "local"
        )
        if local_guarded_operations:
            await self._apply_local_window_guarded_token_increments(local_guarded_operations)
        if redis_guarded_operations:
            await self._apply_redis_window_guarded_token_increments(redis_guarded_operations)

    def get_rate_limit_type(self) -> Literal["output", "input", "total"]:
        from litellm.proxy.proxy_server import general_settings

        specified_rate_limit_type: Final = general_settings.get("token_rate_limit_type", "total")
        if specified_rate_limit_type not in [
            "output",
            "input",
            "total",
        ]:
            return "total"  # default to total
        return specified_rate_limit_type

    @staticmethod
    def _merge_ratelimit_statuses_into_additional_headers(
        additional_headers: dict[str, object],
        statuses: list[RateLimitStatus],
    ) -> dict[str, object]:
        """
        Return ``additional_headers`` extended with
        ``x-ratelimit-{descriptor_key}-{remaining|limit}-{rate_limit_type}``
        entries. Non-mutating so callers pick their own target dict.
        """
        merged: Final[dict[str, object]] = dict(additional_headers)
        for status in statuses:
            prefix = f"x-ratelimit-{status['descriptor_key']}"
            merged[f"{prefix}-remaining-{status['rate_limit_type']}"] = status["limit_remaining"]
            merged[f"{prefix}-limit-{status['rate_limit_type']}"] = status["current_limit"]
        return merged

    @staticmethod
    def _resolve_rerank_token_usage(response_obj: object) -> tuple[int, int, bool] | None:
        if not isinstance(response_obj, RerankResponse) or response_obj.meta is None:
            return None

        rerank_tokens: Final = response_obj.meta.get("tokens")  # pyright: ignore[reportUnknownMemberType]  # TypedDict's optional generic metadata widens get overloads
        if rerank_tokens is not None:
            input_tokens: Final = rerank_tokens.get("input_tokens") or 0  # pyright: ignore[reportUnknownMemberType]  # token fields are typed integers despite the generic get overload
            output_tokens: Final = rerank_tokens.get("output_tokens") or 0  # pyright: ignore[reportUnknownMemberType]  # token fields are typed integers despite the generic get overload
            if input_tokens or output_tokens:
                return max(0, input_tokens), max(0, output_tokens), True

        billed_units: Final = response_obj.meta.get("billed_units")  # pyright: ignore[reportUnknownMemberType]  # TypedDict's optional generic metadata widens get overloads
        if billed_units is not None:
            total_tokens: Final = billed_units.get("total_tokens") or 0  # pyright: ignore[reportUnknownMemberType]  # billed total is a typed integer despite the generic get overload
            if total_tokens:
                return max(0, total_tokens), 0, True
        return None

    def _resolve_reconciled_usage(self, response_obj: object) -> _ReconciledUsage | None:
        """The prompt, completion, and cache-read token counts a response reports, else None
        when it reports no usage at all. Cache-read tokens stay inside ``prompt_tokens`` here;
        each consumer decides what they cost it."""
        rerank_usage: Final = self._resolve_rerank_token_usage(response_obj)
        if rerank_usage is not None:
            return _ReconciledUsage(prompt_tokens=rerank_usage[0], completion_tokens=rerank_usage[1], cached_tokens=0)

        usage: Final = self._response_usage(response_obj)

        if isinstance(usage, Usage):
            prompt_tokens: Final = usage.prompt_tokens or 0
            completion_tokens: Final = usage.completion_tokens or 0
            cached_tokens: Final = (
                getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                if usage.prompt_tokens_details is not None
                else 0
            )
            if prompt_tokens == 0 and completion_tokens == 0:
                return None
            return _ReconciledUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cached_tokens=cached_tokens
            )

        if isinstance(usage, ResponseAPIUsage):
            response_input_tokens: Final = usage.input_tokens or 0
            response_output_tokens: Final = usage.output_tokens or 0
            response_cached_tokens: Final = (
                usage.input_tokens_details.cached_tokens or 0 if usage.input_tokens_details is not None else 0
            )
            if response_input_tokens == 0 and response_output_tokens == 0:
                return None
            return _ReconciledUsage(
                prompt_tokens=response_input_tokens,
                completion_tokens=response_output_tokens,
                cached_tokens=response_cached_tokens,
            )

        if isinstance(usage, Mapping):
            raw_prompt_tokens: Final = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            raw_completion_tokens: Final = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            mapped_prompt_tokens: Final = raw_prompt_tokens if isinstance(raw_prompt_tokens, int) else 0
            mapped_completion_tokens: Final = raw_completion_tokens if isinstance(raw_completion_tokens, int) else 0
            prompt_details: Final = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            raw_cached_tokens: Final = (
                (prompt_details.get("cached_tokens", 0) if isinstance(prompt_details, dict) else 0)
                or usage.get("cache_read_input_tokens")
                or 0
            )
            mapped_cached_tokens: Final = raw_cached_tokens if isinstance(raw_cached_tokens, int) else 0
            if mapped_prompt_tokens == 0 and mapped_completion_tokens == 0:
                return None
            return _ReconciledUsage(
                prompt_tokens=mapped_prompt_tokens,
                completion_tokens=mapped_completion_tokens,
                cached_tokens=mapped_cached_tokens,
            )

        return None

    def _resolve_io_token_reconcile_usage(
        self,
        response_obj: object,
    ) -> tuple[int, int, bool]:
        """
        Resolve ``(billable_input_tokens, completion_tokens, usage_resolved)``
        for ITPM/OTPM reconciliation. Cache-read tokens are excluded from
        billable input -- Bedrock Mantle doesn't count them toward ITPM --
        but they're untouched everywhere else (cost/usage logging still sees
        the full prompt token count).
        """
        usage: Final = self._resolve_reconciled_usage(response_obj)
        if usage is None:
            return 0, 0, False
        return usage.billable_input_tokens, usage.completion_tokens, True

    def _build_io_token_reservation_ops(
        self,
        kwargs: object,
        response_obj: object,
    ) -> list[CounterIncrement] | tuple[ReservationOperation, ...]:
        """
        Reconcile project ITPM/OTPM reservations to actual usage on success:
        ITPM to billable input tokens, OTPM to actual completion tokens.
        Reuses ``_build_reservation_aware_tpm_ops``'s delta pattern -- ITPM/OTPM
        are stored in the same ":tokens" cache bucket as combined TPM, just
        under distinct scope keys, so the reservation-aware increment math is
        identical; only the usage fields being reconciled against differ.
        """
        if not isinstance(kwargs, dict):
            return ()
        stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
        if stash is None:
            return ()

        itpm_reserved: Final = stash.itpm_reserved_tokens
        otpm_reserved: Final = stash.otpm_reserved_tokens
        if itpm_reserved <= 0 and otpm_reserved <= 0:
            return ()

        response_usage: Final = self._resolve_io_token_reconcile_usage(response_obj)
        combined_usage: Final = self._resolve_io_token_reconcile_usage(kwargs.get("combined_usage_object"))
        aggregate_total: Final = self._aggregate_only_total_tokens(
            self._response_usage(response_obj)
        ) or self._aggregate_only_total_tokens(self._response_usage(kwargs.get("combined_usage_object")))

        if (
            not response_usage[2]
            and not combined_usage[2]
            and aggregate_total <= 0
            and not stash.io_reservation_released
        ):
            return ()
        resolved_usage: Final = (
            response_usage
            if response_usage[2]
            else combined_usage
            if combined_usage[2]
            else (aggregate_total, aggregate_total, True)
            if aggregate_total > 0
            else (itpm_reserved, otpm_reserved, False)
        )
        billable_input, completion_tokens, _ = resolved_usage

        if stash.io_reservation_released or (
            not stash.itpm_reserved_window_identities and not stash.otpm_reserved_window_identities
        ):
            return self._build_reservation_aware_tpm_ops(
                targets=tuple(stash.itpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.io_reservation_released else stash.itpm_reserved_scopes,
                actual_tokens=billable_input,
                reserved_tokens=0 if stash.io_reservation_released else itpm_reserved,
            ) + self._build_reservation_aware_tpm_ops(
                targets=tuple(stash.otpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.io_reservation_released else stash.otpm_reserved_scopes,
                actual_tokens=completion_tokens,
                reserved_tokens=0 if stash.io_reservation_released else otpm_reserved,
            )

        itpm_ops: Final[Sequence[ReservationOperation]] = (
            self._build_project_reservation_ops(
                targets=tuple(stash.itpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.io_reservation_released else stash.itpm_reserved_scopes,
                actual_tokens=billable_input,
                reserved_tokens=itpm_reserved,
                reservation_window_identities=stash.itpm_reserved_window_identities,
            )
            if itpm_reserved > 0
            else ()
        )
        otpm_ops: Final[Sequence[ReservationOperation]] = (
            self._build_project_reservation_ops(
                targets=tuple(stash.otpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.io_reservation_released else stash.otpm_reserved_scopes,
                actual_tokens=completion_tokens,
                reserved_tokens=otpm_reserved,
                reservation_window_identities=stash.otpm_reserved_window_identities,
            )
            if otpm_reserved > 0
            else ()
        )
        return tuple((*itpm_ops, *otpm_ops))

    def _collect_tpm_scope_targets(
        self,
        standard_logging_metadata: dict[str, Any],
        kwargs: object,
        model_group: str | None,
        tpm_limited_tags: Set[str] = frozenset(),
        model_groups: tuple[str, ...] = (),
    ) -> list[tuple[str, str]]:
        """
        Enumerate every (scope_key, scope_value) pair that *might* carry a
        TPM counter for this request — independent of whether each scope had
        a configured TPM limit at pre-call. Reservation awareness happens at
        the emitter; this helper just lists the candidate scopes so callers
        can split reserved-vs-unreserved.
        """
        user_api_key: Final = standard_logging_metadata.get("user_api_key_hash")
        user_api_key_user_id: Final = standard_logging_metadata.get("user_api_key_user_id")
        user_api_key_team_id: Final = standard_logging_metadata.get("user_api_key_team_id")
        user_api_key_organization_id: Final = standard_logging_metadata.get("user_api_key_org_id")
        user_api_key_project_id: Final = standard_logging_metadata.get("user_api_key_project_id")
        user_api_key_end_user_id: Final = (
            kwargs.get("user") if isinstance(kwargs, dict) else None
        ) or standard_logging_metadata.get("user_api_key_end_user_id")
        agent_id: Final = standard_logging_metadata.get("agent_id")
        session_id: Final = standard_logging_metadata.get("session_id") or standard_logging_metadata.get("trace_id")

        targets: Final[list[tuple[str, str]]] = []
        if user_api_key:
            targets.append(("api_key", user_api_key))
        if user_api_key_user_id:
            targets.append(("user", user_api_key_user_id))
        if user_api_key_team_id:
            targets.append(("team", user_api_key_team_id))
        if user_api_key_team_id and user_api_key_user_id:
            targets.append(("team_member", f"{user_api_key_team_id}:{user_api_key_user_id}"))
        if user_api_key_end_user_id:
            targets.append(("end_user", user_api_key_end_user_id))
        if user_api_key_organization_id:
            targets.append(("organization", user_api_key_organization_id))
        customer_scope_owners: Final = (
            ("customer_org", standard_logging_metadata.get("user_api_key_customer_org_id")),
            ("customer_team", standard_logging_metadata.get("user_api_key_customer_team_id")),
            ("customer_member", standard_logging_metadata.get("user_api_key_customer_member_id")),
        )
        targets.extend((scope, owner) for scope, owner in customer_scope_owners if owner)
        model_scope_owners: Final = (
            ("model_per_key", user_api_key),
            ("model_per_team", user_api_key_team_id),
            ("model_per_end_user", f"{user_api_key_end_user_id}" if user_api_key_end_user_id else None),
            ("model_per_organization", user_api_key_organization_id),
            ("model_per_project", user_api_key_project_id),
            *((f"model_per_{scope}", owner) for scope, owner in customer_scope_owners),
        )
        if model_group:
            for group in dict.fromkeys(model_groups or (model_group,)):
                targets.extend((scope, f"{owner}:{group}") for scope, owner in model_scope_owners if owner)
        if agent_id:
            targets.append(("agent", agent_id))
            if session_id:
                targets.append(("agent_session", f"{agent_id}:{session_id}"))
        targets.extend(("tag", tag) for tag in sorted(tpm_limited_tags))
        return targets

    def _build_reservation_aware_tpm_ops(
        self,
        targets: Sequence[tuple[str, str]],
        reserved_scopes: Set[tuple[str, str]],
        actual_tokens: int,
        reserved_tokens: int,
    ) -> list[CounterIncrement]:
        """
        Emit per-scope TPM increment ops with reservation awareness.

        - Reserved scope (counter already at +reserved from pre-call):
          reconcile to actual via ``actual - reserved``.
        - Unreserved scope (counter never touched at pre-call):
          charge the full ``actual``.

        Same primitive serves success reconciliation, over-reservation
        release, and failure refund — pass ``actual_tokens=0`` for the pure
        refund case (reserved scopes get -reserved, unreserved get 0/skip).
        """
        return [
            CounterIncrement(
                counter=RateLimitScope(scope_key, scope_value).counter("tokens"),
                amount=increment,
                ttl=self.window_size,
            )
            for scope_key, scope_value in targets
            if (
                increment := (
                    actual_tokens - reserved_tokens if (scope_key, scope_value) in reserved_scopes else actual_tokens
                )
            )
            != 0
        ]

    def _build_project_reservation_op(
        self,
        scope: tuple[str, str],
        reserved_scopes: Set[tuple[str, str]],
        actual_tokens: int,
        reserved_tokens: int,
        reservation_window_identities: frozenset[ReservationWindowIdentity],
    ) -> ReservationOperation | None:
        scope_key, scope_value = scope
        is_reserved_scope: Final = scope in reserved_scopes
        increment: Final = actual_tokens - reserved_tokens if is_reserved_scope else actual_tokens
        if increment == 0:
            return None
        counter: Final = RateLimitScope(scope_key, scope_value).counter("tokens")
        window_identity: Final = next(
            (identity for identity in reservation_window_identities if identity[0] == counter), None
        )
        operation: Final = CounterIncrement(counter=counter, amount=increment, ttl=self.window_size)
        if not is_reserved_scope or window_identity is None:
            return operation
        return WindowGuardedIncrement(
            increment=operation,
            expected_window_start=window_identity[1],
            reservation_backend=window_identity[2],
        )

    def _build_project_reservation_ops(
        self,
        targets: Sequence[tuple[str, str]],
        reserved_scopes: Set[tuple[str, str]],
        actual_tokens: int,
        reserved_tokens: int,
        reservation_window_identities: frozenset[ReservationWindowIdentity],
    ) -> tuple[ReservationOperation, ...]:
        return tuple(
            operation
            for scope in targets
            if (
                operation := self._build_project_reservation_op(
                    scope=scope,
                    reserved_scopes=reserved_scopes,
                    actual_tokens=actual_tokens,
                    reserved_tokens=reserved_tokens,
                    reservation_window_identities=reservation_window_identities,
                )
            )
            is not None
        )

    def _build_success_event_pipeline_operations(
        self,
        kwargs: dict[str, Any],
        response_obj: object,
        rate_limit_type: Literal["output", "input", "total"],
    ) -> list[ReservationOperation]:
        """Build Redis pipeline increment ops for TPM / parallel-request counters."""
        from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs

        # Get metadata from standard_logging_object - this correctly handles both
        # 'metadata' and 'litellm_metadata' fields from litellm_params
        standard_logging_object: Final = kwargs.get("standard_logging_object") or {}
        request_metadata: Final = get_litellm_metadata_from_kwargs(kwargs)
        origin: Final = request_metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY)
        if origin and origin != "autorouter_compaction":
            # Background evaluations keep their exemption; foreground compaction
            # is necessary caller traffic and consumes the caller's token limits.
            return []
        standard_logging_metadata: Final = standard_logging_object.get("metadata") or {}

        model_group: Final = get_model_group_from_litellm_kwargs(kwargs)

        # Get total tokens from response. Responses LiteLLM does not model
        # (e.g. pass-through, whose usage is reported by the upstream rather
        # than parsed out of the body) carry their usage in
        # ``combined_usage_object`` instead (a Responses WebSocket turn in its
        # raw events) and would otherwise never charge the TPM window.
        _usage: Final[Usage | dict[str, object] | None] = (
            getattr(response_obj, "usage", None)
            if isinstance(
                response_obj,
                (
                    ModelResponse,
                    EmbeddingResponse,
                    TextCompletionResponse,
                    BaseLiteLLMOpenAIResponseObject,
                ),
            )
            else get_usage_object(response_obj)
            if isinstance(response_obj, ExtractionResponse)
            else _unmodeled_response_usage(kwargs, response_obj)
        )
        total_tokens = self._get_total_tokens_from_usage(usage=_usage, rate_limit_type=rate_limit_type)
        if total_tokens == 0:
            total_tokens = self._aggregate_only_total_tokens(usage=_usage)

        stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
        open_reservation: Final = stash if stash is not None and not stash.reservation_released else None
        reserved_tokens: Final = open_reservation.reserved_tokens if open_reservation is not None else 0
        reserved_model: Final = stash.reserved_model if stash is not None else None
        reserved_scopes: Final[frozenset[tuple[str, str]]] = stash.reserved_scopes if stash is not None else frozenset()
        served_model: Final = self._rate_limited_model(model_group)
        # ``reserved_model`` (resolved at admission, so an alias map reload
        # mid-flight cannot move the charge) settles the request unless the
        # router fell back to a group this limiter admitted as a fallback
        # target: then the served group is charged and the abandoned group's
        # per-model reservation is refunded. Client-supplied ``model_group``
        # metadata alone never moves the charge.
        fell_back: Final = (
            stash is not None
            and reserved_model is not None
            and served_model is not None
            and served_model.requested != reserved_model.requested
            and served_model.requested in stash.fallback_models
        )
        reconcile_model: Final = served_model if fell_back or reserved_model is None else reserved_model
        reconcile_models: Final = self._rate_limited_models(reconcile_model)

        pipeline_operations: Final[list[ReservationOperation]] = []

        # ----------------------------------------------------------------
        # TPM reconciliation
        # Per-scope behavior:
        #   reserved scope    -> apply (actual - reserved) delta to settle
        #                        the counter at +actual.
        #   unreserved scope  -> charge the full actual usage (the
        #                        reservation never incremented this scope).
        # When no reservation was made, reserved_tokens=0 and reserved_scopes
        # is empty, so every scope falls through the unreserved branch and
        # gets the full actual charge — matching pre-PR behavior.
        # ----------------------------------------------------------------
        targets: Final = self._collect_tpm_scope_targets(
            standard_logging_metadata=standard_logging_metadata,
            kwargs=kwargs,
            tpm_limited_tags=_tpm_limited_tags(stash, kwargs),
            model_group=reconcile_model.group if reconcile_model is not None else None,
            model_groups=tuple(model.group for model in reconcile_models),
        )
        targets.extend(
            scope
            for scope in sorted(reserved_scopes)
            if scope[0] in ("agent", "agent_session") and scope not in targets
        )
        logging_metadata: Final = _callback_kwargs_mapping(standard_logging_metadata)
        key_owned_team_scopes: Final = frozenset(
            ("model_per_team", f"{logging_metadata.get('user_api_key_team_id')}:{model.group}")
            for model in reconcile_models
            if all(
                self._key_owns_model_tpm_limit_from_request_metadata(request_metadata, candidate)
                for candidate in reconcile_models
                if candidate.group == model.group
            )
        )
        charged_targets: Final = [target for target in targets if target not in key_owned_team_scopes]
        abandoned_scopes: Final = (
            tuple(
                sorted(
                    scope
                    for scope in reserved_scopes
                    if scope[0] in PER_MODEL_RATE_LIMIT_DESCRIPTOR_KEYS and scope not in charged_targets
                )
            )
            if fell_back
            else ()
        )
        if reserved_tokens > 0 and total_tokens < reserved_tokens:
            verbose_proxy_logger.debug(
                "Releasing unused TPM budget on success: reserved=%s, actual=%s, release=%s",
                reserved_tokens,
                total_tokens,
                reserved_tokens - total_tokens,
            )
        pipeline_operations.extend(
            self._build_reservation_aware_tpm_ops(
                targets=charged_targets,
                reserved_scopes=reserved_scopes,
                actual_tokens=total_tokens,
                reserved_tokens=reserved_tokens,
            )
        )
        pipeline_operations.extend(
            self._build_reservation_aware_tpm_ops(
                targets=abandoned_scopes,
                reserved_scopes=reserved_scopes,
                actual_tokens=0,
                reserved_tokens=reserved_tokens,
            )
        )
        call_type: Final[object] = kwargs.get("call_type")
        ptu_skips_socket_usage: Final = call_type == CallTypes.aresponses_websocket.value
        pipeline_operations.extend(
            self._build_team_ptu_tpm_ops(
                standard_logging_metadata,  # pyright: ignore[reportUnknownArgumentType]  # untyped logging metadata
                usage_source=response_obj if _usage is None or ptu_skips_socket_usage else _usage,
                reconcile_model=reconcile_model,
                reserved_scopes=reserved_scopes,
                reserved_ceiling=stash.ptu_ceiling if stash is not None else None,
                reserved_tokens=open_reservation.ptu_reserved_tokens if open_reservation is not None else 0,
                reserved_windows=stash.ptu_reserved_window_identities if stash is not None else frozenset(),
                total_tokens=0 if ptu_skips_socket_usage else total_tokens,
            )
        )

        return pipeline_operations

    def _build_team_ptu_tpm_ops(
        self,
        standard_logging_metadata: Mapping[str, object],
        usage_source: object,
        reconcile_model: RateLimitedModel | None,
        reserved_scopes: Set[tuple[str, str]],
        reserved_ceiling: PTUTeamCeiling | None,
        reserved_tokens: int,
        reserved_windows: frozenset[ReservationWindowIdentity],
        total_tokens: int,
    ) -> Sequence[ReservationOperation]:
        """Settle the team's PTU counter in Azure normalized tokens: uncached input in full,
        cached input at the model's cached ratio, output weighted by its output-to-input
        ratio, the way Azure sizes a PTU.

        The reservation was taken in those units against the ceiling admission resolved, so that
        ceiling settles it even after the share changed or went away or the logging metadata lost
        the team; when usage cannot be resolved it charges the raw total the other scopes charge
        so the reservation is never left standing.
        """
        reserved_ptu_scopes: Final = tuple(scope for scope in reserved_scopes if scope[0] == PTU_TEAM_DESCRIPTOR_KEY)
        if reserved_ptu_scopes:
            return self._build_project_reservation_ops(
                targets=reserved_ptu_scopes,
                reserved_scopes=reserved_scopes,
                actual_tokens=self._ptu_settlement_tokens(
                    reserved_ceiling, self._resolve_reconciled_usage(usage_source), total_tokens
                ),
                reserved_tokens=reserved_tokens,
                reservation_window_identities=reserved_windows,
            )
        team_id: Final = standard_logging_metadata.get("user_api_key_team_id")
        if reconcile_model is None or not isinstance(team_id, str) or not team_id:
            return ()
        ceiling: Final = (
            reserved_ceiling
            if reserved_ceiling is not None
            else self._ptu_team_ceiling_resolver(team_id, reconcile_model.group)
        )
        if ceiling is None:
            return ()
        return self._build_reservation_aware_tpm_ops(
            targets=((PTU_TEAM_DESCRIPTOR_KEY, f"{team_id}:{ceiling.model_group}"),),
            reserved_scopes=reserved_scopes,
            actual_tokens=self._ptu_settlement_tokens(
                ceiling, self._resolve_reconciled_usage(usage_source), total_tokens
            ),
            reserved_tokens=reserved_tokens,
        )

    @staticmethod
    def _ptu_settlement_tokens(ceiling: PTUTeamCeiling | None, usage: _ReconciledUsage | None, raw_total: int) -> int:
        if ceiling is None or usage is None:
            return raw_total
        return round(
            normalized_tokens(
                ceiling,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cache_read_tokens=usage.cached_tokens,
            )
        )

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        Update TPM usage on successful API calls by incrementing counters using pipeline
        """
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )

        rate_limit_type: Final = self.get_rate_limit_type()

        litellm_parent_otel_span: Final[Span | None] = get_parent_otel_span_from_kwargs(kwargs)
        try:
            verbose_proxy_logger.debug("INSIDE parallel request limiter ASYNC SUCCESS LOGGING")

            callback_kwargs: Final = _callback_kwargs_mapping(kwargs)
            stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(callback_kwargs))
            await self._release_stashed_parallel_slot(
                stash,
                litellm_parent_otel_span,
                in_logging_callback=True,
                round_ended_at=end_time
                if isinstance(end_time, datetime) and not callback_kwargs.get("stream")
                else None,
            )

            usage_operations: Final = self._build_success_event_pipeline_operations(
                kwargs=kwargs,
                response_obj=response_obj,
                rate_limit_type=rate_limit_type,
            )
            io_token_operations: Final = self._build_io_token_reservation_ops(
                kwargs=kwargs,
                response_obj=response_obj,
            )
            if stash is not None:
                stash.clear_reserved_tokens()
            await self._apply_usage_operations(usage_operations, litellm_parent_otel_span)
            if io_token_operations:
                if isinstance(io_token_operations, list):
                    await self.async_increment_tokens_with_ttl_preservation(
                        pipeline_operations=io_token_operations,
                        parent_otel_span=litellm_parent_otel_span,
                    )
                else:
                    await self.async_increment_reservation_aware_tokens(
                        pipeline_operations=io_token_operations,
                        parent_otel_span=litellm_parent_otel_span,
                    )

        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit success event: %s", e)

    async def _apply_usage_operations(
        self, pipeline_operations: Sequence[ReservationOperation], parent_otel_span: Span | None
    ) -> None:
        unguarded_operations: Final = [
            operation for operation in pipeline_operations if isinstance(operation, CounterIncrement)
        ]
        window_guarded_operations: Final = tuple(
            operation for operation in pipeline_operations if isinstance(operation, WindowGuardedIncrement)
        )
        if unguarded_operations:
            await self.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=unguarded_operations,
                parent_otel_span=parent_otel_span,
            )
        if window_guarded_operations:
            await self.async_increment_reservation_aware_tokens(
                pipeline_operations=window_guarded_operations,
                parent_otel_span=parent_otel_span,
            )

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_logging_hook(
        self,
        kwargs: dict,
        result: object,
        call_type: str,
    ) -> tuple[dict, object]:
        """
        Mirror the pre-call rate-limit snapshot into the SLP so streaming
        success callbacks see the same ``x-ratelimit-*`` headers the
        non-streaming path writes via ``async_post_call_success_hook``.
        Runs in the earlier of the two callback loops inside
        ``async_success_handler`` so downstream callbacks see the values
        regardless of registration order. Idempotent for non-streaming.
        """
        self._mirror_ratelimit_response_into_logging_payload(
            kwargs=kwargs,
            response_obj=result,
        )
        return kwargs, result

    def _mirror_ratelimit_response_into_logging_payload(
        self,
        kwargs: object,
        response_obj: object,
    ) -> None:
        """
        Copy the stashed ``RateLimitResponse`` into the SLP's
        ``hidden_params.additional_headers`` and the response object's
        ``_hidden_params.additional_headers`` (when the latter is a dict).
        """
        if not isinstance(kwargs, dict):
            return

        stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
        rate_limit_response: Final = stash.rate_limit_response if stash is not None else None
        statuses: Final = rate_limit_response["statuses"] if rate_limit_response is not None else []
        if not statuses:
            return

        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            hidden_params = standard_logging_object.get("hidden_params")
            if not isinstance(hidden_params, dict):
                hidden_params = {}
            existing = hidden_params.get("additional_headers")
            hidden_params["additional_headers"] = self._merge_ratelimit_statuses_into_additional_headers(
                additional_headers=existing if isinstance(existing, dict) else {},
                statuses=statuses,
            )
            standard_logging_object["hidden_params"] = hidden_params

        response_hidden: Final = getattr(response_obj, "_hidden_params", None)
        if isinstance(response_hidden, dict):
            existing = response_hidden.get("additional_headers")
            response_hidden["additional_headers"] = self._merge_ratelimit_statuses_into_additional_headers(
                additional_headers=existing if isinstance(existing, dict) else {},
                statuses=statuses,
            )

    def _recovered_partial_usage(self, source: Mapping[str, object]) -> Usage | None:
        usage: Final = source.get("combined_usage_object")
        if not isinstance(usage, Usage) or (usage.completion_tokens or 0) <= 0:
            return None
        return usage

    def _recovered_partial_usage_tokens(self, source: Mapping[str, object]) -> tuple[int, int, int]:
        usage: Final = self._recovered_partial_usage(source)
        if usage is None:
            return 0, 0, 0
        billable_input, completion_tokens, _ = self._resolve_io_token_reconcile_usage(usage)
        return (
            self._get_total_tokens_from_usage(usage=usage, rate_limit_type=self.get_rate_limit_type()),
            billable_input,
            completion_tokens,
        )

    def _build_ptu_failure_settlement_ops(
        self, stash: RequestRateLimiterStash, source: Mapping[str, object], raw_actual_tokens: int
    ) -> tuple[ReservationOperation, ...]:
        """Settle the team PTU reservation on failure in the normalized tokens it was taken in:
        at the recovered partial usage when there is one, else a full refund."""
        ptu_scopes: Final = tuple(scope for scope in stash.reserved_scopes if scope[0] == PTU_TEAM_DESCRIPTOR_KEY)
        if not ptu_scopes:
            return ()
        usage: Final = self._recovered_partial_usage(source)
        return self._build_project_reservation_ops(
            targets=ptu_scopes,
            reserved_scopes=stash.reserved_scopes,
            actual_tokens=self._ptu_settlement_tokens(
                stash.ptu_ceiling,
                self._resolve_reconciled_usage(usage) if usage is not None else None,
                raw_actual_tokens,
            ),
            reserved_tokens=stash.ptu_reserved_tokens,
            reservation_window_identities=stash.ptu_reserved_window_identities,
        )

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        On failure: decrement max_parallel_requests and refund the upfront
        TPM reservation only against the scopes the reservation actually
        charged. Unreserved scopes were never incremented at pre-call, so
        refunding them would drive their counter negative. A failed stream
        whose partial usage was recovered settles the reservation at that
        usage instead of refunding it.
        """
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )

        try:
            failed_call: Final = _callback_kwargs_mapping(kwargs)
            litellm_parent_otel_span: Final[Span | None] = get_parent_otel_span_from_kwargs(failed_call)

            pipeline_operations: Final[list[CounterIncrement]] = []

            stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(failed_call))
            await self._release_stashed_parallel_slot(stash, litellm_parent_otel_span, in_logging_callback=True)

            # Skip the reservation refund if async_post_call_failure_hook
            # already released it (proxy-level rejection that also bubbles up
            # here as an LLM-error callback). max_parallel_requests is its
            # own counter and is always decremented per call.
            reserved_tokens: Final = 0 if stash is None or stash.reservation_released else stash.reserved_tokens
            itpm_reserved, otpm_reserved = (
                (0, 0)
                if stash is None or stash.io_reservation_released
                else (stash.itpm_reserved_tokens, stash.otpm_reserved_tokens)
            )
            tpm_actual, itpm_actual, otpm_actual = self._recovered_partial_usage_tokens(failed_call)

            if stash is not None and reserved_tokens > 0:
                verbose_proxy_logger.debug(
                    "Settling reserved TPM tokens on failure: reserved=%s actual=%s", reserved_tokens, tpm_actual
                )
                # Settle only against the scopes the reservation actually
                # charged: unreserved scopes were never incremented, so a
                # refund there would drive their counter negative.
                pipeline_operations.extend(
                    self._build_reservation_aware_tpm_ops(
                        targets=tuple(scope for scope in stash.reserved_scopes if scope[0] != PTU_TEAM_DESCRIPTOR_KEY),
                        reserved_scopes=stash.reserved_scopes,
                        actual_tokens=tpm_actual,
                        reserved_tokens=reserved_tokens,
                    )
                )

            # Settle project ITPM/OTPM reservations the same way: at the
            # recovered partial usage, or a full refund when there is none.
            itpm_operations: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=itpm_actual,
                    reserved_tokens=itpm_reserved,
                    reservation_window_identities=stash.itpm_reserved_window_identities,
                )
                if stash is not None and itpm_reserved > 0 and stash.itpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=itpm_actual,
                    reserved_tokens=itpm_reserved,
                )
                if stash is not None and itpm_reserved > 0
                else ()
            )

            otpm_operations: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=otpm_actual,
                    reserved_tokens=otpm_reserved,
                    reservation_window_identities=stash.otpm_reserved_window_identities,
                )
                if stash is not None and otpm_reserved > 0 and stash.otpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=otpm_actual,
                    reserved_tokens=otpm_reserved,
                )
                if stash is not None and otpm_reserved > 0
                else ()
            )

            ptu_operations: Final = (
                self._build_ptu_failure_settlement_ops(
                    stash,
                    kwargs,  # pyright: ignore[reportUnknownArgumentType]  # hook kwargs are unannotated
                    tpm_actual,
                )
                if stash is not None and reserved_tokens > 0
                else ()
            )

            if stash is not None and (reserved_tokens > 0 or itpm_reserved > 0 or otpm_reserved > 0):
                stash.mark_reservation_released()
            if pipeline_operations:
                await self.rate_limits.increment_many_post_call(pipeline_operations)
            for project_operations in (itpm_operations, otpm_operations, ptu_operations):
                if isinstance(project_operations, list):
                    await self.rate_limits.increment_many_post_call(project_operations)
                elif project_operations:
                    await self.async_increment_reservation_aware_tokens(
                        pipeline_operations=project_operations,
                        parent_otel_span=litellm_parent_otel_span,
                    )
            if stash is None and tpm_actual > 0 and _is_socket_turn(failed_call):
                await self._apply_usage_operations(
                    self._build_success_event_pipeline_operations(
                        kwargs=dict(failed_call), response_obj=None, rate_limit_type=self.get_rate_limit_type()
                    ),
                    litellm_parent_otel_span,
                )
        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit failure event: %s", e)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_release_max_parallel_requests_on_disconnect(
        self,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """
        Release the api-key ``max_parallel_requests`` slot that
        ``async_pre_call_hook`` acquired, for a request that ended without
        either logging callback firing.

        The slot is normally released by ``async_log_success_event`` (natural
        stream completion) or ``async_log_failure_event`` (LLM error). When a
        client cancels a stream mid-flight, the cancellation surfaces as
        ``asyncio.CancelledError`` / ``GeneratorExit`` and neither callback
        runs, so without this the slot leaks per cancelled stream until its
        TTL prunes it. The stashed acquisition's presence (not the key
        object's current max_parallel_requests configuration, which can
        change mid-request) decides whether there is anything to release.
        """
        await self._release_stashed_parallel_slot(get_request_stash(), None)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response) -> None:
        """
        Release completed-request slots and update rate limit headers in the response.
        """
        try:
            slot_stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(data))
            await self._release_stashed_parallel_slot(slot_stash, user_api_key_dict.parent_otel_span)
        except Exception as e:
            verbose_proxy_logger.exception("Error releasing parallel request slot in post-call hook: %s", e)

        try:
            header_stash: Final = get_request_stash()
            litellm_proxy_rate_limit_response: Final = (
                header_stash.rate_limit_response if header_stash is not None else None
            )

            if litellm_proxy_rate_limit_response is not None and response_has_hidden_params(response):
                additional_headers: Final = ensure_response_additional_headers(response)
                additional_headers.update(
                    self._merge_ratelimit_statuses_into_additional_headers(
                        additional_headers={},
                        statuses=litellm_proxy_rate_limit_response["statuses"],
                    )
                )

        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit post-call hook: %s", e)

        try:
            await self._handle_batch_enqueued_post_call(user_api_key_dict=user_api_key_dict, response=response)
        except Exception as e:  # noqa: BLE001  # post-call batch accounting must never fail the response
            verbose_proxy_logger.exception("Error in batch enqueued-token post-call hook: %s", e)

    async def _handle_batch_enqueued_post_call(self, user_api_key_dict: UserAPIKeyAuth, response: object) -> None:
        view: Final = batch_response_view(response)
        if view is None:
            return
        span: Final = user_api_key_dict.parent_otel_span
        stash: Final = get_request_stash()
        if stash is not None and stash.batch_enqueued_reservation is not None:
            await self.batch_enqueued_token_store.save_reservation(
                batch_id=canonical_provider_batch_id(view.id),
                reservation=stash.batch_enqueued_reservation,
                litellm_parent_otel_span=span,
            )
            stash.batch_enqueued_reservation = None
        if view.status.lower() in BATCH_ENQUEUED_REFUND_STATUSES:
            popped: Final = await self.batch_enqueued_token_store.pop_reservation(
                batch_id=canonical_provider_batch_id(view.id),
                litellm_parent_otel_span=span,
            )
            if popped is not None:
                await self.batch_enqueued_token_store.refund(reservation=popped, litellm_parent_otel_span=span)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """
        Release the parallel-request slot and any TPM/ITPM/OTPM reservation
        when the request is rejected after the pre-call hook acquired them
        but before the LLM call ran (e.g. a downstream guardrail/auth hook
        raised). Without this, those resources are stranded —
        async_log_failure_event is a litellm completion-level callback and
        never fires for proxy-side rejections, so a leaked slot would occupy
        the gauge for the full PARALLEL_REQUEST_SLOT_TTL_SECONDS.

        Idempotent: the slot release clears the stashed acquisition (and slot
        removal is a no-op ZREM on a second run), and the TPM/ITPM/OTPM
        refund is guarded by the stash's two released flags — if
        both this hook and async_log_failure_event end up running in the same
        flow, only the first release/refund applies. A mid-stream failure
        relayed here with recovered partial usage settles the reservation at
        that usage instead of refunding it.
        """
        try:
            stash: Final = get_request_stash()
            if stash is None:
                return
            await self._release_stashed_parallel_slot(stash, user_api_key_dict.parent_otel_span)

            if stash.batch_enqueued_reservation is not None:
                await self.batch_enqueued_token_store.refund(
                    reservation=stash.batch_enqueued_reservation,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                stash.batch_enqueued_reservation = None

            if stash.batch_tpd_refund_ops:
                await self.async_increment_reservation_aware_tokens(
                    pipeline_operations=stash.batch_tpd_refund_ops,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                stash.batch_tpd_refund_ops = ()

            reserved_tokens: Final = 0 if stash.reservation_released else stash.reserved_tokens
            itpm_reserved: Final = 0 if stash.io_reservation_released else stash.itpm_reserved_tokens
            otpm_reserved: Final = 0 if stash.io_reservation_released else stash.otpm_reserved_tokens
            if reserved_tokens <= 0 and itpm_reserved <= 0 and otpm_reserved <= 0:
                return
            tpm_actual, itpm_actual, otpm_actual = self._recovered_partial_usage_tokens(request_data)

            combined_ops: Final = (
                self._build_reservation_aware_tpm_ops(
                    targets=tuple(scope for scope in stash.reserved_scopes if scope[0] != PTU_TEAM_DESCRIPTOR_KEY),
                    reserved_scopes=stash.reserved_scopes,
                    actual_tokens=tpm_actual,
                    reserved_tokens=reserved_tokens,
                )
                if reserved_tokens > 0
                else ()
            )
            ptu_ops: Final = (
                self._build_ptu_failure_settlement_ops(
                    stash,
                    request_data,  # pyright: ignore[reportUnknownArgumentType]  # request_data is a bare dict
                    tpm_actual,
                )
                if reserved_tokens > 0
                else ()
            )
            itpm_ops: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=itpm_actual,
                    reserved_tokens=itpm_reserved,
                    reservation_window_identities=stash.itpm_reserved_window_identities,
                )
                if itpm_reserved > 0 and stash.itpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=itpm_actual,
                    reserved_tokens=itpm_reserved,
                )
                if itpm_reserved > 0
                else ()
            )
            otpm_ops: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=otpm_actual,
                    reserved_tokens=otpm_reserved,
                    reservation_window_identities=stash.otpm_reserved_window_identities,
                )
                if otpm_reserved > 0 and stash.otpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=otpm_actual,
                    reserved_tokens=otpm_reserved,
                )
                if otpm_reserved > 0
                else ()
            )
            if combined_ops or ptu_ops or itpm_ops or otpm_ops:
                verbose_proxy_logger.debug(
                    "Releasing reserved tokens on proxy-level rejection: tpm=%s, itpm=%s, otpm=%s",
                    reserved_tokens,
                    itpm_reserved,
                    otpm_reserved,
                )
            stash.mark_reservation_released()
            if combined_ops:
                await self.rate_limits.increment_many(combined_ops)
            for project_ops in (itpm_ops, otpm_ops, ptu_ops):
                if isinstance(project_ops, list):
                    await self.rate_limits.increment_many(project_ops)
                elif project_ops:
                    await self.async_increment_reservation_aware_tokens(
                        pipeline_operations=project_ops,
                        parent_otel_span=user_api_key_dict.parent_otel_span,
                    )
        except Exception as e:
            verbose_proxy_logger.exception("Error releasing TPM reservation on post-call failure: %s", e)
        return


PROXY_MaxParallelRequestsHandler_v3 = _PROXY_MaxParallelRequestsHandler_v3
