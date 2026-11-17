"""
This is a rate limiter implementation based on a similar one by Envoy proxy.

This is currently in development and not yet ready for production.
"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
)

from fastapi import HTTPException
from pydantic import Field, TypeAdapter, ValidationError
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_503_SERVICE_UNAVAILABLE
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm import DualCache
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import Batch
from litellm.caching.redis_cache import log_redis_failure
from litellm.constants import (
    DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE,
    MODEL_LIMIT_MAX_BORROW_MULTIPLE_METADATA_KEY,
    NO_MAX_TOKENS_TPM_RESERVATION_FRACTION,
    RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY,
)
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
    split_embedding_input_tokens,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    get_key_own_model_rate_limit,
    get_model_rate_limit_from_metadata,
)
from litellm.proxy.common_utils.callback_utils import get_model_group_from_litellm_kwargs
from litellm.proxy.common_utils.proxy_rate_limit_error import (
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
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.proxy.spend_tracking.input_tokens import (
    inline_pdf_file_count,
    models_counted_for,
    openai_chat_pdf_file_tokens,
)
from litellm.redis.stores.connection import connect
from litellm.redis.stores.rate_limits import (
    RATE_LIMITS_TARGET,
    CounterIncrement,
    CounterKind,
    RateLimitCounter,
    RateLimitScope,
    RateLimitStore,
)
from litellm.responses.litellm_completion_transformation.compaction_items import billable_input_since_newest_compaction
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.proxy.auth.customer_hierarchy import CustomerScopeAuth
from litellm.types.utils import (
    CallTypes,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.router import Router
    from litellm.rust_bridge._native import NativeParallelRequestLimiter, NativeParallelRequestLimiterState
    from litellm.types.agents import AgentResponse

    Span = _Span
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any


_REQUEST_RATE_LIMIT_DATA: Final = TypeAdapter(Mapping[str, object])
_MCP_KEY_RPM_LIMITS: Final[TypeAdapter[dict[str, int] | None]] = TypeAdapter(dict[str, int] | None)


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


MODEL_LIMIT_MAX_BORROW_MULTIPLE: Final = MODEL_LIMIT_MAX_BORROW_MULTIPLE_METADATA_KEY
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


BorrowerKind: TypeAlias = Literal["team", "key", "customer_org"]
BORROWER_KINDS: Final[tuple[BorrowerKind, ...]] = ("team", "key", "customer_org")
_METADATA: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class ModelLimitBorrower:
    """Who holds a per-model limit the dynamic rate limiter enforces at saturation, and that limit."""

    kind: BorrowerKind
    entity_id: str
    alias: str | None
    limits: BorrowingTeamModelLimits


def _max_borrow_multiple(raw: object, setting: str, owner: str) -> float | None:
    if raw is None:
        return None
    try:
        return _MAX_BORROW_MULTIPLE.validate_python(raw)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Ignoring %s=%r on %s: expected a number from 1 to %d, so the value is not used",
            setting,
            raw,
            owner,
            MAX_BORROW_MULTIPLE_CEILING,
        )
        return None


def team_model_limit_max_borrow_multiple(user_api_key_dict: UserAPIKeyAuth) -> float | None:
    return _max_borrow_multiple(
        _METADATA.validate_python(user_api_key_dict.team_metadata or {}).get(MODEL_LIMIT_MAX_BORROW_MULTIPLE),
        MODEL_LIMIT_MAX_BORROW_MULTIPLE,
        f"team {user_api_key_dict.team_id}",
    )


def borrowing_team_model_limits(user_api_key_dict: UserAPIKeyAuth, model: RateLimitedModel) -> BorrowingTeamModelLimits:
    settings: Final = litellm.priority_reservation_settings
    if settings is None or "team" not in settings.model_limit_borrowing_entities or user_api_key_dict.team_id is None:
        return NO_BORROWING
    limits: Final = BorrowingTeamModelLimits(
        tpm=inherited_team_model_limit(user_api_key_dict, model, "model_tpm_limit"),
        rpm=inherited_team_model_limit(user_api_key_dict, model, "model_rpm_limit"),
    )
    if limits == NO_BORROWING or not dynamic_rate_limiter_covers(model.group):
        return NO_BORROWING
    return BorrowingTeamModelLimits(
        tpm=limits.tpm,
        rpm=limits.rpm,
        max_borrow_multiple=team_model_limit_max_borrow_multiple(user_api_key_dict)
        or settings.model_limit_borrowing_config.max_borrow_multiple,
    )


def borrowing_key_model_limits(user_api_key_dict: UserAPIKeyAuth, model: RateLimitedModel) -> BorrowingTeamModelLimits:
    """The key's own per-model limits when keys are listed to borrow idle capacity."""
    settings: Final = litellm.priority_reservation_settings
    if settings is None or "key" not in settings.model_limit_borrowing_entities or user_api_key_dict.api_key is None:
        return NO_BORROWING
    limits: Final = BorrowingTeamModelLimits(
        tpm=model.limit_in(get_key_own_model_rate_limit(user_api_key_dict, "model_tpm_limit")),
        rpm=model.limit_in(get_key_own_model_rate_limit(user_api_key_dict, "model_rpm_limit")),
    )
    if limits == NO_BORROWING or not dynamic_rate_limiter_covers(model.group):
        return NO_BORROWING
    return BorrowingTeamModelLimits(
        tpm=limits.tpm,
        rpm=limits.rpm,
        max_borrow_multiple=_max_borrow_multiple(
            _METADATA.validate_python(
                user_api_key_dict.metadata or {}  # pyright: ignore[reportUnknownMemberType]  # validates the untyped key metadata
            ).get(MODEL_LIMIT_MAX_BORROW_MULTIPLE),
            MODEL_LIMIT_MAX_BORROW_MULTIPLE,
            "a key",
        )
        or settings.model_limit_borrowing_config.max_borrow_multiple,
    )


def _customer_org_scope(user_api_key_dict: UserAPIKeyAuth) -> CustomerScopeAuth | None:
    hierarchy: Final = user_api_key_dict.customer_hierarchy
    if hierarchy is None:
        return None
    return next((scope for scope in hierarchy.scopes if scope.key == "customer_org"), None)


def borrowing_customer_org_model_limits(
    user_api_key_dict: UserAPIKeyAuth, model: RateLimitedModel
) -> BorrowingTeamModelLimits:
    """The customer organization's per-model limits when customer organizations are listed to borrow idle
    capacity. The share is one counter per organization and model, across every key the customer calls."""
    settings: Final = litellm.priority_reservation_settings
    scope: Final = _customer_org_scope(user_api_key_dict)
    if settings is None or "customer_organization" not in settings.model_limit_borrowing_entities or scope is None:
        return NO_BORROWING
    limits: Final = BorrowingTeamModelLimits(
        tpm=model.limit_in(scope.model_tpm_limit), rpm=model.limit_in(scope.model_rpm_limit)
    )
    if limits == NO_BORROWING or not dynamic_rate_limiter_covers(model.group):
        return NO_BORROWING
    return BorrowingTeamModelLimits(
        tpm=limits.tpm, rpm=limits.rpm, max_borrow_multiple=settings.model_limit_borrowing_config.max_borrow_multiple
    )


def model_limit_borrowers(user_api_key_dict: UserAPIKeyAuth, model: RateLimitedModel) -> tuple[ModelLimitBorrower, ...]:
    """Every level of the request whose per-model limit on ``model`` is a share that borrows idle capacity."""
    org_scope: Final = _customer_org_scope(user_api_key_dict)
    candidates: Final = (
        ModelLimitBorrower(
            kind="team",
            entity_id=str(user_api_key_dict.team_id),
            alias=user_api_key_dict.team_alias,
            limits=borrowing_team_model_limits(user_api_key_dict, model),
        ),
        ModelLimitBorrower(
            kind="key",
            entity_id=str(user_api_key_dict.api_key),
            alias=user_api_key_dict.key_alias,
            limits=borrowing_key_model_limits(user_api_key_dict, model),
        ),
        ModelLimitBorrower(
            kind="customer_org",
            entity_id="" if org_scope is None else org_scope.entity_id,
            alias=None if org_scope is None else org_scope.alias,
            limits=borrowing_customer_org_model_limits(user_api_key_dict, model),
        ),
    )
    return tuple(borrower for borrower in candidates if borrower.limits != NO_BORROWING)


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


ReservationWindowIdentity: TypeAlias = tuple[RateLimitCounter, str, Literal["redis", "local"]]


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


def coalesce_rate_limit_descriptors(descriptors: Sequence[RateLimitDescriptor]) -> Sequence[RateLimitDescriptor]:
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


_NATIVE_STATUSES: Final = TypeAdapter(list[RateLimitStatus])
_NativeWindow: TypeAlias = tuple[str, str, Literal["requests", "tokens"], str, Literal["redis", "local"]]
_NATIVE_WINDOWS: Final = TypeAdapter[list[_NativeWindow] | None](list[_NativeWindow] | None)


class _NativeSlot(TypedDict):
    slot_id: ReadOnly[str]
    gauges: ReadOnly[tuple[tuple[str, str], ...]]


class _NativeResponse(TypedDict):
    overall_code: ReadOnly[str]
    statuses: ReadOnly[tuple[RateLimitStatus, ...]]


class _NativeIo(TypedDict):
    tokens: ReadOnly[int]
    scopes: ReadOnly[tuple[tuple[str, str], ...]]
    windows: ReadOnly[tuple[_NativeWindow, ...]]


_NativeOp: TypeAlias = tuple[
    str, str, CounterKind, int | float, int | None, str | None, Literal["redis", "local"] | None
]


def reservation_operation(op: _NativeOp) -> ReservationOperation:
    key, value, kind, amount, ttl, window_start, backend = op
    increment: Final = CounterIncrement(counter=RateLimitScope(key, value).counter(kind), amount=amount, ttl=ttl)
    if window_start is None or backend is None:
        return increment
    return WindowGuardedIncrement(increment=increment, expected_window_start=window_start, reservation_backend=backend)


class _NativeStash(TypedDict):
    rate_limit_response: ReadOnly[_NativeResponse | None]
    parallel_slot: ReadOnly[_NativeSlot | None]
    reserved_tokens: ReadOnly[int]
    reserved_model: ReadOnly[tuple[str, str] | None]
    reserved_scopes: ReadOnly[tuple[tuple[str, str], ...]]
    fallback_models: ReadOnly[tuple[str, ...]]
    reservation_released: ReadOnly[bool]
    io_reservation_released: ReadOnly[bool]
    tpm_limited_tags: ReadOnly[tuple[str, ...]]
    ptu_reserved_tokens: ReadOnly[int]
    ptu_windows: ReadOnly[tuple[_NativeWindow, ...]]
    itpm: ReadOnly[_NativeIo]
    otpm: ReadOnly[_NativeIo]
    batch_enqueued_reservation: ReadOnly[BatchEnqueuedTokenReservation | None]
    batch_tpd_refunds: ReadOnly[tuple[_NativeOp, ...]]


_NATIVE_STASH: Final = TypeAdapter(_NativeStash)


def _window_identities(windows: Sequence[_NativeWindow]) -> frozenset[ReservationWindowIdentity]:
    return frozenset(
        (RateLimitScope(key, value).counter(kind), window_start, backend)
        for key, value, kind, window_start, backend in windows
    )


class _MemorySection:
    def __init__(self, native: "NativeParallelRequestLimiter | None") -> None:
        self.native: Final = native

    def __enter__(self) -> None:
        if self.native is not None:
            self.native.lock_memory()

    def __exit__(self, *_: object) -> None:
        if self.native is not None:
            self.native.unlock_memory()


class _ScopedReply(NamedTuple):
    """The descriptors the native builder made for one entry point, and the bridge's reply for its action."""

    descriptors: tuple[RateLimitDescriptor, ...]
    reply: Mapping[str, object]


class WindowKeyMetadata(TypedDict):
    requests_limit: int | None
    tokens_limit: int | None
    window_size: int
    descriptor_key: str
    descriptor_value: ReadOnly[str]


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
class BorrowingShareState:
    """One borrower's share on the admitting model for this request: what was enforced, what was held on it, and the
    holds taken on router fallback targets."""

    limit: RateLimitDescriptor | None = None
    reserved_tokens: int = 0
    reserved_window: ReservationWindowIdentity | None = None
    share: RateLimitDescriptor | None = None
    cap: RateLimitDescriptor | None = None
    cap_reserved_tokens: int = 0
    cap_reserved_window: ReservationWindowIdentity | None = None
    fallback_limits: Mapping[str, BorrowingTeamModelHold] = field(default_factory=lambda: MappingProxyType({}))

    def clear_reservation(self) -> None:
        self.reserved_tokens = 0
        self.cap_reserved_tokens = 0
        self.reserved_window = None
        self.cap_reserved_window = None

    def hold_fallback(self, names: tuple[str, ...], hold: BorrowingTeamModelHold) -> None:
        self.fallback_limits = MappingProxyType({**self.fallback_limits, **dict.fromkeys(names, hold)})

    def drop_fallback_hold(self, hold: BorrowingTeamModelHold | None) -> None:
        if hold is not None:
            self.fallback_limits = MappingProxyType({k: h for k, h in self.fallback_limits.items() if h is not hold})


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
    borrowing_shares: Mapping[BorrowerKind, BorrowingShareState] = field(
        default_factory=lambda: MappingProxyType({kind: BorrowingShareState() for kind in BORROWER_KINDS})
    )
    priority_class_share: RateLimitDescriptor | None = None
    priority_class_share_holder: str | None = None
    dynamic_limiter_priority: str | None = None
    reservation_released: bool = False
    io_reservation_released: bool = False
    tpm_limited_tags: frozenset[str] = field(default_factory=frozenset)
    socket_agent_windows: tuple[RateLimitDescriptor, ...] = ()
    native_state: "NativeParallelRequestLimiterState | None" = None


@dataclass(frozen=True, slots=True)
class TagRateLimit:
    rpm_limit: int | None
    tpm_limit: int | None


class TagRateLimitResolver(Protocol):
    def __call__(self, tag_names: Sequence[str], /) -> Awaitable[Mapping[str, TagRateLimit]]: ...


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


def request_identity(data: Mapping[str, object]) -> str | None:
    call_id: Final = data.get("litellm_call_id")
    if isinstance(call_id, str):
        return call_id
    logged: Final = getattr(data.get("litellm_logging_obj"), "litellm_call_id", None)
    return logged if isinstance(logged, str) else None


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


_SOCKET_TURN_UNCOUNTED_DESCRIPTOR_KEYS: Final = frozenset(
    {PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY, PTU_TEAM_DESCRIPTOR_KEY}
)


def socket_turn_tpm_windows(descriptors: Sequence[RateLimitDescriptor]) -> tuple[RateLimitDescriptor, ...]:
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


def _mcp_scope(user_api_key_dict: UserAPIKeyAuth | None, server: MCPServer) -> Mapping[str, object]:
    """The server an MCP tool call reaches and the per-server RPM limits its key and teams configure."""
    from litellm.proxy.auth.auth_utils import get_team_mcp_rpm_limit

    server_name: Final = server.alias or server.server_name or server.name
    keyed: Final = user_api_key_dict is not None and bool(server_name) and bool(user_api_key_dict.api_key)
    return {
        "server_name": server_name,
        "server_id": server.server_id,
        "server_rpm": server.rpm,
        "key_rpm_limit": _MCP_KEY_RPM_LIMITS.validate_python(
            _REQUEST_RATE_LIMIT_DATA.validate_python(user_api_key_dict.metadata or {}).get("mcp_rpm_limit")
        )
        if keyed and user_api_key_dict is not None
        else None,
        "team_rpm_limit": None if user_api_key_dict is None else get_team_mcp_rpm_limit(user_api_key_dict),
        "source_team_rpm_limits": [
            [team_id, limits]
            for team_id, limits in (
                (user_api_key_dict.mcp_source_team_rpm_limits if user_api_key_dict is not None else None) or {}
            ).items()
        ],
    }


def _parse_output_cap_value(raw_value: object) -> int | None:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, str)):
        return None
    try:
        return int(float(raw_value))
    except (ValueError, OverflowError):
        return None


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
    base_capped_floor: Final = no_max_tokens_output_floor(min_configured_limit)
    capped_floor: Final = (
        max(base_capped_floor, RESPONSES_API_MIN_OUTPUT_TOKENS)
        if call_type in RESPONSES_API_CALL_TYPES
        else base_capped_floor
    )
    baseline_floor: Final = int(DEFAULT_MAX_TOKENS_ESTIMATE * NO_MAX_TOKENS_TPM_RESERVATION_FRACTION)
    is_embedding: Final = _is_embedding_request(data, call_type)
    if (
        capped_floor >= baseline_floor
        or _has_explicit_output_cap(data, call_type)
        or is_embedding
        or endpoint_type == EndpointType.DECISIONS
        or call_type in EXTRACTION_API_CALL_TYPES
    ):
        return
    write_implicit_output_cap(data, call_type, max(capped_floor, configured_output_tokens or 0))


def _extraction_input_text(data: Mapping[str, object]) -> str:
    state: Final = data.get("state")
    return f"{state if isinstance(state, str) else ''}\n{json.dumps(data.get('fields'), default=str)}"


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


def _has_explicit_output_cap(data: object, call_type: str | None) -> bool:
    """Whether the caller explicitly set an output-token cap.

    Checked via ``is not None`` (not truthiness) so an explicit 0 --
    a legitimate zero-output request -- counts as explicit.
    """
    return _get_explicit_output_cap(data, call_type) is not None


class ParallelRequestLimiterHook(CustomLogger):
    """The ``parallel_request_limiter`` proxy hook: holds the native engine, forwards the logging events to it,
    and answers the callback-discovered frame checks. Everything else is a module function taking the hook."""

    runs_on_no_log_requests: ClassVar[bool] = True

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
        self.time_provider = time_provider or datetime.now
        self.tag_rate_limit_resolver = tag_rate_limit_resolver
        self.model_group_resolver = model_group_resolver
        self.counted_models_resolver = counted_models_resolver
        self.media_model_resolver = media_model_resolver
        self.ptu_team_ceiling_resolver = ptu_team_ceiling_resolver
        self.fail_closed_resolver = fail_closed_resolver
        self.force_hash_tag_grouping_resolver = force_hash_tag_grouping_resolver
        self.rate_limits: Final = rate_limit_store or RateLimitStore(
            connect(internal_usage_cache.dual_cache.redis_cache).rate_limits(
                internal_usage_cache.dual_cache.default_in_memory_ttl
            )
        )

        self.window_size = rate_limit_window_size()

        # When disabled, TPM is enforced post-call from actual usage (pre-v1.82
        # behavior) instead of reserving an estimated budget upfront, shedding
        # the extra per-request Redis Lua round-trip and the global-lock
        # in-memory fallback that the reservation path incurs.
        self.tpm_reservation_enabled = tpm_token_reservation_enabled()

        # Batch rate limiter (lazy loaded)
        self.batch_limiter: CallTypeRateLimiter | None = None
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
        self.check_and_increment_lock = asyncio.Lock()
        from litellm.proxy.hooks.parallel_request_limiter_v3_native import (
            NativeBatchEnqueuedTokenStore,
            limiter_engine,
        )

        self.native_companion: Final[NativeParallelRequestLimiter] = limiter_engine(self)
        self.batch_enqueued_token_store = NativeBatchEnqueuedTokenStore(internal_usage_cache, self.native_companion)

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
        scoped: Final = await run_scoped(
            self,
            user_api_key_dict,
            MappingProxyType({"model": requested_model}),
            {"kind": "project_io"},
            {
                "kind": "reserve_io",
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
            },
        )
        response: Final = _native_reply(scoped.reply)
        if response["overall_code"] == "OVER_LIMIT":
            handle_rate_limit_error(self, response, scoped.descriptors, requested_model)

    @with_service_target(RATE_LIMITS_TARGET)
    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        """Read one Responses WebSocket turn's token windows: every TPM limit the socket's request would
        check, plus the agent windows its admission held, without the slot and RPM it already took."""
        connection_stash: Final = get_request_stash()
        tags: Final = sorted(connection_stash.tpm_limited_tags) if connection_stash is not None else []
        agent_windows: Final = connection_stash.socket_agent_windows if connection_stash is not None else ()
        try:
            scoped: Final = await run_scoped(
                self,
                user_api_key_dict,
                MappingProxyType({"model": requested_model, "metadata": {"tags": tags}}),
                {"kind": "socket_turn", "agent_windows": [dict(window) for window in agent_windows]},
                {"kind": "check", "read_only": True, "skip_parallel_check": True},
                call_type=CallTypes.aresponses_websocket.value,
            )
            response: Final = _native_reply(scoped.reply)
        except RateLimitUnverifiableError as unverifiable:
            raise ServiceUnavailableError(
                message=RATE_LIMIT_UNVERIFIABLE_MESSAGE, llm_provider="litellm_proxy", model=requested_model or ""
            ) from unverifiable
        if response["overall_code"] == "OVER_LIMIT":
            handle_rate_limit_error(self, response, scoped.descriptors, requested_model)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        Settle the request's TPM usage and release its parallel slot in the native limiter, which reads the
        usage, the charged scopes and the slot reuse off the callback's kwargs and response itself.
        """
        from litellm.proxy.hooks.parallel_request_limiter_v3_native import post_call_request

        native_state: NativeParallelRequestLimiterState | None = None
        try:
            callback_kwargs: Final = _callback_kwargs_mapping(kwargs)
            stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(callback_kwargs))
            turn: Final = stash is None and _is_socket_turn(callback_kwargs)
            native_state = _native_state(self, get_request_stash() if turn else stash)
            await self.native_companion.log_success(
                native_state,
                callback_kwargs,
                response_obj,  # pyright: ignore[reportUnknownArgumentType]  # hook arguments are unannotated
                end_time,  # pyright: ignore[reportUnknownArgumentType]  # hook arguments are unannotated
                None if stash is None else stash.parallel_slot_reused_at,
                _settlement_resolved(self, callback_kwargs, native_state),
                token_rate_limit_type(),
                request=post_call_request() if self.rate_limits.has_redis else None,
                turn=turn,
            )
        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit success event: %s", e)
            if native_state is not None:
                await self.native_companion.release_on_disconnect(native_state)

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
        _mirror_ratelimit_response_into_logging_payload(
            kwargs=kwargs,
            response_obj=result,
        )
        return kwargs, result

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        On failure: release max_parallel_requests and refund the upfront TPM reservation, in the native
        limiter, only against the scopes the reservation actually charged. A failed stream whose partial
        usage was recovered settles the reservation at that usage instead of refunding it.
        """
        from litellm.proxy.hooks.parallel_request_limiter_v3_native import post_call_request

        try:
            failed_call: Final = _callback_kwargs_mapping(kwargs)
            stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(failed_call))
            turn: Final = stash is None and _is_socket_turn(failed_call)
            native_state: Final = _native_state(self, get_request_stash() if turn else stash)
            if native_state is None and not turn:
                return
            await self.native_companion.log_failure(
                native_state,
                failed_call,
                _settlement_resolved(self, failed_call, native_state) if turn else {},
                token_rate_limit_type(),
                request=post_call_request() if self.rate_limits.has_redis else None,
                turn=turn,
            )
        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit failure event: %s", e)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_release_max_parallel_requests_on_disconnect(
        self,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """
        Release the api-key ``max_parallel_requests`` slot the pre-call acquired, for a request that ended
        without either logging callback firing.

        The slot is normally released by ``async_log_success_event`` (natural stream completion) or
        ``async_log_failure_event`` (LLM error). When a client cancels a stream mid-flight, the cancellation
        surfaces as ``asyncio.CancelledError`` / ``GeneratorExit`` and neither callback runs, so without this
        the slot leaks per cancelled stream until its TTL prunes it. The native's held acquisition (not the
        key object's current max_parallel_requests configuration, which can change mid-request) decides
        whether there is anything to release.
        """
        native_state: Final = _native_state(self, get_request_stash())
        if native_state is not None:
            await self.native_companion.release_on_disconnect(native_state)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response) -> None:
        """
        Release completed-request slots and update rate limit headers in the response.
        """
        await limiter_post_call_success(limiter_post_call(self), data, user_api_key_dict, response)

    @with_service_target(RATE_LIMITS_TARGET)
    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],  # mutable-ok: the base hook declares this parameter as a dict
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """
        Release the parallel-request slot and any TPM/ITPM/OTPM reservation when the request is rejected
        after the pre-call hook acquired them but before the LLM call ran (e.g. a downstream guardrail/auth
        hook raised). async_log_failure_event is a litellm completion-level callback and never fires for
        proxy-side rejections, so without this a leaked slot would occupy the gauge for the full
        PARALLEL_REQUEST_SLOT_TTL_SECONDS.

        Idempotent in the native limiter: the slot release takes the held acquisition, and the refund is
        guarded by the reservation's released flags. A mid-stream failure relayed here with recovered
        partial usage settles the reservation at that usage instead of refunding it.
        """
        try:
            native_state: Final = _native_state(self, get_request_stash())
            if native_state is None:
                return
            await self.native_companion.log_post_call_failure(native_state, request_data, token_rate_limit_type())
        except Exception as e:
            verbose_proxy_logger.exception("Error releasing TPM reservation on post-call failure: %s", e)


PROXY_MaxParallelRequestsHandler_v3 = ParallelRequestLimiterHook
_PROXY_MaxParallelRequestsHandler_v3 = ParallelRequestLimiterHook


def limiter_post_call(limiter: ParallelRequestLimiterHook) -> "LimiterPostCall":
    return LimiterPostCall(
        rate_limits=limiter.rate_limits,
        native=limiter.native_companion,
        batch_enqueued_token_store=limiter.batch_enqueued_token_store,
        memory_lock=limiter.check_and_increment_lock,
    )


def _native_state(
    limiter: ParallelRequestLimiterHook, stash: RequestRateLimiterStash | None
) -> "NativeParallelRequestLimiterState | None":
    return limiter_native_state(limiter_post_call(limiter), stash)


def get_batch_rate_limiter(limiter: ParallelRequestLimiterHook) -> CallTypeRateLimiter | None:
    """Get or lazy-load the batch rate limiter."""
    if limiter.batch_limiter is None:
        try:
            from litellm.proxy.hooks.batch_rate_limiter import (
                PROXY_BatchRateLimiter,
            )

            limiter.batch_limiter = PROXY_BatchRateLimiter(  # rebind-ok: the hook caches its lazily built batch limiter
                internal_usage_cache=limiter.internal_usage_cache,
                parallel_request_limiter=limiter,
                time_provider=limiter.time_provider,
            )
        except Exception as e:
            verbose_proxy_logger.debug("Could not load batch rate limiter: %s", e)
    return limiter.batch_limiter


def current_time(limiter: ParallelRequestLimiterHook) -> datetime:
    """Return the current time for rate limiting calculations."""
    return limiter.time_provider()


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


def write_implicit_output_cap(data: object, call_type: str | None, effective_cap: int) -> bool:
    if not isinstance(data, dict):
        return False
    if call_type in GOOGLE_GENAI_NATIVE_CALL_TYPES:
        config_field: Final = "config" if "config" in data or "generationConfig" not in data else "generationConfig"
        config: Final = data.get(config_field)
        if config is None or isinstance(config, dict):
            data[config_field] = {  # rebind-ok: routed request needs cap
                **(config or {}),
                "maxOutputTokens": effective_cap,
            }
            return True
        return False
    cap_field: Final = "max_output_tokens" if call_type in RESPONSES_API_CALL_TYPES else "max_tokens"
    existing_cap: Final = data.get(cap_field)
    if existing_cap is None or effective_cap < existing_cap:
        data[cap_field] = effective_cap  # rebind-ok: downstream routing requires the bounded output cap
        return True
    return False


def estimate_request_tokens(
    limiter: ParallelRequestLimiterHook,
    data: object,
    model: str,
    min_configured_tpm_limit: int,
    call_type: str | None,
    team_id: str | None,
    rate_limit_type: str = "total",
) -> int:
    return request_token_estimate(
        data=data,
        model=model,
        min_configured_tpm_limit=min_configured_tpm_limit,
        call_type=call_type,
        team_id=team_id,
        rate_limit_type=rate_limit_type,
        media_model_resolver=limiter.media_model_resolver,
    )


def _native_reply(reply: Mapping[str, object]) -> RateLimitResponse:
    if reply.get("unverifiable"):
        raise RateLimitUnverifiableError()
    windows: Final = _NATIVE_WINDOWS.validate_python(reply.get("windows"))
    response: Final = RateLimitResponse(
        overall_code="OVER_LIMIT" if reply.get("overall_code") == "OVER_LIMIT" else "OK",
        statuses=_NATIVE_STATUSES.validate_python(reply.get("statuses")),
    )
    if windows is None:
        return response
    return RateLimitResponse(
        overall_code=response["overall_code"],
        statuses=response["statuses"],
        reservation_windows=_window_identities(windows),
    )


def _native_descriptors(
    limiter: ParallelRequestLimiterHook, descriptors: Sequence[RateLimitDescriptor], skip_tpm_check: bool
) -> str:
    try:
        return json.dumps([dict(descriptor) for descriptor in descriptors])
    except TypeError:
        _collect_windowed_keys_and_gauges(limiter, descriptors=descriptors, skip_tpm_check=skip_tpm_check)
        raise


async def _native_should_rate_limit(
    limiter: ParallelRequestLimiterHook,
    descriptors: Sequence[RateLimitDescriptor],
    read_only: bool,
    skip_tpm_check: bool,
    parallel_slot_id: str | None,
    skip_parallel_check: bool,
) -> RateLimitResponse:
    from litellm.proxy.hooks.parallel_request_limiter_v3_native import limiter_policy, pre_call_request

    reply: Final = await limiter.native_companion.should_rate_limit(
        _native_descriptors(limiter, descriptors, skip_tpm_check),
        read_only,
        skip_tpm_check,
        parallel_slot_id,
        skip_parallel_check,
        json.dumps(limiter_policy(limiter)),
        pre_call_request() if limiter.rate_limits.has_request_batch else None,
    )
    response: Final = _native_reply(reply)
    return RateLimitResponse(overall_code=response["overall_code"], statuses=response["statuses"])


async def _native_atomic_check_and_increment(
    limiter: ParallelRequestLimiterHook,
    descriptors: Sequence[RateLimitDescriptor],
    increments: Sequence[Mapping[Literal["requests", "tokens"], int]],
) -> RateLimitResponse:
    from litellm.proxy.hooks.parallel_request_limiter_v3_native import limiter_policy, pre_call_request

    reply: Final = await limiter.native_companion.atomic_check_and_increment(
        _native_descriptors(limiter, descriptors, skip_tpm_check=False),
        [(int(increment.get("requests", 0) or 0), int(increment.get("tokens", 0) or 0)) for increment in increments],
        json.dumps(limiter_policy(limiter)),
        pre_call_request() if limiter.rate_limits.has_request_batch else None,
    )
    return _native_reply(reply)


@with_service_target(RATE_LIMITS_TARGET)
async def should_rate_limit(
    limiter: ParallelRequestLimiterHook,
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
            pass. Callers that reserve TPM through the native
            reservation path should set this to
            avoid the +1-per-key Lua / in-memory increment double-charging
            the tokens counter.
        skip_parallel_check: If True, ``max_parallel_requests`` gauges are
            neither checked nor acquired, for a call that runs inside a
            request already holding its slot.

    ``max_parallel_requests`` descriptors are enforced by the dedicated
    concurrency-gauge path in the native limiter, never by the windowed
    counters. The gauge phase must stay AFTER the windowed
    check so a windowed rejection never strands an acquired slot; the
    reverse order would leak one gauge slot per RPM/TPM rejection.
    ``parallel_slot_id`` names the slot an admission registers; callers
    that enforce (not read_only) should pass the id they will later
    release with — when omitted, a generated slot id is used and the slot
    can only be reclaimed by TTL expiry.
    """
    return await _native_should_rate_limit(
        limiter, descriptors, read_only, skip_tpm_check, parallel_slot_id, skip_parallel_check
    )


def _collect_windowed_keys_and_gauges(
    limiter: ParallelRequestLimiterHook,
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
        window_size = rate_limit.get("window_size") or limiter.window_size

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


async def _release_parallel_request_slots(
    limiter: ParallelRequestLimiterHook,
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
    await release_parallel_request_slots(limiter_post_call(limiter), acquisition)


@with_service_target(RATE_LIMITS_TARGET)
async def atomic_check_and_increment_by_n(
    limiter: ParallelRequestLimiterHook,
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
    return await _native_atomic_check_and_increment(limiter, descriptors, increments)


def rate_limited_model(limiter: ParallelRequestLimiterHook, requested_model: str | None) -> RateLimitedModel | None:
    """The requested name and the model group it resolves to, so limits keyed by either name are found."""
    return resolve_rate_limited_model(
        requested_model=requested_model, model_group_resolver=limiter.model_group_resolver
    )


def _rate_limited_models(
    limiter: ParallelRequestLimiterHook, model: RateLimitedModel | None
) -> tuple[RateLimitedModel, ...]:
    if model is None:
        return ()
    return (
        model,
        *(
            target
            for candidate in expand_model_latest_alias(model.requested)[1:]
            if (target := rate_limited_model(limiter, candidate)) is not None
        ),
    )


async def enforce_mcp_server_rate_limits(
    limiter: ParallelRequestLimiterHook,
    user_api_key_dict: UserAPIKeyAuth | None,
    server: MCPServer,
) -> None:
    """Charge one MCP tool call to the key's, the teams' and the server's per-server RPM.

    MCP tool calls have no token usage, so only requests are counted. A keyless MCP-admitted subject
    reaches servers through several teams at once, so every team that grants the server is charged.
    """
    scoped: Final = await run_scoped(
        limiter,
        user_api_key_dict,
        MappingProxyType({}),
        {"kind": "mcp", **_mcp_scope(user_api_key_dict, server)},
        {"kind": "charge", "requests": 1},
        lookups=False,
    )
    response: Final = _native_reply(scoped.reply)
    if response["overall_code"] == "OVER_LIMIT":
        handle_rate_limit_error(limiter, response, scoped.descriptors)


def registry_agent(agent_id: str) -> "AgentResponse | None":
    """Look up an agent from the in-memory registry by ID."""
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    return global_agent_registry.get_agent_by_id(agent_id=agent_id)


async def model_has_recent_failures(
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


def get_rate_limiter_for_call_type(limiter: ParallelRequestLimiterHook, call_type: str) -> CallTypeRateLimiter | None:
    """Get the rate limiter for the call type."""
    if call_type == "acreate_batch":
        batch_limiter: Final = get_batch_rate_limiter(limiter)
        return batch_limiter
    return None


def _key_owns_model_limit(
    user_api_key_dict: UserAPIKeyAuth,
    model: RateLimitedModel,
    rate_limit_key: Literal["model_rpm_limit", "model_tpm_limit"],
) -> bool:
    return model.limit_in(get_key_own_model_rate_limit(user_api_key_dict, rate_limit_key)) is not None


def _key_owns_model_tpm_limit_from_request_metadata(
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
    return _key_owns_model_limit(key_view, model, "model_tpm_limit")


def handle_rate_limit_error(
    limiter: ParallelRequestLimiterHook,
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

            now = current_time(limiter).timestamp()
            reset_time = now + limiter.window_size
            reset_time_formatted = datetime.fromtimestamp(reset_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

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
                    "retry-after": str(limiter.window_size),
                    "rate_limit_type": str(status["rate_limit_type"]),
                    "rate_limited_by": descriptor_key,
                    "reset_at": reset_time_formatted,
                },
                rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                model=resolved_model,
                llm_provider=llm_provider,
                descriptor_key=descriptor_key,
            )


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


def _estimate_audio_content_tokens(messages: object) -> int:
    """
    Sum of per-block audio token estimates across all ``messages``.
    Returns 0 when there are no ``input_audio`` blocks, which the caller
    uses to skip the (relatively expensive) strip pass.
    """
    if not isinstance(messages, list):
        return 0
    return sum(
        _estimate_audio_block_tokens(block)
        for message in messages
        if isinstance(message, dict)
        for content in (message.get("content"),)
        if isinstance(content, list)
        for block in content
        if isinstance(block, dict) and block.get("type") == "input_audio"
    )


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


def _count_pretokenized_embedding_input(value: object) -> int | None:
    if not isinstance(value, list):
        return None
    if all(isinstance(token, int) for token in value):
        return len(value)
    if all(isinstance(token_ids, list) and all(isinstance(token, int) for token in token_ids) for token_ids in value):
        return sum(len(token_ids) for token_ids in value)
    return None


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


def estimate_precise_input_tokens(
    limiter: ParallelRequestLimiterHook,
    data: object,
    model: str | None,
    call_type: str | None = None,
    team_id: str | None = None,
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
        None if is_responses_request else _translate_google_genai_native_request(data, call_type)
    )
    is_embedding_request: Final = _is_embedding_request(data, call_type)
    request: Final = as_str_mapping(data)
    raw_embedding_input: Final = request.get("input") if is_embedding_request and request is not None else None
    pretokenized_input_tokens: Final = (
        _count_pretokenized_embedding_input(raw_embedding_input) if is_embedding_request else None
    )
    if pretokenized_input_tokens is not None:
        return pretokenized_input_tokens
    embedding_text, embedding_media_tokens = (
        split_embedding_input_tokens(
            raw_embedding_input, lambda: limiter.media_model_resolver(model or "", team_id), model
        )
        if is_embedding_request
        else (None, 0)
    )

    prompt: Final = data.get("prompt")
    fallback_text: Final = prompt if prompt is not None else data.get("input")
    selected_inputs: Final[tuple[object | None, object | None, object | None, object | None]] = (
        (_responses_input_to_chat_messages(data), None, data.get("tools"), data.get("tool_choice"))
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
        else (None, _rerank_input_to_text(data), data.get("tools"), data.get("tool_choice"))
        if call_type in RERANK_API_CALL_TYPES
        else (None, prompt, data.get("tools"), data.get("tool_choice"))
        if call_type in TEXT_COMPLETION_API_CALL_TYPES
        else (None, _extraction_input_text(_REQUEST_RATE_LIMIT_DATA.validate_python(data)), None, None)
        if call_type in EXTRACTION_API_CALL_TYPES
        else (data.get("messages"), fallback_text, data.get("tools"), data.get("tool_choice"))
    )
    messages, selected_text, countable_tools, countable_tool_choice = selected_inputs

    audio_token_estimate: Final = _estimate_audio_content_tokens(messages)
    countable_messages: Final = _strip_audio_content_blocks(messages) if audio_token_estimate > 0 else messages
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
                for tokenizer_model in (limiter.counted_models_resolver(model, team_id) if model else ("",))
            ),
        )
        return estimate + audio_token_estimate + embedding_media_tokens
    except Exception:  # noqa: BLE001  # tokenizer failures degrade to the cheap estimate
        if call_type in RERANK_API_CALL_TYPES and isinstance(selected_text, str):
            return max(0, len(selected_text) // DEFAULT_CHARS_PER_TOKEN)
        estimated_input_tokens, _ = estimate_input_and_output_tokens(
            data=data,
            call_type=call_type,
            include_media=is_embedding_request,
            team_id=team_id,
            media_model_resolver=limiter.media_model_resolver,
        )
        return estimated_input_tokens + audio_token_estimate


async def run_scoped(
    limiter: ParallelRequestLimiterHook,
    user_api_key_dict: UserAPIKeyAuth | None,
    data: Mapping[str, object],
    selection: Mapping[str, object],
    action: Mapping[str, object],
    *,
    call_type: str | None = None,
    state: "NativeParallelRequestLimiterState | None" = None,
    lookups: bool = True,
) -> _ScopedReply:
    """Build an entry point's descriptors with the native pre-call's builder from the same typed key
    records and lookups, then run ``action`` on them. A Python error the builder reproduces is raised."""
    from litellm.proxy.hooks.parallel_request_limiter_v3_native import (
        _descriptors,  # pyright: ignore[reportPrivateUsage]  # the native's own descriptor shape
        pre_call_request,
        raise_wire_abort,
        scoped_inputs,
    )

    inputs: Final = await scoped_inputs(limiter, user_api_key_dict, data, call_type, lookups=lookups)
    reply: Final = await limiter.native_companion.scoped(
        {**inputs, "selection": selection, "action": action},
        state=state,
        request=pre_call_request() if state is not None or limiter.rate_limits.has_request_batch else None,
    )
    raise_wire_abort(reply.get("abort"))
    return _ScopedReply(descriptors=_descriptors().validate_python(reply.get("descriptors")), reply=reply)


async def raise_if_over_limit_before_body(
    limiter: ParallelRequestLimiterHook, user_api_key_dict: UserAPIKeyAuth
) -> None:
    """Refuse a request from its headers alone once a limit that needs no body is exhausted.

    Counters are only read, never incremented, so an admitted request is still charged by
    ``async_pre_call_hook`` after its body is parsed. Limits that need the model are not checked here.
    """
    scoped: Final = await run_scoped(
        limiter,
        user_api_key_dict,
        MappingProxyType({}),
        {"kind": "request"},
        {"kind": "check", "read_only": True, "skip_tpm_check": limiter.tpm_reservation_enabled},
    )
    response: Final = _native_reply(scoped.reply)
    if response["overall_code"] == "OVER_LIMIT":
        handle_rate_limit_error(limiter, response=response, descriptors=scoped.descriptors)


async def caller_model_rate_limit_response(
    limiter: ParallelRequestLimiterHook, user_api_key_dict: UserAPIKeyAuth, model: str
) -> RateLimitResponse:
    """Read the limits a proxy sub-call on ``model`` runs under, inside its caller's admitted request.

    Nothing is reserved or incremented. ``max_parallel_requests`` and the RPM of every caller-wide
    scope are left out, because the caller's own slot and request would otherwise count against it.
    """
    scoped: Final = await run_scoped(
        limiter,
        user_api_key_dict,
        MappingProxyType({"model": model}),
        {"kind": "sub_call"},
        {"kind": "check", "read_only": True},
    )
    return _native_reply(scoped.reply)


async def batch_rate_limit_descriptors(
    limiter: ParallelRequestLimiterHook, user_api_key_dict: UserAPIKeyAuth, data: Mapping[str, object]
) -> list[RateLimitDescriptor]:  # mutable-ok: the batch limiter extends the list it charges
    """The key, user, team, model and agent limits a batch submission is charged against, with a
    scope's daily token limit replacing its per-minute descriptor."""
    scoped: Final = await run_scoped(
        limiter,
        user_api_key_dict,
        data,
        {
            "kind": "batch",
            "tpd_limits": [
                {"key": key, "value": value, "limit": limit}
                for key, value, limit in (
                    ("api_key", user_api_key_dict.api_key, user_api_key_dict.tpd_limit),
                    ("team", user_api_key_dict.team_id, user_api_key_dict.team_tpd_limit),
                    ("end_user", user_api_key_dict.end_user_id, user_api_key_dict.end_user_tpd_limit),
                )
            ],
        },
        {"kind": "build"},
    )
    return list(scoped.descriptors)


async def project_io_token_rate_limit_descriptors(
    limiter: ParallelRequestLimiterHook, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
) -> tuple[RateLimitDescriptor, ...]:
    """The project's ITPM and OTPM quotas on ``requested_model``."""
    scoped: Final = await run_scoped(
        limiter,
        user_api_key_dict,
        MappingProxyType({"model": requested_model}),
        {"kind": "project_io"},
        {"kind": "build"},
    )
    return scoped.descriptors


async def charge_skill_hub_publish(limiter: ParallelRequestLimiterHook, identity: str) -> RateLimitResponse:
    """Charge one Skills Hub upload or import to the caller's per-minute publishing budget."""
    scoped: Final = await run_scoped(
        limiter,
        None,
        MappingProxyType({}),
        {"kind": "skill_hub_publish", "value": identity},
        {"kind": "charge", "requests": 1},
        lookups=False,
    )
    return _native_reply(scoped.reply)


async def _release_request_capacity_when_admitted(
    limiter: ParallelRequestLimiterHook,
    admission: "asyncio.Task[_ScopedReply]",
    slot_id: str,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    scoped: Final = await admission
    if scoped.reply.get("overall_code") == "OK":
        await _release_parallel_request_slots(
            limiter,
            ParallelSlotAcquisition(
                slot_id=slot_id,
                gauges=tuple(
                    RateLimitScope(d["key"], d["value"])
                    for d in scoped.descriptors
                    if d["rate_limit"] is not None and d["rate_limit"].get("max_parallel_requests") is not None
                ),
            ),
            user_api_key_dict.parent_otel_span,
        )


async def fallback_target_allowed(
    limiter: ParallelRequestLimiterHook,
    user_api_key_dict: UserAPIKeyAuth,
    model: str,
    request_data: Mapping[str, object],
) -> bool:
    """Reserve the caller's per-model limits on a router fallback target. False when any is exhausted."""
    stash: Final = _request_stash.get()
    native_state: Final = _native_state(limiter, stash)
    scoped: Final = await run_scoped(
        limiter,
        user_api_key_dict,
        MappingProxyType({**request_data, "model": model}),
        {"kind": "fallback_target"},
        {"kind": "fallback_target", "model": model},
        state=native_state,
    )
    allowed: Final = bool(scoped.reply.get("allowed"))
    if allowed and native_state is None and stash is not None:
        stash.fallback_models = stash.fallback_models | {model}
    return allowed


@asynccontextmanager
async def request_capacity(
    limiter: ParallelRequestLimiterHook,
    user_api_key_dict: UserAPIKeyAuth,
    model: str,
    *,
    request_data: Mapping[str, object] | None = None,
) -> AsyncGenerator[None, None]:
    """Charge one non-generation provider request to RPM and hold its concurrency slot."""
    data: Final = MappingProxyType({**(request_data or MappingProxyType({})), "model": model})
    slot_id: Final = uuid.uuid4().hex
    admission: Final = asyncio.create_task(
        run_scoped(
            limiter,
            user_api_key_dict,
            data,
            {"kind": "request"},
            {"kind": "check", "skip_tpm_check": True, "parallel_slot_id": slot_id},
        )
    )
    try:
        scoped: Final = await asyncio.shield(admission)
        response: Final = _native_reply(scoped.reply)
        if response["overall_code"] == "OVER_LIMIT":
            handle_rate_limit_error(limiter, response, scoped.descriptors, model)
        yield
    finally:
        cleanup: Final = asyncio.create_task(
            _release_request_capacity_when_admitted(limiter, admission, slot_id, user_api_key_dict)
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


def _native_operation(operation: ReservationOperation) -> _NativeOp:
    if isinstance(operation, CounterIncrement):
        scope: Final = operation.counter.scope
        return (
            scope.descriptor_key,
            scope.descriptor_value,
            operation.counter.kind,
            operation.amount,
            operation.ttl,
            None,
            None,
        )
    increment: Final = operation.increment
    return (
        increment.counter.scope.descriptor_key,
        increment.counter.scope.descriptor_value,
        increment.counter.kind,
        increment.amount,
        increment.ttl,
        operation.expected_window_start,
        operation.reservation_backend,
    )


@with_service_target(RATE_LIMITS_TARGET)
async def increment_reservation_aware_tokens(
    limiter: ParallelRequestLimiterHook,
    pipeline_operations: Sequence[ReservationOperation],
    parent_otel_span: Span | None = None,
) -> None:
    """Apply reservation settlements in the native limiter: plain increments now, window-guarded ones only
    while the window they were reserved in is still the open one."""
    if not pipeline_operations:
        return
    await limiter.native_companion.reservation_aware(
        [_native_operation(operation) for operation in pipeline_operations]
    )


def _settlement_resolved(
    limiter: ParallelRequestLimiterHook,
    kwargs: Mapping[str, object],
    native_state: "NativeParallelRequestLimiterState | None",
) -> Mapping[str, object]:
    """What only the proxy resolves for a native settlement: the served and reserved models through the
    router's aliases, whether the key owns each one's TPM limit, the team's PTU share, and the live policy.
    The usage, targets and totals the native reads off the callback itself."""
    from litellm.litellm_core_utils.core_helpers import (
        get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper is untyped
    )
    from litellm.proxy.hooks.parallel_request_limiter_v3_native import limiter_policy

    empty: Final[Mapping[str, object]] = MappingProxyType({})
    request_metadata: Final = (
        as_str_mapping(get_litellm_metadata_from_kwargs(dict(kwargs)))  # pyright: ignore[reportUnknownArgumentType]  # the helper is untyped
        or empty
    )
    logging_metadata: Final = (
        as_str_mapping((as_str_mapping(kwargs.get("standard_logging_object")) or empty).get("metadata")) or empty
    )
    team_id: Final = logging_metadata.get("user_api_key_team_id")
    served_model: Final = rate_limited_model(limiter, get_model_group_from_litellm_kwargs(dict(kwargs)))
    reserved: Final = None if native_state is None else limiter.native_companion.reserved_model(native_state)
    reserved_model: Final = None if reserved is None else RateLimitedModel(requested=reserved[0], group=reserved[1])
    return {
        "served_model": None
        if served_model is None
        else {"requested": served_model.requested, "group": served_model.group},
        "models": {
            model.requested: {
                "candidates": [
                    {
                        "group": candidate.group,
                        "key_owns": _key_owns_model_tpm_limit_from_request_metadata(request_metadata, candidate),
                    }
                    for candidate in _rate_limited_models(limiter, model)
                ],
                "ptu_ceiling": None
                if ceiling is None
                else {
                    "model_group": ceiling.model_group,
                    "tpm_limit": ceiling.tpm_limit,
                    "output_to_input_ratio": ceiling.output_to_input_ratio,
                    "cached_input_ratio": ceiling.cached_input_ratio,
                },
            }
            for model in (served_model, reserved_model)
            if model is not None
            for ceiling in (
                limiter.ptu_team_ceiling_resolver(team_id, model.group)
                if isinstance(team_id, str) and team_id
                else None,
            )
        },
        "policy": limiter_policy(limiter),
    }


def _mirror_ratelimit_response_into_logging_payload(
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
        hidden_params["additional_headers"] = _merge_ratelimit_statuses_into_additional_headers(
            additional_headers=existing if isinstance(existing, dict) else {},
            statuses=statuses,
        )
        standard_logging_object["hidden_params"] = hidden_params

    response_hidden: Final = getattr(response_obj, "_hidden_params", None)
    if isinstance(response_hidden, dict):
        existing = response_hidden.get("additional_headers")
        response_hidden["additional_headers"] = _merge_ratelimit_statuses_into_additional_headers(
            additional_headers=existing if isinstance(existing, dict) else {},
            statuses=statuses,
        )


def rate_limit_window_size() -> int:
    return int(os.getenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", 60))


def tpm_token_reservation_enabled() -> bool:
    return os.getenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "true").lower() == "true"


def estimate_input_and_output_tokens(
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
    translated_data: Final = _translate_google_genai_native_request(data, call_type)
    estimable_data: Final = translated_data if translated_data is not None else request_data
    selected_fields: Final[tuple[object | None, object | None, object | None]] = (
        (None, None, estimable_data.get("input"))
        if call_type in RESPONSES_API_CALL_TYPES or call_type in EMBEDDING_API_CALL_TYPES
        else (None, estimable_data.get("prompt"), None)
        if call_type in TEXT_COMPLETION_API_CALL_TYPES
        else (
            None,
            _extraction_input_text(_REQUEST_RATE_LIMIT_DATA.validate_python(estimable_data)),
            None,
        )
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

    explicit_max_tokens: Final = _get_explicit_output_cap(data, call_type)
    generates_no_output: Final = _is_embedding_request(data, call_type) or call_type in EXTRACTION_API_CALL_TYPES

    base_output_floor: Final = no_max_tokens_output_floor(min_configured_tpm_limit)
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

    return estimated_input_tokens, max_tokens_estimate * get_output_candidate_count(data, call_type)


def estimate_tokens_for_request(
    data: object,
    model: str | None = None,
    min_configured_tpm_limit: int | None = None,
    call_type: str | None = None,
    configured_output_tokens: int | None = None,
    team_id: str | None = None,
    media_model_resolver: Callable[[str, str | None], tuple[str, ...]] = _resolve_media_models_via_proxy_router,
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
    estimated_input_tokens, max_tokens_estimate = estimate_input_and_output_tokens(
        data=data,
        min_configured_tpm_limit=min_configured_tpm_limit,
        call_type=call_type,
        configured_output_tokens=configured_output_tokens,
        team_id=team_id,
        media_model_resolver=media_model_resolver,
    )
    total_estimated: Final = estimated_input_tokens + max_tokens_estimate

    verbose_proxy_logger.debug(
        "TPM reservation estimate: input=%s, max_tokens=%s, total=%s",
        estimated_input_tokens,
        max_tokens_estimate,
        total_estimated,
    )

    return total_estimated


def request_token_estimate(
    data: object,
    model: str,
    min_configured_tpm_limit: int,
    call_type: str | None,
    team_id: str | None,
    rate_limit_type: str = "total",
    media_model_resolver: Callable[[str, str | None], tuple[str, ...]] = _resolve_media_models_via_proxy_router,
) -> int:
    """The reservation this limiter would hold for the request, for another limiter settling the same call."""
    if rate_limit_type == "total":
        return estimate_tokens_for_request(
            data=data,
            model=model,
            min_configured_tpm_limit=min_configured_tpm_limit,
            call_type=call_type,
            team_id=team_id,
            media_model_resolver=media_model_resolver,
        )
    estimated_input, output_budget = estimate_input_and_output_tokens(
        data=data,
        min_configured_tpm_limit=min_configured_tpm_limit,
        call_type=call_type,
        team_id=team_id,
        media_model_resolver=media_model_resolver,
    )
    return output_budget if rate_limit_type == "output" else estimated_input


def cap_output_for_limit(data: object, min_configured_limit: int, call_type: str | None) -> None:
    """Cap a request without an explicit output cap the way this limiter does for its own small TPM limits."""
    _apply_implicit_output_cap(data=data, min_configured_limit=min_configured_limit, call_type=call_type)


def token_rate_limit_type() -> Literal["output", "input", "total"]:
    from litellm.proxy.proxy_server import general_settings

    specified_rate_limit_type: Final = general_settings.get("token_rate_limit_type", "total")
    if specified_rate_limit_type not in [
        "output",
        "input",
        "total",
    ]:
        return "total"  # default to total
    return specified_rate_limit_type


def resolve_rate_limited_model(
    requested_model: str | None,
    model_group_resolver: Callable[[str], str | None] = _resolve_model_group_alias_via_proxy_router,
) -> RateLimitedModel | None:
    if not requested_model:
        return None
    return RateLimitedModel(
        requested=requested_model,
        group=model_group_resolver(requested_model) or requested_model,
    )


@dataclass(frozen=True)
class LimiterPostCall:
    rate_limits: RateLimitStore
    native: "NativeParallelRequestLimiter | None"
    batch_enqueued_token_store: BatchEnqueuedTokenStore
    memory_lock: asyncio.Lock


SlotRelease = Callable[[ParallelSlotAcquisition, "Span | None"], Awaitable[None]]


def limiter_native_state(
    limiter: LimiterPostCall, stash: RequestRateLimiterStash | None
) -> "NativeParallelRequestLimiterState | None":
    if limiter.native is None or stash is None:
        return None
    return stash.native_state


async def take_native_slot(limiter: LimiterPostCall, stash: RequestRateLimiterStash) -> ParallelSlotAcquisition | None:
    native_state: Final = limiter_native_state(limiter, stash)
    if limiter.native is None or native_state is None:
        return None
    taken: Final = await limiter.native.take_parallel_slot(native_state)
    if taken is None:
        return None
    slot_id, gauges = taken
    return ParallelSlotAcquisition(slot_id=slot_id, gauges=tuple(RateLimitScope(key, value) for key, value in gauges))


def mirror_released_parallel_slots(
    rate_limits: RateLimitStore, gauges: Sequence[RateLimitScope], remaining_by_gauge: Sequence[int]
) -> None:
    for scope, remaining in zip(gauges, remaining_by_gauge):
        rate_limits.set_local(scope.gauge, max(0, remaining), PARALLEL_REQUEST_SLOT_TTL_SECONDS)


async def release_parallel_request_slots_in_memory(
    limiter: LimiterPostCall, gauges: Sequence[RateLimitScope], slot_id: str
) -> None:
    async with limiter.memory_lock:
        with _MemorySection(limiter.native):
            for scope in gauges:
                raw_value = limiter.rate_limits.local((scope.gauge,))[0]
                if isinstance(raw_value, dict):
                    if slot_id not in raw_value:
                        continue
                    new_value: dict[str, float] | int = {  # mutable-ok: the memory gauge keeps its slots as a dict
                        key: ts for key, ts in raw_value.items() if key != slot_id
                    }
                elif raw_value is None:
                    continue
                else:
                    new_value = max(0, int(raw_value) - 1)
                limiter.rate_limits.set_local(scope.gauge, new_value, PARALLEL_REQUEST_SLOT_TTL_SECONDS)


async def release_parallel_request_slots(limiter: LimiterPostCall, acquisition: ParallelSlotAcquisition) -> None:
    if limiter.native is not None:
        await limiter.native.release_slots(
            acquisition["slot_id"],
            [(scope.descriptor_key, scope.descriptor_value) for scope in acquisition["gauges"]],
        )
        return
    gauges: Final = acquisition["gauges"]
    slot_id: Final = acquisition["slot_id"]
    if not gauges or not slot_id:
        return
    if limiter.rate_limits.has_redis:
        try:
            remaining: Final = await limiter.rate_limits.parallel_release(gauges, slot_id, batch=Batch.NONE)
            mirror_released_parallel_slots(limiter.rate_limits, gauges, remaining)
            return
        except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to the in-memory release, never a 500
            log_redis_failure(
                verbose_proxy_logger,
                logging.WARNING,
                "parallel_release_script failed, falling back to in-memory release",
                e,
            )
    await release_parallel_request_slots_in_memory(limiter, gauges, slot_id)


async def defer_parallel_slot_release(limiter: LimiterPostCall, acquisition: ParallelSlotAcquisition) -> bool:
    gauges: Final = acquisition["gauges"]
    slot_id: Final = acquisition["slot_id"]
    if not limiter.rate_limits.has_request_batch or not gauges or not slot_id:
        return False
    await release_parallel_request_slots_in_memory(limiter, gauges, slot_id)
    return limiter.rate_limits.parallel_release_post_call(gauges, slot_id)


async def release_stashed_parallel_slot(
    limiter: LimiterPostCall,
    stash: RequestRateLimiterStash | None,
    parent_otel_span: "Span | None",
    *,
    in_logging_callback: bool = False,
    round_ended_at: datetime | None = None,
    release: SlotRelease | None = None,
) -> None:
    if stash is None:
        return
    async with stash.parallel_slot_release_lock:
        if _ended_before_slot_reuse(round_ended_at, stash.parallel_slot_reused_at):
            return
        acquisition: Final = stash.parallel_slot or await take_native_slot(limiter, stash)
        if acquisition is None:
            return
        deferred: Final = in_logging_callback and await defer_parallel_slot_release(limiter, acquisition)
        if not deferred:
            if release is None:
                await release_parallel_request_slots(limiter, acquisition)
            else:
                await release(acquisition, parent_otel_span)
        stash.parallel_slot = None  # rebind-ok: marks this request's slot as released


async def handle_batch_enqueued_post_call(
    limiter: LimiterPostCall, user_api_key_dict: UserAPIKeyAuth, response: object
) -> None:
    stash: Final = get_request_stash()
    if limiter_native_state(limiter, stash) is not None:
        return
    view: Final = batch_response_view(response)
    if view is None:
        return
    span: Final = user_api_key_dict.parent_otel_span
    if stash is not None and stash.batch_enqueued_reservation is not None:
        await limiter.batch_enqueued_token_store.save_reservation(
            batch_id=canonical_provider_batch_id(view.id),
            reservation=stash.batch_enqueued_reservation,
            litellm_parent_otel_span=span,
        )
        stash.batch_enqueued_reservation = None
    if view.status.lower() in BATCH_ENQUEUED_REFUND_STATUSES:
        popped: Final = await limiter.batch_enqueued_token_store.pop_reservation(
            batch_id=canonical_provider_batch_id(view.id),
            litellm_parent_otel_span=span,
        )
        if popped is not None:
            await limiter.batch_enqueued_token_store.refund(reservation=popped, litellm_parent_otel_span=span)


async def limiter_post_call_success(
    limiter: LimiterPostCall,
    data: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    response: object,
    *,
    release: SlotRelease | None = None,
) -> None:
    try:
        slot_stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(data))
        await release_stashed_parallel_slot(limiter, slot_stash, user_api_key_dict.parent_otel_span, release=release)
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
                _merge_ratelimit_statuses_into_additional_headers(
                    additional_headers={},
                    statuses=litellm_proxy_rate_limit_response["statuses"],
                )
            )

    except Exception as e:
        verbose_proxy_logger.exception("Error in rate limit post-call hook: %s", e)

    try:
        await handle_batch_enqueued_post_call(limiter, user_api_key_dict=user_api_key_dict, response=response)
    except Exception as e:  # noqa: BLE001  # post-call batch accounting must never fail the response
        verbose_proxy_logger.exception("Error in batch enqueued-token post-call hook: %s", e)
