"""
``RustCore``: the router core served by litellm/rust/crates/router-core through ``_native``.

The Rust core owns the registry, selection, retries, fallbacks and cooldowns. It reaches outside
itself only through its host, and the host never calls back into a Python router:

- every attempt runs through the host: ``router_attempt`` builds the request for the one
  deployment Rust picked and ``PythonHost.invoke(op)`` calls ``litellm.<op>`` with it
- the core records every outcome (cooldowns, allowed-fails counters, usage) itself, in the routing
  cache the host builds and owns (``PythonHost.routing_cache``), as far as the outcome callbacks the
  Python router would have running for the request reach (``OptionalChecks.outcome_callbacks``)
- sleeps, fallback access and budget checks, and cache reads and writes are host steps

Settings are resolved from the ``RouterConfig`` and the host's process settings; the views
(``registry()``, ``settings()``, ``health()``) are read from the Rust handle. A member the Rust
core does not serve raises ``RustCoreUnsupported`` (an ``AttributeError``).
"""

import asyncio
import contextlib
import dataclasses
import inspect
import itertools
import json
from collections.abc import Awaitable, Callable, Coroutine, Generator, Iterable, Mapping, Sequence
from contextvars import ContextVar
from datetime import date, datetime, time
from functools import partial
from time import perf_counter
from time import time as unix_time
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, TypeAlias, TypeVar, final, overload

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator
from typing_extensions import assert_never

from litellm._internal_context import request_parent_otel_span, service_target
from litellm.constants import DEFAULT_COOLDOWN_TIME_SECONDS
from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
    apply_quality_router_decision_headers,
    apply_remaining_usage_headers,
    apply_response_model_id,
    complexity_router_decision_headers,
    ensure_response_additional_headers,
    prepare_response_for_header_attachment,
)
from litellm.litellm_core_utils.agentic_followup_failure import is_agentic_followup_failure
from litellm.litellm_core_utils.attempt_settlement import (
    ATTEMPT_SETTLEMENT_GRACE_SECONDS,
    RoutedAttempt,
    released,
    routed_attempt,
)
from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type
from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    get_parent_otel_span_from_kwargs,
    without,
)
from litellm.litellm_core_utils.routing_common_utils import (
    compact_fallback_error_detail,
    format_content_policy_fallback_hint,
    format_context_window_fallback_hint,
    format_fallback_outcome_message,
    format_no_fallback_group_message,
    get_request_team_id,
    is_web_search_request,
    truncate_fallback_error_detail,
)
from litellm.litellm_core_utils.routing_groups import validate_routing_strategy
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.litellm_core_utils.total_deadline import (
    TotalTimeoutNotApplicable,
    current_total_deadline,
    enforce_total_timeout_scope,
)
from litellm.redis.stores.connection import connect
from litellm.redis.stores.deployment_health import DeploymentHealthStore
from litellm.redis.stores.router_cooldowns import RouterCooldownStore
from litellm.redis.stores.router_session_pins import RouterSessionPinStore
from litellm.redis.stores.router_usage import RouterUsage
from litellm.router_api import (
    ROUTER_INIT_ARGS,
    SETTING_FIELDS,
    ApplyResult,
    ChangeSet,
    RouterHealth,
    RouterRegistry,
    RouterSettings,
    RoutingContext,
    RoutingObservers,
)
from litellm.router_attempt import for_requested_model, request_from_call
from litellm.router_config import RouterConfig, RouterDynamicConfig, detached
from litellm.router_python_host import PythonHost
from litellm.router_rust.invoke.endpoints import (
    AttemptCall,
    Plan,
    RustEndpoints,
    arm_compaction_state,
    attempt_call,
    clean_realtime_session_deployment,
    compacted_attempt,
    endpoint_names,
    failover_fields,
    generic_plan,
    is_batch_upload,
    log_refused_call,
    log_refused_call_sync,
    pinned_vector_store_file_deployment,
    pinned_video_deployment,
    plan_for,
    reports_own_success,
    scoped_fallback_fields,
    selection_endpoint,
    stamp_error,
    unrouted_fields,
)
from litellm.router_rust.invoke.shaping import Shaping, non_stream_timeout
from litellm.router_rust.invoke.slots import enter_deployment_slot, max_parallel_requests_limit
from litellm.router_rust.invoke.streaming import (
    StreamAttempt,
    continuation_flag,
    continuation_routing,
    open_attempt_stream,
    open_attempt_stream_sync,
    register_continuation_prefill_check,
    shape_attempt_kwargs,
)
from litellm.router_rust.invoke.walk import FallbackOutcomes, WalkMetadata, await_all, log_retry_into
from litellm.router_rust.invoke.websocket import turns_counted_their_requests
from litellm.router_rust.routing_stores import (
    CACHE_STEP,
    CooldownsStep,
    CountFailureStep,
    CountSuccessStep,
    LocalStep,
    OutcomesStep,
    RoutingStores,
    service_target_of,
)
from litellm.router_rust.steps.access import caller_fields, pinned_deployment_wildcard_grant, selection_refusal
from litellm.router_rust.steps.callbacks import OptionalChecks, RustOptionalChecks
from litellm.router_rust.steps.plugins import SYNC_PATH_MESSAGE, candidate_models, run_plugins
from litellm.router_rust.steps.precall import (
    DECISIONS_MEDIA_CALL_TYPES,
    PreCallFacts,
    cannot_serve_route,
    declares_no_vision,
    refuses_decisions_media,
    request_carries_image,
    serves_route_capability,
)
from litellm.router_rust.steps.prerouting import PreRouter, PreRoutingStrategies, picks_async
from litellm.router_rust.steps.selection import (
    SELECTOR_ATTRIBUTES,
    StrategySelectors,
    group_documents,
    group_settings,
    member_map,
    normalize_strategy,
    parse_groups,
    pick_async,
    pick_sync,
    picked_id,
    routing_pre_call_checks,
)
from litellm.router_rust.surface.alerting import initialize_alerting
from litellm.router_rust.surface.fallbacks import (
    RustFallbacks,
    mock_error,
    mock_failure,
    refuse_malformed_fallbacks,
    rejected_fallback_target,
)
from litellm.router_rust.surface.health import (
    RustHealthCallbacks,
    failure_already_logged,
    failure_status,
    first_cooldown,
    logged_call,
)
from litellm.router_rust.surface.health_routing import (
    HealthRouting,
    failover_document,
    fully_unhealthy_model_names,
    resolved_allowed_fails_policy,
)
from litellm.router_rust.surface.health_routing import all_blocked as deployments_all_blocked
from litellm.router_rust.surface.health_routing import fully_blocked_model_names as blocked_model_names
from litellm.router_rust.surface.health_routing import is_blocked as deployment_is_blocked
from litellm.router_rust.surface.legacy_members import RustLegacyMembers
from litellm.router_rust.surface.members import RustRegistryMembers
from litellm.router_rust.surface.model_list import typed_deployment
from litellm.router_rust.surface.registry import (
    DEFAULT_DEPLOYMENT_COPIED_ID,
    DEFAULT_DEPLOYMENT_FAILING_ID,
    INDEX_ATTRIBUTES,
    BuildInstall,
    HostRegistry,
    RustRegistry,
    default_deployment_call_error,
    default_deployment_copy,
    default_deployment_copy_error,
    hosted_registry_view,
    index_document,
    served_default_deployment,
)
from litellm.router_rust.surface.replay import track_live_registry, untrack_live_registry
from litellm.router_rust.surface.retries import (
    GroupPolicies,
    RustRetries,
    group_policies,
    is_policy_per_group,
    policy_lookup_error,
    raising_groups,
    readable_retry_policy,
    reads_as_policy,
    request_retry_fields,
    resolved_retry_policy,
    retry_count_error,
    static_group_policies,
)
from litellm.router_rust.surface.settings import (
    MISSING,
    PLAIN_DEFAULTS,
    SettingsUpdate,
    apply_construction_effects,
    argument_attribute,
    default_litellm_params,
    initial_plain,
    read_settings_update,
    request_timeout,
    settings_view,
)
from litellm.router_strategy.least_busy import (
    CALL_IDENTITY,
    call_id_of,
    owning_picks,
    pick_reserves,
    picks_reserve,
    release_started_attempt,
)
from litellm.router_utils.cooldown_handlers import get_cooldown_deployment_info, mark_deployment_selection_failure
from litellm.router_utils.endpoint_eligibility import endpoint_for_call, selected_endpoint
from litellm.router_utils.kubernetes_pod_discovery import (
    KubernetesPodDiscovery,
    async_resolve_pods_after_bound,
    resolve_pods_after_bound,
)
from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import EncryptedContentAffinityCheck
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import release_io_token_reservation
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    find_deployment_metadata,
    get_deployment_failures_for_current_minute,
    get_deployment_successes_for_current_minute,
    increment_deployment_failures_for_current_minute,
    increment_deployment_successes_for_current_minute,
)
from litellm.router_utils.routing_read_batch import RoutingReadBatch
from litellm.types.router import (
    AllowedFailsPolicy,
    Deployment,
    RouterErrors,
    RouterGeneralSettings,
    RouterRateLimitError,
    RoutingGroup,
    RoutingStrategy,
)
from litellm.types.utils import CooldownDeploymentInfo, LlmProviders

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.caching.dual_cache import DualCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.router_api import UntypedDict
    from litellm.router_utils.cooldown_cache import CooldownCache
    from litellm.router_utils.health_state_cache import DeploymentHealthCache
    from litellm.rust_bridge._native import (
        RouterCall,
        RouterCoreHandle,
        _CallSteps,  # pyright: ignore[reportPrivateUsage]  # the stub's step shapes
        _FailureReportSteps,  # pyright: ignore[reportPrivateUsage]  # see above
        _SuccessReportSteps,  # pyright: ignore[reportPrivateUsage]  # see above
    )
    from litellm.types.router import (
        CustomRoutingStrategyBase,
        DeploymentTypedDict,
        GuardrailTypedDict,
        PreRoutingHookResponse,
        RoutingPlugin,
    )

_FAILURE_KINDS: Final = (
    "RateLimitError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "GuardrailIntervention",
    "CallerCredentialAuthenticationError",
    "CallerCredentialRateLimitError",
)
_DECLINED: Final = frozenset({"MalformedRequestFieldError", "UnsupportedEndpointError"})
_PROVIDERS: Final = tuple(provider.value for provider in LlmProviders)
_DEFAULT_COOLDOWN_SECONDS: Final = 5.0
# Past this a retry-after no longer fits the Rust core's durations; it cools down as good as forever.
_LONGEST_RETRY_AFTER_SECONDS: Final = 2**63

# Constructor arguments this core reads. Any other argument set to something other than its
# unset value keeps the router on the Python core while both exist (make_core decides that from
# the config alone, before any core is built).
_SERVED_ARGUMENTS: Final = frozenset(
    {
        "model_list",
        "routing_strategy",
        "routing_strategy_args",
        "routing_groups",
        "enable_tag_filtering",
        "tag_filtering_match_any",
        "tag_routing_prefix",
        "model_group_alias",
        "fallbacks",
        "default_fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "cooldown_time",
        "disable_cooldowns",
        "allowed_fails_policy",
        "retry_after",
        "timeout",
        "default_litellm_params",
        "fallback_access_check",
        "fallback_budget_check",
        "fallback_rate_limit_check",
        "enable_mid_stream_fallback_continuation",
        "redis_url",
        "redis_host",
        "redis_port",
        "redis_password",
        "redis_db",
        "cache_kwargs",
        "set_verbose",
        "debug_level",
        # Read host-side by the endpoint plans (litellm/router_rust/invoke/endpoints.py).
        "stream_timeout",
        "cache_responses",
        "assistants_config",
        "search_tools",
        "guardrail_list",
        "default_priority",
        "polling_interval",
        "ignore_invalid_deployments",
        # Read host-side by the pre-routing strategies (litellm/router_rust/steps/prerouting.py).
        "auto_router_capability_limit",
        # Decided in router-core (precall.rs); its facts come from litellm/router_rust/steps/precall.py.
        "enable_pre_call_checks",
        # Run host-side (litellm/router_rust/steps/plugins.py); router-core filters on what they leave.
        "plugins",
        "retry_policy",
        "model_group_retry_policy",
        # Health-check routing and the failover flags (litellm/router_rust/surface/health_routing.py).
        "enable_health_check_routing",
        "health_check_staleness_threshold",
        "health_check_ignore_transient_errors",
        "background_health_check_model_groups",
        "enable_weighted_failover",
        "fallback_on_unsupported_params",
        # Built host-side as CustomLogger checks (litellm/router_rust/steps/callbacks.py); router-core runs
        # their deployment filters through its callback_filters stage.
        "optional_pre_call_checks",
        "provider_budget_config",
        "deployment_affinity_ttl_seconds",
        "session_affinity_repin_on_failover",
        "model_group_affinity_config",
        # Kept as given (litellm/router_rust/surface/settings.py) and read host-side by the endpoint plans
        # (default_litellm_params' caching groups) and the slot limits.
        "client_ttl",
        "caching_groups",
        "default_max_parallel_requests",
        # Built host-side into a Slack alerting logger (litellm/router_rust/surface/alerting.py).
        "alerting_config",
        # Kept as given and reported by settings(); nothing in the router reads its fields.
        "router_general_settings",
    }
)
# Arguments that may be passed at their default without asking for anything.
_UNSET_DEFAULTS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "cache_responses": False,
        "enable_pre_call_checks": False,
        "enable_tag_filtering": False,
        "retry_policy": None,
        "model_group_retry_policy": {},
    }
)
# Settings a changeset may change. Static ones rebuild the Rust core (settings are fixed for a
# core's life); dynamic ones are handed to it as a new dynamic document.
_STATIC_SETTINGS: Final = frozenset(
    {
        "routing_strategy",
        "routing_groups",
        "cooldown_time",
        "disable_cooldowns",
        "allowed_fails_policy",
        "retry_after",
        "timeout",
        "enable_pre_call_checks",
    }
)
_DYNAMIC_SETTINGS: Final = frozenset(
    {
        "default_deployment",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "model_group_alias",
        "default_litellm_params",
        "fallback_access_check",
        "fallback_budget_check",
        "fallback_rate_limit_check",
        "routing_strategy_args",
        "enable_tag_filtering",
        "tag_filtering_match_any",
        "tag_routing_prefix",
    }
)
# Settings only the host reads (the endpoint plans, the slot limits, the budget limiter, the
# settings view). A changeset assigns them as given, the Rust core never sees them, and
# `update_settings` ignores them (none is runtime-updatable).
_HOST_SETTINGS: Final = frozenset(
    {
        "cache_responses",
        "search_tools",
        "guardrail_list",
        "default_max_parallel_requests",
        "provider_budget_config",
        "health_check_ignore_transient_errors",
        "router_general_settings",
        "provider_default_deployment_ids",
        "kubernetes_pod_discovery",
        # Read by router-core's health-check filter: assigning one rebuilds the Rust core.
        "enable_health_check_routing",
        "background_health_check_model_groups",
    }
)
# The constructor's default `router_general_settings`: one object every router built without the
# argument shares, as the Python core's default argument is.
_DEFAULT_GENERAL_SETTINGS: Final = RouterGeneralSettings()


# Plain settings `update_settings` sets that the Rust core's retry loop reads.
_RETRY_POLICY_SETTINGS: Final = ("retry_policy", "model_group_retry_policy")
# Attributes a `del` unsets beyond the setting fields, whose readers then raise AttributeError.
_DELETABLE_ATTRIBUTES: Final = frozenset({"model_group_alias", "num_retries", "scheduler"})


# Calls whose pick least-busy does not hold (`picks_reserve(False)`): a batch cancel makes no
# call that logs a start.
_UNRESERVED_OPS: Final = frozenset({"acancel_batch"})


# Settings converted by `Resolved._selection_setting`.
_SELECTION_SETTINGS: Final = frozenset(
    {
        "routing_strategy",
        "routing_strategy_args",
        "enable_tag_filtering",
        "tag_filtering_match_any",
        "tag_routing_prefix",
    }
)
_NUMBER_SETTINGS: Final = frozenset(
    {"num_retries", "max_fallbacks", "allowed_fails", "cooldown_time", "retry_after", "timeout"}
)
# The settings `_core_update_settings` casts with `int(value)`.
_UPDATE_INT_SETTINGS: Final = frozenset({"timeout", "num_retries", "retry_after", "allowed_fails", "cooldown_time"})


def _raised_before_provider(error: BaseException) -> bool:
    """A request litellm refused before it reached the provider (a parameter the provider does not
    take, a malformed priority): neither a failure nor a request of the deployment it picked."""
    from litellm.exceptions import UnsupportedParamsError

    body: Final = getattr(error, "body", None)
    return isinstance(error, UnsupportedParamsError) or (isinstance(body, Mapping) and body.get("param") == "priority")


def _shares_logging(request: Mapping[str, object]) -> bool:
    """The caller passed the logging object every attempt of the request logs on, which logs one
    success a request (``should_run_logging``)."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    return isinstance(request.get("litellm_logging_obj"), Logging)


class RustCoreUnsupported(AttributeError):
    """A member or request the Rust core does not serve. It is never forwarded to a Python core."""


# --- failures -----------------------------------------------------------------------------------


def failure_kind(error: BaseException) -> str:
    names: Final = tuple(cls.__name__ for cls in type(error).__mro__)
    if not _DECLINED.isdisjoint(names):
        return "Declined"
    if is_guardrail_intervention(error):  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
        return "GuardrailIntervention"
    return next((name for name in names if name in _FAILURE_KINDS), "APIError")


def retry_after_seconds(error: BaseException) -> float | None:
    """The header cooldown of ``deployment_callback_on_failure``: ``retry-after`` alone, as a whole
    number or a date, with a negative or unreadable one read as none."""
    from litellm.litellm_core_utils.exception_mapping_utils import (
        _get_response_headers,  # pyright: ignore[reportPrivateUsage]  # the failure callback's own read
    )
    from litellm.utils import _get_retry_after_from_exception_header  # pyright: ignore[reportPrivateUsage]  # see above

    headers: Final = _get_response_headers(original_exception=error)  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
    if headers is None:
        return None
    seconds: Final = _get_retry_after_from_exception_header(response_headers=headers)  # pyright: ignore[reportUnknownVariableType]  # an untyped helper's int or None
    return float(min(seconds, _LONGEST_RETRY_AFTER_SECONDS)) if seconds is not None and seconds >= 0 else None


def attempt_failure(error: BaseException) -> Mapping[str, object]:
    status: Final = getattr(error, "status_code", None)
    num_retries: Final = getattr(error, "num_retries", None)
    # `_deployment_ids_to_skip_on_retry` reads the id off the exception.
    named_deployment: Final = getattr(error, "retry_skip_deployment_id", None) or getattr(
        error, "failed_deployment_id", None
    )
    return {
        "ok": False,
        "kind": failure_kind(error),
        "status": status if isinstance(status, int) and 100 <= status <= 599 else None,
        "status_absent": getattr(error, "status_code", "") == "",
        "retry_after": retry_after_seconds(error),
        "num_retries": num_retries if isinstance(num_retries, int) and num_retries >= 0 else None,
        "failed_deployment_id": named_deployment if isinstance(named_deployment, str) else None,
        "detail": str(error),
        "classes": [cls.__name__ for cls in type(error).__mro__],
        "llm_provider": bool(getattr(error, "llm_provider", "")),
        "agentic_followup": is_agentic_followup_failure(error),
    }


def total_tokens(response: object) -> int:
    usage: Final = getattr(response, "usage", None)
    tokens: Final = getattr(usage, "total_tokens", None)
    return tokens if isinstance(tokens, int) else 0


# --- settings -----------------------------------------------------------------------------------


def _json_default(value: object) -> object:
    # ISO 8601, which ``typed_deployment`` reads back as the value it was.
    if isinstance(value, date | time):
        return value.isoformat()
    dump: Final = getattr(value, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else str(value)


def _plain(value: object) -> object:
    """``value`` as plain JSON values, the shape the Rust core reads."""
    loaded: Final[object] = json.loads(json.dumps(value, default=_json_default))  # pyright: ignore[reportAny]  # plain JSON
    return loaded


def _document(value: object) -> Mapping[str, object]:
    """A mapping as the plain JSON document the Rust core reads."""
    return as_str_mapping(_plain(value)) or {}


def _as_list(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()  # pyright: ignore[reportUnknownVariableType]  # a list's items are read as objects


def _strategy_name(value: object) -> str | None:
    """A strategy as a setting stores it: ``None`` stays no strategy (``_normalize_strategy``)."""
    return None if value is None else normalize_strategy(value) or "simple-shuffle"


_U32_MAX: Final = 2**32 - 1


def _crate_count(value: int | None) -> int | None:
    """A count setting as the crate's u32 reads it: a negative one is 0, a huge one the most it holds."""
    return None if value is None else min(max(value, 0), _U32_MAX)


def _lenient_int(value: object, default: int) -> int:
    """A count setting kept as given, as the crate reads it: an int, or a number spelled another way
    (``1.5``, ``"3"``) truncated to one; ``default`` when it is no number. The Python core stores the
    value as given and never checks it at construction."""
    if isinstance(value, bool | int):
        return int(value)
    try:
        return int(float(str(value)))
    except (ValueError, OverflowError):
        return default


def _lenient_float(value: object, default: float | None) -> float | None:
    """A seconds setting kept as given, as the crate reads it (see ``_lenient_int``)."""
    if isinstance(value, bool | int | float):
        return float(value)
    try:
        return float(str(value))
    except (ValueError, OverflowError):
        return default


# Settings the Python core stores as given, so that a read returns the value passed (``"1"``,
# ``None``, ``[1]``); ``Resolved`` keeps them in ``given`` and the typed fields hold what the crate reads.
_KEPT_AS_GIVEN: Final = frozenset(
    {
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "retry_after",
        "timeout",
        "disable_cooldowns",
        "enable_tag_filtering",
        "tag_filtering_match_any",
        "tag_routing_prefix",
        "routing_strategy_args",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Resolved:
    """The router's settings, resolved from its arguments and the process settings the way the
    Python router's constructor resolves them. Replaced, never changed, by ``apply``."""

    routing_strategy: str | None
    routing_strategy_args: Mapping[str, object]
    num_retries: int
    max_fallbacks: int
    allowed_fails: int
    explicit_allowed_fails: int | None
    cooldown_time: float
    disable_cooldowns: bool | None
    allowed_fails_policy: object
    retry_after: float
    timeout: float | None
    fallbacks: list[object] | None  # mutable-ok: the setting as Python spells it
    context_window_fallbacks: list[object] | None  # mutable-ok: same
    content_policy_fallbacks: list[object] | None  # mutable-ok: same
    model_group_alias: Mapping[str, object]
    default_litellm_params: Mapping[str, object]
    fallback_access_check: object
    fallback_budget_check: object
    fallback_rate_limit_check: object
    enable_mid_stream_fallback_continuation: bool = False
    routing_groups: "tuple[RoutingGroup, ...]" = ()
    enable_tag_filtering: bool = False
    tag_filtering_match_any: bool = True
    tag_routing_prefix: str = ""
    enable_pre_call_checks: bool = False
    routing_plugins: "tuple[RoutingPlugin, ...]" = ()
    retry_policy: object = None
    model_group_retry_policy: object = None
    health_routing: HealthRouting = dataclasses.field(default_factory=HealthRouting)
    # `_core_default_deployment`, as assigned (detached).
    default_deployment: object = None
    # `_model_to_group` as the Python core last built it, kept once a raw `routing_groups` write
    # replaced the groups without rebuilding it.
    routing_group_members: Mapping[str, str] | None = None
    # `_routing_groups_input`: the groups last passed to the constructor or `update_settings`, which
    # a strategy change through `update_settings` re-inits the routing groups from.
    routing_groups_input: object = None
    assigned: frozenset[str] = frozenset()
    # The `_KEPT_AS_GIVEN` settings as the Python core stores them, for reads.
    given: Mapping[str, object] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def read(self, name: str) -> object:
        """A setting as the Python core reads it back: as given when it keeps it so."""
        return self.given[name] if name in self.given else getattr(self, name)  # pyright: ignore[reportAny]  # a setting field

    @classmethod
    def of(cls, config: RouterConfig, dynamic: RouterDynamicConfig) -> "Resolved":
        get: Final = config.get
        # The process list is the router's own, as in the Python router: default_fallbacks are
        # appended to it in place.
        configured: Final[list[object] | None] = (  # mutable-ok: the router's live list, as Python keeps it
            list(_as_list(get("fallbacks"))) or (dynamic.fallbacks if dynamic.fallbacks else None)  # pyright: ignore[reportAssignmentType]  # the process list as given
        )
        refuse_malformed_fallbacks(configured)
        default_fallbacks: Final = get("default_fallbacks") or dynamic.default_fallbacks
        if isinstance(default_fallbacks, str):
            raise TypeError(
                f"default_fallbacks must be a list of model names, got '{default_fallbacks}'. Use ['{default_fallbacks}']"
            )
        if default_fallbacks is not None and configured is not None:
            configured.append({"*": default_fallbacks})
        fallbacks: Final[list[object] | None] = (  # mutable-ok: see above
            [{"*": default_fallbacks}] if default_fallbacks is not None and configured is None else configured
        )
        content_policy: Final = list(_as_list(get("content_policy_fallbacks"))) or (
            list(dynamic.content_policy_fallbacks) if dynamic.content_policy_fallbacks else None
        )
        refuse_malformed_fallbacks(content_policy)
        num_retries: Final = get("num_retries")
        max_fallbacks: Final = get("max_fallbacks")
        allowed_fails: Final = get("allowed_fails")
        timeout: Final = get("timeout")
        strategy: Final = get("routing_strategy")
        validate_routing_strategy(strategy)  # pyright: ignore[reportArgumentType]  # validated as given
        # The values the Python core's constructor stores (`self.num_retries = num_retries`, ...).
        given_num_retries: Final[object] = (
            num_retries if num_retries is not None else dynamic.num_retries if dynamic.num_retries is not None else 2
        )
        given_max_fallbacks: Final[object] = (
            max_fallbacks
            if max_fallbacks is not None
            else dynamic.max_fallbacks
            if dynamic.max_fallbacks is not None
            else dynamic.router_max_fallbacks
        )
        given_allowed_fails: Final[object] = allowed_fails if allowed_fails is not None else dynamic.allowed_fails
        given_timeout: Final[object] = timeout or dynamic.request_timeout
        given_retry_after: Final[object] = get("retry_after") if "retry_after" in config.arguments else 0
        kept: Final = {
            "num_retries": given_num_retries,
            "max_fallbacks": given_max_fallbacks,
            "allowed_fails": given_allowed_fails,
            "timeout": given_timeout,
            "retry_after": given_retry_after,
            "disable_cooldowns": get("disable_cooldowns"),
            **{
                name: get(name)
                for name in ("enable_tag_filtering", "tag_filtering_match_any", "tag_routing_prefix")
                if name in config.arguments
            },
            **(
                {"routing_strategy_args": get("routing_strategy_args")}
                if "routing_strategy_args" in config.arguments
                else {}
            ),
        }
        return cls(
            # An explicit `routing_strategy=None` is stored as None (no strategy picks).
            routing_strategy=None
            if strategy is None and "routing_strategy" in config.arguments
            else normalize_strategy(strategy) or "simple-shuffle",
            # A plain dict, as `self.routing_strategy_args`: `/router/settings` serializes it as read.
            routing_strategy_args=dict(as_str_mapping(get("routing_strategy_args")) or {}),
            num_retries=_lenient_int(given_num_retries, 2),
            max_fallbacks=_lenient_int(given_max_fallbacks, dynamic.router_max_fallbacks),
            allowed_fails=_lenient_int(given_allowed_fails, dynamic.allowed_fails),
            explicit_allowed_fails=_lenient_int(allowed_fails, dynamic.allowed_fails)
            if allowed_fails is not None
            else None,
            cooldown_time=_lenient_float(get("cooldown_time") or _DEFAULT_COOLDOWN_SECONDS, _DEFAULT_COOLDOWN_SECONDS)
            or 0.0,
            disable_cooldowns=_optional_bool(get("disable_cooldowns")),
            allowed_fails_policy=resolved_allowed_fails_policy(get("allowed_fails_policy")),
            retry_after=_lenient_float(given_retry_after or 0, 0.0) or 0.0,
            timeout=_lenient_float(given_timeout, None) if given_timeout else None,
            given=MappingProxyType(kept),
            fallbacks=fallbacks,
            context_window_fallbacks=list(_as_list(get("context_window_fallbacks")))
            or (list(dynamic.context_window_fallbacks) if dynamic.context_window_fallbacks else None),
            content_policy_fallbacks=content_policy,
            model_group_alias=MappingProxyType(dict(as_str_mapping(get("model_group_alias")) or {})),
            default_litellm_params=MappingProxyType(dict(as_str_mapping(get("default_litellm_params")) or {})),
            fallback_access_check=get("fallback_access_check"),
            fallback_budget_check=get("fallback_budget_check"),
            fallback_rate_limit_check=get("fallback_rate_limit_check"),
            enable_mid_stream_fallback_continuation=continuation_flag(get("enable_mid_stream_fallback_continuation")),
            enable_tag_filtering=bool(get("enable_tag_filtering")),
            tag_filtering_match_any=_true_unless_given(get("tag_filtering_match_any")),
            tag_routing_prefix=str(get("tag_routing_prefix") or ""),
            enable_pre_call_checks=bool(get("enable_pre_call_checks")),
            routing_plugins=tuple(_as_list(get("plugins"))),  # pyright: ignore[reportArgumentType]  # the plugins as passed
            routing_groups_input=get("routing_groups"),
            routing_groups=parse_groups(
                get("routing_groups"),
                frozenset(
                    str(name)
                    for entry in _as_list(get("model_list"))
                    if (name := (as_str_mapping(entry) or {}).get("model_name"))
                ),
                frozenset(as_str_mapping(get("model_group_alias")) or {}),
            ),
            retry_policy=resolved_retry_policy(get("retry_policy")),
            model_group_retry_policy=get("model_group_retry_policy"),
            health_routing=HealthRouting.of(config),
        )

    def changed(self, values: Mapping[str, object], updated: frozenset[str] = frozenset()) -> "Resolved":
        """``values`` applied; ``updated`` names those passed as ``ChangeSet.settings`` (which go
        through ``update_settings``: only they may name ``routing_strategy_args``)."""
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        unknown: Final = sorted(
            set(values)
            - {*SETTING_FIELDS, *PLAIN_SETTING_ATTRIBUTES, *ASSIGNABLE_ALIASES}
            - (updated & {"routing_strategy_args", "routing_groups", "enable_tag_filtering", "tag_routing_prefix"})
        )
        if unknown:
            raise ValueError(f"not assignable router settings: {unknown}")
        unserved: Final = sorted(set(values) - _STATIC_SETTINGS - _DYNAMIC_SETTINGS)
        if unserved:
            raise RustCoreUnsupported(f"the Rust router core does not serve the setting(s) {', '.join(unserved)} yet")
        updates: Final[dict[str, object]] = {}  # mutable-ok: built here, then applied once
        for name, value in values.items():
            if name in {"model_group_alias", "default_litellm_params"}:
                updates[name] = MappingProxyType(dict(as_str_mapping(value) or {}))
            elif name in _NUMBER_SETTINGS:
                # `allowed_fails` saved later is no explicit one: only the constructor's argument is
                # (`explicit_allowed_fails`), so a value equal to the default keeps the rate-based default.
                updates[name] = self._number_setting(name, value)
            elif name in _SELECTION_SETTINGS:
                updates[name] = self._selection_setting(name, value)
            elif name == "disable_cooldowns":
                updates[name] = _optional_bool(value)
            elif name == "enable_pre_call_checks":
                updates[name] = bool(value)
            elif name in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks"):
                updates[name] = list(_as_list(value)) or None
            elif name == "default_deployment":
                updates[name] = detached(value)
            else:
                updates[name] = value
        if "routing_groups" in values:
            # `_init_routing_groups` rebuilds the member map from the new groups.
            updates["routing_group_members"] = None
        # `_core_update_settings` casts its int settings (`int(value)`, which raises as Python's
        # does); every other write is stored as given.
        kept: Final = {
            name: int(value) if name in updated and name in _UPDATE_INT_SETTINGS else value  # pyright: ignore[reportArgumentType]  # int() of the value as given, as Python casts it
            for name, value in values.items()
            if name in _KEPT_AS_GIVEN
        }
        return dataclasses.replace(
            self,
            **updates,
            assigned=self.assigned | frozenset(values),
            given=MappingProxyType({**self.given, **kept}),
        )  # pyright: ignore[reportArgumentType]  # each value converted to its field's type above

    def _number_setting(self, name: str, value: object) -> object:
        """A numeric setting as the crate reads it; the value as given is kept in ``given``."""
        if name in ("num_retries", "max_fallbacks", "allowed_fails"):
            return _lenient_int(value, getattr(self, name))  # pyright: ignore[reportAny]  # the field's current int
        if name == "cooldown_time":
            # Stored as given (`_core_cooldown_time`), so a failure cools down for that long:
            # 0 for no time at all; None leaves the cooldown cache's own default, set by the caller.
            return _lenient_float(value, self.cooldown_time) if value is not None else self.cooldown_time
        if name == "retry_after":
            return _lenient_float(value or 0, self.retry_after)
        return _lenient_float(value, self.timeout) if value is not None else None

    def _selection_setting(self, name: str, value: object) -> object:
        """A selection setting converted to its field's type."""
        if name == "routing_strategy":
            return _strategy_name(value)
        if name == "routing_strategy_args":
            return dict(as_str_mapping(value) or {})
        if name == "tag_routing_prefix":
            return str(value or "")
        return bool(value)

    def static_document(self, model_list: Sequence[object]) -> Mapping[str, object]:
        values: Final = {
            "model_list": _plain(list(model_list)),
            "routing_strategy": self.routing_strategy,
            "cooldown_time": self.cooldown_time,
            "disable_cooldowns": self.disable_cooldowns,
            # A policy that is no policy or mapping (an attribute write of a list) is none, as the
            # Python core's `get_allowed_fails_from_policy` reads no field off it.
            "allowed_fails_policy": _plain(self.allowed_fails_policy) if self._has_fails_policy() else None,
            "retry_after": self.retry_after,
            "timeout": self.timeout,
            "routing_groups": group_documents(self.routing_groups) if self.routing_groups else None,
            "enable_pre_call_checks": True if self.enable_pre_call_checks else None,
            "routing_plugins": True if self.routing_plugins else None,
            # `async_callback_filter_deployments` runs on every async candidate list.
            "callback_filters": True,
            "retry_policy": _plain(self.retry_policy),
            "model_group_retry_policy": _plain(static_group_policies(self.model_group_retry_policy)) or None,
            "routing_group_members": dict(self.routing_group_members)
            if self.routing_group_members is not None
            else None,
            **self.health_routing.static_document(),
        }
        kept: Final = {name: value for name, value in values.items() if value is not None}
        # An explicit `routing_strategy=None` reaches the Rust core as "None": no strategy picks.
        return kept if self.routing_strategy is not None else {**kept, "routing_strategy": "None"}

    def _has_fails_policy(self) -> bool:
        return isinstance(self.allowed_fails_policy, AllowedFailsPolicy | Mapping)

    def dynamic_document(self, version: int, model_names: Sequence[str]) -> Mapping[str, object]:
        values: Final = {
            "version": version,
            "model_group_alias": _plain(dict(self.model_group_alias)),
            "fallbacks": _plain(self.fallbacks),
            "context_window_fallbacks": _plain(self.context_window_fallbacks),
            "content_policy_fallbacks": _plain(self.content_policy_fallbacks),
            # A negative count reads as none at all in the Python core (`range(num_retries)`,
            # `fallback_depth >= max_fallbacks`, `fails > allowed_fails`); the crate reads a u32.
            "num_retries": _crate_count(self.num_retries),
            "max_fallbacks": _crate_count(self.max_fallbacks),
            "allowed_fails": _crate_count(self.allowed_fails),
            "allowed_fails_is_default": True if self.allowed_fails_is_default() else None,
            "fallback_keys": self._fallback_keys(model_names),
            "default_deployment": _plain(served_default_deployment(self.default_deployment)[0]),
            "default_deployment_raises": True if served_default_deployment(self.default_deployment)[1] else None,
        }
        return {name: value for name, value in values.items() if value is not None}

    def allowed_fails_is_default(self) -> bool:
        """``allowed_fails`` does not count as set on the router (``_is_allowed_fails_set_on_router``):
        not passed explicitly, and the value copied at init still equals the live process default."""
        import litellm

        return self.explicit_allowed_fails is None and self.allowed_fails == litellm.allowed_fails

    def _fallback_keys(self, model_names: Sequence[str]) -> Mapping[str, object]:
        """The spellings a fallback key matches besides the group name (`get_fallback_model_group`)."""
        from litellm.router_utils.fallback_event_handlers import (
            _provider_prefixed_model_group,  # pyright: ignore[reportPrivateUsage]  # the fallback walk's own spelling rule
        )

        keys: Final = tuple(  # comprehension-ok: the first group of each fallback entry, across the three lists
            str(next(iter(mapping)))
            for entries in (self.fallbacks, self.context_window_fallbacks, self.content_policy_fallbacks)
            for entry in _as_list(entries)
            if (mapping := as_str_mapping(entry))
        )
        # A bare name a wildcard serves is in no list of names: the keys' own bare spellings stand for it.
        bare: Final = (key.split("/", 1)[1] for key in keys if "/" in key)
        names: Final = {*model_names, *self.model_group_alias, *bare}
        prefixed: Final = {
            name: spelled for name in names if (spelled := _provider_prefixed_model_group(name, keys)) is not None
        }
        from litellm.types.router import LiteLLMParamsTypedDict

        return {
            "providers": _PROVIDERS,
            "provider_prefixed": prefixed,
            "param_keys": list(LiteLLMParamsTypedDict.__annotations__),
        }


def _alias_target(target: object) -> str | None:
    mapping: Final = as_str_mapping(target)
    if mapping is not None:
        model: Final = mapping.get("model")
        return model if isinstance(model, str) else None
    return target if isinstance(target, str) else None


def _true_unless_given(value: object) -> bool:
    return True if value is None else bool(value)


def _group_selectors(groups: "Sequence[RoutingGroup]") -> Sequence[tuple[str, str, Mapping[str, object]]]:
    return [
        (
            group.group_name,
            group.routing_strategy,
            as_str_mapping(group.routing_strategy_args) or {},  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # RoutingGroup's untyped dict
        )
        for group in groups
    ]


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def unsupported_arguments(config: RouterConfig) -> frozenset[str]:
    """Arguments the Rust core does not serve."""
    unserved: Final = frozenset(
        name
        for name, value in config.arguments.items()
        if name not in _SERVED_ARGUMENTS and value not in (None, _UNSET_DEFAULTS.get(name))
    )
    return unserved


def _listing_team_id(request: Mapping[str, object]) -> str | None:
    """``(kwargs.get("metadata", {}) or {}).get("user_api_key_team_id")``: the team
    ``async_function_with_fallbacks_common_utils`` lists a group for reads this bucket alone."""
    metadata: Final = as_str_mapping(request.get("metadata"))
    team_id: Final = metadata.get("user_api_key_team_id") if metadata is not None else None
    return team_id if isinstance(team_id, str) else None


def _strings(value: object) -> "tuple[str, ...]":
    """Every string in ``value``, a name or a fallback list of names and dict entries."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        entries: Final[Mapping[object, object]] = value  # pyright: ignore[reportUnknownVariableType]  # read as objects
        return tuple(itertools.chain.from_iterable(map(_strings, itertools.chain.from_iterable(entries.items()))))
    if isinstance(value, Sequence):
        items: Final[Sequence[object]] = value  # pyright: ignore[reportUnknownVariableType]  # see above
        return tuple(itertools.chain.from_iterable(map(_strings, items)))  # pyright: ignore[reportUnknownArgumentType]  # read as objects
    return ()


def _routed_model(request: Mapping[str, object]) -> object:
    """The request's ``model`` as a recovery of its stream names it (``RustCore.routed_name``)."""
    model: Final = request.get("model")
    return model if model is None or isinstance(model, str) else str(model)


class RequestStartFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    fallbacks: Sequence[str | Mapping[str, object]] | None = None
    context_window_fallbacks: Sequence[str | Mapping[str, object]] | None = None
    content_policy_fallbacks: Sequence[str | Mapping[str, object]] | None = None
    max_fallbacks: Annotated[int, Field(ge=0, le=_U32_MAX)] | None = None
    specific_deployment: bool = False
    disable_fallbacks: bool = False
    cooldown_time: object | None = None
    router_weights: object | None = Field(default=None, validation_alias="_router_weights")
    team_id: str | None = None
    listing_team_id: str | None = None
    routing_strategy: str | None = None

    @model_validator(mode="before")
    @classmethod
    def ignore_standalone_fallback_strings(cls, value: object) -> object:
        request_fields: Final = as_str_mapping(value)
        if request_fields is None:
            return value
        return {
            key: item
            for key, item in request_fields.items()
            if key not in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks")
            or not isinstance(item, str)
        }

    @field_validator("specific_deployment", "disable_fallbacks", mode="before")
    @classmethod
    def literal_true_only(cls, value: object) -> bool:
        return value is True


def _request_start_fields(request: Mapping[str, object]) -> RequestStartFields:
    strategy: Final = request.get("routing_strategy")
    override: Final = normalize_strategy(strategy) if isinstance(strategy, str | RoutingStrategy) else None
    fields: Final = {
        **request,
        "team_id": get_request_team_id(request),
        "listing_team_id": _listing_team_id(request),
        "routing_strategy": override,
    }
    try:
        return RequestStartFields.model_validate(fields)
    except ValidationError:
        import litellm.exceptions as litellm_exceptions

        model: Final = request.get("model")
        raise litellm_exceptions.BadRequestError(
            message="Invalid router request fields",
            model=model if isinstance(model, str) else None,
            llm_provider="",
        ) from None


def _start_fields(request: Mapping[str, object]) -> Mapping[str, object]:
    """The request fields the Rust core routes on."""
    normalized: Final = _request_start_fields(request)
    excluded: Final = {
        name: True
        for name, omit in (
            ("specific_deployment", not normalized.specific_deployment),
            ("disable_fallbacks", not normalized.disable_fallbacks),
            ("cooldown_time", normalized.cooldown_time is None),
            ("router_weights", normalized.router_weights is None),
            ("team_id", not normalized.team_id),
            ("listing_team_id", normalized.listing_team_id is None),
            ("routing_strategy", normalized.routing_strategy is None),
        )
        if omit
    }
    request_fields: Final = normalized.model_dump(exclude_unset=True, exclude=excluded)
    return {
        **request_retry_fields(request.get("num_retries")),
        **request_fields,
        **caller_fields(request),
    }


# --- views --------------------------------------------------------------------------------------


class RustSettings(RouterSettings):
    """``settings()``: the resolved settings as a frozen snapshot."""

    def guardrail(self, name: str) -> "GuardrailTypedDict":
        router: Final = self._router
        if not isinstance(router, _SettingsSource):
            return super().guardrail(name)
        return router.available_guardrail(name)  # pyright: ignore[reportReturnType]  # the configured guardrail entry

    def as_dict(self) -> Mapping[str, object]:
        router: Final = self._router
        if not isinstance(router, _SettingsSource):
            return super().as_dict()
        return router.settings_dict()


class _SettingsSource:
    def settings_dict(self) -> Mapping[str, object]:
        raise NotImplementedError

    def available_guardrail(self, name: str) -> object:
        raise NotImplementedError


class RustHealth(RouterHealth):
    """``health()``: cooldowns, fail counts and usage, read through the host's routing cache and
    written through the Rust core."""

    __slots__ = ("_core",)

    def __init__(self, core: "RustCore") -> None:  # pyright: ignore[reportMissingSuperCall]  # reads the Rust core, not a Router
        self._core = core

    def __getattribute__(self, name: str) -> object:
        if name in _HEALTH_SERVED or name.startswith("__") or name == "_core":
            found: Final[object] = object.__getattribute__(self, name)  # pyright: ignore[reportAny]  # the view's own member
            return found
        raise RustCoreUnsupported(f"health().{name} is not served by the Rust router core", name=name, obj=self)

    @property
    def store(self) -> "DualCache":
        return self._core._cache  # pyright: ignore[reportPrivateUsage]  # the health view reads its own core

    @property
    def cooldowns(self) -> "CooldownCache":
        return self._core._cooldowns  # pyright: ignore[reportPrivateUsage]  # see above

    @property
    def session_pins(self) -> RouterSessionPinStore:
        return self._core._session_pins  # pyright: ignore[reportPrivateUsage]  # see above

    def failures_this_minute(self, deployment_id: str) -> int:
        count: Final[object] = self.store.get_cache(f"{deployment_id}:fails", local_only=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache's reads are untyped
        return int(count) if isinstance(count, int | float) else 0

    def record_failure(self, deployment_id: str, exception: Exception, exception_status: str | int) -> bool:
        self._core._sync_allowed_fails()  # pyright: ignore[reportPrivateUsage]  # see above
        failure: Final = {
            **attempt_failure(exception),
            "status": failure_status(exception_status),
            "caller_scoped": self._core.per_user_deployment(deployment_id),
        }
        handle: Final = self._core._handle  # pyright: ignore[reportPrivateUsage]  # see above
        call: Final = handle.report_failure(deployment_id, failure, count_only=exception_status == "")
        done: Final = self._core._drive_sync_steps(call)  # pyright: ignore[reportPrivateUsage]  # see above
        return self._core.cooled_down(done, deployment_id, exception_status)

    def team_ptu_ceiling(self, team_id: str, model_group: str) -> "PTUTeamCeiling | None":
        return self._core._hosted._core_team_ptu_ceiling(team_id, model_group)  # pyright: ignore[reportPrivateUsage]  # the host registry's ported ceiling

    def split_priority(self, model: str, kwargs: Mapping[str, object]) -> tuple[int | None, Mapping[str, object]]:
        """``_core_split_request_priority``, as the endpoints resolve a request's priority."""
        return self._core._endpoints._split_priority(model, kwargs)  # pyright: ignore[reportPrivateUsage]  # see store

    def priority_header_fields(
        self, model: str, kwargs: Mapping[str, object], header: str | None
    ) -> Mapping[str, int | str]:
        """``priority_header_update``, as the endpoints read a priority header."""
        return self._core._endpoints.priority_header_update(model=model, kwargs=kwargs, header=header)  # pyright: ignore[reportPrivateUsage]  # see store

    @property
    def health_checks(self) -> "DeploymentHealthCache":
        return self._core._health_checks  # pyright: ignore[reportPrivateUsage]  # see store

    @property
    def latency(self) -> "UntypedDict":  # the live latency map, as the Python core keeps it
        return self._core._hosted._core_deployment_latency_map  # pyright: ignore[reportPrivateUsage]  # see team_ptu_ceiling

    @staticmethod
    def is_blocked(deployment: "Deployment") -> bool:
        return deployment_is_blocked(deployment)

    @staticmethod
    def all_blocked(deployments: "Sequence[DeploymentTypedDict]") -> bool:
        return deployments_all_blocked(deployments)

    def fully_blocked_model_names(self) -> set[str]:  # mutable-ok: the core's type, a fresh set
        return blocked_model_names(self._core._hosted._core_get_model_list() or [])  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # see team_ptu_ceiling

    async def fully_unhealthy_model_names(self) -> set[str]:  # mutable-ok: the core's type, a fresh set
        core: Final = self._core
        return await fully_unhealthy_model_names(
            core._hosted._core_get_model_list() or [],  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # see team_ptu_ceiling
            core._health_checks,  # pyright: ignore[reportPrivateUsage]  # see store
            core._resolved.allowed_fails_policy,  # pyright: ignore[reportPrivateUsage]  # see store
        )

    @overload
    async def usage(
        self, model_group: str, *, kind: Literal["current"] = "current"
    ) -> tuple[int | None, int | None]: ...

    @overload
    async def usage(self, model_group: str, *, kind: Literal["remaining"]) -> Mapping[str, int]: ...

    async def usage(  # pyright: ignore[reportIncompatibleMethodOverride]  # the same two kinds
        self, model_group: str, *, kind: Literal["current", "remaining"] = "current"
    ) -> tuple[int | None, int | None] | Mapping[str, int]:
        """The group's usage this minute, read from the routing cache over the host registry's
        live model list, as ``get_model_group_usage`` / ``get_remaining_model_group_usage`` read it."""
        hosted: Final = self._core._hosted  # pyright: ignore[reportPrivateUsage]  # see above
        if kind == "current":
            return await hosted._core_get_model_group_usage(model_group)  # pyright: ignore[reportPrivateUsage]  # the ported usage read
        if kind == "remaining":
            return await hosted._core_get_remaining_model_group_usage(model_group)  # pyright: ignore[reportPrivateUsage]  # see above
        raise ValueError(f"unknown usage kind {kind!r}")


# Deprecated attribute writes (``router.<name> = ...`` reaches the core as ``_core_<name>``) of
# registry state this core keeps on its host registry, and of routing state it rebinds.
_HOSTED_STATE: Final = frozenset(
    {
        "_core_model_names",
        "_core_deployment_names",
        "_core_team_public_model_names",
        "_core_model_name_to_deployment_indices",
        "_core_team_model_to_deployment_indices",
        "_core_cached_deployment_model_info",
        "_core_pattern_router",
        "_core_team_pattern_routers",
        "_core_routing_groups",
        "_core_auto_routers",
        "_core_complexity_routers",
        "_core_quality_routers",
        "_core_switchyard_routers",
        "_core_adaptive_routers",
        "_core_routing_plugins",
        "_core_deployment_latency_map",
    }
)
_REBOUND_STATE: Final = MappingProxyType(
    {
        "_core_cache": "_cache",
        "_core_cooldown_cache": "_cooldowns",
        "_core_session_pins": "_session_pins",
        "_core_health_state_cache": "_health_checks",
        "_core_slack_alerting_logger": "_slack_alerting_logger",
    }
)


_HEALTH_SERVED: Final = frozenset(
    {
        "store",
        "cooldowns",
        "session_pins",
        "failures_this_minute",
        "record_failure",
        "usage",
        "team_ptu_ceiling",
        "split_priority",
        "priority_header_fields",
        "health_checks",
        "latency",
        "is_blocked",
        "all_blocked",
        "fully_blocked_model_names",
        "fully_unhealthy_model_names",
    }
)


# --- the core -----------------------------------------------------------------------------------


_NO_INPUT: Final = object()
_EndpointImpl: TypeAlias = Callable[..., Awaitable[object]]
# A strategy select step's strategy, selector, keyed name and candidate deployments.
_StrategyInputs: TypeAlias = tuple[str, object, str, list[dict[str, object]]]


def _pick_answer(
    step: Mapping[str, object], candidates: Sequence[object], picked: object
) -> list[object]:  # mutable-ok: a JSON list
    """A strategy step's answer: the picked id, or its position when the candidates share ids (the
    step then carries ``records``), so the pick serves the entry it chose as Python's list does."""
    found: Final = picked_id(picked)
    if step.get("records") is None or found is None:
        return [found] if found is not None else []
    # The entry the selector returned, else (a selector that returned a copy) the first with its id.
    position: Final = next((index for index, record in enumerate(candidates) if record is picked), None)
    if position is not None:
        return [position]
    return [next((index for index, record in enumerate(candidates) if picked_id(record) == found), 0)]


@final
class _PickCounts(Protocol):
    """The router cache as ``track_deployment_pick`` uses it: each deployment id holds its pick count."""

    def get_cache(
        self, key: object, parent_otel_span: "Span | None" = None, local_only: bool = False
    ) -> int | None: ...

    def set_cache(self, key: object, value: int, local_only: bool = False, *, ttl: int = ...) -> None: ...


class _EndpointPort:
    """``RustCore`` as its endpoints see it (``EndpointCore``): each member forwards to the core's
    own state or call drivers, read live, since the core replaces its handle and settings. The
    port is the core's own adapter, so it is the one place its private members are read from
    outside the class; every ``reportPrivateUsage`` ignore below is for that reason."""

    __slots__ = ("_core",)

    def __init__(self, core: "RustCore") -> None:
        self._core: Final = core

    @property
    def router(self) -> "RustCore":
        return self._core

    @property
    def public(self) -> object:
        """The ``Router`` that owns the core, which litellm functions taking ``router=`` get; the core
        itself when built alone."""
        owner: Final[object] = self._core.__dict__.get("_core_owner")
        public: Final[object] = owner() if callable(owner) else None
        return self._core if public is None else public

    @property
    def host(self) -> PythonHost:
        return self._core._host  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    @property
    def handle(self) -> "RouterCoreHandle":
        return self._core._handle  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    @property
    def resolved(self) -> "Resolved":
        return self._core._resolved  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    @property
    def shaping(self) -> Shaping:
        return self._core._shaping  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    @property
    def routing_cache(self) -> "DualCache":
        return self._core._cache  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    @property
    def router_usage(self) -> RouterUsage:
        return self._core._usage  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    @property
    def selectors(self) -> StrategySelectors:
        return self._core._selectors  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    @property
    def endpoints(self) -> RustEndpoints:
        return self._core._endpoints  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    async def drive(
        self,
        plan: Plan,
        request: Mapping[str, object],
        overrides: Mapping[str, object] | None = None,
        *,
        passthrough: Callable[[Mapping[str, object], str], Awaitable[object]] | None = None,
        prepared_already: bool = False,
    ) -> object:
        return await self._core._drive(  # pyright: ignore[reportPrivateUsage]  # see the class docstring
            plan, request, overrides, passthrough=passthrough, prepared_already=prepared_already
        )

    def call_sync(
        self, plan: Plan, request: Mapping[str, object], overrides: Mapping[str, object] | None = None
    ) -> object:
        return self._core._call_sync(plan, request, overrides)  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    def routing_callbacks(self) -> Iterable[object]:
        return self._core._checks.routing_callbacks()  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    def resolve_alias(self, model: str) -> str | None:
        return self._core._hosted._core_get_model_from_alias(model)  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    def group_deployments(self, model_name: str) -> Sequence[Mapping[str, object]] | None:
        return self._core._hosted._core_get_model_list(model_name=model_name)  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    def group_members(self, model_name: str) -> Sequence[tuple[object, object]] | None:
        return self._core._hosted._group_members(model_name)  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    def routed_name(self, model: object) -> str:
        return self._core.routed_name(model)

    def service_logged(self, call_type: str, start_time: float, end_time: float, request: Mapping[str, object]) -> None:
        self._core.service_logged(call_type, start_time, end_time, request)

    async def set_response_headers(
        self, response: object, model_group: str | None, request_kwargs: Mapping[str, object]
    ) -> object:
        return await self._core.set_response_headers(response, model_group, request_kwargs)

    def configured_deployment(self, model_id: str) -> "Deployment | None":
        return self._core._hosted._core_get_deployment(model_id)  # pyright: ignore[reportPrivateUsage]  # see the class docstring

    def registry(self) -> RouterRegistry:
        return self._core.registry()

    async def select(
        self,
        model: str,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the RouterCore signature
        *,
        kind: str = "pick",
        specific_deployment: bool | None = None,
        messages: object = None,
        batch_upload: bool = False,
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        return await self._core.select(
            model,
            request_kwargs,
            kind=kind,
            specific_deployment=specific_deployment,
            messages=messages,
            batch_upload=batch_upload,
        )

    async def acompletion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._core.acompletion(*args, **kwargs)


class RustCore(
    _SettingsSource,
    RustLegacyMembers,
    RustRegistryMembers,
    RustRetries,
    RustFallbacks,
    RustHealthCallbacks,
    RustOptionalChecks,
):
    """A ``RouterCore`` served by the Rust router core over a ``PythonHost``. It holds no Python core."""

    def __init__(self, config: RouterConfig, host: PythonHost, resolved: Resolved) -> None:
        from litellm._service_logger import ServiceLogging
        from litellm.router_utils.cooldown_cache import CooldownCache

        self._config: Final = config
        self._host: Final = host
        self.service_logger_obj: ServiceLogging = ServiceLogging()
        apply_construction_effects(config, host)
        self._resolved: Resolved = resolved
        # `cooldown_time` as the Python core stores it (`cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS`,
        # then each value assigned or updated, as given): what `settings()` reads back.
        self._cooldown_given: object = config.get("cooldown_time") or DEFAULT_COOLDOWN_TIME_SECONDS
        # Settings the Python core keeps as plain attributes, and settings deleted with `del`.
        self._plain: Final = initial_plain(config)
        self._deleted: frozenset[str] = frozenset()
        self._request_timeout: Final = request_timeout(config)
        legacy: Final[dict[str, object]] = {"deployment_stats": {}}  # mutable-ok: the core's own legacy attributes
        self._legacy: Final = legacy
        self._health_view: RustHealth | None = None
        self._cache: DualCache = host.routing_cache(config)
        redis: Final = connect(self._cache.redis_cache)
        self._usage: RouterUsage = host.router_usage(redis)
        self._cooldowns: CooldownCache = CooldownCache(
            store=RouterCooldownStore(redis.router_cooldowns()),
            default_cooldown_time=self._cooldown_given,  # pyright: ignore[reportArgumentType]  # the constructor's value, as the Python core passes it
        )
        self._health_checks: DeploymentHealthCache = resolved.health_routing.health_checks(
            DeploymentHealthStore(redis.deployment_health())
        )
        self._session_pins: Final = RouterSessionPinStore(redis.router_session_pins())
        self._version: int = 1
        self._settings: RustSettings | None = None
        self._tag_filtering_models: tuple[int, bool] | None = None
        # The registry indexes written through the deprecated attributes (`_core_<name>`), and the
        # document router-core last received for them.
        self._index_written: frozenset[str] = frozenset()
        self._index_sent: Mapping[str, object] | None = None
        self._selectors: Final = StrategySelectors(host, self._cache, self._usage)
        self._selectors.init_default(resolved.routing_strategy, resolved.routing_strategy_args)
        self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        self._port: Final = _EndpointPort(self)
        self._endpoints: Final = RustEndpoints(self._port, config)
        # The resolver every picked deployment with `kubernetes_pod_discovery` goes through.
        self._pod_discovery: KubernetesPodDiscovery = KubernetesPodDiscovery()
        general: Final = config.get("router_general_settings", _DEFAULT_GENERAL_SETTINGS)
        self._general_settings: object = general or RouterGeneralSettings()
        self._registry_view: RouterRegistry | None = None
        self._hosted: Final = HostRegistry(
            host,
            model_group_alias=dict(resolved.model_group_alias),  # pyright: ignore[reportArgumentType]  # the configured alias map
            ignore_invalid_deployments=bool(config.get("ignore_invalid_deployments")),
            on_change=self._sync_registry,
        )
        self._hosted._core_cache = self._cache  # pyright: ignore[reportPrivateUsage]  # usage reads come from the cache the Rust core writes
        self._hosted._core_usage_store = self._usage  # pyright: ignore[reportPrivateUsage]  # tpm/rpm reads come from the store the Rust core writes
        ctx: Final = RoutingContext(self)  # pyright: ignore[reportArgumentType]  # the strategies read registry(), settings(), health() and select()
        self._ctx: Final = ctx
        strategies: Final = PreRoutingStrategies(
            self._hosted, host, ctx, config.get("auto_router_capability_limit"), self._session_pins
        )
        self._hosted.strategies = strategies
        self._prerouter: Final = PreRouter(strategies, self._hosted, host, ctx, self._cache)
        # `register_plugin`: a custom strategy that replaces every pick (the host answers the
        # Rust core's pre_route step with its deployment).
        self._custom_strategy: CustomRoutingStrategyBase | None = None
        self._custom_discovery: KubernetesPodDiscovery | None = None
        # router-core builds the deployments, asking the host registry about each one: the handle
        # starts empty, and is rebuilt over the built list for the settings the list's names feed.
        self._handle_default_ids: tuple[str, ...] = ()  # the provider default ids the handle was built with
        self._allowed_fails_default_sent: bool = False  # what the handle's dynamic document last said
        self._walked_fallbacks: Mapping[str, object] = {}  # the fallback lists that document held
        self._handle: RouterCoreHandle = self._new_handle([])
        self._hosted.build_runner = self._run_build
        self._hosted.missing_attribute = self._no_router_attribute
        self._hosted.changes_runner = self._apply_registry_changes
        self._hosted.handle_lookup = self._live_handle
        self._hosted.build(_as_list(config.get("model_list")) if config.get("model_list") is not None else None)
        self._hosted.set_routing_groups(resolved.routing_groups)
        track_live_registry(self._hosted)
        self._hosted.routable_lookup = self._routable
        self._hosted.max_parallel_requests_lookup = self._max_parallel_requests_client
        self._hosted._core_routing_plugins = list(resolved.routing_plugins)  # pyright: ignore[reportPrivateUsage]  # the core fills its own host registry
        self._rebuild()
        self._join_callback_holders()
        self._checks: Final = OptionalChecks(
            self, host, self._cache, self._usage, self._session_pins, self._hosted, ctx, config.arguments
        )
        self._hosted.checks = self._checks
        strategies.ensure_affinity = self._checks.ensure_deployment_affinity_callback
        strategies.hand_over_affinity(self._checks.adopt)
        self._checks.build(config.get("optional_pre_call_checks"), config.get("model_list"))
        self._slack_alerting_logger: SlackAlerting | None = initialize_alerting(host, config.get("alerting_config"))  # pyright: ignore[reportArgumentType]  # the argument as passed
        if self._slack_alerting_logger is not None:
            self._checks.slack_alerting_loggers = (self._slack_alerting_logger,)
        if resolved.enable_mid_stream_fallback_continuation:
            register_continuation_prefill_check(host)
        if resolved.routing_strategy == "lar1":
            from litellm.router_strategy.lar1_routing import apply_lar1_routing_strategy

            apply_lar1_routing_strategy(ctx, resolved.routing_strategy_args)  # pyright: ignore[reportArgumentType]  # the core's RoutingContext

    @property
    def _core_ctx(self) -> RoutingContext:
        """The router's ``RoutingContext``: what a routing helper gets where it once took the router."""
        return self._ctx

    @property
    def _registry_state(self) -> HostRegistry:
        return self._hosted

    @property
    def _python_host(self) -> PythonHost:
        return self._host

    @property
    def _written_retry_policy(self) -> object:
        return self._plain["retry_policy"]

    @classmethod
    def build(cls, config: RouterConfig, host: PythonHost) -> "RustCore":
        from litellm.rust_bridge._native import RustBridgeDeclined

        extra: Final = unsupported_arguments(config)
        if extra:
            raise RustBridgeDeclined(f"router arguments not served by the Rust core: {sorted(extra)}")
        RustEndpoints.prepare_host(config, host)
        return cls(config, host, Resolved.of(config, host.dynamic()))

    @property
    def enable_mid_stream_fallback_continuation(self) -> bool:
        return self._resolved.enable_mid_stream_fallback_continuation

    def _set_continuation(self, enabled: bool) -> None:
        """``_set_mid_stream_fallback_continuation``: an enabled flag registers the process-wide prefill check."""
        self._resolved = dataclasses.replace(self._resolved, enable_mid_stream_fallback_continuation=enabled)
        self._settings = None
        if enabled:
            register_continuation_prefill_check(self._host)

    def __dir__(self) -> list[str]:  # mutable-ok: dir() returns a list
        return sorted({*super().__dir__(), *endpoint_names()})

    def __getattr__(self, name: str) -> object:
        if name in self.__dict__.get("_deleted", ()):
            raise self._no_attribute(name)
        if name in SELECTOR_ATTRIBUTES.values():
            return self._selectors.attribute(name)
        if name == "_core_async_get_available_deployment" and self.__dict__.get("_custom_strategy") is not None:
            return self._custom_async_pick()
        # Only reached for members this class does not define: the endpoints served through the
        # host, else refused. Never forwarded anywhere.
        endpoints: Final[object] = self.__dict__.get("_endpoints")
        endpoint: Final = endpoints.member(name) if isinstance(endpoints, RustEndpoints) else None
        if endpoint is not None:
            return endpoint
        attribute: Final = self._settings_attribute(name) if "_resolved" in self.__dict__ else MISSING
        if attribute is not MISSING:
            return attribute
        raise RustCoreUnsupported(f"'{name}' is not served by the Rust router core", name=name, obj=self)

    @property
    def _unset_attributes(self) -> frozenset[str]:
        """The instance attributes ``del`` removed, which ``_settings_attribute`` no longer serves."""
        unset: Final[object] = self.__dict__.get("_unset_attributes")
        return unset if isinstance(unset, frozenset) else frozenset()  # pyright: ignore[reportUnknownVariableType]  # only names are stored

    def _settings_attribute(self, name: str) -> object:
        """A router attribute that reads a setting, the constructor's arguments or a constant."""
        if name in self._unset_attributes:
            return MISSING
        r: Final = self._resolved
        resolved: Final[Mapping[str, object]] = {
            "num_retries": r.read("num_retries"),
            "max_fallbacks": r.read("max_fallbacks"),
            "timeout": r.read("timeout"),
            "retry_after": r.read("retry_after"),
            "explicit_allowed_fails": r.given["allowed_fails"]
            if r.explicit_allowed_fails is not None and "allowed_fails" in r.given
            else r.explicit_allowed_fails,
            "routing_strategy_args": r.read("routing_strategy_args"),
            "request_timeout": self._request_timeout,
            "default_priority": self._endpoints._default_priority,  # pyright: ignore[reportPrivateUsage]  # parsed by the endpoint plans
        }
        if name in resolved:
            return resolved[name]
        if name in self._plain:
            return self._plain[name]
        if name in self._legacy:
            return self._legacy[name]
        if name == "healthy_deployments":
            return self._hosted._core_model_list  # pyright: ignore[reportPrivateUsage]  # the live list, as the core keeps it
        if name == "get_model_group_io_token_usage":
            return self._hosted.get_model_group_io_token_usage
        if name == "scheduler":
            return self._endpoints._get_scheduler()  # pyright: ignore[reportPrivateUsage]  # the endpoint plans' scheduler
        if name == "chat":
            from litellm.main import Chat

            self._legacy["chat"] = Chat(params=default_litellm_params(self._config), router_obj=self)  # pyright: ignore[reportArgumentType]  # Chat calls the router's endpoints, which this core serves
            return self._legacy["chat"]
        return argument_attribute(self._config, name)

    @property
    def _shaping(self) -> Shaping:
        return self._endpoints.shaping(self._resolved)

    def _core_deployment_call_timeout(self, litellm_params: Mapping[str, object]) -> object:
        return non_stream_timeout(self._shaping, MappingProxyType({}), litellm_params)

    # -- building --------------------------------------------------------------------------------

    def _new_handle(self, model_list: Sequence[object]) -> "RouterCoreHandle":
        from litellm.rust_bridge._native import RouterCoreHandle

        names: Final = [
            str(name)
            for entry in model_list
            if isinstance(name := (as_str_mapping(entry) or {}).get("model_name"), str)
        ]
        default_ids: Final = self._provider_default_ids()
        self._handle_default_ids = default_ids
        static: Final = {
            **self._resolved.static_document(model_list),
            **{name: value for name, value in failover_document(self._plain).items() if value is not None},
            **({"provider_default_deployment_ids": list(default_ids)} if default_ids else {}),
        }
        return RouterCoreHandle(static, self._dynamic_document(names))

    def _dynamic_document(self, model_names: Sequence[str]) -> Mapping[str, object]:
        self._allowed_fails_default_sent = self._resolved.allowed_fails_is_default()
        # Settings are static: an in-place edit of a live list reaches neither the walk nor its notes.
        resolved: Final = self._resolved
        self._walked_fallbacks = {
            "fallbacks": detached(resolved.fallbacks),
            "context_window_fallbacks": detached(resolved.context_window_fallbacks),
            "content_policy_fallbacks": detached(resolved.content_policy_fallbacks),
        }
        return resolved.dynamic_document(self._version, model_names)

    def _provider_default_ids(self) -> tuple[str, ...]:
        """``provider_default_deployment_ids`` as the cooldown check reads it (``deployment in
        settings().provider_default_deployment_ids``): the ids it holds now. router-core holds them
        with its static settings."""
        handle: Final = self._live_handle()
        return () if handle is None else tuple(handle.provider_default_ids())

    def _pick_discovery(self, *, sync: bool) -> KubernetesPodDiscovery | None:
        """The pod discovery a pick resolves through: the one bound when a custom strategy was
        registered (``resolve_pods_after_bound``), else the router's own (``resolve_pods_after``);
        none for an async pick assigned through the deprecated attribute, which Python calls bare."""
        if not sync and callable(self.__dict__.get("_core_async_get_available_deployment")):
            return None
        if self._custom_strategy is not None and self._custom_discovery is not None:
            return self._custom_discovery
        return self._pod_discovery

    # -- the plugin context (``RoutingContext``) -----------------------------------------------

    def _core_arm_routing_read_prefetch(self, model: str, request_kwargs: Mapping[str, object] | None = None) -> None:
        """``RoutingContext.arm_read_prefetch``: declare the cooldown reads (and, for
        usage-based-routing-v2, the usage reads) the pick for ``model`` makes on the request's
        Redis batch, so admission's flush carries them. A miss costs nothing."""
        from litellm.router_utils.routing_read_batch import RoutingPrefetch

        try:
            usage_selector: Final = self._prefetch_usage_selector(model, request_kwargs)
            deployments: Final = self._hosted._core_get_model_list(model_name=model)  # pyright: ignore[reportPrivateUsage]  # see __init__
            if deployments:
                RoutingPrefetch.arm(RoutingContext(self), usage_selector, deployments)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # the prefetch reads registry() and health(); its deployments list is untyped
        except Exception as e:  # noqa: BLE001  # a prefetch is an optimisation, never a reason to fail the request
            from litellm._logging import verbose_router_logger

            verbose_router_logger.debug(
                "routing read prefetch not armed for %s: %s",
                str(model).replace("\r", "").replace("\n", ""),
                str(e).replace("\r", "").replace("\n", ""),
            )

    @contextlib.contextmanager
    def _usage_read_scope(self, model: str, request: Mapping[str, object]) -> Generator[None, None, None]:
        """``RoutingReadBatch.for_strategy``: a usage-based-routing-v2 pick reads its tpm/rpm counters
        in the cooldown read's MGET (see ``_batch_read``), and the strategy reads them from there."""
        from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2

        try:
            selector: Final = self._prefetch_usage_selector(model, request)
            deployments: Final = (
                self._hosted._core_get_model_list(model_name=model) or []  # pyright: ignore[reportPrivateUsage]  # see __init__
                if isinstance(selector, LowestTPMLoggingHandler_v2)
                else []
            )
        except Exception:  # noqa: BLE001  # the batched read is an optimisation: the reads stay apart
            yield
            return
        if not isinstance(selector, LowestTPMLoggingHandler_v2) or not deployments:
            yield
            return
        token: Final = _USAGE_READ.set((RoutingReadBatch(usage_selector=selector), list(deployments)))
        try:
            yield
        finally:
            _USAGE_READ.reset(token)

    def _prefetch_usage_selector(self, model: str, request_kwargs: Mapping[str, object] | None) -> object:
        """``_get_routing_context``'s selector when it is a usage-based-routing-v2 one: the request's
        strategy override, else the routing group ``model`` resolves into, else the router's own."""
        from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2

        strategy, selector = self._prefetch_strategy_and_selector(model, request_kwargs or {})
        if strategy == "usage-based-routing-v2" and isinstance(selector, LowestTPMLoggingHandler_v2):
            return selector
        return None

    def _prefetch_strategy_and_selector(self, model: str, request: Mapping[str, object]) -> tuple[str | None, object]:
        raw: Final = request.get("routing_strategy")
        override: Final = normalize_strategy(raw) if isinstance(raw, str | RoutingStrategy) else None
        if override is not None and override in {"simple-shuffle", *SELECTOR_ATTRIBUTES}:
            return override, self._selectors.selector_for(override, "override", None, request)
        groups: Final = self._hosted._core_routing_groups  # pyright: ignore[reportPrivateUsage]  # see __init__
        resolved: Final = self._hosted._core_get_model_from_alias(model=model) or model  # pyright: ignore[reportPrivateUsage]  # see __init__
        group_name: Final = (
            resolved
            if self._hosted.get_routing_group(resolved) is not None
            else next(
                (
                    group.group_name
                    for group in groups.values()
                    if group.routing_strategy != "priority" and resolved in group.models
                ),
                None,
            )
        )
        if group_name is None:
            return normalize_strategy(self._resolved.routing_strategy), self._selectors.default
        if groups[group_name].routing_strategy == "priority":
            return None, None
        group_strategy: Final = normalize_strategy(groups[group_name].routing_strategy)
        return group_strategy, self._selectors.selector_for(group_strategy or "", "group", group_name, None)

    async def _core_async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: the request kwargs the hook stamps
        messages: object = None,
        input: object = None,
        specific_deployment: bool | None = False,
    ) -> "PreRoutingHookResponse | None":
        """``RoutingContext.pre_routing_hook``: the pre-routing strategies' answer for ``model``
        (the tier group and its params), with the routing plugins run first."""
        r: Final = self._resolved
        owner: Final[object] = self.__dict__.get("_core_owner")
        public: Final[object] = owner() if callable(owner) else None
        return await self._prerouter._hook(  # pyright: ignore[reportPrivateUsage]  # the core's own pre-router
            model,
            request_kwargs,
            messages=messages,
            input=input,
            fallback_depth=0,
            enable_tag_filtering=r.enable_tag_filtering,
            match_any=r.tag_filtering_match_any,
            public_router=public if public is not None else self,
            run_plugins=partial(self._run_plugins, request_kwargs, messages)
            if self._hosted._core_routing_plugins  # pyright: ignore[reportPrivateUsage]  # the live plugin list
            else None,
            call_type=None,
        )

    def _core_log_retry(
        self,
        kwargs: dict[str, object],  # mutable-ok: the request kwargs the record joins
        e: Exception,
    ) -> dict[str, object]:  # mutable-ok: the same kwargs, as Python's returns them
        """``RoutingContext.log_retry``."""
        return log_retry_into(kwargs, e, self._deployment_params_base_model)

    def _deployment_params_base_model(self, deployment_id: str) -> str | None:
        """``_deployment_params_base_model``: the ``base_model`` a deployment declares under ``litellm_params``."""
        deployment: Final = self._hosted._core_get_deployment(model_id=deployment_id)  # pyright: ignore[reportPrivateUsage]  # see __init__
        if deployment is None:
            return None
        base_model: Final = (deployment.litellm_params.model_extra or {}).get("base_model")
        return base_model if isinstance(base_model, str) else None

    def _sync_routing_state(self) -> None:
        """Before a pick: sync the routing state that callers can change in place, which the Python
        core reads live on every pick. This exists only for the deprecated writable attributes and
        goes away with them.

        Only these cases send anything; otherwise both checks return without touching router-core::

            router.model_names = {...}            # assigning syncs at once (__setattr__)
            router.model_names.add("new-group")   # an in-place change is picked up here, on the next pick
            router.model_name_to_deployment_indices["g"].append(3)   # likewise

            router.routing_plugins.append(plugin)  # first plugin: rebuilds the handle with the plugin stage
            router.routing_plugins.clear()         # list emptied: rebuilds without it

        Until one of the index attributes has been assigned, ``_sync_index`` sends nothing. After
        that it sends the index document only when it differs from the last one it sent.
        """
        self._sync_plugin_stage()
        self._sync_index()
        self._sync_allowed_fails()

    def _sync_allowed_fails(self) -> None:
        """``_is_allowed_fails_set_on_router`` compares with the live ``litellm.allowed_fails`` on
        every failure, so a write to the global can change the rule; ``_settings`` stays as copied."""
        if self._resolved.allowed_fails_is_default() != self._allowed_fails_default_sent:
            names: Final = [str(d.get("model_name")) for d in self._handle.deployments()]
            self._handle.apply({"settings": self._dynamic_document(names)})

    def _sync_index(self) -> None:
        """Hand router-core the registry indexes written through the deprecated attributes
        (``model_names``, ``model_name_to_deployment_indices``, ``team_model_to_deployment_indices``)
        as the host registry holds them now (its own deployment changes update the written objects,
        as the Python core's do), whenever they differ from what it holds."""
        if not self._index_written:
            return
        document: Final = index_document(self._hosted, self._index_written)
        if document != self._index_sent:
            _ = self._handle.apply({"index": document})
            self._index_sent = document

    def _sync_plugin_stage(self) -> None:
        """Give router-core the routing-plugin stage when the live plugin list (``routing_plugins``,
        which callers may assign or append to) gains its first plugin, and drop it when the list
        empties: the Python core checks the list on every pick."""
        live: Final = _as_list(self._hosted._core_routing_plugins)  # pyright: ignore[reportPrivateUsage]  # the live plugin list
        if bool(live) == bool(self._resolved.routing_plugins):
            return
        self._resolved = dataclasses.replace(self._resolved, routing_plugins=tuple(live))  # pyright: ignore[reportArgumentType]  # the plugins as listed
        self._rebuild()

    def _rebuild(self) -> None:
        """A new Rust core for changed static settings, over the same deployments and the
        providers the old one's builds resolved for them."""
        with self._hosted._deployment_index_lock:  # pyright: ignore[reportPrivateUsage]  # the registry's lock, held across the swap and the read back
            old: Final = self._handle
            model_list: Final = list(self._hosted._core_model_list)  # pyright: ignore[reportPrivateUsage]  # see __init__
            self._handle = self._new_handle(model_list)
            self._handle.adopt_providers(old)
            old.retire()
            # The live list and indexes read the new handle's (the same entries, kept as the same objects).
            self._hosted.refresh()
        # The new handle builds its indexes from the deployments: written ones are handed again.
        self._index_sent = None

    def _run_build(
        self,
        entries: Sequence[object],
        path: str,
        resolve_secrets: bool,
        install: BuildInstall,
        answer: Callable[[Mapping[str, object]], object],
    ) -> None:
        """Drive one router-core deployment build (``RouterCoreHandle.build``), answering each of
        its steps with ``answer``; router-core installs each deployment as it accepts it, and the
        host registry reads the list and indexes back before answering the next step."""
        call: Final = self._handle.build(_plain(list(entries)), path, resolve_secrets, install)  # pyright: ignore[reportArgumentType]  # plain JSON entries
        outcome: object = None  # rebind-ok: each step's answer
        while True:
            step = call.next(outcome)
            self._hosted.refresh()
            if step["kind"] == "done":
                break
            if step["kind"] == "failed":
                raise ValueError(str(step.get("error")))
            outcome = _plain(answer(step))

    def _live_handle(self) -> "RouterCoreHandle | None":
        """The handle the live model_list reads, once the constructor made one."""
        from litellm.rust_bridge._native import RouterCoreHandle

        handle: Final[object] = self.__dict__.get("_handle")
        return handle if isinstance(handle, RouterCoreHandle) else None

    def _apply_registry_changes(self, changes: Mapping[str, object]) -> None:
        """Apply a host registry write router-core's build did not make (a removal is a delete, a
        swapped list a replace) as a changeset."""
        _ = self._handle.apply(_plain(dict(changes)))  # pyright: ignore[reportArgumentType]  # a plain JSON changeset

    def _sync_registry(self) -> None:
        """After a host registry write: router-core already holds it (the build installed what it
        added, a removal came as a changeset), so only the settings it feeds move."""
        from litellm.rust_bridge._native import RouterCoreHandle

        self._tag_filtering_models = None
        handle: Final[object] = self.__dict__.get("_handle")
        # A wildcard deployment added an id the cooldown check skips.
        if isinstance(handle, RouterCoreHandle) and self._provider_default_ids() != self._handle_default_ids:
            self._rebuild()

    # -- entry points ----------------------------------------------------------------------------

    def registry(self) -> RouterRegistry:
        """The registry view, built once (``registry()`` returns the same object each time)."""
        if self._registry_view is None:
            self._registry_view = hosted_registry_view(self._hosted)
        return self._registry_view

    def settings(self) -> RustSettings:
        """A cached snapshot, replaced when a changeset changes a setting."""
        if self._settings is None:
            self._settings = self._settings_snapshot()
        return self._settings

    def _settings_snapshot(self) -> RustSettings:
        r: Final = self._resolved
        snapshot: Final = RustSettings(
            routing_strategy=r.routing_strategy,
            fallbacks=r.read("fallbacks"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            context_window_fallbacks=r.read("context_window_fallbacks"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            content_policy_fallbacks=r.read("content_policy_fallbacks"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            default_deployment=r.default_deployment,
            default_litellm_params=dict(r.default_litellm_params)
            if "default_litellm_params" in r.assigned
            else default_litellm_params(self._config),
            default_max_parallel_requests=self._endpoints.default_max_parallel_requests,  # pyright: ignore[reportArgumentType]  # the setting as given
            cooldown_time=self._cooldown_given,  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            allowed_fails=r.read("allowed_fails"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            allowed_fails_policy=r.allowed_fails_policy,  # pyright: ignore[reportArgumentType]  # the policy as passed
            disable_cooldowns=r.read("disable_cooldowns"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            enable_pre_call_checks=r.enable_pre_call_checks,
            enable_health_check_routing=r.health_routing.enable_health_check_routing,
            enable_tag_filtering=r.read("enable_tag_filtering"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            tag_filtering_match_any=r.read("tag_filtering_match_any"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            tag_routing_prefix=r.read("tag_routing_prefix"),  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            fallback_access_check=r.fallback_access_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            fallback_budget_check=r.fallback_budget_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            fallback_rate_limit_check=r.fallback_rate_limit_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            cache_responses=self._endpoints.cache_responses,
            search_tools=self._endpoints.search_tools,  # pyright: ignore[reportArgumentType]  # the configured entries, as passed
            guardrail_list=self._endpoints.guardrail_list,  # pyright: ignore[reportArgumentType]  # same
            provider_default_deployment_ids=self._hosted._core_provider_default_deployment_ids,  # pyright: ignore[reportPrivateUsage]  # the host registry's live list
            health_check_ignore_transient_errors=r.health_routing.health_check_ignore_transient_errors,
            background_health_check_model_groups=r.health_routing.background_health_check_model_groups,
            router_general_settings=self._general_settings,  # pyright: ignore[reportArgumentType]  # the setting as given
            provider_budget_config=self._checks.provider_budget_config,  # pyright: ignore[reportArgumentType]  # the budget limiter's live config
            _router=self,  # pyright: ignore[reportArgumentType]  # the snapshot reads as_dict from this core
            kubernetes_pod_discovery=self._pod_discovery,
        )
        if not self._deleted:
            return snapshot
        # A deleted setting reads as unset.
        return dataclasses.replace(
            snapshot,
            **dict.fromkeys(self._deleted.intersection(SETTING_FIELDS), None),  # pyright: ignore[reportArgumentType]  # settings fields only: an alias map or retry count is deletable too
            _unset=snapshot._unset | self._deleted,  # pyright: ignore[reportPrivateUsage]  # the snapshot's own unset set
        )

    def available_guardrail(self, name: str) -> object:
        return self._endpoints.available_guardrail(name)

    def settings_dict(self) -> Mapping[str, object]:
        r: Final = self._resolved
        latency: Final = self._selectors.latency_args() if r.routing_strategy == "latency-based-routing" else None
        # `get_settings` reports the latency selector's full routing args.
        args: Final = (
            latency if latency is not None else r.given.get("routing_strategy_args", dict(r.routing_strategy_args))
        )
        values: Final = {
            "routing_strategy_args": args,
            "routing_strategy": r.routing_strategy,
            "allowed_fails": r.read("allowed_fails"),
            "cooldown_time": self._cooldown_given,
            "num_retries": r.read("num_retries"),
            "timeout": r.read("timeout"),
            "retry_after": r.read("retry_after"),
            "fallbacks": r.read("fallbacks"),
            "context_window_fallbacks": r.read("context_window_fallbacks"),
            "model_group_alias": dict(r.model_group_alias),
            "enable_mid_stream_fallback_continuation": r.enable_mid_stream_fallback_continuation,
            "enable_tag_filtering": r.read("enable_tag_filtering"),
            "tag_routing_prefix": r.read("tag_routing_prefix"),
            "routing_groups": group_settings(r.routing_groups),
        }
        return settings_view(values, self._plain, self._deleted)

    def health(self) -> RustHealth:
        """The health view, built once."""
        if self._health_view is None:
            self._health_view = RustHealth(self)
        return self._health_view

    def observe(self, logging_obj: "LiteLLMLogging | None" = None, *, request_scoped: bool = False) -> RoutingObservers:
        """The routing observers: the Slack alerting logger ``alerting_config`` built, if any.
        ``request_scoped=True`` takes the router's callbacks off the host lists; passing a request's
        ``logging_obj`` binds its selectors and policies to it. This core registers no outcome
        callbacks, so it binds none either: router-core reads what they would do for the request
        (``OptionalChecks.outcome_callbacks``)."""
        if request_scoped:
            self._serve_through_request_logging()
        if logging_obj is not None:
            self._checks.bind_to_request_logging(logging_obj)
        return RoutingObservers(slack_alerting_logger=self._slack_alerting_logger)

    def register_plugin(self, plugin: "CustomRoutingStrategyBase") -> None:
        """``_core_set_custom_routing_strategy``: ``plugin`` picks every deployment from now on, until
        the routing strategy changes."""
        self._custom_strategy = plugin
        # `resolve_pods_after_bound`: its picks resolve pods through the discovery bound now.
        self._custom_discovery = self._pod_discovery
        # Python assigns the strategy's picks over its own: a caller of the pick reaches the
        # strategy with the arguments it passed and no others.
        self.__dict__["get_available_deployment"] = self._custom_sync_pick(plugin)
        # It replaces an async pick assigned through the deprecated attribute, as Python's does.
        if "_core_async_get_available_deployment" in self.__dict__:
            del self.__dict__["_core_async_get_available_deployment"]

    def _custom_sync_pick(self, plugin: "CustomRoutingStrategyBase") -> object:
        discovery: Final = self._custom_discovery
        pick: Final = plugin.get_available_deployment
        return pick if discovery is None else resolve_pods_after_bound(pick, discovery)

    def _custom_async_pick(self) -> object:
        """``_core_set_custom_routing_strategy``'s ``_core_async_get_available_deployment``."""
        custom: Final = self._custom_strategy
        discovery: Final = self._custom_discovery
        pick: Final = custom.async_get_available_deployment  # pyright: ignore[reportOptionalMemberAccess]  # __getattr__ checked it
        return pick if discovery is None else async_resolve_pods_after_bound(pick, discovery)

    def set_custom_routing_strategy(self, CustomRoutingStrategy: "CustomRoutingStrategyBase") -> None:
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the deprecated alias notes its use like Python's
        )

        _note("set_custom_routing_strategy", "register_plugin(strategy)")
        self.register_plugin(CustomRoutingStrategy)

    def retire(self) -> None:
        """``_core_discard``: only the router's global registrations go; it keeps serving calls."""
        untrack_live_registry(self._hosted)
        self._hosted.retire_shared_capabilities()
        self._retire_callbacks()
        if self._hosted.strategies is not None:
            self._hosted.strategies.retire()

    def apply(self, changes: ChangeSet) -> ApplyResult:
        """``apply_changes``: assigned settings, then ``update_settings``, upserts, deletes and the
        Redis cache, in that order. An unknown name refuses the changeset; a setting
        ``update_settings`` raises for ends it there, with the settings before it applied."""
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        if changes.refresh_model_info:
            raise TypeError("ChangeSet(refresh_model_info=True) needs aapply()")
        unknown: Final = sorted(set(changes.assign) - {*SETTING_FIELDS, *PLAIN_SETTING_ATTRIBUTES, *ASSIGNABLE_ALIASES})
        if unknown:
            raise ValueError(f"not assignable router settings: {unknown}")
        unserved: Final = sorted(set(changes.assign) - _STATIC_SETTINGS - _DYNAMIC_SETTINGS - _HOST_SETTINGS)
        if unserved:
            raise RustCoreUnsupported(f"the Rust router core does not serve the setting(s) {', '.join(unserved)} yet")
        update: Final = read_settings_update(changes.settings, self._resolved.routing_strategy)
        # `apply_changes` assigns before it runs `update_settings`: a budget config assigned beside
        # `optional_pre_call_checks` builds the budget limiter with the new config.
        self._assign_host_settings(changes.assign)
        if update.optional_pre_call_checks is not MISSING:
            self._checks.set(update.optional_pre_call_checks)
        if update.continuation is not MISSING:
            self._set_continuation(continuation_flag(update.continuation))
        self._apply_settings(changes.assign, update)
        if update.failure is not None:
            raise update.failure
        # Upserts first, then deletes (`apply_changes`), through the host registry, which hands
        # each change to the Rust core.
        upserted: Final = [self._hosted._core_upsert_deployment(deployment=d) for d in changes.upsert]  # pyright: ignore[reportPrivateUsage]  # see __init__
        deleted: Final = [self._hosted._core_delete_deployment(id=i) for i in changes.delete]  # pyright: ignore[reportPrivateUsage]  # see __init__
        if changes.redis_cache is not None:
            # `_core_update_redis_cache`: the routing state's cache gains the Redis cache.
            self._cache.attach_redis_cache(changes.redis_cache)
            redis: Final = connect(changes.redis_cache)
            self._usage.attach_redis(redis)
            self._cooldowns.cooldown_store.attach_redis(redis)
            self._health_checks.store.attach_redis(redis)
            self._session_pins.attach_redis(redis)
            self._checks.prompt_caching_pin_cache.attach_redis_cache(changes.redis_cache)
            self._prerouter.attach_redis_cache(changes.redis_cache)
        return ApplyResult(
            version=self._hosted._registry_version,  # pyright: ignore[reportPrivateUsage]  # see __init__
            upserted=upserted,
            deleted=deleted,  # pyright: ignore[reportArgumentType]  # the removed model_list entries, as the core returns them
        )

    async def aapply(self, changes: ChangeSet) -> ApplyResult:
        """``apply``, then the model info refresh, which needs I/O."""
        result: Final = self.apply(dataclasses.replace(changes, refresh_model_info=False, refresh_client=None))
        if changes.refresh_model_info:
            await self._hosted.arefresh_model_info(changes.refresh_client)
        return result

    def _apply_settings(self, assign: Mapping[str, object], update: SettingsUpdate) -> None:
        values: Final = {
            **without(assign, *_HOST_SETTINGS),
            **update.values,
        }
        given: Final = dict(update.values)  # the settings passed through `update_settings`
        groups_input: Final[object] = (
            update.routing_groups if update.routing_groups is not MISSING else self._resolved.routing_groups_input
        )
        # What `_core_update_settings` runs after its loop, which a setting that raised never reaches.
        settled: Final = update.failure is None
        groups_failure: Exception | None = None  # rebind-ok: set when the groups passed do not parse
        if settled and (update.routing_groups is not MISSING or self._changes_strategy(update.values)):
            # `update_settings` re-inits the routing groups from the groups passed, else (on a
            # strategy change) from the groups last passed, whatever a raw write assigned since.
            try:
                values["routing_groups"] = parse_groups(
                    groups_input,
                    frozenset(str(d.get("model_name")) for d in self._handle.deployments()),
                    frozenset(self._resolved.model_group_alias),
                )
            except Exception as raised:  # noqa: BLE001  # raised below: `_init_routing_groups` runs after the loop
                groups_failure = raised  # rebind-ok: None until the groups fail
            else:
                given["routing_groups"] = groups_input
        self._store_plain(update.plain)
        if self._deleted & set(values):
            self._deleted = self._deleted - set(values)
            self._settings = None
        if values:
            previous: Final = self._resolved
            resolved: Final = dataclasses.replace(
                previous.changed(values, frozenset(given)),
                routing_groups_input=groups_input if "routing_groups" in values else previous.routing_groups_input,
            )
            static_changed: Final = bool(_STATIC_SETTINGS & set(values))
            self._resolved = resolved
            if set(values) & set(SETTING_FIELDS):
                self._settings = None  # the snapshot holds settings fields, not the plain attributes
            if "model_group_alias" in values:
                self._hosted._core_model_group_alias = dict(resolved.model_group_alias)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # see __init__
                # A group's info reads through the alias map.
                self._hosted._invalidate_model_group_info_cache()  # pyright: ignore[reportPrivateUsage]  # see __init__
            self._apply_strategy(previous, resolved, given, settled=settled)
            if "routing_groups" in values:
                self._settings = None
                self._hosted.set_routing_groups(resolved.routing_groups)
            if "cooldown_time" in values:
                self._cooldown_given = values["cooldown_time"]
                if values["cooldown_time"] is None:
                    # `add_deployment_to_cooldown(cooldown_time=None)` cools down for the cooldown
                    # cache's default, which keeps the constructor's value.
                    self._resolved = dataclasses.replace(
                        resolved, cooldown_time=float(self._cooldowns.default_cooldown_time)
                    )
            if static_changed:
                self._rebuild()
            else:
                names: Final = [str(d.get("model_name")) for d in self._handle.deployments()]
                self._handle.apply({"settings": self._dynamic_document(names)})
        if groups_failure is not None:
            raise groups_failure

    def _store_plain(self, plain: Mapping[str, object]) -> None:
        """Settings the Python core keeps as plain attributes, read from the next call on."""
        flags_before: Final = failover_document(self._plain)
        self._plain.update(plain)
        if failover_document(self._plain) != flags_before:
            self._rebuild()
        policies: Final = {name: plain[name] for name in _RETRY_POLICY_SETTINGS if name in plain}
        if policies:
            self._set_retry_policies(policies)

    def _changes_strategy(self, updated: Mapping[str, object]) -> bool:
        """``update_settings`` names a ``routing_strategy`` other than the router's."""
        if "routing_strategy" not in updated:
            return False
        return _strategy_name(updated["routing_strategy"]) != self._resolved.routing_strategy

    def _assign_host_settings(self, assign: Mapping[str, object]) -> None:
        """The assigned settings only the host reads, each stored as given (detached, as
        ``apply_changes`` stores it) where its readers find it from the next call on."""
        given: Final = {name: detached(value) for name, value in assign.items() if name in _HOST_SETTINGS}
        if not given:
            return
        for name, value in given.items():
            if name in ("enable_health_check_routing", "background_health_check_model_groups"):
                self._resolved = dataclasses.replace(
                    self._resolved,
                    health_routing=dataclasses.replace(self._resolved.health_routing, **{name: value}),  # pyright: ignore[reportArgumentType]  # stored as given, as the Python core stores it
                )
            elif name == "provider_default_deployment_ids":
                # The list wildcard deployments added later are appended to, as on the Python core.
                self._hosted._core_provider_default_deployment_ids = value  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # stored as given, as the Python core stores it
            elif name == "cache_responses":
                self._endpoints.set_cache_responses(bool(value))
            elif name == "provider_budget_config":
                # Read when the budget limiter is built; a limiter built already keeps its config.
                self._checks.provider_budget_config = value
            elif name == "health_check_ignore_transient_errors":
                self._resolved = dataclasses.replace(
                    self._resolved,
                    health_routing=dataclasses.replace(
                        self._resolved.health_routing,
                        health_check_ignore_transient_errors=value,  # pyright: ignore[reportArgumentType]  # stored as given, as the Python core stores it
                    ),
                )
            elif name == "router_general_settings":
                self._general_settings = value
            elif name == "kubernetes_pod_discovery":
                self._pod_discovery = value  # pyright: ignore[reportAttributeAccessIssue]  # stored as given, as the Python core stores it
            else:
                self._endpoints.assign_setting(name, value)
        self._deleted = self._deleted - set(given)
        self._settings = None
        if {"enable_health_check_routing", "background_health_check_model_groups"} & set(given) or (
            self._provider_default_ids() != self._handle_default_ids
        ):
            self._rebuild()

    def _set_retry_policies(self, policies: Mapping[str, object]) -> None:
        """The ``retry_policy`` and ``model_group_retry_policy`` the retry loop reads from then on,
        from ``update_settings`` or an attribute write: router-core holds them with its static
        settings."""
        previous: Final = self._resolved
        resolved: Final = dataclasses.replace(
            previous,
            retry_policy=readable_retry_policy(policies.get("retry_policy", previous.retry_policy)),
            model_group_retry_policy=policies.get("model_group_retry_policy", previous.model_group_retry_policy),
        )
        if resolved != previous:
            self._resolved = resolved
            self._rebuild()

    def _apply_strategy(
        self, previous: Resolved, resolved: Resolved, values: Mapping[str, object], *, settled: bool
    ) -> None:
        """``_core_update_settings``: a changed strategy re-inits the selectors with the args passed
        alongside it (or none); changed args alone rebuild the default selector. What runs after
        the loop needs ``settled``: no setting raised, and the groups are in ``values``."""
        if "routing_strategy" in values and resolved.routing_strategy != previous.routing_strategy:
            if resolved.routing_strategy == "lar1":
                # `_switch_routing_strategy`: LAR-1 is a custom strategy, registered in place of
                # `routing_strategy_init`, so the selectors the router held stay as they were.
                from litellm.router_strategy.lar1_routing import apply_lar1_routing_strategy

                apply_lar1_routing_strategy(
                    RoutingContext(self),  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
                    as_str_mapping(values.get("routing_strategy_args")),
                )
            else:
                # `routing_strategy_init` drops a custom strategy (`_reset_custom_routing_strategy`),
                # and an async pick assigned through the deprecated attribute.
                self._custom_strategy = None
                self.__dict__.pop("get_available_deployment", None)
                if "_core_async_get_available_deployment" in self.__dict__:
                    del self.__dict__["_core_async_get_available_deployment"]
                self._selectors.init_default(
                    resolved.routing_strategy, as_str_mapping(values.get("routing_strategy_args")) or {}
                )
            if "routing_groups" in values:
                self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        elif "routing_groups" in values:
            self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        elif (
            settled
            and "routing_strategy_args" in values
            and resolved.routing_strategy_args != previous.routing_strategy_args
        ):
            if resolved.routing_strategy == "lar1":
                from litellm.router_strategy.lar1_routing import apply_lar1_routing_strategy

                apply_lar1_routing_strategy(RoutingContext(self), resolved.routing_strategy_args)  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
                return
            self._selectors.rebuild_default(resolved.routing_strategy_args)

    def upsert_deployment(self, deployment: Deployment) -> Deployment | None:
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the router's deprecation notice
        )

        _note("upsert_deployment", "apply(ChangeSet(upsert=[deployment]))")
        return self.apply(ChangeSet(upsert=[deployment])).upserted[0]

    def delete_deployment(self, id: str) -> Deployment | None:
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see above

        _note("delete_deployment", "apply(ChangeSet(delete=[id]))")
        return self.apply(ChangeSet(delete=[id])).deleted[0]

    def update_settings(self, **kwargs: object) -> None:  # kwargs-ok: the deprecated signature, unchanged
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see upsert_deployment

        _note("update_settings", "apply(ChangeSet(settings={...}))")
        _ = self.apply(ChangeSet(settings=kwargs))

    def update_redis_cache(self, cache: "RedisCache") -> None:
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see upsert_deployment

        _note("update_redis_cache", "apply(ChangeSet(redis_cache=cache))")
        _ = self.apply(ChangeSet(redis_cache=cache))

    def get_model_list(
        self, model_name: str | None = None, team_id: str | None = None
    ) -> "list[DeploymentTypedDict] | None":  # mutable-ok: the old return type, unchanged
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see upsert_deployment

        _note("get_model_list", "registry().deployments(model_name, team_id=...)")
        return self._hosted._core_get_model_list(model_name=model_name, team_id=team_id)  # pyright: ignore[reportPrivateUsage, reportReturnType]  # the host registry's listing, as the deprecated member returns it

    @staticmethod
    def get_valid_args() -> list[str]:  # mutable-ok: the old return type, a fresh list
        """The arguments ``Router.__init__`` accepts."""
        return list(ROUTER_INIT_ARGS)

    async def arefresh_model_info(self, *, client: "AsyncHTTPHandler | None" = None) -> None:
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see upsert_deployment

        _note("arefresh_model_info", "aapply(ChangeSet(refresh_model_info=True))")
        _ = await self.aapply(ChangeSet(refresh_model_info=True, refresh_client=client))

    def flush_cache(self) -> None:
        """Drop the host's response cache and the routing state."""
        self._host.set_global_cache(None)
        self._usage.flush_local_counters()
        self._cooldowns.cooldown_store.flush_local_counters()
        self._cache.flush_cache()
        self._checks.prompt_caching_pin_cache.in_memory_cache.flush_cache()

    def reset(self) -> None:
        """Empty the host's success and failure callback lists (every entry, not only this core's),
        drop the retry policy and flush the caches."""
        for name in ("success_callback", "_async_success_callback", "failure_callback", "_async_failure_callback"):
            self._host.set_callback_list(name, [])
        self._plain["retry_policy"] = None
        self._set_retry_policies({"retry_policy": None})
        self.flush_cache()

    def __delattr__(self, name: str) -> None:
        # `del router.<setting>` unsets it: the settings view reports it unset from then on.
        from litellm.router_api import PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        # The deprecated attribute deletes the core's own name (``_core_<setting>``).
        setting: Final = name.removeprefix("_core_")
        if (
            setting not in SETTING_FIELDS
            and setting not in PLAIN_SETTING_ATTRIBUTES
            and setting not in _DELETABLE_ATTRIBUTES
        ):
            if name in self.__dict__ or name in self._unset_attributes:
                object.__delattr__(self, name)
                return
            if self._settings_attribute(name) is MISSING:
                object.__delattr__(self, name)
                return
            # An attribute the Python core keeps on the instance (`alerting_config`, an argument
            # as passed): reading it raises AttributeError from now on.
            self.__dict__["_unset_attributes"] = self._unset_attributes | {name}
            return
        if setting in self._deleted:
            raise self._no_attribute(setting)
        self._deleted = self._deleted | {setting}
        self._settings = None
        if setting == "model_group_alias":
            del self._hosted._core_model_group_alias  # pyright: ignore[reportPrivateUsage]  # every read of it raises from now on, as on the Python core
        elif setting == "scheduler":
            self._endpoints.delete_scheduler()

    def _no_attribute(self, name: str) -> AttributeError:
        """What reading an attribute deleted from the core raises (the facade renames it)."""
        return AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'", name=name, obj=self)

    def _no_router_attribute(self, name: str) -> AttributeError:
        """``_no_attribute`` for a read through ``registry()``, which the deprecated attribute
        renames when it is the router's own."""
        owner: Final[object] = self.__dict__.get("_core_owner")
        public: Final[object] = owner() if callable(owner) else None
        return AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
            name=name,
            obj=self if public is None else public,
        )

    def _refuse_deleted_reads(self, request: Mapping[str, object]) -> None:
        """What a call reads of the router before any attempt, once deleted: routing reads
        ``_core_model_group_alias``, and ``async_function_with_retries`` reads ``self.num_retries``
        when the request names none. Every hop raises it, so no fallback serves the call."""
        if "model_group_alias" in self._deleted:
            raise self._no_attribute("_core_model_group_alias")
        if "num_retries" in self._deleted and request.get("num_retries") is None:
            raise self._no_attribute("num_retries")

    def __setattr__(self, name: str, value: object) -> None:
        # A setting write is a changeset (`router.num_retries = 2` is `apply(ChangeSet(assign=...))`),
        # a `model_list` write swaps the live list, and any other attribute is kept as written.
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        if name in PLAIN_SETTING_ATTRIBUTES or name in SETTING_FIELDS or name in ASSIGNABLE_ALIASES:
            _ = self.apply(ChangeSet(assign={name: value}))
        elif name in PLAIN_DEFAULTS or name == "max_retries":
            # Plain attributes on the Python core, outside the setting lists: stored as given.
            self._store_plain({name: value})
        elif name == "default_priority":
            # Queued calls read it from the next call on, kept as given, as the Python core's attribute is.
            self._endpoints.assign_setting(name, value)
            self._deleted = self._deleted - {name}
        elif name == "scheduler":
            self._endpoints.set_scheduler(value)
            self._deleted = self._deleted - {name}
        elif name in ("model_list", "_core_model_list"):
            # `router.model_list = [...]` reaches the core as `_core_model_list` through the shim.
            self._assign_model_list(_as_list(value))
        elif name in _HOSTED_STATE:
            # A deprecated attribute write of the registry state: it lands where this core keeps it.
            setattr(self._hosted, name, value)
            if name in INDEX_ATTRIBUTES:
                self._index_written = self._index_written | {name}
                self._sync_index()
            elif name == "_core_routing_groups":
                # The Python core routes on the assigned groups from the next call on, while its
                # member map and group selectors stay as they were; router-core holds the groups and
                # the member map with its static settings.
                assigned: Final = as_str_mapping(value) or {}
                previous: Final = self._resolved
                self._resolved = dataclasses.replace(
                    previous,
                    routing_groups=tuple(g for g in assigned.values() if isinstance(g, RoutingGroup)),
                    routing_group_members=previous.routing_group_members
                    if previous.routing_group_members is not None
                    else member_map(previous.routing_groups),
                )
                self._settings = None
                self._rebuild()
            elif name == "_core_routing_plugins":
                self._sync_plugin_stage()
        elif name == "_core_prompt_caching_pin_cache":
            # Read when a prompt caching check is built, by `update_redis_cache` and by `flush_cache`.
            self._checks.prompt_caching_pin_cache = value  # pyright: ignore[reportAttributeAccessIssue]  # stored as given, as the Python core stores it
        elif name in _REBOUND_STATE:
            # A deprecated attribute write of the routing state (``router.cache = ...``): this core
            # reads the new object from now on, as the Python core reads its rebound attribute.
            object.__setattr__(self, _REBOUND_STATE[name], value)
            if name == "_core_cache":
                self._hosted._core_cache = value  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # usage reads come from the routing cache
            elif name == "_core_slack_alerting_logger":
                from litellm.integrations.custom_logger import CustomLogger

                # `_slack_alerting_loggers` reads the attribute each time: the new logger is the one bound.
                self._checks.slack_alerting_loggers = (value,) if isinstance(value, CustomLogger) else ()
        else:
            # Any other attribute is kept as written; `_core_async_get_available_deployment` (the
            # deprecated async pick) is then asked by every async pick (see `_pre_route`).
            object.__setattr__(self, name, value)

    async def select(  # pyright: ignore[reportIncompatibleMethodOverride]  # the RouterCore selection
        self,
        model: str,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the RouterCore signature
        *,
        kind: str = "pick",
        specific_deployment: bool | None = None,
        messages: object = None,
        input: object = None,
        health_check_probe: bool = False,
        batch_upload: bool = False,
        **_: object,  # kwargs-ok: the RouterCore select options this core ignores
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Pick a deployment for ``model`` without calling it (``pick``), list the deployments a
        pick would choose among (``healthy``), or pick one for a pass-through route
        (``pass_through``, which picks with the sync strategies, as Python does)."""
        if kind not in ("pick", "healthy", "pass_through"):
            raise ValueError(f"unknown select kind {kind!r}")
        self._refuse_deleted_reads({"num_retries": 0})  # a pick reads no retry budget
        request: Final = request_kwargs if request_kwargs is not None else {}
        selection: Final = _Selection(
            self,
            model,
            request,
            kind,
            specific_deployment,
            messages,
            input,
            probe=kind == "healthy" and health_check_probe,
            batch_upload=batch_upload,
        )
        # A pick that makes no call holds no least-busy reservation (`select` on the Python core).
        with picks_reserve(False):
            if kind == "pass_through":
                return selection.run_sync()
            return await selection.run()

    async def async_get_available_deployment_for_pass_through(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: the legacy signature
        messages: object = None,
        input: object = None,
        specific_deployment: bool | None = False,
    ) -> Mapping[str, object]:
        """The full healthy chain, then only ``use_in_pass_through`` deployments, picked with the
        async strategies."""
        selection: Final = _Selection(
            self, model, request_kwargs, "async_pass_through", specific_deployment, messages, input
        )
        picked: Final = await selection.run()
        return picked if isinstance(picked, Mapping) else {}

    def get_available_deployment(
        self,
        model: str,
        messages: object = None,
        input: object = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the legacy signature
    ) -> Mapping[str, object]:
        """The sync pick: no pre-routing, the sync strategies."""
        selection: Final = _Selection(
            self,
            model,
            request_kwargs if request_kwargs is not None else {},
            "sync_pick",
            specific_deployment,
            messages,
            input,
        )
        picked: Final = selection.run_sync()
        return picked if isinstance(picked, Mapping) else {}

    def get_available_deployment_for_pass_through(
        self,
        model: str,
        messages: object = None,
        input: object = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the legacy signature
    ) -> Mapping[str, object]:
        """Deprecated: ``select(model, kind="pass_through")``."""
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the router's deprecation notice
        )

        _note("get_available_deployment_for_pass_through", 'select(model, kind="pass_through")')
        selection: Final = _Selection(
            self,
            model,
            request_kwargs if request_kwargs is not None else {},
            "pass_through",
            specific_deployment,
            messages,
            input,
        )
        picked: Final = selection.run_sync()
        return picked if isinstance(picked, Mapping) else {}

    # -- endpoints -------------------------------------------------------------------------------

    async def call(self, op: str, request: Mapping[str, object]) -> object:
        """Run one endpoint with the request's keyword payload, as ``getattr(core, op)(**request)``."""
        member: Final[object] = getattr(self, op)  # pyright: ignore[reportAny]  # litellm's untyped helpers and request values
        if not callable(member):
            raise TypeError(f"{op!r} is not an endpoint")
        result: Final[object] = member(**request)
        return await result if inspect.isawaitable(result) else result

    async def acompletion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.acompletion(request_from_call("acompletion", args, kwargs))

    async def atext_completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.atext_completion(request_from_call("atext_completion", args, kwargs))

    async def aembedding(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._drive(plan_for("aembedding"), request_from_call("aembedding", args, kwargs))

    async def aresponses(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.generic(plan_for("aresponses"), request_from_call("aresponses", args, kwargs))

    async def aanthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.generic(plan_for("aanthropic_messages"), kwargs)

    async def anthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        """The proxy's ``/v1/messages`` route name for ``aanthropic_messages``; async."""
        return await self._endpoints.generic(plan_for("anthropic_messages"), kwargs)

    async def aimage_generation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._drive(plan_for("aimage_generation"), request_from_call("aimage_generation", args, kwargs))

    async def atranscription(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._drive(plan_for("atranscription"), request_from_call("atranscription", args, kwargs))

    async def aspeech(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        request: Final = dict(request_from_call("aspeech", args, kwargs))
        request.setdefault("voice", None)
        return await self._drive(plan_for("aspeech"), request)

    async def arerank(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        # `arerank` sets kwargs["input"] = input: the Python builtin, which reaches the deployment call.
        request: Final = {**request_from_call("arerank", args, kwargs), "input": input}
        return await self._drive(plan_for("arerank"), request)

    async def amoderation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.amoderation(request_from_call("amoderation", args, kwargs))

    def completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("completion"), request_from_call("completion", args, kwargs))

    def text_completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._endpoints.text_completion(request_from_call("text_completion", args, kwargs))

    def embedding(
        self,
        model: str,
        input: str | list[object],  # mutable-ok: the request field, as the Python core takes it
        is_async: bool | None = False,
        **kwargs: object,  # kwargs-ok: endpoint payload
    ) -> object:
        """``embedding``'s own signature; as there, ``is_async`` is named and goes no further."""
        return self._call_sync(plan_for("embedding"), {**kwargs, "model": model, "input": input})

    def image_generation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("image_generation"), request_from_call("image_generation", args, kwargs))

    # -- driving a call --------------------------------------------------------------------------

    def _start(
        self,
        plan: Plan,
        request: Mapping[str, object],
        overrides: Mapping[str, object] | None = None,
        *,
        prepared_already: bool = False,
    ) -> "tuple[RouterCall[_CallSteps], Mapping[str, object]]":
        self._refuse_deleted_reads({**request, **(overrides or {})})
        self._sync_routing_state()
        model: Final = request.get("model")
        if "model" not in request:
            raise TypeError(f"{plan.endpoint}() needs a model name")
        name: Final = (
            model if isinstance(model, str) else self._no_model_name() if model is None else self.routed_name(model)
        )
        field: Final = plan.prepare_field or "metadata"
        if request.get(field) is not None and not isinstance(request.get(field), Mapping):
            raise TypeError(f"{field} must be a mapping")
        prepared: Final = dict(request) if prepared_already else self._endpoints.prepare(plan, request, model)  # pyright: ignore[reportArgumentType]  # the name as given, which the metadata keeps
        if not prepared_already:
            # `async_function_with_fallbacks`: a selection the caller sent is not one the router made,
            # so it never exempts a fallback target from the access gate.
            from litellm.router_utils.fallback_event_handlers import clear_pre_routing_selection

            clear_pre_routing_selection(prepared)
        # `async_function_with_fallbacks` raises a pick for no model group as it is, with no fallback.
        routed: Final = {**(overrides or {}), "disable_fallbacks": True} if model is None else overrides
        return self._handle.start(plan.endpoint, self._routing_inputs(plan, request, prepared, name, routed)), prepared

    def service_logged(self, call_type: str, start_time: float, end_time: float, request: Mapping[str, object]) -> None:
        """The ``ServiceTypes.ROUTER`` success event of a timed step, as a task of the running loop."""
        import asyncio

        from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
        from litellm.types.services import ServiceTypes

        _ = asyncio.create_task(  # fire and forget, as the Python core schedules it
            self.service_logger_obj.async_service_success_hook(  # pyright: ignore[reportUnknownMemberType]  # its unused `event_metadata` is an untyped dict
                service=ServiceTypes.ROUTER,
                duration=end_time - start_time,
                call_type=call_type,
                start_time=start_time,
                end_time=end_time,
                parent_otel_span=get_parent_otel_span_from_kwargs(request),
            )
        )

    def track_deployment_pick(self, deployment_id: object, request: Mapping[str, object]) -> None:
        """``_track_deployment_metrics`` for a pick: ``_update_usage``'s count of the deployment's
        picks in the router cache, on this worker only. A default deployment served under a
        placeholder id counts under the id Python's copy of it holds."""
        from litellm._logging import verbose_router_logger
        from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs

        parent_otel_span: Final = get_parent_otel_span_from_kwargs(request)
        try:
            model_id: Final = (
                self._default_deployment_id()
                if deployment_id in (DEFAULT_DEPLOYMENT_FAILING_ID, DEFAULT_DEPLOYMENT_COPIED_ID)
                else deployment_id
            )
            if model_id is None:
                return
            counts: Final = self._pick_counts()
            count: Final = counts.get_cache(key=model_id, parent_otel_span=parent_otel_span, local_only=True)
            if count is None:
                counts.set_cache(key=model_id, value=1, local_only=True, ttl=60)  # only store for 60s
            else:
                counts.set_cache(key=model_id, value=count + 1, local_only=True)
        except Exception as e:  # noqa: BLE001  # as `_track_deployment_metrics` catches it
            verbose_router_logger.error("Error in _track_deployment_metrics: %s", e)

    def _pick_counts(self) -> _PickCounts:
        return self._cache

    def _default_deployment_id(self) -> object:
        return self._resolved.default_deployment.get("model_info", {}).get("id", None)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # read as Python reads it

    def routed_name(self, model: object, *, read_as_member: bool = False) -> str:
        """The name a ``model`` that is no string routes under, for the native calls, which take strings
        alone. Python's first reads raise for most values: the alias lookup hashes it (a sync pick reads
        ``model in _core_model_names`` first, ``read_as_member``), then ``_core_get_model_list`` reaches
        ``declared_provider``, which searches it. A value both accept (falsy, a tuple) is a name no
        deployment serves, and ``str(model)`` is how every error names it."""
        from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider

        if read_as_member:
            _ = model in frozenset[object]()
        else:
            _ = self._hosted._core_get_model_from_alias(model)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # raising is the point
        _ = inferred_provider(model)  # pyright: ignore[reportArgumentType]  # see above
        return str(model)

    def _no_model_name(self) -> str:
        """``model=None``: ``_common_checks_available_deployment`` picks for the first default
        fallback when there is one, and else for the name ``None``, which only a wildcard matches."""
        return next(
            (
                str(targets[0])
                for entry in _as_list(self._resolved.fallbacks)
                if (targets := _as_list((as_str_mapping(entry) or {}).get("*")))
            ),
            "None",
        )

    def _routing_inputs(
        self,
        plan: Plan,
        request: Mapping[str, object],
        prepared: Mapping[str, object],
        model: str,
        overrides: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        """The document router-core starts a call from: what it routes on, then what this call changes."""
        fields: dict[str, object] = {"model": model}  # mutable-ok: built up below, later keys win  # rebind-ok: same
        fields.update(self._routing_fields(prepared, plan.pick_input(prepared)[0]))
        fields.update(pinned_video_deployment(plan, request))
        fields.update(pinned_vector_store_file_deployment(plan, request))
        fields.update(unrouted_fields(plan))
        fields.update(failover_fields(plan))
        fields.update(scoped_fallback_fields(plan, request))
        fields.update(self._selection_fields(plan, prepared))
        fields.update(self._outcome_fields(plan, request))
        # The sync path never runs pre-routing; a custom strategy picks on both.
        if not plan.sync or self._custom_strategy is not None:
            fields["pre_routes"] = True
        if (mock := mock_failure(request)) is not None:
            fields["mock_failure"] = mock
        fields.update(self._retry_policy_raises(request))
        if overrides is not None and "num_retries" in overrides:
            fields.pop("uncounted_retries", None)
        fields.update(overrides or {})
        return _document(fields)

    def _outcome_fields(self, plan: Plan | None, request: Mapping[str, object]) -> Mapping[str, object]:
        """Which outcome callbacks run for the request: an endpoint that makes one attempt skips
        ``make_call``, and a recovered stream is async."""
        sync: Final = plan is not None and plan.sync
        make_call: Final = plan is None or not plan.single_attempt
        callbacks: Final = {
            **self.outcome_callbacks(request, sync=sync, make_call=make_call),
            **({"shared_logging": True} if _shares_logging(request) else {}),
        }
        return {"outcome_callbacks": callbacks} if callbacks else {}

    def _selection_fields(self, plan: Plan, request: Mapping[str, object]) -> Mapping[str, object]:
        """What an endpoint's pick reads beyond the routing fields: the narrowing keys the caller
        passed, the endpoint ``endpoint_selection`` names, and whether it asks for web search,
        which only an async pick filters by."""
        endpoint: Final = selection_endpoint(self._port, plan)
        web_search: Final = not plan.sync and self._picks_async() and is_web_search_request(request)
        # The generic handler `_route_capability_pin` names: a call only some providers serve
        # drops the others, and a Decisions call carrying media skips the deployments refusing it.
        handler: Final = plan.handler
        decisions_media: Final = handler in DECISIONS_MEDIA_CALL_TYPES and bool(
            request.get("audio") or request.get("videos")
        )
        return {
            **_request_narrowing(request),
            **({"endpoint": endpoint} if endpoint is not None else {}),
            **({"web_search": True} if web_search else {}),
            **({"route_handler": handler} if serves_route_capability(handler) else {}),
            **({"decisions_media": True} if decisions_media else {}),
            # `batch_create_routing`: a batch create and its file upload keep to the deployments that can run batches.
            **({"batch_create": True} if _routes_as_batch_create(plan, request) else {}),
            **({"batch_upload": True} if plan.kind == "create_file" and is_batch_upload(request) else {}),
            # `deployments_admitting_embedding_batch`: an embedding keeps to the deployments admitting its input.
            **({"embedding_batch": True} if plan.kind == "embedding" else {}),
        }

    def _picks_async(self) -> bool:
        return picks_async(normalize_strategy(self._resolved.routing_strategy))

    def _process_renamed(self, model: str, group: str) -> str:
        """``model`` as the error names it (``_common_checks_available_deployment``): the group a name
        resolved to by name, alias or routing group, or the group of the deployment an id names,
        renamed through the process ``model_alias_map``."""
        resolution: Final = self._handle.resolve(model)
        if resolution is None or resolution["kind"] not in ("Group", "Alias", "RoutingGroup", "Id"):
            return model
        named: Final = str(resolution["model_group"]) if resolution["kind"] == "Id" else group
        renames: Final = self._host.dynamic().model_alias_map
        return renames[named] if named in renames else named  # noqa: SIM401  # `renames` is any process mapping, read as Python reads it

    def _max_parallel_requests_client(self, deployment: Mapping[str, object], kwargs: Mapping[str, object]) -> object:
        """``_get_client(client_type="max_parallel_requests")``: the deployment's slot limit."""
        return max_parallel_requests_limit(self._port, deployment, kwargs)

    def _routable(
        self, model: str, request: Mapping[str, object], call_type: str | None = None
    ) -> Sequence[Mapping[str, object]]:
        """``_core_get_routable_deployments``: ``call_type`` names the endpoint it asks about."""
        endpoint: Final = endpoint_for_call(call_type)
        return self._handle.routable(
            model,
            _document(
                {
                    **_start_fields(request),
                    **({"endpoint": endpoint} if endpoint is not None else {}),
                    "wildcard_request": wildcard_request(model),
                }
            ),
        )

    def _routing_fields(self, request: Mapping[str, object], messages: object = None) -> Mapping[str, object]:
        """The request fields the Rust core routes on, and whether its tag stage may apply: the
        router's ``enable_tag_filtering``, the request's own, or a deployment's
        ``model_info.enable_tag_filtering`` (``get_deployments_for_tag``'s effective setting)."""
        filter_tags: Final = (
            self._resolved.enable_tag_filtering
            or request.get("enable_tag_filtering") is True
            or self._model_tag_filtering()
        )
        fields: dict[str, object] = dict(_start_fields(request))  # mutable-ok: built up below  # rebind-ok: same
        fields.update(self._group_policy_fields(request))
        pinned_grant: Final = pinned_deployment_wildcard_grant(
            request.get("model"), fields.get("allowed_models"), get_request_team_id(request), self.registry()
        )
        if pinned_grant is not None:
            fields["allowed_models"] = list(pinned_grant)
        if filter_tags:
            fields["filter_tags"] = True
        if request_carries_image(messages, request):
            fields["has_image"] = True
        return fields

    def _group_policies(self, request: Mapping[str, object]) -> tuple[object, GroupPolicies]:
        """The ``model_group_retry_policy`` the retry loop reads for this request (the request's own,
        even ``None``, replaces the router's: ``kwargs.pop(..., self.model_group_retry_policy)``),
        resolved for every group name the call may look up."""
        policies: Final = request.get("model_group_retry_policy", self._resolved.model_group_retry_policy)
        names: Final = self._policy_lookup_names(request)
        resolved: Final = group_policies(policies, names)
        if reads_as_policy(self._written_retry_policy):
            return policies, resolved
        # A group no entry names looks up the router's `retry_policy`, and that lookup raises.
        unnamed: Final = names - frozenset(resolved.document)
        return policies, dataclasses.replace(resolved, raising=resolved.raising | unnamed)

    def _policy_lookup_names(self, request: Mapping[str, object]) -> frozenset[str]:
        """The model names a retry policy lookup may run on (each hop looks up the name it calls):
        the requested one and every name in the fallback lists the walk may take."""
        resolved: Final = self._resolved
        normalized: Final = _request_start_fields(request)
        lists: Final = (
            resolved.fallbacks,
            resolved.context_window_fallbacks,
            resolved.content_policy_fallbacks,
            normalized.fallbacks,
            normalized.context_window_fallbacks,
            normalized.content_policy_fallbacks,
        )
        return frozenset((*_strings(request.get("model")), *_strings(lists)))

    def _retry_policy_raises(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """The groups whose ``model_group_retry_policy`` lookup raises on a hop's first failure, for
        router-core to end that hop with the error the lookup raises (see ``_Driver.take_events``)."""
        raising: Final = raising_groups(request, self._group_policies(request)[1])
        return {"retry_policy_raises": list(raising)} if raising else {}

    def _group_policy_fields(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """The request's ``model_group_retry_policy`` for router-core; a router policy that is not a
        mapping of policies rides on every request, resolved for it."""
        policies, resolved = self._group_policies(request)
        if "model_group_retry_policy" not in request and is_policy_per_group(policies):
            return {}
        return {"model_group_retry_policy": _plain(resolved.document)}

    def _model_tag_filtering(self) -> bool:
        """Whether any deployment sets ``model_info.enable_tag_filtering``, cached per registry version."""
        cached: Final = self._tag_filtering_models
        if cached is not None and cached[0] == self._version:
            return cached[1]
        found: Final = any(
            (as_str_mapping((as_str_mapping(record) or {}).get("model_info")) or {}).get("enable_tag_filtering")
            is not None
            for record in self._handle.deployments()
        )
        self._tag_filtering_models = (self._version, found)
        return found

    async def _drive(
        self,
        plan: Plan,
        request: Mapping[str, object],
        overrides: Mapping[str, object] | None = None,
        *,
        passthrough: "Callable[[Mapping[str, object], str], Awaitable[object]] | None" = None,
        prepared_already: bool = False,
    ) -> object:
        routed: Final = (
            overrides if passthrough is None else {**(overrides or {}), "passthrough_on_no_deployment": True}
        )
        # `owning_picks`: least-busy frees every pick this request made but never started.
        with (
            request_parent_otel_span(get_parent_otel_span_from_kwargs(request)),
            owning_picks(self._selectors.reservations, call_id_of(CALL_IDENTITY.validate_python(request))),
        ):
            call, prepared = self._start(plan, request, routed, prepared_already=prepared_already)
            arm_compaction_state(self._port, plan, prepared)  # pyright: ignore[reportArgumentType]  # the call's own dict
            with (
                picks_reserve(plan.endpoint not in _UNRESERVED_OPS),
                self._usage_read_scope(str(prepared.get("model")), prepared),
            ):
                driver, step = await self._run(plan, call, prepared, fresh=True, passthrough=passthrough)
            if step["kind"] == "failed":
                raise driver.failed(step)
            return await driver.finish_async(step)

    async def _run(
        self,
        plan: Plan,
        call: "RouterCall[_CallSteps]",
        prepared: Mapping[str, object],
        cut_deployment: object = None,
        stream_failure: tuple[int, BaseException] | None = None,
        *,
        fresh: bool = False,
        passthrough: "Callable[[Mapping[str, object], str], Awaitable[object]] | None" = None,
    ) -> "tuple[_Driver, Mapping[str, object]]":
        """Drive ``call`` to its ``done`` or ``failed`` step, answering every step on the host.
        ``cut_deployment`` is the attempt whose stream failed, for a recovery call, and
        ``stream_failure`` the level it served with the failure a fallback walk re-enters from.
        ``fresh`` is a call from the top, through the retry and fallback wrappers, and
        ``passthrough`` serves an attempt whose pick failed."""
        driver: Final = _Driver(
            self, plan, prepared, cut_deployment, stream_failure, fresh=fresh, passthrough=passthrough
        )
        try:
            return await self._run_steps(driver, call)
        except BaseException:
            driver.close_route_phase()
            raise

    async def _run_steps(
        self, driver: "_Driver", call: "RouterCall[_CallSteps]"
    ) -> "tuple[_Driver, Mapping[str, object]]":
        """``_run``'s loop over the call's steps."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = call.next(outcome)
            driver.take_events(step.get("events"))
            # `run_async_fallback` stamps a served response before it logs the hop's success: `finish`
            # runs what a served call still has pending.
            if step["kind"] != "done":
                await await_all(driver.outcomes.take_pending())
            match step["kind"]:
                case "attempt":
                    outcome = await driver.attempt(step)
                case "sleep":
                    if (outlasted := _error_outlasting_total_timeout(driver, _sleep_seconds(step))) is not None:
                        driver.close_route_phase()
                        raise outlasted
                    await self._host.sleep(_sleep_seconds(step))
                    outcome = None
                case "allow_fallback":
                    outcome = await driver.allow_fallback(str(step["target"]))
                case "refuse_fallbacks":
                    outcome = driver.refuse_fallbacks(step)
                case "cache":
                    outcome = await self._answer_cache(step)
                case "select":
                    outcome = await driver.select(step)
                case "pre_route":
                    outcome = await driver.pre_route(step)
                case "done" | "failed":
                    driver.close_route_phase()
                    return driver, step
                case other:  # pyright: ignore[reportUnnecessaryComparison]  # a kind the bridge returns but the stub does not list
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def _call_sync(
        self, plan: Plan, request: Mapping[str, object], overrides: Mapping[str, object] | None = None
    ) -> object:
        # The sync routed calls pick with `picks_reserve(True)` inside `owning_picks`.
        with (
            request_parent_otel_span(get_parent_otel_span_from_kwargs(request)),
            owning_picks(self._selectors.reservations, call_id_of(CALL_IDENTITY.validate_python(request))),
        ):
            call, prepared = self._start(plan, request, overrides)
            with picks_reserve(plan.endpoint not in _UNRESERVED_OPS):
                driver, step = self._run_sync(plan, call, prepared, fresh=True)
            if step["kind"] == "failed":
                raise driver.failed(step)
            return driver.finish(step)

    def _run_sync(
        self,
        plan: Plan,
        call: "RouterCall[_CallSteps]",
        prepared: Mapping[str, object],
        cut_deployment: object = None,
        stream_failure: tuple[int, BaseException] | None = None,
        *,
        fresh: bool = False,
    ) -> "tuple[_Driver, Mapping[str, object]]":
        driver: Final = _Driver(self, plan, prepared, cut_deployment, stream_failure, sync=True, fresh=fresh)
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = call.next(outcome)
            driver.take_events(step.get("events"))
            if step["kind"] != "done":  # see _run
                driver.log_outcomes_sync()
            match step["kind"]:
                case "attempt":
                    outcome = driver.attempt_sync(step)
                case "sleep":
                    self._host.sleep_sync(_sleep_seconds(step))
                    outcome = None
                case "allow_fallback":
                    from litellm.litellm_core_utils import asyncify

                    outcome = asyncify.run_async_function(driver.allow_fallback, str(step["target"]))  # pyright: ignore[reportUnknownMemberType]  # the untyped sync bridge
                case "refuse_fallbacks":
                    outcome = driver.refuse_fallbacks(step)
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "select":
                    outcome = driver.select_sync(step)
                case "pre_route":
                    outcome = driver.pre_route_sync(step)
                case "done" | "failed":
                    return driver, step
                case other:  # pyright: ignore[reportUnnecessaryComparison]  # a kind the bridge returns but the stub does not list
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    # -- stream recovery (the host's stream wrappers, router_rust.invoke.streaming) ---------------------

    def _recovery_call(
        self,
        recovery: str,
        op: str,
        request: Mapping[str, object],
        origin: Mapping[str, object],
        failure: BaseException,
    ) -> "RouterCall[_CallSteps]":
        start: Final = self._handle.retry_stream if recovery == "retry" else self._handle.fallback_stream
        plan: Final = generic_plan(op)
        routing: Final = _document(
            {
                "model": _routed_model(request),
                **self._routing_fields(request),
                **(self._selection_fields(plan, request) if plan is not None else {}),
                **self._retry_policy_raises(request),
                **continuation_routing(self, request),
                **self._outcome_fields(plan, request),
            }
        )
        return start(op, routing, _document(origin), attempt_failure(failure))

    def stream_policy(
        self, request: Mapping[str, object], origin: Mapping[str, object], trigger: BaseException | None
    ) -> Mapping[str, object]:
        routing: Final = _document(
            {
                "model": _routed_model(request),
                **_start_fields(request),
                **self._group_policy_fields(request),
                **self._retry_policy_raises(request),
            }
        )
        kind: Final = None if trigger is None else str(attempt_failure(trigger)["kind"])
        return self._handle.stream_policy(routing, _document(origin), kind)

    async def recover_stream(
        self,
        recovery: str,
        op: str,
        request: Mapping[str, object],
        origin: Mapping[str, object],
        failure: BaseException,
    ) -> object:
        """The fallback walk for a stream that failed before content: the fallback's response with the
        router's headers, or the error the walk ended on."""
        call: Final = self._recovery_call(recovery, op, request, origin, failure)
        driver, step = await self._run(
            plan_for(op), call, request, origin.get("deployment_id"), (_origin_level(origin), failure)
        )
        if step["kind"] == "failed":
            raise driver.failed(step)
        return await driver.finish_async(step)

    def recover_stream_sync(
        self,
        recovery: str,
        op: str,
        request: Mapping[str, object],
        origin: Mapping[str, object],
        failure: BaseException,
    ) -> object:
        call: Final = self._recovery_call(recovery, op, request, origin, failure)
        driver, step = self._run_sync(
            plan_for(op), call, request, origin.get("deployment_id"), (_origin_level(origin), failure)
        )
        if step["kind"] == "failed":
            raise driver.failed(step)
        return driver.finish(step)

    async def retry_stream(
        self, op: str, request: Mapping[str, object], origin: Mapping[str, object], failure: BaseException
    ) -> "tuple[object, int, int] | BaseException":
        """A same-group retry of a stream that failed before content: the retried response (without the
        router's headers) with the retry count and budget, or the newest failure (``failure`` itself
        when no retry ran)."""
        call: Final = self._recovery_call("retry", op, request, origin, failure)
        driver, step = await self._run(plan_for(op), call, request, origin.get("deployment_id"))
        await await_all(driver.outcomes.take_pending())
        if step["kind"] == "failed":
            return driver.last_error if driver.last_error is not None else failure
        attempted: Final = step.get("attempted_retries")
        budget: Final = step.get("max_retries")
        return (
            driver.response,
            attempted if isinstance(attempted, int) else 0,
            budget if isinstance(budget, int) else 0,
        )

    def params_sent_to_deployment(self, request: Mapping[str, object], deployment_id: object) -> Mapping[str, object]:
        """The request as a deployment received it: its params, the router defaults, its tool choice, then
        the request (``_params_sent_to_deployment``)."""
        record: Final = self._handle.deployment(deployment_id) if isinstance(deployment_id, str) else None
        params: Final = as_str_mapping((as_str_mapping(record) or {}).get("litellm_params")) or {}
        defaults: Final = {k: v for k, v in self._resolved.default_litellm_params.items() if v is not None}
        tool_choice: Final = {"tool_choice": params["tool_choice"]} if params.get("tool_choice") is not None else {}
        return MappingProxyType({**params, **defaults, **tool_choice, **request})

    def _drive_sync_steps(self, call: "RouterCall[_FailureReportSteps | _SuccessReportSteps]") -> Mapping[str, object]:
        """Drive a call whose only steps are cache ops (``report_failure``) to its end."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = call.next(outcome)
            match step["kind"]:
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "done":
                    return step
                case "failed":
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:  # pyright: ignore[reportUnnecessaryComparison]  # a kind the bridge returns but the stub does not list
                    raise RuntimeError(f"router-core asked for {other!r} while recording an outcome")

    async def _drive_cache_steps(
        self, call: "RouterCall[_FailureReportSteps | _SuccessReportSteps]"
    ) -> Mapping[str, object]:
        """``_drive_sync_steps`` answered through the async cache (``report_success``)."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = call.next(outcome)
            match step["kind"]:
                case "cache":
                    outcome = await self._answer_cache(step)
                case "done":
                    return step
                case "failed":
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:  # pyright: ignore[reportUnnecessaryComparison]  # a kind the bridge returns but the stub does not list
                    raise RuntimeError(f"router-core asked for {other!r} while recording an outcome")

    # -- pre-route steps -------------------------------------------------------------------------

    async def _pre_route(
        self,
        step: Mapping[str, object],
        request: Mapping[str, object],
        messages: object = _NO_INPUT,
        input: object = _NO_INPUT,
        specific_deployment: object = _NO_INPUT,
        *,
        pass_through: bool = False,
        call_type: str | None = None,
    ) -> object:
        """The host's pre-routing of one pick (router_rust.steps.prerouting): the model group to resolve
        instead of the step's ``model``, a custom strategy's deployment, or ``None``. ``pass_through``
        is ``_async_get_available_deployment_for_pass_through``, which runs the hook whatever the
        router's strategy and which no replaced pick answers."""
        assigned: Final[object] = None if pass_through else self.__dict__.get("_core_async_get_available_deployment")
        if callable(assigned):
            # The async pick assigned through the deprecated attribute answers, as Python's
            # replaced `_core_async_get_available_deployment` does.
            answer: Final[object] = assigned(
                model=str(step.get("model")),
                request_kwargs=request,
                messages=request.get("messages") if messages is _NO_INPUT else messages,
                input=request.get("input") if input is _NO_INPUT else input,
                specific_deployment=request.get("specific_deployment")
                if specific_deployment is _NO_INPUT
                else specific_deployment,
            )
            chosen: Final[object] = await answer if inspect.isawaitable(answer) else answer
            return _custom_pick(chosen)
        custom: Final = None if pass_through else self._custom_strategy
        if custom is not None:
            picked: Final[object] = await custom.async_get_available_deployment(  # pyright: ignore[reportUnknownMemberType]  # the strategy's untyped pick
                model=str(step.get("model")),
                messages=request.get("messages") if messages is _NO_INPUT else messages,  # pyright: ignore[reportArgumentType]  # the request's messages, as given
                input=request.get("input") if input is _NO_INPUT else input,  # pyright: ignore[reportArgumentType]  # the request's input, as given
                specific_deployment=request.get("specific_deployment")
                if specific_deployment is _NO_INPUT
                else specific_deployment,  # pyright: ignore[reportArgumentType]  # same
                request_kwargs=request,  # pyright: ignore[reportArgumentType]  # the request kwargs, as given
            )
            return _custom_pick(picked)
        if not isinstance(request, dict):
            return None
        depth: Final = step.get("fallback_depth")
        if isinstance(depth, int) and depth > 0:
            # A fallback hop routes on its own copy of the metadata buckets (`run_async_fallback`'s
            # kwargs): what its hook clears and stamps leaves the caller's metadata as the first hop left it.
            for name in ("metadata", "litellm_metadata"):
                bucket = request.get(name)
                if isinstance(bucket, dict):
                    request[name] = dict(bucket)  # pyright: ignore[reportUnknownArgumentType]  # the request's metadata dict
        r: Final = self._resolved
        owner: Final[object] = self.__dict__.get("_core_owner")
        public: Final[object] = owner() if callable(owner) else None
        routed: Final[object] = await self._prerouter.pre_route(
            str(step.get("model")),
            request,  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs the hook stamps
            messages=request.get("messages") if messages is _NO_INPUT else messages,
            input=request.get("input") if input is _NO_INPUT else input,
            fallback_depth=depth if isinstance(depth, int) else 0,
            router_strategy=normalize_strategy(r.routing_strategy),
            pass_through=pass_through,
            call_type=call_type,
            enable_tag_filtering=r.enable_tag_filtering,
            match_any=r.tag_filtering_match_any,
            public_router=public if public is not None else self,
            run_plugins=partial(
                self._run_plugins, request, request.get("messages") if messages is _NO_INPUT else messages
            )  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs the plugins stamp
            if self._hosted._core_routing_plugins  # pyright: ignore[reportPrivateUsage]  # the live plugin list
            else None,
        )
        return _served_as(routed, request)

    async def _run_plugins(
        self,
        request: dict[str, object],  # mutable-ok: the plugins stamp their outcome on the caller's request
        messages: object,
        model: str,
    ) -> None:
        """``_run_routing_plugins`` over the pre-routing hook's session-resolved ``model``."""
        await run_plugins(
            tuple(self._hosted._core_routing_plugins),  # pyright: ignore[reportPrivateUsage]  # the live plugin list
            self._hosted,  # pyright: ignore[reportArgumentType]  # the host registry serves the listing
            self._resolved.model_group_alias,
            model,
            request,
            messages,
        )

    def _custom_pick_sync(
        self,
        step: Mapping[str, object],
        request: Mapping[str, object],
        messages: object = _NO_INPUT,
        input: object = _NO_INPUT,
        specific_deployment: object = _NO_INPUT,
    ) -> object:
        """A custom strategy's sync pick (``get_available_deployment``); without one, no pre-routing."""
        custom: Final = self._custom_strategy
        if custom is None:
            return None
        picked: Final[object] = custom.get_available_deployment(  # pyright: ignore[reportUnknownMemberType]  # the strategy's untyped pick
            model=str(step.get("model")),
            messages=request.get("messages") if messages is _NO_INPUT else messages,  # pyright: ignore[reportArgumentType]  # see _pre_route
            input=request.get("input") if input is _NO_INPUT else input,  # pyright: ignore[reportArgumentType]  # same
            specific_deployment=request.get("specific_deployment")
            if specific_deployment is _NO_INPUT
            else specific_deployment,  # pyright: ignore[reportArgumentType]  # same
            request_kwargs=request,  # pyright: ignore[reportArgumentType]  # same
        )
        return _custom_pick(picked)

    # -- select steps ----------------------------------------------------------------------------

    def _strategy_inputs(self, step: Mapping[str, object], request: Mapping[str, object]) -> "_StrategyInputs | None":
        """The strategy, its selector, the name it keys state by and the candidate deployments of a
        ``strategy`` select step; ``None`` for a stage the host passes through."""
        if step.get("stage") != "strategy":
            return None
        strategy: Final = str(step.get("strategy"))
        group: Final = step.get("group")
        selector: Final = self._selectors.selector_for(
            strategy, str(step.get("scope")), str(group) if group is not None else None, request
        )
        # Candidates sharing an id come with their own records, so the pick can land on any of them.
        records: Final = step.get("records")
        candidates: Final = (
            [dict(as_str_mapping(record) or {}) for record in _as_list(records)]
            if records is not None
            else [
                dict(as_str_mapping(self._handle.deployment(str(id))) or {}) for id in _as_list(step.get("candidates"))
            ]
        )
        return strategy, selector, str(step.get("model")), candidates

    def _records_of(self, ids: object) -> list[dict[str, object]]:  # mutable-ok: the deployment dicts filters take
        return [dict(as_str_mapping(self._handle.deployment(str(id))) or {}) for id in _as_list(ids)]

    async def _answer_tags(self, step: Mapping[str, object], request: Mapping[str, object]) -> object:
        """``get_deployments_for_tag`` over the step's candidates (bridged: not ported to Rust yet)."""
        from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
        from litellm.router_strategy.tag_based_routing import get_deployments_for_tag

        kept: Final = await get_deployments_for_tag(
            ctx=RoutingContext(self),  # pyright: ignore[reportArgumentType]  # the filter reads registry() and settings()
            model=str(step.get("model")),
            healthy_deployments=self._records_of(step.get("candidates")),  # pyright: ignore[reportArgumentType]  # deployment dicts
            request_kwargs=dict(request),  # pyright: ignore[reportArgumentType]  # the request kwargs
            metadata_variable_name=get_metadata_variable_name_from_kwargs(dict(request)),
            request_deployments=self._records_of(step.get("pool")),  # pyright: ignore[reportArgumentType]  # deployment dicts
        )
        return [found for record in kept if (found := picked_id(record)) is not None]

    def _pre_call_facts(
        self, step: Mapping[str, object], request: Mapping[str, object], messages: object, input: object
    ) -> PreCallFacts:
        hosted: Final = self._hosted

        def record(id: str) -> Mapping[str, object]:
            return hosted._core_get_model_info(id) or as_str_mapping(self._handle.deployment(id)) or {}  # pyright: ignore[reportPrivateUsage]  # see __init__

        return PreCallFacts(self._host, hosted, step, record, request, messages, input)  # pyright: ignore[reportArgumentType]  # the host registry serves the reads

    def _text_only(self, step: Mapping[str, object]) -> list[str]:  # mutable-ok: the step's answer
        """The candidates that declare no vision support (bridged: needs the cost map)."""
        group: Final = str(step.get("model_group"))
        return [
            str(id)
            for id in _as_list(step.get("candidates"))
            if declares_no_vision(as_str_mapping(self._handle.deployment(str(id))) or {}, group)
        ]

    def _route_incapable(self, step: Mapping[str, object]) -> list[str]:  # mutable-ok: the step's answer
        """The candidates whose provider cannot serve the call the step's handler names."""
        group: Final = str(step.get("model_group"))
        handler: Final = str(step.get("handler"))
        return [
            str(id)
            for id in _as_list(step.get("candidates"))
            if cannot_serve_route(as_str_mapping(self._handle.deployment(str(id))) or {}, group, handler)
        ]

    def _media_refusing(
        self, step: Mapping[str, object], request: Mapping[str, object]
    ) -> list[str]:  # mutable-ok: the step's answer
        """The candidates that refuse the audio or videos a Decisions call carries."""
        group: Final = str(step.get("model_group"))
        defaults: Final = {k: v for k, v in self._resolved.default_litellm_params.items() if v is not None}
        sent: Final = {**defaults, **request}
        audio: Final = bool(request.get("audio"))
        videos: Final = bool(request.get("videos"))
        return [
            str(id)
            for id in _as_list(step.get("candidates"))
            if refuses_decisions_media(
                as_str_mapping(self._handle.deployment(str(id))) or {}, group, sent, audio=audio, videos=videos
            )
        ]

    def _override_selector(self, step: Mapping[str, object], request: Mapping[str, object]) -> object:
        """The request's own strategy selector when it overrides the router's (``_override_selectors``),
        whose pre-call check is in no callback list."""
        strategy: Final = str(step.get("strategy"))
        selector: Final = self._selectors.selector_for(strategy, "override", None, request)
        return selector if selector is not None and self._selectors.is_override(strategy, selector) else None

    def _embedding_batch(
        self, step: Mapping[str, object], request: Mapping[str, object], input: object, *, tags: bool
    ) -> list[str]:  # mutable-ok: the step's answer
        """``deployments_admitting_embedding_batch`` over the candidates; when none admits the input,
        nothing while a routable, unpaused cooling deployment does (the pick waits out its cooldown),
        else the candidates with the largest limit, so the 400 names the most the group accepts."""
        from litellm.router_utils.embedding_batch_size_check import (
            deployments_admitting_embedding_batch,
            deployments_when_none_admits_embedding_batch,
            embedding_deployment_selection,
        )

        sent: Final = request.get("input", input)
        candidates: Final = self._records_of(step.get("candidates"))
        with embedding_deployment_selection():
            admitting: Final = deployments_admitting_embedding_batch(candidates, sent, dict(request))
        if admitting is not None:
            return [found for record in admitting if (found := picked_id(record)) is not None]
        pool: Final = self._records_of(step.get("pool"))
        pool_ids: Final = [found for record in pool if (found := picked_id(record)) is not None]
        cooling: Final = [id for id, _ in self._cooldowns.get_active_cooldowns(pool_ids, None)]  # pyright: ignore[reportArgumentType]  # a list of ids
        routable: Final = _region_routable(pool, request, True, self._resolved.enable_pre_call_checks)
        routable_cooling: Final = (
            self._tag_routable_ids(str(step.get("model")), pool, routable, frozenset(cooling), request)
            if tags
            else frozenset(id for record in routable if (id := picked_id(record)) in cooling)
        )
        kept: Final = deployments_when_none_admits_embedding_batch(
            candidates, pool, routable_cooling, sent, dict(request)
        )
        return [found for record in kept if (found := picked_id(record)) is not None]

    async def _answer_select(
        self,
        step: Mapping[str, object],
        request: Mapping[str, object],
        messages: object,
        input: object,
        *,
        pass_through: bool = False,
    ) -> object:
        if step.get("stage") == "pre_call":
            return await self._pre_call_facts(step, request, messages, input).answer_async()
        if step.get("stage") == "routing_plugins":
            # The plugins already ran in this pick's pre_route step (see `_pre_route`), on the
            # session-resolved name and before a strategy was selected, as Python's hook runs them.
            return candidate_models(request)
        if step.get("stage") == "tags":
            return await self._answer_tags(step, request)
        if step.get("stage") == "text_only":
            return self._text_only(step)
        if step.get("stage") == "route_incapable":
            return self._route_incapable(step)
        if step.get("stage") == "media_refusing":
            return self._media_refusing(step, request)
        if step.get("stage") == "embedding_batch":
            return self._embedding_batch(step, request, input, tags=True)
        if step.get("stage") == "override_check":
            selector = self._override_selector(step, request)
            if selector is not None:
                for record in self._records_of(step.get("candidates")):
                    await selector.async_pre_call_check(dict(record), None)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # every selector is a CustomLogger
            return step["candidates"]
        if step.get("stage") == "provider_request":
            return _provider_request(step)
        inputs: Final = self._strategy_inputs(step, request)
        if inputs is None:
            return step["candidates"]
        strategy, selector, model, candidates = inputs
        pinned: Final = bool(request.get("_encrypted_content_affinity_pinned")) and len(candidates) == 1
        if selector is None and not pinned:
            return []
        # The strategy's service event: `_async_get_available_deployment` times it for the strategies
        # with an async pick, `_async_get_available_deployment_for_pass_through` for any, on another clock.
        timed: Final = not pinned and strategy != "simple-shuffle" and (pass_through or self._picks_async())
        clock: Final = perf_counter if pass_through else unix_time
        start_time: Final = clock()
        # `_async_get_available_deployment`: a pin by encrypted content takes no strategy's pick.
        picked: Final = (
            candidates[0]
            if pinned
            else await _pick_with_prefetched_usage(
                strategy,
                selector,  # pyright: ignore[reportArgumentType]  # a selector from selector_for
                model,
                candidates,
                messages,
                input,
                dict(request),
                # `_async_get_available_deployment` reserves a request's pick unless told not to;
                # the pass-through pick only when told to.
                reserve=pick_reserves(default=not pass_through) and bool(request),
            )
        )
        if picked is not None and selector is not None and self._selectors.is_override(strategy, selector):
            await selector.async_pre_call_check(dict(as_str_mapping(picked) or {}), None)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # every selector is a CustomLogger
        if picked is not None and timed:
            self.service_logged("<routing_strategy>.async_get_available_deployments", start_time, clock(), request)
        return _pick_answer(step, candidates, picked)

    def _answer_select_sync(
        self, step: Mapping[str, object], request: Mapping[str, object], messages: object, input: object
    ) -> object:
        if step.get("stage") == "pre_call":
            return self._pre_call_facts(step, request, messages, input).answer()
        if step.get("stage") == "routing_plugins":
            return candidate_models(request)
        if step.get("stage") == "callback_filters":
            # Only the async candidate lists run the callback filters.
            return step["candidates"]
        if step.get("stage") == "text_only":
            return self._text_only(step)
        if step.get("stage") == "route_incapable":
            return self._route_incapable(step)
        if step.get("stage") == "media_refusing":
            return self._media_refusing(step, request)
        if step.get("stage") == "embedding_batch":
            return self._embedding_batch(step, request, input, tags=False)
        if step.get("stage") == "override_check":
            selector = self._override_selector(step, request)
            if selector is not None:
                for record in self._records_of(step.get("candidates")):
                    selector.pre_call_check(dict(record))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # every selector is a CustomLogger
            return step["candidates"]
        if step.get("stage") == "provider_request":
            return _provider_request(step)
        inputs: Final = self._strategy_inputs(step, request)
        if inputs is None:
            return step["candidates"]
        strategy, selector, model, candidates = inputs
        if selector is None:
            return []
        picked: Final = pick_sync(
            strategy,
            selector,  # pyright: ignore[reportArgumentType]  # a selector from selector_for
            model,
            candidates,
            messages,
            input,
            dict(request),
            # The sync picks reserve only when a routed call site says so (`picks_reserve(True)`).
            reserve=pick_reserves(default=False) and bool(request),
        )
        if picked is not None and self._selectors.is_override(strategy, selector):
            selector.pre_call_check(dict(as_str_mapping(picked) or {}))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # see above
        return _pick_answer(step, candidates, picked)

    # -- cache steps -----------------------------------------------------------------------------

    async def _cooldowns_with_usage(self, step: CooldownsStep) -> object:
        """A usage-based-routing-v2 pick reads its tpm/rpm counters alongside its cooldowns (``_usage_read_scope``)."""
        usage_read: Final = _USAGE_READ.get()
        if usage_read is None:
            return await self._stores.answer(step)
        batch, deployments = usage_read
        stored, _ = await asyncio.gather(self._stores.answer(step), batch.read_usage(deployments, None))
        return stored

    async def _answer_cache(self, raw: Mapping[str, object]) -> object:
        step: Final = CACHE_STEP.validate_python(raw)
        with service_target(service_target_of(step)):
            match step:
                case CooldownsStep():
                    return _plain(await self._cooldowns_with_usage(step))
                case OutcomesStep() | CountSuccessStep() | CountFailureStep():
                    return self._answer_outcomes(step)
                case _:
                    return _plain(await self._stores.answer(step))

    def _answer_outcomes(self, step: LocalStep) -> object:
        """This worker's per-minute outcomes, kept where the Python router's cooldown handlers keep them."""
        ctx: Final = self._core_ctx
        match step:
            case OutcomesStep(id=model_id):
                return [
                    get_deployment_successes_for_current_minute(ctx, model_id),
                    get_deployment_failures_for_current_minute(ctx, model_id),
                ]
            case CountSuccessStep(id=model_id):
                _ = increment_deployment_successes_for_current_minute(ctx, model_id)
                return None
            case CountFailureStep(id=model_id):
                increment_deployment_failures_for_current_minute(ctx, model_id)
                return None
            case _:
                assert_never(step)

    def run_routing_cache_sync(self, read_or_write: "Callable[[], Coroutine[object, object, object]]") -> object:
        """A routing-cache coroutine on a sync call, whose `make_call` runs on the one event loop
        `run_async_function` opened for the whole call. Without Redis the coroutine never suspends, so
        it runs here: a loop of its own would hand the GIL to the logging thread, which then reads the
        response before the router's headers are on it."""
        if self._cache.redis_cache is not None:
            from litellm.litellm_core_utils import asyncify

            return asyncify.run_async_function(read_or_write)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the untyped sync bridge
        running: Final = read_or_write()
        try:
            while True:
                _ = running.send(None)
        except StopIteration as finished:
            return finished.value  # pyright: ignore[reportAny]  # the coroutine's result

    @property
    def _stores(self) -> RoutingStores:
        """The stores router-core's cache steps reach, over the cooldowns as last assigned."""
        return RoutingStores(self._cooldowns, self._usage, self._health_checks)

    def io_token_usage_sync(self, model_group: str) -> tuple[int | None, int | None]:
        """``get_model_group_io_token_usage``, whose counters stay in the routing cache, on a sync call."""
        read: Final = self.run_routing_cache_sync(lambda: self._hosted.get_model_group_io_token_usage(model_group))
        return _IO_TOKEN_USAGE.validate_python(read)

    def _answer_cache_sync(self, raw: Mapping[str, object]) -> object:
        step: Final = CACHE_STEP.validate_python(raw)
        with service_target(service_target_of(step)):
            match step:
                case OutcomesStep() | CountSuccessStep() | CountFailureStep():
                    return self._answer_outcomes(step)
                case _:
                    return _plain(self._stores.answer_sync(step))

    # -- errors the core raises when no attempt failed ------------------------------------------

    def _routable_cooled_info(
        self,
        step: Mapping[str, object],
        group: str,
        cooling: Sequence[str],
        request: Mapping[str, object],
        *,
        has_input: bool,
        tags: bool,
    ) -> tuple[CooldownDeploymentInfo, ...] | None:
        """``RouterRateLimitError.cooldown_deployments``: the cooling deployments the request could
        route to (region and, on an async pick, tag routing over the deployments the core listed
        before its cooldown filter); ``None`` when the strategy refused every candidate."""
        request_ids: Final = step.get("error_request_ids")
        if request_ids is None:
            return None
        records: Final = self._records_of(request_ids)
        routable: Final = _region_routable(records, request, has_input, self._resolved.enable_pre_call_checks)
        route: Final = (
            self._tag_routable_ids(group, records, routable, frozenset(cooling), request)
            if tags
            else frozenset(id for record in routable if (id := picked_id(record)) is not None)
        )
        return get_cooldown_deployment_info(RoutingContext(self), route, cooling)  # pyright: ignore[reportArgumentType]  # reads registry()

    def _tag_routable_ids(
        self,
        group: str,
        pool: Sequence[Mapping[str, object]],
        routable: Sequence[Mapping[str, object]],
        cooled: frozenset[str],
        request: Mapping[str, object],
    ) -> frozenset[str]:
        """The cooling deployments in ``routable`` that tag routing would send the request to, each
        pick counted along with the picks it falls back to once that one is unavailable, judged
        without recording a tag match on the request."""
        from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
        from litellm.router_strategy.tag_based_routing import get_deployments_for_tag

        field: Final = get_metadata_variable_name_from_kwargs(dict(request))
        metadata: Final = request.get(field)
        if field in request and not isinstance(metadata, Mapping):
            return frozenset()
        tag_request: Final = {**request, field: {**metadata}} if isinstance(metadata, Mapping) else dict(request)
        found = frozenset[str]()  # rebind-ok: gains each pass's tag routing picks
        remaining = [  # rebind-ok: each pass drops what tag routing just picked
            record for record in routable if picked_id(record) in cooled
        ]
        while remaining:
            try:
                # Tag routing awaits nothing: its coroutine finishes on the first send.
                picked = _finished(
                    get_deployments_for_tag(
                        ctx=RoutingContext(self),  # pyright: ignore[reportArgumentType]  # reads registry() and settings()
                        model=group,
                        healthy_deployments=remaining,  # pyright: ignore[reportArgumentType]  # deployment dicts
                        request_kwargs=tag_request,  # pyright: ignore[reportArgumentType]  # the request kwargs
                        metadata_variable_name=field,
                        request_deployments=pool,  # pyright: ignore[reportArgumentType]  # deployment dicts
                    )
                )
            except ValueError:  # tag routing sends this request to none of the rest
                break
            picked_ids = frozenset(
                id
                for record in (picked if isinstance(picked, list | tuple) else ())
                if (id := picked_id(record)) is not None
            )  # pyright: ignore[reportArgumentType]  # deployment dicts
            rest = [record for record in remaining if picked_id(record) not in picked_ids]
            if len(rest) == len(remaining):
                break
            found |= picked_ids
            remaining = rest
        return found

    def _failure_error(  # noqa: C901  # one branch per failure Python tells apart
        self,
        step: Mapping[str, object],
        model: str,
        last: BaseException | None,
        *,
        pass_through: str = "",
        request: Mapping[str, object] | None = None,
        has_input: bool = True,
        tags: bool = True,
    ) -> BaseException:
        """The error a call or pick that failed before any attempt raises; ``pass_through`` names
        the selection kind (``pass_through``/``async_pass_through``) whose errors it words."""
        import litellm.exceptions as litellm_exceptions

        kind: Final = step.get("error_kind")
        aliases: Final = self._resolved.model_group_alias
        refused: Final = selection_refusal(step, model, request or {})
        if refused is not None:
            return refused
        if kind == "paused":
            # The message names the deployment's litellm model, not the id that was asked for; the
            # default deployment's copy is named by the requested name.
            record: Final = as_str_mapping(self._handle.deployment(model))
            litellm_model: Final = (
                str((as_str_mapping(record.get("litellm_params")) or {}).get("model")) if record is not None else model
            )
            return litellm_exceptions.ServiceUnavailableError(
                message=f"Model '{litellm_model}' is currently paused and cannot accept requests.",
                model=litellm_model,
                llm_provider="",
            )
        if kind == "only_markers":
            named_model: Final = str(step.get("error_model") or model)
            return litellm_exceptions.BadRequestError(
                message=f"You passed in model={named_model}. {RouterErrors.only_strategy_marker_deployments.value}",
                model=named_model,
                llm_provider="",
            )
        if kind == "endpoint_not_supported":
            return litellm_exceptions.EndpointNotSupportedByModelError(
                message=str(step.get("error_detail")),
                model=str(step.get("error_model") or model),
                llm_provider="",
            )
        if kind == "ptu_reserved":
            reserved: Final = str(step.get("error_model") or model)
            return litellm_exceptions.BadRequestError(
                message=f"Deployment {reserved} is reserved for the teams holding a PTU share of it",
                model=reserved,
                llm_provider="",
            )
        if kind == "not_pass_through":
            # The sync pick names what `_common_checks_available_deployment` resolved the id to.
            named_record: Final = (
                as_str_mapping(self._handle.deployment(model)) if pass_through == "pass_through" else None
            )
            error_model: Final = (
                str((as_str_mapping(named_record.get("litellm_params")) or {}).get("model"))
                if named_record is not None
                else model
            )
            return litellm_exceptions.BadRequestError(
                message=f"Deployment {model} does not support pass-through endpoint (use_in_pass_through=False)",
                model=error_model,
                llm_provider="",
            )
        if kind == "no_pass_through_deployments":
            # The sync pick names what `_common_checks_available_deployment` resolved the name to.
            named_group: Final = str(step.get("error_model") or model)
            listed_model: Final = (
                named_group if pass_through == "async_pass_through" else self._process_renamed(model, named_group)
            )
            message: Final = (
                f"Model {listed_model} has no deployments configured with use_in_pass_through=True. "
                "Please add use_in_pass_through: true to the deployment configuration"
                if pass_through == "async_pass_through"
                else f"Model {listed_model} has no deployment configured with use_in_pass_through=True. "
                "Please add use_in_pass_through: true in the deployment configuration"
            )
            return litellm_exceptions.BadRequestError(message=message, model=listed_model, llm_provider="")
        if kind == "context_window_exceeded":
            return litellm_exceptions.ContextWindowExceededError(
                message="litellm._pre_call_checks: Context Window exceeded for given call. No models have context "
                f"window large enough for this call.\n{step.get('error_detail')}",
                model=str(step.get("error_model") or model),
                llm_provider="",
            )
        if kind == "unsupported_params":
            params: Final = sorted(str(param) for param in _as_list(step.get("error_params")))
            unsupported_model: Final = str(step.get("error_model") or model)
            return litellm_exceptions.UnsupportedParamsError(
                message=(
                    f"no deployment of model={unsupported_model} supports {params}. "
                    "To drop these, set `litellm.drop_params=True` or for proxy `litellm_settings: drop_params: true`, "
                    f"or send allowed_openai_params={params} to pass them through"
                ),
                model=unsupported_model,
            )
        if kind == "routing_plugins_left_nothing":
            left: Final = [str(name) for name in _as_list(step.get("error_params"))]
            return ValueError(f"No deployments left after routing-plugin filtering. candidate_models={left}")
        if kind == "routing_plugins_need_async":
            return ValueError(SYNC_PATH_MESSAGE)
        looked_up: Final = _lookup_error(kind, str(step.get("error_model") or model), self._resolved.default_deployment)
        if looked_up is not None:
            return looked_up
        if kind == "unknown_model":
            # router-core names what `_common_checks_available_deployment` looked for last.
            aliased: Final = _alias_target(aliases.get(model)) or model if model in aliases else model
            named: Final = str(step.get("error_model") or aliased)
            return litellm_exceptions.BadRequestError(
                message=f"You passed in model={named}. {RouterErrors.no_healthy_deployments.value}",
                model=named,
                llm_provider="",
            )
        if kind == "no_healthy_deployments":
            group: Final = self.registry().resolve(model)
            group_ids: Final = self.registry().names(str(group), kind="ids")
            # A listed pass-through pick names only the pass-through deployments.
            ids: Final = (
                [id for id in group_ids if _uses_pass_through(self._handle.deployment(id))]
                if pass_through == "pass_through"
                else list(group_ids)
            )
            active: Final = self._cooldowns.get_active_cooldowns(list(ids), None)  # pyright: ignore[reportArgumentType]  # a list of ids
            # `async_get_cooldown_deployments_with_debug_info`: every deployment cooling, in any group.
            listed: Final = self._cooldowns.get_active_cooldowns(list(self.registry().names(kind="ids")), None)  # pyright: ignore[reportArgumentType]  # a list of ids
            cooling: Final = [id for id, _ in listed]
            min_cooldown: Final = min(
                (_seconds(entry.get("cooldown_time"), self._cooldowns.default_cooldown_time) for _, entry in active),
                default=_seconds(None, self._cooldowns.default_cooldown_time),
            )
            return RouterRateLimitError(
                model=self._process_renamed(model, str(group)),
                cooldown_time=min_cooldown,
                enable_pre_call_checks=self._resolved.enable_pre_call_checks,
                cooldown_list=cooling,
                model_ids=list(ids),  # pyright: ignore[reportArgumentType]  # a list of ids
                cooldown_deployments=self._routable_cooled_info(
                    step, str(group), cooling, request or {}, has_input=has_input, tags=tags
                ),
            )
        if last is not None:
            return last
        return RuntimeError(f"router-core ({kind}): {step.get('error')}")


def wildcard_request(model: str) -> str:
    """``f"{provider}/{model}"``, the spelling ``PatternMatchRouter.get_pattern`` tries second."""
    from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider

    return f"{inferred_provider(model)}/{model}"


def _provider_request(step: Mapping[str, object]) -> Mapping[str, object]:
    return {"models": [wildcard_request(str(step.get("model")))]}


def _lookup_error(kind: object, model: str, default_deployment: object) -> BaseException | None:
    """The plain Python error a pick's lookup raised for ``model``, or None for any other failure."""
    if kind == "default_deployment_raises":
        return default_deployment_copy_error(default_deployment, model)
    if kind == "index_out_of_range":
        # `self._core_model_list[index]` for a written index position past the list.
        return IndexError("list index out of range")
    if kind == "missing_routing_group":
        # `_get_routing_context` reads the stale member map's group from the replaced groups.
        return KeyError(model)
    return None


def _origin_level(origin: Mapping[str, object]) -> int:
    level: Final = origin.get("level")
    return level if isinstance(level, int) else 0


def _error_outlasting_total_timeout(driver: "_Driver", seconds: float) -> BaseException | None:
    """The error a retry would have answered when its backoff outlasts the request's total_timeout: the
    call ends with it now, rather than with a timeout after waiting out the rest of the budget."""
    deadline: Final = current_total_deadline()
    if deadline is None or deadline.allows_wait(seconds):
        return None
    return driver.last_error


def _sleep_seconds(step: Mapping[str, object]) -> float:
    """A retry's sleep: ``_time_to_sleep_before_retry`` returns a plain ``0`` when another
    deployment is healthy, else a backoff in float seconds."""
    seconds: Final = float(str(step["seconds"]))
    return 0 if seconds == 0 else seconds


def _served_as(routed: object, request: Mapping[str, object]) -> object:
    """A pre-routed group, paired with the group the hook stamped the request as served for
    (``fallback_lookup_groups``' routed group) when that differs: the level's fallbacks look it up."""
    from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs

    if not isinstance(routed, str):
        return routed
    metadata: Final = request.get(get_metadata_variable_name_from_kwargs(dict(request)))
    stamped: Final = metadata.get("model_group") if isinstance(metadata, Mapping) else None
    return (routed, stamped) if isinstance(stamped, str) and stamped and stamped != routed else routed


_USAGE_READ: Final[ContextVar["tuple[RoutingReadBatch, list[Mapping[str, object]]] | None"]] = ContextVar(
    "rust_core_usage_read", default=None
)


async def _pick_with_prefetched_usage(*args: object, **kwargs: object) -> object:
    """``pick_async`` with the counters the pick's cooldown read fetched (``PrefetchedUsage``)."""
    from litellm.router_strategy.lowest_tpm_rpm_v2 import PrefetchedUsage

    usage_read: Final = _USAGE_READ.get()
    with PrefetchedUsage.scoped(usage_read[0].prefetched_usage if usage_read is not None else None):
        return await pick_async(*args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]  # pick_async's own arguments


def _deployment_pick_attributes(model: str, request: Mapping[str, object]) -> Mapping[str, str | int]:
    """Bounded attributes for one deployment pick; the attempt is 1-based within its model group."""
    metadata: Final = request.get("litellm_metadata", request.get("metadata"))
    attempted: Final = metadata.get("attempted_retries") if isinstance(metadata, Mapping) else None
    retries: Final = attempted if isinstance(attempted, int) else 0
    depth: Final = request.get("fallback_depth")
    hop: Final = (isinstance(depth, int) and depth > 0) or model != request.get("model")
    reason: Final = "retry" if retries > 0 else "fallback" if hop else "initial"
    return MappingProxyType(
        {
            "litellm.deployment.attempt": retries + 1,
            "litellm.deployment.reason": reason,
            "litellm.deployment.model_group": model,
        }
    )


def _region_routable(
    records: Sequence[Mapping[str, object]],
    request: Mapping[str, object],
    has_input: bool,
    pre_call_checks: bool,
) -> Sequence[Mapping[str, object]]:
    """``records`` without the ones ``_pre_call_checks`` keeps the request away from for being
    outside its ``allowed_model_region``."""
    from litellm.types.router import LiteLLM_Params
    from litellm.utils import is_region_allowed

    allowed: Final = request.get("allowed_model_region")
    if allowed is None or not pre_call_checks or not has_input:
        return records
    if not isinstance(allowed, str):
        return ()
    return [
        record
        for record in records
        if is_region_allowed(
            litellm_params=LiteLLM_Params.model_validate(record.get("litellm_params") or {}),
            allowed_model_region=allowed,
        )
    ]


_Value = TypeVar("_Value")


def _finished(coroutine: Coroutine[object, object, _Value]) -> _Value:
    """The value of a coroutine that awaits nothing, run to its end without an event loop."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value  # pyright: ignore[reportAny]  # the coroutine's return value
    coroutine.close()
    raise RuntimeError("the coroutine awaited something")


def _uses_pass_through(record: object) -> bool:
    params: Final = as_str_mapping((as_str_mapping(record) or {}).get("litellm_params")) or {}
    return bool(params.get("use_in_pass_through"))


def _seconds(value: object, default: float) -> float | int:
    """A cooldown time as stored: an int stays an int, so error texts read ``60`` like Python's."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return int(default) if float(default).is_integer() else default


_IO_TOKEN_USAGE: Final = TypeAdapter(tuple[int | None, int | None])


# The plans whose kwargs carry `original_generic_function`, which the fallback cooldown reads.
_GENERIC_API_KINDS: Final = frozenset(("generic", "search"))


class _NotSynchronous(TypeError):
    """A sync attempt's host endpoint returned an awaitable: a host bug, never an attempt failure."""


_MEMBER_READ_PICKS: Final = frozenset({"healthy", "pass_through", "sync_pick"})


class _Selection:
    """One ``select``: the Rust selection call, its select and cache steps answered on the host."""

    def __init__(
        self,
        core: RustCore,
        model: object,
        request: dict[str, object],  # mutable-ok: the caller's request, whose narrowing keys are popped
        kind: str,
        specific_deployment: bool | None,
        messages: object,
        input: object,
        *,
        probe: bool = False,
        batch_upload: bool = False,
    ) -> None:
        self._core: Final = core
        # `_async_get_available_deployment` reads the alias map first, every other pick its names.
        self._model: Final = (
            model if isinstance(model, str) else core.routed_name(model, read_as_member=kind in _MEMBER_READ_PICKS)
        )
        self._request: Final = request
        self._kind: Final = kind
        self._messages: Final = messages
        self._input: Final = input
        fields: Final = {
            **core._routing_fields(request, messages),  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            **_narrowing_fields(request, kind),
        }  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
        # The pick's argument alone: a `specific_deployment` key in its request is no argument, and
        # `_common_checks_available_deployment` reads nothing but True.
        fields.pop("specific_deployment", None)
        if specific_deployment is True:
            fields["specific_deployment"] = True
        # `filter_web_search_deployments` is in the async chain, which a strategy without an async
        # selector never reaches on a pick.
        if (kind in ("healthy", "async_pass_through") or (kind == "pick" and core._picks_async())) and (  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            is_web_search_request(request)
        ):
            fields["web_search"] = True
        # `_text_completion` wraps its sync pick in `endpoint_selection`.
        endpoint: Final = selected_endpoint()
        if endpoint is not None:
            fields["endpoint"] = endpoint
        if probe:
            fields["health_check_probe"] = True
        if batch_upload:
            fields["batch_create"] = True
            fields["batch_upload"] = True
        if kind in ("pick", "async_pass_through") or (kind == "sync_pick" and core._custom_strategy is not None):  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            fields["pre_routes"] = True
        self._specific_deployment: Final = specific_deployment
        # The error a callback filter raised, which a failed selection raises.
        self._filter_error: BaseException | None = None
        if kind == "pass_through":
            fields["pass_through"] = "listed"
        elif kind == "async_pass_through":
            fields["pass_through"] = "healthy"
        core._sync_routing_state()  # pyright: ignore[reportPrivateUsage]  # see above
        handle: Final = core._handle  # pyright: ignore[reportPrivateUsage]  # see above
        self._call: Final = (handle.healthy if kind == "healthy" else handle.select)(self._model, _document(fields))

    def _done(self, step: Mapping[str, object]) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        if self._kind == "healthy":
            if step.get("single"):
                # A deployment named by its id is listed alone, as a dict.
                (record,) = _as_list(step.get("deployments"))
                return self._served(record)
            # The deployments as served: a priority group's members carry their priority as order.
            return [self._served(record) for record in _as_list(step.get("deployments"))]
        if step.get("deployment_id") in (DEFAULT_DEPLOYMENT_FAILING_ID, DEFAULT_DEPLOYMENT_COPIED_ID):
            # A default deployment every call to fails, or whose id is not a string, is picked as Python copies it.
            picked: Final = default_deployment_copy(self._core._resolved.default_deployment, self._model)  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            return picked  # pyright: ignore[reportReturnType]  # the assigned value's copy, as the Python core returns it
        return self._served(step.get("deployment"))

    def _served(self, record: object) -> Mapping[str, object]:
        """A deployment as served: its record, with the litellm_params values JSON carries as text."""
        return _with_live_params(
            self._core, for_requested_model(typed_deployment(as_str_mapping(record) or {}), self._model)
        )

    def _failed(self, step: Mapping[str, object], *, sync: bool) -> BaseException:
        core: Final = self._core
        error: Final = core._failure_error(  # pyright: ignore[reportPrivateUsage]  # see above
            step,
            self._model,
            self._filter_error,
            pass_through=self._kind,
            request=self._request,
            has_input=self._messages is not None or self._input is not None,
            tags=not sync,
        )
        if self._kind not in ("pass_through", "sync_pick"):
            _log_refused_pick(self._request, error)
        return error

    async def run(self) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        from litellm.integrations.otel.runtime import phase_span

        with (
            phase_span(f"route {self._model}"),
            self._core._usage_read_scope(str(self._model), self._request),  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
        ):
            return await self._run()

    async def _run(self) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        core: Final = self._core
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = self._call.next(outcome)
            match step["kind"]:
                case "cache":
                    outcome = await core._answer_cache(step)  # pyright: ignore[reportPrivateUsage]  # see above
                case "select" if step.get("stage") == "callback_filters":
                    outcome, self._filter_error = await core._answer_callback_filters(  # pyright: ignore[reportPrivateUsage]  # see above
                        step, self._request, self._messages
                    )
                case "select":
                    outcome = await core._answer_select(  # pyright: ignore[reportPrivateUsage]  # see above
                        step,
                        self._request,
                        self._messages,
                        self._input,
                        pass_through=self._kind == "async_pass_through",
                    )
                case "pre_route":
                    outcome = await core._pre_route(  # pyright: ignore[reportPrivateUsage]  # see above
                        step,
                        self._request,
                        self._messages,
                        self._input,
                        self._specific_deployment,
                        pass_through=self._kind == "async_pass_through",
                    )
                case "done":
                    done = self._done(step)
                    if self._kind != "pick" or not isinstance(done, Mapping):
                        return done
                    # `_core_async_get_available_deployment` resolves its pick to a pod.
                    return await self._resolved_pod(core, done)
                case "failed":
                    raise self._failed(step, sync=False)
                case other:  # pyright: ignore[reportUnnecessaryComparison]  # a kind the bridge returns but the stub does not list
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    async def _resolved_pod(self, core: "RustCore", done: Mapping[str, object]) -> Mapping[str, object]:
        discovery: Final = core._pick_discovery(sync=False)  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
        if discovery is None:
            return done
        resolved: Final[object] = await discovery.async_resolve_deployment(done, self._request)
        return resolved  # pyright: ignore[reportReturnType]  # the deployment, resolved

    def _resolved_pod_sync(self, core: "RustCore", done: Mapping[str, object]) -> Mapping[str, object]:
        discovery: Final = core._pick_discovery(sync=True)  # pyright: ignore[reportPrivateUsage]  # see _resolved_pod
        return done if discovery is None else discovery.resolve_deployment(done, self._request)

    def run_sync(self) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        core: Final = self._core
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = self._call.next(outcome)
            match step["kind"]:
                case "cache":
                    outcome = core._answer_cache_sync(step)  # pyright: ignore[reportPrivateUsage]  # see above
                case "select":
                    outcome = core._answer_select_sync(step, self._request, self._messages, self._input)  # pyright: ignore[reportPrivateUsage]  # see above
                case "pre_route":
                    outcome = core._custom_pick_sync(  # pyright: ignore[reportPrivateUsage]  # see above
                        step, self._request, self._messages, self._input, self._specific_deployment
                    )
                case "done":
                    done = self._done(step)
                    if self._kind != "sync_pick" or not isinstance(done, Mapping):
                        return done
                    # `get_available_deployment` resolves its pick to a pod.
                    return self._resolved_pod_sync(core, done)
                case "failed":
                    raise self._failed(step, sync=True)
                case other:  # pyright: ignore[reportUnnecessaryComparison]  # a kind the bridge returns but the stub does not list
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")


def _log_refused_pick(request: Mapping[str, object], error: BaseException) -> None:
    """A refused pick is logged to the request's logging object (`_async_get_available_deployment`)."""
    dispatch: Final[object] = getattr(request.get("litellm_logging_obj"), "dispatch_failure_handlers", None)
    if not callable(dispatch):
        return
    import asyncio
    import traceback

    trace: Final = "Traceback (most recent call last):\n" + "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    pending: Final[object] = dispatch(exception=error, traceback_exception=trace, prefer_async_handlers=True)
    if inspect.isawaitable(pending):
        _ = asyncio.ensure_future(pending)


def _log_refused_image_pick(request: Mapping[str, object], error: BaseException) -> None:
    """``_aimage_generation``'s refused pick, for a request with no logging object of its own."""
    from litellm._uuid import uuid
    from litellm.router_rust.invoke.endpoints import ROUTING_KEYS
    from litellm.utils import (
        Rules,
        _restore_correlation_context_if_supported,  # pyright: ignore[reportPrivateUsage]  # same task-local cleanup as the Python router
        function_setup,
    )

    kwargs: Final = without(request, *(ROUTING_KEYS - {"model"}))
    kwargs.setdefault("litellm_call_id", str(uuid.uuid4()))
    logging_obj, _ = function_setup("aimage_generation", Rules(), datetime.now(), **kwargs)  # noqa: DTZ005  # SDK logging uses naive local timestamps  # pyright: ignore[reportArgumentType]  # opaque SDK options, as the Python router passes them
    try:
        _log_refused_pick({"litellm_logging_obj": logging_obj}, error)
    finally:
        _restore_correlation_context_if_supported(logging_obj)


def _with_live_params(core: "RustCore", deployment: Mapping[str, object]) -> Mapping[str, object]:
    """``deployment`` with the values its JSON record carries as text: litellm_params holding an
    Exception ``mock_response``, a callable or a str subclass that marks the value (an admin's secret
    reference), and model_info values with keys that are not strings."""
    info: Final = as_str_mapping(deployment.get("model_info"))
    model_id: Final = (info or {}).get("id")
    live: Final = core._hosted.live_params(model_id)  # pyright: ignore[reportPrivateUsage]  # the core's own registry
    live_info: Final = core._hosted.live_info(model_id)  # pyright: ignore[reportPrivateUsage]  # same
    if not live and not live_info:
        return deployment
    return {
        **deployment,
        **({"litellm_params": {**(as_str_mapping(deployment.get("litellm_params")) or {}), **live}} if live else {}),
        **({"model_info": {**(info or {}), **live_info}} if live_info else {}),
    }


def _custom_pick(picked: object) -> Mapping[str, object]:
    """A custom strategy's answer as the deployment document the Rust core serves."""
    document: Final = as_str_mapping(picked)
    if document is None:
        raise TypeError(f"the custom routing strategy returned {type(picked).__name__}, not a deployment")
    return _document(document)


def _with_target_order(kwargs: Mapping[str, object], order: object) -> Mapping[str, object]:
    """The attempt's kwargs with the order level its hop targets, as `run_async_fallback` sets
    `_target_order` before `_acreate_file` picks every healthy deployment of that level."""
    if not isinstance(order, int) or isinstance(order, bool):
        return kwargs
    return {**kwargs, "_target_order": order}


def _narrowing_fields(request: dict[str, object], kind: str) -> Mapping[str, object]:  # mutable-ok: popped
    """``_target_order``, ``_excluded_deployment_ids`` and ``_retry_skipped_deployment_ids``, popped
    from the request as the healthy chain pops them (a listed pass-through pick only reads the order)."""
    fields: Final = _request_narrowing(request)
    if kind == "pass_through":
        return {key: value for key, value in fields.items() if key == "target_order"}
    for key in ("_target_order", "_excluded_deployment_ids", "_retry_skipped_deployment_ids"):
        request.pop(key, None)
    return fields


def _routes_as_batch_create(plan: Plan, request: Mapping[str, object]) -> bool:
    return plan.kind == "batch" or (plan.kind == "create_file" and is_batch_upload(request))


def _request_narrowing(request: Mapping[str, object]) -> Mapping[str, object]:
    """The narrowing keys a request carries, as the fields router-core reads them."""
    from litellm.litellm_core_utils.routing_common_utils import (
        as_excluded_deployment_ids,
        as_retry_skipped_deployment_ids,
    )

    target_order: Final = request.get("_target_order")
    excluded: Final = as_excluded_deployment_ids(request.get("_excluded_deployment_ids"))
    skipped: Final = as_retry_skipped_deployment_ids(request.get("_retry_skipped_deployment_ids"))
    return {
        **(
            {"target_order": target_order}
            if isinstance(target_order, int) and not isinstance(target_order, bool)
            else {}
        ),
        **({"excluded_ids": list(excluded)} if excluded else {}),
        **({"retry_skipped_ids": list(skipped)} if skipped else {}),
    }


_NO_ROUTING_PRE_CALL_CHECKS: Final = frozenset({"search", "rerank", "moderation"})
# Attempts whose deployment is no pick (`_core_async_get_available_deployment` /
# `get_available_deployment`), so Kubernetes pod discovery never resolves it.
_UNPICKED_KINDS: Final = frozenset({"search", "create_file"})


# The info lines the Python router's endpoint helpers logged for an attempt: its success, and for
# these its failure too.
_ATTEMPT_LOG_NAMES: Final = MappingProxyType(
    {
        "acompletion": "litellm.acompletion",
        "completion": "litellm.completion",
        "aembedding": "litellm.aembedding",
        "embedding": "litellm.embedding",
        "aimage_generation": "litellm.aimage_generation",
        "image_generation": "litellm.image_generation",
        "atranscription": "litellm.atranscription",
        "aspeech": "litellm.aspeech",
        "arerank": "litellm.arerank",
        "atext_completion": "litellm.atext_completion",
        "aadapter_completion": "litellm.aadapter_completion",
    }
)
_SUCCESS_ONLY_LOG_NAMES: Final = MappingProxyType(
    {
        "acancel_batch": "litellm.acancel_batch",
        "acreate_batch": "litellm.acreate_batch",
        "acreate_file": "litellm.acreate_file",
        "avector_store_create": "litellm.avector_store_create",
    }
)


def _log_level_traceback(error: BaseException | None) -> None:
    """``async_function_with_fallbacks_common_utils``' DEBUG traceback of the error a level hands
    its fallbacks: the stack the ERROR line leaves out."""
    import logging
    import traceback

    from litellm._logging import redact_string, verbose_router_logger

    if error is None or not verbose_router_logger.isEnabledFor(logging.DEBUG):
        return
    formatted: Final = "".join(traceback.format_exception(error))
    verbose_router_logger.debug("Traceback%s", redact_string(formatted))


def _log_attempt(plan: Plan, model_name: object, error: BaseException | None = None) -> None:
    """``litellm.<endpoint>(model=...) 200 OK`` or ``... Exception ...``, as the Python router logged each attempt."""
    from litellm._logging import verbose_router_logger

    name: Final = _ATTEMPT_LOG_NAMES.get(plan.endpoint)
    if error is not None:
        if name is not None:
            verbose_router_logger.info("%s(model=%s)\x1b[31m Exception %s\x1b[0m", name, model_name, str(error))
        return
    served_name: Final = (
        name
        or _SUCCESS_ONLY_LOG_NAMES.get(plan.endpoint)
        or ("ageneric_api_call_with_fallbacks" if plan.kind == "generic" else None)
    )
    if served_name is not None:
        verbose_router_logger.info("%s(model=%s)\x1b[32m 200 OK\x1b[0m", served_name, model_name)


class _Driver:
    """Answers the attempt and fallback steps of one call, and shapes its result.

    Each attempt is one host call against the deployment Rust picked. The driver keeps the error
    each fallback level ended on: a level that fails raises its own error, whatever its fallbacks
    raised, as the Python router does, unless an order chain handed it another
    (``order_chain_failed``).
    """

    def __init__(
        self,
        core: RustCore,
        plan: Plan,
        request: Mapping[str, object],
        cut_deployment: object = None,
        stream_failure: tuple[int, BaseException] | None = None,
        *,
        sync: bool = False,
        fresh: bool = False,
        passthrough: "Callable[[Mapping[str, object], str], Awaitable[object]] | None" = None,
    ) -> None:
        self._core: Final = core
        self._sync: Final = sync
        self._passthrough: Final = passthrough
        self._plan: Final = plan
        self._op: Final = plan.endpoint
        self._request: Final = request
        self._cut_deployment: Final = cut_deployment if isinstance(cut_deployment, str) else None
        self._last_response: object = None
        self._last_error: BaseException | None = None
        # The level the call raises from: a fallback walk re-entered from a failed stream starts at
        # the level that stream served, on the failure ``async_function_with_fallbacks_common_utils``
        # was handed, which is what it raises when no fallback answers.
        self._base_level: Final = stream_failure[0] if stream_failure is not None else 0
        # How many fallback hops deep the walk is: the "Successful fallback" lines a served hop logs.
        self._fallback_depth: int = 0
        self._level_errors: Final[dict[int, BaseException]] = dict(  # mutable-ok: filled as the call runs
            (stream_failure,) if stream_failure else ()
        )
        # The retry counts an exhausted retry loop stamps on the pick error its level then raises.
        self._pending_stamps: Final[dict[int, tuple[object, object]]] = {}  # mutable-ok: filled as the call runs
        # Each level's current hop's first failure, once it retries: the error its retry loop raises
        # last is raised while that one is handled, so it chains to it.
        self._hop_first: Final[dict[int, BaseException]] = {}  # mutable-ok: filled as the call runs
        # Each level's latest pick error: the one object it logged, recorded and raises.
        self._level_picks: Final[dict[int, BaseException]] = {}  # mutable-ok: filled as the call runs
        # Failures a retry or a later hop raised past the request's first logged failure, which the
        # attempt itself recorded: the failed hop's fallback cooldown leaves them alone.
        self._recorded_blocked: Final[set[int]] = set()  # mutable-ok: filled as the call runs
        self._served: tuple[str | None, Mapping[str, object]] = (None, MappingProxyType({}))
        # The latest attempt's deployment and call, which a blocked response is raised from.
        self._latest_call: tuple[Mapping[str, object], AttemptCall] | None = None
        # A sync attempt's stream is reading its first chunk, so a failure now is the stream's to log.
        self._reading_first_chunk: bool = False
        self._attempts: int = 0
        # When the latest attempt started: Python reads it off the logging object, when one is set up.
        self._attempt_started: datetime | None = None
        # A single-attempt endpoint runs neither of the wrappers that stamp the walk on the request.
        # The open ``route {model_group}`` phase span of the pick in progress (async calls only).
        self._route_phase: contextlib.ExitStack | None = None
        self._route_model: str = ""
        self._walk: Final = WalkMetadata(
            request, plan.metadata_field, self._params_base_model, live=fresh and not plan.single_attempt
        )
        self.outcomes: Final = FallbackOutcomes(request)
        # The fallback level the walk is at, from its fallback events.
        self._level: int = self._base_level
        # The primary level reached the end of `async_function_with_fallbacks_common_utils`, the
        # only place the outcome message is appended: not its early raises, nor its order path.
        self._explains_failure: bool = False

    def _blocked(self, response: object, endpoint_name: str) -> str | None:
        """How the provider blocked a served response, the first check of
        ``_should_raise_content_policy_error`` (a chat response with no choices, or whose first
        choice was content-filtered) and of ``_should_raise_anthropic_refusal_error`` (a
        /v1/messages safeguard refusal carrying ``stop_details``)."""
        from litellm.llms.anthropic.messages.native.utils import get_safeguard_refusal_stop_details
        from litellm.types.utils import ModelResponse

        if endpoint_name == "anthropic_messages" and get_safeguard_refusal_stop_details(response) is not None:
            return "refusal"
        if self._op not in ("acompletion", "completion") or not isinstance(response, ModelResponse):
            return None
        return (
            "content_filter" if not response.choices or response.choices[0].finish_reason == "content_filter" else None
        )

    def _blocked_error(self, event: Mapping[str, object]) -> BaseException:
        """The error the router raises for a blocked response a fallback would serve."""
        import litellm.exceptions as litellm_exceptions
        from litellm.llms.anthropic.messages.native.utils import (
            get_safeguard_refusal_stop_details,
            safeguard_refusal_error,
        )

        model: Final = str(event.get("model"))
        stop_details: Final = get_safeguard_refusal_stop_details(self._last_response)
        if event.get("blocked") == "refusal" and stop_details is not None:
            return safeguard_refusal_error(model=model, stop_details=stop_details)
        return litellm_exceptions.ContentPolicyViolationError(
            message="Response output was blocked.", model=model, llm_provider=""
        )

    def _stamp_blocked(self, error: BaseException) -> None:
        """The attempt helper's ``except`` runs for the error it raises over a blocked response: the
        response then counts as a failure, not a success."""
        if self._latest_call is None:
            return
        deployment, call = self._latest_call
        endpoints: Final = self._core._endpoints  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        stamp_error(self._plan, error, deployment, call.served)
        endpoints.counters.retract_success(self._plan, call)
        endpoints.attempt_failed(self._plan, error, deployment, call, call.group)

    def _params_base_model(self, deployment_id: str) -> str | None:
        """``_deployment_params_base_model``: the ``base_model`` under the deployment's ``litellm_params``."""
        record: Final = as_str_mapping(self._core._handle.deployment(deployment_id)) or {}  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        base_model: Final = (as_str_mapping(record.get("litellm_params")) or {}).get("base_model")
        return base_model if isinstance(base_model, str) else None

    def _logged_error(self, event: Mapping[str, object]) -> BaseException | None:
        """The error a ``retry_logged`` event records."""
        return self._cause_error(event)

    def _policy_raised(self, event: Mapping[str, object]) -> None:
        """A hop's first failure looked its group's retry policy up and the lookup raised: the hop
        ends with that error, raised from the failure the event names."""
        model: Final = str(event.get("model"))
        original: Final = self._cause_error(event) or RuntimeError(f"router-core: the hop serving {model} failed")
        policies: Final = self._request.get(
            "model_group_retry_policy",
            self._core._resolved.model_group_retry_policy,  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        )
        raised: Final = policy_lookup_error(original, model, self._core._written_retry_policy, policies)  # pyright: ignore[reportPrivateUsage]  # see above
        level: Final = event.get("level")
        self._last_error = raised
        self._level_errors[level if isinstance(level, int) else 0] = raised

    def _stamped_budget(self, event: Mapping[str, object]) -> Mapping[str, object]:
        """A ``retries_opened`` event as ``_metadata["max_retries"]`` reads it: a request
        ``num_retries`` that is no count is stamped as given."""
        given: Final = self._request.get("num_retries")
        return {**event, "max_retries": given} if "uncounted_retries" in request_retry_fields(given) else event

    def _count_raised(self, event: Mapping[str, object]) -> None:
        """A hop's first failure reached the retry loop's read of a request ``num_retries`` that
        raises: the hop ends with that error, raised from the failure the event names."""
        model: Final = str(event.get("model"))
        original: Final = self._cause_error(event) or RuntimeError(f"router-core: the hop serving {model} failed")
        raised: Final = retry_count_error(original, self._request.get("num_retries"))
        level: Final = event.get("level")
        self._last_error = raised
        self._level_errors[level if isinstance(level, int) else 0] = raised

    def _cause_error(self, event: Mapping[str, object]) -> BaseException | None:
        """The error the ``cause`` of a ``retry_logged`` or ``retry_policy_raised`` event names."""
        match event.get("cause"):
            case "level":
                level: Final = event.get("level")
                return self._level_errors.get(level if isinstance(level, int) else 0, self._last_error)
            case "pick":
                at_level: Final = event.get("level")
                held: Final = self._level_picks.get(at_level if isinstance(at_level, int) else 0)
                return held if held is not None else self._picked_error(event)
            case _:
                return self._last_error

    def _picked_error(self, event: Mapping[str, object]) -> BaseException:
        """The error a pick an event names raised, as a failed step with that error raises it."""
        model: Final = str(event.get("failed_model") or event.get("model"))
        return self._core._failure_error(  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            event, model, None, request=self._request, has_input=self._has_input(), tags=not self._sync
        )

    def _pick_raised(self, error: Exception, *, logged: bool) -> Mapping[str, object]:
        """A pick raised on the host (a strategy, a tag filter, a pre-routing hook): the try fails with
        that error, which the Rust walk retries and falls back from like any attempt's. ``logged``
        when the async pick's ``except`` reports it, as ``_async_get_available_deployment`` does."""
        if isinstance(error, RustCoreUnsupported):
            raise error
        if logged:
            self._log_pick_failure(error)
        else:
            mark_deployment_selection_failure(error)
        self._chain_to_hop_first(error, self._level)
        self._last_error = error
        self._level_errors[self._level] = error
        return {**attempt_failure(error), "raised": True}

    def _log_pick_failure(self, error: BaseException) -> None:
        """The ``except`` of ``_async_get_available_deployment``, which only the async selection runs:
        the router's strategy may have no async selector, and a custom strategy replaces the method.
        ``_aimage_generation`` logs a refused pick itself when the request set up no logging object."""
        core: Final = self._core
        assigned: Final = "_core_async_get_available_deployment" in core.__dict__
        custom: Final = assigned or core._custom_strategy is not None  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        if not custom and core._picks_async():  # pyright: ignore[reportPrivateUsage]  # see above
            mark_deployment_selection_failure(error)
            _log_refused_pick(self._request, error)
        if self._plan.kind == "image" and self._request.get("litellm_logging_obj") is None:
            _log_refused_image_pick(self._request, error)

    def _pick_failed(self, event: Mapping[str, object]) -> None:
        """A pick failed in the Rust core: its error is raised once, however many events name it."""
        error: Final = self._picked_error(event)
        level: Final = event.get("level")
        self._level_picks[level if isinstance(level, int) else 0] = error
        if not self._sync:
            self._log_pick_failure(error)

    def _unroutable_error(self, model: object) -> BaseException:
        """What a hop to a dict target's non-string ``model`` raises: no group answers ``None``, and
        any other value fails the level's fallback lookup, which hashes it and then searches it."""
        if model is None:
            return self._core._failure_error({"error_kind": "unknown_model"}, "None", None)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        try:
            _ = model in {} or "/" in model  # pyright: ignore[reportOperatorIssue]  # raising is the point
        except TypeError as error:
            return error
        return TypeError(f"{model!r} is not a model group")

    def _level_failed(self, event: Mapping[str, object]) -> None:
        """A level raises the error a pick raised (``level_failed``)."""
        level: Final = event.get("level")
        at_level: Final = level if isinstance(level, int) else 0
        picked: Final = (
            self._picked_error(event)
            if event.get("handling") is True
            else self._level_picks.pop(at_level, None) or self._picked_error(event)
        )
        if event.get("handling") is True:
            # Raised by the level's fallback handling, while the level's error was handled.
            picked.__context__ = self._level_errors.get(at_level, self._last_error)
            self._level_errors[at_level] = picked
            if at_level == self._base_level:
                self._explains_failure = False
            return
        stamp: Final = self._pending_stamps.pop(at_level, None)
        if stamp is not None and type(picked) in self._core._host.dynamic().exception_types:  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            # The exhausted retry loop raised this pick's error: stamped as `retries_exhausted` stamps.
            picked.max_retries, picked.num_retries = stamp  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
        self._chain_to_hop_first(picked, at_level)
        self._level_errors[at_level] = picked

    def _log_hop(self, event: Mapping[str, object]) -> None:
        """``run_async_fallback``'s "Falling back to model_group" line, for a hop to ``event``'s target."""
        from litellm._logging import verbose_router_logger
        from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure

        target: Final = event["unroutable"] if "unroutable" in event else event.get("to")
        verbose_router_logger.info("Falling back to model_group = %s", mask_sensitive_structure(target))
        depth: Final = event.get("depth")
        self._fallback_depth = depth if isinstance(depth, int) else self._fallback_depth + 1

    def _carry_superseded_usage(self) -> None:
        """``carry_superseded_attempt_usage``, before each same-group retry and each fallback hop."""
        from litellm.router_utils.fallback_event_handlers import carry_superseded_attempt_usage

        carry_superseded_attempt_usage(self._request)

    def _order_chain_failed(self, event: Mapping[str, object]) -> None:
        """The order path of ``async_function_with_fallbacks_common_utils`` raised: the chain's last
        hop's error, which the hop's level ended on, unless the level keeps its own."""
        level: Final = event.get("level")
        at: Final = level if isinstance(level, int) else 0
        last_hop: Final = self._level_errors.get(at + 1)
        if event.get("last_hop") is True and last_hop is not None:
            self._level_errors[at] = last_hop
        if at == self._base_level:
            self._explains_failure = False

    def _logged_off_the_loop(self) -> bool:
        """Whether litellm logged the attempt's failure on a thread with no running loop. An async
        call's mocked failure raises in ``completion``, which ``acompletion`` runs in an executor
        thread; a sync stream's first chunk logs in a ``threading.Thread``."""
        from litellm.utils import should_run_mock_completion

        if self._sync:
            return self._reading_first_chunk
        kwargs: Final[Mapping[str, object]] = self._latest_call[1].kwargs if self._latest_call is not None else {}
        return self._plan.kind in ("chat", "text") and should_run_mock_completion(
            kwargs.get("mock_response"), kwargs.get("mock_tool_calls"), kwargs.get("mock_timeout")
        )

    def _cooldown_set(self, event: Mapping[str, object]) -> None:
        """An attempt's failure cooled its deployment down: ``set_cooldown_deployments`` as the failure
        callback runs it. Outside a running loop it raises, which ``CustomLogger.log_event`` swallows;
        a sync walk with a private loop has one (``cooldown_event_on_own_loop``)."""
        deployment_id: Final = event.get("deployment_id")
        seconds: Final = event.get("cooldown_time")
        if self._logged_off_the_loop() or not isinstance(deployment_id, str):
            return
        fire: Final = self._core.cooldown_event_on_own_loop if self._plan.on_its_own_loop else self._core.cooldown_event
        with contextlib.suppress(RuntimeError):
            fire(
                deployment_id,
                getattr(self._last_error, "status_code", ""),
                self._python_cooldown_time(seconds) if isinstance(seconds, float) else None,
            )

    def _python_cooldown_time(self, seconds: float) -> float:
        """``_time_to_cooldown`` of ``deployment_callback_on_failure`` for the attempt that failed."""
        deployment: Final[Mapping[str, object]] = self._latest_call[0] if self._latest_call is not None else {}
        configured: Final = first_cooldown(
            as_str_mapping(deployment.get("model_info")),
            self._request,
            as_str_mapping(deployment.get("litellm_params")),
        )
        return self._core.python_cooldown(seconds, configured, self._last_error)

    def _selected(self, event: Mapping[str, object]) -> None:
        """``_acompletion``'s service event for the pick it awaited, and its ``_track_deployment_metrics``."""
        seconds: Final = event.get("pick_seconds")
        if self._plan.kind == "chat" and not self._sync and isinstance(seconds, float):
            end_time: Final = unix_time()
            self._core.service_logged("async_get_available_deployment", end_time - seconds, end_time, self._request)
            self._core.track_deployment_pick(event.get("deployment_id"), self._request)

    def _fallbacks_failed(self, event: Mapping[str, object]) -> None:
        """The ``except`` around the regular fallbacks of ``async_function_with_fallbacks_common_utils``:
        it logs what the ``try`` raised, the last hop's error or the level's own. A level with no
        target raises its own error into it, except where no fallback list was set to search."""
        import litellm.exceptions as litellm_exceptions
        from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs

        level: Final = event.get("level")
        at: Final = level if isinstance(level, int) else 0
        own: Final = self._level_errors.get(at, self._last_error)
        request: Final = self._request
        walked: Final = self._core._walked_fallbacks  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        if event.get("chain") is not True:
            policy_list: Final = request.get("content_policy_fallbacks", walked["content_policy_fallbacks"])
            if not isinstance(own, litellm_exceptions.ContentPolicyViolationError) or policy_list is None:
                if request.get("fallbacks", walked["fallbacks"]) is None:
                    return
                self._explain_no_fallback_group(own, at)
        raised: Final = self._level_errors.get(at + 1, own) if event.get("last_hop") is True else own
        self.outcomes.failed_fallbacks(
            RoutingContext(self._core),  # pyright: ignore[reportArgumentType]  # the read uses registry() and health()
            truncate_fallback_error_detail(redact_string(str(raised))),
            get_parent_otel_span_from_kwargs(dict(request)),
        )

    def _explain_no_fallback_group(self, primary: BaseException | None, at: int) -> None:
        """The primary level names the missing fallback group on its error before raising it into
        the ``except``, so the log reads it and no outcome message follows."""
        if (
            at == self._base_level == 0
            and self._core._host.dynamic().expose_router_debug_in_errors  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            and primary is not None
            and hasattr(primary, "message")
        ):
            primary.message += self._explanation(primary, str(self._request.get("model")))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the message is what this explains
            self._explains_failure = False

    def open_route_phase(self, model: object) -> None:
        """``_async_get_available_deployment_unresolved``'s ``route {model}`` phase: the pick's reads
        nest in it, until the pick ends."""
        from litellm.integrations.otel.runtime import phase_span

        if self._sync or self._route_phase is not None:
            return
        stack: Final = contextlib.ExitStack()
        stack.enter_context(phase_span(f"route {model}"))
        self._route_phase = stack
        self._route_model = str(model)

    def close_route_phase(self, selected_in: object = None) -> None:
        """The pick ended: a served pick marks ``litellm.request.deployment_selected`` inside it first."""
        phase: Final = self._route_phase
        if phase is None:
            return
        if isinstance(selected_in, str):
            from litellm.integrations.otel.runtime import phase_event

            phase_event("litellm.request.deployment_selected", _deployment_pick_attributes(selected_in, self._request))
        self._route_phase = None
        phase.close()

    def take_events(self, events: object) -> None:  # noqa: C901  # one branch per router-core event
        for raw_event in _as_list(events):
            event = as_str_mapping(raw_event)
            if event is None:
                continue
            if event.get("kind") == "pick_started":
                self.close_route_phase()
                self.open_route_phase(event.get("model"))
                continue
            if event.get("kind") == "retry":
                self._carry_superseded_usage()
                continue
            if event.get("kind") == "cooldown_set":
                self._cooldown_set(event)
                continue
            if event.get("kind") == "selected":
                self._selected(event)
                self.close_route_phase(self._route_model)
                continue
            if event.get("kind") == "retries_opened":
                self._walk.retries_opened(self._stamped_budget(event))
                continue
            if event.get("kind") == "retry_budget":
                self._walk.retry_budget(event)
                continue
            if event.get("kind") == "pick_failed":
                self.close_route_phase()
                self._pick_failed(event)
                continue
            if event.get("kind") == "order_chain_failed":
                self._order_chain_failed(event)
                continue
            if event.get("kind") == "mock_failed":
                resolved = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                mocked = mock_error(
                    str(event.get("mock")),
                    str(event.get("model")),
                    event.get("num_retries"),
                    self._request,
                    {
                        "fallbacks": resolved.fallbacks,
                        "context_window_fallbacks": resolved.context_window_fallbacks,
                        "content_policy_fallbacks": resolved.content_policy_fallbacks,
                    },
                )
                level = event.get("level")
                self._last_error = mocked
                self._level_errors[level if isinstance(level, int) else 0] = mocked
                continue
            if event.get("kind") == "retry_policy_raised":
                self._policy_raised(event)
                continue
            if event.get("kind") == "retry_count_raised":
                self._count_raised(event)
                continue
            if event.get("kind") == "level_failed":
                self._level_failed(event)
                continue
            if event.get("kind") == "fallbacks_failed":
                self._fallbacks_failed(event)
                continue
            if event.get("kind") == "fallback":
                self._log_hop(event)
                hop_to = event.get("level")
                self._hop_first.pop(hop_to if isinstance(hop_to, int) else 0, None)
                self._walk.hop(event)
                self._carry_superseded_usage()
                hop_level = event.get("level")
                self._level = hop_level if isinstance(hop_level, int) else self._level
                if "unroutable" in event:
                    self._level_errors[self._level] = self._unroutable_error(event["unroutable"])
                continue
            if event.get("kind") == "blocked":
                level = event.get("level")
                blocked = self._blocked_error(event)
                self._stamp_blocked(blocked)
                self._last_error = blocked
                self._level_errors[level if isinstance(level, int) else 0] = blocked
                continue
            if event.get("kind") == "falling_back":
                level = event.get("level")
                at = level if isinstance(level, int) else 0
                _log_level_traceback(self._level_errors.get(at, self._last_error))
                self._explains_failure = self._explains_failure or at == self._base_level
                if at == self._base_level == 0:
                    self._hint_level_failure(self._level_errors.get(at, self._last_error), str(event.get("model")))
                continue
            if event.get("kind") == "fallback_done":
                level = event.get("level")
                at = level if isinstance(level, int) else 0
                self._level = at
                self.outcomes.done(
                    event,
                    self._level_errors.get(at, self._last_error),
                    self._level_errors.get(at + 1),
                    self._walk.hop_kwargs(event),
                )
                if (
                    event.get("succeeded") is not True
                    and id(self._level_errors.get(at + 1)) not in self._recorded_blocked
                ):
                    self._core.trigger_fallback_cooldown(
                        self._request,
                        self._level_errors.get(at + 1),
                        self._plan.kind in _GENERIC_API_KINDS,
                        own_loop=self._plan.on_its_own_loop,
                    )
                continue
            if event.get("kind") == "retry_logged":
                logged = self._logged_error(event)
                self._walk.log_retry(event, logged)
                logged_level = event.get("level")
                if event.get("attempted_retries") == 0 and logged is not None:
                    self._hop_first[logged_level if isinstance(logged_level, int) else 0] = logged
                continue
            if event.get("kind") != "retries_exhausted":
                continue
            if event.get("picked") is True:
                # The level raises the last retry's pick error, which its `level_failed` names.
                exhausted_level = event.get("level")
                self._pending_stamps[exhausted_level if isinstance(exhausted_level, int) else 0] = (
                    event.get("max"),
                    event.get("attempted"),
                )
                continue
            error: BaseException | None = self._last_error
            # `async_function_with_retries` stamps the error its exhausted retry loop raises.
            if error is not None and type(error) in self._core._host.dynamic().exception_types:  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                error.max_retries = event.get("max")  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
                error.num_retries = event.get("attempted")  # pyright: ignore[reportAttributeAccessIssue]  # same

    def _call(self, step: Mapping[str, object]) -> "tuple[Mapping[str, object], AttemptCall]":
        name: Final = step.get("model")
        requested: Final = str(name) if name is not None else str(self._request.get("model"))
        deployment: Mapping[str, object] = (  # rebind-ok: given its live params below
            # A default deployment whose id is not a string is called as Python copies it, id and all.
            as_str_mapping(default_deployment_copy(self._core._resolved.default_deployment, requested)) or {}  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            if step.get("deployment_id") == DEFAULT_DEPLOYMENT_COPIED_ID
            else for_requested_model(typed_deployment(as_str_mapping(step.get("deployment")) or {}), requested)
        )
        # The values its JSON record carries as text (an Exception `mock_response`) called as configured.
        deployment = _with_live_params(self._core, deployment)
        self._attempts += 1
        self._attempt_started = datetime.now()  # noqa: DTZ005  # naive to match the logging pipeline's api_call_start_time
        level: Final = step.get("fallback_level")
        # The request params the walk's dict targets merged in (`kwargs.update(target)`).
        params: Final = as_str_mapping(step.get("params"))
        call: Final = attempt_call(
            self._core._port,  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            self._plan,
            self._request if params is None else {**self._request, **params},
            deployment,
            group=requested,
            requested=requested,
            fallback_level=level if isinstance(level, int) else 0,
        )
        self._strip_undecryptable_reasoning(step, deployment, call.kwargs)
        shaped: Final = shape_attempt_kwargs(call.kwargs, deployment)
        # The walk's metadata for the attempt's level (retries, fallback hops) and its hop kwargs.
        call.kwargs = self._walk.stamp(step, shaped, deployment) if self._plan.stamps_walk else shaped
        field: Final = self._plan.metadata_field
        if isinstance(call.served, dict) and field in call.kwargs:
            call.served[field] = call.kwargs[field]
        self._served = (requested, call.served)
        self._latest_call = (deployment, call)
        if call.refused is None:
            self._core._endpoints.counters.started(self._plan, call)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        return deployment, call

    def _strip_undecryptable_reasoning(
        self, step: Mapping[str, object], deployment: Mapping[str, object], kwargs: Mapping[str, object]
    ) -> None:
        """``_update_kwargs_with_deployment`` on a fallback hop (``_is_fallback_hop``): the encrypted
        reasoning the target cannot decrypt is dropped; reasoning with no origin marker came from
        the deployment that just failed."""
        depth: Final = step.get("fallback_depth")
        if not isinstance(depth, int) or depth <= 0:
            return
        EncryptedContentAffinityCheck.strip_reasoning_the_targets_cannot_decrypt(
            RoutingContext(self._core),  # pyright: ignore[reportArgumentType]  # the check reads registry()
            kwargs.get("input"),
            kwargs.get("messages"),
            (deployment,),
            unmarked_origin=self._walk.deployment_that_just_failed(step) or self._cut_deployment,
        )

    @property
    def response(self) -> object:
        return self._last_response

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def _stream_attempt(
        self, step: Mapping[str, object], kwargs: Mapping[str, object], slot: "contextlib.ExitStack | None" = None
    ) -> StreamAttempt:
        def number(key: str) -> int:
            value: Final = step.get(key)
            return value if isinstance(value, int) else 0

        order: Final = step.get("target_order")
        deployment_id: Final = step.get("deployment_id")
        routed: Final = step.get("routed")
        return StreamAttempt(
            core=self._core,
            op=self._op,
            request=self._request,
            kwargs=kwargs,
            group=str(step.get("model") or step.get("model_group")),
            order=order if isinstance(order, int) else None,
            depth=number("fallback_depth"),
            level=number("fallback_level"),
            deployment_id=deployment_id if isinstance(deployment_id, str) else None,
            attempted_retries=number("retry"),
            routed=routed if isinstance(routed, str) else None,
            slot=slot,
        )

    def _failed(
        self,
        step: Mapping[str, object],
        deployment: Mapping[str, object],
        call: "AttemptCall | None",
        error: BaseException,
    ) -> Mapping[str, object]:
        # What litellm's failure logging read, before the attempt helper's `except` wrote on it.
        detail: Final = str(error)
        if call is not None:
            _log_attempt(self._plan, call.model_name, error)
        served: Final = call.served if call is not None else {}  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        stamp_error(self._plan, error, deployment, served)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
        self._core._endpoints.attempt_failed(self._plan, error, deployment, call, str(step.get("model")))  # pyright: ignore[reportPrivateUsage]  # see above
        self._last_error = error
        level: Final = step.get("fallback_level")
        retry: Final = step.get("retry")
        if isinstance(retry, int) and retry > 0:
            self._chain_to_hop_first(error, level if isinstance(level, int) else 0)
        self._level_errors[level if isinstance(level, int) else 0] = error
        logged: Final = logged_call(
            call.host_op if call is not None else self._plan.host_op,
            served or self._request,  # pyright: ignore[reportUnknownArgumentType]  # see above
            self._attempt_started,
        )
        model_info: Final = as_str_mapping(deployment.get("model_info")) or {}
        exits: Final = self._core.failure_exits(error, logged, model_info.get("id"))
        # A call under credentials the request supplied ran as a request-scoped deployment
        # (`_handle_clientside_credential`): its failure is never the shared deployment's.
        ran_as: Final = (as_str_mapping(as_str_mapping(served).get("model_info") if served else None) or {}).get("id")  # pyright: ignore[reportUnknownArgumentType]  # see above
        request_scoped: Final = self._core.ran_as_request_deployment(
            ran_as, model_info.get("id")
        ) or self._caller_entra_failure(error, served, model_info.get("id"))  # pyright: ignore[reportUnknownArgumentType]  # see above
        return {
            **attempt_failure(error),
            "detail": detail,
            **exits,
            **({"outside_deployment": True} if request_scoped or _raised_before_provider(error) else {}),
            # `async_deployment_callback_on_failure` does not run for a sync call, and skips a batch retrieve.
            "uncounted_request": self._plan.sync
            or is_batch_retrieve_call_type(logged.get("call_type"))
            or _raised_before_provider(error),
        }

    def _records_blocked(
        self, failed: Mapping[str, object], call: "AttemptCall | None", error: BaseException, in_call: bool
    ) -> bool:
        """Record a failure the shared logging object skipped (a retry or later hop), like the first one."""
        from litellm.router_utils.fallback_event_handlers import (
            _failure_callbacks_blocked_logging_object,  # pyright: ignore[reportPrivateUsage]  # the fallback path's own rule
        )

        if not in_call or call is None or failed.get("outside_deployment") is True:
            return False
        if self._plan.sync:
            if not failure_already_logged(call.kwargs, "sync_failure"):
                return False
        elif _failure_callbacks_blocked_logging_object(call.kwargs) is None:
            return False
        if not self._core.outcome_callbacks(call.kwargs, sync=self._plan.sync).get("failure", True):
            return False
        if self._core.blocked_failure_exempt(error, call.kwargs, self._plan.kind in _GENERIC_API_KINDS):
            return False
        self._recorded_blocked.add(id(error))
        return True

    def _caller_entra_failure(self, error: BaseException, sent: Mapping[str, object], deployment_id: object) -> bool:
        """``is_caller_entra_credential_failure``: Entra ID rejected a credential the caller sent, not
        the deployment's own, so the failure is the caller's."""
        from litellm.router_utils.cooldown_handlers import is_caller_entra_credential_failure

        if not isinstance(deployment_id, str):
            return False
        hosted: Final = self._core._hosted  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        defaults: Final = {k: v for k, v in self._core._resolved.default_litellm_params.items() if v is not None}  # pyright: ignore[reportPrivateUsage]  # see above
        return is_caller_entra_credential_failure(
            hosted._core_get_deployment(deployment_id),  # pyright: ignore[reportPrivateUsage]  # the host registry's id lookup
            defaults,
            {"litellm_params": sent},
            getattr(error, "status_code", ""),
            error,
        )

    def _chain_to_hop_first(self, error: BaseException, level: int) -> None:
        """A retry's failure is raised while the hop's first failure is handled (the retry loop
        runs in its ``except`` block), so it chains to it."""
        first: Final = self._hop_first.get(level)
        if first is not None and first is not error:
            error.__context__ = first  # rebind-ok: chained as the retry loop's raise chains it

    def _succeeded(
        self, call: "AttemptCall", response: object, host_op: str, stream: "StreamAttempt | None" = None
    ) -> Mapping[str, object]:
        self._last_response = response
        _log_attempt(self._plan, call.model_name)
        if self._fallback_depth > 0:
            from litellm._logging import verbose_router_logger

            # One line per `run_async_fallback` the served hop returns through.
            for _ in range(self._fallback_depth):
                verbose_router_logger.info("Successful fallback b/w models.")
        tokens: Final = total_tokens(response)
        # A wrapped stream reports its success or failure when it ends; the attempt counts the
        # request and the tokens read so far, which the end's report leaves out.
        # A realtime attempt counts its session's success itself (`attempt`), when it ended cleanly.
        deferred: Final = (stream is not None and stream.reports_outcome) or reports_own_success(call)
        if stream is not None and deferred:
            stream.counted_tokens = tokens  # rebind-ok: what the end's report leaves out
        return {
            "ok": True,
            "total_tokens": tokens,
            "blocked": self._blocked(response, host_op),
            "deferred": deferred,
            "tracks_usage": self._tracks_usage(call, response, deferred),
        }

    def _tracks_usage(self, call: "AttemptCall", response: object, deferred: bool) -> bool:
        """Whether the attempt counts against its deployment's usage (``_deployment_tracks_usage``). A
        ``None`` response counts nothing (``increment_deployment_usage_for_response``), and an attempt
        that reports no success then is a realtime session, which logs none to count it either. A
        Responses socket whose finished turns each counted their own request adds none at close."""
        if deferred and response is None:
            return False
        if turns_counted_their_requests(call.kwargs):
            return False
        model: Final = call.kwargs.get("model")
        return self._core.deployment_tracks_usage(
            (as_str_mapping(call.kwargs.get("model_info")) or {}).get("id"),
            model if isinstance(model, str) else call.group,
        )

    async def _refused(
        self,
        step: Mapping[str, object],
        deployment: Mapping[str, object],
        call: "AttemptCall",
        error: BaseException,
    ) -> Mapping[str, object]:
        """A routing pre-call check refused the attempt (``async_routing_strategy_pre_call_checks``):
        a rate limit cools the deployment down (``set_cooldown_deployments``); the attempt's failure
        is recorded by nothing else."""
        import litellm.exceptions as litellm_exceptions

        core: Final = self._core
        if isinstance(error, litellm_exceptions.RateLimitError):
            deployment_id: Final = (as_str_mapping(deployment.get("model_info")) or {}).get("id")
            if isinstance(deployment_id, str):
                failure: Final = {**attempt_failure(error), "caller_scoped": core.per_user_deployment(deployment_id)}
                report: Final = core._handle.report_failure(deployment_id, failure, uncounted=True)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                done: Final = await core._drive_cache_steps(report)  # pyright: ignore[reportPrivateUsage]  # see above
                _ = core.cooled_down(done, deployment_id, error.status_code)
        return {**self._failed(step, deployment, call, error), "recorded": True}

    def _pick_kwargs(self, step: Mapping[str, object]) -> Mapping[str, object]:
        """The request kwargs the pick sees: the request with its level's retry count."""
        count: Final = self._walk.request_retry_count(step)
        if count is None:
            return self._request
        field: Final = self._plan.metadata_field
        metadata: Final = as_str_mapping(self._request.get(field)) or {}
        return {**self._request, field: {**metadata, "request_retry_count": count}}

    def _with_deployment(self, step: Mapping[str, object], resolved: object) -> Mapping[str, object]:
        return step if resolved is step.get("deployment") else {**step, "deployment": resolved}

    async def _resolve_pods(self, step: Mapping[str, object]) -> Mapping[str, object]:
        """``async_resolve_pods_after``: a picked deployment with Kubernetes pod discovery on is
        sent to one of its service's pods."""
        if self._plan.kind in _UNPICKED_KINDS or step.get("deployment") is None:
            return step
        discovery: Final = self._core._pick_discovery(sync=False)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        if discovery is None:
            return step
        resolved: Final[object] = await discovery.async_resolve_deployment(
            step.get("deployment"), self._pick_kwargs(step)
        )
        return self._with_deployment(step, resolved)

    def _resolve_pods_sync(self, step: Mapping[str, object]) -> Mapping[str, object]:
        """``resolve_pods_after``, on the sync pick."""
        if self._plan.kind in _UNPICKED_KINDS or step.get("deployment") is None:
            return step
        discovery: Final = self._core._pick_discovery(sync=True)  # pyright: ignore[reportPrivateUsage]  # see _resolve_pods
        if discovery is None:
            return step
        return self._with_deployment(
            step, discovery.resolve_deployment(step.get("deployment"), self._pick_kwargs(step))
        )

    def _default_copy(self, step: Mapping[str, object]) -> Mapping[str, object]:
        """The default deployment as Python copied it for the call that fails on it: the error is
        stamped from this copy, never from the id router-core serves it under."""
        copied: Final = default_deployment_copy(self._core._resolved.default_deployment, str(step.get("model")))  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        return as_str_mapping(copied) or {}

    def _default_call_error(self, step: Mapping[str, object]) -> BaseException | None:
        """The error a call to a default deployment every call fails to raises (its first
        reads of the copy, before any endpoint runs): nothing records it against a deployment."""
        if step.get("deployment_id") != DEFAULT_DEPLOYMENT_FAILING_ID:
            return None
        return default_deployment_call_error(self._core._resolved.default_deployment)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core

    async def _passthrough_attempt(self, step: Mapping[str, object]) -> Mapping[str, object]:
        """``_ageneric_api_call_with_fallbacks_helper`` with ``passthrough_on_no_deployment``: the pick
        failed, so the handler is called under the attempt's name. Nothing was picked, so nothing is
        stamped on a failure or recorded against a deployment."""
        passthrough: Final = self._passthrough
        if passthrough is None:
            raise RuntimeError("router-core asked for an unrouted attempt the host did not offer")
        name: Final = str(step.get("model"))
        params: Final = as_str_mapping(step.get("params"))
        self._attempts += 1
        self._attempt_started = datetime.now()  # noqa: DTZ005  # see _call
        try:
            answered: Final = await passthrough(self._request if params is None else {**self._request, **params}, name)
        except Exception as error:  # noqa: BLE001  # handed to the Rust call loop like any attempt failure
            return {**self._failed(step, {}, None, error), "recorded": True}
        self._last_response = answered
        self._served = (name, self._request)
        return {"ok": True, "total_tokens": total_tokens(answered)}

    async def attempt(self, step: Mapping[str, object]) -> Mapping[str, object]:
        if self._passthrough is not None and not step.get("deployment_id"):
            return await self._passthrough_attempt(step)
        with routed_attempt() as attempt:
            return await self._routed_attempt(step, attempt)

    def _release(
        self, call: AttemptCall, attempt: RoutedAttempt, *, refund_io_tokens: bool
    ) -> "tuple[asyncio.Task[None], ...]":
        """What a failed or cancelled attempt still holds, whatever its logging object (shared across
        attempts, it logs only the first failure). Tokens a provider already spent are kept counted."""
        core: Final = self._core
        releases: Final = (
            release_started_attempt(core._selectors.reservations, core._cache, attempt.attempt_id),  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            release_io_token_reservation(
                core._usage,  # pyright: ignore[reportPrivateUsage]  # see above
                call.kwargs,  # pyright: ignore[reportArgumentType]  # the call's kwargs dict
            )
            if refund_io_tokens
            else None,
        )
        return tuple(release for release in releases if release is not None)

    async def _routed_attempt(self, step: Mapping[str, object], attempt: RoutedAttempt) -> Mapping[str, object]:
        step = await self._resolve_pods(step)  # rebind-ok: the picked deployment, resolved to a pod
        deployment: Mapping[str, object] = as_str_mapping(step.get("deployment")) or {}  # rebind-ok: the picked one
        call: AttemptCall | None = None  # rebind-ok: built inside the try
        # Whether a failure raised now runs the failure callbacks: none raised before the litellm
        # call or after it answered does, nor any but the first on a shared logging object.
        logs_failure: bool = False  # rebind-ok: set around the litellm call
        in_call: bool = False  # rebind-ok: set around the litellm call, whichever attempt logs its failure
        reading_stream: StreamAttempt | None = None  # rebind-ok: set while the first chunk is read
        refused_call: Final = self._default_call_error(step)
        if refused_call is not None:
            return {**self._failed(step, self._default_copy(step), None, refused_call), "recorded": True}
        try:
            deployment, call = self._call(step)  # rebind-ok: rebound as the Python router rebinds it
            enforce_total_timeout_scope(call.kwargs)
            if call.refused is not None:
                await log_refused_call(call, call.refused)
                raise call.refused
            host: Final = self._core._host  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            try:
                compacted: Final = await compacted_attempt(self._core._port, self._plan, deployment, call)  # pyright: ignore[reportPrivateUsage]  # see above
            except Exception as refused:  # noqa: BLE001  # raised before the provider call, so no cooldown reads it
                return {**self._failed(step, deployment, call, refused), "recorded": True}
            # The deployment's max_parallel_requests slot; an open chat stream takes it along.
            port = self._core._port  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core  # rebind-ok: the same port on every attempt
            with enter_deployment_slot(port, self._plan, deployment, call.kwargs) as slot:
                # A search tool is no deployment: nothing to check; Python's `_arerank` and
                # `amoderation` run no routing pre-call checks either.
                if self._plan.kind not in _NO_ROUTING_PRE_CALL_CHECKS:
                    try:
                        await routing_pre_call_checks(
                            self._core._checks.routing_callbacks(),  # pyright: ignore[reportPrivateUsage]  # see above
                            as_str_mapping(step.get("deployment")) or {},
                        )
                    except Exception as refused:  # noqa: BLE001  # handed to the Rust call loop as the attempt's failure
                        # Only `_acompletion` hands the check its logging object.
                        if self._plan.kind == "chat":
                            _log_refused_pick(self._request, refused)
                        return await self._refused(step, deployment, call, refused)
                first_to_log: Final = not failure_already_logged(call.kwargs, "async_failure")
                logs_failure = first_to_log  # rebind-ok: whether a failure here is this attempt's to log
                in_call = True  # rebind-ok: see above
                pending: Final[object] = (
                    self._core._endpoints.create_file_everywhere(  # pyright: ignore[reportPrivateUsage]  # see above
                        _with_target_order(call.kwargs, step.get("target_order")),
                        call.model_name,
                        self._request.get("specific_deployment"),
                    )
                    if self._plan.kind == "create_file"
                    else host.invoke(call.host_op)(**compacted)
                )
                # The generic helper awaits whatever the handler returns, as the Python router does.
                awaited: Final = self._plan.kind == "generic" or inspect.isawaitable(pending)
                answered: Final[object] = await pending if awaited else pending  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]  # see above
                logs_failure = False  # rebind-ok: see above
                in_call = False  # rebind-ok: see above
                self._core._endpoints.check_response(self._plan, answered, call)  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]  # see above
                stream: Final = self._stream_attempt(step, call.kwargs, slot)
                logs_failure = first_to_log  # rebind-ok: the stream's own failure logging, for its first chunk
                in_call = True  # rebind-ok: see above
                reading_stream = stream  # rebind-ok: None until the attempt opens a stream
                response: Final = await open_attempt_stream(stream, answered)  # pyright: ignore[reportUnknownArgumentType]  # see above
                logs_failure = False  # rebind-ok: see above
                in_call = False  # rebind-ok: see above
                # `self.success_calls[model_name] += 1`, inside the try: a counter replaced with a
                # plain dict fails the attempt.
                self._core._endpoints.counters.succeeded(self._plan, call)  # pyright: ignore[reportPrivateUsage]  # see above
        except asyncio.CancelledError:
            if call is not None:
                await released(
                    self._release(call, attempt, refund_io_tokens=not attempt.answered),
                    ATTEMPT_SETTLEMENT_GRACE_SECONDS,
                )
            raise
        except TotalTimeoutNotApplicable:
            raise
        except Exception as error:  # noqa: BLE001  # every attempt failure is classified and handed to the Rust call loop
            if call is not None:
                await released(self._release(call, attempt, refund_io_tokens=True))
            failed: Final = self._failed(step, deployment, call, error)
            if reading_stream is None or not logs_failure:
                return {
                    **failed,
                    "recorded": not logs_failure and not self._records_blocked(failed, call, error, in_call),
                }
            # `_dispatch_stream_failure`: the stream logs its failure in a task of its own.
            reading_stream.report_first_chunk_failure(error)
            return {**failed, "recorded": True}
        served: Final = self._succeeded(call, response, call.host_op, stream)
        # The session re-pin the Python core's success callback runs; a wrapped stream runs it when it ends.
        if not served.get("deferred") and self._core.outcome_callbacks(call.kwargs, sync=False).get("success", True):
            await self._core.repin_after_success(stream.logged(), response)
        session_deployment: Final = clean_realtime_session_deployment(call)
        if session_deployment is not None:
            await self._core.count_realtime_session_success(session_deployment)
        return served

    def attempt_sync(self, step: Mapping[str, object]) -> Mapping[str, object]:
        step = self._resolve_pods_sync(step)  # rebind-ok: see attempt
        deployment: Mapping[str, object] = as_str_mapping(step.get("deployment")) or {}  # rebind-ok: see attempt
        call: AttemptCall | None = None  # rebind-ok: see attempt
        logs_failure: bool = False  # rebind-ok: see attempt
        in_call: bool = False  # rebind-ok: see attempt
        self._reading_first_chunk = False
        refused_call: Final = self._default_call_error(step)
        if refused_call is not None:
            return {**self._failed(step, self._default_copy(step), None, refused_call), "recorded": True}
        try:
            deployment, call = self._call(step)  # rebind-ok: rebound as the Python router rebinds it
            if call.refused is not None:
                log_refused_call_sync(call, call.refused)
                raise call.refused
            try:
                self._core._endpoints.sync_pre_call_checks(self._plan, deployment, self._request)  # pyright: ignore[reportPrivateUsage]  # see above
            except Exception as refused:  # noqa: BLE001  # handed to the Rust call loop like any attempt failure
                # The sync path records no outcome for a refused pre-call check: no cooldown.
                return {**self._failed(step, deployment, call, refused), "recorded": True}
            first_to_log: Final = not failure_already_logged(call.kwargs, "sync_failure")
            logs_failure = first_to_log  # rebind-ok: see attempt
            in_call = True  # rebind-ok: see attempt
            answered: Final = self._core._host.invoke(call.host_op)(**call.kwargs)  # pyright: ignore[reportPrivateUsage]  # a sync endpoint
            logs_failure = False  # rebind-ok: see attempt
            in_call = False  # rebind-ok: see attempt
            if inspect.isawaitable(answered):
                raise _NotSynchronous(f"the host's {call.host_op} endpoint is not synchronous")
            self._core._endpoints.check_response(self._plan, answered, call)  # pyright: ignore[reportPrivateUsage]  # see above
            stream: Final = self._stream_attempt(step, call.kwargs)
            logs_failure = first_to_log  # rebind-ok: see attempt
            in_call = True  # rebind-ok: see attempt
            self._reading_first_chunk = True
            response: Final = open_attempt_stream_sync(stream, answered)
            self._reading_first_chunk = False
            logs_failure = False  # rebind-ok: see attempt
            in_call = False  # rebind-ok: see attempt
            self._core._endpoints.counters.succeeded(self._plan, call)  # pyright: ignore[reportPrivateUsage]  # see attempt
        except _NotSynchronous:
            raise
        except Exception as error:  # noqa: BLE001  # see attempt
            failed: Final = self._failed(step, deployment, call, error)
            return {**failed, "recorded": not logs_failure and not self._records_blocked(failed, call, error, in_call)}
        return self._succeeded(call, response, "", stream)

    async def select(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        messages, input = self._plan.pick_input(request)
        if step.get("stage") == "callback_filters":
            answer, error = await self._core._answer_callback_filters(step, request, messages)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            if isinstance(error, Exception):
                # The try failed with the filter's error: the level raises it if nothing else serves.
                _ = self._pick_raised(error, logged=True)
            return answer
        try:
            return await self._core._answer_select(step, request, messages, input)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        except Exception as error:  # noqa: BLE001  # handed to the Rust call loop as the try's failure
            return self._pick_raised(error, logged=True)

    def select_sync(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        try:
            return self._core._answer_select_sync(step, request, *self._plan.pick_input(request))  # pyright: ignore[reportPrivateUsage]  # see above
        except Exception as error:  # noqa: BLE001  # see select
            return self._pick_raised(error, logged=False)

    async def pre_route(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        try:
            return await self._core._pre_route(  # pyright: ignore[reportPrivateUsage]  # see above
                step, request, *self._plan.pick_input(request), call_type=self._plan.handler
            )
        except Exception as error:  # noqa: BLE001  # see select
            return self._pick_raised(error, logged=True)

    def pre_route_sync(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        try:
            return self._core._custom_pick_sync(step, request, *self._plan.pick_input(request))  # pyright: ignore[reportPrivateUsage]  # see above
        except Exception as error:  # noqa: BLE001  # see select
            return self._pick_raised(error, logged=False)

    async def allow_fallback(self, target: str) -> object:
        """`_is_fallback_target_authorized`, then `_is_fallback_target_within_budget`. A check that raises
        ends the chain outside any hop's ``try``, as a chain's own failure: the level raises what it
        would have (``async_function_with_fallbacks_common_utils`` catches it), and its note or an
        order path reads the error as the next level's."""
        from litellm.router_utils.fallback_event_handlers import get_pre_routing_selection
        from litellm.types.router import FallbackSkipReason

        ctx: Final = RoutingContext(self._core)  # pyright: ignore[reportArgumentType]  # the checks read registry() and settings()
        request: Final = dict(self._request)
        if self._plan.kind in _GENERIC_API_KINDS:
            # The generic helpers' kwargs carry the function they call, which the checks read (a
            # search fallback runs the search tool grants).
            request["original_generic_function"] = self._core._host.invoke(self._plan.host_op)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the driver serves its own core
        try:
            if target != get_pre_routing_selection(request) and not await ctx.fallback_access_allowed(
                model=target, request_kwargs=request
            ):
                self._walk.targets().record_skip(target, FallbackSkipReason.NOT_AUTHORIZED)
                return False
            if not await ctx.fallback_budget_allowed(model=target, request_kwargs=request):
                self._walk.targets().record_skip(target, FallbackSkipReason.OVER_BUDGET)
                return False
            if target not in self._walk.targets() and not await ctx.fallback_rate_limit_allowed(
                model=target, request_kwargs=request
            ):
                self._walk.targets().record_skip(target, FallbackSkipReason.RATE_LIMITED)
                return False
        except Exception as error:  # noqa: BLE001  # handed to the Rust call loop, which ends the chain
            self._level_errors[self._level + 1] = error
            return {**attempt_failure(error), "raised": True}
        return True

    def refuse_fallbacks(self, step: Mapping[str, object]) -> bool:
        """``run_async_fallback``'s check of a chain's dict targets: a refused chain ends the walk
        with the check's error, which the first hop's level would have raised."""
        refused: Final = rejected_fallback_target(_as_list(step.get("targets")))
        if refused is None:
            return False
        level: Final = step.get("level")
        self._level_errors[(level if isinstance(level, int) else 0) + 1] = refused
        return True

    def _head_response(self) -> object | None:
        """``set_response_headers`` up to its usage read: the deployment id, the model group the
        response was served as and the routers' decisions."""
        response: Final = prepare_response_for_header_attachment(self._last_response)
        if response is None:
            return None
        name, kwargs = self._served
        additional: Final = ensure_response_additional_headers(response)
        apply_response_model_id(response, find_deployment_metadata(kwargs))
        additional["x-litellm-model-group"] = name
        apply_quality_router_decision_headers(additional, kwargs)
        additional.update(complexity_router_decision_headers(kwargs))
        return response

    def _count_response(self, response: object, step: Mapping[str, object]) -> object:
        """What the wrappers above `make_call` stamp: ``async_function_with_retries``'s retry count,
        then the fallback count of ``async_function_with_fallbacks`` and ``run_async_fallback``."""
        retries: Final = step.get("attempted_retries")
        if isinstance(retries, int):
            max_retries: Final = step.get("max_retries")
            response = add_retry_headers_to_response(  # rebind-ok: gains the counts
                response=response,
                attempted_retries=retries,
                max_retries=max_retries if isinstance(max_retries, int) and retries > 0 else None,
            )
        fallbacks: Final = step.get("attempted_fallbacks")
        attempted: Final = fallbacks if isinstance(fallbacks, int) else 0
        return add_fallback_headers_to_response(
            response=self.outcomes.add_headers(response, attempted), attempted_fallbacks=attempted
        )

    def log_outcomes_sync(self) -> None:
        if pending := self.outcomes.take_pending():
            from litellm.litellm_core_utils import asyncify

            asyncify.run_async_function(await_all, pending)  # pyright: ignore[reportUnknownMemberType]  # the untyped sync bridge

    def finish(self, step: Mapping[str, object]) -> object:
        """The response with the router's headers, in the order ``make_call`` and the wrappers above it
        stamp them: a usage read that raises leaves the counts off."""
        if not self._plan.sets_headers:
            self.log_outcomes_sync()
            return self._last_response
        response: Final = self._head_response()
        if response is None:
            self.log_outcomes_sync()
            return self._last_response
        name: Final = self._served[0]
        if name is not None:
            core: Final = self._core
            remaining: Final = core._hosted._core_get_remaining_model_group_usage_sync(  # pyright: ignore[reportPrivateUsage]  # the core reads its own host registry
                name, core.io_token_usage_sync
            )
            apply_remaining_usage_headers(ensure_response_additional_headers(response), remaining)
        counted: Final = self._count_response(response, step)
        self.log_outcomes_sync()
        return counted

    async def finish_async(self, step: Mapping[str, object]) -> object:
        """``finish``, with the usage read on the call's own loop."""
        if not self._plan.sets_headers:
            await await_all(self.outcomes.take_pending())
            return self._last_response
        response: Final = self._head_response()
        if response is None:
            await await_all(self.outcomes.take_pending())
            return self._last_response
        name: Final = self._served[0]
        if name is not None:
            remaining: Final = await self._core.health().usage(name, kind="remaining")
            if isinstance(remaining, Mapping):
                apply_remaining_usage_headers(ensure_response_additional_headers(response), dict(remaining))
        counted: Final = self._count_response(response, step)
        await await_all(self.outcomes.take_pending())
        return counted

    def _has_input(self) -> bool:
        """The request carries messages or an input, which region routing reads."""
        return self._request.get("messages") is not None or self._request.get("input") is not None

    def failed(self, step: Mapping[str, object]) -> BaseException:
        model: Final = str(self._request.get("model"))
        primary: Final = self._level_errors.get(self._base_level)
        if primary is None:
            return self._core._failure_error(  # pyright: ignore[reportPrivateUsage]  # see above
                step, model, self._last_error, request=self._request, has_input=self._has_input(), tags=not self._sync
            )
        if (
            self._base_level == 0
            and self._explains_failure
            and self._core._host.dynamic().expose_router_debug_in_errors  # pyright: ignore[reportPrivateUsage]  # see above
            and hasattr(primary, "message")
        ):
            primary.message += self._explanation(primary, model)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
        return primary

    def _hint_level_failure(self, error: BaseException | None, model: str) -> None:
        from litellm.router_utils.fallback_event_handlers import check_non_standard_fallback_format

        if (
            error is None
            or not hasattr(error, "message")
            or not self._core._host.dynamic().expose_router_debug_in_errors  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        ):
            return
        walked: Final = self._core._walked_fallbacks  # pyright: ignore[reportPrivateUsage]  # see above
        fallbacks: Final = self._request.get("fallbacks", walked["fallbacks"])
        if check_non_standard_fallback_format(fallbacks=_as_list(fallbacks)):
            return
        hint: Final = self._kind_hint(error, model, fallbacks)
        if hint:
            error.message += hint  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the router appends to the error it raises

    def _kind_hint(self, error: BaseException, model: str, fallbacks: object) -> str:
        """The hint a context-window or content-policy error gets when its own list is unset."""
        import litellm.exceptions as litellm_exceptions
        from litellm.router_utils.fallback_event_handlers import fallback_lookup_groups

        walked: Final = self._core._walked_fallbacks  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        lookup: Final = fallback_lookup_groups(self._request, model)
        if isinstance(error, litellm_exceptions.ContextWindowExceededError):
            if self._request.get("context_window_fallbacks", walked["context_window_fallbacks"]) is not None:
                return ""
            return "\n" + format_context_window_fallback_hint(model, None, fallbacks, lookup)
        if isinstance(error, litellm_exceptions.ContentPolicyViolationError):
            if self._request.get("content_policy_fallbacks", walked["content_policy_fallbacks"]) is not None:
                return ""
            return "\n" + format_content_policy_fallback_hint(model, None, lookup)
        return ""

    def _explanation(self, primary: BaseException, model: str) -> str:
        """What ``async_function_with_fallbacks_common_utils`` appends to the error it re-raises
        under ``expose_router_debug_in_errors``, from the resolved settings and what the call did."""
        import litellm.exceptions as litellm_exceptions
        from litellm.router_utils.fallback_event_handlers import (
            check_non_standard_fallback_format,
            fallback_lookup_groups,
            get_fallback_model_group_for_lookup_groups,
        )

        walked: Final = self._core._walked_fallbacks  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        request: Final = self._request
        # The request's own lists replace the router's (`kwargs.get("fallbacks", router)`).
        fallbacks: Final = request.get("fallbacks", walked["fallbacks"])
        window_list: Final = request.get("context_window_fallbacks", walked["context_window_fallbacks"])
        policy_list: Final = request.get("content_policy_fallbacks", walked["content_policy_fallbacks"])
        lookup: Final = fallback_lookup_groups(request, model)

        def group_of(source: object) -> tuple[Sequence[object] | None, int | None]:
            return get_fallback_model_group_for_lookup_groups(_as_list(source), lookup)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape

        def outcome(group: Sequence[object] | None) -> str:
            return format_fallback_outcome_message(
                model, group, self._fallback_detail(primary), skipped=self._walk.targets().skipped
            )

        # `context_window_fallback_model_group`: only a chain the group matched takes precedence.
        window_group: Final = (
            group_of(window_list)[0]
            if isinstance(primary, litellm_exceptions.ContextWindowExceededError) and window_list is not None
            else None
        )
        if window_group is None and check_non_standard_fallback_format(fallbacks=_as_list(fallbacks)):
            return outcome(_as_list(fallbacks))
        if window_group is not None:
            return outcome(window_group)
        # A content-policy list decides on its own: one with no match raises into the `except`.
        if isinstance(primary, litellm_exceptions.ContentPolicyViolationError) and policy_list is not None:
            return outcome(group_of(policy_list)[0])
        if fallbacks is None or not lookup:
            return outcome(None)
        found, generic = group_of(fallbacks)
        group: Final[Sequence[object] | None] = (
            found if found is not None or generic is None else _as_list(_as_list(fallbacks)[generic]["*"])  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType, reportUnknownArgumentType]  # the entry the lookup matched
        )
        return format_no_fallback_group_message(lookup, _as_list(fallbacks)) if group is None else outcome(group)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape

    def _fallback_detail(self, primary: BaseException) -> str:
        """``run_async_fallback`` raises its last hop's error, or the primary's when no hop ran."""
        return compact_fallback_error_detail(redact_string(str(self._level_errors.get(1, primary))))


__all__ = [
    "Resolved",
    "RustCore",
    "RustCoreUnsupported",
    "RustHealth",
    "RustRegistry",
    "RustSettings",
    "attempt_failure",
    "failure_kind",
    "total_tokens",
    "unsupported_arguments",
]
