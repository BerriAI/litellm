"""
``RustCore``: the router core served by litellm-rust/crates/router-core through ``_native``.

The Rust core owns the registry, selection, retries, fallbacks and cooldowns. It reaches outside
itself only through its host, and the host never calls back into a Python router:

- every attempt runs through the host: ``router_attempt`` builds the request for the one
  deployment Rust picked and ``PythonHost.invoke(op)`` calls ``litellm.<op>`` with it
- the core records every outcome (cooldowns, allowed-fails counters, usage) itself, in the routing
  cache the host builds and owns (``PythonHost.routing_cache``)
- sleeps, fallback access and budget checks, and cache reads and writes are host steps

``RustCore`` does not build, wrap or read a ``PythonCore``. Settings are resolved from the
``RouterConfig`` and the host's process settings; the views (``registry()``, ``settings()``,
``health()``) are read from the Rust handle. A member the Rust core does not serve raises
``RustCoreUnsupported`` (an ``AttributeError``) instead of reaching a Python core, so deleting
``router_python_core.py`` cannot change what this core does.
"""

import contextlib
import dataclasses
import inspect
import itertools
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._internal_context import service_target
from litellm.constants import DEFAULT_COOLDOWN_TIME_SECONDS, HEALTH_CHECKS_TARGET
from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
    apply_remaining_usage_headers,
    apply_response_model_id,
    ensure_response_additional_headers,
    prepare_response_for_header_attachment,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.routing_common_utils import (
    format_content_policy_fallback_hint,
    format_fallback_outcome_message,
    format_no_fallback_group_message,
    truncate_fallback_error_detail,
)
from litellm.litellm_core_utils.routing_groups import validate_routing_strategy
from litellm.litellm_core_utils.secret_redaction import redact_string
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
from litellm.router_rust_access import caller_fields, selection_refusal
from litellm.router_rust_alerting import initialize_alerting
from litellm.router_rust_callbacks import OptionalChecks, RustOptionalChecks
from litellm.router_rust_endpoints import (
    AttemptCall,
    Plan,
    RustEndpoints,
    attempt_call,
    pinned_video_deployment,
    plan_for,
    stamp_error,
    unrouted_fields,
)
from litellm.router_rust_fallbacks import (
    MOCK_FLAGS,
    RustFallbacks,
    mock_error,
    mock_failure,
    refuse_malformed_fallbacks,
    rejected_fallback_target,
)
from litellm.router_rust_health import (
    RustHealthCallbacks,
    cooled_in_running_loop,
    failure_status,
)
from litellm.router_rust_health_routing import (
    HEALTH_STATE_KEY,
    HealthRouting,
    failover_document,
    fully_unhealthy_model_names,
    resolved_allowed_fails_policy,
)
from litellm.router_rust_health_routing import all_blocked as deployments_all_blocked
from litellm.router_rust_health_routing import fully_blocked_model_names as blocked_model_names
from litellm.router_rust_health_routing import is_blocked as deployment_is_blocked
from litellm.router_rust_members import RustRegistryMembers
from litellm.router_rust_plugins import SYNC_PATH_MESSAGE, candidate_models, run_plugins
from litellm.router_rust_precall import PreCallFacts, declares_no_vision, request_carries_image
from litellm.router_rust_prerouting import PreRouter, PreRoutingStrategies
from litellm.router_rust_registry import (
    DEFAULT_DEPLOYMENT_COPIED_ID,
    DEFAULT_DEPLOYMENT_FAILING_ID,
    INDEX_ATTRIBUTES,
    HostRegistry,
    RustRegistry,
    default_deployment_call_error,
    default_deployment_copy,
    default_deployment_copy_error,
    hosted_registry_view,
    index_document,
    served_default_deployment,
)
from litellm.router_rust_replay import track_live_registry, untrack_live_registry
from litellm.router_rust_retries import (
    GroupPolicies,
    RustRetries,
    group_policies,
    is_policy_per_group,
    policy_lookup_error,
    raising_groups,
    resolved_retry_policy,
    static_group_policies,
)
from litellm.router_rust_selection import (
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
from litellm.router_rust_settings import (
    MISSING,
    SettingsUpdate,
    apply_construction_effects,
    argument_attribute,
    default_litellm_params,
    initial_plain,
    read_settings_update,
    request_timeout,
    settings_view,
)
from litellm.router_rust_shaping import Shaping
from litellm.router_rust_slots import enter_deployment_slot, max_parallel_requests_limit
from litellm.router_rust_streaming import (
    StreamAttempt,
    continuation_flag,
    continuation_routing,
    open_attempt_stream,
    open_attempt_stream_sync,
    register_continuation_prefill_check,
    shape_attempt_kwargs,
)
from litellm.router_rust_walk import FallbackOutcomes, WalkMetadata, await_all, log_retry_into
from litellm.router_utils.cooldown_cache import ROUTER_COOLDOWNS_TARGET
from litellm.router_utils.kubernetes_pod_discovery import KubernetesPodDiscovery
from litellm.router_utils.router_callbacks.track_deployment_metrics import find_deployment_metadata
from litellm.router_utils.routing_read_batch import ROUTER_USAGE_TARGET
from litellm.types.router import (
    Deployment,
    RouterErrors,
    RouterGeneralSettings,
    RouterRateLimitError,
    RoutingGroup,
    RoutingStrategy,
)
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.router_api import UntypedDict
    from litellm.router_utils.cooldown_cache import CooldownCache
    from litellm.router_utils.health_state_cache import DeploymentHealthCache
    from litellm.rust_bridge._native import (  # pyright: ignore[reportPrivateUsage]  # the stub's step shapes
        RouterCall,
        RouterCoreHandle,
    )
    from litellm.types.router import (
        CustomRoutingStrategyBase,
        DeploymentTypedDict,
        GuardrailTypedDict,
        PreRoutingHookResponse,
        RoutingPlugin,
    )

# Async endpoints the Rust core routes: select a deployment, then run the attempt on the host.
RUST_SERVED_OPS: Final[frozenset[str]] = frozenset(
    {
        "acompletion",
        "atext_completion",
        "aembedding",
        "aresponses",
        "aanthropic_messages",
        "aimage_generation",
        "atranscription",
        "aspeech",
        "arerank",
        "amoderation",
    }
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
    "CallerCredentialError",
)
_DECLINED: Final = frozenset({"MalformedRequestFieldError", "UnsupportedEndpointError"})
_STEP: Final = TypeAdapter(Mapping[str, object])
_PROVIDERS: Final = tuple(provider.value for provider in LlmProviders)
_DEFAULT_COOLDOWN_SECONDS: Final = 5.0

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
        "enable_mid_stream_fallback_continuation",
        "redis_url",
        "redis_host",
        "redis_port",
        "redis_password",
        "redis_db",
        "cache_kwargs",
        "set_verbose",
        "debug_level",
        # Read host-side by the endpoint plans (litellm/router_rust_endpoints.py).
        "stream_timeout",
        "cache_responses",
        "assistants_config",
        "search_tools",
        "guardrail_list",
        "default_priority",
        "polling_interval",
        "ignore_invalid_deployments",
        # Read host-side by the pre-routing strategies (litellm/router_rust_prerouting.py).
        "auto_router_capability_limit",
        # Decided in router-core (precall.rs); its facts come from litellm/router_rust_precall.py.
        "enable_pre_call_checks",
        # Run host-side (litellm/router_rust_plugins.py); router-core filters on what they leave.
        "plugins",
        "retry_policy",
        "model_group_retry_policy",
        # Health-check routing and the failover flags (litellm/router_rust_health_routing.py).
        "enable_health_check_routing",
        "health_check_staleness_threshold",
        "health_check_ignore_transient_errors",
        "background_health_check_model_groups",
        "enable_weighted_failover",
        "fallback_on_unsupported_params",
        # Built host-side as CustomLogger checks (litellm/router_rust_callbacks.py); router-core runs
        # their deployment filters through its callback_filters stage.
        "optional_pre_call_checks",
        "provider_budget_config",
        "deployment_affinity_ttl_seconds",
        "session_affinity_repin_on_failover",
        "model_group_affinity_config",
        # Kept as given (litellm/router_rust_settings.py) and read host-side by the endpoint plans
        # (default_litellm_params' caching groups) and the slot limits.
        "client_ttl",
        "caching_groups",
        "default_max_parallel_requests",
        # Built host-side into a Slack alerting logger (litellm/router_rust_alerting.py).
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


class RustCoreUnsupported(AttributeError):
    """A member or request the Rust core does not serve. It is never forwarded to a Python core."""


def _cache_target(key: str) -> str:
    """The service target a routing cache read or write is attributed to, as the Python core's."""
    if key == HEALTH_STATE_KEY:
        return HEALTH_CHECKS_TARGET
    return ROUTER_COOLDOWNS_TARGET if key.endswith(":cooldown") else ROUTER_USAGE_TARGET


# --- failures -----------------------------------------------------------------------------------


def failure_kind(error: BaseException) -> str:
    names: Final = tuple(cls.__name__ for cls in type(error).__mro__)
    if not _DECLINED.isdisjoint(names):
        return "Declined"
    if is_guardrail_intervention(error):  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
        return "GuardrailIntervention"
    return next((name for name in names if name in _FAILURE_KINDS), "APIError")


def _retry_after_seconds(error: BaseException) -> float | None:
    response: Final = getattr(error, "response", None)
    headers: Final[object] = getattr(error, "litellm_response_headers", None) or getattr(response, "headers", None)
    get: Final[object] = getattr(headers, "get", None)
    if headers is None or not callable(get):
        return None
    milliseconds: Final[object] = get("retry-after-ms")
    seconds: Final[object] = get("retry-after")
    try:
        value: Final = float(str(milliseconds)) / 1000 if milliseconds is not None else float(str(seconds))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def attempt_failure(error: BaseException) -> Mapping[str, object]:
    status: Final = getattr(error, "status_code", None)
    num_retries: Final = getattr(error, "num_retries", None)
    return {
        "ok": False,
        "kind": failure_kind(error),
        "status": status if isinstance(status, int) and 100 <= status <= 599 else None,
        "retry_after": _retry_after_seconds(error),
        "num_retries": num_retries if isinstance(num_retries, int) and num_retries >= 0 else None,
        "detail": str(error)[:500],
        "classes": [cls.__name__ for cls in type(error).__mro__],
        "llm_provider": bool(getattr(error, "llm_provider", "")),
    }


def total_tokens(response: object) -> int:
    usage: Final = getattr(response, "usage", None)
    tokens: Final = getattr(usage, "total_tokens", None)
    return tokens if isinstance(tokens, int) else 0


# --- settings -----------------------------------------------------------------------------------


def _json_default(value: object) -> object:
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
        return cls(
            # An explicit `routing_strategy=None` is stored as None (no strategy picks).
            routing_strategy=None
            if strategy is None and "routing_strategy" in config.arguments
            else normalize_strategy(strategy) or "simple-shuffle",
            routing_strategy_args=MappingProxyType(dict(as_str_mapping(get("routing_strategy_args")) or {})),
            num_retries=int(str(num_retries))
            if num_retries is not None
            else dynamic.num_retries
            if dynamic.num_retries is not None
            else 2,
            max_fallbacks=int(str(max_fallbacks))
            if max_fallbacks is not None
            else dynamic.max_fallbacks
            if dynamic.max_fallbacks is not None
            else dynamic.router_max_fallbacks,
            allowed_fails=int(str(allowed_fails)) if allowed_fails is not None else dynamic.allowed_fails,
            explicit_allowed_fails=int(str(allowed_fails)) if allowed_fails is not None else None,
            cooldown_time=float(str(get("cooldown_time") or _DEFAULT_COOLDOWN_SECONDS)),
            disable_cooldowns=_optional_bool(get("disable_cooldowns")),
            allowed_fails_policy=resolved_allowed_fails_policy(get("allowed_fails_policy")),
            retry_after=float(str(get("retry_after") or 0)),
            timeout=float(str(timeout)) if timeout else dynamic.request_timeout,
            fallbacks=fallbacks,
            context_window_fallbacks=list(_as_list(get("context_window_fallbacks")))
            or (list(dynamic.context_window_fallbacks) if dynamic.context_window_fallbacks else None),
            content_policy_fallbacks=content_policy,
            model_group_alias=MappingProxyType(dict(as_str_mapping(get("model_group_alias")) or {})),
            default_litellm_params=MappingProxyType(dict(as_str_mapping(get("default_litellm_params")) or {})),
            fallback_access_check=get("fallback_access_check"),
            fallback_budget_check=get("fallback_budget_check"),
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
        through ``update_settings``: only they may name ``routing_strategy_args``, and only they
        validate a strategy)."""
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
                updates[name] = self._number_setting(name, value)
                if name == "allowed_fails":
                    updates["explicit_allowed_fails"] = updates[name]
            elif name in _SELECTION_SETTINGS:
                updates[name] = self._selection_setting(name, value, validate=name in updated)
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
        return dataclasses.replace(self, **updates, assigned=self.assigned | frozenset(values))  # pyright: ignore[reportArgumentType]  # each value converted to its field's type above

    def _number_setting(self, name: str, value: object) -> object:
        """A numeric setting converted to its field's type."""
        if name in ("num_retries", "max_fallbacks", "allowed_fails"):
            return int(str(value))
        if name == "cooldown_time":
            # Stored as given (`_core_cooldown_time`), so a failure cools down for that long:
            # 0 for no time at all; None leaves the cooldown cache's own default, set by the caller.
            return float(str(value)) if value is not None else self.cooldown_time
        if name == "retry_after":
            return float(str(value or 0))
        return float(str(value)) if value is not None else None

    def _selection_setting(self, name: str, value: object, *, validate: bool) -> object:
        """A selection setting converted to its field's type; a strategy changed through
        ``update_settings`` is validated."""
        if name == "routing_strategy":
            normalized: Final = normalize_strategy(value) or "simple-shuffle"
            if validate and normalized != self.routing_strategy:
                validate_routing_strategy(value)  # pyright: ignore[reportArgumentType]  # validated as given
            return normalized
        if name == "routing_strategy_args":
            return MappingProxyType(dict(as_str_mapping(value) or {}))
        if name == "tag_routing_prefix":
            return str(value or "")
        return bool(value)

    def static_document(self, model_list: Sequence[object]) -> Mapping[str, object]:
        values: Final = {
            "model_list": _plain(list(model_list)),
            "routing_strategy": self.routing_strategy,
            "cooldown_time": self.cooldown_time,
            "disable_cooldowns": self.disable_cooldowns,
            "allowed_fails_policy": _plain(self.allowed_fails_policy),
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

    def dynamic_document(self, version: int, model_names: Sequence[str]) -> Mapping[str, object]:
        values: Final = {
            "version": version,
            "model_group_alias": _plain(dict(self.model_group_alias)),
            "fallbacks": _plain(self.fallbacks),
            "context_window_fallbacks": _plain(self.context_window_fallbacks),
            "content_policy_fallbacks": _plain(self.content_policy_fallbacks),
            "num_retries": self.num_retries,
            "max_fallbacks": self.max_fallbacks,
            "allowed_fails": self._router_allowed_fails(),
            "fallback_keys": self._fallback_keys(model_names),
            "default_deployment": _plain(served_default_deployment(self.default_deployment)[0]),
            "default_deployment_raises": True if served_default_deployment(self.default_deployment)[1] else None,
        }
        return {name: value for name, value in values.items() if value is not None}

    def _router_allowed_fails(self) -> int | None:
        """``allowed_fails`` when it counts as set on the router (``_is_allowed_fails_set_on_router``):
        passed explicitly, or different from the process default."""
        import litellm

        if self.explicit_allowed_fails is not None:
            return self.explicit_allowed_fails
        if self.allowed_fails != litellm.allowed_fails:
            return self.allowed_fails
        return None

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
        names: Final = {*model_names, *self.model_group_alias}
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


def _string_ids(ids: object) -> tuple[str, ...]:
    """The string ids of an id list as assigned (anything else holds none)."""
    return tuple(i for i in _as_list(ids) if isinstance(i, str))


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


def _team_id(request: Mapping[str, object]) -> str | None:
    metadata: Final = as_str_mapping(request.get("metadata") or request.get("litellm_metadata"))
    team_id: Final = metadata.get("user_api_key_team_id") if metadata is not None else None
    return team_id if isinstance(team_id, str) else None


# Request fallback settings, which replace the router's for the call.
_REQUEST_FALLBACK_KEYS: Final = ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks", "max_fallbacks")


def _strings(value: object) -> "tuple[str, ...]":
    """Every string in ``value``, a name or a fallback list of names and dict entries."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        entries: Final[Mapping[object, object]] = value  # pyright: ignore[reportUnknownVariableType]  # read as objects
        return tuple(itertools.chain.from_iterable(map(_strings, itertools.chain.from_iterable(entries.items()))))
    if isinstance(value, list | tuple):
        items: Final[Sequence[object]] = value  # pyright: ignore[reportUnknownVariableType]  # see above
        return tuple(itertools.chain.from_iterable(map(_strings, items)))  # pyright: ignore[reportUnknownArgumentType]  # read as objects
    return ()


def _start_fields(request: Mapping[str, object]) -> Mapping[str, object]:
    """The request fields the Rust core routes on."""
    fields: Final = {
        key: request[key]
        for key in ("specific_deployment", "num_retries", "cooldown_time")
        if request.get(key) is not None
    }
    # `record_disable_fallbacks(kwargs, disable_fallbacks is True)`: only True turns them off.
    disabled: Final = {"disable_fallbacks": True} if request.get("disable_fallbacks") is True else {}
    lists: Final = {key: _plain(request[key]) for key in _REQUEST_FALLBACK_KEYS if key in request}
    weights: Final = request.get("_router_weights")
    team_id: Final = _team_id(request)
    strategy: Final = request.get("routing_strategy")
    override: Final = normalize_strategy(strategy) if isinstance(strategy, str | RoutingStrategy) else None
    return {
        **fields,
        **({"routing_strategy": override} if override is not None else {}),
        **disabled,
        **({"router_weights": weights} if weights is not None else {}),
        **lists,
        **({"team_id": team_id} if team_id else {}),
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

    def failures_this_minute(self, deployment_id: str) -> int:
        count: Final[object] = self.store.get_cache(f"{deployment_id}:fails", local_only=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache's reads are untyped
        return int(count) if isinstance(count, int | float) else 0

    def record_failure(self, deployment_id: str, exception: Exception, exception_status: str | int) -> bool:
        failure: Final = {**attempt_failure(exception), "status": failure_status(exception_status)}
        handle: Final = self._core._handle  # pyright: ignore[reportPrivateUsage]  # see above
        call: Final = handle.report_failure(deployment_id, failure, count_only=exception_status == "")
        done: Final = self._core._drive_sync_steps(call)  # pyright: ignore[reportPrivateUsage]  # see above
        return cooled_in_running_loop(bool(done.get("cooled")))

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
        "_core_adaptive_routers",
        "_core_routing_plugins",
        "_core_deployment_latency_map",
    }
)
_REBOUND_STATE: Final = MappingProxyType(
    {
        "_core_cache": "_cache",
        "_core_cooldown_cache": "_cooldowns",
        "_core_health_state_cache": "_health_checks",
        "_core_slack_alerting_logger": "_slack_alerting_logger",
    }
)


_HEALTH_SERVED: Final = frozenset(
    {
        "store",
        "cooldowns",
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


class RustCore(
    _SettingsSource, RustRegistryMembers, RustRetries, RustFallbacks, RustHealthCallbacks, RustOptionalChecks
):
    """A ``RouterCore`` served by the Rust router core over a ``PythonHost``. It holds no Python core."""

    def __init__(self, config: RouterConfig, host: PythonHost, resolved: Resolved) -> None:
        from litellm.router_utils.cooldown_cache import CooldownCache

        self._config: Final = config
        self._host: Final = host
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
        self._cooldowns: CooldownCache = CooldownCache(
            cache=self._cache,
            default_cooldown_time=self._cooldown_given,  # pyright: ignore[reportArgumentType]  # the constructor's value, as the Python core passes it
        )
        self._health_checks: DeploymentHealthCache = resolved.health_routing.health_checks(self._cache)
        self._version: int = 1
        self._retired: bool = False
        self._settings: RustSettings | None = None
        self._tag_filtering_models: tuple[int, bool] | None = None
        # The registry indexes written through the deprecated attributes (`_core_<name>`), and the
        # document router-core last received for them.
        self._index_written: frozenset[str] = frozenset()
        self._index_sent: Mapping[str, object] | None = None
        self._selectors: Final = StrategySelectors(host, self._cache)
        self._selectors.init_default(resolved.routing_strategy, resolved.routing_strategy_args)
        self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        self._endpoints: Final = RustEndpoints(self, config)
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
        ctx: Final = RoutingContext(self)  # pyright: ignore[reportArgumentType]  # the strategies read registry(), settings(), health() and select()
        strategies: Final = PreRoutingStrategies(
            self._hosted, host, ctx, config.get("auto_router_capability_limit"), self._cache
        )
        self._hosted.strategies = strategies
        self._prerouter: Final = PreRouter(strategies, self._hosted, host, ctx, self._cache)
        # `register_plugin`: a custom strategy that replaces every pick (the host answers the
        # Rust core's pre_route step with its deployment).
        self._custom_strategy: CustomRoutingStrategyBase | None = None
        self._custom_discovery: KubernetesPodDiscovery | None = None
        self._hosted.build(_as_list(config.get("model_list")) if config.get("model_list") is not None else None)
        self._hosted.set_routing_groups(resolved.routing_groups)
        track_live_registry(self._hosted)
        self._hosted.routable_lookup = self._routable
        self._handle_default_ids: tuple[str, ...] = ()  # the provider default ids the handle was built with
        self._hosted.max_parallel_requests_lookup = self._max_parallel_requests_client
        self._hosted._core_routing_plugins = list(resolved.routing_plugins)  # pyright: ignore[reportPrivateUsage]  # the core fills its own host registry
        self._handle: RouterCoreHandle = self._new_handle(list(self._hosted._core_model_list))  # pyright: ignore[reportPrivateUsage]  # the core reads its own host registry
        self._join_callback_holders()
        self._checks: Final = OptionalChecks(self, host, self._cache, self._hosted, ctx, config.arguments)
        self._hosted.checks = self._checks
        strategies.ensure_affinity = self._checks.ensure_deployment_affinity_callback
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
    def _registry_state(self) -> HostRegistry:
        return self._hosted

    @property
    def _python_host(self) -> PythonHost:
        return self._host

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

    def __getattr__(self, name: str) -> object:
        if name in SELECTOR_ATTRIBUTES.values():
            return self._selectors.attribute(name)
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

    def _settings_attribute(self, name: str) -> object:
        """A router attribute that reads a setting, the constructor's arguments or a constant."""
        r: Final = self._resolved
        resolved: Final[Mapping[str, object]] = {
            "num_retries": r.num_retries,
            "max_fallbacks": r.max_fallbacks,
            "timeout": r.timeout,
            "retry_after": r.retry_after,
            "explicit_allowed_fails": r.explicit_allowed_fails,
            "routing_strategy_args": r.routing_strategy_args,
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
        return RouterCoreHandle(static, self._resolved.dynamic_document(self._version, names))

    def _provider_default_ids(self) -> tuple[str, ...]:
        """``provider_default_deployment_ids`` as the cooldown check reads it (``deployment in
        settings().provider_default_deployment_ids``): the ids it holds now. router-core holds them
        with its static settings."""
        return _string_ids(self._hosted._core_provider_default_deployment_ids)  # pyright: ignore[reportPrivateUsage]  # the host registry's live list

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
        """Before a pick: the plugin stage and the registry indexes written through the deprecated
        attributes, as the Python core reads both live on every pick."""
        self._sync_plugin_stage()
        self._sync_index()

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
        """A new Rust core for changed static settings, over the same deployments."""
        old: Final = self._handle
        self._handle = self._new_handle(list(self._hosted._core_model_list))  # pyright: ignore[reportPrivateUsage]  # see __init__
        old.retire()
        # The new handle builds its indexes from the deployments: written ones are handed again.
        self._index_sent = None

    def _sync_registry(self) -> None:
        """Hand the host registry's model_list to the Rust core after a write."""
        from litellm.rust_bridge._native import RouterCoreHandle

        self._tag_filtering_models = None
        handle: Final[object] = self.__dict__.get("_handle")
        if isinstance(handle, RouterCoreHandle):
            _ = handle.apply({"replace": _plain(list(self._hosted._core_model_list))})  # pyright: ignore[reportPrivateUsage]  # see __init__
            # A wildcard deployment added an id the cooldown check skips.
            if self._provider_default_ids() != self._handle_default_ids:
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
            fallbacks=r.fallbacks,
            context_window_fallbacks=r.context_window_fallbacks,
            content_policy_fallbacks=r.content_policy_fallbacks,
            default_deployment=r.default_deployment,
            default_litellm_params=dict(r.default_litellm_params)
            if "default_litellm_params" in r.assigned
            else default_litellm_params(self._config),
            default_max_parallel_requests=self._endpoints.default_max_parallel_requests,  # pyright: ignore[reportArgumentType]  # the setting as given
            cooldown_time=self._cooldown_given,  # pyright: ignore[reportArgumentType]  # the value as given, as the Python core stores it
            allowed_fails=r.allowed_fails,
            allowed_fails_policy=r.allowed_fails_policy,  # pyright: ignore[reportArgumentType]  # the policy as passed
            disable_cooldowns=r.disable_cooldowns,
            enable_pre_call_checks=r.enable_pre_call_checks,
            enable_health_check_routing=r.health_routing.enable_health_check_routing,
            enable_tag_filtering=r.enable_tag_filtering,
            tag_filtering_match_any=r.tag_filtering_match_any,
            tag_routing_prefix=r.tag_routing_prefix,
            fallback_access_check=r.fallback_access_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            fallback_budget_check=r.fallback_budget_check,  # pyright: ignore[reportArgumentType]  # the check as passed
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
            **dict.fromkeys(self._deleted, None),  # pyright: ignore[reportArgumentType]  # settings fields only
            _unset=snapshot._unset | self._deleted,  # pyright: ignore[reportPrivateUsage]  # the snapshot's own unset set
        )

    def available_guardrail(self, name: str) -> object:
        return self._endpoints.available_guardrail(name)

    def settings_dict(self) -> Mapping[str, object]:
        r: Final = self._resolved
        latency: Final = self._selectors.latency_args() if r.routing_strategy == "latency-based-routing" else None
        # `get_settings` reports the latency selector's full routing args.
        args: Final = latency if latency is not None else dict(r.routing_strategy_args)
        values: Final = {
            "routing_strategy_args": args,
            "routing_strategy": r.routing_strategy,
            "allowed_fails": r.allowed_fails,
            "cooldown_time": self._cooldown_given,
            "num_retries": r.num_retries,
            "timeout": r.timeout,
            "retry_after": r.retry_after,
            "fallbacks": r.fallbacks,
            "context_window_fallbacks": r.context_window_fallbacks,
            "model_group_alias": dict(r.model_group_alias),
            "enable_mid_stream_fallback_continuation": r.enable_mid_stream_fallback_continuation,
            "enable_tag_filtering": r.enable_tag_filtering,
            "tag_routing_prefix": r.tag_routing_prefix,
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
        callbacks, so it binds none either."""
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
        # It replaces an async pick assigned through the deprecated attribute, as Python's does.
        if "_core_async_get_available_deployment" in self.__dict__:
            del self.__dict__["_core_async_get_available_deployment"]

    def set_custom_routing_strategy(self, CustomRoutingStrategy: "CustomRoutingStrategyBase") -> None:
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the deprecated alias notes its use like Python's
        )

        _note("set_custom_routing_strategy", "register_plugin(strategy)")
        self.register_plugin(CustomRoutingStrategy)

    def retire(self) -> None:
        self._retired = True
        untrack_live_registry(self._hosted)
        self._retire_callbacks()
        if self._hosted.strategies is not None:
            self._hosted.strategies.retire()
        self._handle.retire()

    def apply(self, changes: ChangeSet) -> ApplyResult:
        """``apply_changes``: assigned settings, then ``update_settings``, upserts, deletes and the
        Redis cache, in that order. A refused changeset changes nothing."""
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        if changes.refresh_model_info:
            raise TypeError("ChangeSet(refresh_model_info=True) needs aapply()")
        unknown: Final = sorted(set(changes.assign) - {*SETTING_FIELDS, *PLAIN_SETTING_ATTRIBUTES, *ASSIGNABLE_ALIASES})
        if unknown:
            raise ValueError(f"not assignable router settings: {unknown}")
        unserved: Final = sorted(set(changes.assign) - _STATIC_SETTINGS - _DYNAMIC_SETTINGS - _HOST_SETTINGS)
        if unserved:
            raise RustCoreUnsupported(f"the Rust router core does not serve the setting(s) {', '.join(unserved)} yet")
        update: Final = read_settings_update(changes.settings)
        # `apply_changes` assigns before it runs `update_settings`: a budget config assigned beside
        # `optional_pre_call_checks` builds the budget limiter with the new config.
        self._assign_host_settings(changes.assign)
        if update.optional_pre_call_checks is not MISSING:
            self._checks.set(update.optional_pre_call_checks)
        if update.continuation is not MISSING:
            self._set_continuation(continuation_flag(update.continuation))
        self._apply_settings(changes.assign, update)
        # Upserts first, then deletes (`apply_changes`), through the host registry, which hands
        # each change to the Rust core.
        upserted: Final = [self._hosted._core_upsert_deployment(deployment=d) for d in changes.upsert]  # pyright: ignore[reportPrivateUsage]  # see __init__
        deleted: Final = [self._hosted._core_delete_deployment(id=i) for i in changes.delete]  # pyright: ignore[reportPrivateUsage]  # see __init__
        if changes.redis_cache is not None:
            # `_core_update_redis_cache`: the routing state's cache gains the Redis cache.
            self._cache.attach_redis_cache(changes.redis_cache)
            self._checks.prompt_caching_pin_cache.attach_redis_cache(changes.redis_cache)
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
        policies: Final = {name: update.plain[name] for name in _RETRY_POLICY_SETTINGS if name in update.plain}
        values: Final = {
            **{name: value for name, value in assign.items() if name not in _HOST_SETTINGS},
            **update.values,
        }
        given: Final = dict(update.values)  # the settings passed through `update_settings`
        groups_input: Final[object] = (
            update.routing_groups if update.routing_groups is not MISSING else self._resolved.routing_groups_input
        )
        if update.routing_groups is not MISSING or self._changes_strategy(update.values):
            # `update_settings` re-inits the routing groups from the groups passed, else (on a
            # strategy change) from the groups last passed, whatever a raw write assigned since.
            given["routing_groups"] = groups_input
            values["routing_groups"] = parse_groups(
                groups_input,
                frozenset(str(d.get("model_name")) for d in self._handle.deployments()),
                frozenset(self._resolved.model_group_alias),
            )
        flags_before: Final = failover_document(self._plain)
        self._plain.update(update.plain)
        if failover_document(self._plain) != flags_before:
            self._rebuild()
        if policies:
            self._set_retry_policies(policies)
        if self._deleted & set(values):
            self._deleted = self._deleted - set(values)
            self._settings = None
        if values:
            previous: Final = self._resolved
            resolved: Final = dataclasses.replace(
                previous.changed(values, frozenset(given)), routing_groups_input=groups_input
            )
            static_changed: Final = bool(_STATIC_SETTINGS & set(values))
            self._resolved = resolved
            if set(values) & set(SETTING_FIELDS):
                self._settings = None  # the snapshot holds settings fields, not the plain attributes
            if "model_group_alias" in values:
                self._hosted._core_model_group_alias = dict(resolved.model_group_alias)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # see __init__
            self._apply_strategy(previous, resolved, given)
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
                self._handle.apply({"settings": resolved.dynamic_document(self._version, names)})

    def _changes_strategy(self, updated: Mapping[str, object]) -> bool:
        """``update_settings`` names a ``routing_strategy`` other than the router's."""
        if "routing_strategy" not in updated:
            return False
        return (normalize_strategy(updated["routing_strategy"]) or "simple-shuffle") != self._resolved.routing_strategy

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
        """``update_settings``' ``retry_policy`` and ``model_group_retry_policy``, which the retry
        loop reads from then on: router-core holds them with its static settings."""
        previous: Final = self._resolved
        resolved: Final = dataclasses.replace(
            previous,
            retry_policy=resolved_retry_policy(policies.get("retry_policy", previous.retry_policy)),
            model_group_retry_policy=policies.get("model_group_retry_policy", previous.model_group_retry_policy),
        )
        if resolved != previous:
            self._resolved = resolved
            self._rebuild()

    def _apply_strategy(self, previous: Resolved, resolved: Resolved, values: Mapping[str, object]) -> None:
        """``_core_update_settings``: a changed strategy re-inits the selectors with the args passed
        alongside it (or none); changed args alone rebuild the default selector."""
        if "routing_strategy" in values and resolved.routing_strategy != previous.routing_strategy:
            # `routing_strategy_init` drops a custom strategy (`_reset_custom_routing_strategy`),
            # and an async pick assigned through the deprecated attribute.
            self._custom_strategy = None
            if "_core_async_get_available_deployment" in self.__dict__:
                del self.__dict__["_core_async_get_available_deployment"]
            self._selectors.init_default(
                resolved.routing_strategy, as_str_mapping(values.get("routing_strategy_args")) or {}
            )
            self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        elif "routing_groups" in values:
            self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        elif "routing_strategy_args" in values and resolved.routing_strategy_args != previous.routing_strategy_args:
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
        from litellm.router_api import SETTING_FIELDS

        # The deprecated attribute deletes the core's own name (``_core_<setting>``).
        setting: Final = name.removeprefix("_core_")
        if setting not in SETTING_FIELDS:
            object.__delattr__(self, name)
            return
        if setting in self._deleted:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{setting}'")
        self._deleted = self._deleted | {setting}
        self._settings = None

    def __setattr__(self, name: str, value: object) -> None:
        # A setting write is a changeset (`router.num_retries = 2` is `apply(ChangeSet(assign=...))`),
        # a `model_list` write swaps the live list, and any other attribute is kept as written.
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        if name in PLAIN_SETTING_ATTRIBUTES or name in SETTING_FIELDS or name in ASSIGNABLE_ALIASES:
            _ = self.apply(ChangeSet(assign={name: value}))
        elif name == "model_list":
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
        **_: object,  # kwargs-ok: the RouterCore select options this core ignores
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Pick a deployment for ``model`` without calling it (``pick``), list the deployments a
        pick would choose among (``healthy``), or pick one for a pass-through route
        (``pass_through``, which picks with the sync strategies, as Python does)."""
        if kind not in ("pick", "healthy", "pass_through"):
            raise ValueError(f"unknown select kind {kind!r}")
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
        )
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
        """The proxy's ``/v1/messages`` route name for ``aanthropic_messages``; async, as in PythonCore."""
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
        return await self._drive(plan_for("arerank"), request_from_call("arerank", args, kwargs))

    async def amoderation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.amoderation(request_from_call("amoderation", args, kwargs))

    def completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("completion"), request_from_call("completion", args, kwargs))

    def text_completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._endpoints.text_completion(request_from_call("text_completion", args, kwargs))

    def embedding(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("embedding"), request_from_call("embedding", args, kwargs))

    def image_generation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("image_generation"), request_from_call("image_generation", args, kwargs))

    @property
    def total_calls(self) -> Mapping[str, int]:
        return self._endpoints.counters.total_calls

    @property
    def success_calls(self) -> Mapping[str, int]:
        return self._endpoints.counters.success_calls

    @property
    def fail_calls(self) -> Mapping[str, int]:
        return self._endpoints.counters.fail_calls

    # -- driving a call --------------------------------------------------------------------------

    def _start(
        self,
        plan: Plan,
        request: Mapping[str, object],
        overrides: Mapping[str, object] | None = None,
        *,
        prepared_already: bool = False,
    ) -> "tuple[RouterCall, Mapping[str, object]]":
        if self._retired:
            raise RuntimeError("this router core was retired")
        self._sync_routing_state()
        unserved: Final = sorted(key for key in request if key.startswith("mock_testing_") and key not in MOCK_FLAGS)
        if unserved:
            raise RustCoreUnsupported(f"request keys not served by the Rust router core yet: {unserved}")
        model: Final = request.get("model")
        if not isinstance(model, str):
            raise TypeError(f"{plan.endpoint}() needs a model name")
        field: Final = plan.prepare_field or "metadata"
        if request.get(field) is not None and not isinstance(request.get(field), Mapping):
            raise TypeError(f"{field} must be a mapping")
        prepared: Final = dict(request) if prepared_already else self._endpoints.prepare(plan, request, model)
        mock: Final = mock_failure(request)
        fields: Final[Mapping[str, object]] = {
            **self._routing_fields(prepared),
            **pinned_video_deployment(plan, request),
            **unrouted_fields(plan),
            # The sync path never runs pre-routing; a custom strategy picks on both.
            **({} if plan.sync and self._custom_strategy is None else {"pre_routes": True}),
            **({"mock_failure": mock} if mock is not None else {}),
            # Every target of the walk is refused.
            **({"fallbacks": []} if rejected_fallback_target(request) is not None else {}),
            **self._retry_policy_raises(plan.endpoint, request),
            **(overrides or {}),
        }
        return self._handle.start(plan.endpoint, _document({"model": model, **fields})), prepared

    def _process_renamed(self, model: str, group: str) -> str:
        """``model`` as the error names it: a group the name resolved to by name, alias or routing
        group is renamed through the process ``model_alias_map`` (``_common_checks_available_deployment``)."""
        renames: Final = self._host.dynamic().model_alias_map
        resolution: Final = self._handle.resolve(model) if group in renames else None
        if resolution is None or resolution["kind"] not in ("Group", "Alias", "RoutingGroup"):
            return model
        return renames[group]

    def _max_parallel_requests_client(self, deployment: Mapping[str, object], kwargs: Mapping[str, object]) -> object:
        """``_get_client(client_type="max_parallel_requests")``: the deployment's slot limit."""
        return max_parallel_requests_limit(self, deployment, kwargs)

    def _routable(self, model: str, request: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
        return self._handle.routable(model, _document(_start_fields(request)))

    def _routing_fields(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """The request fields the Rust core routes on, and whether its tag stage may apply: the
        router's ``enable_tag_filtering``, the request's own, or a deployment's
        ``model_info.enable_tag_filtering`` (``get_deployments_for_tag``'s effective setting)."""
        filter_tags: Final = (
            self._resolved.enable_tag_filtering
            or request.get("enable_tag_filtering") is True
            or self._model_tag_filtering()
        )
        return {
            **_start_fields(request),
            **self._group_policy_fields(request),
            **({"filter_tags": True} if filter_tags else {}),
            **({"has_image": True} if request_carries_image(None, request) else {}),
        }

    def _group_policies(self, request: Mapping[str, object]) -> tuple[object, GroupPolicies]:
        """The ``model_group_retry_policy`` the retry loop reads for this request (the request's own,
        even ``None``, replaces the router's: ``kwargs.pop(..., self.model_group_retry_policy)``),
        resolved for every group name the call may look up."""
        policies: Final = request.get("model_group_retry_policy", self._resolved.model_group_retry_policy)
        return policies, group_policies(policies, self._policy_lookup_names(request))

    def _policy_lookup_names(self, request: Mapping[str, object]) -> frozenset[str]:
        """The model names a retry policy lookup may run on (each hop looks up the name it calls):
        the requested one and every name in the fallback lists the walk may take."""
        resolved: Final = self._resolved
        lists: Final = (
            resolved.fallbacks,
            resolved.context_window_fallbacks,
            resolved.content_policy_fallbacks,
            *(request.get(key) for key in _REQUEST_FALLBACK_KEYS),
        )
        return frozenset((*_strings(request.get("model")), *_strings(lists)))

    def _retry_policy_raises(self, endpoint: str, request: Mapping[str, object]) -> Mapping[str, object]:
        """The groups whose ``model_group_retry_policy`` lookup raises on a hop's first failure, for
        router-core to end that hop with the error the lookup raises (see ``_Driver.take_events``)."""
        raising: Final = raising_groups(endpoint, request, *self._group_policies(request))
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
        passthrough: "Callable[[Mapping[str, object]], Awaitable[object]] | None" = None,
        prepared_already: bool = False,
    ) -> object:
        call, prepared = self._start(plan, request, overrides, prepared_already=prepared_already)
        driver, step = await self._run(plan, call, prepared)
        if step["kind"] == "failed":
            if passthrough is not None and driver.attempted == 0:
                return await passthrough(prepared)
            raise driver.failed(step)
        return await driver.finish_async(step)

    async def _run(
        self, plan: Plan, call: "RouterCall", prepared: Mapping[str, object]
    ) -> "tuple[_Driver, Mapping[str, object]]":
        """Drive ``call`` to its ``done`` or ``failed`` step, answering every step on the host."""
        driver: Final = _Driver(self, plan, prepared)
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # Rust validates each outcome against the step it answers
            driver.take_events(step.get("events"))
            await await_all(driver.outcomes.take_pending())
            match step["kind"]:
                case "attempt":
                    outcome = await driver.attempt(step)
                case "sleep":
                    await self._host.sleep(_sleep_seconds(step))
                    outcome = None
                case "allow_fallback":
                    outcome = await driver.allow_fallback(str(step["target"]))
                case "cache":
                    outcome = await self._answer_cache(step)
                case "select":
                    outcome = await driver.select(step)
                case "pre_route":
                    outcome = await self._pre_route(step, prepared)
                case "done" | "failed":
                    return driver, step
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def _call_sync(
        self, plan: Plan, request: Mapping[str, object], overrides: Mapping[str, object] | None = None
    ) -> object:
        call, prepared = self._start(plan, request, overrides)
        driver, step = self._run_sync(plan, call, prepared)
        if step["kind"] == "failed":
            raise driver.failed(step)
        return driver.finish(step)

    def _run_sync(
        self, plan: Plan, call: "RouterCall", prepared: Mapping[str, object]
    ) -> "tuple[_Driver, Mapping[str, object]]":
        driver: Final = _Driver(self, plan, prepared)
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _run
            driver.take_events(step.get("events"))
            if pending := driver.outcomes.take_pending():
                from litellm.litellm_core_utils import asyncify

                asyncify.run_async_function(await_all, pending)  # pyright: ignore[reportUnknownMemberType]  # the untyped sync bridge
            match step["kind"]:
                case "attempt":
                    outcome = driver.attempt_sync(step)
                case "sleep":
                    self._host.sleep_sync(_sleep_seconds(step))
                    outcome = None
                case "allow_fallback":
                    from litellm.litellm_core_utils import asyncify

                    outcome = asyncify.run_async_function(driver.allow_fallback, str(step["target"]))  # pyright: ignore[reportUnknownMemberType]  # the untyped sync bridge
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "select":
                    outcome = driver.select_sync(step)
                case "pre_route":
                    outcome = self._custom_pick_sync(step, prepared)
                case "done" | "failed":
                    return driver, step
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    # -- stream recovery (the host's stream wrappers, router_rust_streaming) ---------------------

    def _recovery_call(
        self,
        recovery: str,
        op: str,
        request: Mapping[str, object],
        origin: Mapping[str, object],
        failure: BaseException,
    ) -> "RouterCall":
        start: Final = self._handle.retry_stream if recovery == "retry" else self._handle.fallback_stream
        routing: Final = _document(
            {
                "model": request.get("model"),
                **self._routing_fields(request),
                **self._retry_policy_raises(op, request),
                **continuation_routing(self, request),
            }
        )
        return start(op, routing, _document(origin), attempt_failure(failure))

    def stream_policy(
        self, request: Mapping[str, object], origin: Mapping[str, object], trigger: BaseException | None
    ) -> Mapping[str, object]:
        routing: Final = _document(
            {"model": request.get("model"), **_start_fields(request), **self._group_policy_fields(request)}
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
        driver, step = await self._run(plan_for(op), call, request)
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
        driver, step = self._run_sync(plan_for(op), call, request)
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
        driver, step = await self._run(plan_for(op), call, request)
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

    def _drive_sync_steps(self, call: "RouterCall") -> Mapping[str, object]:
        """Drive a call whose only steps are cache ops (``report_failure``) to its end."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _drive
            match step["kind"]:
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "done":
                    return step
                case "failed":
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:
                    raise RuntimeError(f"router-core asked for {other!r} while recording an outcome")

    async def _drive_cache_steps(self, call: "RouterCall") -> Mapping[str, object]:
        """``_drive_sync_steps`` answered through the async cache (``report_success``)."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _drive
            match step["kind"]:
                case "cache":
                    outcome = await self._answer_cache(step)
                case "done":
                    return step
                case "failed":
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:
                    raise RuntimeError(f"router-core asked for {other!r} while recording an outcome")

    # -- pre-route steps -------------------------------------------------------------------------

    async def _pre_route(
        self,
        step: Mapping[str, object],
        request: Mapping[str, object],
        messages: object = _NO_INPUT,
        input: object = _NO_INPUT,
        specific_deployment: object = _NO_INPUT,
    ) -> object:
        """The host's pre-routing of one pick (router_rust_prerouting): the model group to resolve
        instead of the step's ``model``, a custom strategy's deployment, or ``None``."""
        assigned: Final[object] = self.__dict__.get("_core_async_get_available_deployment")
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
        custom: Final = self._custom_strategy
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
        r: Final = self._resolved
        owner: Final[object] = self.__dict__.get("_core_owner")
        public: Final[object] = owner() if callable(owner) else None
        return await self._prerouter.pre_route(
            str(step.get("model")),
            request,  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs the hook stamps
            messages=request.get("messages") if messages is _NO_INPUT else messages,
            input=request.get("input") if input is _NO_INPUT else input,
            fallback_depth=depth if isinstance(depth, int) else 0,
            router_strategy=normalize_strategy(r.routing_strategy),
            enable_tag_filtering=r.enable_tag_filtering,
            match_any=r.tag_filtering_match_any,
            public_router=public if public is not None else self,
            run_plugins=partial(
                self._run_plugins, request, request.get("messages") if messages is _NO_INPUT else messages
            )  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs the plugins stamp
            if self._hosted._core_routing_plugins  # pyright: ignore[reportPrivateUsage]  # the live plugin list
            else None,
        )

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
        candidates: Final = [
            dict(as_str_mapping(self._handle.deployment(str(id))) or {}) for id in _as_list(step.get("candidates"))
        ]
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

    async def _answer_select(
        self, step: Mapping[str, object], request: Mapping[str, object], messages: object, input: object
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
        inputs: Final = self._strategy_inputs(step, request)
        if inputs is None:
            return step["candidates"]
        strategy, selector, model, candidates = inputs
        if selector is None:
            return []
        picked: Final = await pick_async(
            strategy,
            selector,  # pyright: ignore[reportArgumentType]  # a selector from selector_for
            model,
            candidates,
            messages,
            input,
            dict(request),
        )
        if picked is not None and self._selectors.is_override(strategy, selector):
            await selector.async_pre_call_check(dict(as_str_mapping(picked) or {}), None)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # every selector is a CustomLogger
        found: Final = picked_id(picked)
        return [found] if found is not None else []

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
        )
        if picked is not None and self._selectors.is_override(strategy, selector):
            selector.pre_call_check(dict(as_str_mapping(picked) or {}))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # see above
        found: Final = picked_id(picked)
        return [found] if found is not None else []

    # -- cache steps -----------------------------------------------------------------------------

    def _store_for(self, key: str) -> "DualCache":
        """Cooldowns live in the cooldown store; every other routing key in the routing cache."""
        return self._cooldowns.cooldown_store if key.endswith(":cooldown") else self._cache

    async def _answer_cache(self, step: Mapping[str, object]) -> object:
        with service_target(_cache_target(str(step.get("key")))):
            return await self._answer_cache_in_target(step)

    async def _answer_cache_in_target(self, step: Mapping[str, object]) -> object:
        key: Final = str(step.get("key"))
        ttl: Final = _ttl(step)
        local: Final = step.get("local") is True
        match step.get("op"):
            case "get":
                return _plain(await self._store_for(key).async_get_cache(key, local_only=local))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # DualCache's reads are untyped
            case "batch_get":
                return _plain(
                    [await self._store_for(str(k)).async_get_cache(str(k)) for k in _as_list(step.get("keys"))]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # see above
                )
            case "set":
                store: Final = self._store_for(key)
                if ttl is None:
                    await store.async_set_cache(key, step.get("value"))  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
                else:
                    await store.async_set_cache(key, step.get("value"), ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                return None
            case "increment":
                by: Final = _amount(step.get("value"))
                counter: Final = self._store_for(key)
                total: Final = (
                    await counter.async_increment_cache(key, by, local_only=local)  # pyright: ignore[reportUnknownMemberType]  # see above
                    if ttl is None
                    else await counter.async_increment_cache(key, by, local_only=local, ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                )
                return by if total is None else total
            case "delete":
                await self._store_for(key).async_delete_cache(key)
                return None
            case other:
                raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")

    def _answer_cache_sync(self, step: Mapping[str, object]) -> object:
        with service_target(_cache_target(str(step.get("key")))):
            return self._answer_cache_sync_in_target(step)

    def _answer_cache_sync_in_target(self, step: Mapping[str, object]) -> object:
        key: Final = str(step.get("key"))
        ttl: Final = _ttl(step)
        local: Final = step.get("local") is True
        match step.get("op"):
            case "get":
                return _plain(self._store_for(key).get_cache(key, local_only=local))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # DualCache's reads are untyped
            case "batch_get":
                return _plain([self._store_for(str(k)).get_cache(str(k)) for k in _as_list(step.get("keys"))])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # see above
            case "set":
                store: Final = self._store_for(key)
                if ttl is None:
                    store.set_cache(key, step.get("value"))  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
                else:
                    store.set_cache(key, step.get("value"), ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                return None
            case "increment":
                by: Final = _amount(step.get("value"))
                counter: Final = self._store_for(key)
                # DualCache.increment_cache is annotated int but adds floats as given.
                return (
                    counter.increment_cache(key, by, local_only=local)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # see above
                    if ttl is None
                    else counter.increment_cache(key, by, local_only=local, ttl=ttl)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # see above
                )
            case "delete":
                self._store_for(key).delete_cache(key)  # pyright: ignore[reportUnknownMemberType]  # DualCache's delete is untyped
                return None
            case other:
                raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")

    # -- errors the core raises when no attempt failed ------------------------------------------

    def _failure_error(
        self,
        step: Mapping[str, object],
        model: str,
        last: BaseException | None,
        *,
        pass_through: str = "",
        request: Mapping[str, object] | None = None,
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
        if kind == "ptu_reserved":
            reserved: Final = str(step.get("error_model") or model)
            return litellm_exceptions.BadRequestError(
                message=f"Deployment {reserved} is reserved for the teams holding a PTU share of it",
                model=reserved,
                llm_provider="",
            )
        if kind == "not_pass_through":
            return litellm_exceptions.BadRequestError(
                message=f"Deployment {model} does not support pass-through endpoint (use_in_pass_through=False)",
                model=model,
                llm_provider="",
            )
        if kind == "no_pass_through_deployments":
            message: Final = (
                f"Model {model} has no deployments configured with use_in_pass_through=True. "
                "Please add use_in_pass_through: true to the deployment configuration"
                if pass_through == "async_pass_through"
                else f"Model {model} has no deployment configured with use_in_pass_through=True. "
                "Please add use_in_pass_through: true in the deployment configuration"
            )
            return litellm_exceptions.BadRequestError(message=message, model=model, llm_provider="")
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
            named: Final = _alias_target(aliases.get(model)) or model if model in aliases else model
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
            cooling: Final = [id for id, _ in active]
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
            )
        if last is not None:
            return last
        return RuntimeError(f"router-core ({kind}): {step.get('error')}")


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


def _sleep_seconds(step: Mapping[str, object]) -> float:
    """A retry's sleep: ``_time_to_sleep_before_retry`` returns a plain ``0`` when another
    deployment is healthy, else a backoff in float seconds."""
    seconds: Final = float(str(step["seconds"]))
    return 0 if seconds == 0 else seconds


def _uses_pass_through(record: object) -> bool:
    params: Final = as_str_mapping((as_str_mapping(record) or {}).get("litellm_params")) or {}
    return bool(params.get("use_in_pass_through"))


def _seconds(value: object, default: float) -> float | int:
    """A cooldown time as stored: an int stays an int, so error texts read ``60`` like Python's."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return int(default) if float(default).is_integer() else default


def _ttl(step: Mapping[str, object]) -> float | None:
    ttl: Final = step.get("ttl")
    return ttl if isinstance(ttl, int | float) else None


def _amount(value: object) -> float | int:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 1


class _NotSynchronous(TypeError):
    """A sync attempt's host endpoint returned an awaitable: a host bug, never an attempt failure."""


class _Selection:
    """One ``select``: the Rust selection call, its select and cache steps answered on the host."""

    def __init__(
        self,
        core: RustCore,
        model: str,
        request: dict[str, object],  # mutable-ok: the caller's request, whose narrowing keys are popped
        kind: str,
        specific_deployment: bool | None,
        messages: object,
        input: object,
        *,
        probe: bool = False,
    ) -> None:
        self._core: Final = core
        self._model: Final = model
        self._request: Final = request
        self._kind: Final = kind
        self._messages: Final = messages
        self._input: Final = input
        fields: Final = {
            **core._routing_fields(request),  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            **_narrowing_fields(request, kind),
            **({"has_image": True} if request_carries_image(messages, request) else {}),
        }  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
        if specific_deployment:
            fields["specific_deployment"] = True
        if probe:
            fields["health_check_probe"] = True
        if kind == "pick" or (kind == "sync_pick" and core._custom_strategy is not None):  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
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
        self._call: Final = (handle.healthy if kind == "healthy" else handle.select)(model, _document(fields))

    def _done(self, step: Mapping[str, object]) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        if self._kind == "healthy":
            if step.get("single"):
                # A deployment named by its id is listed alone, as a dict.
                (record,) = _as_list(step.get("deployments"))
                return for_requested_model(as_str_mapping(record) or {}, self._model)
            # The deployments as served: a priority group's members carry their priority as order.
            return [
                for_requested_model(as_str_mapping(record) or {}, self._model)
                for record in _as_list(step.get("deployments"))
            ]
        if step.get("deployment_id") in (DEFAULT_DEPLOYMENT_FAILING_ID, DEFAULT_DEPLOYMENT_COPIED_ID):
            # A default deployment every call to fails, or whose id is not a string, is picked as Python copies it.
            picked: Final = default_deployment_copy(self._core._resolved.default_deployment, self._model)  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            return picked  # pyright: ignore[reportReturnType]  # the assigned value's copy, as the Python core returns it
        return for_requested_model(as_str_mapping(step.get("deployment")) or {}, self._model)

    def _failed(self, step: Mapping[str, object]) -> BaseException:
        core: Final = self._core
        error: Final = core._failure_error(  # pyright: ignore[reportPrivateUsage]  # see above
            step, self._model, self._filter_error, pass_through=self._kind, request=self._request
        )
        logging_obj: Final = self._request.get("litellm_logging_obj")
        dispatch: Final[object] = getattr(logging_obj, "dispatch_failure_handlers", None)
        if callable(dispatch) and self._kind not in ("pass_through", "sync_pick"):
            import asyncio
            import traceback

            # A refused pick is logged to the request's logging object (`_async_get_available_deployment`).
            trace: Final = "Traceback (most recent call last):\n" + "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            pending: Final[object] = dispatch(exception=error, traceback_exception=trace, prefer_async_handlers=True)
            if inspect.isawaitable(pending):
                _ = asyncio.ensure_future(pending)
        return error

    async def run(self) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        core: Final = self._core
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(self._call.next(outcome))  # pyright: ignore[reportArgumentType]  # Rust validates each outcome
            match step["kind"]:
                case "cache":
                    outcome = await core._answer_cache(step)  # pyright: ignore[reportPrivateUsage]  # see above
                case "select" if step.get("stage") == "callback_filters":
                    outcome, self._filter_error = await core._answer_callback_filters(  # pyright: ignore[reportPrivateUsage]  # see above
                        step, self._request, self._messages
                    )
                case "select":
                    outcome = await core._answer_select(step, self._request, self._messages, self._input)  # pyright: ignore[reportPrivateUsage]  # see above
                case "pre_route":
                    outcome = await core._pre_route(  # pyright: ignore[reportPrivateUsage]  # see above
                        step, self._request, self._messages, self._input, self._specific_deployment
                    )
                case "done":
                    done = self._done(step)
                    if self._kind != "pick" or not isinstance(done, Mapping):
                        return done
                    # `_core_async_get_available_deployment` resolves its pick to a pod.
                    return await self._resolved_pod(core, done)
                case "failed":
                    raise self._failed(step)
                case other:
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
            step = _STEP.validate_python(self._call.next(outcome))  # pyright: ignore[reportArgumentType]  # see run
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
                    raise self._failed(step)
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")


def _custom_pick(picked: object) -> Mapping[str, object]:
    """A custom strategy's answer as the deployment document the Rust core serves."""
    document: Final = as_str_mapping(picked)
    if document is None:
        raise TypeError(f"the custom routing strategy returned {type(picked).__name__}, not a deployment")
    return _document(document)


def _narrowing_fields(request: dict[str, object], kind: str) -> Mapping[str, object]:  # mutable-ok: popped
    """``_target_order``, ``_excluded_deployment_ids`` and ``_retry_skipped_deployment_ids``, popped
    from the request as the healthy chain pops them (a listed pass-through pick only reads the order)."""
    from litellm.litellm_core_utils.routing_common_utils import (
        as_excluded_deployment_ids,
        as_retry_skipped_deployment_ids,
    )

    if kind == "pass_through":
        order: Final = request.get("_target_order")
        return {"target_order": order} if isinstance(order, int) and not isinstance(order, bool) else {}
    target_order: Final = request.pop("_target_order", None)
    excluded: Final = as_excluded_deployment_ids(request.pop("_excluded_deployment_ids", None))
    skipped: Final = as_retry_skipped_deployment_ids(request.pop("_retry_skipped_deployment_ids", None))
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


class _Driver:
    """Answers the attempt and fallback steps of one call, and shapes its result.

    Each attempt is one host call against the deployment Rust picked. The driver keeps the error
    each fallback level ended on: a level that fails raises its own error, whatever its fallbacks
    raised, as the Python router does.
    """

    def __init__(self, core: RustCore, plan: Plan, request: Mapping[str, object]) -> None:
        self._core: Final = core
        self._plan: Final = plan
        self._op: Final = plan.endpoint
        self._request: Final = request
        self._last_response: object = None
        self._last_error: BaseException | None = None
        self._level_errors: Final[dict[int, BaseException]] = {}  # mutable-ok: filled as the call runs
        # The retry counts an exhausted retry loop stamps on the pick error its level then raises.
        self._pending_stamps: Final[dict[int, tuple[object, object]]] = {}  # mutable-ok: filled as the call runs
        # Each level's current hop's first failure, once it retries: the error its retry loop raises
        # last is raised while that one is handled, so it chains to it.
        self._hop_first: Final[dict[int, BaseException]] = {}  # mutable-ok: filled as the call runs
        self._served: tuple[str | None, Mapping[str, object]] = (None, MappingProxyType({}))
        self._attempts: int = 0
        self._walk: Final = WalkMetadata(request, plan.metadata_field, self._params_base_model)
        self.outcomes: Final = FallbackOutcomes(request)
        # The fallback level the walk is at, from its fallback events.
        self._level: int = 0

    @property
    def attempted(self) -> int:
        """How many attempts reached the host."""
        return self._attempts

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
        raised: Final = policy_lookup_error(original, model, self._core._resolved.retry_policy, policies)  # pyright: ignore[reportPrivateUsage]  # see above
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
                return self._picked_error(event)
            case _:
                return self._last_error

    def _picked_error(self, event: Mapping[str, object]) -> BaseException:
        """The error a pick an event names raised, as a failed step with that error raises it."""
        model: Final = str(event.get("failed_model") or event.get("model"))
        return self._core._failure_error(event, model, None, request=self._request)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core

    def _level_failed(self, event: Mapping[str, object]) -> None:
        """A level raises the error a pick raised (``level_failed``)."""
        level: Final = event.get("level")
        at_level: Final = level if isinstance(level, int) else 0
        picked: Final = self._picked_error(event)
        if event.get("handling") is True:
            # Raised by the level's fallback handling, while the level's error was handled.
            picked.__context__ = self._level_errors.get(at_level, self._last_error)
            self._level_errors[at_level] = picked
            return
        stamp: Final = self._pending_stamps.pop(at_level, None)
        if stamp is not None and type(picked) in self._core._host.dynamic().exception_types:  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            # The exhausted retry loop raised this pick's error: stamped as `retries_exhausted` stamps.
            picked.max_retries, picked.num_retries = stamp  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
        self._chain_to_hop_first(picked, at_level)
        self._level_errors[at_level] = picked

    def take_events(self, events: object) -> None:
        for raw_event in _as_list(events):
            event = as_str_mapping(raw_event)
            if event is None:
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
            if event.get("kind") == "level_failed":
                self._level_failed(event)
                continue
            if event.get("kind") == "fallback":
                hop_to = event.get("level")
                self._hop_first.pop(hop_to if isinstance(hop_to, int) else 0, None)
                self._walk.hop(event)
                hop_level = event.get("level")
                self._level = hop_level if isinstance(hop_level, int) else self._level
                continue
            if event.get("kind") == "blocked":
                level = event.get("level")
                blocked = self._blocked_error(event)
                self._last_error = blocked
                self._level_errors[level if isinstance(level, int) else 0] = blocked
                continue
            if event.get("kind") == "fallback_done":
                level = event.get("level")
                at = level if isinstance(level, int) else 0
                self._level = at
                if event.get("succeeded") is not True:
                    self._hint_nested_failure(self._level_errors.get(at + 1), str(event.get("to")))
                self.outcomes.done(
                    event,
                    self._level_errors.get(at, self._last_error),
                    self._level_errors.get(at + 1),
                    self._walk.hop_kwargs(event),
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
        deployment: Final = (
            # A default deployment whose id is not a string is called as Python copies it, id and all.
            as_str_mapping(default_deployment_copy(self._core._resolved.default_deployment, requested)) or {}  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            if step.get("deployment_id") == DEFAULT_DEPLOYMENT_COPIED_ID
            else for_requested_model(as_str_mapping(step.get("deployment")) or {}, requested)
        )
        self._attempts += 1
        level: Final = step.get("fallback_level")
        # The request params the walk's dict targets merged in (`kwargs.update(target)`).
        params: Final = as_str_mapping(step.get("params"))
        call: Final = attempt_call(
            self._core,
            self._plan,
            self._request if params is None else {**self._request, **params},
            deployment,
            group=requested,
            requested=requested,
            fallback_level=level if isinstance(level, int) else 0,
        )
        shaped: Final = shape_attempt_kwargs(call.kwargs, deployment)
        # The walk's metadata for the attempt's level (retries, fallback hops) and its hop kwargs.
        call.kwargs = self._walk.stamp(step, shaped)
        field: Final = self._plan.metadata_field
        if isinstance(call.served, dict) and field in call.kwargs:
            call.served[field] = call.kwargs[field]
        self._served = (requested, call.served)
        self._core._endpoints.counters.started(self._plan, call)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        return deployment, call

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
            slot=slot,
        )

    def _failed(
        self,
        step: Mapping[str, object],
        deployment: Mapping[str, object],
        call: "AttemptCall | None",
        error: BaseException,
    ) -> Mapping[str, object]:
        served: Final = call.served if call is not None else {}  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        stamp_error(self._plan, error, deployment, served)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
        self._core._endpoints.attempt_failed(self._plan, error, deployment, call, str(step.get("model")))  # pyright: ignore[reportPrivateUsage]  # see above
        self._last_error = error
        level: Final = step.get("fallback_level")
        retry: Final = step.get("retry")
        if isinstance(retry, int) and retry > 0:
            self._chain_to_hop_first(error, level if isinstance(level, int) else 0)
        self._level_errors[level if isinstance(level, int) else 0] = error
        return attempt_failure(error)

    def _chain_to_hop_first(self, error: BaseException, level: int) -> None:
        """A retry's failure is raised while the hop's first failure is handled (the retry loop
        runs in its ``except`` block), so it chains to it."""
        first: Final = self._hop_first.get(level)
        if first is not None and first is not error:
            error.__context__ = first  # rebind-ok: chained as the retry loop's raise chains it

    def _succeeded(self, call: "AttemptCall", response: object, host_op: str) -> Mapping[str, object]:
        self._core._endpoints.counters.succeeded(self._plan, call)  # pyright: ignore[reportPrivateUsage]  # see above
        self._last_response = response
        return {"ok": True, "total_tokens": total_tokens(response), "blocked": self._blocked(response, host_op)}

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
                report: Final = core._handle.report_failure(deployment_id, dict(attempt_failure(error)), uncounted=True)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                await core._drive_cache_steps(report)  # pyright: ignore[reportPrivateUsage]  # see above
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

    def _default_call_error(self, step: Mapping[str, object]) -> BaseException | None:
        """The error a call to a default deployment every call fails to raises (its first
        reads of the copy, before any endpoint runs): nothing records it against a deployment."""
        if step.get("deployment_id") != DEFAULT_DEPLOYMENT_FAILING_ID:
            return None
        return default_deployment_call_error(self._core._resolved.default_deployment)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core

    async def attempt(self, step: Mapping[str, object]) -> Mapping[str, object]:
        step = await self._resolve_pods(step)  # rebind-ok: the picked deployment, resolved to a pod
        deployment: Mapping[str, object] = as_str_mapping(step.get("deployment")) or {}  # rebind-ok: the picked one
        call: AttemptCall | None = None  # rebind-ok: built inside the try
        refused_call: Final = self._default_call_error(step)
        if refused_call is not None:
            return {**self._failed(step, deployment, None, refused_call), "recorded": True}
        try:
            deployment, call = self._call(step)  # rebind-ok: rebound as the Python router rebinds it
            host: Final = self._core._host  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            # The deployment's max_parallel_requests slot; an open chat stream takes it along.
            with enter_deployment_slot(self._core, self._plan, deployment, call.kwargs) as slot:
                # A search tool is no deployment: nothing to check; Python's `_arerank` and
                # `amoderation` run no routing pre-call checks either.
                if self._plan.kind not in _NO_ROUTING_PRE_CALL_CHECKS:
                    try:
                        await routing_pre_call_checks(
                            self._core._checks.routing_callbacks(),  # pyright: ignore[reportPrivateUsage]  # see above
                            as_str_mapping(step.get("deployment")) or {},
                        )
                    except Exception as refused:  # noqa: BLE001  # handed to the Rust call loop as the attempt's failure
                        return await self._refused(step, deployment, call, refused)
                pending: Final[object] = (
                    self._core._endpoints.create_file_everywhere(  # pyright: ignore[reportPrivateUsage]  # see above
                        call.kwargs, call.model_name, self._request.get("specific_deployment")
                    )
                    if self._plan.kind == "create_file"
                    else host.invoke(call.host_op)(**call.kwargs)
                )
                # The generic helper awaits whatever the handler returns, as the Python router does.
                awaited: Final = self._plan.kind == "generic" or inspect.isawaitable(pending)
                answered: Final[object] = await pending if awaited else pending  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]  # see above
                self._core._endpoints.check_response(self._plan, answered, call)  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]  # see above
                response: Final = await open_attempt_stream(self._stream_attempt(step, call.kwargs, slot), answered)  # pyright: ignore[reportUnknownArgumentType]  # see above
        except Exception as error:  # noqa: BLE001  # every attempt failure is classified and handed to the Rust call loop
            return self._failed(step, deployment, call, error)
        return self._succeeded(call, response, call.host_op)

    def attempt_sync(self, step: Mapping[str, object]) -> Mapping[str, object]:
        step = self._resolve_pods_sync(step)  # rebind-ok: see attempt
        deployment: Mapping[str, object] = as_str_mapping(step.get("deployment")) or {}  # rebind-ok: see attempt
        call: AttemptCall | None = None  # rebind-ok: see attempt
        refused_call: Final = self._default_call_error(step)
        if refused_call is not None:
            return {**self._failed(step, deployment, None, refused_call), "recorded": True}
        try:
            deployment, call = self._call(step)  # rebind-ok: rebound as the Python router rebinds it
            try:
                self._core._endpoints.sync_pre_call_checks(self._plan, deployment, self._request)  # pyright: ignore[reportPrivateUsage]  # see above
            except Exception as refused:  # noqa: BLE001  # handed to the Rust call loop like any attempt failure
                # The sync path records no outcome for a refused pre-call check: no cooldown.
                return {**self._failed(step, deployment, call, refused), "recorded": True}
            answered: Final = self._core._host.invoke(call.host_op)(**call.kwargs)  # pyright: ignore[reportPrivateUsage]  # a sync endpoint
            if inspect.isawaitable(answered):
                raise _NotSynchronous(f"the host's {call.host_op} endpoint is not synchronous")
            self._core._endpoints.check_response(self._plan, answered, call)  # pyright: ignore[reportPrivateUsage]  # see above
            response: Final = open_attempt_stream_sync(self._stream_attempt(step, call.kwargs), answered)
        except _NotSynchronous:
            raise
        except Exception as error:  # noqa: BLE001  # see attempt
            return self._failed(step, deployment, call, error)
        return self._succeeded(call, response, "")

    async def select(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        if step.get("stage") == "callback_filters":
            answer, error = await self._core._answer_callback_filters(step, request, request.get("messages"))  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            if error is not None:
                # The try failed with the filter's error: the level raises it if nothing else serves.
                self._last_error = error
                self._level_errors[self._level] = error
            return answer
        return await self._core._answer_select(step, request, request.get("messages"), request.get("input"))  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core

    def select_sync(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        return self._core._answer_select_sync(step, request, request.get("messages"), request.get("input"))  # pyright: ignore[reportPrivateUsage]  # see above

    async def allow_fallback(self, target: str) -> bool:
        """`_is_fallback_target_authorized`, then `_is_fallback_target_within_budget`."""
        if target == self._request.get("model"):
            return True
        from litellm.types.router import FallbackSkipReason

        ctx: Final = RoutingContext(self._core)  # pyright: ignore[reportArgumentType]  # the checks read registry() and settings()
        request: Final = dict(self._request)
        if not await ctx.fallback_access_allowed(model=target, request_kwargs=request):
            self._walk.targets().record_skip(target, FallbackSkipReason.NOT_AUTHORIZED)
            return False
        if not await ctx.fallback_budget_allowed(model=target, request_kwargs=request):
            self._walk.targets().record_skip(target, FallbackSkipReason.OVER_BUDGET)
            return False
        return True

    def finish(self, step: Mapping[str, object]) -> object:
        """The response with the router's headers: the deployment id, the model group it was
        served as, and the retry and fallback counts (``set_response_headers``)."""
        if not self._plan.sets_headers:
            return self._last_response
        response: object = prepare_response_for_header_attachment(self._last_response)  # rebind-ok: gains headers
        if response is None:
            return self._last_response
        name, kwargs = self._served
        additional: Final = ensure_response_additional_headers(response)
        apply_response_model_id(response, find_deployment_metadata(kwargs))
        additional["x-litellm-model-group"] = name
        retries: Final = step.get("attempted_retries")
        if isinstance(retries, int):
            max_retries: Final = step.get("max_retries")
            response = add_retry_headers_to_response(  # rebind-ok: see above
                response=response,
                attempted_retries=retries,
                max_retries=max_retries if isinstance(max_retries, int) and retries > 0 else None,
            )
        fallbacks: Final = step.get("attempted_fallbacks")
        attempted: Final = fallbacks if isinstance(fallbacks, int) else 0
        return add_fallback_headers_to_response(
            response=self.outcomes.add_headers(response, attempted), attempted_fallbacks=attempted
        )

    async def finish_async(self, step: Mapping[str, object]) -> object:
        response: Final = self.finish(step)
        name: Final = self._served[0]
        if name is not None and response is not None and self._plan.sets_headers:
            remaining: Final = await self._core.health().usage(name, kind="remaining")
            if isinstance(remaining, Mapping):
                apply_remaining_usage_headers(ensure_response_additional_headers(response), dict(remaining))
        return response

    def failed(self, step: Mapping[str, object]) -> BaseException:
        model: Final = str(self._request.get("model"))
        primary: Final = self._level_errors.get(0)
        if primary is None:
            return self._core._failure_error(step, model, self._last_error, request=self._request)  # pyright: ignore[reportPrivateUsage]  # see above
        if self._core._host.dynamic().expose_router_debug_in_errors and hasattr(primary, "message"):  # pyright: ignore[reportPrivateUsage]  # see above
            primary.message += self._explanation(primary, model)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
        return primary

    def _hint_nested_failure(self, error: BaseException | None, model: str) -> None:
        """A fallback level that fails on a context-window or content-policy error with no list
        of its own appends the hint to set one, as every level of the Python walk does."""
        from litellm.router_utils.fallback_event_handlers import check_non_standard_fallback_format

        if (
            error is None
            or not hasattr(error, "message")
            or not self._core._host.dynamic().expose_router_debug_in_errors  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        ):
            return
        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # see above
        fallbacks: Final = self._request.get("fallbacks", resolved.fallbacks)
        if check_non_standard_fallback_format(fallbacks=_as_list(fallbacks)):
            return
        hint: Final = self._kind_hint(error, model, fallbacks)
        if hint:
            error.message += hint  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the router appends to the error it raises

    def _kind_hint(self, error: BaseException, model: str, fallbacks: object) -> str:
        """The hint a context-window or content-policy error gets when its own list is unset."""
        import litellm.exceptions as litellm_exceptions

        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        if isinstance(error, litellm_exceptions.ContextWindowExceededError):
            if self._request.get("context_window_fallbacks", resolved.context_window_fallbacks) is not None:
                return ""
            return (
                f"\nmodel={model}. context_window_fallbacks=None. fallbacks={fallbacks}.\n\n"
                "Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
            )
        if isinstance(error, litellm_exceptions.ContentPolicyViolationError):
            if self._request.get("content_policy_fallbacks", resolved.content_policy_fallbacks) is not None:
                return ""
            return "\n" + format_content_policy_fallback_hint(model, None, fallbacks)  # pyright: ignore[reportArgumentType]  # the list as configured
        return ""

    def _explanation(self, primary: BaseException, model: str) -> str:
        """What ``_async_function_with_fallbacks_common_utils`` appends to the error it re-raises
        under ``expose_router_debug_in_errors``, from the resolved settings and what the call did."""
        import litellm.exceptions as litellm_exceptions
        from litellm.router_utils.fallback_event_handlers import (
            check_non_standard_fallback_format,
            get_fallback_model_group_for_lookup_groups,
        )

        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        request: Final = self._request
        # The request's own lists replace the router's (`kwargs.get("fallbacks", router)`).
        fallbacks: Final = request.get("fallbacks", resolved.fallbacks)
        window_list: Final = request.get("context_window_fallbacks", resolved.context_window_fallbacks)
        policy_list: Final = request.get("content_policy_fallbacks", resolved.content_policy_fallbacks)
        lookup: Final = (model,)
        if check_non_standard_fallback_format(fallbacks=_as_list(fallbacks)):
            chain_list: Final = _as_list(fallbacks)
            rejected: Final = rejected_fallback_target(request)
            detail: Final = (
                truncate_fallback_error_detail(redact_string(str(rejected)))
                if rejected is not None
                else self._fallback_detail(chain_list)
            )
            return format_fallback_outcome_message(model, chain_list, detail, skipped=self._walk.targets().skipped)
        context_window: Final = isinstance(primary, litellm_exceptions.ContextWindowExceededError)
        content_policy: Final = isinstance(primary, litellm_exceptions.ContentPolicyViolationError)
        # The error kind's own list when it is set, else the generic one with a hint to set it.
        own: Final = window_list if context_window else policy_list if content_policy else None
        hint: Final = (
            f"\nmodel={model}. context_window_fallbacks=None. fallbacks={fallbacks}.\n\n"
            "Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
            if context_window and own is None
            else "\n" + format_content_policy_fallback_hint(model, None, fallbacks)  # pyright: ignore[reportArgumentType]  # the list as configured
            if content_policy and own is None
            else ""
        )
        source: Final = own if own is not None else fallbacks
        entries: Final = [dict(m) for entry in _as_list(source) if (m := as_str_mapping(entry))]
        found, generic = (
            get_fallback_model_group_for_lookup_groups(entries, lookup)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape
            if entries
            else (None, None)
        )
        chain: Final[Sequence[object] | None] = (
            found if found is not None else _as_list(entries[generic].get("*")) if generic is not None else None
        )
        if chain is None and entries and own is None:
            return hint + format_no_fallback_group_message(lookup, entries)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape
        if chain is None:
            return hint + format_fallback_outcome_message(model, None, "")
        return hint + format_fallback_outcome_message(
            model, chain, self._fallback_detail(chain), skipped=self._walk.targets().skipped
        )

    def _fallback_detail(self, chain: Sequence[object]) -> str:
        """The error the fallback walk ended on: the first fallback level's last error, or the
        unknown-model error of a target with no deployment, which fails before any attempt."""
        last: Final = str(chain[-1]) if chain else None
        attempted: Final = self._level_errors.get(1)
        unknown: Final = last is not None and self._core._handle.resolve(last) is None  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        error: Final = (
            attempted
            if attempted is not None or not unknown or last is None
            else self._core._failure_error({"error_kind": "unknown_model"}, last, None)  # pyright: ignore[reportPrivateUsage]  # see above
        )
        return truncate_fallback_error_detail(redact_string(str(error))) if error is not None else ""


__all__ = [
    "RUST_SERVED_OPS",
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
