# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
# (the ported core code reads the untyped strategy routers, LiteLLM_Params dict fields and request dicts)
"""
Host-side pre-routing for ``RustCore``.

Before an async pick resolves its name, the Rust core asks the host to pre-route it (a
``pre_route`` step, ``router-core/src/prerouting.rs``): the complexity, auto, adaptive and quality
routers registered from ``auto_router/...`` marker deployments, and the Claude Code session
binding, may send the request to another model group. They are live Python objects, so they stay
host-side for now (bridged, to port): this module registers them from the host registry's
deployments the way the Python router's ``init_*_router_deployment``,
``_finalize_adaptive_router_if_configured`` and ``_unregister_pre_routing_strategy_for_deployment``
do, and answers the step the way ``_core_async_pre_routing_hook`` and ``_pin_tier_params_onto_request``
do. It never reaches a Python router core.
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast  # noqa: TID251  # the strategy routers are untyped

from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache, InMemoryCache
from litellm.caching.redis_cache import log_redis_failure
from litellm.constants import (
    CLIENT_OUTPUT_CEILING_METADATA_KEY,
    CONSUMED_REQUEST_TAGS_METADATA_KEY,
    DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS,
    OUTPUT_TOKEN_CEILING_PARAMS,
    SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY,
)
from litellm.integrations.otel.routing import routing_decision_attributes
from litellm.integrations.otel.runtime import phase_attributes
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.litellm_core_utils.credential_accessor import reject_api_key_secret_reference
from litellm.router_api import deployment_id
from litellm.router_host import router_host
from litellm.router_strategy.auto_router.auto_router_model_naming import (
    AUTO_ROUTER_MODEL_PREFIX,
    STRATEGY_ROUTER_PARAM_FIELDS,
    GatedAutoRouterCapability,
    capability_limit_violation,
    claimed_capabilities,
    classify_strategy_router_model,
    count_capability_routers,
)
from litellm.router_strategy.complexity_router.context_compaction import (
    arm_compaction,
    is_native_compaction_call,
    reject_recursive_compactor,
)
from litellm.router_strategy.tag_based_routing import get_routing_tags_from_request_kwargs, is_valid_deployment_tag
from litellm.router_utils.fallback_event_handlers import record_pre_routing_selection
from litellm.types.router import ConsumedRequestTagsStamp, Deployment, LiteLLM_Params, TaggedPreRoutingStrategy
from litellm.types.utils import (
    PROMPT_QUOTING_ROUTING_DECISION_FIELDS,
    CustomPricingLiteLLMParams,
    StandardLoggingRoutingDecision,
    all_litellm_params,
)
from litellm.utils import provider_rejectable_params

if TYPE_CHECKING:
    from litellm.router_python_host import PythonHost
    from litellm.router_rust_registry import HostRegistry
    from litellm.router_strategy.auto_router.auto_router import PreRoutingHookResponse

# The live registry dicts: model_name -> the strategies registered under it, one per tag set.
StrategyRegistry: TypeAlias = "dict[str, list[TaggedPreRoutingStrategy[Any]]]"

_ALIAS_PARAMS_NEVER_FORWARDED: Final = frozenset(
    {"model", "api_base", "api_key", "api_version", "responses_api_version"}
)
_ALIAS_MARKER_FORWARDED_PARAMS_KWARG: Final = "_alias_marker_forwarded_params"
_CLAUDE_CODE_SESSION_ID_RE: Final = re.compile(r"^[a-zA-Z0-9_\-]{8,}$")
_CLAUDE_CODE_SESSION_ROUTER_TTL_SECONDS: Final = 3600
CLAUDE_CODE_SESSION_ROUTER_BINDING_TARGET: Final = "claude_code_session_router_binding"
# `_async_get_available_deployment` runs the pre-routing hook only for the strategies with an async
# pick; any other router strategy takes the sync path, which never pre-routes.
_ASYNC_PICK_STRATEGIES: Final = frozenset(
    {"usage-based-routing-v2", "simple-shuffle", "cost-based-routing", "latency-based-routing", "least-busy"}
)
TIER_PARAMS_NEVER_DROPPED: Final = frozenset(all_litellm_params) | frozenset(
    {
        "additional_drop_params",
        "drop_params",
        "messages",
        "model",
        "extra_headers",
        "max_tokens",
        "max_completion_tokens",
    }
)


def _json_default_stable_id(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _deployment_tags(deployment: Deployment) -> tuple[str, ...]:
    return tuple(deployment.litellm_params.tags or ())


def _has_registered_strategy(registry: StrategyRegistry, model_name: str, tags: tuple[str, ...]) -> bool:
    return any(existing.tags == tags for existing in registry.get(model_name, []))


def _routing_definition_fingerprint(deployment: Deployment) -> str | None:
    """Identify configured routing fields, excluding external files and live strategy state."""
    try:
        definition: Final = json.dumps(
            deployment.litellm_params.model_dump(
                include=set(STRATEGY_ROUTER_PARAM_FIELDS | {"model", "tags"}), exclude_none=True, warnings=False
            ),
            sort_keys=True,
            default=_json_default_stable_id,
        )
    except (TypeError, ValueError):
        return None
    return deployment_id(deployment.model_name, {"routing_definition": definition})


def _unregister(registry: StrategyRegistry, model_name: str, tags: tuple[str, ...]) -> bool:
    existing: Final = registry.get(model_name, [])
    remaining: Final = [entry for entry in existing if entry.tags != tags]
    if len(remaining) == len(existing):
        return False
    if remaining:
        registry[model_name] = remaining  # rebind-ok: the live registry, edited in place as the Python router does
    else:
        registry.pop(model_name, None)
    return True


def is_strategy_marker(deployment: Mapping[str, object]) -> bool:
    params: Final = deployment.get("litellm_params")
    if not isinstance(params, Mapping):
        return False
    model: Final = params.get("model")
    return isinstance(model, str) and classify_strategy_router_model(model) is not None


def _participates_in_adaptive_routing(params: LiteLLM_Params) -> bool:
    kind: Final = classify_strategy_router_model(params.model)
    if kind == "adaptive":
        return True
    if kind != "complexity":
        return False
    config: Final = params.complexity_router_config
    return bool(config.get("adaptive")) if config else False


class PreRoutingStrategies:
    """The pre-routing strategies of one router, kept on its host registry (``registry().plugins``
    reads ``_core_auto_routers`` and its siblings there)."""

    def __init__(
        self, hosted: "HostRegistry", host: "PythonHost", ctx: object, capability_limit: object, cache: DualCache
    ) -> None:
        self._hosted: Final = hosted
        self._host: Final = host
        self._ctx: Final = ctx
        self._capability_limit: object = capability_limit
        self._cache: Final = cache
        self._affinity: object = None
        # The router's own `_ensure_deployment_affinity_callback` (its optional checks), when set.
        self.ensure_affinity: Callable[[], None] | None = None

    # -- the registries ------------------------------------------------------------------------

    @property
    def auto(self) -> StrategyRegistry:
        return self._hosted._core_auto_routers  # pyright: ignore[reportPrivateUsage]  # the registries live on the host registry

    @property
    def complexity(self) -> StrategyRegistry:
        return self._hosted._core_complexity_routers  # pyright: ignore[reportPrivateUsage]  # see above

    @property
    def adaptive(self) -> StrategyRegistry:
        return self._hosted._core_adaptive_routers  # pyright: ignore[reportPrivateUsage]  # see above

    @property
    def quality(self) -> StrategyRegistry:
        return self._hosted._core_quality_routers  # pyright: ignore[reportPrivateUsage]  # see above

    def all(self) -> tuple[StrategyRegistry, ...]:
        """In the order ``_select_pre_routing_strategy`` searches them."""
        return (self.auto, self.complexity, self.adaptive, self.quality)

    def any_registered(self) -> bool:
        return any(self.all())

    # -- registry hooks (HostRegistry calls these) -----------------------------------------------

    def reset(self) -> None:
        """``set_model_list`` drops the auto, complexity and quality routers (never the adaptive ones)."""
        self._hosted._core_quality_routers = {}  # pyright: ignore[reportPrivateUsage]  # see above
        self._hosted._core_complexity_routers = {}  # pyright: ignore[reportPrivateUsage]  # see above
        self._hosted._core_auto_routers = {}  # pyright: ignore[reportPrivateUsage]  # see above

    def register(self, deployment: Deployment) -> None:
        """The end of ``_add_deployment``: an auto, complexity or quality marker registers its router
        (adaptive ones wait for ``finalize``)."""
        match classify_strategy_router_model(deployment.litellm_params.model):
            case "semantic":
                self._init_auto_router(deployment)
            case "complexity":
                self._init_complexity_router(deployment)
            case "quality":
                self._init_quality_router(deployment)
            case _:
                pass

    def release(self, deployment: Deployment) -> None:
        """``_unregister_pre_routing_strategy_for_deployment``."""
        if not deployment.litellm_params.model.startswith("auto_router/"):
            return
        model_name: Final = deployment.model_name
        tags: Final = _deployment_tags(deployment)
        for registry in (self.auto, self.complexity, self.quality):
            _ = _unregister(registry, model_name, tags)
        if _unregister(self.adaptive, model_name, tags):
            self._sync_adaptive_router_hooks()

    def release_entry(self, entry: Mapping[str, object]) -> None:
        """``release`` of a deleted model_list entry; a failure only logs, as the delete has happened."""
        try:
            self.release(Deployment(**entry))  # pyright: ignore[reportArgumentType]  # a model_list entry
        except Exception:  # noqa: BLE001  # the delete already happened; a failed release only logs
            verbose_router_logger.exception(
                "delete_deployment: could not release pre-routing strategies for model_id=%s; "
                "the deployment is out of the model_list and its indices are repaired",
                as_dict(entry.get("model_info")).get("id"),
            )

    def upserted(self, deployment: Deployment, previous: Deployment | None) -> None:
        """After an upsert's re-add: rebuild the adaptive routers when it touched one."""
        if _participates_in_adaptive_routing(deployment.litellm_params) or (
            previous is not None and _participates_in_adaptive_routing(previous.litellm_params)
        ):
            self.finalize()

    def without_capability_limit(self, action: Callable[[], object]) -> None:
        """Run ``action`` without judging the capability ceiling: a rollback re-admits a serving router."""
        resolver: Final = self._capability_limit
        self._capability_limit = None
        try:
            _ = action()
        finally:
            self._capability_limit = resolver

    def finalize(self) -> None:
        """``_finalize_adaptive_router_if_configured``: an adaptive router for every adaptive marker,
        and the adaptive companion of every complexity router configured with ``adaptive``."""
        for entry in self._hosted._core_model_list or []:  # pyright: ignore[reportPrivateUsage]  # the host registry's live list
            params = entry.get("litellm_params")
            lp_model = params.get("model") if isinstance(params, dict) else None
            if not (isinstance(lp_model, str) and lp_model.startswith("auto_router/adaptive_router")):
                continue
            model_name = entry.get("model_name")
            if not model_name or not params:
                continue
            deployment = Deployment(
                model_name=str(model_name),
                litellm_params=LiteLLM_Params.model_validate(params),
                model_info=entry.get("model_info"),  # pyright: ignore[reportArgumentType]  # the entry's model_info
            )
            if _has_registered_strategy(self.adaptive, str(model_name), _deployment_tags(deployment)):
                continue
            self._init_adaptive_router(deployment)
        for model_name, tagged_complexity_routers in self.complexity.items():
            for tagged in tagged_complexity_routers:
                complexity_router = tagged.strategy
                if not complexity_router.config.adaptive:
                    continue
                if _has_registered_strategy(self.adaptive, model_name, tagged.tags):
                    continue
                adaptive_router = complexity_router._ensure_adaptive_router()
                if adaptive_router is not None:
                    self.adaptive[model_name] = [
                        *self.adaptive.get(model_name, []),
                        TaggedPreRoutingStrategy(
                            tags=tagged.tags,
                            strategy=adaptive_router,
                            definition_fingerprint=tagged.definition_fingerprint,
                            deployment=tagged.deployment,
                        ),
                    ]
        self._sync_adaptive_router_hooks()

    def retire(self) -> None:
        """Release the adaptive routers' post-call hooks this router registered."""
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook

        own: Final = frozenset(id(tagged.strategy) for tagged in chain.from_iterable(self.adaptive.values()))
        hooks: Final[list[object]] = [  # mutable-ok: built here
            hook
            for hook in self._host.callbacks.get_custom_loggers_for_type(AdaptiveRouterPostCallHook)
            if isinstance(hook, AdaptiveRouterPostCallHook) and id(hook.adaptive_router) in own
        ]
        if self._affinity is not None:
            hooks.append(self._affinity)
            self._affinity = None
        if hooks:
            self._host.callbacks.drop_callbacks(hooks)

    # -- building --------------------------------------------------------------------------------

    def _register(
        self, registry: StrategyRegistry, deployment: Deployment, strategy: object, strategy_label: str
    ) -> None:
        tags: Final = _deployment_tags(deployment)
        if _has_registered_strategy(registry, deployment.model_name, tags):
            raise ValueError(
                f"{strategy_label} deployment {deployment.model_name} with tags {list(tags)} already exists. "
                "Please use a different model name or set different tags."
            )
        registry[deployment.model_name] = [  # rebind-ok: the live registry, edited in place as the Python router does
            *registry.get(deployment.model_name, []),
            TaggedPreRoutingStrategy(
                tags=tags,
                strategy=strategy,
                definition_fingerprint=_routing_definition_fingerprint(deployment),
                deployment=deployment,
            ),
        ]

    def _capability_violation(self, capability: GatedAutoRouterCapability) -> str | None:
        resolver: Final = self._capability_limit
        limit: Final = resolver() if callable(resolver) else None
        others: Final = count_capability_routers(
            (entry for entry in self._hosted._core_model_list if isinstance(entry, Mapping)),  # pyright: ignore[reportPrivateUsage, reportUnnecessaryIsInstance]  # the live list
            capability=capability,
        )
        return capability_limit_violation(capability=capability, held=others + 1, limit=limit)  # pyright: ignore[reportArgumentType]  # the resolver's limit

    def _init_auto_router(self, deployment: Deployment) -> None:
        from litellm.router_strategy.auto_router.auto_router import AutoRouter

        params: Final = deployment.litellm_params
        if params.auto_router_config_path is None and params.auto_router_config is None:
            raise ValueError(
                "auto_router_config_path or auto_router_config is required for auto-router deployments. Please set it in the litellm_params"
            )
        if params.auto_router_default_model is None:
            raise ValueError(
                "auto_router_default_model is required for auto-router deployments. Please set it in the litellm_params"
            )
        if params.auto_router_embedding_model is None:
            raise ValueError(
                "auto_router_embedding_model is required for auto-router deployments. Please set it in the litellm_params"
            )
        max_input_chars: Final = params.auto_router_max_input_chars
        router: Final = AutoRouter(
            model_name=deployment.model_name,
            auto_router_config_path=params.auto_router_config_path,
            auto_router_config=params.auto_router_config,
            default_model=params.auto_router_default_model,
            embedding_model=params.auto_router_embedding_model,
            ctx=self._ctx,  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
            max_input_chars=max_input_chars if max_input_chars is not None else DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS,
        )
        self._register(self.auto, deployment, router, "Auto-router")

    def _init_complexity_router(self, deployment: Deployment) -> None:
        from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
        from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig

        config: Final = deployment.litellm_params.complexity_router_config
        for capability in claimed_capabilities(config):
            if (violation := self._capability_violation(capability)) is not None:
                raise ValueError(violation)
        default_model: Final = (
            ComplexityRouterConfig.model_validate(config).resolve_default_model(
                deployment.litellm_params.complexity_router_default_model
            )
            if config
            else deployment.litellm_params.complexity_router_default_model
        )
        if default_model is None:
            raise ValueError(
                "complexity_router_default_model is required for complexity-router deployments, "
                "or configure tiers in complexity_router_config. Please set it in the litellm_params"
            )
        router: Final = ComplexityRouter(
            model_name=deployment.model_name,
            default_model=default_model,
            ctx=self._ctx,  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
            complexity_router_config=config,
        )
        self._register(self.complexity, deployment, router, "Complexity-router")
        if router._uses_deployment_pin:  # pyright: ignore[reportPrivateUsage]  # the Python router reads the same flag
            self._ensure_deployment_affinity_callback()

    def _ensure_deployment_affinity_callback(self) -> None:
        """``_ensure_deployment_affinity_callback``: the affinity check (every global flag off) a
        complexity router's session pin needs, registered once."""
        from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck

        if self.ensure_affinity is not None:
            self.ensure_affinity()
            return
        if self._affinity is not None:
            return
        hosted: Final = self._hosted

        def is_priority_group(model: str) -> bool:
            resolved: Final = hosted._core_get_model_from_alias(model=model) or model  # pyright: ignore[reportPrivateUsage]  # the host registry's alias lookup
            group: Final = hosted.get_routing_group(resolved)
            return group is not None and group.routing_strategy == "priority"

        affinity: Final = DeploymentAffinityCheck(
            cache=self._cache,
            ttl_seconds=3600,
            enable_user_key_affinity=False,
            enable_responses_api_affinity=False,
            enable_session_id_affinity=False,
            model_group_affinity_config=None,
            is_priority_group=is_priority_group,
            session_affinity_repin_on_failover=False,
        )
        self._affinity = affinity
        self._host.callbacks.add_litellm_callback(affinity)

    def _init_quality_router(self, deployment: Deployment) -> None:
        from litellm.router_strategy.quality_router.quality_router import QualityRouter

        config: Final = deployment.litellm_params.quality_router_config
        configured: Final = deployment.litellm_params.quality_router_default_model
        default_model: Final = configured if configured is not None or not config else config.get("default_model")
        if default_model is None:
            raise ValueError(
                "quality_router_default_model is required for quality-router deployments, "
                "or set default_model in quality_router_config. Please configure it in the litellm_params"
            )
        router: Final = QualityRouter(
            model_name=deployment.model_name,
            default_model=default_model,
            ctx=self._ctx,  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
            quality_router_config=config,
        )
        self._register(self.quality, deployment, router, "Quality-router")

    def _init_adaptive_router(self, deployment: Deployment) -> None:
        from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook
        from litellm.types.router import AdaptiveRouterConfig, AdaptiveRouterPreferences

        raw_config: Final = deployment.litellm_params.adaptive_router_config
        if raw_config is None:
            raise ValueError("adaptive_router_config is required for adaptive-router deployments.")
        config: Final = AdaptiveRouterConfig(**raw_config)
        model_to_prefs: Final[dict[str, AdaptiveRouterPreferences]] = {}  # mutable-ok: built here
        model_to_cost: Final[dict[str, float]] = {}  # mutable-ok: built here
        for name in config.available_models:
            deployments = self._hosted._core_deployments_named(name)  # pyright: ignore[reportPrivateUsage]  # the host registry's name index
            if not deployments:
                continue
            first = deployments[0]
            model_info = first.get("model_info")
            info: Mapping[str, object] = model_info if isinstance(model_info, dict) else {}
            prefs_raw = info.get("adaptive_router_preferences")
            if isinstance(prefs_raw, dict):
                model_to_prefs[name] = AdaptiveRouterPreferences(**prefs_raw)
            cost = _adaptive_cost(as_dict(first.get("litellm_params")), info)
            if isinstance(cost, int | float | str):
                model_to_cost[name] = float(cost)
        router: Final = AdaptiveRouter(
            router_name=deployment.model_name,
            config=config,
            model_to_prefs=model_to_prefs,
            model_to_cost=model_to_cost,
        )
        self._register(self.adaptive, deployment, router, "Adaptive-router")
        self._host.callbacks.add_litellm_callback(AdaptiveRouterPostCallHook(adaptive_router=router))
        verbose_router_logger.info(
            "AdaptiveRouter[%s] initialized with %d models", deployment.model_name, len(config.available_models)
        )

    def _sync_adaptive_router_hooks(self) -> None:
        """One post-call hook per registered adaptive router, plus the header hook while any exists."""
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook
        from litellm.router_utils.global_callback_holders import (
            adaptive_routers_on_host,
            sync_adaptive_response_header_hook,
        )

        callbacks: Final = self._host.callbacks
        for callback in callbacks.get_custom_loggers_for_type(AdaptiveRouterPostCallHook):
            callbacks.remove_callback_from_all_lists(callback)
        for tagged in chain.from_iterable(self.adaptive.values()):
            callbacks.add_litellm_callback(AdaptiveRouterPostCallHook(adaptive_router=tagged.strategy))
        sync_adaptive_response_header_hook(callbacks, bool(self.adaptive) or adaptive_routers_on_host(self._host))


# -- the request side --------------------------------------------------------------------------


def _stamp_or_clear_metadata_key(
    request_kwargs: dict[str, object],  # mutable-ok: the request kwargs, stamped in place
    key: str,
    value: object | None,
) -> None:
    """Write a proxy-internal metadata key for this routing attempt, or clear it from both buckets."""
    if value is None:
        for bucket in (request_kwargs.get("metadata"), request_kwargs.get("litellm_metadata")):
            if isinstance(bucket, dict):
                bucket.pop(key, None)
        return
    _, metadata_bucket = get_or_create_metadata_bucket(request_kwargs)
    metadata_bucket[key] = value


def _redact_prompt_text_if_needed(
    request_kwargs: Mapping[str, object], routing_decision: StandardLoggingRoutingDecision
) -> StandardLoggingRoutingDecision:
    from litellm.litellm_core_utils.redact_messages import should_redact_message_logging

    if not should_redact_message_logging(
        {
            "litellm_params": request_kwargs,
            "standard_callback_dynamic_params": request_kwargs.get("standard_callback_dynamic_params"),
        }
    ):
        return routing_decision
    kept: Final = {
        field: value for field, value in routing_decision.items() if field not in PROMPT_QUOTING_ROUTING_DECISION_FIELDS
    }
    return cast(StandardLoggingRoutingDecision, kept)  # cast-ok: dropping optional keys preserves the type


def _record_routing_decision(
    request_kwargs: dict[str, object],  # mutable-ok: the request kwargs
    routing_decision: StandardLoggingRoutingDecision | None,
) -> None:
    """Make the request's metadata describe this routing attempt, and only this one."""
    from litellm.router_strategy.auto_router.baseline_request import capture_baseline_parameters
    from litellm.types.router import BaselineRouteStamp

    phase_attributes(routing_decision_attributes(routing_decision))
    baseline_model: Final = routing_decision.get("savings_baseline_model") if routing_decision else None
    baseline_id: Final = routing_decision.get("savings_baseline_deployment_id") if routing_decision else None
    router_name: Final = routing_decision.get("router_model_name") if routing_decision else None
    caller_parameters: Final = capture_baseline_parameters(request_kwargs) if router_name and baseline_model else None
    _stamp_or_clear_metadata_key(
        request_kwargs,
        "_autorouter_baseline_route",
        BaselineRouteStamp(router_name, baseline_model, baseline_id, caller_parameters)
        if router_name and baseline_model and baseline_id
        else None,
    )
    _stamp_or_clear_metadata_key(
        request_kwargs,
        "routing_decision",
        None if routing_decision is None else _redact_prompt_text_if_needed(request_kwargs, routing_decision),
    )


def _as_output_cap(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        cap: Final = int(float(value))
    except (ValueError, OverflowError):
        return None
    return cap if cap >= 0 else None


def _stamped_client_ceiling(request_kwargs: Mapping[str, object]) -> Mapping[str, int] | None:
    stamped: Final = next(
        (
            bucket.get(CLIENT_OUTPUT_CEILING_METADATA_KEY)
            for bucket in (request_kwargs.get("metadata"), request_kwargs.get("litellm_metadata"))
            if isinstance(bucket, dict) and CLIENT_OUTPUT_CEILING_METADATA_KEY in bucket
        ),
        None,
    )
    if not isinstance(stamped, dict):
        return None
    return MappingProxyType(
        {
            str(carrier): cap
            for carrier, value in stamped.items()
            if carrier in OUTPUT_TOKEN_CEILING_PARAMS and (cap := _as_output_cap(value)) is not None
        }
    )


def _restore_client_output_ceiling(
    request_kwargs: MutableMapping[str, object],  # mutable-ok: the request kwargs, restored in place
) -> None:
    callers_ceiling: Final = _stamped_client_ceiling(request_kwargs)
    if callers_ceiling is None:
        return
    for carrier in OUTPUT_TOKEN_CEILING_PARAMS:
        request_kwargs.pop(carrier, None)
    request_kwargs.update(callers_ceiling)


def _stamp_client_output_ceiling(request_kwargs: dict[str, object]) -> None:  # mutable-ok: the request kwargs
    _, metadata_bucket = get_or_create_metadata_bucket(request_kwargs)
    metadata_bucket.setdefault(
        CLIENT_OUTPUT_CEILING_METADATA_KEY,
        {carrier: request_kwargs[carrier] for carrier in OUTPUT_TOKEN_CEILING_PARAMS if carrier in request_kwargs},
    )


def _pop_effort_from_nested_carrier(request_kwargs: dict[str, object], carrier: str) -> None:  # mutable-ok: same
    nested: Final = request_kwargs.get(carrier)
    if not isinstance(nested, Mapping):
        return
    sanitized: Final = {key: value for key, value in nested.items() if key != "effort"}
    if sanitized:
        request_kwargs[carrier] = sanitized  # rebind-ok: copy-on-write of the request's nested carrier
    else:
        request_kwargs.pop(carrier, None)


def _tier_ceiling_under_the_surface_name(
    tier_litellm_params: Mapping[str, object], responses_call: bool
) -> Mapping[str, object]:
    surface_key: Final = "max_output_tokens" if responses_call else "max_tokens"
    carried: Final = tuple(
        key
        for key in (surface_key, "max_tokens", "max_completion_tokens", "max_output_tokens")
        if key in tier_litellm_params
    )
    if not carried:
        return tier_litellm_params
    return MappingProxyType(
        {
            **{k: v for k, v in tier_litellm_params.items() if k not in OUTPUT_TOKEN_CEILING_PARAMS},
            surface_key: tier_litellm_params[carried[0]],
        }
    )


def _drop_client_carriers_a_tier_pin_supersedes(
    request_kwargs: dict[str, object],  # mutable-ok: the request kwargs
    tier_litellm_params: Mapping[str, object],
) -> None:
    if not OUTPUT_TOKEN_CEILING_PARAMS.isdisjoint(tier_litellm_params):
        _stamp_client_output_ceiling(request_kwargs)
        for carrier in OUTPUT_TOKEN_CEILING_PARAMS:
            request_kwargs.pop(carrier, None)
    if "reasoning_effort" not in tier_litellm_params:
        return
    request_kwargs.pop("thinking", None)
    _pop_effort_from_nested_carrier(request_kwargs, "output_config")
    _pop_effort_from_nested_carrier(request_kwargs, "reasoning")


def _request_header(request_kwargs: Mapping[str, object], header_name: str) -> str | None:
    proxy_server_request: Final = request_kwargs.get("proxy_server_request")
    if not isinstance(proxy_server_request, Mapping):
        return None
    headers: Final = proxy_server_request.get("headers")
    if not isinstance(headers, Mapping):
        return None
    return next(
        (
            value
            for key, value in headers.items()
            if isinstance(key, str) and key.lower() == header_name and isinstance(value, str)
        ),
        None,
    )


def _session_router_cache_key(request_kwargs: Mapping[str, object]) -> str | None:
    session_id: Final = _request_header(request_kwargs, "x-claude-code-session-id")
    if session_id is None or _CLAUDE_CODE_SESSION_ID_RE.fullmatch(session_id) is None:
        return None
    metadata_name: Final = "litellm_metadata" if "litellm_metadata" in request_kwargs else "metadata"
    metadata: Final = request_kwargs.get(metadata_name)
    if not isinstance(metadata, Mapping):
        return None
    caller_scope: Final = metadata.get("user_api_key_hash")
    if not isinstance(caller_scope, str) or not caller_scope:
        return None
    return f"claude_code_session_router:v1:{caller_scope}:{session_id}"


class PreRouter:
    """Answers the Rust core's ``pre_route`` steps (``_core_async_pre_routing_hook``)."""

    def __init__(
        self,
        strategies: PreRoutingStrategies,
        hosted: "HostRegistry",
        host: "PythonHost",
        ctx: object,
        routing_cache: DualCache,
    ) -> None:
        self._strategies: Final = strategies
        self._hosted: Final = hosted
        self._host: Final = host
        self._ctx: Final = ctx
        self._session_cache: Final = DualCache(redis_cache=routing_cache.redis_cache, in_memory_cache=InMemoryCache())

    # -- the step --------------------------------------------------------------------------------

    def has_strategy_routers(self) -> bool:
        """Whether any auto, complexity, adaptive or quality router is registered."""
        return self._strategies.any_registered()

    async def pre_route(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: the request kwargs the hook stamps
        *,
        messages: object,
        input: object,
        fallback_depth: int,
        router_strategy: str | None,
        enable_tag_filtering: bool,
        match_any: bool,
        public_router: object,
        run_plugins: Callable[[str], Awaitable[None]] | None = None,
    ) -> str | None:
        """The start of ``_async_get_available_deployment``: run the hook, record its pick and pin the
        tier's litellm_params. The model group to resolve, or ``None`` to resolve ``model`` itself."""
        if router_strategy not in _ASYNC_PICK_STRATEGIES:
            return None
        responses_call: Final = input is not None and messages is None
        response: Final = await self._hook(
            model,
            request_kwargs,
            messages=messages,
            input=input,
            fallback_depth=fallback_depth,
            enable_tag_filtering=enable_tag_filtering,
            match_any=match_any,
            public_router=public_router,
            run_plugins=run_plugins,
        )
        routed: Final = response.model if response is not None else model
        if response is not None:
            record_pre_routing_selection(request_kwargs, routed)
        self._pin_tier_params(
            routed, response.litellm_params if response is not None else None, request_kwargs, responses_call
        )
        return routed if response is not None else None

    # -- the hook --------------------------------------------------------------------------------

    async def _hook(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: see pre_route
        *,
        messages: object,
        input: object,
        fallback_depth: int,
        enable_tag_filtering: bool,
        match_any: bool,
        public_router: object,
        run_plugins: Callable[[str], Awaitable[None]] | None,
    ) -> "PreRoutingHookResponse | None":
        _restore_client_output_ceiling(request_kwargs)
        _record_routing_decision(request_kwargs, None)
        requested: Final = self._hosted._core_get_model_from_alias(model=model) or model  # pyright: ignore[reportPrivateUsage]  # the host registry's alias lookup
        registered: Final = await self._resolve_session_router(
            model, requested, request_kwargs, fallback_depth, enable_tag_filtering, match_any
        )
        # The routing plugins run on the session-resolved name, before a strategy is selected, so a
        # strategy reads their signals; router-core narrows the pick to what they left.
        if run_plugins is not None:
            await run_plugins(registered)
        selected: Final = self._select_strategy(registered, request_kwargs, enable_tag_filtering, match_any)
        if selected is None:
            await arm_compaction(request_kwargs, None)
            _stamp_or_clear_metadata_key(request_kwargs, SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY, None)
            _stamp_or_clear_metadata_key(request_kwargs, CONSUMED_REQUEST_TAGS_METADATA_KEY, None)
            return None
        from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter

        marker: Final = self._selected_marker(registered, selected.tags, request_kwargs)
        definition: Final = selected.deployment
        config_id: Final = definition.model_info.id if definition else None
        updated_at: Final = definition.model_info.updated_at if definition else None
        router_kind: Final = classify_strategy_router_model(definition.litellm_params.model) if definition else None
        provenance: Final[StandardLoggingRoutingDecision] = {
            "router_model_name": definition.model_name if definition else registered,
            **({"router_type": router_kind} if router_kind is not None else {}),
            **({"router_config_id": config_id} if config_id is not None else {}),
            **({"router_config_updated_at": updated_at.isoformat()} if updated_at is not None else {}),
            **(
                {"router_config_fingerprint": selected.definition_fingerprint}
                if selected.definition_fingerprint is not None
                else {}
            ),
        }  # pyright: ignore[reportAssignmentType]  # the optional keys, as the Python router builds them
        phase_attributes(routing_decision_attributes(provenance))
        reject_recursive_compactor(registered)
        strategy: Final = selected.strategy
        complexity: Final = strategy if isinstance(strategy, ComplexityRouter) else None
        await arm_compaction(
            request_kwargs,
            complexity.config.context_compaction if complexity is not None else None,
            _tier_members(complexity.config.tiers) if complexity is not None else (),  # pyright: ignore[reportArgumentType]  # the tiers' model names
            parent_model=model,
            ctx=self._ctx,  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
            allow_escalation=complexity is not None and complexity.config.enable_context_window_escalation,
            messages=messages,  # pyright: ignore[reportArgumentType]  # the request's messages, as given
        )
        await router_host().authorize_strategy_inference(
            deployment=marker,
            request_kwargs=request_kwargs,
            llm_router=public_router,  # pyright: ignore[reportArgumentType]  # the litellm.Router this core serves
        )
        compression_policy: Final = router_host().compression_policy(
            llm_router=public_router,  # pyright: ignore[reportArgumentType]  # same
            model_alias=registered,
            request_kwargs=request_kwargs,
            request_tags=get_routing_tags_from_request_kwargs(request_kwargs),
        )
        independent_compression: Final = compression_policy is not None and not (
            compression_policy.is_same
            and compression_policy.model is not None
            and router_host().model_hop_compression_armed()
        )
        routing_messages: Final = (
            await router_host().messages_for_routing(
                policy=compression_policy,
                messages=messages,  # pyright: ignore[reportArgumentType]  # same
                request_kwargs=request_kwargs,
            )
            if independent_compression
            else None
        )
        routed: Final = await strategy.async_pre_routing_hook(
            model=registered,
            request_kwargs=request_kwargs,
            messages=routing_messages if routing_messages is not None else messages,  # pyright: ignore[reportArgumentType]  # the request's messages, as given
            input=input,  # pyright: ignore[reportArgumentType]  # the request's input, as given
            specific_deployment=False,
        )
        hooked: Final[PreRoutingHookResponse | None] = routed
        traced: Final = (
            hooked.model_copy(update={"routing_decision": {**provenance, **hooked.routing_decision}})
            if hooked is not None and hooked.routing_decision is not None
            else hooked
        )
        response: Final = (
            traced.model_copy(update={"messages": messages})
            if traced is not None and routing_messages is not None and traced.messages == routing_messages
            else traced
        )
        _record_routing_decision(request_kwargs, response.routing_decision if response else None)
        _stamp_or_clear_metadata_key(
            request_kwargs,
            SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY,
            response.session_affinity_ttl_seconds if response else None,
        )
        request_tags: Final = get_routing_tags_from_request_kwargs(request_kwargs)
        consumed: Final = (
            ConsumedRequestTagsStamp(model_group=response.model, tags=selected.tags)
            if response is not None
            and selected.tags
            and request_tags
            and is_valid_deployment_tag(selected.tags, request_tags, match_any)  # pyright: ignore[reportArgumentType]  # the tags as the Python router passes them
            else None
        )
        _stamp_or_clear_metadata_key(request_kwargs, CONSUMED_REQUEST_TAGS_METADATA_KEY, consumed)
        marker_params: Final = (
            self._forwardable_marker_params(registered, selected.tags, request_kwargs) if response is not None else ()
        )
        tier_param_keys: Final = tuple(response.litellm_params or ()) if response is not None else ()
        overridden: Final = frozenset(tier_param_keys) | (
            OUTPUT_TOKEN_CEILING_PARAMS if not OUTPUT_TOKEN_CEILING_PARAMS.isdisjoint(tier_param_keys) else frozenset()
        )
        newly_forwarded: Final = tuple(
            (key, value) for key, value in marker_params if key not in request_kwargs and key not in overridden
        )
        if response is not None:
            _stamp_client_output_ceiling(request_kwargs)
        request_kwargs.pop(_ALIAS_MARKER_FORWARDED_PARAMS_KWARG, None)
        request_kwargs.update(newly_forwarded)
        if newly_forwarded:
            request_kwargs.update({_ALIAS_MARKER_FORWARDED_PARAMS_KWARG: tuple(key for key, _ in newly_forwarded)})
        return response

    # -- strategy selection ----------------------------------------------------------------------

    def _deployments_for_request(
        self, model: str, request_kwargs: Mapping[str, object]
    ) -> Sequence[Mapping[str, object]]:
        return self._hosted._core_deployments_for_request(model, request_kwargs)  # pyright: ignore[reportPrivateUsage]  # the host registry's lookup

    def _select_strategy(
        self, model: str, request_kwargs: Mapping[str, object], enable_tag_filtering: bool, match_any: bool
    ) -> "TaggedPreRoutingStrategy[Any] | None":
        """``_select_pre_routing_strategy``."""
        registries: Final = self._strategies.all()
        if not any(registries):
            return None
        deployments: Final = self._deployments_for_request(model, request_kwargs)
        registered_names: Final = tuple(dict.fromkeys(str(d.get("model_name")) for d in deployments)) or (model,)
        candidates: Final = (
            tuple(  # comprehension-ok: registry by registry, name by name, as the Python router searches
                tagged for registry in registries for name in registered_names for tagged in registry.get(name, [])
            )
        )
        if not candidates:
            return None
        request_tags: Final = get_routing_tags_from_request_kwargs(request_kwargs)
        if request_tags:
            for tagged in candidates:
                if tagged.tags and is_valid_deployment_tag(list(tagged.tags), request_tags, match_any):
                    return tagged
        for tagged in candidates:
            if "default" in tagged.tags:
                return tagged
        request_scoped: Final = request_kwargs.get("enable_tag_filtering") is True
        if (
            (enable_tag_filtering or request_scoped)
            and all(tagged.tags for tagged in candidates)
            and any(not is_strategy_marker(d) for d in deployments)
        ):
            return None
        return candidates[0]

    def _selected_marker(
        self, model: str, strategy_tags: tuple[str, ...], request_kwargs: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        markers: Final = tuple(
            deployment
            for deployment in self._deployments_for_request(model, request_kwargs)
            if str(as_dict(deployment.get("litellm_params")).get("model", "")).startswith(AUTO_ROUTER_MODEL_PREFIX)
            and "model" in as_dict(deployment.get("litellm_params"))
        )
        tag_matched: Final = tuple(
            deployment
            for deployment in markers
            if tuple(_tags_of(as_dict(deployment.get("litellm_params")))) == strategy_tags
        )
        return tag_matched[0] if tag_matched else (markers[0] if markers else None)

    def _forwardable_marker_params(
        self, model: str, strategy_tags: tuple[str, ...], request_kwargs: Mapping[str, object]
    ) -> tuple[tuple[str, object], ...]:
        marker: Final = self._selected_marker(model, strategy_tags, request_kwargs)
        if marker is None:
            return ()
        return tuple(
            (key, value)
            for key, value in as_dict(marker.get("litellm_params")).items()
            if key not in _ALIAS_PARAMS_NEVER_FORWARDED
            and key not in CustomPricingLiteLLMParams.model_fields
            and value is not None
        )

    # -- the Claude Code session binding ---------------------------------------------------------

    @with_service_target(CLAUDE_CODE_SESSION_ROUTER_BINDING_TARGET)
    async def _delete_binding(self, cache_key: str) -> None:
        try:
            await self._session_cache.async_delete_cache(key=cache_key)
        except Exception as e:  # noqa: BLE001  # cache cleanup must not fail an otherwise routable request
            verbose_router_logger.warning(
                "Failed to delete Claude Code session router binding; the binding may remain until its TTL expires: %s",
                e,
            )

    @with_service_target(CLAUDE_CODE_SESSION_ROUTER_BINDING_TARGET)
    async def _get_binding(self, cache_key: str) -> object:
        try:
            if self._session_cache.redis_cache is None:
                return await self._session_cache.async_get_cache(key=cache_key)
            return await self._session_cache.redis_cache.async_get_cache(key=cache_key)
        except Exception as e:  # noqa: BLE001  # an optional binding must not make routing depend on Redis
            log_redis_failure(
                verbose_router_logger,
                logging.WARNING,
                "Failed to read Claude Code session router binding; using the requested model",
                e,
            )
            return None

    @with_service_target(CLAUDE_CODE_SESSION_ROUTER_BINDING_TARGET)
    async def _resolve_session_router(
        self,
        model: str,
        registered: str,
        request_kwargs: dict[str, object],  # mutable-ok: see pre_route
        fallback_depth: int,
        enable_tag_filtering: bool,
        match_any: bool,
    ) -> str:
        """``_resolve_claude_code_session_router``."""
        if is_native_compaction_call() or not self._strategies.any_registered():
            return registered
        cache_key: Final = _session_router_cache_key(request_kwargs)
        if cache_key is None:
            return registered
        depth: Final = fallback_depth or request_kwargs.get("fallback_depth")
        if depth not in (None, 0):
            return registered
        agent_id: Final = _request_header(request_kwargs, "x-claude-code-agent-id")
        if agent_id is not None:
            bound_model: Final = await self._get_binding(cache_key)
            if not isinstance(bound_model, str):
                return registered
            bound_registered: Final = self._hosted._core_get_model_from_alias(model=bound_model) or bound_model  # pyright: ignore[reportPrivateUsage]  # see _hook
            if self._select_strategy(bound_registered, request_kwargs, enable_tag_filtering, match_any) is None:
                await self._delete_binding(cache_key)
                return registered
            await self._session_cache.async_set_cache(
                key=cache_key, value=bound_model, ttl=_CLAUDE_CODE_SESSION_ROUTER_TTL_SECONDS
            )
            _stamp_or_clear_metadata_key(request_kwargs, "model_group", bound_model)
            return bound_registered
        if self._select_strategy(registered, request_kwargs, enable_tag_filtering, match_any) is None:
            return registered
        await self._session_cache.async_set_cache(
            key=cache_key, value=model, ttl=_CLAUDE_CODE_SESSION_ROUTER_TTL_SECONDS
        )
        return registered

    # -- tier litellm_params ---------------------------------------------------------------------

    def _tier_params_the_target_accepts(
        self, model: str, tier_params: Mapping[str, object], request_kwargs: Mapping[str, object]
    ) -> Mapping[str, object]:
        from litellm.router_rust_shaping import deployment_accepts_param

        deployments: Final = self._hosted._core_get_model_list(model_name=model) or ()  # pyright: ignore[reportPrivateUsage]  # the host registry's lookup
        if not deployments:
            return tier_params
        allowlisted: Final = _allowlist(tier_params) | _allowlist(request_kwargs)
        candidates: Final = provider_rejectable_params(tier_params) - TIER_PARAMS_NEVER_DROPPED - allowlisted
        unsupported: Final = frozenset(
            param
            for param in candidates
            if not any(deployment_accepts_param(deployment, model, param, self._host) for deployment in deployments)
        )
        if not unsupported:
            return tier_params
        verbose_router_logger.warning(
            "litellm.router.py: dropping tier params %s for model=%s, no deployment behind it declares them",
            ", ".join(sorted(unsupported)),
            model,
        )
        return MappingProxyType({key: value for key, value in tier_params.items() if key not in unsupported})

    def _pin_tier_params(
        self,
        model: str,
        tier_litellm_params: Mapping[str, object] | None,
        request_kwargs: dict[str, object],  # mutable-ok: see pre_route
        responses_call: bool,
    ) -> None:
        """``_pin_tier_params_onto_request``."""
        if not tier_litellm_params:
            return
        reject_api_key_secret_reference(tier_litellm_params)
        accepted: Final = self._tier_params_the_target_accepts(model, tier_litellm_params, request_kwargs)
        surface: Final = _tier_ceiling_under_the_surface_name(accepted, responses_call=responses_call)
        _drop_client_carriers_a_tier_pin_supersedes(request_kwargs, surface)
        request_kwargs.update(surface)


def as_dict(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)  # cast-ok: a str-keyed config or request dict


def _adaptive_cost(params: Mapping[str, object], info: Mapping[str, object]) -> object:
    """A model's input price for the adaptive router: litellm_params win over model_info."""
    cost: Final = params.get("input_cost_per_token")
    return cost if cost is not None else info.get("input_cost_per_token")


def _tier_members(tiers: Mapping[str, object]) -> tuple[object, ...]:
    """Every model the tier pools name, once, in order."""
    pools: Final = (
        (pool,) if isinstance(pool, str) or not isinstance(pool, Iterable) else pool for pool in tiers.values()
    )
    return tuple(dict.fromkeys(chain.from_iterable(pools)))


def _tags_of(params: Mapping[str, object]) -> Sequence[object]:
    tags: Final = params.get("tags")
    return tags if isinstance(tags, list | tuple) else ()


def _allowlist(params: Mapping[str, object]) -> frozenset[str]:
    declared: Final = params.get("allowed_openai_params")
    if not isinstance(declared, list | tuple | set | frozenset):
        return frozenset()
    return frozenset(entry for entry in declared if isinstance(entry, str))


__all__ = ["PreRouter", "PreRoutingStrategies", "is_strategy_marker"]
