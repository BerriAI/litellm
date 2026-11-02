"""
Host-side optional pre-call checks for ``RustCore``: the ``CustomLogger`` checks a router builds
from ``optional_pre_call_checks``, ``provider_budget_config``, per-deployment budgets and
``model_group_affinity_config``, registered in the host's ``callbacks`` list exactly as the Python
router's ``add_optional_pre_call_checks`` registers them, since they are features users configure.

The Rust core asks the host to run every routing callback's ``async_filter_deployments`` through a
``callback_filters`` select stage (``async_callback_filter_deployments``) and decides itself what
the surviving deployments or the raised error mean: a pick with nothing left, a retry, a fallback.

This module also keeps the process-wide holder rules (``litellm/router_utils/global_callback_holders``):
a retiring router hands the checks it vacates to a survivor, a runtime removal keeps a check
another live router still has, and ``observe(request_scoped=True)`` takes every routing callback
off the host lists while the router keeps running them (``_routing_callbacks``). It never reaches a
Python router core.
"""

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Final

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import PROMPT_CACHE_PIN_CACHE_MAX_SIZE
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_utils.global_callback_holders import (
    OUTCOME_CALLBACK_LISTS,
    adaptive_routers_on_host,
    live_check_owners,
    on_host,
    routers_holding_global_callbacks,
    routers_holding_global_callbacks_lock,
    sync_adaptive_response_header_hook,
    without_router_callbacks,
)
from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
    DeploymentAffinityCheck,
    warn_on_unknown_model_group_affinity_flags,
)
from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import EncryptedContentAffinityCheck
from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import PromptCachingDeploymentCheck

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.router_python_host import PythonHost
    from litellm.router_rust_registry import HostRegistry
    from litellm.router_rust_selection import StrategySelectors
    from litellm.router_utils.global_callback_holders import GlobalCallbackHolder
    from litellm.types.router import Deployment

_AFFINITY_CHECKS: Final = (
    "deployment_affinity",
    "responses_api_deployment_check",
    "session_affinity",
    "encrypted_content_affinity",
)
# `_RUNTIME_TOGGLEABLE_PRE_CALL_CHECKS`: the checks a runtime list drops when it leaves them out.
_RUNTIME_TOGGLEABLE: Final[Mapping[str, type[CustomLogger]]] = {
    "prompt_caching": PromptCachingDeploymentCheck,
    "enforce_model_rate_limits": ModelRateLimitingCheck,
    "encrypted_content_affinity": EncryptedContentAffinityCheck,
}


def _callbacks(host: "PythonHost") -> list[object]:  # mutable-ok: the process list itself
    return host.callback_list("callbacks")  # pyright: ignore[reportArgumentType]  # not in the CallbackList literal


def register_router_selector(host: "PythonHost", selector: CustomLogger) -> None:
    """``_register_router_selector``: least-busy also counts calls as they start."""
    if isinstance(selector, LeastBusyLoggingHandler):
        if isinstance(host.callback_list("input_callback"), list):  # pyright: ignore[reportUnnecessaryIsInstance]  # a process list a user may have reassigned
            host.callbacks.add_litellm_input_callback(selector)
        else:
            host.set_callback_list("input_callback", [selector])
    if isinstance(_callbacks(host), list):  # pyright: ignore[reportUnnecessaryIsInstance]  # see above
        host.callbacks.add_litellm_callback(selector)


def _move_before_deployment_affinity(
    callback_list: list[object],  # mutable-ok: reordered in place
    callback: object,
) -> None:
    if callback not in callback_list:
        return
    callback_list.remove(callback)
    index: Final = next(
        (i for i, entry in enumerate(callback_list) if isinstance(entry, DeploymentAffinityCheck)),
        len(callback_list),
    )
    callback_list.insert(index, callback)


class OptionalChecks:
    """One router's optional pre-call checks (``optional_callbacks``) and the stores they read."""

    def __init__(
        self,
        owner: "GlobalCallbackHolder",
        host: "PythonHost",
        cache: DualCache,
        hosted: "HostRegistry",
        ctx: object,
        arguments: Mapping[str, object],
    ) -> None:
        self._owner: Final = owner
        self._host: Final = host
        self._cache: Final = cache
        self._hosted: Final = hosted
        self._ctx: Final = ctx
        self.optional_callbacks: list[object] | None = None  # mutable-ok: the router's live list, as Python keeps it
        self.router_budget_logger: RouterBudgetLimiting | None = None
        self.provider_budget_config: object = arguments.get("provider_budget_config")
        ttl: Final = arguments.get("deployment_affinity_ttl_seconds")
        self.deployment_affinity_ttl_seconds: object = 3600 if ttl is None else ttl
        self.session_affinity_repin_on_failover: object = bool(arguments.get("session_affinity_repin_on_failover"))
        affinity_config: Final = arguments.get("model_group_affinity_config")
        self.model_group_affinity_config: dict[str, list[str]] | None = (  # mutable-ok: the argument as passed
            affinity_config if isinstance(affinity_config, dict) else None  # pyright: ignore[reportAttributeAccessIssue]  # the argument as passed
        )
        warn_on_unknown_model_group_affinity_flags(self.model_group_affinity_config)
        self.prompt_caching_pin_cache: DualCache = DualCache(
            redis_cache=cache.redis_cache,
            in_memory_cache=InMemoryCache(max_size_in_memory=PROMPT_CACHE_PIN_CACHE_MAX_SIZE),
        )
        self._request_scoped_selectors: tuple[CustomLogger, ...] = ()
        self._request_scoped_policies: tuple[CustomLogger, ...] = ()
        # ``_slack_alerting_loggers``: the Slack alerting logger ``alerting_config`` builds, if any.
        self.slack_alerting_loggers: tuple[CustomLogger, ...] = ()

    # -- construction ---------------------------------------------------------------------------

    def build(self, checks: object, model_list: object) -> None:
        """The constructor's checks: a budget appends ``router_budget_limiting`` to the list given (in
        place, as Python does), the list is added, a per-group affinity config adds the affinity
        check, and ``apply_default_settings`` adds the empty default list."""
        listed: list[object] | None = checks if isinstance(checks, list) else None  # pyright: ignore[reportUnknownVariableType]  # mutable-ok: appended in place, as Python does  # rebind-ok: the argument as passed
        if RouterBudgetLimiting.should_init_router_budget_limiter(  # pyright: ignore[reportUnknownMemberType]  # untyped model_list
            model_list=model_list if isinstance(model_list, list) else None,  # pyright: ignore[reportUnknownArgumentType]  # the argument as passed
            provider_budget_config=self.provider_budget_config,  # pyright: ignore[reportArgumentType]  # the argument as passed
        ):
            if listed is not None:
                listed.append("router_budget_limiting")
            else:
                listed = ["router_budget_limiting"]  # rebind-ok: the list Python builds
        if listed is not None:
            self.add(listed)  # pyright: ignore[reportUnknownArgumentType]  # names as given
        if self.model_group_affinity_config:
            self.ensure_deployment_affinity_callback()
        self.add([])

    def _is_priority_group(self, model: str) -> bool:
        """``_is_priority_routing_group``."""
        hosted: Final = self._hosted
        resolved: Final = hosted._core_get_model_from_alias(model=model) or model  # pyright: ignore[reportPrivateUsage]  # the host registry's alias lookup
        group: Final = hosted.get_routing_group(resolved)
        return group is not None and group.routing_strategy == "priority"

    def _affinity_check(self, user_key: bool, responses_api: bool, session_id: bool) -> DeploymentAffinityCheck:
        return DeploymentAffinityCheck(
            cache=self._cache,
            ttl_seconds=self.deployment_affinity_ttl_seconds,  # pyright: ignore[reportArgumentType]  # the argument as passed
            enable_user_key_affinity=user_key,
            enable_responses_api_affinity=responses_api,
            enable_session_id_affinity=session_id,
            model_group_affinity_config=self.model_group_affinity_config,
            is_priority_group=self._is_priority_group,
            session_affinity_repin_on_failover=self.session_affinity_repin_on_failover,  # pyright: ignore[reportArgumentType]  # same
        )

    def _own(self, callback: CustomLogger) -> None:
        if self.optional_callbacks is None:
            self.optional_callbacks = []
        self.optional_callbacks.append(callback)
        self._host.callbacks.add_litellm_callback(callback)

    def ensure_deployment_affinity_callback(self) -> None:
        """``_ensure_deployment_affinity_callback``: the affinity check with every global flag off."""
        if any(isinstance(cb, DeploymentAffinityCheck) for cb in self.optional_callbacks or ()):
            return
        self._own(self._affinity_check(False, False, False))

    def _add_affinity(self, checks: Sequence[object]) -> None:
        user_key: Final = "deployment_affinity" in checks
        responses_api: Final = "responses_api_deployment_check" in checks
        session_id: Final = "session_affinity" in checks
        if not (user_key or responses_api or session_id):
            return
        if self.optional_callbacks is None:
            self.optional_callbacks = []
        existing: Final = next(
            (cb for cb in self.optional_callbacks if isinstance(cb, DeploymentAffinityCheck)),
            None,
        )
        if existing is None:
            self._own(self._affinity_check(user_key, responses_api, session_id))
            return
        existing.enable_user_key_affinity = existing.enable_user_key_affinity or user_key
        existing.enable_responses_api_affinity = existing.enable_responses_api_affinity or responses_api
        existing.enable_session_id_affinity = existing.enable_session_id_affinity or session_id
        existing.ttl_seconds = self.deployment_affinity_ttl_seconds  # pyright: ignore[reportAttributeAccessIssue]  # the argument as passed
        if self.model_group_affinity_config:
            existing.model_group_affinity_config = self.model_group_affinity_config

    def _add_encrypted_content_affinity(self, enable_global_affinity: bool) -> None:
        """``_add_encrypted_content_affinity_check``."""
        if not (
            enable_global_affinity
            or EncryptedContentAffinityCheck.has_model_group_affinity_enabled(self.model_group_affinity_config)
        ):
            return
        if self.optional_callbacks is None:
            self.optional_callbacks = []
        existing: Final = next(
            (cb for cb in self.optional_callbacks if isinstance(cb, EncryptedContentAffinityCheck)), None
        )
        if existing is not None:
            existing.ctx = self._ctx  # pyright: ignore[reportAttributeAccessIssue]  # the core's RoutingContext serves registry()
            existing.enable_global_affinity = existing.enable_global_affinity or enable_global_affinity
            existing.model_group_affinity_config = self.model_group_affinity_config or {}
            check: EncryptedContentAffinityCheck = existing  # rebind-ok: the check to move
        else:
            check = EncryptedContentAffinityCheck(  # rebind-ok: the check to move
                ctx=self._ctx,  # pyright: ignore[reportArgumentType]  # see above
                enable_global_affinity=enable_global_affinity,
                model_group_affinity_config=self.model_group_affinity_config,
            )
            self._own(check)
        _move_before_deployment_affinity(self.optional_callbacks, check)
        _move_before_deployment_affinity(_callbacks(self._host), check)

    def add(self, checks: Sequence[object] | None) -> None:
        """``add_optional_pre_call_checks``."""
        if checks is None:
            return
        self._add_affinity(checks)
        self._add_encrypted_content_affinity("encrypted_content_affinity" in checks)
        for name in checks:
            if name in _AFFINITY_CHECKS:
                continue
            callback: CustomLogger | None = None
            if name == "prompt_caching":
                callback = PromptCachingDeploymentCheck(
                    cache=self.prompt_caching_pin_cache, is_priority_group=self._is_priority_group
                )
            elif name == "router_budget_limiting":
                if self.budget_limiter() is not None:
                    continue
                callback = RouterBudgetLimiting(
                    dual_cache=self._cache,
                    provider_budget_config=self.provider_budget_config,  # pyright: ignore[reportArgumentType]  # the argument as passed
                    model_list=self._hosted._core_model_list,  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # the live model_list
                )
                self.router_budget_logger = callback
            elif name == "enforce_model_rate_limits":
                callback = ModelRateLimitingCheck(dual_cache=self._cache)
            if callback is None:
                continue
            if self.optional_callbacks is not None and any(
                isinstance(existing, type(callback)) for existing in self.optional_callbacks
            ):
                continue
            self._own(callback)

    def set(self, checks: object) -> None:
        """``set_optional_pre_call_checks``: drop the toggleable checks the list leaves out, then add it."""
        if checks is None:
            return
        listed: Final[tuple[object, ...]] = tuple(checks) if isinstance(checks, list | tuple) else ()  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # names as given
        requested: Final = frozenset(listed)
        for name, callback_cls in _RUNTIME_TOGGLEABLE.items():
            if name not in requested:
                self._remove_of_type(callback_cls)
        self.add(listed)

    def _remove_of_type(self, callback_cls: type[CustomLogger]) -> None:
        """``_remove_optional_callbacks_of_type``: kept on the host while any other live router of
        the process has a check of that type."""
        if self.optional_callbacks is None or not any(type(cb) is callback_cls for cb in self.optional_callbacks):
            return
        self.optional_callbacks = [cb for cb in self.optional_callbacks if type(cb) is not callback_cls]
        others: Final = tuple(owner for owner in on_host(live_check_owners, self._host) if id(owner) != id(self._owner))
        if any(any(type(cb) is callback_cls for cb in (owner.optional_callbacks or ())) for owner in others):
            return
        callbacks: Final = _callbacks(self._host)
        for cb in tuple(callbacks):
            if type(cb) is callback_cls:
                self._host.callbacks.remove_callback_from_list_by_object(callbacks, cb, require_self=False)  # pyright: ignore[reportUnknownMemberType]  # the manager's untyped helper

    # -- budgets --------------------------------------------------------------------------------

    def budget_limiter(self) -> RouterBudgetLimiting | None:
        """``_core_get_router_deployment_budget_limiter``: exact type, so a proxy subclass is not taken."""
        if self.router_budget_logger is not None:
            return self.router_budget_logger
        for cb in self.optional_callbacks or ():
            if isinstance(cb, RouterBudgetLimiting) and type(cb) is RouterBudgetLimiting:
                self.router_budget_logger = cb
                return cb
        return None

    def sync_deployment_budget(self, deployment: "Deployment") -> None:
        """``_sync_deployment_budget_config``, after a deployment is added."""
        model_id: Final = deployment.model_info.id
        if model_id is None:
            return
        limiter: RouterBudgetLimiting | None = self.budget_limiter()  # rebind-ok: built when first needed
        params: Final = deployment.litellm_params
        if params.max_budget is None or params.budget_duration is None:
            if limiter is not None:
                limiter.unregister_deployment_budget(model_id=model_id)
            return
        if limiter is None:
            self.add(["router_budget_limiting"])
            limiter = self.budget_limiter()  # rebind-ok: the one just built
        if limiter is not None:
            limiter.register_deployment_budget(deployment=deployment.to_json(exclude_none=True))  # pyright: ignore[reportUnknownMemberType]  # pydantic's untyped dump

    def unregister_deployment_budget(self, model_id: str) -> None:
        limiter: Final = self.budget_limiter()
        if limiter is not None:
            limiter.unregister_deployment_budget(model_id=model_id)

    # -- the routing callbacks ------------------------------------------------------------------

    def loggers(self) -> tuple[CustomLogger, ...]:
        """``_optional_loggers``."""
        return tuple(cb for cb in self.optional_callbacks or () if isinstance(cb, CustomLogger))

    def _policies_not_set_globally(self) -> tuple[CustomLogger, ...]:
        keys: Final = self._host.callbacks.get_custom_logger_key
        global_keys: Final = frozenset(keys(cb) for cb in _callbacks(self._host) if isinstance(cb, CustomLogger))
        return tuple(policy for policy in self._request_scoped_policies if keys(policy) not in global_keys)

    def bind_to_request_logging(self, logging_obj: "Logging") -> None:
        """``_core_bind_to_request_logging``'s routing half: the request-scoped selectors replace the
        host's same-keyed callbacks for the request, and the policies the host does not run are
        added to it. The outcome callbacks are not bound: the Rust core records every attempt's
        outcome itself, as it registers no outcome callbacks on the host."""
        for selector in self._request_scoped_selectors:
            logging_obj.replace_global_callback(selector)
        for policy in self._policies_not_set_globally():
            logging_obj.add_dynamic_callback(policy)

    def routing_callbacks(self) -> Iterable[object]:
        """``_routing_callbacks``: the host's callbacks, or after ``observe(request_scoped=True)`` the
        host's callbacks with the router's own selectors and policies in place of theirs."""
        if not (self._request_scoped_selectors or self._request_scoped_policies):
            return tuple(_callbacks(self._host))
        keys: Final = self._host.callbacks.get_custom_logger_key
        replaced: Final = frozenset(map(keys, self._request_scoped_selectors))
        return (
            *self._host.callbacks.without_custom_logger_keys(_callbacks(self._host), replaced),  # pyright: ignore[reportArgumentType]  # the process list holds any callback
            *self._request_scoped_selectors,
            *self._policies_not_set_globally(),
        )

    async def filter_deployments(
        self,
        model: str,
        deployments: list[dict[str, object]],  # mutable-ok: CustomLogger deployment-list contract
        messages: object,
        request: Mapping[str, object],
    ) -> list[object]:  # mutable-ok: as the filters return it
        """``async_callback_filter_deployments``: each routing callback narrows the deployments in
        turn; a filter's error is raised for the Rust core to decide on."""
        remaining: list[object] = list(deployments)  # mutable-ok: filter contract  # rebind-ok: narrowed
        base: Final[object] = CustomLogger.async_filter_deployments  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped deployment dicts
        request_kwargs: Final = dict(request)
        for callback in tuple(self.routing_callbacks()):
            # The base filter returns its input unchanged: only overriding filters are run.
            if not isinstance(callback, CustomLogger) or type(callback).async_filter_deployments is base:  # pyright: ignore[reportUnknownMemberType]  # see above
                continue
            remaining = await callback.async_filter_deployments(  # pyright: ignore[reportUnknownMemberType, reportAssignmentType, reportUnknownVariableType]  # see above
                model=model,
                healthy_deployments=remaining,
                messages=messages,  # pyright: ignore[reportArgumentType]  # the request's messages, as given
                request_kwargs=request_kwargs,
                parent_otel_span=None,
            )
        return remaining

    # -- release --------------------------------------------------------------------------------

    def serve_through_request_logging(self, selectors: Sequence[CustomLogger], dropped: Sequence[object]) -> None:
        """``_core_serve_through_request_logging``: keep running ``selectors`` (and request-scoped
        adaptive hooks) and the optional checks, but take everything off the host lists."""
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook

        adaptive: Final = tuple(
            AdaptiveRouterPostCallHook(adaptive_router=tagged.strategy)  # pyright: ignore[reportArgumentType]  # an adaptive router
            for tagged in chain.from_iterable(self._hosted._core_adaptive_routers.values())  # pyright: ignore[reportPrivateUsage]  # the host registry's live registry
        )
        self._request_scoped_selectors = (*selectors, *adaptive)
        self._request_scoped_policies = (*self.loggers(), *self.slack_alerting_loggers)
        self.release(dropped)

    def own_adaptive_hooks(self) -> tuple[CustomLogger, ...]:
        """``_adaptive_router_hooks``: the host's adaptive post-call hooks of this router's routers."""
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook

        own: Final = frozenset(
            id(tagged.strategy)
            for tagged in chain.from_iterable(self._hosted._core_adaptive_routers.values())  # pyright: ignore[reportPrivateUsage]  # see above
        )
        return tuple(
            hook
            for hook in self._host.callbacks.get_custom_loggers_for_type(AdaptiveRouterPostCallHook)
            if isinstance(hook, AdaptiveRouterPostCallHook) and id(hook.adaptive_router) in own
        )

    def retire_state(self, selectors: Sequence[object]) -> None:
        """``retire_routing_state``: stop the background work of the selectors and the budget limiter."""
        for strategy in (*selectors, *self.loggers()):
            if isinstance(strategy, BaseRoutingStrategy | RouterBudgetLimiting):
                strategy.retire()

    def release(self, dropped: Sequence[object]) -> None:
        """``_release_global_callbacks``: drop ``dropped``, the optional checks and the adaptive hooks
        from the host lists, then hand each vacated check to a surviving router's same-keyed one."""
        with routers_holding_global_callbacks_lock:
            routers_holding_global_callbacks.discard(self._owner)
        adaptive: Final = self.own_adaptive_hooks()
        slack: Final = self.slack_alerting_loggers
        vacated: Final = self._host.callbacks.drop_callbacks(
            (*dropped, *(self.optional_callbacks or ()), *slack, *adaptive), method_owners=(self._owner, *slack)
        )
        with routers_holding_global_callbacks_lock:
            survivors: Final = on_host(tuple(routers_holding_global_callbacks), self._host)
        candidates: Final = chain.from_iterable(router._global_callback_candidates() for router in survivors)  # pyright: ignore[reportPrivateUsage]  # the holder protocol
        for candidate in self._host.callbacks.with_custom_logger_keys(candidates, vacated):
            register_router_selector(self._host, candidate)
        adaptive_keys: Final = frozenset(map(self._host.callbacks.get_custom_logger_key, adaptive))
        survivor: Final = next((router for router in survivors if router._core_adaptive_routers), None)  # pyright: ignore[reportPrivateUsage]  # see above
        if survivor is not None and vacated & adaptive_keys:
            survivor._sync_adaptive_router_hooks()  # pyright: ignore[reportPrivateUsage]  # see above


def filtered_ids(remaining: Iterable[object], candidates: Sequence[str]) -> list[str]:  # mutable-ok: a JSON list
    """The ids of the deployments the filters left, in their order, among the candidates."""
    ids: Final[list[str]] = []  # mutable-ok: built here
    for deployment in remaining:
        found = (as_str_mapping((as_str_mapping(deployment) or {}).get("model_info")) or {}).get("id")
        if isinstance(found, str) and found in candidates and found not in ids:
            ids.append(found)
    return ids


class RustOptionalChecks:
    """``RustCore`` members for its optional pre-call checks and routing callbacks."""

    # RustCore sets these as Final attributes in its constructor.
    @property
    def _opt(self) -> OptionalChecks:
        checks: Final[OptionalChecks] = self.__dict__["_checks"]  # pyright: ignore[reportAny]  # RustCore's own attribute
        return checks

    @property
    def _sel(self) -> "StrategySelectors":
        selectors: Final[StrategySelectors] = self.__dict__["_selectors"]  # pyright: ignore[reportAny]  # same
        return selectors

    @property
    def _reg(self) -> "HostRegistry":
        hosted: Final[HostRegistry] = self.__dict__["_hosted"]  # pyright: ignore[reportAny]  # same
        return hosted

    @property
    def optional_callbacks(self) -> "list[object] | None":  # mutable-ok: the router's live list
        return self._opt.optional_callbacks

    @property
    def router_budget_logger(self) -> RouterBudgetLimiting | None:
        return self._opt.router_budget_logger

    @property
    def deployment_affinity_ttl_seconds(self) -> object:
        return self._opt.deployment_affinity_ttl_seconds

    @property
    def session_affinity_repin_on_failover(self) -> object:
        return self._opt.session_affinity_repin_on_failover

    @property
    def model_group_affinity_config(self) -> "dict[str, list[str]] | None":  # mutable-ok: the argument as passed
        return self._opt.model_group_affinity_config

    def add_optional_pre_call_checks(self, optional_pre_call_checks: "Sequence[object] | None") -> None:
        self._opt.add(optional_pre_call_checks)

    def apply_default_settings(self) -> None:
        """Add the (empty) default checks; an enabled continuation registers the prefill check."""
        from litellm.router_rust_streaming import register_continuation_prefill_check

        self._opt.add([])
        if getattr(self, "enable_mid_stream_fallback_continuation", False):
            register_continuation_prefill_check(self._opt._host)  # pyright: ignore[reportPrivateUsage]  # the checks' host

    # The members another router's retire reads (`global_callback_holders`).

    def _global_callback_candidates(self) -> tuple[CustomLogger, ...]:
        return (*self._sel.callback_selectors(), *self._opt.loggers(), *self._opt.slack_alerting_loggers)

    @property
    def _core_adaptive_routers(self) -> Mapping[str, object]:
        return self._reg._core_adaptive_routers  # pyright: ignore[reportPrivateUsage]  # the host registry's live registry

    def _sync_adaptive_router_hooks(self) -> None:
        strategies: Final = self._reg.strategies
        if strategies is not None:
            strategies._sync_adaptive_router_hooks()  # pyright: ignore[reportPrivateUsage]  # the strategies' own hook sync

    def _join_callback_holders(self) -> None:
        with routers_holding_global_callbacks_lock:
            routers_holding_global_callbacks.add(self)
        live_check_owners.add(self)

    def _serve_through_request_logging(self) -> None:
        selectors: Final = self._sel
        self._opt.serve_through_request_logging(
            selectors.callback_selectors(), (*selectors.callback_selectors(), *selectors.override_selectors())
        )

    def _release_outcome_callbacks(self) -> None:
        """``reset``'s callback half: this core's own entries leave the host's success and failure
        lists; every other entry stays."""
        checks: Final = self._opt
        slack: Final = checks.slack_alerting_loggers
        owned: Final = (
            *self._sel.callback_selectors(),
            *self._sel.override_selectors(),
            *(checks.optional_callbacks or ()),
            *slack,
            *checks.own_adaptive_hooks(),
        )
        host: Final = checks._host  # pyright: ignore[reportPrivateUsage]  # the checks' host
        for name in OUTCOME_CALLBACK_LISTS:
            host.set_callback_list(name, without_router_callbacks(host.callback_list(name), owned, (self, *slack)))

    def _retire_callbacks(self) -> None:
        """``_core_discard``'s callback half: retire the routing state, release the callbacks."""
        live_check_owners.discard(self)
        selectors: Final = (*self._sel.callback_selectors(), *self._sel.override_selectors())
        self._opt.retire_state(selectors)
        self._opt.release(selectors)
        self._sel.clear()
        # The shared adaptive response header hook leaves with the host's last adaptive router.
        host: Final = self._opt._host  # pyright: ignore[reportPrivateUsage]  # the checks' host
        sync_adaptive_response_header_hook(host.callbacks, adaptive_routers_on_host(host))

    async def _answer_callback_filters(
        self, step: Mapping[str, object], request: Mapping[str, object], messages: object
    ) -> tuple[object, BaseException | None]:
        """A ``callback_filters`` step: the remaining ids, or a filter's error as an attempt
        failure with the error itself, which the caller keeps to raise."""
        from litellm.router_rust_core import attempt_failure

        candidates: Final = [str(id) for id in step.get("candidates") or ()]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType, reportUnknownArgumentType]  # the step's id list
        records: Final = self._records_of(candidates)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # RustCore's deployment dicts
        try:
            remaining: Final = await self._opt.filter_deployments(
                str(step.get("model_group")),
                records,  # pyright: ignore[reportUnknownArgumentType]  # see above
                messages,
                request,
            )
        except Exception as error:  # noqa: BLE001  # handed to the Rust core as the try's failure
            return attempt_failure(error), error
        return filtered_ids(remaining, candidates), None  # pyright: ignore[reportUnknownArgumentType]  # see above


__all__ = [
    "OptionalChecks",
    "RustOptionalChecks",
    "filtered_ids",
    "register_router_selector",
]
