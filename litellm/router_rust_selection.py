"""
Host-side strategy selectors for ``RustCore``.

The Rust core decides which strategy picks and whose state it reads (``strategy::routing_context``:
a request override, then the routing group, then the router's own strategy) and asks the host to
run the stateful strategies it does not implement natively through a ``select`` step. The
strategies themselves are the standalone ``litellm/router_strategy`` loggers, which double as
``CustomLogger`` callbacks that record traffic in the host's routing cache. This module builds,
registers and retires them the way the Python router's ``routing_strategy_init``,
``_init_routing_groups`` and ``_get_override_strategy_selector`` do. It never reaches a Python
router core.
"""

import threading
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.types.router import RoutingGroup, RoutingStrategy

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.router_python_host import PythonHost

Selector: TypeAlias = (
    LeastBusyLoggingHandler
    | LowestCostLoggingHandler
    | LowestLatencyLoggingHandler
    | LowestTPMLoggingHandler
    | LowestTPMLoggingHandler_v2
)

# The legacy router attribute that holds the default group's selector, per strategy.
SELECTOR_ATTRIBUTES: Final = MappingProxyType(
    {
        "least-busy": "leastbusy_logger",
        "usage-based-routing": "lowesttpm_logger",
        "usage-based-routing-v2": "lowesttpm_logger_v2",
        "latency-based-routing": "lowestlatency_logger",
        "cost-based-routing": "lowestcost_logger",
    }
)


def normalize_strategy(strategy: object) -> str | None:
    """A strategy as the string Python compares against (``_normalize_strategy``)."""
    if isinstance(strategy, RoutingStrategy):
        return strategy.value
    return strategy if isinstance(strategy, str) else None


def build_selector(strategy: str | None, args: Mapping[str, object], cache: "DualCache") -> Selector | None:
    """``_build_strategy_selector`` without registering: ``None`` for simple-shuffle and every
    strategy without a selector."""
    routing_args: Final = dict(args)
    match strategy:
        case "least-busy":
            return LeastBusyLoggingHandler(router_cache=cache)
        case "usage-based-routing":
            return LowestTPMLoggingHandler(router_cache=cache, routing_args=routing_args)
        case "usage-based-routing-v2":
            return LowestTPMLoggingHandler_v2(router_cache=cache, routing_args=routing_args)
        case "latency-based-routing":
            return LowestLatencyLoggingHandler(router_cache=cache, routing_args=routing_args)
        case "cost-based-routing":
            return LowestCostLoggingHandler(router_cache=cache, routing_args={})
        case _:
            return None


def _callbacks(host: "PythonHost") -> list[object]:  # mutable-ok: the process list itself
    """``litellm.callbacks``, which the host serves like its other callback lists."""
    return host.callback_list("callbacks")  # pyright: ignore[reportArgumentType]  # not in the CallbackList literal


class StrategySelectors:
    """The selectors of one router: the default group's, each routing group's, and the lazily
    built per-request override ones."""

    def __init__(self, host: "PythonHost", cache: "DualCache") -> None:
        self._host: Final = host
        self._cache: Final = cache
        self._strategy: str | None = None
        self._default: Selector | None = None
        self._groups: dict[str, Selector] = {}  # mutable-ok: replaced as routing groups change
        self._group_strategies: frozenset[str] = frozenset()
        self._overrides: dict[str, Selector | None] = {}  # mutable-ok: filled lazily per strategy
        self._lock: Final = threading.Lock()

    # -- building ------------------------------------------------------------------------------

    def _register(self, selector: CustomLogger) -> None:
        """``_register_router_selector``: least-busy also counts calls as they start."""
        host: Final = self._host
        if isinstance(selector, LeastBusyLoggingHandler):
            if isinstance(host.callback_list("input_callback"), list):  # pyright: ignore[reportUnnecessaryIsInstance]  # a process list a user may have reassigned
                host.callbacks.add_litellm_input_callback(selector)
            else:
                host.set_callback_list("input_callback", [selector])
        if isinstance(_callbacks(host), list):  # pyright: ignore[reportUnnecessaryIsInstance]  # see above
            host.callbacks.add_litellm_callback(selector)

    def _unregister(self, selectors: Sequence[object]) -> None:
        live: Final = [selector for selector in selectors if selector is not None]
        for selector in live:
            if isinstance(selector, BaseRoutingStrategy):
                selector.retire()
        if live:
            self._host.callbacks.drop_callbacks(live)

    def init_default(self, strategy: object, args: Mapping[str, object]) -> None:
        """``routing_strategy_init``: drop the old default and override selectors, build the new one."""
        self._unregister([self._default, *self._overrides.values()])
        self._overrides = {}
        self._strategy = normalize_strategy(strategy)
        self._default = build_selector(self._strategy, args, self._cache)
        if self._default is not None:
            self._register(self._default)

    def rebuild_default(self, args: Mapping[str, object]) -> None:
        """``_apply_updated_routing_strategy_args``: relink the default selector to new args, keeping
        the old one when they are invalid."""
        from pydantic import ValidationError

        from litellm._logging import verbose_router_logger

        if self._default is None:
            return
        try:
            rebuilt: Final = build_selector(self._strategy, args, self._cache)
        except (TypeError, ValidationError):
            verbose_router_logger.exception(
                "Invalid routing_strategy_args %s for '%s'; keeping the previous ones", dict(args), self._strategy
            )
            return
        self._unregister([self._default])
        self._default = rebuilt
        if rebuilt is not None:
            self._register(rebuilt)

    def init_groups(self, groups: Sequence[tuple[str, str, Mapping[str, object]]]) -> None:
        """``_replace_routing_groups``: one selector per group, from (name, strategy, args)."""
        self._unregister(list(self._groups.values()))
        built: Final = {
            name: selector
            for name, strategy, args in groups
            if (selector := build_selector(None if strategy == "priority" else strategy, args, self._cache)) is not None
        }
        for selector in built.values():
            self._register(selector)
        self._groups = built
        self._group_strategies = frozenset(strategy for _, strategy, _ in groups)

    def retire(self) -> None:
        self._unregister([self._default, *self._groups.values(), *self._overrides.values()])
        self.clear()

    def clear(self) -> None:
        """Forget every selector, after the caller released them from the host lists."""
        self._default = None
        self._groups = {}
        self._overrides = {}

    def callback_selectors(self) -> tuple[Selector, ...]:
        """``_callback_selectors``: the default group's selector and each routing group's."""
        return tuple(selector for selector in (self._default, *self._groups.values()) if selector is not None)

    def override_selectors(self) -> tuple[Selector, ...]:
        """The per-request override selectors built so far."""
        return tuple(selector for selector in self._overrides.values() if selector is not None)

    @property
    def default(self) -> Selector | None:
        return self._default

    def attribute(self, name: str) -> Selector | None:
        """A legacy ``*_logger`` attribute: the default selector when it is that strategy's."""
        return self._default if SELECTOR_ATTRIBUTES.get(self._strategy or "") == name else None

    def latency_args(self) -> Mapping[str, object] | None:
        """The default latency selector's full routing args, as ``get_settings`` reports them."""
        if not isinstance(self._default, LowestLatencyLoggingHandler):
            return None
        reported: Final[object] = self._default.routing_args.json()  # pyright: ignore[reportUnknownMemberType]  # RoutingArgs.json is untyped
        return as_str_mapping(reported)

    # -- picking -------------------------------------------------------------------------------

    def selector_for(
        self, strategy: str, scope: str, group: str | None, request: Mapping[str, object] | None
    ) -> Selector | None:
        """The selector a Rust ``select`` step names (``_get_routing_context``)."""
        if scope == "group":
            return self._groups.get(group or "")
        if scope == "override":
            return self._override(strategy, request)
        return self._default if strategy == self._strategy else None

    def _override(self, strategy: str, request: Mapping[str, object] | None) -> Selector | None:
        """``_get_override_strategy_selector`` and ``_bind_override_selector_to_request``."""
        if strategy == self._strategy:
            return self._default
        with self._lock:
            if strategy not in self._overrides:
                self._overrides[strategy] = build_selector(strategy, {}, self._cache)
            selector: Final = self._overrides[strategy]
        if selector is not None and request is not None and strategy not in self._registered_strategies():
            from litellm.litellm_core_utils.litellm_logging import Logging

            logging_obj: Final = request.get("litellm_logging_obj")
            if isinstance(logging_obj, Logging):
                logging_obj.add_dynamic_callback(selector)
        return selector

    def _registered_strategies(self) -> frozenset[str]:
        return frozenset({*((self._strategy,) if self._strategy else ()), *self._group_strategies})

    def is_override(self, strategy: str, selector: object) -> bool:
        """Whether ``selector`` is a per-request override one, whose pre-call check the router runs
        itself because it is not in the callback lists."""
        return selector is not None and self._overrides.get(strategy) is selector


async def pick_async(
    strategy: str,
    selector: Selector,
    model: str,
    deployments: list[dict[str, object]],  # mutable-ok: the selectors take a list of deployment dicts
    messages: object,
    input: object,
    request_kwargs: dict[str, object] | None,  # mutable-ok: the latency selector reads it as given
) -> object:
    """``_select_deployment_async``."""
    match selector:
        case LeastBusyLoggingHandler() if strategy == "least-busy":
            return await selector.async_get_available_deployments(model_group=model, healthy_deployments=deployments)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the selectors return untyped deployment dicts
        case LowestTPMLoggingHandler() if strategy == "usage-based-routing":
            return selector.get_available_deployments(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the selectors return untyped deployment dicts
                model_group=model,
                healthy_deployments=deployments,
                messages=messages,  # pyright: ignore[reportArgumentType]  # the request's messages, as given
                input=input,  # pyright: ignore[reportArgumentType]  # same
            )
        case LowestTPMLoggingHandler_v2() | LowestCostLoggingHandler() if strategy in (
            "usage-based-routing-v2",
            "cost-based-routing",
        ):
            return await selector.async_get_available_deployments(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the selectors return untyped deployment dicts
                model_group=model,
                healthy_deployments=deployments,
                messages=messages,  # pyright: ignore[reportArgumentType]  # same
                input=input,  # pyright: ignore[reportArgumentType]  # same
            )
        case LowestLatencyLoggingHandler() if strategy == "latency-based-routing":
            return await selector.async_get_available_deployments(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the selectors return untyped deployment dicts
                model_group=model,
                healthy_deployments=deployments,
                messages=messages,  # pyright: ignore[reportArgumentType]  # same
                input=input,  # pyright: ignore[reportArgumentType]  # same
                request_kwargs=request_kwargs,
            )
        case _:
            return None


def pick_sync(
    strategy: str,
    selector: Selector,
    model: str,
    deployments: list[dict[str, object]],  # mutable-ok: see pick_async
    messages: object,
    input: object,
    request_kwargs: dict[str, object] | None,  # mutable-ok: see pick_async
) -> object:
    """``_select_deployment_sync``."""
    match selector:
        case LeastBusyLoggingHandler() if strategy == "least-busy":
            return selector.get_available_deployments(model_group=model, healthy_deployments=deployments)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the selectors return untyped deployment dicts
        case LowestTPMLoggingHandler() | LowestTPMLoggingHandler_v2() | LowestCostLoggingHandler() if strategy in (
            "usage-based-routing",
            "usage-based-routing-v2",
            "cost-based-routing",
        ):
            return selector.get_available_deployments(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the selectors return untyped deployment dicts
                model_group=model,
                healthy_deployments=deployments,
                messages=messages,  # pyright: ignore[reportArgumentType]  # same
                input=input,  # pyright: ignore[reportArgumentType]  # same
            )
        case LowestLatencyLoggingHandler() if strategy == "latency-based-routing":
            return selector.get_available_deployments(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the selectors return untyped deployment dicts
                model_group=model,
                healthy_deployments=deployments,
                messages=messages,  # pyright: ignore[reportArgumentType]  # same
                input=input,  # pyright: ignore[reportArgumentType]  # same
                request_kwargs=request_kwargs,
            )
        case _:
            return None


async def routing_pre_call_checks(callbacks: Iterable[object], deployment: Mapping[str, object]) -> None:
    """``async_routing_strategy_pre_call_checks``: every ``CustomLogger`` among the router's routing
    callbacks (``_routing_callbacks``) checks the picked deployment before the call
    (usage-based-routing-v2 counts its rpm here). A check's error fails the attempt; the core
    records that failure like any other."""
    for callback in tuple(callbacks):
        if isinstance(callback, CustomLogger):
            await callback.async_pre_call_check(dict(deployment), None)  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes an untyped deployment dict


def routing_pre_call_checks_sync(callbacks: Iterable[object], deployment: Mapping[str, object]) -> None:
    """``routing_strategy_pre_call_checks``: the sync check of each ``CustomLogger`` callback."""
    for callback in tuple(callbacks):
        if isinstance(callback, CustomLogger):
            callback.pre_call_check(dict(deployment))  # pyright: ignore[reportUnknownMemberType]  # see above


def picked_id(picked: object) -> str | None:
    """The id of the deployment a selector returned."""
    info: Final = as_str_mapping((as_str_mapping(picked) or {}).get("model_info")) or {}
    found: Final = info.get("id")
    return str(found) if found is not None else None


__all__ = [
    "SELECTOR_ATTRIBUTES",
    "Selector",
    "StrategySelectors",
    "build_selector",
    "group_documents",
    "group_settings",
    "member_map",
    "normalize_strategy",
    "parse_groups",
    "pick_async",
    "pick_sync",
    "picked_id",
    "routing_pre_call_checks",
    "routing_pre_call_checks_sync",
]


def parse_groups(
    groups_input: object, model_names: frozenset[str], aliases: frozenset[str]
) -> tuple[RoutingGroup, ...]:
    """``_init_routing_groups``' validation: ``parse_routing_groups``, then a priority group may not
    share its name with a model group or alias (a plain group only warns: its name is shadowed)."""
    from litellm._logging import verbose_router_logger
    from litellm.litellm_core_utils.routing_groups import (
        parse_routing_groups,  # pyright: ignore[reportUnknownVariableType]  # takes untyped dicts
    )

    if not groups_input or not isinstance(groups_input, list | tuple):
        return ()
    groups: Final = parse_routing_groups(  # pyright: ignore[reportUnknownVariableType]  # returns RoutingGroup models
        list(groups_input),  # pyright: ignore[reportUnknownArgumentType]  # validated by parse_routing_groups
        known_model_names=model_names,
    )
    for group in groups:
        if group.group_name in model_names or group.group_name in aliases:
            if group.routing_strategy == "priority":
                raise ValueError("Priority routing group names must not shadow a model or alias")
            verbose_router_logger.warning(
                "routing_groups: group_name '%s' is shadowed by an existing model_name or model_group_alias; "
                "the group's strategy still applies to its members, but the name is not callable until renamed.",
                group.group_name,
            )
    return groups


def group_documents(groups: Sequence[RoutingGroup]) -> list[Mapping[str, object]]:  # mutable-ok: a JSON list
    """The routing groups as the Rust core reads them."""
    return [
        {
            "group_name": group.group_name,
            "models": list(group.models),
            "routing_strategy": group.routing_strategy,
            **({"model_priorities": dict(group.model_priorities)} if group.model_priorities is not None else {}),
        }
        for group in groups
    ]


def group_settings(groups: Sequence[RoutingGroup]) -> list[object]:  # mutable-ok: the settings snapshot's list
    """``get_settings()['routing_groups']``."""
    return [
        group.model_dump(exclude={"model_priorities"}) if group.model_priorities is None else group.model_dump()
        for group in groups
    ]


def member_map(groups: Sequence[RoutingGroup]) -> Mapping[str, str]:
    """``_model_to_group``: each member model group of a non-priority routing group, to the group
    (a model listed by two groups maps to the last)."""
    kept: Final = [group for group in groups if group.routing_strategy != "priority"]
    return dict(chain.from_iterable(dict.fromkeys(group.models, group.group_name).items() for group in kept))
