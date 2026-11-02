"""
The deprecated Router members (``DeprecatedRouterMembers``, litellm/router_deprecated.py) on a Router
over each core: every old attribute reads what the Python core's ``_core_<name>`` holds, through the
entry point named in its DeprecationWarning, and the old writes have the Python core's effect.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.router import Router
from litellm.router_api import SETTING_FIELDS, ChangeSet
from litellm.router_core import RouterCore
from litellm.router_utils.cooldown_cache import CooldownCache
from litellm.router_utils.kubernetes_pod_discovery import KubernetesPodDiscovery
from litellm.types.router import AllowedFailsPolicy, Deployment, RouterGeneralSettings, RoutingContext
from tests.unit.test_router.core.registry_helpers import MESSAGES, MakeCore, ids, team_deployment
from tests.unit.test_router.core.scripted_host import deployment


def as_router(core: RouterCore) -> Router:
    """A ``litellm.Router`` facade over ``core``, as ``Router.from_config`` attaches the core it builds."""
    router: Final = Router.__new__(Router)
    router._attach(core)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]  # the facade's own attach, as from_config calls it
    return router


def picked_id(picked: object) -> object:
    """The ``model_info.id`` of a picked deployment dict."""
    return (as_str_mapping((as_str_mapping(picked) or {}).get("model_info")) or {}).get("id")


def read(router: Router, name: str) -> object:
    with pytest.warns(DeprecationWarning, match=rf"Router\.{name} is deprecated"):
        return getattr(router, name)  # pyright: ignore[reportAny]  # each old member's own type


def write(router: Router, name: str, value: object) -> None:
    with pytest.warns(DeprecationWarning, match=rf"Router\.{name} is deprecated"):
        setattr(router, name, value)


_CONFIGURED: Final[dict[str, object]] = {  # mutable-ok: constructor arguments, copied per router
    "model_list": [
        deployment("gpt", "a"),
        deployment("gpt", "b"),
        deployment("claude", "c", model="anthropic/claude-x"),
        team_deployment("int-t", "t1", "TA", "pub", model="openai/team-model"),
        deployment("openai/*", "w", model="openai/*"),
    ],
    "fallbacks": [{"gpt": ["claude"]}],
    "context_window_fallbacks": [{"gpt": ["claude"]}],
    "content_policy_fallbacks": [{"claude": ["gpt"]}],
    "cooldown_time": 7,
    "allowed_fails": 2,
    "allowed_fails_policy": AllowedFailsPolicy(RateLimitErrorAllowedFails=4),
    "disable_cooldowns": True,
    "enable_pre_call_checks": True,
    "enable_tag_filtering": True,
    "tag_filtering_match_any": False,
    "tag_routing_prefix": "x-",
    "default_litellm_params": {"metadata": {"k": "v"}},
    "routing_strategy": "least-busy",
    "default_max_parallel_requests": 5,
    "guardrail_list": [{"guardrail_name": "g", "litellm_params": {"guardrail": "x"}}],
    "search_tools": [{"search_tool_name": "s", "litellm_params": {"search_provider": "tavily"}}],
    "health_check_ignore_transient_errors": True,
    "enable_health_check_routing": True,
    "background_health_check_model_groups": ["gpt"],
    "model_group_alias": {"fast": "gpt"},
    "cache_responses": True,
}

# The Python core's values for _CONFIGURED, by setting (the objects it builds are compared by type).
_CONFIGURED_SETTINGS: Final[dict[str, object]] = {  # mutable-ok: expected values
    "routing_strategy": "least-busy",
    "fallbacks": [{"gpt": ["claude"]}],
    "context_window_fallbacks": [{"gpt": ["claude"]}],
    "content_policy_fallbacks": [{"claude": ["gpt"]}],
    "default_deployment": None,
    "default_litellm_params": {"timeout": None, "max_retries": 0, "metadata": {"k": "v", "caching_groups": None}},
    "default_max_parallel_requests": 5,
    "cooldown_time": 7,
    "allowed_fails": 2,
    "allowed_fails_policy": AllowedFailsPolicy(RateLimitErrorAllowedFails=4),
    "disable_cooldowns": True,
    "enable_pre_call_checks": True,
    "enable_health_check_routing": True,
    "health_check_ignore_transient_errors": True,
    "background_health_check_model_groups": frozenset({"gpt"}),
    "enable_tag_filtering": True,
    "tag_filtering_match_any": False,
    "tag_routing_prefix": "x-",
    "router_general_settings": RouterGeneralSettings(),
    "provider_budget_config": None,
    "provider_default_deployment_ids": ["w"],
    "fallback_access_check": None,
    "fallback_budget_check": None,
    "search_tools": [{"search_tool_name": "s", "litellm_params": {"search_provider": "tavily"}}],
    "guardrail_list": [{"guardrail_name": "g", "litellm_params": {"guardrail": "x"}}],
    "cache_responses": True,
}


def _configured(make_core: MakeCore) -> Router:
    return as_router(make_core(**{name: _copy(value) for name, value in _CONFIGURED.items()}))


def _copy(value: object) -> object:
    import copy

    return copy.deepcopy(value)


# -- settings ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(set(SETTING_FIELDS) - {"kubernetes_pod_discovery"}))
def test_a_setting_attribute_reads_the_configured_setting(make_core: MakeCore, name: str) -> None:
    router: Final = _configured(make_core)
    value: Final = read(router, name)
    assert value == _CONFIGURED_SETTINGS[name]
    assert value is getattr(router.settings(), name)


def test_the_pod_discovery_setting_is_the_routers_pod_lookup(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    discovery: Final = read(router, "kubernetes_pod_discovery")
    assert isinstance(discovery, KubernetesPodDiscovery)
    assert discovery is router.settings().kubernetes_pod_discovery


def test_default_settings_read_as_the_python_cores_defaults(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    assert read(router, "routing_strategy") == "simple-shuffle"
    assert read(router, "cooldown_time") == 5
    assert read(router, "allowed_fails") == 3
    assert read(router, "fallbacks") is None
    assert read(router, "default_litellm_params") == {
        "timeout": None,
        "max_retries": 0,
        "metadata": {"caching_groups": None},
    }
    assert read(router, "tag_filtering_match_any") is True
    assert read(router, "tag_routing_prefix") == ""
    assert read(router, "cache_responses") is False
    assert read(router, "search_tools") == []
    assert read(router, "guardrail_list") == []


def test_a_setting_attribute_write_is_a_changeset(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    write(router, "cooldown_time", 9)
    write(router, "fallbacks", [{"gpt": ["gpt"]}])
    assert read(router, "cooldown_time") == 9
    assert router.settings().cooldown_time == 9
    assert read(router, "fallbacks") == [{"gpt": ["gpt"]}]


def test_a_deleted_setting_attribute_no_longer_reads(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    with pytest.warns(DeprecationWarning, match=r"Router\.fallbacks is deprecated"):
        del router.fallbacks
    assert not router.settings().is_set("fallbacks")
    with pytest.raises(AttributeError, match="'fallbacks'"):
        read(router, "fallbacks")


# -- registry state ------------------------------------------------------------------------------

_REGISTRY_READS: Final[dict[str, Callable[[Router], object]]] = {
    "model_list": lambda r: r.registry().all(),
    "model_names": lambda r: r.registry().names(kind="group_set"),
    "deployment_names": lambda r: r.registry().names(kind="deployment_names"),
    "team_public_model_names": lambda r: r.registry().names(kind="team_public"),
    "model_name_to_deployment_indices": lambda r: r.registry().index("model_name"),
    "team_model_to_deployment_indices": lambda r: r.registry().index("team_model"),
    "cached_deployment_model_info": lambda r: r.registry().deployment_model_info_cache(),
    "model_group_alias": lambda r: r.registry().aliases(),
    "pattern_router": lambda r: r.registry().patterns(),
    "team_pattern_routers": lambda r: r.registry().patterns(team=True),
    "routing_groups": lambda r: r.registry().routing_groups(),
    "auto_routers": lambda r: r.registry().plugins("auto"),
    "complexity_routers": lambda r: r.registry().plugins("complexity"),
    "quality_routers": lambda r: r.registry().plugins("quality"),
    "adaptive_routers": lambda r: r.registry().plugins("adaptive"),
    "routing_plugins": lambda r: r.registry().plugins("routing"),
    "prompt_caching_pin_cache": lambda r: r.registry().plugins("prompt_caching_pins"),
    "cache": lambda r: r.health().store,
    "cooldown_cache": lambda r: r.health().cooldowns,
    "health_state_cache": lambda r: r.health().health_checks,
    "deployment_latency_map": lambda r: r.health().latency,
}


@pytest.mark.parametrize("name", sorted(_REGISTRY_READS))
def test_a_state_attribute_is_the_live_object_its_entry_point_returns(make_core: MakeCore, name: str) -> None:
    router: Final = _configured(make_core)
    assert read(router, name) is _REGISTRY_READS[name](router)


def test_the_registry_index_attributes_hold_the_python_cores_values(make_core: MakeCore) -> None:
    router: Final = _configured(make_core)
    assert read(router, "model_names") == {"gpt", "claude", "int-t", "openai/*"}
    assert read(router, "deployment_names") == [
        "openai/a",
        "openai/b",
        "anthropic/claude-x",
        "openai/team-model",
        "openai/*",
    ]
    assert read(router, "team_public_model_names") == frozenset({"pub"})
    assert read(router, "model_name_to_deployment_indices") == {
        "gpt": [0, 1],
        "claude": [2],
        "int-t": [3],
        "openai/*": [4],
    }
    assert read(router, "team_model_to_deployment_indices") == {("TA", "pub"): [3]}
    assert read(router, "model_group_alias") == {"fast": "gpt"}
    assert read(router, "routing_groups") == {}
    with pytest.warns(DeprecationWarning, match=r"Router\.routing_groups is deprecated"):
        assert router._routing_groups == {}  # pyright: ignore[reportPrivateUsage]  # the old private alias
    assert read(router, "routing_plugins") == []
    assert read(router, "deployment_latency_map") == {
        "openai/a": 0,
        "openai/b": 0,
        "anthropic/claude-x": 0,
        "openai/team-model": 0,
        "openai/*": 0,
    }
    assert ids(read(router, "model_list")) == ["a", "b", "c", "t1", "w"]  # pyright: ignore[reportArgumentType]  # the live model list


def test_the_cached_model_info_attribute_is_the_lru_cache(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    with pytest.warns(DeprecationWarning, match=r"Router\.cached_deployment_model_info is deprecated"):
        cached: Final = router.cached_deployment_model_info
    first: Final = cached("a", "gpt")
    assert cached("a", "gpt") is first
    assert cached.cache_info().hits == 1
    assert first == router.registry().deployment_model_info("a", "gpt")


def test_a_configured_routing_plugin_is_listed(make_core: MakeCore) -> None:
    class Keep:
        async def run(self, context: RoutingContext) -> RoutingContext:
            return context

    plugin: Final = Keep()
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")], plugins=[plugin]))
    assert read(router, "routing_plugins") == [plugin]


def test_a_routing_plugin_appended_to_the_live_list_runs(make_core: MakeCore) -> None:
    class Narrow:
        async def run(self, context: RoutingContext) -> RoutingContext:
            context.candidate_models = [m for m in context.candidate_models if m == "openai/b"]
            return context

    class Keep:
        async def run(self, context: RoutingContext) -> RoutingContext:
            return context

    router: Final = as_router(make_core(model_list=[deployment("g", "a"), deployment("g", "b")], plugins=[Keep()]))
    with pytest.warns(DeprecationWarning, match=r"Router\.routing_plugins is deprecated"):
        plugins: Final = router.routing_plugins
    plugins.append(Narrow())

    import asyncio

    for _ in range(4):
        picked = asyncio.run(router.select("g", {}, messages=MESSAGES))  # pyright: ignore[reportCallIssue]
        assert picked_id(picked) == "b"


# -- state writes --------------------------------------------------------------------------------


def test_a_registry_state_write_is_what_the_registry_reads(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    write(router, "model_names", {"x"})
    write(router, "deployment_names", ["openai/x"])
    write(router, "model_name_to_deployment_indices", {"x": [0]})
    write(router, "auto_routers", {})
    assert read(router, "model_names") == {"x"}
    assert router.registry().names(kind="group_set") == {"x"}
    assert router.registry().names(kind="deployment_names") == ["openai/x"]
    assert router.registry().index("model_name") == {"x": [0]}
    assert router.registry().plugins("auto") == {}


def test_a_routing_state_write_is_what_health_reads(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    store: Final = DualCache()
    cooldowns: Final = CooldownCache(cache=store, default_cooldown_time=1)
    latency: Final[dict[str, float]] = {"openai/a": 0.5}
    write(router, "cache", store)
    write(router, "cooldown_cache", cooldowns)
    write(router, "deployment_latency_map", latency)
    assert read(router, "cache") is store
    assert router.health().store is store
    assert router.health().cooldowns is cooldowns
    assert router.health().latency is latency


def test_an_assigned_slack_alerting_logger_is_the_routers_observer(make_core: MakeCore) -> None:
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting

    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    with (
        pytest.raises(AttributeError, match="slack_alerting_logger"),
        pytest.warns(DeprecationWarning, match=r"Router\.slack_alerting_logger is deprecated"),
    ):
        _ = router.slack_alerting_logger
    logger: Final = SlackAlerting(alerting=["slack"], default_webhook_url="https://example.invalid/hook")
    write(router, "slack_alerting_logger", logger)
    assert read(router, "slack_alerting_logger") is logger
    assert router.observe().slack_alerting_logger is logger


@pytest.mark.asyncio
async def test_an_assigned_async_pick_answers_select(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a"), deployment("gpt", "b")]))
    calls: Final[list[str]] = []  # mutable-ok: the models the pick was asked for

    async def pick(
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: the old pick's signature
        messages: object = None,
        input: object = None,
        specific_deployment: object = False,
    ) -> dict[str, object]:  # mutable-ok: a deployment dict
        calls.append(model)
        return deployment("gpt", "b")

    write(router, "async_get_available_deployment", pick)
    assert read(router, "async_get_available_deployment") is pick
    picked: Final = await router.select("gpt", {})
    assert picked_id(picked) == "b"
    assert calls == ["gpt"]


@pytest.mark.asyncio
async def test_the_async_pick_attribute_picks_like_select(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    with pytest.warns(DeprecationWarning, match=r"Router\.async_get_available_deployment is deprecated"):
        pick: Final = router.async_get_available_deployment
    picked: Final = await pick(model="gpt", request_kwargs={})
    assert picked_id(picked) == "a"


# -- methods -------------------------------------------------------------------------------------


def _call(router: Router, name: str, *args: object, **kwargs: object) -> object:  # kwargs-ok: the old member's
    with pytest.warns(DeprecationWarning, match=rf"Router\.{name} is deprecated"):
        return getattr(router, name)(*args, **kwargs)  # pyright: ignore[reportAny]  # each old member's own type


def test_registry_methods_answer_through_the_registry(make_core: MakeCore) -> None:
    router: Final = _configured(make_core)
    registry: Final = router.registry()
    found: Final = _call(router, "get_deployment", "a")
    assert isinstance(found, Deployment)
    assert found.model_info.id == "a"
    assert _call(router, "has_model_id", "a") is True
    assert _call(router, "get_model_info", "zzz") is None
    assert ids(_call(router, "get_model_list", "gpt")) == ["a", "b"]  # pyright: ignore[reportArgumentType]
    assert ids(_call(router, "get_all_deployments", "gpt")) == ["a", "b"]  # pyright: ignore[reportArgumentType]
    assert ids(_call(router, "get_model_list_from_model_alias", "fast")) == ["a", "b"]  # pyright: ignore[reportArgumentType]
    assert _call(router, "get_model_names") == registry.names()
    assert _call(router, "get_model_ids", "gpt") == ["a", "b"]
    assert _call(router, "get_model_from_alias", "fast") == "gpt"
    assert _call(router, "routable_model_group", "fast") == "gpt"
    assert _call(router, "is_recognized_model", "claude") is True
    assert _call(router, "get_model_group", "a") == registry.resolve("a", kind="group_of_id")
    assert _call(router, "map_team_model", "pub", "TA") == "pub"
    assert _call(router, "get_model_access_groups") == registry.access_groups()
    assert _call(router, "get_deployment_credentials", "a") == {"api_key": "sk-test"}
    assert _call(router, "get_router_deployment_budget_limiter") is None
    assert _call(router, "should_include_deployment", "gpt", deployment("gpt", "a")) is True
    assert _call(router, "get_router_model_info", None, "gpt", id="a") == registry.router_model_info(
        None, "gpt", id="a"
    )


def test_settings_methods_answer_through_settings(make_core: MakeCore) -> None:
    router: Final = _configured(make_core)
    assert _call(router, "get_settings") == dict(router.settings().as_dict())
    assert _call(router, "get_available_guardrail", "g") == {
        "guardrail_name": "g",
        "litellm_params": {"guardrail": "x"},
    }
    with (
        pytest.raises(ValueError, match="No guardrail found with name: nope"),
        pytest.warns(DeprecationWarning, match=r"Router\.get_available_guardrail is deprecated"),
    ):
        router.get_available_guardrail("nope")


@pytest.mark.asyncio
async def test_health_and_selection_methods_answer_through_their_entry_points(make_core: MakeCore) -> None:
    router: Final = as_router(
        make_core(model_list=[deployment("gpt", "a"), deployment("pt", "p", use_in_pass_through=True)])
    )
    assert _call(router, "get_fully_blocked_model_names") == set()
    with pytest.warns(DeprecationWarning, match=r"Router\.async_get_fully_unhealthy_model_names is deprecated"):
        unhealthy: Final = await router.async_get_fully_unhealthy_model_names()
    assert unhealthy == set()
    with pytest.warns(DeprecationWarning, match=r"Router\.get_model_group_usage is deprecated"):
        usage: Final = await router.get_model_group_usage("gpt")
    assert usage == (None, None)
    with pytest.warns(DeprecationWarning, match=r"Router\.get_remaining_model_group_usage is deprecated"):
        remaining: Final = await router.get_remaining_model_group_usage("gpt")
    assert remaining == {}
    with pytest.warns(DeprecationWarning, match=r"Router\.async_get_healthy_deployments is deprecated"):
        healthy = await router.async_get_healthy_deployments("gpt", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the old untyped dicts
    assert isinstance(healthy, list)
    assert [picked_id(entry) for entry in healthy] == ["a"]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # see above
    assert picked_id(_call(router, "get_available_deployment_for_pass_through", "pt")) == "p"


def test_write_methods_go_through_apply(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    _call(router, "update_settings", allowed_fails=6)
    assert router.settings().allowed_fails == 6
    router.apply(ChangeSet())
    assert _call(router, "delete_deployment", "a") is not None
    assert router.registry().all() == []


def test_the_priority_split_answers_through_health(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")], default_priority=2))
    assert _call(router, "split_request_priority", "gpt", {"priority": 3, "x": 1}) == (3, {"x": 1})
    assert _call(router, "split_request_priority", "gpt", {"x": 1}) == (2, {"x": 1})
    assert router.health().priority_header_fields("gpt", {}, "5") == {"priority": 5}
    assert router.health().priority_header_fields("gpt", {}, None) == {}
    with pytest.raises(litellm.BadRequestError, match="priority must be an integer, got str 'bad'"):
        router.health().split_priority("gpt", {"priority": "bad"})


# -- plugin context, request logging -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_pre_routing_hook_answers_through_the_routing_context(make_core: MakeCore) -> None:
    router: Final = as_router(
        make_core(model_list=[deployment("small", "s"), deployment("big", "b"), _complexity_marker()])
    )
    request: Final[dict[str, object]] = {}
    with pytest.warns(DeprecationWarning, match=r"Router\.async_pre_routing_hook is deprecated"):
        response = await router.async_pre_routing_hook("smart", request, messages=MESSAGES)  # pyright: ignore[reportUnknownMemberType]  # the old signature's bare ``list``
    assert response is not None
    assert response.model == "small"
    decision: Final = as_str_mapping((as_str_mapping(request.get("metadata")) or {}).get("routing_decision")) or {}
    assert decision.get("router_model_name") == "smart"
    assert decision.get("routed_model") == "small"
    plain: Final[dict[str, object]] = {}
    with pytest.warns(DeprecationWarning, match="async_pre_routing_hook"):
        hooked: Final = await router.async_pre_routing_hook("small", plain)  # pyright: ignore[reportUnknownMemberType]  # see above
    assert hooked is None
    assert plain == {}


def _complexity_marker() -> dict[str, object]:  # mutable-ok: a model_list entry
    return {
        "model_name": "smart",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": {"SIMPLE": "small", "MEDIUM": "big", "COMPLEX": "big"}},
            "complexity_router_default_model": "big",
        },
        "model_info": {"id": "marker"},
    }


def test_log_retry_records_the_failure_on_the_request_metadata(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a", base_model="bm")]))
    kwargs: Final[dict[str, object]] = {
        "model": "gpt",
        "metadata": {"model_info": {"id": "a"}, "deployment": "openai/a", "attempted_retries": 1},
    }
    assert _call(router, "log_retry", kwargs, ValueError("boom")) is kwargs
    assert kwargs["metadata"] == {
        "model_info": {"id": "a"},
        "deployment": "openai/a",
        "attempted_retries": 1,
        "previous_models": (
            {
                "model_group": "gpt",
                "deployment_id": "a",
                "model": "openai/a",
                "base_model": "bm",
                "exception_type": "ValueError",
                "exception_string": "boom",
                "attempted_retries": 1,
            },
        ),
        "request_retry_count": 1,
    }
    _call(router, "log_retry", kwargs, KeyError("again"))
    metadata: Final = as_str_mapping(kwargs["metadata"]) or {}
    assert metadata["request_retry_count"] == 2
    previous: Final = metadata["previous_models"]
    assert isinstance(previous, tuple)
    assert len(previous) == 2  # pyright: ignore[reportUnknownArgumentType]  # an untyped tuple of breadcrumbs


def test_arming_the_read_prefetch_without_a_request_batch_does_nothing(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")], routing_strategy="usage-based-routing-v2"))
    assert _call(router, "arm_routing_read_prefetch", "gpt", {}) is None
    assert _call(router, "arm_routing_read_prefetch", "unknown-model") is None


def test_binding_to_request_logging_binds_through_observe(make_core: MakeCore) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    logging_obj: Final = Logging(
        model="gpt",
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    assert _call(router, "bind_to_request_logging", logging_obj) is None
    with (
        pytest.raises(AttributeError, match="add_dynamic_function_callbacks"),
        pytest.warns(DeprecationWarning, match=r"Router\.bind_to_request_logging is deprecated"),
    ):
        router.bind_to_request_logging(None)  # pyright: ignore[reportArgumentType]


# -- writes with a routing effect ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_assigned_routing_groups_route_from_the_next_call(make_core: MakeCore) -> None:
    from litellm.types.router import RoutingGroup

    router: Final = as_router(make_core(model_list=[deployment("gpt", "a"), deployment("claude", "c")]))
    with pytest.raises(Exception, match="grp"):
        await router.select("grp", {})
    write(
        router,
        "routing_groups",
        {"grp": RoutingGroup(group_name="grp", models=["claude"], routing_strategy="simple-shuffle")},
    )
    assert picked_id(await router.select("grp", {})) == "c"
    assert router.registry().routing_groups().keys() == {"grp"}


class _NarrowTo:
    def __init__(self, keep: str) -> None:
        self.keep: Final = keep

    async def run(self, context: RoutingContext) -> RoutingContext:
        context.candidate_models = [m for m in context.candidate_models if m == self.keep]
        return context


@pytest.mark.asyncio
async def test_a_plugin_added_to_an_empty_plugin_list_runs(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("g", "a"), deployment("g", "b")]))
    with pytest.warns(DeprecationWarning, match=r"Router\.routing_plugins is deprecated"):
        plugins: Final = router.routing_plugins
    plugins.append(_NarrowTo("openai/b"))
    for _ in range(4):
        assert picked_id(await router.select("g", {}, messages=MESSAGES)) == "b"  # pyright: ignore[reportCallIssue]
    plugins.clear()
    assert {picked_id(await router.select("g", {}, messages=MESSAGES)) for _ in range(40)} == {"a", "b"}  # pyright: ignore[reportCallIssue]


@pytest.mark.asyncio
async def test_an_assigned_plugin_list_runs(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("g", "a"), deployment("g", "b")]))
    write(router, "routing_plugins", [_NarrowTo("openai/a")])
    for _ in range(4):
        assert picked_id(await router.select("g", {}, messages=MESSAGES)) == "a"  # pyright: ignore[reportCallIssue]


def test_an_assigned_prompt_caching_pin_cache_is_the_one_flushed(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    pins: Final = DualCache()
    write(router, "prompt_caching_pin_cache", pins)
    assert read(router, "prompt_caching_pin_cache") is pins
    pins.in_memory_cache.set_cache("k", "v")  # pyright: ignore[reportUnknownMemberType]  # the in-memory cache is untyped
    router.flush_cache()  # pyright: ignore[reportAttributeAccessIssue]
    assert pins.in_memory_cache.get_cache("k") is None  # pyright: ignore[reportUnknownMemberType]  # see above


# -- pod resolution and cooldown_time --------------------------------------------------------------


class _TaggingDiscovery:
    """Stands in for KubernetesPodDiscovery: stamps which discovery resolved the pick."""

    def __init__(self, name: str) -> None:
        self.name: Final = name

    def resolve_deployment(self, deployment: object, request_kwargs: object) -> object:
        return {**(as_str_mapping(deployment) or {}), "resolved_by": self.name}

    async def async_resolve_deployment(self, deployment: object, request_kwargs: object) -> object:
        return self.resolve_deployment(deployment, request_kwargs)


@pytest.mark.asyncio
async def test_a_custom_strategy_resolves_pods_through_the_discovery_bound_when_registered(
    make_core: MakeCore,
) -> None:
    from litellm.types.router import CustomRoutingStrategyBase

    class Pick(CustomRoutingStrategyBase):
        async def async_get_available_deployment(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            model: str,
            messages: object = None,
            input: object = None,
            specific_deployment: object = False,
            request_kwargs: object = None,
        ) -> object:
            return deployment("gpt", "b")

        def get_available_deployment(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            model: str,
            messages: object = None,
            input: object = None,
            specific_deployment: object = False,
            request_kwargs: object = None,
        ) -> object:
            return deployment("gpt", "b")

    router: Final = as_router(make_core(model_list=[deployment("gpt", "a"), deployment("gpt", "b")]))
    router.apply(ChangeSet(assign={"kubernetes_pod_discovery": _TaggingDiscovery("first")}))
    assert (as_str_mapping(await router.select("gpt", {})) or {}).get("resolved_by") == "first"
    router.register_plugin(Pick())
    router.apply(ChangeSet(assign={"kubernetes_pod_discovery": _TaggingDiscovery("second")}))
    picked: Final = as_str_mapping(await router.select("gpt", {})) or {}
    assert picked_id(picked) == "b"
    assert picked.get("resolved_by") == "first"


def test_cooldown_time_reads_back_as_given(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")], cooldown_time=7))
    value: Final = read(router, "cooldown_time")
    assert value == 7 and type(value) is int
    assert router.settings().as_dict()["cooldown_time"] == 7
    assert type(router.settings().as_dict()["cooldown_time"]) is int
    write(router, "cooldown_time", 2.5)
    assert router.settings().cooldown_time == 2.5
    _call(router, "update_settings", cooldown_time="3")
    updated: Final = router.settings().cooldown_time
    assert updated == 3 and type(updated) is int
    default: Final = as_router(make_core(model_list=[deployment("gpt", "a")])).settings().cooldown_time
    assert default == 5 and type(default) is int


@pytest.mark.asyncio
async def test_the_old_healthy_listing_takes_the_retry_skip_list_off_the_callers_request(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a"), deployment("gpt", "b")]))
    request: Final[dict[str, object]] = {"_retry_skipped_deployment_ids": ("a",)}
    with pytest.warns(DeprecationWarning, match=r"Router\.async_get_healthy_deployments is deprecated"):
        healthy: Final = await router.async_get_healthy_deployments("gpt", request)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the old untyped dicts
    assert isinstance(healthy, list)
    assert [picked_id(entry) for entry in healthy] == ["b"]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # see above
    assert request == {}
