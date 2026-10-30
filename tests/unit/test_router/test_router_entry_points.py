"""
Router entry points and DeprecatedRouterMembers.

Every member that moved into DeprecatedRouterMembers must return what its entry point returns,
warn once per access from outside the router, and count the access. Internal access (the router's
own modules) must stay silent.
"""

import asyncio
import copy
import dataclasses
import warnings

import httpx
import pytest

import litellm
from litellm import Router
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.router_api import SETTING_FIELDS, ChangeSet, RouterSettings
from litellm.router_deprecated import DEPRECATED_ROUTER_MEMBER_USES, DeprecatedRouterMembers
from litellm.types.router import Deployment


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-a"},
                "model_info": {"id": "dep-a", "access_groups": ["grp"]},
            },
            {
                "model_name": "gpt",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-b"},
                "model_info": {"id": "dep-b"},
            },
            {
                "model_name": "team-internal",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-t"},
                "model_info": {"id": "dep-team", "team_id": "team-1", "team_public_model_name": "team-gpt"},
            },
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-w"},
                "model_info": {"id": "dep-wild"},
            },
        ],
        model_group_alias={"alias-gpt": "gpt"},
        fallbacks=[{"gpt": ["team-internal"]}],
        cooldown_time=7,
    )


@pytest.fixture
def router():
    r = _router()
    yield r
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.retire()


# (old member, how to read it through the old name, how to read it through the entry point)
READS = [
    ("model_list", lambda r: r.model_list, lambda r: r.registry().all()),
    ("model_names", lambda r: r.model_names, lambda r: r.registry().names(kind="group_set")),
    ("deployment_names", lambda r: r.deployment_names, lambda r: r.registry().names(kind="deployment_names")),
    ("team_public_model_names", lambda r: r.team_public_model_names, lambda r: r.registry().names(kind="team_public")),
    (
        "model_name_to_deployment_indices",
        lambda r: r.model_name_to_deployment_indices,
        lambda r: r.registry().index("model_name"),
    ),
    (
        "team_model_to_deployment_indices",
        lambda r: r.team_model_to_deployment_indices,
        lambda r: r.registry().index("team_model"),
    ),
    ("model_group_alias", lambda r: r.model_group_alias, lambda r: r.registry().aliases()),
    ("pattern_router", lambda r: r.pattern_router, lambda r: r.registry().patterns()),
    ("team_pattern_routers", lambda r: r.team_pattern_routers, lambda r: r.registry().patterns(team=True)),
    ("routing_groups", lambda r: r.routing_groups, lambda r: r.registry().routing_groups()),
    ("auto_routers", lambda r: r.auto_routers, lambda r: r.registry().plugins("auto")),
    ("complexity_routers", lambda r: r.complexity_routers, lambda r: r.registry().plugins("complexity")),
    ("quality_routers", lambda r: r.quality_routers, lambda r: r.registry().plugins("quality")),
    ("adaptive_routers", lambda r: r.adaptive_routers, lambda r: r.registry().plugins("adaptive")),
    ("routing_plugins", lambda r: r.routing_plugins, lambda r: r.registry().plugins("routing")),
    (
        "prompt_caching_pin_cache",
        lambda r: r.prompt_caching_pin_cache,
        lambda r: r.registry().plugins("prompt_caching_pins"),
    ),
    ("cache", lambda r: r.cache, lambda r: r.health().store),
    ("cooldown_cache", lambda r: r.cooldown_cache, lambda r: r.health().cooldowns),
    ("health_state_cache", lambda r: r.health_state_cache, lambda r: r.health().health_checks),
    ("deployment_latency_map", lambda r: r.deployment_latency_map, lambda r: r.health().latency),
    ("get_deployment", lambda r: r.get_deployment("dep-a"), lambda r: r.registry().deployment("dep-a")),
    ("has_model_id", lambda r: r.has_model_id("dep-b"), lambda r: r.registry().has_deployment("dep-b")),
    ("get_model_info", lambda r: r.get_model_info("dep-a"), lambda r: r.registry().deployment_dict("dep-a")),
    ("get_model_list", lambda r: r.get_model_list("gpt"), lambda r: r.registry().deployments("gpt")),
    ("get_model_list_all", lambda r: r.get_model_list(), lambda r: r.registry().deployments()),
    ("get_model_list_alias", lambda r: r.get_model_list("alias-gpt"), lambda r: r.registry().deployments("alias-gpt")),
    (
        "get_all_deployments",
        lambda r: r.get_all_deployments("gpt"),
        lambda r: r.registry().deployments("gpt", source="all"),
    ),
    (
        "config_deployments",
        lambda r: list(r.config_deployments()),
        lambda r: list(r.registry().deployments(source="config")),
    ),
    (
        "deployments_named",
        lambda r: r.deployments_named("gpt"),
        lambda r: r.registry().deployments("gpt", source="named"),
    ),
    (
        "get_model_list_from_model_alias",
        lambda r: r.get_model_list_from_model_alias("alias-gpt"),
        lambda r: r.registry().deployments("alias-gpt", source="alias"),
    ),
    (
        "get_deployment_by_model_group_name",
        lambda r: r.get_deployment_by_model_group_name("gpt"),
        lambda r: r.registry().deployments("gpt", source="first_in_group"),
    ),
    ("get_model_names", lambda r: r.get_model_names(), lambda r: r.registry().names()),
    (
        "get_model_names_team",
        lambda r: r.get_model_names(team_id="team-1"),
        lambda r: r.registry().names(team_id="team-1"),
    ),
    ("get_model_ids", lambda r: r.get_model_ids("gpt"), lambda r: r.registry().names("gpt", kind="ids")),
    (
        "get_model_ids_all",
        lambda r: r.get_model_ids(exclude_team_models=True),
        lambda r: r.registry().names(kind="ids", exclude_team_models=True),
    ),
    (
        "get_model_from_alias",
        lambda r: r.get_model_from_alias("alias-gpt"),
        lambda r: r.registry().resolve("alias-gpt", kind="alias"),
    ),
    (
        "is_recognized_model",
        lambda r: r.is_recognized_model("gpt"),
        lambda r: r.registry().resolve("gpt", kind="recognized"),
    ),
    (
        "routable_model_group",
        lambda r: r.routable_model_group("alias-gpt"),
        lambda r: r.registry().resolve("alias-gpt"),
    ),
    (
        "map_team_model",
        lambda r: r.map_team_model("team-gpt", "team-1"),
        lambda r: r.registry().resolve("team-gpt", kind="team_model", team_id="team-1"),
    ),
    (
        "get_model_group",
        lambda r: r.get_model_group("dep-a"),
        lambda r: r.registry().resolve("dep-a", kind="group_of_id"),
    ),
    (
        "get_search_tool_support",
        lambda r: r.get_search_tool_support("gpt"),
        lambda r: r.registry().resolve("gpt", kind="search_tool_support"),
    ),
    ("get_model_access_groups", lambda r: r.get_model_access_groups(), lambda r: r.registry().access_groups()),
    (
        "get_deployment_credentials",
        lambda r: r.get_deployment_credentials("dep-a"),
        lambda r: r.registry().credentials("dep-a"),
    ),
    (
        "get_deployment_credentials_with_provider",
        lambda r: r.get_deployment_credentials_with_provider("dep-a"),
        lambda r: r.registry().credentials("dep-a", form="with_provider"),
    ),
    (
        "get_credential_deployment",
        lambda r: r.get_credential_deployment("dep-a"),
        lambda r: r.registry().credentials("dep-a", form="deployment"),
    ),
    ("get_configured_mode", lambda r: r.get_configured_mode("gpt"), lambda r: r.registry().group("gpt", field="mode")),
    (
        "get_configured_token_limits",
        lambda r: r.get_configured_token_limits("gpt"),
        lambda r: r.registry().group("gpt", field="token_limits"),
    ),
    (
        "get_fully_blocked_model_names",
        lambda r: r.get_fully_blocked_model_names(),
        lambda r: r.health().fully_blocked_model_names(),
    ),
    ("get_settings", lambda r: r.get_settings(), lambda r: r.settings().as_dict()),
    ("fallbacks", lambda r: r.fallbacks, lambda r: r.settings().fallbacks),
    ("cooldown_time", lambda r: r.cooldown_time, lambda r: r.settings().cooldown_time),
    ("routing_strategy", lambda r: r.routing_strategy, lambda r: r.settings().routing_strategy),
]


@pytest.mark.parametrize("name, old, new", READS, ids=[r[0] for r in READS])
def test_deprecated_read_matches_entry_point(router, name, old, new):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_value = old(router)
    new_value = new(router)
    if isinstance(old_value, (list, dict, set)) or hasattr(old_value, "__dict__"):
        assert old_value == new_value
    else:
        assert old_value == new_value
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), name


def test_live_objects_not_copies(router):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert router.model_list is router.registry().all()
        assert router.fallbacks is router.settings().fallbacks
        assert router.cache is router.health().store
        assert router.model_group_alias is router.registry().aliases()


def test_assignment_goes_through_apply(router):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router.fallbacks = [{"gpt": ["openai/gpt-4.1"]}]
        assert router.settings().fallbacks == [{"gpt": ["openai/gpt-4.1"]}]
        router.search_tools = [{"search_tool_name": "s", "litellm_params": {"search_provider": "tavily"}}]
        assert router.settings().search_tools[0]["search_tool_name"] == "s"
    router.apply(ChangeSet(assign={"cooldown_time": 3}))
    assert router.settings().cooldown_time == 3
    with pytest.raises(ValueError, match="not assignable router settings"):
        router.apply(ChangeSet(assign={"not_a_setting": 1}))


def test_apply_upsert_and_delete_bump_version(router):
    v0 = router.registry().version
    added = Deployment(
        model_name="new-group",
        litellm_params={"model": "openai/gpt-4o-mini", "api_key": "sk-n"},
        model_info={"id": "dep-new"},
    )
    result = router.apply(ChangeSet(upsert=[added]))
    assert result.version == v0 + 1
    assert router.registry().has_deployment("dep-new")
    assert router.registry().deployments("new-group")[0]["model_info"]["id"] == "dep-new"
    result = router.apply(ChangeSet(delete=["dep-new"]))
    assert result.version == v0 + 2
    assert result.deleted[0] is not None
    assert not router.registry().has_deployment("dep-new")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert router.upsert_deployment(added) is added
        assert router.upsert_deployment(added) is None  # unchanged
        assert router.delete_deployment("dep-new") is not None


def test_old_deployment_writes_also_advance_the_version(router):
    v0 = router.registry().version
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router.add_deployment(
            Deployment(
                model_name="added",
                litellm_params={"model": "openai/gpt-4o-mini", "api_key": "sk-x"},
                model_info={"id": "dep-added"},
            )
        )
        v1 = router.registry().version
        router.set_model_list([{"model_name": "only", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-o"}}])
    assert v0 < v1 < router.registry().version
    assert router.registry().names() == ["only"]
    v2 = router.registry().version
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router.model_list = []
    assert router.registry().version > v2


def test_update_settings_through_apply(router):
    router.apply(ChangeSet(settings={"cooldown_time": "11", "not_allowed": 1}))
    assert router.settings().cooldown_time == 11


def test_aapply_applies_changes_then_refreshes_model_info(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    router = Router(
        model_list=[
            {
                "model_name": "local",
                "litellm_params": {
                    "model": "hosted_vllm/local-model",
                    "api_base": "https://capacity.test/v1",
                    "api_key": "local-key",
                },
                "model_info": {"id": "dep-local"},
            }
        ]
    )
    with pytest.raises(TypeError, match="needs aapply"):
        router.apply(ChangeSet(refresh_model_info=True))

    async def refresh() -> None:
        handler = AsyncHTTPHandler()
        await handler.client.aclose()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": "local-model", "max_model_len": 4096}]})
            )
        ) as client:
            handler.client = client
            await router.aapply(ChangeSet(assign={"cooldown_time": 4}, refresh_model_info=True, refresh_client=handler))

    asyncio.run(refresh())
    assert router.settings().cooldown_time == 4
    assert router.registry().group("local", field="token_limits") == (4096, 4096)


def test_settings_snapshot_is_reused_until_a_setting_is_written(router):
    first = router.settings()
    assert router.settings() is first
    router.apply(ChangeSet(assign={"cooldown_time": 7}))
    assert router.settings() is not first
    assert router.settings().cooldown_time == 7
    router.apply(ChangeSet(settings={"cooldown_time": "8"}))
    assert router.settings().cooldown_time == 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router.fallbacks = [{"gpt": ["team-gpt"]}]
    assert router.settings().fallbacks == [{"gpt": ["team-gpt"]}]
    router._core_fallbacks = [{"gpt": ["other"]}]  # the router's own writes invalidate it too
    assert router.settings().fallbacks == [{"gpt": ["other"]}]
    router.settings().fallbacks.append({"x": ["y"]})  # the snapshot holds the router's own objects
    assert router._core_fallbacks[-1] == {"x": ["y"]}
    unrelated = router.settings()
    router._core_model_list = list(router._core_model_list)
    assert router.settings() is unrelated


def test_deleting_a_setting_shows_as_unset(router):
    router.apply(ChangeSet(assign={"cooldown_time": 3}))
    assert router.settings().is_set("cooldown_time")
    del router._core_cooldown_time
    assert router.settings().is_set("cooldown_time") is (hasattr(router, "_core_cooldown_time"))


def test_settings_snapshot_is_frozen(router):
    snapshot = router.settings()
    assert isinstance(snapshot, RouterSettings)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.cooldown_time = 1  # type: ignore[misc]
    assert set(SETTING_FIELDS) <= set(RouterSettings.__dataclass_fields__)


def test_unset_setting_raises_attribute_error_through_old_name():
    r = _router()
    try:
        del r.__dict__["_core_kubernetes_pod_discovery"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert getattr(r, "kubernetes_pod_discovery", "missing") == "missing"
        assert r.settings().is_set("kubernetes_pod_discovery") is False
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.retire()


def test_unset_attribute_error_names_the_public_attribute(router):
    # Reading (or deleting) an old attribute the router never set raises what reading it raised before.
    router.__dict__.pop("_core_slack_alerting_logger", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AttributeError) as read:
            _ = router.slack_alerting_logger
        with pytest.raises(AttributeError) as deleted:
            del router.slack_alerting_logger
        del router.__dict__["_core_kubernetes_pod_discovery"]
        with pytest.raises(AttributeError) as setting:
            _ = router.kubernetes_pod_discovery
    for caught, name in ((read, "slack_alerting_logger"), (setting, "kubernetes_pod_discovery")):
        assert str(caught.value) == f"'Router' object has no attribute '{name}'"
        assert caught.value.name == name
        assert caught.value.obj is router
    # A failed del sets no name/obj in Python either.
    assert str(deleted.value) == "'Router' object has no attribute 'slack_alerting_logger'"


def test_bind_to_request_logging_none_raises_as_before(router):
    # The old member bound unconditionally, so None fails on the first callback registration.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AttributeError, match="add_dynamic_function_callbacks"):
            router.bind_to_request_logging(None)


def test_select_picks_like_async_get_available_deployment(router):
    picked = asyncio.run(router.select("gpt", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = asyncio.run(router.async_get_available_deployment(model="gpt", request_kwargs={}))
    assert picked["model_name"] == old["model_name"] == "gpt"
    healthy = asyncio.run(router.select("gpt", {}, kind="healthy"))
    assert {d["model_info"]["id"] for d in healthy} == {"dep-a", "dep-b"}


def test_counter_and_internal_silence(router):
    before = DEPRECATED_ROUTER_MEMBER_USES["get_model_ids"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        router.get_model_ids()
        # Router internals call the core names; a completion must not warn.
        asyncio.run(router.acompletion(model="gpt", messages=[{"role": "user", "content": "hi"}], mock_response="ok"))
    assert DEPRECATED_ROUTER_MEMBER_USES["get_model_ids"] == before + 1
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning) and "Router." in str(w.message)]
    assert [str(w.message).split(" ")[0] for w in deprecations] == ["Router.get_model_ids"]
    assert deprecations[0].filename == __file__


def test_every_deprecated_member_names_a_replacement():
    import inspect

    from litellm.router_deprecated import _DeprecatedAttribute

    for name, member in vars(DeprecatedRouterMembers).items():
        if name.startswith("__"):
            continue
        if isinstance(member, _DeprecatedAttribute):
            assert member.replacement
            continue
        func = member.__func__ if isinstance(member, staticmethod) else member
        if callable(func):
            assert "_note(" in inspect.getsource(func), name


def test_generate_model_id_is_the_module_function():
    from litellm.router_api import deployment_id

    params = {"model": "openai/gpt-4o", "api_key": "k", "extra": {"a": 1}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert Router.generate_model_id("g", params) == deployment_id("g", params)


def test_subclass_override_of_an_old_member_is_still_called():
    calls = []

    class Sub(litellm.Router):
        async def async_pre_routing_hook(
            self, model, request_kwargs, messages=None, input=None, specific_deployment=False
        ):
            calls.append(model)
            return await super().async_pre_routing_hook(model, request_kwargs, messages, input, specific_deployment)

    r = Sub(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x", "mock_response": "ok"},
            }
        ]
    )
    response = asyncio.run(r.acompletion(model="m", messages=[{"role": "user", "content": "hi"}]))
    assert response.choices[0].message.content == "ok"
    assert calls == ["m"]


def test_direct_call_of_a_subclass_override_runs_it_once():
    calls = []

    class Sub(litellm.Router):
        def get_model_access_groups(self, model_name=None, model_access_group=None, team_id=None):
            calls.append(model_name)
            return super().get_model_access_groups(model_name, model_access_group, team_id)

    r = Sub(model_list=[{"model_name": "m", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"}}])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert r.get_model_access_groups("m") == {}
    assert calls == ["m"]
    assert asyncio.run(asyncio.to_thread(r.registry().access_groups, model_name="m")) == {}
    assert calls == ["m", "m"]


def test_subclass_staticmethod_override_is_still_consulted():
    calls = []

    class Sub(litellm.Router):
        @staticmethod
        def is_deployment_blocked(deployment):
            calls.append(deployment.model_info.id)
            return True

    r = Sub(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"},
                "model_info": {"id": "d1"},
            }
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert r.get_deployment_credentials(model_id="d1") is None
    assert calls == ["d1"]


def test_private_alias_skips_a_subclass_override_of_the_public_name():
    calls = []

    class Sub(litellm.Router):
        def get_all_deployments(self, *args, **kwargs):
            calls.append("get_all_deployments")
            return super().get_all_deployments(*args, **kwargs)

    r = Sub(model_list=[{"model_name": "m", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"}}])
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(r._get_all_deployments(model_name="m")) == 1
    assert calls == []


def test_old_name_writes_and_deletes_warn(router):
    with pytest.warns(DeprecationWarning, match=r"^Router\.fallbacks is deprecated"):
        router.fallbacks = []
    with pytest.warns(DeprecationWarning, match=r"^Router\.fallbacks is deprecated"):
        del router.fallbacks


def test_alias_attribute_writes_the_same_storage(router):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        replacement = {}
        router._routing_groups = replacement
        assert router.routing_groups is replacement
    assert router.registry().routing_groups() is replacement


def test_core_members_keep_their_old_names_in_errors():
    from litellm.router import _name_core_members_after_old_names

    _name_core_members_after_old_names()

    assert Router._core_get_deployment_model_info.__qualname__ == "Router.get_deployment_model_info"
    assert Router._core_get_deployment.__qualname__ == "Router.get_deployment"


def test_wrong_arity_errors_still_name_router(router):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match=r"^Router\.get_deployment\(\) takes 2 positional arguments"):
            router.get_deployment("dep-a", "extra")  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError, match=r"^Router\.get_available_guardrail\(\) missing 1 required"):
            router.get_available_guardrail()  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError, match=r"^litellm\.router\.Router\.has_model_id\(\) argument after \*\*"):
            router.has_model_id(**None)  # pyright: ignore[reportCallIssue, reportArgumentType]
        with pytest.raises(TypeError, match=r"^Router\.is_deployment_blocked\(\) missing 1 required"):
            router.is_deployment_blocked()  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError, match=r"^Router\.get_deployment_model_info\(\) missing 2 required"):
            router.cached_deployment_model_info()  # pyright: ignore[reportCallIssue]


_SENTINEL = object()

# (entry point call, Router core member it forwards to, the arguments the core member must get)
FORWARDS = [
    (
        lambda r: r.registry().deployment_model_info("dep-a", "gpt", cached=True),
        "_core_cached_deployment_model_info",
        (("dep-a", "gpt"), {}),
    ),
    (
        lambda r: r.registry().deployment_model_info("dep-a", "gpt"),
        "_core_get_deployment_model_info",
        ((), {"model_id": "dep-a", "model_name": "gpt"}),
    ),
    (
        lambda r: r.registry().router_model_info(None, "gpt", id="dep-a"),
        "_core_get_router_model_info",
        ((), {"deployment": None, "received_model_name": "gpt", "id": "dep-a"}),
    ),
    (lambda r: r.registry().discovered_model_info("dep-a"), "_core_get_discovered_model_info", (("dep-a",), {})),
    (
        lambda r: r.registry().deployments("gpt", source="routing_groups"),
        "_core_get_model_list_from_routing_groups",
        ((), {"model_name": "gpt"}),
    ),
    (
        lambda r: r.registry().deployments("gpt", source="routed_group", team_id="t"),
        "_core_get_model_list_of_routed_group",
        (("gpt", "t"), {}),
    ),
    (
        lambda r: r.registry().deployments("gpt", source="wildcard", team_id="t"),
        "_core_wildcard_route_deployments",
        (("gpt", "t"), {}),
    ),
    (
        lambda r: r.registry().deployments("gpt", source="routable", request={"a": 1}, call_type="aembedding"),
        "_core_get_routable_deployments",
        (("gpt", {"a": 1}, "aembedding"), {}),
    ),
    (
        lambda r: r.registry().deployments("gpt", source="for_request"),
        "_core_deployments_for_request",
        (("gpt", {}), {}),
    ),
    (
        lambda r: r.registry().includes({"model_name": "gpt"}, model_name="gpt", team_id="t"),
        "_core_should_include_deployment",
        ((), {"model_name": "gpt", "model": {"model_name": "gpt"}, "team_id": "t"}),
    ),
    (
        lambda r: r.registry().names("gpt", kind="litellm_models", team_id="t"),
        "_core_resolved_litellm_models",
        (("gpt",), {"team_id": "t"}),
    ),
    (
        lambda r: r.registry().names("gpt", kind="litellm_base_models"),
        "_core_resolved_litellm_models",
        (("gpt",), {"team_id": None, "use_base_model": True}),
    ),
    (
        lambda r: r.registry().names("gpt", kind="wildcard", team_id="t"),
        "_core_wildcard_routed_models",
        (("gpt", "t"), {}),
    ),
    (
        lambda r: r.registry().names("gpt", kind="route_candidates"),
        "_core_get_candidate_model_ids_for_route",
        (("gpt",), {"team_id": None}),
    ),
    (
        lambda r: r.registry().resolve("dep-a", kind="from_id", custom_llm_provider="openai"),
        "_core_resolve_model_name_from_model_id",
        (("dep-a",), {"custom_llm_provider": "openai"}),
    ),
    (
        lambda r: r.registry().resolve("rg", kind="routing_group_alternatives"),
        "_core_routing_group_has_alternatives",
        (("rg",), {}),
    ),
    (
        lambda r: r.registry().resolve("gpt", kind="served", served_model="gpt-4o"),
        "_core_resolve_served_model_name",
        (("gpt", "gpt-4o"), {}),
    ),
    (
        lambda r: r.registry().resolve("gpt", kind="upstream", team_id="t"),
        "_core_get_routable_upstream_model",
        (("gpt",), {"team_id": "t"}),
    ),
    (
        lambda r: r.registry().resolve("gpt", kind="search_tool_support", team_id="t"),
        "_core_get_search_tool_support",
        (("gpt",), {"team_id": "t"}),
    ),
    (
        lambda r: r.registry().resolve("grp", kind="wildcard_access_group"),
        "_core_is_model_access_group_for_wildcard_route",
        ((), {"model_access_group": "grp"}),
    ),
    (
        lambda r: r.registry().resolve("dep-a", kind="team_model_alternatives"),
        "_core_team_model_has_alternatives",
        (("dep-a",), {}),
    ),
    (
        lambda r: r.registry().group("gpt", include_hidden=True, team_id="t"),
        "_core_get_model_group_info",
        (("gpt",), {"include_hidden": True, "team_id": "t"}),
    ),
    (lambda r: r.registry().group("gpt", field="cached_info"), "_core_cached_model_group_info", (("gpt",), {})),
    (lambda r: r.registry().group("gpt", field="display_name"), "_core_get_configured_display_name", (("gpt",), {})),
    (
        lambda r: r.registry().group("gpt", field="service_tiers", team_id="t"),
        "_core_get_configured_service_tiers",
        (("gpt",), {"team_id": "t"}),
    ),
    (lambda r: r.registry().group("gpt", field="listing_info"), "_core_get_model_listing_info", (("gpt",), {})),
    (lambda r: r.registry().plugins("budget_limiter"), "_core_get_router_deployment_budget_limiter", ((), {})),
    (lambda r: r.settings().allowed_fails_for(ValueError("x")), "_core_get_allowed_fails_from_policy", None),
    (lambda r: r.settings().guardrail("g"), "_core_get_available_guardrail", (("g",), {})),
    (lambda r: r.health().team_ptu_ceiling("t", "gpt"), "_core_team_ptu_ceiling", (("t", "gpt"), {})),
    (
        lambda r: r.health().split_priority("gpt", {"priority": 1}),
        "_core_split_request_priority",
        (("gpt", {"priority": 1}), {}),
    ),
]


@pytest.mark.parametrize("call, core, expected", FORWARDS, ids=[f[1] for f in FORWARDS])
def test_entry_point_forwards_to_its_core_member(router, monkeypatch, call, core, expected):
    from unittest.mock import MagicMock

    mock = MagicMock(return_value=_SENTINEL)
    monkeypatch.setattr(router, core, mock)
    assert call(router) is _SENTINEL
    if expected is not None:
        assert mock.call_args == expected
    else:
        mock.assert_called_once()


ASYNC_FORWARDS = [
    (lambda r: r.health().fully_unhealthy_model_names(), "_core_async_get_fully_unhealthy_model_names", ((), {})),
    (lambda r: r.health().usage("gpt"), "_core_get_model_group_usage", ((), {"model_group": "gpt"})),
    (
        lambda r: r.health().usage("gpt", kind="remaining"),
        "_core_get_remaining_model_group_usage",
        ((), {"model_group": "gpt"}),
    ),
]


@pytest.mark.parametrize("call, core, expected", ASYNC_FORWARDS, ids=[f[1] for f in ASYNC_FORWARDS])
def test_async_entry_point_forwards_to_its_core_member(router, monkeypatch, call, core, expected):
    from unittest.mock import AsyncMock

    mock = AsyncMock(return_value=_SENTINEL)
    monkeypatch.setattr(router, core, mock)
    assert asyncio.run(call(router)) is _SENTINEL
    assert mock.call_args == expected


def test_registry_reads_without_a_core_forward(router):
    registry = router.registry()
    assert registry.zero_cost() is getattr(router, "_zero_cost_cache", None)
    assert registry.includes(router.registry().deployment("dep-a").model_dump(), team_id=None) is True


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda r: r.registry().deployments(source="named"), TypeError),
        (lambda r: r.registry().deployments("gpt", source="nope"), ValueError),
        (lambda r: r.registry().index("nope"), ValueError),
        (lambda r: r.registry().names(kind="litellm_models"), TypeError),
        (lambda r: r.registry().names("gpt", kind="nope"), ValueError),
        (lambda r: r.registry().resolve("team-gpt", kind="team_model"), TypeError),
        (lambda r: r.registry().resolve(None, kind="alias"), TypeError),
        (lambda r: r.registry().resolve("gpt", kind="nope"), ValueError),
        (lambda r: r.registry().group("gpt", field="nope"), ValueError),
        (lambda r: r.registry().credentials("dep-a", form="nope"), ValueError),
        (lambda r: r.registry().plugins("nope"), ValueError),
        (lambda r: asyncio.run(r.health().usage("gpt", kind="nope")), ValueError),
        (lambda r: RouterSettings().as_dict(), AttributeError),
    ],
)
def test_entry_points_reject_unknown_kinds_and_missing_names(router, call, error):
    with pytest.raises(error):
        call(router)


def test_health_is_blocked_and_the_old_static_name_agree():
    paused = Deployment(
        model_name="m",
        litellm_params={"model": "openai/gpt-4o-mini"},  # pyright: ignore[reportArgumentType]
        model_info={"id": "d1", "blocked": True},  # pyright: ignore[reportArgumentType]
    )
    from litellm.router_api import RouterHealth

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert Router.is_deployment_blocked(paused) is RouterHealth.is_blocked(paused)


def test_old_members_refuse_objects_that_are_not_routers():
    from litellm.router_deprecated import _as_router

    with pytest.raises(TypeError, match=r"is not a litellm\.Router"):
        _as_router(object())  # pyright: ignore[reportArgumentType]


def test_direct_call_of_an_async_subclass_override_runs_it_once():
    calls = []

    class Sub(litellm.Router):
        async def async_pre_routing_hook(
            self, model, request_kwargs, messages=None, input=None, specific_deployment=False
        ):
            calls.append(model)
            return await super().async_pre_routing_hook(model, request_kwargs, messages, input, specific_deployment)

    r = Sub(model_list=[{"model_name": "m", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"}}])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        asyncio.run(r.async_pre_routing_hook("m", {}))
    assert calls == ["m"]


def test_routing_context_forwards_to_the_router(router, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.router_api import RoutingContext

    ctx = RoutingContext(router)
    assert ctx.registry().version == router.registry().version
    assert ctx.settings() is router.settings()
    assert isinstance(ctx.health(), type(router.health()))
    for name in (
        "_pre_call_checks",
        "_count_pre_call_check_tokens",
        "_update_kwargs_before_fallbacks",
        "_core_arm_routing_read_prefetch",
        "_core_log_retry",
    ):
        monkeypatch.setattr(router, name, MagicMock(return_value=_SENTINEL))
    monkeypatch.setattr(router, "_core_async_pre_routing_hook", AsyncMock(return_value=_SENTINEL))
    assert ctx.pre_call_checks("gpt", []) is _SENTINEL
    assert ctx.count_pre_call_check_tokens(None, "hi") is _SENTINEL
    assert ctx.update_kwargs_before_fallbacks("gpt", {}) is _SENTINEL
    assert ctx.arm_read_prefetch("gpt") is _SENTINEL
    assert asyncio.run(ctx.pre_routing_hook("gpt", {})) is _SENTINEL
    assert ctx.log_retry({}, ValueError("x")) is _SENTINEL


def test_attribute_errors_about_other_objects_pass_through_unchanged(router):
    from litellm.router_deprecated import _public_attribute_error

    other = AttributeError("'dict' object has no attribute 'x'", name="x", obj={})
    assert _public_attribute_error(router, "_core_x", "x", other) is other
