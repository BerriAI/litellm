"""
Router entry points and DeprecatedRouterMembers.

Every member that moved into DeprecatedRouterMembers must return what its entry point returns,
warn once per access from outside the router, and count the access. Internal access (the router's
own modules) must stay silent.
"""

import asyncio
import dataclasses
import warnings

import pytest

import litellm
from litellm import Router
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


def test_update_settings_through_apply(router):
    router.apply(ChangeSet(settings={"cooldown_time": "11", "not_allowed": 1}))
    assert router.settings().cooldown_time == 11


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


def test_alias_attribute_writes_the_same_storage(router):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        replacement = {}
        router._routing_groups = replacement
        assert router.routing_groups is replacement
    assert router.registry().routing_groups() is replacement
