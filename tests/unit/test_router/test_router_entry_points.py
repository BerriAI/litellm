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


def test_router_cannot_be_subclassed():
    with pytest.raises(TypeError, match=r"^litellm\.Router cannot be subclassed"):

        class Sub(litellm.Router):
            pass


def test_core_classes_can_still_be_subclassed():
    from litellm.router_rust.core import RustCore

    class RustSub(RustCore):
        pass

    assert issubclass(RustSub, RustCore)


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


def test_registry_reads_without_a_core_forward(router):
    registry = router.registry()
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


def test_routing_context_forwards_to_the_router(router):
    from litellm.router_api import RoutingContext

    ctx = RoutingContext(router)
    assert ctx.registry().version == router.registry().version
    assert ctx.settings() is router.settings()
    assert isinstance(ctx.health(), type(router.health()))


def test_attribute_errors_about_other_objects_pass_through_unchanged(router):
    from litellm.router_deprecated import _public_attribute_error

    other = AttributeError("'dict' object has no attribute 'x'", name="x", obj={})
    assert _public_attribute_error(router, "_core_x", "x", other) is other


_MINTING_DEPLOYMENT = "dep-minting-org"
_OTHER_ORG_DEPLOYMENT = "dep-other-org"
_BOTH_ORGS = (_MINTING_DEPLOYMENT, _OTHER_ORG_DEPLOYMENT)
_GROUP_AFFINITY = ["encrypted_content_affinity"]


@pytest.mark.parametrize(
    "affinity, target_ids, reasoning_stays",
    [
        pytest.param({}, _BOTH_ORGS, False, id="no affinity, one target cannot decrypt"),
        pytest.param({}, (_MINTING_DEPLOYMENT,), True, id="no affinity, the only target minted it"),
        pytest.param(
            {"optional_pre_call_checks": _GROUP_AFFINITY}, _BOTH_ORGS, True, id="global affinity, origin among targets"
        ),
        pytest.param(
            {"optional_pre_call_checks": _GROUP_AFFINITY},
            (_OTHER_ORG_DEPLOYMENT,),
            False,
            id="global affinity, origin not among targets",
        ),
        pytest.param(
            {"optional_pre_call_checks": [], "model_group_affinity_config": {"pair": _GROUP_AFFINITY}},
            _BOTH_ORGS,
            True,
            id="affinity on for this group",
        ),
        pytest.param(
            {"optional_pre_call_checks": [], "model_group_affinity_config": {"other-group": _GROUP_AFFINITY}},
            _BOTH_ORGS,
            False,
            id="affinity on for another group only",
        ),
    ],
)
def test_decryptable_reasoning_keeps_encrypted_reasoning_only_when_the_router_sends_it_to_a_deployment_that_decrypts_it(
    affinity, target_ids, reasoning_stays
):
    from litellm.responses.utils import ResponsesAPIRequestUtils

    r = Router(
        model_list=[
            {
                "model_name": "pair",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": api_base, "api_key": "sk-pair"},
                "model_info": {"id": model_id},
            }
            for api_base, model_id in (
                ("https://minting-org.test/v1", _MINTING_DEPLOYMENT),
                ("https://other-org.test/v1", _OTHER_ORG_DEPLOYMENT),
            )
        ],
        **affinity,
    )
    summary = [{"type": "summary_text", "text": "multiply 17 by 23"}]
    user_turn = {"type": "message", "role": "user", "content": "What is 17*23?"}
    reasoning = {
        "type": "reasoning",
        "id": "rs_minted",
        "encrypted_content": ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id(
            "gAAAAA-blob", _MINTING_DEPLOYMENT
        ),
        "summary": summary,
    }
    request_input = [user_turn, reasoning]
    targets = tuple(r.registry().deployment(model_id) for model_id in target_ids)
    try:
        kept = r.registry().decryptable_reasoning("pair", request_input, targets)
    finally:
        r.retire()

    readable_only = {"type": "reasoning", "summary": summary}
    assert kept == (user_turn, reasoning if reasoning_stays else readable_only)
    assert request_input == [user_turn, reasoning], "the caller's input must not change"


# Router-family helpers whose Router parameter became a RoutingContext keep the old spellings.


def test_renamed_router_keyword_still_works_and_warns(router):
    from litellm.router_utils.router_callbacks.track_deployment_metrics import (
        get_deployment_failures_for_current_minute,
        increment_deployment_failures_for_current_minute,
    )

    deployment_id = router.registry().names(kind="ids")[0]
    label = (
        "litellm.router_utils.router_callbacks.track_deployment_metrics."
        "increment_deployment_failures_for_current_minute"
    )
    before = DEPRECATED_ROUTER_MEMBER_USES[label]

    with pytest.warns(DeprecationWarning, match=r"litellm_router_instance= is deprecated; pass ctx="):
        increment_deployment_failures_for_current_minute(litellm_router_instance=router, deployment_id=deployment_id)

    assert DEPRECATED_ROUTER_MEMBER_USES[label] == before + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_deployment_failures_for_current_minute(ctx=router._core_ctx, deployment_id=deployment_id) == 1


def test_router_passed_where_the_context_goes_is_converted_with_a_warning(router):
    from litellm.router_utils.cooldown_handlers import get_cooldown_deployments

    with pytest.warns(DeprecationWarning, match="passing a Router is deprecated"):
        assert get_cooldown_deployments(router, None) == []
    with pytest.warns(DeprecationWarning, match="a Router as ctx= is deprecated"):
        assert get_cooldown_deployments(ctx=router, parent_otel_span=None) == []


def test_renamed_router_keyword_alongside_the_new_one_is_rejected(router):
    from litellm.router_utils.cooldown_handlers import get_cooldown_deployments

    with pytest.raises(TypeError, match="got both litellm_router_instance= and ctx="):
        get_cooldown_deployments(ctx=router._core_ctx, litellm_router_instance=router, parent_otel_span=None)


def test_renamed_router_param_maps_to_a_differently_named_context(router):
    from litellm.router_api import RoutingContext
    from litellm.router_deprecated import renamed_router_param

    @renamed_router_param("litellm_router", new="litellm_routing_ctx")
    def helper(*, litellm_routing_ctx: RoutingContext) -> RoutingContext:
        return litellm_routing_ctx

    with pytest.warns(DeprecationWarning, match="litellm_router= is deprecated; pass litellm_routing_ctx="):
        assert helper(litellm_router=router) is router._core_ctx
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert helper(litellm_routing_ctx=router._core_ctx) is router._core_ctx


def test_renamed_router_attribute_reads_and_writes_the_context(router):
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    check = EncryptedContentAffinityCheck(ctx=router._core_ctx)
    with pytest.warns(DeprecationWarning, match=r"EncryptedContentAffinityCheck\.router: reading this attribute"):
        assert check.router is router._core_ctx

    other = _router()
    with pytest.warns(DeprecationWarning, match="setting this attribute is deprecated; set .ctx"):
        check.router = other
    assert check.ctx is other._core_ctx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other.retire()
