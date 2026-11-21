"""
Characterization of the settings only the host reads, changed after construction: ``search_tools``,
``guardrail_list``, ``default_max_parallel_requests``, ``provider_budget_config``,
``health_check_ignore_transient_errors`` and ``router_general_settings``.

``apply(ChangeSet(assign=...))`` stores each as given, ``settings()`` reports it, and whatever reads
it afterwards sees it from then on. None of them is runtime-updatable, so ``ChangeSet(settings=...)``
(``update_settings``) ignores them.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.router_utils.kubernetes_pod_discovery import KubernetesPodDiscovery
from litellm.types.router import Deployment, RouterGeneralSettings
from tests.unit.test_router.core.endpoints_helpers import EndpointHost, patch_direct_endpoints
from tests.unit.test_router.core.retries_health_helpers import cooled_down
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit
from tests.unit.test_router.core.selection_helpers import ids_of
from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts

MakeCore = Callable[..., RouterCore]
MESSAGES: Final = [{"role": "user", "content": "hi"}]
SEARCH_TOOLS: Final = [{"search_tool_name": "s", "litellm_params": {"search_provider": "tavily", "api_key": "tk"}}]
GUARDRAILS: Final = [{"guardrail_name": "gr", "litellm_params": {"guardrail": "presidio", "mode": "pre_call"}}]
PROVIDER_BUDGET: Final = {"openai": {"budget_limit": 1, "time_period": "1d"}}

ASSIGNED: Final[list[tuple[str, object]]] = [  # mutable-ok: pytest parameters
    ("search_tools", SEARCH_TOOLS),
    ("guardrail_list", GUARDRAILS),
    ("default_max_parallel_requests", 3),
    ("provider_budget_config", PROVIDER_BUDGET),
    ("health_check_ignore_transient_errors", True),
    ("router_general_settings", RouterGeneralSettings(pass_through_all_models=True)),
    ("provider_default_deployment_ids", ["d1"]),
    ("enable_health_check_routing", True),
    ("background_health_check_model_groups", ["g"]),
]


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> EndpointHost:
    endpoint_host: Final = EndpointHost()
    patch_direct_endpoints(monkeypatch, endpoint_host)
    return endpoint_host


def _endpoint(core: RouterCore, name: str) -> Callable[..., Awaitable[object]]:
    """An async endpoint method, which the core serves by name."""
    method: Final[Callable[..., Awaitable[object]]] = getattr(core, name)  # pyright: ignore[reportAny]  # endpoint methods are served by name
    return method


async def _open_stream(core: RouterCore) -> object:
    return await _endpoint(core, "acompletion")(model="a", messages=MESSAGES, stream=True)


# --- settings() ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "value"), ASSIGNED)
def test_an_assigned_host_setting_is_reported_by_settings(make_core: MakeCore, name: str, value: object) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")])
    before: Final = core.settings()
    _ = core.apply(ChangeSet(assign={name: value}))
    assert getattr(core.settings(), name) == value
    assert core.settings() is not before


@pytest.mark.parametrize(("name", "value"), [("search_tools", SEARCH_TOOLS), ("guardrail_list", GUARDRAILS)])
def test_an_assigned_host_setting_is_copied_from_the_callers_list(
    make_core: MakeCore, name: str, value: list[dict[str, object]]
) -> None:
    given: Final = [dict(entry) for entry in value]
    core: Final = make_core(model_list=[deployment("g", "d1")])
    _ = core.apply(ChangeSet(assign={name: given}))
    given.clear()
    assert getattr(core.settings(), name) == value  # pyright: ignore[reportAny]  # a settings field named by the parameter


@pytest.mark.parametrize(("name", "value"), ASSIGNED)
def test_update_settings_ignores_a_host_setting(make_core: MakeCore, name: str, value: object) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")])
    before: Final[object] = getattr(core.settings(), name)  # pyright: ignore[reportAny]  # a settings field named by the parameter
    _ = core.apply(ChangeSet(settings={name: value}))
    assert getattr(core.settings(), name) == before


def test_a_router_attribute_write_assigns_a_host_setting(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")])
    core.search_tools = SEARCH_TOOLS  # pyright: ignore[reportAttributeAccessIssue]  # the deprecated attribute write
    assert core.settings().search_tools == SEARCH_TOOLS


# --- router_general_settings as a constructor argument ------------------------------------------


def test_router_general_settings_passed_to_the_constructor_are_reported(make_core: MakeCore) -> None:
    general: Final = RouterGeneralSettings(async_only_mode=True, pass_through_all_models=True)
    core: Final = make_core(model_list=[deployment("g", "d1")], router_general_settings=general)
    assert core.settings().router_general_settings == general


def test_router_general_settings_of_none_become_the_defaults(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], router_general_settings=None)
    assert core.settings().router_general_settings == RouterGeneralSettings()


def test_routers_built_without_general_settings_share_one_default_object(make_core: MakeCore) -> None:
    """The Python core's default argument is one ``RouterGeneralSettings()`` every router shares."""
    first: Final = make_core(model_list=[deployment("g", "d1")])
    second: Final = make_core(model_list=[deployment("g", "d1")])
    assert first.settings().router_general_settings is second.settings().router_general_settings


async def test_router_general_settings_leave_routing_unchanged(make_core: MakeCore, host: EndpointHost) -> None:
    general: Final = RouterGeneralSettings(async_only_mode=True, pass_through_all_models=True)
    core: Final = make_core(model_list=[deployment("g", "d1")], router_general_settings=general)
    _ = await _endpoint(core, "acompletion")(model="g", messages=MESSAGES)
    completion: Final[Callable[..., object]] = getattr(core, "completion")  # pyright: ignore[reportAny]  # see _endpoint
    _ = completion(model="g", messages=MESSAGES)
    assert host.deployments_called() == ["d1", "d1"]


# --- readers ------------------------------------------------------------------------------------


async def test_asearch_reads_search_tools_assigned_after_construction(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    _ = core.apply(ChangeSet(assign={"search_tools": SEARCH_TOOLS}))
    _ = await _endpoint(core, "asearch")(model="s", query="q")
    call: Final = host.calls[0]
    assert (call.op, call.kwargs["search_provider"], call.kwargs["api_key"]) == ("asearch", "tavily", "tk")


async def test_asearch_no_longer_finds_a_tool_dropped_by_a_changeset(make_core: MakeCore, host: EndpointHost):
    core: Final = make_core(model_list=[deployment("g", "d1")], search_tools=SEARCH_TOOLS, num_retries=0)
    _ = core.apply(ChangeSet(assign={"search_tools": []}))
    with pytest.raises(ValueError, match=r"Search tool 's' not found in router\.search_tools"):
        _ = await _endpoint(core, "asearch")(model="s", query="q")
    assert host.calls == []


async def test_aguardrail_reads_a_guardrail_list_assigned_after_construction(make_core: MakeCore, host: ScriptedHost):
    seen: Final[list[Mapping[str, object]]] = []

    async def guard(**kwargs: object) -> str:  # kwargs-ok: the guardrail's keyword payload
        seen.append(kwargs)
        return "verdict"

    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    _ = core.apply(ChangeSet(assign={"guardrail_list": GUARDRAILS}))
    assert await _endpoint(core, "aguardrail")(guardrail_name="gr", original_function=guard) == "verdict"
    assert seen[0]["selected_guardrail"] == GUARDRAILS[0]


async def test_a_default_max_parallel_requests_assigned_later_limits_the_next_requests(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("x"))
    core: Final = make_core(model_list=[deployment("a", "a")], num_retries=0)
    _ = core.apply(ChangeSet(assign={"default_max_parallel_requests": 1}))
    first: Final = await _open_stream(core)
    with pytest.raises(litellm.RateLimitError):
        await asyncio.wait_for(_open_stream(core), 5)
    assert await collect_texts(first) == "x"


async def test_a_deployment_keeps_the_slot_limit_it_was_given_before_a_change(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The limit is cached per deployment on first use: a later default does not replace it."""
    host.answer("openai/a", chat_stream("x"))
    core: Final = make_core(model_list=[deployment("a", "a")], num_retries=0, default_max_parallel_requests=1)
    assert await collect_texts(await _open_stream(core)) == "x"
    _ = core.apply(ChangeSet(assign={"default_max_parallel_requests": 5}))
    first: Final = await _open_stream(core)
    with pytest.raises(litellm.RateLimitError):
        await asyncio.wait_for(_open_stream(core), 5)
    assert await collect_texts(first) == "x"


async def test_a_budget_config_assigned_beside_the_budget_check_builds_the_limiter_with_it(make_core: MakeCore) -> None:
    """``apply_changes`` assigns before ``update_settings`` adds the check."""
    core: Final = make_core(model_list=[deployment("g", "a")])
    _ = core.apply(
        ChangeSet(
            assign={"provider_budget_config": PROVIDER_BUDGET},
            settings={"optional_pre_call_checks": ["router_budget_limiting"]},
        )
    )
    limiter: Final = core.registry().plugins("budget_limiter")
    assert isinstance(limiter, RouterBudgetLimiting)
    assert limiter.provider_budget_config is core.settings().provider_budget_config
    assert list(limiter.provider_budget_config or {}) == ["openai"]
    core.retire()


async def test_a_budget_limiter_built_already_keeps_its_config(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], provider_budget_config=PROVIDER_BUDGET)
    limiter: Final = core.registry().plugins("budget_limiter")
    assert isinstance(limiter, RouterBudgetLimiting)
    _ = core.apply(ChangeSet(assign={"provider_budget_config": {"azure": {"budget_limit": 2, "time_period": "1d"}}}))
    assert core.registry().plugins("budget_limiter") is limiter
    assert list(limiter.provider_budget_config or {}) == ["openai"]
    assert list(core.settings().provider_budget_config or {}) == ["azure"]
    core.retire()


# --- settings().guardrail(name) -----------------------------------------------------------------


def test_settings_guardrail_returns_the_one_configured_under_the_name(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=GUARDRAILS)
    assert core.settings().guardrail("gr") == GUARDRAILS[0]


def test_settings_guardrail_with_an_unknown_name_raises_value_error(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=GUARDRAILS)
    with pytest.raises(ValueError, match="No guardrail found with name: nope"):
        _ = core.settings().guardrail("nope")


def test_settings_guardrail_picks_among_same_named_guardrails_by_weight(make_core: MakeCore) -> None:
    guardrails: Final = [
        {"guardrail_name": "gr", "litellm_params": {"guardrail": "never", "weight": 0}},
        {"guardrail_name": "gr", "litellm_params": {"guardrail": "always", "weight": 1}},
    ]
    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=guardrails)
    assert {core.settings().guardrail("gr")["litellm_params"]["guardrail"] for _ in range(20)} == {"always"}


def test_settings_guardrail_reads_a_guardrail_list_assigned_after_construction(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")])
    _ = core.apply(ChangeSet(assign={"guardrail_list": GUARDRAILS}))
    assert core.settings().guardrail("gr") == GUARDRAILS[0]


# --- a default_max_parallel_requests that is no int ---------------------------------------------


@pytest.mark.parametrize("assigned", [False, True])
async def test_a_float_default_max_parallel_requests_limits_the_slots(
    make_core: MakeCore, host: ScriptedHost, assigned: bool
) -> None:
    host.answer("openai/a", chat_stream("x"))
    given: Final = {} if assigned else {"default_max_parallel_requests": 1.0}
    core: Final = make_core(model_list=[deployment("a", "a")], num_retries=0, **given)
    if assigned:
        _ = core.apply(ChangeSet(assign={"default_max_parallel_requests": 1.0}))
    first: Final = await _open_stream(core)
    with pytest.raises(litellm.RateLimitError):
        await asyncio.wait_for(_open_stream(core), 5)
    assert await collect_texts(first) == "x"


# --- health-check routing assigned after construction -------------------------------------------


async def _healthy_ids(core: RouterCore, model: str) -> list[str]:  # mutable-ok: a fresh list
    select: Final[Callable[..., Awaitable[object]]] = getattr(core, "select")  # pyright: ignore[reportAny]  # the core's select, which takes kind
    return ids_of(await select(model, kind="healthy"))


def _unhealthy(*ids: str) -> dict[str, dict[str, object]]:  # mutable-ok: health states
    return {i: {"is_healthy": False, "timestamp": time.time(), "reason": "probe failed"} for i in ids}


async def test_health_check_routing_turned_on_later_drops_unhealthy_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    core.health().health_checks.set_deployment_health_states(_unhealthy("a"))  # pyright: ignore[reportArgumentType]  # health states
    assert await _healthy_ids(core, "g") == ["a", "b"]
    _ = core.apply(ChangeSet(assign={"enable_health_check_routing": True}))
    assert await _healthy_ids(core, "g") == ["b"]
    _ = core.apply(ChangeSet(assign={"enable_health_check_routing": False}))
    assert await _healthy_ids(core, "g") == ["a", "b"]


async def test_background_health_check_model_groups_assigned_later_scope_the_filter(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b"), deployment("h", "c"), deployment("h", "d")],
        enable_health_check_routing=True,
    )
    core.health().health_checks.set_deployment_health_states(_unhealthy("a", "c"))  # pyright: ignore[reportArgumentType]  # health states
    assert await _healthy_ids(core, "h") == ["d"]
    _ = core.apply(ChangeSet(assign={"background_health_check_model_groups": ["g"]}))
    assert await _healthy_ids(core, "g") == ["b"]
    assert await _healthy_ids(core, "h") == ["c", "d"]


# --- provider_default_deployment_ids assigned after construction --------------------------------


@pytest.mark.parametrize(("default_ids", "cooled"), [([], ["a"]), (["a"], [])])
async def test_an_assigned_provider_default_deployment_never_cools_down(
    make_core: MakeCore, host: ScriptedHost, default_ids: list[str], cooled: list[str]
) -> None:
    host.answer("openai/a", rate_limit())
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0)
    _ = core.apply(ChangeSet(assign={"provider_default_deployment_ids": default_ids}))
    with pytest.raises(litellm.RateLimitError):
        await core.call("acompletion", {"model": "a", "messages": MESSAGES})
    assert sorted(cooled_down(core, ["a", "b"])) == cooled


async def test_wildcard_deployments_never_cool_down(make_core: MakeCore, host: ScriptedHost) -> None:
    """Each is a provider default deployment, which the cooldown check skips."""
    host.answer("openai/x", rate_limit())
    core: Final = make_core(
        model_list=[deployment("openai/*", "w", model="openai/*"), deployment("openai/*", "w2", model="openai/*")],
        num_retries=0,
    )
    for _ in range(3):
        with pytest.raises(litellm.RateLimitError):
            await core.call("acompletion", {"model": "openai/x", "messages": MESSAGES})
    assert cooled_down(core, ["w", "w2"]) == {}
    assert len(host.calls) == 3


async def test_an_added_wildcard_deployment_never_cools_down(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/x", rate_limit())
    core: Final = make_core(model_list=[deployment("openai/*", "w", model="openai/*")], num_retries=0)
    _ = core.apply(ChangeSet(upsert=[Deployment.model_validate(deployment("openai/*", "w2", model="openai/*"))]))
    assert core.settings().provider_default_deployment_ids == ["w", "w2"]
    for _ in range(3):
        with pytest.raises(litellm.RateLimitError):
            await core.call("acompletion", {"model": "openai/x", "messages": MESSAGES})
    assert cooled_down(core, ["w", "w2"]) == {}


# --- Kubernetes pod discovery -------------------------------------------------------------------


class _RecordingDiscovery(KubernetesPodDiscovery):
    """Resolves every pod-discovery deployment to one pod IP, and records the retry counts seen."""

    def __init__(self) -> None:
        super().__init__(proxy_environment=lambda: {})
        self.seen: list[object] = []  # mutable-ok: filled as picks resolve

    def _pod(self, deployment: object) -> object:
        record: Final = dict(as_str_mapping(deployment) or {})
        params: Final = as_str_mapping(record.get("litellm_params")) or {}
        if params.get("kubernetes_pod_discovery") is not True:
            return deployment
        record["litellm_params"] = {**params, "api_base": "http://10.0.0.7:4000"}
        return record

    def resolve_deployment(self, deployment: object, request_kwargs: Mapping[str, object] | None = None) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # the resolver's generic signature, read as objects
        self.seen.append(self._request_retry_count(request_kwargs))
        return self._pod(deployment)

    async def async_resolve_deployment(  # pyright: ignore[reportIncompatibleMethodOverride]  # see resolve_deployment
        self, deployment: object, request_kwargs: Mapping[str, object] | None = None
    ) -> object:
        self.seen.append(self._request_retry_count(request_kwargs))
        return self._pod(deployment)


def _pod_deployment(deployment_id: str) -> dict[str, object]:  # mutable-ok: a model_list entry
    return deployment("g", deployment_id, api_base="http://svc.ns:4000", kubernetes_pod_discovery=True)


def test_settings_report_the_routers_pod_discovery(make_core: MakeCore) -> None:
    settings: Final = make_core(model_list=[deployment("g", "d1")]).settings()
    assert settings.is_set("kubernetes_pod_discovery")
    assert isinstance(settings.kubernetes_pod_discovery, KubernetesPodDiscovery)


async def test_an_assigned_pod_discovery_resolves_each_attempt(make_core: MakeCore, host: ScriptedHost) -> None:
    host.answer("openai/a", rate_limit())
    discovery: Final = _RecordingDiscovery()
    core: Final = make_core(model_list=[_pod_deployment("a")], num_retries=1)
    _ = core.apply(ChangeSet(assign={"kubernetes_pod_discovery": discovery}))
    assert core.settings().kubernetes_pod_discovery is discovery
    with pytest.raises(litellm.RateLimitError):
        await core.call("acompletion", {"model": "g", "messages": MESSAGES})
    assert [call.kwargs["api_base"] for call in host.calls] == ["http://10.0.0.7:4000"] * 2
    assert discovery.seen == [0, 1]


def test_an_assigned_pod_discovery_resolves_a_sync_attempt(make_core: MakeCore, host: ScriptedHost) -> None:
    discovery: Final = _RecordingDiscovery()
    core: Final = make_core(model_list=[_pod_deployment("a")])
    _ = core.apply(ChangeSet(assign={"kubernetes_pod_discovery": discovery}))
    completion: Final[Callable[..., object]] = getattr(core, "completion")  # pyright: ignore[reportAny]  # see _endpoint
    _ = completion(model="g", messages=MESSAGES)
    assert host.calls[0].kwargs["api_base"] == "http://10.0.0.7:4000"


def _api_base(record: object) -> object:
    return (as_str_mapping((as_str_mapping(record) or {}).get("litellm_params")) or {}).get("api_base")


async def test_a_pick_is_resolved_to_a_pod_and_a_healthy_listing_is_not(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[_pod_deployment("a")])
    _ = core.apply(ChangeSet(assign={"kubernetes_pod_discovery": _RecordingDiscovery()}))
    assert _api_base(await core.select("g")) == "http://10.0.0.7:4000"
    select: Final[Callable[..., Awaitable[object]]] = getattr(core, "select")  # pyright: ignore[reportAny]  # see _healthy_ids
    listed: Final[object] = await select("g", kind="healthy")
    assert isinstance(listed, list)
    assert [_api_base(record) for record in listed] == ["http://svc.ns:4000"]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # the listing


def test_a_sync_pick_is_resolved_to_a_pod(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[_pod_deployment("a")])
    _ = core.apply(ChangeSet(assign={"kubernetes_pod_discovery": _RecordingDiscovery()}))
    pick: Final[Callable[..., object]] = getattr(core, "get_available_deployment")  # pyright: ignore[reportAny]  # the deprecated sync pick
    assert _api_base(pick(model="g")) == "http://10.0.0.7:4000"


async def test_the_default_pod_discovery_resolves_a_service_host(make_core: MakeCore, host: ScriptedHost) -> None:
    """``localhost`` resolves to a loopback address, which takes the service host's place."""
    core: Final = make_core(
        model_list=[deployment("g", "a", api_base="http://localhost:4000", kubernetes_pod_discovery=True)]
    )
    _ = core.apply(ChangeSet(assign={"kubernetes_pod_discovery": KubernetesPodDiscovery(proxy_environment=lambda: {})}))
    await core.call("acompletion", {"model": "g", "messages": MESSAGES})
    assert str(host.calls[0].kwargs["api_base"]) in ("http://127.0.0.1:4000", "http://[::1]:4000")
