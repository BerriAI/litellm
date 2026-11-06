"""
The old public Router members outside the entry points: registry lookups, the
routing-state lifecycle, the routing callbacks' checks, one provider call's accounting and the
pre-routing strategy builders (``RustLegacyMembers``, litellm/router_rust/surface/legacy_members.py).
"""

import asyncio
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.constants import ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.token_counter import served_openai_models
from litellm.router_api import RoutingContext
from litellm.router_core import RouterCore
from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook
from litellm.router_strategy.auto_router.auto_router_model_naming import HEURISTIC_V2_CAPABILITY
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import PromptCachingDeploymentCheck
from litellm.types.router import CustomRoutingStrategyBase, Deployment
from tests.unit.test_router.core.construction_precall_helpers import CallbackHost
from tests.unit.test_router.core.gaps_l11100_helpers import (
    TIERS,
    adaptive_marker,
    complexity_marker,
    marker,
    registered_tags,
)
from tests.unit.test_router.core.registry_helpers import MESSAGES, MakeCore, team_deployment
from tests.unit.test_router.core.retries_health_helpers import cooled_down
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    completion_response,
    deployment,
    rate_limit,
    rust_core,
    server_error,
)

GROUP: Final = [deployment("g", "a"), deployment("g", "b")]
PLAIN: Final = [deployment("small", "s"), deployment("big", "b")]


@pytest.fixture
def host() -> ScriptedHost:
    """Switching an optional pre-call check off removes it from the host's lists."""
    return CallbackHost()


class Gate(CustomLogger):
    """Records what its pre-call checks saw and raises ``error`` (when set) from them."""

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error
        self.seen: Final[list[tuple[object, object]]] = []

    def pre_call_check(self, deployment: dict) -> dict | None:
        self.seen.append((deployment, None))
        if self.error is not None:
            raise self.error
        return None

    async def async_pre_call_check(self, deployment: dict, parent_otel_span: object) -> dict | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.seen.append((deployment, parent_otel_span))
        if self.error is not None:
            raise self.error
        return None


class Narrow(CustomLogger):
    """Keeps the deployments other than ``dropped``, or raises ``error``."""

    def __init__(self, dropped: str = "", error: Exception | None = None) -> None:
        super().__init__()
        self.dropped = dropped
        self.error = error
        self.calls: Final[list[Mapping[str, object]]] = []

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: object,
        request_kwargs: object = None,
        parent_otel_span: object = None,
    ) -> list:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "request_kwargs": request_kwargs,
                "parent_otel_span": parent_otel_span,
            }
        )
        if self.error is not None:
            raise self.error
        return [d for d in healthy_deployments if d["model_info"]["id"] != self.dropped]


class RecordingLogging:
    """The logging object's failure dispatch, recorded."""

    def __init__(self) -> None:
        self.dispatched: Final[list[Mapping[str, object]]] = []

    async def dispatch_failure_handlers(self, **kwargs: object) -> None:  # kwargs-ok: recorded as passed
        self.dispatched.append(kwargs)


def counters(core: RouterCore, deployment_id: str, kind: str) -> list[object]:  # mutable-ok: a fresh list
    """The per-minute ``kind`` (``tpm``, ``rpm``) counters of ``deployment_id`` in the routing cache."""
    memory: Final = core.health().store.in_memory_cache
    return [
        memory.get_cache(key)
        for key in tuple(memory.cache_dict)
        if key.startswith(f"global_router:{deployment_id}:") and f":{kind}:" in key
    ]


def call_metadata(deployment_id: str = "a") -> dict[str, object]:  # mutable-ok: the request's own bucket
    return {"model_info": {"id": deployment_id}, "deployment": f"openai/{deployment_id}", "model_group": "g"}


# --- registry lookups --------------------------------------------------------------------------


def test_get_routing_group_names_a_group_a_deployment_does_not_shadow(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("m2", "b")],
        routing_groups=[{"group_name": "grp", "models": ["m1", "m2"], "routing_strategy": "simple-shuffle"}],
    )
    group: Final = core.get_routing_group("grp")  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert group is not None
    assert group.group_name == "grp"
    assert core.get_routing_group("m1") is None  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.get_routing_group("nope") is None  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_team_public_model_names_of_lists_one_teams_public_names(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            team_deployment("int-1", "t1", "TA", "pub-a"),
            team_deployment("int-2", "t2", "TB", "pub-b"),
        ]
    )

    assert core.team_public_model_names_of("TA") == frozenset({"pub-a"})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
    assert core.team_public_model_names_of("nobody") == frozenset()  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_served_openai_models_reads_the_group_and_follows_a_model_list_change(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o")])
    before: Final = core.served_openai_models("g")  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert before == served_openai_models(core.registry().deployments("g", source="named") or ())
    assert core.served_openai_models("nope") == ()  # pyright: ignore[reportAttributeAccessIssue]  # see above

    core.set_model_list([deployment("g", "b", model="openai/gpt-4o-mini")])  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert core.served_openai_models("g") == served_openai_models(
        core.registry().deployments("g", source="named") or ()
    )  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_auto_router_capability_violation_resolves_the_limit_on_every_call(make_core: MakeCore) -> None:
    limits: Final[list[int | None]] = [None]  # mutable-ok: what the resolver answers

    core: Final = make_core(
        model_list=[*PLAIN, complexity_marker(config={"classifier_type": "heuristic_v2", "tiers": dict(TIERS)})],
        auto_router_capability_limit=lambda: limits[0],
    )

    assert core.auto_router_capability_violation(HEURISTIC_V2_CAPABILITY) is None  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
    limits[0] = 1
    violation: Final = core.auto_router_capability_violation(HEURISTIC_V2_CAPABILITY)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert violation is not None
    assert violation.startswith("At most 1 auto-router(s) using the heuristic_v2 classifier can be registered")


# --- routing state ------------------------------------------------------------------------------


def test_retire_routing_state_retires_the_selectors_and_keeps_them_registered(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    core: Final = make_core(model_list=list(GROUP), routing_strategy="usage-based-routing-v2")
    selector: Final = core.lowesttpm_logger_v2  # pyright: ignore[reportAttributeAccessIssue]  # an old public attribute
    assert isinstance(selector, BaseRoutingStrategy)
    retired: Final[list[object]] = []  # mutable-ok: what retire() saw
    monkeypatch.setattr(selector, "retire", lambda: retired.append(selector))

    core.retire_routing_state()  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert retired == [selector]
    assert core.lowesttpm_logger_v2 is selector  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert selector in host.lists["callbacks"]


def test_routing_strategy_init_refuses_an_unknown_strategy_and_changes_nothing(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP), routing_strategy="least-busy")
    selector: Final = core.leastbusy_logger  # pyright: ignore[reportAttributeAccessIssue]  # an old public attribute

    with pytest.raises(ValueError, match="not-a-real-strategy"):
        core.routing_strategy_init(routing_strategy="not-a-real-strategy", routing_strategy_args={})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert core.leastbusy_logger is selector  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_routing_strategy_init_replaces_the_selector_and_leaves_the_setting(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(GROUP), routing_strategy="least-busy")
    old: Final = core.leastbusy_logger  # pyright: ignore[reportAttributeAccessIssue]  # an old public attribute

    core.routing_strategy_init("usage-based-routing-v2", {})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert core.leastbusy_logger is None  # pyright: ignore[reportAttributeAccessIssue]  # see above
    selector: Final = core.lowesttpm_logger_v2  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert isinstance(selector, LowestTPMLoggingHandler_v2)
    assert selector in host.lists["callbacks"]
    assert old not in host.lists["callbacks"]
    assert core.settings().routing_strategy == "least-busy"


@pytest.mark.asyncio
async def test_a_request_override_of_the_initialized_strategy_is_not_the_routers_own(make_core: MakeCore) -> None:
    """``_get_override_strategy_selector`` compares with the router's ``routing_strategy``, which
    ``routing_strategy_init`` leaves, so the request gets its own selector."""
    core: Final = make_core(model_list=list(GROUP))
    core.routing_strategy_init("least-busy", {})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
    initialized: Final = core.leastbusy_logger  # pyright: ignore[reportAttributeAccessIssue]  # an old public attribute
    assert isinstance(initialized, LeastBusyLoggingHandler)

    picked: Final = await core.select("g", {"routing_strategy": "least-busy", "messages": list(MESSAGES)})

    assert picked["model_info"]["id"] in {"a", "b"}  # pyright: ignore[reportIndexIssue]  # the pick
    assert core.leastbusy_logger is initialized  # pyright: ignore[reportAttributeAccessIssue]  # see above


@pytest.mark.asyncio
async def test_routing_strategy_init_drops_a_custom_strategy(make_core: MakeCore) -> None:
    class Custom(CustomRoutingStrategyBase):
        async def async_get_available_deployment(self, *args: object, **kwargs: object) -> dict:  # kwargs-ok: any pick
            return dict(GROUP[0])

        def get_available_deployment(self, *args: object, **kwargs: object) -> dict:  # kwargs-ok: any pick
            return dict(GROUP[0])

    core: Final = make_core(model_list=list(GROUP))
    core.register_plugin(Custom())
    assert {(await core.select("g", {"messages": list(MESSAGES)}))["model_info"]["id"] for _ in range(20)} == {"a"}  # pyright: ignore[reportIndexIssue]  # the pick

    core.routing_strategy_init("simple-shuffle", {})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    picks: Final = {(await core.select("g", {"messages": list(MESSAGES)}))["model_info"]["id"] for _ in range(60)}  # pyright: ignore[reportIndexIssue]  # the pick
    assert picks == {"a", "b"}


def test_set_optional_pre_call_checks_adds_and_drops_the_toggleable_checks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(GROUP))

    core.set_optional_pre_call_checks(["prompt_caching"])  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
    assert [type(callback) for callback in core.optional_callbacks] == [PromptCachingDeploymentCheck]  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert any(isinstance(callback, PromptCachingDeploymentCheck) for callback in host.lists["callbacks"])

    core.set_optional_pre_call_checks(None)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert [type(callback) for callback in core.optional_callbacks] == [PromptCachingDeploymentCheck]  # pyright: ignore[reportAttributeAccessIssue]  # see above

    core.set_optional_pre_call_checks([])  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.optional_callbacks == []  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert not any(isinstance(callback, PromptCachingDeploymentCheck) for callback in host.lists["callbacks"])


def test_initialize_assistants_endpoint_replaces_an_assigned_endpoint(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP))
    assigned: Final = object()
    core.aget_assistants = assigned  # pyright: ignore[reportAttributeAccessIssue]  # an old public endpoint
    core.amoderation = assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above

    core.initialize_assistants_endpoint()  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert core.aget_assistants is not assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert callable(core.aget_assistants)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.amoderation is assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_initialize_router_endpoints_replaces_only_what_it_binds(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP))
    assigned: Final = object()
    for name in ("amoderation", "aresponses", "aupload_container_file", "aget_assistants", "acreate_file"):
        setattr(core, name, assigned)

    core.initialize_router_endpoints()  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert core.amoderation is not assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.aresponses is not assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.aupload_container_file is not assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above
    # `initialize_router_endpoints` binds neither of these: the assistants have their own, and
    # `acreate_file` is a method an assignment shadows.
    assert core.aget_assistants is assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.acreate_file is assigned  # pyright: ignore[reportAttributeAccessIssue]  # see above


# --- pre-routing strategies ----------------------------------------------------------------------


def test_init_complexity_router_deployment_registers_it_once(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PLAIN))
    router: Final = Deployment(**complexity_marker())

    core.init_complexity_router_deployment(router)  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert registered_tags(core, "complexity") == {"smart": [()]}
    with pytest.raises(ValueError, match="already exists"):
        core.init_complexity_router_deployment(router)  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_init_quality_router_deployment_needs_a_default_model(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PLAIN))

    with pytest.raises(ValueError, match="quality_router_default_model is required"):
        core.init_quality_router_deployment(Deployment(**marker("q", "m", "auto_router/quality_router")))  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    core.init_quality_router_deployment(  # pyright: ignore[reportAttributeAccessIssue]  # see above
        Deployment(**marker("q", "m", "auto_router/quality_router", quality_router_default_model="big"))
    )
    assert registered_tags(core, "quality") == {"q": [()]}


def test_init_adaptive_router_deployment_registers_it_and_its_hook(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=list(PLAIN))

    core.init_adaptive_router_deployment(Deployment(**adaptive_marker("ad", "m", ["big"])))  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert registered_tags(core, "adaptive") == {"ad": [()]}
    assert any(isinstance(callback, AdaptiveRouterPostCallHook) for callback in host.lists["callbacks"])


def test_init_auto_router_deployment_needs_its_config(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PLAIN))

    with pytest.raises(ValueError, match="auto_router_config_path or auto_router_config is required"):
        core.init_auto_router_deployment(Deployment(**marker("auto", "m", "auto_router/semantic")))  # pyright: ignore[reportAttributeAccessIssue]  # an old public member


# --- routing callbacks ---------------------------------------------------------------------------


def test_routing_strategy_pre_call_checks_hands_each_callback_the_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    gate: Final = Gate()
    host.lists["callbacks"].append(gate)
    core: Final = make_core(model_list=list(GROUP))
    picked: Final = dict(GROUP[0])

    core.routing_strategy_pre_call_checks(picked)  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert gate.seen[0][0] is picked

    gate.error = rate_limit(message="sync gate")
    with pytest.raises(litellm.RateLimitError, match="sync gate"):
        core.routing_strategy_pre_call_checks(picked)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert cooled_down(core, ["a"]) == {}


@pytest.mark.asyncio
async def test_a_rate_limited_async_pre_call_check_cools_down_and_logs_the_failure(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    gate: Final = Gate(rate_limit(message="gate says no"))
    host.lists["callbacks"].append(gate)
    core: Final = make_core(model_list=list(GROUP), cooldown_time=33)
    logging_obj: Final = RecordingLogging()
    span: Final = object()

    with pytest.raises(litellm.RateLimitError, match="gate says no") as raised:
        await core.async_routing_strategy_pre_call_checks(  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
            deployment=dict(GROUP[0]), parent_otel_span=span, logging_obj=logging_obj
        )
    await asyncio.sleep(0)

    assert gate.seen[0][1] is span
    assert logging_obj.dispatched[0]["exception"] is raised.value
    assert logging_obj.dispatched[0]["prefer_async_handlers"] is True
    assert set(cooled_down(core, ["a", "b"])) == {"a"}


@pytest.mark.asyncio
async def test_another_async_pre_call_check_error_logs_the_failure_without_a_cooldown(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.lists["callbacks"].append(Gate(ValueError("no")))
    core: Final = make_core(model_list=list(GROUP))
    logging_obj: Final = RecordingLogging()

    with pytest.raises(ValueError, match="no"):
        await core.async_routing_strategy_pre_call_checks(  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
            deployment=dict(GROUP[0]), parent_otel_span=None, logging_obj=logging_obj
        )
    await asyncio.sleep(0)

    assert len(logging_obj.dispatched) == 1
    assert cooled_down(core, ["a", "b"]) == {}


@pytest.mark.asyncio
async def test_async_callback_filter_deployments_narrows_through_each_callback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    first: Final = Narrow(dropped="a")
    second: Final = Narrow()
    host.lists["callbacks"].extend((first, second))
    core: Final = make_core(model_list=list(GROUP))
    span: Final = object()
    request: Final = {"metadata": {}}

    remaining: Final = await core.async_callback_filter_deployments(  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
        model="g",
        healthy_deployments=[dict(entry) for entry in GROUP],
        messages=list(MESSAGES),
        parent_otel_span=span,
        request_kwargs=request,
    )

    assert [entry["model_info"]["id"] for entry in remaining] == ["b"]
    assert second.calls[0]["parent_otel_span"] is span
    assert second.calls[0]["request_kwargs"] is request


@pytest.mark.asyncio
async def test_a_failing_callback_filter_logs_the_failure_and_raises(make_core: MakeCore, host: ScriptedHost) -> None:
    host.lists["callbacks"].append(Narrow(error=ValueError("filter refused")))
    core: Final = make_core(model_list=list(GROUP))
    logging_obj: Final = RecordingLogging()

    with pytest.raises(ValueError, match="filter refused"):
        await core.async_callback_filter_deployments(  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
            model="g",
            healthy_deployments=[dict(entry) for entry in GROUP],
            messages=None,
            parent_otel_span=None,
            logging_obj=logging_obj,
        )
    await asyncio.sleep(0)

    assert len(logging_obj.dispatched) == 1


@pytest.mark.asyncio
async def test_async_admit_deployment_turn_runs_the_checks_on_the_deployment_as_configured_now(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    gate: Final = Gate()
    host.lists["callbacks"].append(gate)
    core: Final = make_core(model_list=list(GROUP))

    admission: Final = await core.async_admit_deployment_turn("g", {"model_info": {"id": "a"}}, {})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert dict(admission) == {}
    assert gate.seen[0][0]["litellm_params"]["model"] == "openai/a"  # pyright: ignore[reportIndexIssue]  # the deployment dict


@pytest.mark.asyncio
async def test_a_rate_limited_turn_cools_the_deployment_down(make_core: MakeCore, host: ScriptedHost) -> None:
    host.lists["callbacks"].append(Gate(rate_limit(message="turn refused")))
    core: Final = make_core(model_list=list(GROUP))

    with pytest.raises(litellm.RateLimitError, match="turn refused"):
        await core.async_admit_deployment_turn("g", {"model_info": {"id": "a"}}, {})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert set(cooled_down(core, ["a", "b"])) == {"a"}


# --- one provider call ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_call_counts_a_tracked_deployments_usage_and_notes_it_on_the_request(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o", rpm=5)])
    metadata: Final = call_metadata()

    async def endpoint(**kwargs: object) -> object:  # kwargs-ok: the request, as make_call passes it
        return completion_response(total_tokens=10)

    await core.make_call(endpoint, model="g", metadata=metadata)  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert counters(core, "a", "tpm") == [10]
    assert counters(core, "a", "rpm") == [1]
    assert metadata[ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY] == 10


@pytest.mark.asyncio
async def test_make_call_runs_a_sync_endpoint_and_stamps_the_response(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o")])
    answer: Final = completion_response()

    def endpoint(**kwargs: object) -> object:  # kwargs-ok: the request, as make_call passes it
        return answer

    response: Final = await core.make_call(endpoint, model="g", metadata=call_metadata())  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert response is answer
    assert answer._hidden_params["model_id"] == "a"  # pyright: ignore[reportPrivateUsage]  # what set_response_headers stamps
    assert counters(core, "a", "tpm") == []


@pytest.mark.asyncio
async def test_make_call_raises_the_endpoints_error(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o", rpm=5)])
    metadata: Final = call_metadata()

    async def endpoint(**kwargs: object) -> object:  # kwargs-ok: the request, as make_call passes it
        raise server_error()

    with pytest.raises(litellm.InternalServerError):
        await core.make_call(endpoint, model="g", metadata=metadata)  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert counters(core, "a", "rpm") == []
    assert ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY not in metadata


@pytest.mark.asyncio
async def test_increment_usage_notes_tokens_even_for_a_deployment_it_does_not_track(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o")])
    metadata: Final = call_metadata()

    await core.increment_deployment_usage_for_response(  # pyright: ignore[reportAttributeAccessIssue]  # an old public member
        response=completion_response(total_tokens=7), request_kwargs={"model": "g", "metadata": metadata}
    )

    assert metadata[ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY] == 7
    assert counters(core, "a", "rpm") == []


@pytest.mark.asyncio
async def test_increment_usage_takes_the_note_back_when_the_count_fails(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o", rpm=5)])
    metadata: Final = call_metadata()

    async def refused(**kwargs: object) -> None:  # kwargs-ok: the pipeline's arguments
        raise ConnectionError("the usage store is down")

    monkeypatch.setattr(core.health().store, "async_increment_cache_pipeline", refused)

    await core.increment_deployment_usage_for_response(completion_response(), {"model": "g", "metadata": metadata})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY not in metadata
    assert counters(core, "a", "rpm") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_kwargs",
    [{"model": "g", "metadata": {}}, {"metadata": call_metadata()}, {"model": "g"}],
    ids=["no-model-info", "no-model", "no-metadata"],
)
async def test_increment_usage_skips_a_request_with_nothing_to_count_under(
    make_core: MakeCore, request_kwargs: dict[str, object]
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o", rpm=5)])

    await core.increment_deployment_usage_for_response(completion_response(), dict(request_kwargs))  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert counters(core, "a", "rpm") == []


@pytest.mark.asyncio
async def test_increment_usage_counts_nothing_for_no_response(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model="openai/gpt-4o", rpm=5)])
    metadata: Final = call_metadata()

    await core.increment_deployment_usage_for_response(None, {"model": "g", "metadata": metadata})  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY not in metadata


@pytest.mark.asyncio
async def test_async_deployment_callback_on_failure_counts_one_request(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP))
    logged: Final = {"litellm_params": {"metadata": call_metadata(), "model_info": {"id": "a"}}}

    await core.async_deployment_callback_on_failure(logged, None, None, None)  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert counters(core, "a", "rpm") == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logged",
    [
        {"call_type": "aretrieve_batch", "litellm_params": {"metadata": call_metadata(), "model_info": {"id": "a"}}},
        {"litellm_params": {"metadata": {"deployment": "x"}, "model_info": {"id": "a"}}},
        {"litellm_params": {"metadata": call_metadata()}},
    ],
    ids=["batch-retrieve", "no-group", "no-id"],
)
async def test_async_deployment_callback_on_failure_skips(make_core: MakeCore, logged: dict[str, object]) -> None:
    core: Final = make_core(model_list=list(GROUP))

    await core.async_deployment_callback_on_failure(logged, None, None, None)  # pyright: ignore[reportAttributeAccessIssue]  # an old public member

    assert counters(core, "a", "rpm") == []


@pytest.mark.asyncio
async def test_async_deployment_callback_on_failure_needs_the_params(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP))

    with pytest.raises(KeyError, match="litellm_params"):
        await core.async_deployment_callback_on_failure({}, None, None, None)  # pyright: ignore[reportAttributeAccessIssue]  # an old public member


# --- the plugin context --------------------------------------------------------------------------


def test_deployment_params_merge_in_the_order_the_deployment_receives_them(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", temperature=0.1, tool_choice="auto", top_p=0.5)],
        default_litellm_params={"temperature": 0.9, "tool_choice": "none"},
    )
    target: Final = core.registry().deployment("a")

    params: Final = RoutingContext(core).deployment_params({"top_p": 0.7}, target)  # pyright: ignore[reportArgumentType]  # the core a context wraps

    assert params["model"] == "openai/a"
    assert params["temperature"] == 0.9
    assert params["tool_choice"] == "auto"
    assert params["top_p"] == 0.7
    assert RoutingContext(core).deployment_params({}, None).get("temperature") == 0.9  # pyright: ignore[reportArgumentType]  # see above


def test_update_kwargs_before_fallbacks_writes_the_group_into_the_named_bucket(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP), model_group_alias={"al": "g"})
    request: Final[dict[str, object]] = {"litellm_trace_id": "t", "litellm_metadata": {"k": "v"}}

    RoutingContext(core).update_kwargs_before_fallbacks("al", request, "litellm_metadata")  # pyright: ignore[reportArgumentType]  # see above

    assert request == {
        "litellm_trace_id": "t",
        "litellm_metadata": {"k": "v", "model_group": "al", "model_group_alias": "al"},
    }

    plain: Final[dict[str, object]] = {}  # mutable-ok: the request the core writes into
    RoutingContext(core).update_kwargs_before_fallbacks("g", plain)  # pyright: ignore[reportArgumentType]  # see above

    assert isinstance(plain["litellm_trace_id"], str)
    assert plain["metadata"] == {"model_group": "g", "model_group_alias": None}


# --- what stays refused --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "async_function_with_fallbacks",
        "async_function_with_fallbacks_common_utils",
        "async_function_with_retries",
        "function_with_fallbacks",
        "factory_function",
        "model_id_to_deployment_index_map",
    ],
)
def test_the_rust_core_refuses_what_it_cannot_serve(name: str) -> None:
    from litellm.router_rust.core import RustCoreUnsupported

    core: Final = rust_core(ScriptedHost(), model_list=list(GROUP))

    with pytest.raises(RustCoreUnsupported):
        getattr(core, name)
