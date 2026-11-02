"""
Gaps in the routing-strategy pre-call checks and the Responses-websocket turn admission of ``PythonCore``
(router_python_core.py ``routing_strategy_pre_call_checks``, ``async_routing_strategy_pre_call_checks``,
``async_admit_deployment_turn``). A ``CustomLogger`` on the host's ``callbacks`` list (``litellm.callbacks``)
takes part in those checks through ``pre_call_check`` / ``async_pre_call_check``.
"""

from collections.abc import Callable
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_core import RouterCore
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.retries_health_helpers import attempt, attempt_sync, cooled_down
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit

MakeCore = Callable[..., RouterCore]


class PreCallGate(CustomLogger):
    """Records the deployment ids its pre-call checks saw and raises ``error`` (when set) from them."""

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error
        self.seen: Final[list[tuple[str, str]]] = []

    async def async_pre_call_check(self, deployment: dict, parent_otel_span: object) -> dict | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.seen.append(("async", deployment["model_info"]["id"]))
        if self.error is not None:
            raise self.error
        return None

    def pre_call_check(self, deployment: dict) -> dict | None:
        self.seen.append(("sync", deployment["model_info"]["id"]))
        if self.error is not None:
            raise self.error
        return None


def gated(host: ScriptedHost, error: Exception | None = None) -> PreCallGate:
    gate: Final = PreCallGate(error)
    host.lists["callbacks"].append(gate)
    return gate


GROUP: Final = [deployment("g", "a"), deployment("g", "b")]


# --- async pre-call checks ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rate_limit_from_a_pre_call_check_cools_the_deployment_down_and_is_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUP, num_retries=0, cooldown_time=33)
    gate: Final = gated(host, rate_limit(message="gate says no"))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RateLimitError)
    assert str(error) == "litellm.RateLimitError: gate says no"
    assert host.calls == []
    ((kind, checked),) = gate.seen
    assert kind == "async"
    cooled: Final = cooled_down(core, ["a", "b"])
    assert list(cooled) == [checked]
    assert cooled[checked]["cooldown_time"] == 33


@pytest.mark.asyncio
async def test_any_other_error_from_a_pre_call_check_is_raised_without_a_cooldown(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUP, num_retries=0, cooldown_time=33)
    gate: Final = gated(host, ValueError("gate broke"))

    error: Final = await attempt(core)

    assert type(error) is ValueError
    assert str(error) == "gate broke"
    assert host.calls == []
    assert len(gate.seen) == 1
    assert cooled_down(core, ["a", "b"]) == {}


@pytest.mark.asyncio
async def test_a_passing_pre_call_check_sees_the_deployment_that_is_then_called(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUP, num_retries=0)
    gate: Final = gated(host)

    await attempt(core)

    assert gate.seen == [("async", host.deployments_called("acompletion")[0])]


def test_a_rate_limit_from_a_sync_pre_call_check_cools_the_deployment_down_and_is_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The sync twin cools the refused deployment down as the async path does."""
    core: Final = make_core(model_list=GROUP, num_retries=0, cooldown_time=33)
    gate: Final = gated(host, rate_limit(message="gate says no"))

    error: Final = attempt_sync(core)

    assert isinstance(error, litellm.RateLimitError)
    assert host.calls == []
    ((kind, checked),) = gate.seen
    assert kind == "sync"
    cooled: Final = cooled_down(core, ["a", "b"])
    assert list(cooled) == [checked]
    assert cooled[checked]["cooldown_time"] == 33


def test_any_other_error_from_a_sync_pre_call_check_is_raised_without_a_cooldown(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUP, num_retries=0, cooldown_time=33)
    gated(host, ValueError("gate broke"))

    error: Final = attempt_sync(core)

    assert type(error) is ValueError
    assert host.calls == []
    assert cooled_down(core, ["a", "b"]) == {}


# --- Responses websocket turn admission ----------------------------------------------------------


class TestResponsesWebsocketTurnAdmission:
    """The core hands the websocket handler a ``turn_admission`` callable that re-runs the deployment's
    admission (budget limiter, routing strategy, pre-call checks) before every later turn."""

    @pytest.fixture
    def host(self) -> EndpointHost:
        return EndpointHost()

    @staticmethod
    async def open_socket(core: RouterCore, host: EndpointHost) -> Callable[[], object]:
        await core.call("_aresponses_websocket", {"model": "g"})
        (call,) = host.calls_to("_aresponses_websocket")
        admission: Final = call.kwargs["turn_admission"]
        assert callable(admission)
        return admission

    @pytest.mark.asyncio
    async def test_simple_shuffle_admits_every_turn_and_runs_the_pre_call_checks(
        self, make_core: MakeCore, host: EndpointHost
    ) -> None:
        core: Final = make_core(model_list=[deployment("g", "a", rpm=1)])
        gate: Final = gated(host)
        admit: Final = await self.open_socket(core, host)

        admitted: Final = [dict(await admit()) for _ in range(3)]  # pyright: ignore[reportGeneralTypeIssues]

        assert admitted == [{}, {}, {}]
        assert gate.seen == [("async", "a")] * 4  # the socket opening, then each turn

    @pytest.mark.asyncio
    async def test_a_rate_limit_from_a_pre_call_check_refuses_the_turn(
        self, make_core: MakeCore, host: EndpointHost
    ) -> None:
        core: Final = make_core(model_list=[deployment("g", "a")])
        gate: Final = gated(host)
        admit: Final = await self.open_socket(core, host)
        gate.error = rate_limit(message="turn refused")

        with pytest.raises(litellm.RateLimitError, match="turn refused"):
            await admit()  # pyright: ignore[reportGeneralTypeIssues]

    @pytest.mark.asyncio
    async def test_usage_based_routing_admits_turns_until_the_deployment_rpm_is_used_up(
        self, make_core: MakeCore, host: EndpointHost
    ) -> None:
        """Opening the socket already counts two requests (its pre-call check and its success), so an
        rpm of 5 leaves room for three turns."""
        core: Final = make_core(model_list=[deployment("g", "a", rpm=5)], routing_strategy="usage-based-routing-v2")
        admit: Final = await self.open_socket(core, host)

        admitted: Final = [dict(await admit()) for _ in range(3)]  # pyright: ignore[reportGeneralTypeIssues]
        with pytest.raises(litellm.RateLimitError) as refused:
            await admit()  # pyright: ignore[reportGeneralTypeIssues]

        assert admitted == [{}, {}, {}]
        assert str(refused.value).startswith(
            "litellm.RateLimitError: No deployments available for selected model. Passed model=g. Deployments="
        )
