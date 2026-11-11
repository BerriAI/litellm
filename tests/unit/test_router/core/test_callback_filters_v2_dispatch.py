from collections.abc import Generator, Mapping
from typing import Final

import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Hook, build_manifest, on
from litellm.integrations.custom_logger_v2.host import has_natives, register_native, reset_host_for_tests
from litellm.router_rust.steps.callbacks import OptionalChecks
from tests.unit.test_router.core.registry_helpers import MakeCore
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment

GROUP: Final = [deployment("g", "a"), deployment("g", "b"), deployment("g", "c")]


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    yield
    reset_host_for_tests()


class Narrow(CustomLogger):
    def __init__(self, dropped: str, order: list[str]) -> None:  # mutable-ok: the shared call log
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger's untyped **kwargs
        self.dropped = dropped
        self.order = order

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, dict[str, str]]],  # mutable-ok: the filter contract
        messages: object,
        request_kwargs: object = None,
        parent_otel_span: object = None,
    ) -> list[dict[str, dict[str, str]]]:  # mutable-ok: the filter contract
        self.order.append("legacy")
        return [d for d in healthy_deployments if d["model_info"]["id"] != self.dropped]


class Gate(CustomLoggerV2):
    def __init__(self, order: list[str]) -> None:  # mutable-ok: the shared call log
        self.order = order

    @on(Hook.filter_deployments, may_abort=True)
    async def gate(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")


class Recorded:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[tuple[object, ...]] = []  # mutable-ok: each call's arguments

    async def __call__(self, model: str, deployments: object, messages: object, request: object) -> object:
        self.calls.append((model, deployments, messages, request))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _checks(make_core: MakeCore) -> OptionalChecks:
    core: Final = make_core(model_list=list(GROUP))
    checks: Final[OptionalChecks] = core.__dict__["_checks"]  # pyright: ignore[reportAny]  # RustCore's own attribute
    return checks


async def test_without_natives_the_filters_run_once_and_their_answer_comes_back(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer: Final = (["kept"], "preferred")
    recorded: Final = Recorded(answer)
    checks: Final = _checks(make_core)
    monkeypatch.setattr(checks, "_filter_deployments", recorded)
    deployments: Final[list[dict[str, object]]] = [{"model_info": {"id": "a"}}]  # mutable-ok: filter contract
    messages: Final = object()
    request: Final[Mapping[str, object]] = {"model": "g"}

    result: Final = await checks.filter_deployments("g", deployments, messages, request)

    assert not has_natives(Hook.filter_deployments)
    assert result is answer
    assert len(recorded.calls) == 1
    model, seen_deployments, seen_messages, seen_request = recorded.calls[0]
    assert model == "g"
    assert seen_deployments is deployments
    assert seen_messages is messages
    assert seen_request is request


async def test_without_natives_a_filter_error_reaches_the_caller_unchanged(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    error: Final = RuntimeError("a filter raised")
    checks: Final = _checks(make_core)
    monkeypatch.setattr(checks, "_filter_deployments", Recorded(error))

    with pytest.raises(RuntimeError) as raised:
        await checks.filter_deployments("g", [], None, {})

    assert raised.value is error


async def test_a_blocking_native_runs_before_the_legacy_filters_and_the_pick_still_honours_them(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    order: Final[list[str]] = []  # mutable-ok: the shared call log
    register_native(build_manifest(Gate(order)))
    host.lists["callbacks"].append(Narrow("b", order))
    core: Final = make_core(model_list=list(GROUP))

    picked: Final[set[object]] = {(await core.select("g"))["model_info"]["id"] for _ in range(10)}  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]  # the picked deployment dict

    assert picked <= {"a", "c"}
    assert order[:2] == ["native", "legacy"]
    assert order.count("native") == order.count("legacy")
