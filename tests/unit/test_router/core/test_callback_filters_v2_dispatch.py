from collections.abc import Callable, Sequence
from typing import Final

import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import (
    Abort,
    CallContext,
    CustomLoggerV2,
    Field,
    Hook,
    InvalidPatchError,
    Patch,
    on,
)
from litellm.integrations.custom_logger_v2.host import V2Outcome
from litellm.router_rust.steps.callbacks import OptionalChecks
from tests.unit.test_router.core.registry_helpers import MakeCore
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment

GROUP: Final = [deployment("g", "a"), deployment("g", "b"), deployment("g", "c")]

Deployments = list[dict[str, object]]  # mutable-ok: the filter contract


def _id(deployment: object) -> object:
    return deployment["model_info"]["id"] if isinstance(deployment, dict) else None  # pyright: ignore[reportUnknownVariableType]  # a deployment dict


def _ids(deployments: object) -> tuple[object, ...]:
    return tuple(map(_id, deployments)) if isinstance(deployments, list | tuple) else ()  # pyright: ignore[reportUnknownArgumentType]  # see above


class Narrow(CustomLogger):
    def __init__(
        self, name: str, journal: list[object], dropped: str = "", error: Exception | None = None
    ) -> None:  # mutable-ok: the shared call log
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger's untyped **kwargs
        self.name = name
        self.journal = journal
        self.dropped = dropped
        self.error = error
        self.seen: list[object] = []  # mutable-ok: each call's deployments

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: Deployments,
        messages: object,
        request_kwargs: object = None,
        parent_otel_span: object = None,
    ) -> Deployments:
        self.journal.append(self.name)
        self.seen.append(healthy_deployments)
        if self.error is not None:
            raise self.error
        return [d for d in healthy_deployments if _id(d) != self.dropped]


class Bystander(CustomLogger):
    def __init__(self, journal: list[object]) -> None:  # mutable-ok: the shared call log
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # see above
        self.journal = journal


class Picker(CustomLoggerV2):
    def __init__(
        self,
        journal: list[object],
        keep: Callable[[Sequence[object]], object] | None = None,
        outcome: Abort | None = None,
    ) -> None:  # mutable-ok: the shared call log
        self.journal = journal
        self.keep = keep
        self.outcome = outcome

    @on(
        Hook.filter_deployments,
        reads={Field.candidate_deployments, Field.model},
        writes={Field.candidate_deployments},
        may_abort=True,
    )
    async def pick(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        candidates: Final = getattr(view, "candidate_deployments")  # noqa: B009  # the view's declared field
        self.journal.append(("v2", getattr(view, "model"), _ids(candidates)))  # noqa: B009  # see above
        if self.outcome is not None:
            return self.outcome
        return None if self.keep is None else Patch().set(Field.candidate_deployments, self.keep(candidates))


class Inert(CustomLoggerV2):
    @on(Hook.pre_call)
    async def ignore(self, view: object, ctx: CallContext[object]) -> None:
        return None


def _checks(make_core: MakeCore) -> OptionalChecks:
    core: Final = make_core(model_list=list(GROUP))
    checks: Final[OptionalChecks] = core.__dict__["_checks"]  # pyright: ignore[reportAny]  # RustCore's own attribute
    return checks


def _install(host: ScriptedHost, callbacks: Sequence[object], inert: bool) -> None:
    host.lists["callbacks"].extend((*callbacks, Picker([])) if inert else callbacks)


def _deployments() -> Deployments:
    return [{"model_info": {"id": name}} for name in ("a", "b", "c")]


@pytest.mark.parametrize("inert", [False, True])
async def test_filters_run_in_list_order_and_each_sees_what_the_previous_one_kept(
    make_core: MakeCore, host: ScriptedHost, inert: bool
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    first: Final = Narrow("first", journal, dropped="a")
    second: Final = Narrow("second", journal, dropped="c")
    checks: Final = _checks(make_core)
    _install(host, [first, Bystander(journal), "a-string-callback", second], inert)
    deployments: Final = _deployments()

    remaining, preferred = await checks.filter_deployments("g", deployments, None, {"model": "g"})

    assert journal == ["first", "second"]
    assert first.seen[0] is not deployments
    assert _ids(first.seen[0]) == ("a", "b", "c")
    assert _ids(second.seen[0]) == ("b", "c")
    assert _ids(remaining) == ("b",)
    assert preferred is None


@pytest.mark.parametrize("inert", [False, True])
async def test_a_filter_error_reaches_the_caller_unchanged_and_stops_later_filters(
    make_core: MakeCore, host: ScriptedHost, inert: bool
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    error: Final = RuntimeError("a filter raised")
    checks: Final = _checks(make_core)
    _install(host, [Narrow("first", journal, error=error), Narrow("second", journal)], inert)

    with pytest.raises(RuntimeError) as raised:
        await checks.filter_deployments("g", _deployments(), None, {})

    assert raised.value is error
    assert journal == ["first"]


@pytest.mark.parametrize("inert", [False, True])
async def test_an_empty_result_is_returned_for_the_core_to_fail_the_pick(
    make_core: MakeCore, host: ScriptedHost, inert: bool
) -> None:
    checks: Final = _checks(make_core)
    _install(
        host, [Narrow("only", [], dropped="a"), Narrow("all", [], dropped="b"), Narrow("rest", [], dropped="c")], inert
    )

    remaining, _ = await checks.filter_deployments("g", _deployments(), None, {})

    assert remaining == []


async def test_a_v2_callback_between_two_filters_sees_the_first_cut_and_the_second_sees_its_own(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    second: Final = Narrow("second", journal)
    checks: Final = _checks(make_core)
    v2_callback: Final = Picker(journal, keep=lambda candidates: [d for d in candidates if _id(d) != "b"])
    _install(host, [Narrow("first", journal, dropped="a"), v2_callback, second], inert=False)

    remaining, _ = await checks.filter_deployments("g", _deployments(), None, {})

    assert journal == ["first", ("v2", "g", ("b", "c")), "second"]
    assert _ids(second.seen[0]) == ("c",)
    assert _ids(remaining) == ("c",)


@pytest.mark.parametrize(
    "keep",
    [
        pytest.param(lambda candidates: [*candidates, {"model_info": {"id": "z"}}], id="adds"),
        pytest.param(lambda candidates: [candidates[0], candidates[0]], id="repeats"),
    ],
)
async def test_a_v2_callback_may_only_narrow_the_candidates(
    make_core: MakeCore, host: ScriptedHost, keep: Callable[[Sequence[object]], object]
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    checks: Final = _checks(make_core)
    _install(host, [Picker(journal, keep=keep), Narrow("after", journal)], inert=False)

    with pytest.raises(InvalidPatchError, match="only narrow"):
        await checks.filter_deployments("g", _deployments(), None, {})

    assert journal == [("v2", "g", ("a", "b", "c"))]


async def test_a_v2_callback_abort_ends_the_filters_as_an_error(make_core: MakeCore, host: ScriptedHost) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    abort: Final = Abort(status=429, message="no capacity")
    checks: Final = _checks(make_core)
    _install(host, [Narrow("before", journal), Picker(journal, outcome=abort), Narrow("after", journal)], inert=False)

    with pytest.raises(V2Outcome) as raised:
        await checks.filter_deployments("g", _deployments(), None, {})

    assert raised.value.outcome is abort
    assert journal == ["before", ("v2", "g", ("a", "b", "c"))]


async def test_a_v2_callback_abort_carrying_an_exception_raises_it_as_a_filter_error_would(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    error: Final = RuntimeError("no capacity")
    checks: Final = _checks(make_core)
    _install(
        host,
        [Picker(journal, outcome=Abort(status=429, message="ignored", exception=error)), Narrow("after", journal)],
        inert=False,
    )

    with pytest.raises(RuntimeError) as raised:
        await checks.filter_deployments("g", _deployments(), None, {})

    assert raised.value is error
    assert journal == [("v2", "g", ("a", "b", "c"))]


async def test_only_a_v2_callback_that_filters_makes_the_core_ask_for_filters(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    checks: Final = _checks(make_core)
    host.lists["callbacks"].append(Inert())
    without_filter: Final = checks.has_filters()
    host.lists["callbacks"].append(Picker([]))

    assert (without_filter, checks.has_filters()) == (False, True)


async def test_the_pick_honours_a_v2_callback_between_legacy_filters(make_core: MakeCore, host: ScriptedHost) -> None:
    journal: Final[list[object]] = []  # mutable-ok: the shared call log
    v2_callback: Final = Picker(journal, keep=lambda candidates: [d for d in candidates if _id(d) != "c"])
    host.lists["callbacks"].extend((Narrow("first", journal, dropped="a"), v2_callback, Narrow("second", journal)))
    core: Final = make_core(model_list=list(GROUP))

    picked: Final = {_id(await core.select("g")) for _ in range(10)}

    assert picked == {"b"}
    assert journal[:3] == ["first", ("v2", "g", ("b", "c")), "second"]
