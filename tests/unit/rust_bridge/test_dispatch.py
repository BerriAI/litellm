from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeVar

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import AsyncPublicRoute, PublicRoute
from litellm.rust_bridge.runtime import NO_PYTHON, NoPythonImplementationError

NativeT: Final = TypeVar("NativeT")

ROUTE: Final = Route.MESSAGES
ARGS: Final[tuple[object, ...]] = ("claude-model",)
KWARGS: Final[Mapping[str, object]] = MappingProxyType({"stream": True, "max_tokens": 8})


@dataclass(frozen=True, slots=True)
class Request:
    provider: str


ANTHROPIC: Final = Request(provider="anthropic")
OPENAI: Final = Request(provider="openai")


@dataclass(frozen=True, slots=True)
class ServedByPython:
    args: tuple[object, ...]
    kwargs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ServedByRust:
    request: Request


PYTHON: Final = ServedByPython(ARGS, KWARGS)


@dataclass(frozen=True, slots=True)
class Case:
    rules: Rules
    projected: Request | None = ANTHROPIC
    bypassed: bool = False
    native_available: bool = True
    has_python: bool = True
    switch: bool | None = None


@dataclass(frozen=True, slots=True)
class Outcome:
    served: ServedByPython | ServedByRust
    projected: bool


class Projector:
    def __init__(self, request: Request | None) -> None:
        self.request: Final = request
        self.seen: tuple[tuple[object, ...], Mapping[str, object]] | None = None

    def __call__(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> Request | None:
        assert self.seen is None, "a public call must be projected at most once"
        self.seen = (args, kwargs)
        return self.request


def context(request: Request) -> RouteContext:
    return RouteContext(ROUTE, provider=request.provider)


def binding(case: Case, native: NativeT) -> NativeBinding[NativeT]:
    bound: Final[NativeBinding[NativeT]] = NativeBinding(ROUTE.value, validate=lambda _: None)
    bound.override(native if case.native_available else None)
    return bound


def run_sync(case: Case, projector: Projector) -> object:
    return PublicRoute(
        route=ROUTE,
        request=projector,
        context=context,
        binding=binding(case, ServedByRust),
        python=(lambda *args, **kwargs: ServedByPython(args, kwargs)) if case.has_python else NO_PYTHON,
        bypass=lambda _: case.bypassed,
    ).run(ARGS, KWARGS, rules=case.rules)


async def run_async(case: Case, projector: Projector) -> object:
    async def native(request: Request) -> ServedByRust:
        return ServedByRust(request)

    async def python(*args: object, **kwargs: object) -> ServedByPython:  # kwargs-ok: public pass-through shape
        return ServedByPython(args, kwargs)

    return await AsyncPublicRoute(
        route=ROUTE,
        request=projector,
        context=context,
        binding=binding(case, native),
        python=python if case.has_python else NO_PYTHON,
        bypass=lambda _: case.bypassed,
    ).arun(ARGS, KWARGS, rules=case.rules)


async def serve(case: Case, *, asynchronous: bool) -> Outcome:
    projector: Final = Projector(case.projected)
    configuration.rust(case.switch)
    served: Final = await run_async(case, projector) if asynchronous else run_sync(case, projector)
    assert isinstance(served, ServedByPython | ServedByRust)
    assert projector.seen in (None, (ARGS, KWARGS)), "projection must see the public call unchanged"
    return Outcome(served=served, projected=projector.seen is not None)


@pytest.fixture(autouse=True)
def clean_rust_switch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    yield
    configuration.rust(None)


def rule(rollout: Rollout, *, route: Route = ROUTE, providers: frozenset[str] | None = None) -> RouteRule:
    return RouteRule(route, rollout, providers=providers)


ONLY_ANTHROPIC: Final = frozenset({"anthropic"})
SYNC_AND_ASYNC: Final = pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))


@pytest.mark.asyncio
@SYNC_AND_ASYNC
@pytest.mark.parametrize(
    ("case", "expected"),
    (
        pytest.param(Case(rules=()), Outcome(PYTHON, projected=False), id="no-rules"),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED, route=Route.OCR),)),
            Outcome(PYTHON, projected=False),
            id="rule-for-another-route",
        ),
        pytest.param(Case(rules=(rule(Rollout.PYTHON_ONLY),)), Outcome(PYTHON, projected=False), id="python-only"),
        pytest.param(
            Case(rules=(rule(Rollout.PYTHON_ONLY), rule(Rollout.RUST_REQUIRED))),
            Outcome(PYTHON, projected=False),
            id="first-match-python-shadows-rust",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),)),
            Outcome(ServedByRust(ANTHROPIC), projected=True),
            id="rust-required",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), switch=False),
            Outcome(ServedByRust(ANTHROPIC), projected=True),
            id="rust-required-ignores-switch-off",
        ),
        pytest.param(Case(rules=(rule(Rollout.RUST_OPT_IN),)), Outcome(PYTHON, projected=False), id="opt-in-default"),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_OPT_IN),), switch=True),
            Outcome(ServedByRust(ANTHROPIC), projected=True),
            id="opt-in-switched-on",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_OPT_OUT),)),
            Outcome(ServedByRust(ANTHROPIC), projected=True),
            id="opt-out-default",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_OPT_OUT),), switch=False),
            Outcome(PYTHON, projected=False),
            id="opt-out-switched-off",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_OPT_OUT),), native_available=False),
            Outcome(PYTHON, projected=True),
            id="optional-rust-unavailable-falls-back",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED, providers=ONLY_ANTHROPIC),)),
            Outcome(ServedByRust(ANTHROPIC), projected=True),
            id="provider-scoped-rust-matches",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED, providers=ONLY_ANTHROPIC),), projected=OPENAI),
            Outcome(PYTHON, projected=True),
            id="provider-scoped-rust-misses",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.PYTHON_ONLY, providers=ONLY_ANTHROPIC),)),
            Outcome(PYTHON, projected=False),
            id="provider-scoped-python-cannot-select-rust",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.PYTHON_ONLY, providers=ONLY_ANTHROPIC), rule(Rollout.RUST_REQUIRED))),
            Outcome(PYTHON, projected=True),
            id="provider-scoped-python-carves-out-of-rust",
        ),
        pytest.param(
            Case(
                rules=(rule(Rollout.PYTHON_ONLY, providers=ONLY_ANTHROPIC), rule(Rollout.RUST_REQUIRED)),
                projected=OPENAI,
            ),
            Outcome(ServedByRust(OPENAI), projected=True),
            id="provider-scoped-python-leaves-others-on-rust",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), projected=None),
            Outcome(PYTHON, projected=True),
            id="unprojectable-call-stays-on-python",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), bypassed=True),
            Outcome(PYTHON, projected=True),
            id="bypassed-call-stays-on-python",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), has_python=False),
            Outcome(ServedByRust(ANTHROPIC), projected=True),
            id="no-python-rust-required",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), has_python=False, switch=False),
            Outcome(ServedByRust(ANTHROPIC), projected=True),
            id="no-python-ignores-switch-off",
        ),
    ),
)
async def test_public_route_serves_each_call_from_the_backend_its_rules_select(
    case: Case, expected: Outcome, asynchronous: bool
) -> None:
    assert await serve(case, asynchronous=asynchronous) == expected


@pytest.mark.asyncio
@SYNC_AND_ASYNC
@pytest.mark.parametrize(
    ("case", "error", "message"),
    (
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), native_available=False),
            RuntimeError,
            "Rust messages bridge is unavailable",
            id="required-rust-unavailable",
        ),
        pytest.param(
            Case(rules=(), has_python=False),
            NoPythonImplementationError,
            "must resolve to RUST_REQUIRED",
            id="no-python-no-rules",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_OPT_OUT),), has_python=False),
            NoPythonImplementationError,
            "must resolve to RUST_REQUIRED",
            id="no-python-optional-rust",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED, providers=ONLY_ANTHROPIC),), projected=OPENAI, has_python=False),
            NoPythonImplementationError,
            "must resolve to RUST_REQUIRED",
            id="no-python-provider-scoped-miss",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), projected=None, has_python=False),
            NoPythonImplementationError,
            "must project to a native request",
            id="no-python-unprojectable-call",
        ),
        pytest.param(
            Case(rules=(rule(Rollout.RUST_REQUIRED),), bypassed=True, has_python=False),
            NoPythonImplementationError,
            "bypass predicate matches",
            id="no-python-bypassed-call",
        ),
    ),
)
async def test_public_route_fails_loudly_when_no_backend_can_serve_the_call(
    case: Case, error: type[Exception], message: str, asynchronous: bool
) -> None:
    with pytest.raises(error, match=message):
        await serve(case, asynchronous=asynchronous)


async def chunks() -> AsyncGenerator[int, None]:
    yield 1


RESULTS: Final = pytest.mark.parametrize(
    "make_result", (lambda: iter((1, 2)), chunks, lambda: None), ids=("stream", "async-stream", "websocket-none")
)
BACKENDS: Final = pytest.mark.parametrize("rules", ((), (rule(Rollout.RUST_REQUIRED),)), ids=("python", "rust"))


@RESULTS
@BACKENDS
def test_public_route_returns_the_backend_result_untouched(make_result: Callable[[], object], rules: Rules) -> None:
    result: Final = make_result()
    route: Final = PublicRoute(
        route=ROUTE,
        request=lambda args, kwargs: ANTHROPIC,
        context=context,
        binding=binding(Case(rules=rules), lambda _: result),
        python=lambda *args, **kwargs: result,
    )
    assert route.run(ARGS, KWARGS, rules=rules) is result


@pytest.mark.asyncio
@RESULTS
@BACKENDS
async def test_async_public_route_returns_the_backend_result_untouched(
    make_result: Callable[[], object], rules: Rules
) -> None:
    result: Final = make_result()

    async def native(_: Request) -> object:
        return result

    async def python(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        return result

    route: Final = AsyncPublicRoute(
        route=ROUTE,
        request=lambda args, kwargs: ANTHROPIC,
        context=context,
        binding=binding(Case(rules=rules), native),
        python=python,
    )
    assert await route.arun(ARGS, KWARGS, rules=rules) is result
