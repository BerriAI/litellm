import asyncio
import dataclasses
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeVar

import pytest

from litellm.rust_bridge import catalog, configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import AsyncPublicRoute, AsyncVariant, PublicRoute
from litellm.rust_bridge.runtime import NO_PYTHON, NoPythonImplementation, NoPythonImplementationError

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


ASYNC_FLAG: Final = "is_async"
FLAGGED: Final[Mapping[str, object]] = MappingProxyType({**KWARGS, ASYNC_FLAG: True})


@dataclass(frozen=True, slots=True)
class FlaggedCase:
    rules: Rules
    native_available: bool = True
    has_python: bool = True
    legacy_returns_coroutine: bool = True


def served_by_python(*args: object, **kwargs: object) -> ServedByPython:  # kwargs-ok: public pass-through shape
    return ServedByPython(args, kwargs)


async def deferred_python(*args: object, **kwargs: object) -> ServedByPython:  # kwargs-ok: public pass-through shape
    return ServedByPython(args, kwargs)


def legacy_python(
    case: FlaggedCase,
) -> Callable[..., ServedByPython | Coroutine[object, object, ServedByPython]] | NoPythonImplementation:
    if not case.has_python:
        return NO_PYTHON
    return deferred_python if case.legacy_returns_coroutine else served_by_python


def flagged_route(case: FlaggedCase, projector: Projector) -> PublicRoute[Request, object, ServedByPython, object]:
    async def native(request: Request) -> ServedByRust:
        return ServedByRust(request)

    return PublicRoute(
        route=ROUTE,
        request=projector,
        context=context,
        binding=binding(Case(rules=()), lambda _: pytest.fail("a flagged call must not use the sync binding")),
        python=legacy_python(case),
        asynchronous=AsyncVariant(ASYNC_FLAG, binding(Case(rules=(), native_available=case.native_available), native)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected"),
    (
        pytest.param(FlaggedCase(rules=(rule(Rollout.RUST_REQUIRED),)), ServedByRust(ANTHROPIC), id="rust"),
        pytest.param(
            FlaggedCase(rules=(rule(Rollout.RUST_REQUIRED),), has_python=False),
            ServedByRust(ANTHROPIC),
            id="no-python-rust",
        ),
        pytest.param(
            FlaggedCase(rules=(rule(Rollout.RUST_OPT_OUT),), native_available=False),
            ServedByPython(ARGS, FLAGGED),
            id="fallback-awaits-the-legacy-coroutine",
        ),
        pytest.param(
            FlaggedCase(rules=(rule(Rollout.RUST_OPT_OUT),), native_available=False, legacy_returns_coroutine=False),
            ServedByPython(ARGS, FLAGGED),
            id="fallback-keeps-a-legacy-value",
        ),
    ),
)
async def test_flagged_sync_call_selected_for_rust_returns_a_coroutine_of_the_sync_projection(
    case: FlaggedCase, expected: ServedByPython | ServedByRust
) -> None:
    projector: Final = Projector(ANTHROPIC)

    pending: Final = flagged_route(case, projector).run(ARGS, FLAGGED, rules=case.rules)

    assert asyncio.iscoroutine(pending)
    assert await pending == expected
    assert projector.seen == (ARGS, KWARGS)


@pytest.mark.parametrize("legacy_returns_coroutine", (False, True), ids=("value", "coroutine"))
@pytest.mark.parametrize(
    "rules",
    ((), (rule(Rollout.PYTHON_ONLY),), (rule(Rollout.RUST_REQUIRED, providers=frozenset({"openai"})),)),
    ids=("no-rules", "python-only", "provider-miss"),
)
def test_flagged_sync_call_selected_for_python_returns_exactly_what_the_legacy_call_returns(
    rules: Rules, legacy_returns_coroutine: bool
) -> None:
    route: Final = flagged_route(
        FlaggedCase(rules=rules, legacy_returns_coroutine=legacy_returns_coroutine), Projector(ANTHROPIC)
    )

    served: Final = route.run(ARGS, FLAGGED, rules=rules)

    if legacy_returns_coroutine:
        assert asyncio.iscoroutine(served)
        served.close()
        return
    assert served == ServedByPython(ARGS, FLAGGED)


@pytest.mark.parametrize("flag", (False, None))
def test_sync_call_without_a_true_async_flag_stays_synchronous(flag: bool | None) -> None:
    route: Final = dataclasses.replace(
        flagged_route(FlaggedCase(rules=()), Projector(ANTHROPIC)), python=served_by_python
    )

    assert route.run(ARGS, {**KWARGS, ASYNC_FLAG: flag}, rules=()) == ServedByPython(ARGS, {**KWARGS, ASYNC_FLAG: flag})


OUTER: Final = Route.OCR
NESTED_RUST: Final = (RouteRule(OUTER, Rollout.PYTHON_ONLY), rule(Rollout.RUST_REQUIRED))


def nested_routes() -> tuple[PublicRoute[Request, object, object], AsyncPublicRoute[Request, object, object]]:
    async def anative(request: Request) -> ServedByRust:
        return ServedByRust(request)

    async def apython(*args: object, **kwargs: object) -> ServedByPython:  # kwargs-ok: public pass-through shape
        return ServedByPython(args, kwargs)

    inner: Final[PublicRoute[Request, object, object]] = PublicRoute(
        route=ROUTE,
        request=lambda args, kwargs: ANTHROPIC,
        context=context,
        binding=binding(Case(rules=()), ServedByRust),
        python=lambda *args, **kwargs: ServedByPython(args, kwargs),
    )
    ainner: Final[AsyncPublicRoute[Request, object, object]] = AsyncPublicRoute(
        route=ROUTE,
        request=lambda args, kwargs: ANTHROPIC,
        context=context,
        binding=binding(Case(rules=()), anative),
        python=apython,
    )
    return inner, ainner


def outer_context(request: Request) -> RouteContext:
    return RouteContext(OUTER, provider=request.provider)


def serve_nested_sync() -> object:
    inner: Final = nested_routes()[0]

    def outer_python(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        return inner.run(args, kwargs)

    return PublicRoute(
        route=OUTER,
        request=lambda args, kwargs: ANTHROPIC,
        context=outer_context,
        binding=binding(Case(rules=()), ServedByRust),
        python=outer_python,
    ).run(ARGS, KWARGS, rules=NESTED_RUST)


async def serve_nested_async() -> object:
    ainner: Final = nested_routes()[1]

    async def outer_python(*args: object, **kwargs: object) -> object:  # kwargs-ok: public pass-through shape
        return await ainner.arun(args, kwargs)

    return await AsyncPublicRoute(
        route=OUTER,
        request=lambda args, kwargs: ANTHROPIC,
        context=outer_context,
        binding=binding(Case(rules=()), outer_python),
        python=outer_python,
    ).arun(ARGS, KWARGS, rules=NESTED_RUST)


@pytest.mark.asyncio
@SYNC_AND_ASYNC
async def test_a_call_made_while_serving_another_follows_the_outer_rules(asynchronous: bool) -> None:
    served: Final = await serve_nested_async() if asynchronous else serve_nested_sync()

    assert served == ServedByRust(ANTHROPIC)
    assert catalog.active_rules() is catalog.RULES
