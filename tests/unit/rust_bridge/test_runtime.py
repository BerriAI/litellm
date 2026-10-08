from __future__ import annotations

from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Final, Protocol

import pytest

from litellm.exceptions import APIError
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import bindings, catalog, configuration, runtime
from litellm.rust_bridge.catalog import Decision, Python, Route, RouteContext, Rust
from litellm.rust_bridge.lifecycle import Complete, Open, Yield
from litellm.rust_bridge.streams import Stream, SyncStream


class RustBridgeDeclined(Exception):
    pass


class RustUpstreamError(Exception):
    pass


@pytest.fixture(autouse=True)
def native_exceptions(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    native: Final = SimpleNamespace(RustBridgeDeclined=RustBridgeDeclined, RustUpstreamError=RustUpstreamError)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


class NativeFn(Protocol):
    def __call__(self) -> object: ...


CONTEXT: Final = RouteContext(Route.MESSAGES, provider="anthropic", model="model")
RUST: Final = "rust"
PYTHON: Final = "python"
OPTIONAL: Final = Rust()
REQUIRED: Final = Rust(required=True)
STAY: Final = Python("test keeps the call on Python")
ASYNC: Final = pytest.mark.parametrize("asynchronous", (False, True), ids=("run", "arun"))


class Recorder:
    """Records which side served the call; native raises ``native_effect`` or returns ``native_result``."""

    def __init__(self, native_effect: BaseException | None = None, native_result: object = RUST) -> None:
        self._native_effect: Final = native_effect
        self._native_result: Final = native_result
        self.calls: tuple[str, ...] = ()

    def rust(self) -> object:
        self.calls = (*self.calls, RUST)
        if self._native_effect is not None:
            raise self._native_effect
        return self._native_result

    def python(self) -> object:
        self.calls = (*self.calls, PYTHON)
        return PYTHON


def binding(native: NativeFn | None) -> bindings.NativeBinding[NativeFn]:
    bound: Final[bindings.NativeBinding[NativeFn]] = bindings.NativeBinding("_messages", validate=lambda _: None)
    bound.override(native)
    return bound


def never_loaded() -> bindings.NativeBinding[NativeFn]:
    def reject_load(value: object) -> NativeFn | None:
        pytest.fail("a Python decision must not load the native binding")

    return bindings.NativeBinding("_messages", validate=reject_load)


async def run(
    policy: Decision | Callable[[RouteContext], Decision] | None,
    calls: Recorder,
    *,
    asynchronous: bool,
    bound: bindings.NativeBinding[NativeFn] | None = None,
    native_missing: bool = False,
) -> object:
    selected: Final = bound if bound is not None else binding(None if native_missing else calls.rust)
    if not asynchronous:
        return runtime.run(CONTEXT, binding=selected, native=lambda fn: fn(), python=calls.python, policy=policy)

    async def native(fn: NativeFn) -> object:
        return fn()

    async def python() -> object:
        return calls.python()

    return await runtime.arun(CONTEXT, binding=selected, native=native, python=python, policy=policy)


async def run_native(policy: Decision, calls: Recorder, *, asynchronous: bool, native_missing: bool = False) -> object:
    bound: Final = binding(None if native_missing else calls.rust)
    if not asynchronous:
        return runtime.run_native(CONTEXT, binding=bound, native=lambda fn: fn(), policy=policy)

    async def native(fn: NativeFn) -> object:
        return fn()

    return await runtime.arun_native(CONTEXT, binding=bound, native=native, policy=policy)


@ASYNC
@pytest.mark.parametrize(
    ("policy", "expected"),
    ((STAY, (PYTHON,)), (OPTIONAL, (RUST,)), (REQUIRED, (RUST,))),
    ids=("python", "optional", "required"),
)
async def test_decision_selects_native_or_python(
    asynchronous: bool, policy: Decision, expected: tuple[str, ...]
) -> None:
    calls: Final = Recorder()

    assert await run(policy, calls, asynchronous=asynchronous) == expected[-1]
    assert calls.calls == expected


@ASYNC
@pytest.mark.parametrize("enabled", (False, True))
async def test_shipped_catalog_decides_when_no_policy_is_given(asynchronous: bool, enabled: bool) -> None:
    calls: Final = Recorder()
    configuration.rust(enabled)
    shipped: Final = catalog.decide(CONTEXT)

    assert await run(None, calls, asynchronous=asynchronous) == (RUST if isinstance(shipped, Rust) else PYTHON)


def test_policy_callable_receives_the_context() -> None:
    seen: Final[list[RouteContext]] = []

    def policy(context: RouteContext) -> Decision:
        seen.append(context)
        return REQUIRED

    result: Final = runtime.run(
        CONTEXT, binding=binding(Recorder().rust), native=lambda fn: fn(), python=lambda: PYTHON, policy=policy
    )

    assert result == RUST
    assert seen == [CONTEXT]


@ASYNC
async def test_python_decision_never_loads_the_native_binding(asynchronous: bool) -> None:
    calls: Final = Recorder()

    assert await run(STAY, calls, asynchronous=asynchronous, bound=never_loaded()) == PYTHON
    assert calls.calls == (PYTHON,)


@ASYNC
async def test_optional_native_decline_falls_back_to_python_once(asynchronous: bool) -> None:
    calls: Final = Recorder(RustBridgeDeclined("unsupported"))

    assert await run(OPTIONAL, calls, asynchronous=asynchronous) == PYTHON
    assert calls.calls == (RUST, PYTHON)


@ASYNC
async def test_optional_unavailable_native_falls_back_to_python(asynchronous: bool) -> None:
    calls: Final = Recorder()

    assert await run(OPTIONAL, calls, asynchronous=asynchronous, native_missing=True) == PYTHON
    assert calls.calls == (PYTHON,)


@ASYNC
@pytest.mark.parametrize("native_missing", (False, True), ids=("declined", "unavailable"))
async def test_python_fallback_result_is_not_marked_as_rust(asynchronous: bool, native_missing: bool) -> None:
    expected: Final = OCRResponse(pages=[], model="python")
    bound: Final = binding(None if native_missing else Recorder(RustBridgeDeclined("unsupported")).rust)

    async def anative(fn: NativeFn) -> object:
        return fn()

    async def apython() -> OCRResponse:
        return expected

    result: Final = (
        await runtime.arun(CONTEXT, binding=bound, native=anative, python=apython, policy=OPTIONAL)
        if asynchronous
        else runtime.run(CONTEXT, binding=bound, native=lambda fn: fn(), python=lambda: expected, policy=OPTIONAL)
    )

    assert result is expected
    assert get_hidden_params_dict(result) == {}


@ASYNC
@pytest.mark.parametrize("shape", ("model", "dict"))
@pytest.mark.parametrize("cache_key", (None, "test-cache-key"))
async def test_native_response_is_marked_as_rust_keeping_existing_metadata(
    asynchronous: bool, shape: str, cache_key: str | None
) -> None:
    hidden: Final = {
        "additional_headers": {"x-request-id": "upstream"},
        "response_cost": 0.01,
        **({"cache_key": cache_key} if cache_key is not None else {}),
    }
    response: Final[OCRResponse | dict[str, object]] = (
        OCRResponse(pages=[], model="native") if shape == "model" else {"content": "native", "_hidden_params": hidden}
    )
    if isinstance(response, OCRResponse):
        response._hidden_params = hidden  # pyright: ignore[reportPrivateUsage]  # seed SDK metadata to verify it survives native marking

    result: Final = await run(REQUIRED, Recorder(native_result=response), asynchronous=asynchronous)

    assert result is response
    assert get_hidden_params_dict(result) == {
        "response_cost": 0.01,
        "additional_headers": {
            "x-request-id": "upstream",
            "x-litellm-rust": "true",
            **({"x-litellm-cache-key": cache_key} if cache_key is not None else {}),
        },
        **({"cache_key": cache_key} if cache_key is not None else {}),
    }


async def test_native_none_result_reaches_the_caller() -> None:
    calls: Final = Recorder(native_result=None)

    assert await run(REQUIRED, calls, asynchronous=True) is None
    assert calls.calls == (RUST,)


class ScriptedStreamExecution:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._steps: Final = iter((*(Yield(chunk) for chunk in chunks), Complete(None)))
        self.closed = False

    def start(self) -> Open:
        return Open(None)

    def resume_value(self, value: object) -> Yield | Complete:
        return next(self._steps)

    def resume_error(self, error: BaseException) -> Complete:
        return Complete(None)

    def close(self) -> None:
        self.closed = True


@ASYNC
async def test_native_stream_is_marked_without_wrapping_or_consuming_it(asynchronous: bool) -> None:
    chunks: Final = (b"event: message_start\n\n", b"event: message_stop\n\n")
    execution: Final = ScriptedStreamExecution(chunks)
    stream: Final[Stream | SyncStream] = Stream(execution) if asynchronous else SyncStream(execution)

    result: Final = await run(REQUIRED, Recorder(native_result=stream), asynchronous=asynchronous)

    assert result is stream
    assert get_hidden_params_dict(result) == {"additional_headers": {"x-litellm-rust": "true"}}
    assert not execution.closed
    delivered: Final = tuple([chunk async for chunk in stream]) if isinstance(stream, Stream) else tuple(stream)
    assert delivered == chunks
    assert execution.closed


@ASYNC
@pytest.mark.parametrize("policy", (OPTIONAL, REQUIRED), ids=("optional", "required"))
async def test_upstream_error_maps_to_api_error_without_fallback(asynchronous: bool, policy: Decision) -> None:
    calls: Final = Recorder(RustUpstreamError(429, "rate limited"))

    with pytest.raises(APIError, match="rate limited") as caught:
        await run(policy, calls, asynchronous=asynchronous)

    assert caught.value.status_code == 429
    assert calls.calls == (RUST,)


@ASYNC
@pytest.mark.parametrize("policy", (OPTIONAL, REQUIRED), ids=("optional", "required"))
async def test_other_native_errors_propagate_without_fallback(asynchronous: bool, policy: Decision) -> None:
    failure: Final = ValueError("admitted")
    calls: Final = Recorder(failure)

    with pytest.raises(ValueError, match="admitted") as caught:
        await run(policy, calls, asynchronous=asynchronous)

    assert caught.value is failure
    assert calls.calls == (RUST,)


@ASYNC
async def test_required_route_rejects_unavailable_bridge(asynchronous: bool) -> None:
    calls: Final = Recorder()

    with pytest.raises(RuntimeError, match="Rust messages bridge is unavailable"):
        await run(REQUIRED, calls, asynchronous=asynchronous, native_missing=True)

    assert calls.calls == ()


@ASYNC
async def test_required_route_rejects_native_decline(asynchronous: bool) -> None:
    calls: Final = Recorder(RustBridgeDeclined("unsupported"))

    with pytest.raises(RuntimeError, match="declined the request: unsupported"):
        await run(REQUIRED, calls, asynchronous=asynchronous)

    assert calls.calls == (RUST,)


@ASYNC
async def test_route_without_python_runs_native(asynchronous: bool) -> None:
    calls: Final = Recorder()

    assert await run_native(REQUIRED, calls, asynchronous=asynchronous) == RUST
    assert calls.calls == (RUST,)


@ASYNC
@pytest.mark.parametrize(
    ("native_missing", "effect", "message"),
    (
        (True, None, "Rust messages bridge is unavailable"),
        (False, RustBridgeDeclined("unsupported"), "Rust messages bridge declined the request: unsupported"),
    ),
    ids=("unavailable", "declined"),
)
async def test_route_without_python_raises_when_native_cannot_serve_the_call(
    asynchronous: bool, native_missing: bool, effect: BaseException | None, message: str
) -> None:
    calls: Final = Recorder(effect)

    with pytest.raises(RuntimeError, match=message) as raised:
        await run_native(REQUIRED, calls, asynchronous=asynchronous, native_missing=native_missing)

    assert not isinstance(raised.value, runtime.NoPythonImplementationError)


@ASYNC
@pytest.mark.parametrize("policy", (STAY, OPTIONAL), ids=("python", "optional"))
async def test_route_without_python_rejects_decisions_that_could_select_python(
    asynchronous: bool, policy: Decision
) -> None:
    calls: Final = Recorder()

    with pytest.raises(runtime.NoPythonImplementationError, match="messages has no Python implementation"):
        await run_native(policy, calls, asynchronous=asynchronous)

    assert calls.calls == ()
