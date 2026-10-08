from __future__ import annotations

from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Final, Protocol

import pytest

from litellm.exceptions import APIError
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import bindings, configuration, runtime
from litellm.rust_bridge.catalog import Decision, Python, Route, RouteContext, Rust
from litellm.rust_bridge.lifecycle import Complete, Open, Yield
from litellm.rust_bridge.streams import Stream, SyncStream


class RustBridgeDeclined(Exception):
    pass


class RustUpstreamError(Exception):
    pass


@pytest.fixture(autouse=True)
def native_exceptions(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    native: Final = SimpleNamespace(
        RustBridgeDeclined=RustBridgeDeclined,
        RustUpstreamError=RustUpstreamError,
    )
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


class NativeFn(Protocol):
    def __call__(self) -> str: ...


CONTEXT: Final = RouteContext(Route.MESSAGES, provider="anthropic", model="model")
RUST: Final = "rust"
PYTHON: Final = "python"


def binding(native: NativeFn | None) -> bindings.NativeBinding[NativeFn]:
    bound: Final[bindings.NativeBinding[NativeFn]] = bindings.NativeBinding("_messages", validate=lambda _: None)
    bound.override(native)
    return bound


class Recorder:
    def __init__(self, native_effect: BaseException | None = None) -> None:
        self._native_effect: Final = native_effect
        self.calls: tuple[str, ...] = ()

    def rust(self) -> str:
        self.calls = (*self.calls, RUST)
        if self._native_effect is not None:
            raise self._native_effect
        return RUST

    def python(self) -> str:
        self.calls = (*self.calls, PYTHON)
        return PYTHON


def recorder(native_effect: BaseException | None = None) -> Recorder:
    return Recorder(native_effect)


OPTIONAL: Final = Rust()
REQUIRED: Final = Rust(required=True)
STAY: Final = Python("test keeps the call on Python")


def run(policy: Decision, calls: Recorder, *, native_missing: bool = False, context: RouteContext = CONTEXT) -> str:
    return runtime.run(
        context,
        binding=binding(None if native_missing else calls.rust),
        native=lambda fn: fn(),
        python=calls.python,
        policy=policy,
    )


@pytest.mark.parametrize(
    ("policy", "expected"),
    (
        (STAY, (PYTHON,)),
        (OPTIONAL, (RUST,)),
        (REQUIRED, (RUST,)),
    ),
)
def test_decision_selects_native_or_python(policy: Decision, expected: tuple[str, ...]) -> None:
    calls: Final = recorder()

    assert run(policy, calls) == expected[-1]
    assert calls.calls == expected


@pytest.mark.parametrize("switch", (None, "0", "1"))
def test_shipped_policy_is_consulted_when_none_is_given(monkeypatch: pytest.MonkeyPatch, switch: str | None) -> None:
    calls: Final = recorder()
    if switch is not None:
        monkeypatch.setenv("LITELLM_RUST", switch)

    result: Final = runtime.run(CONTEXT, binding=binding(calls.rust), native=lambda fn: fn(), python=calls.python)

    assert result == (RUST if switch == "1" else PYTHON)


def test_policy_callable_receives_the_context() -> None:
    calls: Final = recorder()
    seen: Final[list[RouteContext]] = []

    def policy(context: RouteContext) -> Decision:
        seen.append(context)
        return REQUIRED

    assert (
        runtime.run(CONTEXT, binding=binding(calls.rust), native=lambda fn: fn(), python=calls.python, policy=policy)
        == RUST
    )
    assert seen == [CONTEXT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    (
        RouteContext(Route.CHAT_COMPLETIONS, provider="anthropic"),
        RouteContext(Route.CHAT_COMPLETIONS, provider="bedrock"),
        RouteContext(Route.RESPONSES, provider="openai"),
        RouteContext(Route.TRANSCRIPTION, provider="openai"),
    ),
)
async def test_shipped_python_routes_never_load_native(monkeypatch: pytest.MonkeyPatch, context: RouteContext) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    configuration.rust(True)
    calls: Final = recorder()

    def reject_load(value: object) -> NativeFn | None:
        pytest.fail("Python-only dispatch must not load a native binding")

    bound: Final = bindings.NativeBinding("_messages", validate=reject_load)

    async def native(fn: NativeFn) -> str:
        return fn()

    async def python() -> str:
        return calls.python()

    assert runtime.run(context, binding=bound, native=lambda fn: fn(), python=calls.python) == PYTHON
    assert await runtime.arun(context, binding=bound, native=native, python=python) == PYTHON
    assert calls.calls == (PYTHON, PYTHON)


def test_native_decline_falls_back_to_python_once() -> None:
    calls: Final = recorder(RustBridgeDeclined("unsupported"))

    assert run(OPTIONAL, calls) == "python"
    assert calls.calls == (RUST, PYTHON)


def test_unavailable_native_falls_back_to_python() -> None:
    calls: Final = recorder()

    assert run(OPTIONAL, calls, native_missing=True) == "python"
    assert calls.calls == (PYTHON,)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_python_fallback_does_not_claim_rust_execution(missing: bool) -> None:
    calls: Final = recorder(RustBridgeDeclined("unsupported"))
    bound: Final = binding(None if missing else calls.rust)
    expected: Final = OCRResponse(pages=[], model="python")

    def native(fn: NativeFn) -> OCRResponse:
        fn()
        pytest.fail("native must decline before constructing a response")

    async def anative(fn: NativeFn) -> OCRResponse:
        return native(fn)

    async def python() -> OCRResponse:
        return expected

    assert runtime.run(CONTEXT, binding=bound, native=native, python=lambda: expected, policy=OPTIONAL) is expected
    assert await runtime.arun(CONTEXT, binding=bound, native=anative, python=python, policy=OPTIONAL) is expected
    assert get_hidden_params_dict(expected) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("model", "dict"))
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("cache_key", (None, "test-cache-key"))
async def test_native_response_marker_reaches_caller_with_existing_metadata(
    shape: str, asynchronous: bool, cache_key: str | None
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
    bound: Final[bindings.NativeBinding[Callable[[], object]]] = bindings.NativeBinding("ocr", validate=lambda _: None)
    bound.override(lambda: response)

    def python() -> object:
        pytest.fail("native success must not fall back")

    async def anative(fn: Callable[[], object]) -> object:
        return fn()

    async def apython() -> object:
        return python()

    result: Final = (
        await runtime.arun(CONTEXT, binding=bound, native=anative, python=apython, policy=REQUIRED)
        if asynchronous
        else runtime.run(CONTEXT, binding=bound, native=lambda fn: fn(), python=python, policy=REQUIRED)
    )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_stream_marker_reaches_caller_without_wrapping_or_consuming_the_stream(
    asynchronous: bool,
) -> None:
    chunks: Final = (b"event: message_start\n\n", b"event: message_stop\n\n")
    execution: Final = ScriptedStreamExecution(chunks)
    stream: Final[Stream | SyncStream] = Stream(execution) if asynchronous else SyncStream(execution)
    bound: Final[bindings.NativeBinding[Callable[[], object]]] = bindings.NativeBinding(
        "messages", validate=lambda _: None
    )
    bound.override(lambda: stream)

    def python() -> object:
        pytest.fail("native success must not fall back")

    async def anative(fn: Callable[[], object]) -> object:
        return fn()

    async def apython() -> object:
        return python()

    result: Final = (
        await runtime.arun(CONTEXT, binding=bound, native=anative, python=apython, policy=REQUIRED)
        if asynchronous
        else runtime.run(CONTEXT, binding=bound, native=lambda fn: fn(), python=python, policy=REQUIRED)
    )
    assert result is stream
    assert get_hidden_params_dict(result) == {"additional_headers": {"x-litellm-rust": "true"}}
    assert not execution.closed
    delivered: Final = tuple([chunk async for chunk in result]) if isinstance(result, Stream) else tuple(result)
    assert delivered == chunks
    assert execution.closed


def test_upstream_error_maps_to_api_error_without_fallback() -> None:
    calls: Final = recorder(RustUpstreamError(429, "rate limited"))

    with pytest.raises(APIError, match="rate limited") as caught:
        run(OPTIONAL, calls)

    assert caught.value.status_code == 429
    assert calls.calls == (RUST,)


def test_other_native_errors_propagate_without_fallback() -> None:
    failure: Final = ValueError("admitted")
    calls: Final = recorder(failure)

    with pytest.raises(ValueError, match="admitted") as caught:
        run(OPTIONAL, calls)

    assert caught.value is failure
    assert calls.calls == (RUST,)


def test_required_route_rejects_unavailable_bridge() -> None:
    calls: Final = recorder()

    with pytest.raises(RuntimeError, match="Rust messages bridge is unavailable"):
        run(REQUIRED, calls, native_missing=True)

    assert PYTHON not in calls.calls


def test_required_route_rejects_native_decline() -> None:
    calls: Final = recorder(RustBridgeDeclined("unsupported"))

    with pytest.raises(RuntimeError, match="declined the request: unsupported"):
        run(REQUIRED, calls)

    assert PYTHON not in calls.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native_effect", "native_missing", "expected"),
    (
        (None, False, (RUST,)),
        (RustBridgeDeclined("unsupported"), False, (RUST, PYTHON)),
        (None, True, (PYTHON,)),
    ),
)
async def test_arun_mirrors_sync_fallback(
    native_effect: BaseException | None, native_missing: bool, expected: tuple[str, ...]
) -> None:
    calls: Final = recorder(native_effect)

    async def native(fn: NativeFn) -> str:
        return fn()

    async def python() -> str:
        return calls.python()

    result: Final = await runtime.arun(
        CONTEXT,
        binding=binding(None if native_missing else calls.rust),
        native=native,
        python=python,
        policy=OPTIONAL,
    )

    assert result == expected[-1]
    assert calls.calls == expected


@pytest.mark.asyncio
async def test_arun_required_route_rejects_unavailable_bridge() -> None:
    async def python() -> str:
        pytest.fail("fallback must not run")

    with pytest.raises(RuntimeError, match="is unavailable"):
        await runtime.arun(
            CONTEXT,
            binding=binding(None),
            native=lambda fn: python(),
            python=python,
            policy=REQUIRED,
        )


@pytest.mark.asyncio
async def test_arun_upstream_error_maps_to_api_error_without_fallback() -> None:
    calls: Final = recorder(RustUpstreamError(503, "upstream unavailable"))

    async def native(fn: NativeFn) -> str:
        return fn()

    async def python() -> str:
        return calls.python()

    with pytest.raises(APIError, match="upstream unavailable") as caught:
        await runtime.arun(
            CONTEXT,
            binding=binding(calls.rust),
            native=native,
            python=python,
            policy=OPTIONAL,
        )

    assert caught.value.status_code == 503
    assert calls.calls == (RUST,)


async def run_without_python(
    policy: Decision,
    calls: Recorder,
    *,
    asynchronous: bool,
    native_missing: bool = False,
    context: RouteContext = CONTEXT,
) -> str:
    bound: Final = binding(None if native_missing else calls.rust)
    if not asynchronous:
        return runtime.run(context, binding=bound, native=lambda fn: fn(), python=runtime.NO_PYTHON, policy=policy)

    async def native(fn: NativeFn) -> str:
        return fn()

    return await runtime.arun(context, binding=bound, native=native, python=runtime.NO_PYTHON, policy=policy)


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("switch", (None, False, True))
async def test_route_without_python_runs_native_whatever_the_rust_switch(
    asynchronous: bool, switch: bool | None
) -> None:
    calls: Final = recorder()
    if switch is not None:
        configuration.rust(switch)

    assert await run_without_python(REQUIRED, calls, asynchronous=asynchronous) == RUST
    assert calls.calls == (RUST,)


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize(
    ("native_missing", "effect", "message"),
    (
        (True, None, "Rust messages bridge is unavailable"),
        (False, RustBridgeDeclined("unsupported"), "Rust messages bridge declined the request: unsupported"),
    ),
)
async def test_route_without_python_raises_when_native_cannot_serve_the_call(
    asynchronous: bool, native_missing: bool, effect: BaseException | None, message: str
) -> None:
    calls: Final = recorder(effect)

    with pytest.raises(RuntimeError, match=message) as raised:
        await run_without_python(REQUIRED, calls, asynchronous=asynchronous, native_missing=native_missing)

    assert not isinstance(raised.value, runtime.NoPythonImplementationError)


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("switch", (None, False, True))
@pytest.mark.parametrize("policy", (STAY, OPTIONAL), ids=("python", "optional"))
async def test_route_without_python_rejects_decisions_that_could_select_python(
    asynchronous: bool, switch: bool | None, policy: Decision
) -> None:
    calls: Final = recorder()
    if switch is not None:
        configuration.rust(switch)

    with pytest.raises(runtime.NoPythonImplementationError, match="messages has no Python implementation"):
        await run_without_python(policy, calls, asynchronous=asynchronous)

    assert calls.calls == ()
