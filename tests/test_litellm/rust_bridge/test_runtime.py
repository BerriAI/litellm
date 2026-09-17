from __future__ import annotations

from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Final, Protocol

import pytest

from litellm.exceptions import APIError
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import bindings, configuration, runtime
from litellm.rust_bridge.catalog import Context, Delivery, Route, Rule
from litellm.rust_bridge.configuration import Rollout


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


CONTEXT: Final = Context(Route.MESSAGES, provider="anthropic", model="model")
RUST: Final = "rust"
PYTHON: Final = "python"


def binding(native: NativeFn | None) -> bindings.NativeBinding[NativeFn]:
    bound: Final[bindings.NativeBinding[NativeFn]] = bindings.NativeBinding("_messages", validate=lambda _: None)
    bound.override(native)
    return bound


def rules(rollout: Rollout) -> tuple[Rule, ...]:
    return (Rule(Route.MESSAGES, rollout, providers=frozenset({"anthropic"})),)


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


def run(rollout: Rollout, calls: Recorder, *, native_missing: bool = False, context: Context = CONTEXT) -> str:
    return runtime.run(
        context,
        binding=binding(None if native_missing else calls.rust),
        native=lambda fn: fn(),
        python=calls.python,
        rules=rules(rollout),
    )


@pytest.mark.parametrize(
    ("rollout", "switch", "expected"),
    (
        (Rollout.PYTHON_ONLY, None, (PYTHON,)),
        (Rollout.PYTHON_ONLY, True, (PYTHON,)),
        (Rollout.RUST_OPT_IN, None, (PYTHON,)),
        (Rollout.RUST_OPT_IN, True, (RUST,)),
        (Rollout.RUST_OPT_OUT, None, (RUST,)),
        (Rollout.RUST_OPT_OUT, False, (PYTHON,)),
        (Rollout.RUST_REQUIRED, None, (RUST,)),
        (Rollout.RUST_REQUIRED, False, (RUST,)),
    ),
)
def test_rollout_and_switch_select_native_or_python(
    rollout: Rollout, switch: bool | None, expected: tuple[str, ...]
) -> None:
    calls: Final = recorder()
    if switch is not None:
        configuration.rust(switch)

    assert run(rollout, calls) == expected[-1]
    assert calls.calls == expected


def test_environment_switch_enables_opt_in_route(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Final = recorder()
    monkeypatch.setenv("LITELLM_RUST", "1")

    assert run(Rollout.RUST_OPT_IN, calls) == "rust"
    assert calls.calls == (RUST,)


@pytest.mark.parametrize(
    ("rollout", "environment", "switch", "expected"),
    (
        (Rollout.RUST_OPT_IN, "0", True, (PYTHON,)),
        (Rollout.RUST_OPT_OUT, "0", True, (PYTHON,)),
        (Rollout.RUST_OPT_IN, "1", False, (RUST,)),
        (Rollout.RUST_OPT_OUT, "1", False, (RUST,)),
        (Rollout.RUST_REQUIRED, "0", False, (RUST,)),
        (Rollout.PYTHON_ONLY, "1", True, (PYTHON,)),
    ),
)
def test_environment_switch_wins_over_process_switch(
    monkeypatch: pytest.MonkeyPatch,
    rollout: Rollout,
    environment: str,
    switch: bool,
    expected: tuple[str, ...],
) -> None:
    calls: Final = recorder()
    monkeypatch.setenv("LITELLM_RUST", environment)
    configuration.rust(switch)

    assert run(rollout, calls) == expected[-1]
    assert calls.calls == expected


def test_context_outside_rule_stays_on_python() -> None:
    calls: Final = recorder()
    configuration.rust(True)

    assert run(Rollout.RUST_REQUIRED, calls, context=Context(Route.MESSAGES, provider="openai")) == "python"
    assert run(Rollout.RUST_REQUIRED, calls, context=Context(Route.RESPONSES, provider="anthropic")) == "python"
    assert calls.calls == (PYTHON, PYTHON)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    (
        Context(Route.CHAT_COMPLETIONS, provider="anthropic"),
        Context(Route.CHAT_COMPLETIONS, provider="bedrock"),
        Context(Route.MESSAGES, provider="anthropic"),
        Context(Route.RESPONSES, provider="openai"),
        Context(Route.TRANSCRIPTION, provider="openai"),
    ),
)
@pytest.mark.parametrize("delivery", tuple(Delivery))
async def test_shipped_python_routes_never_load_native(
    monkeypatch: pytest.MonkeyPatch, context: Context, delivery: Delivery
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    configuration.rust(True)
    calls: Final = recorder()
    request: Final = Context(context.route, provider=context.provider, delivery=delivery)

    def reject_load(value: object) -> NativeFn | None:
        pytest.fail("Python-only dispatch must not load a native binding")

    bound: Final = bindings.NativeBinding("_messages", validate=reject_load)

    async def native(fn: NativeFn) -> str:
        return fn()

    async def python() -> str:
        return calls.python()

    assert runtime.run(request, binding=bound, native=lambda fn: fn(), python=calls.python) == PYTHON
    assert await runtime.arun(request, binding=bound, native=native, python=python) == PYTHON
    assert calls.calls == (PYTHON, PYTHON)


def test_native_decline_falls_back_to_python_once() -> None:
    calls: Final = recorder(RustBridgeDeclined("unsupported"))

    assert run(Rollout.RUST_OPT_OUT, calls) == "python"
    assert calls.calls == (RUST, PYTHON)


def test_unavailable_native_falls_back_to_python() -> None:
    calls: Final = recorder()

    assert run(Rollout.RUST_OPT_OUT, calls, native_missing=True) == "python"
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

    assert (
        runtime.run(CONTEXT, binding=bound, native=native, python=lambda: expected, rules=rules(Rollout.RUST_OPT_OUT))
        is expected
    )
    assert (
        await runtime.arun(CONTEXT, binding=bound, native=anative, python=python, rules=rules(Rollout.RUST_OPT_OUT))
        is expected
    )
    assert get_hidden_params_dict(expected) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("model", "dict"))
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_response_marker_reaches_caller_with_existing_metadata(shape: str, asynchronous: bool) -> None:
    hidden: Final = {"additional_headers": {"x-request-id": "upstream"}, "response_cost": 0.01}
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
        await runtime.arun(CONTEXT, binding=bound, native=anative, python=apython, rules=rules(Rollout.RUST_REQUIRED))
        if asynchronous
        else runtime.run(
            CONTEXT, binding=bound, native=lambda fn: fn(), python=python, rules=rules(Rollout.RUST_REQUIRED)
        )
    )
    assert result is response
    assert get_hidden_params_dict(result) == {
        "response_cost": 0.01,
        "additional_headers": {"x-request-id": "upstream", "x-litellm-rust": "true"},
    }


def test_upstream_error_maps_to_api_error_without_fallback() -> None:
    calls: Final = recorder(RustUpstreamError(429, "rate limited"))

    with pytest.raises(APIError, match="rate limited") as caught:
        run(Rollout.RUST_OPT_OUT, calls)

    assert caught.value.status_code == 429
    assert calls.calls == (RUST,)


def test_other_native_errors_propagate_without_fallback() -> None:
    failure: Final = ValueError("admitted")
    calls: Final = recorder(failure)

    with pytest.raises(ValueError, match="admitted") as caught:
        run(Rollout.RUST_OPT_OUT, calls)

    assert caught.value is failure
    assert calls.calls == (RUST,)


def test_required_route_rejects_unavailable_bridge() -> None:
    calls: Final = recorder()

    with pytest.raises(RuntimeError, match="Rust messages bridge is unavailable"):
        run(Rollout.RUST_REQUIRED, calls, native_missing=True)

    assert PYTHON not in calls.calls


def test_required_route_rejects_native_decline() -> None:
    calls: Final = recorder(RustBridgeDeclined("unsupported"))

    with pytest.raises(RuntimeError, match="declined the request: unsupported"):
        run(Rollout.RUST_REQUIRED, calls)

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
        rules=rules(Rollout.RUST_OPT_OUT),
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
            rules=rules(Rollout.RUST_REQUIRED),
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
            rules=rules(Rollout.RUST_OPT_OUT),
        )

    assert caught.value.status_code == 503
    assert calls.calls == (RUST,)
