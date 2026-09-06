from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.exceptions import APIError, AuthenticationError, InternalServerError, RateLimitError
from litellm.rust_bridge import bindings, loader, runtime


class RustBridgeDeclined(Exception):
    pass


class RustUpstreamError(Exception):
    pass


@pytest.fixture(autouse=True)
def native_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bindings,
        "get_native_bridge",
        lambda: SimpleNamespace(
            RustBridgeDeclined=RustBridgeDeclined,
            RustUpstreamError=RustUpstreamError,
        ),
    )


def context() -> runtime.BridgeErrorContext:
    return runtime.BridgeErrorContext(provider="anthropic", model="model")


def enabled() -> bool:
    return True


@dataclass(frozen=True, slots=True)
class FallbackCase:
    process_enabled: bool | None = None
    eligible: bool = True
    binding_available: bool = True
    declined: bool = False
    expected_events: tuple[str, ...] = ()


FALLBACK_CASES: Final = (
    pytest.param(
        FallbackCase(process_enabled=False, expected_events=("python",)),
        id="process-disabled",
    ),
    pytest.param(
        FallbackCase(eligible=False, expected_events=("python",)),
        id="request-ineligible",
    ),
    pytest.param(
        FallbackCase(binding_available=False, expected_events=("load", "python")),
        id="bridge-unavailable",
    ),
    pytest.param(
        FallbackCase(declined=True, expected_events=("load", "prepare", "rust", "python")),
        id="bridge-declined",
    ),
)


@pytest.mark.parametrize("case", FALLBACK_CASES)
def test_invoke_falls_back_only_before_provider_success(case: FallbackCase) -> None:
    events: list[str] = []

    def load() -> object | None:
        events.append("load")
        return object() if case.binding_available else None

    def call(_binding: object, _request: object) -> int:
        events.append("rust")
        if case.declined:
            raise RustBridgeDeclined("unsupported")
        return 3

    bridge: Final = runtime.EndpointBinding(
        route="messages", load=load, enabled=lambda: case.process_enabled is not False
    )
    result: Final = bridge.invoke(
        prepare=lambda: events.append("prepare"),
        call=call,
        fallback=lambda: events.append("python") or "fallback",
        adapt=str,
        error_context=context(),
        eligible=case.eligible,
    )

    assert result == "fallback"
    assert tuple(events) == case.expected_events


@pytest.mark.asyncio
@pytest.mark.parametrize("case", FALLBACK_CASES)
async def test_ainvoke_matches_sync_fallback_contract(case: FallbackCase) -> None:
    events: list[str] = []

    def load() -> object | None:
        events.append("load")
        return object() if case.binding_available else None

    async def call(_binding: object, _request: object) -> int:
        events.append("rust")
        if case.declined:
            raise RustBridgeDeclined("unsupported")
        return 3

    async def fallback() -> str:
        events.append("python")
        return "fallback"

    bridge: Final = runtime.EndpointBinding(
        route="messages", load=load, enabled=lambda: case.process_enabled is not False
    )
    result: Final = await bridge.ainvoke(
        prepare=lambda: events.append("prepare"),
        call=call,
        fallback=fallback,
        adapt=str,
        error_context=context(),
        eligible=case.eligible,
    )

    assert result == "fallback"
    assert tuple(events) == case.expected_events


def test_invoke_adapts_native_success_without_fallback() -> None:
    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    result: Final = bridge.invoke(
        prepare=lambda: 3,
        call=lambda _binding, request: request * 2,
        fallback=lambda: pytest.fail("fallback must not run"),
        adapt=lambda value: f"adapted-{value}",
        error_context=context(),
    )

    assert result == "adapted-6"


@pytest.mark.asyncio
async def test_ainvoke_adapts_native_success_without_fallback() -> None:
    async def call(_binding: object, request: int) -> int:
        return request * 2

    async def fallback() -> str:
        pytest.fail("fallback must not run")

    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)
    result: Final = await bridge.ainvoke(
        prepare=lambda: 3,
        call=call,
        fallback=fallback,
        adapt=lambda value: f"adapted-{value}",
        error_context=context(),
    )

    assert result == "adapted-6"


@pytest.mark.parametrize(
    ("error", "expected_type", "expected_status", "expected_message"),
    (
        pytest.param(RustUpstreamError(401, "unauthorized"), AuthenticationError, 401, "unauthorized", id="auth"),
        pytest.param(RustUpstreamError(429, "rate limited"), RateLimitError, 429, "rate limited", id="rate-limit"),
        pytest.param(RustUpstreamError(500, "failed"), InternalServerError, 500, "failed", id="server-error"),
        pytest.param(RustUpstreamError(0, "connection reset"), APIError, 500, "connection reset", id="transport"),
        pytest.param(RustUpstreamError(403, "forbidden"), APIError, 403, "forbidden", id="other-status"),
    ),
)
@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_upstream_failure_maps_to_api_error_without_fallback(
    asynchronous: bool,
    error: RustUpstreamError,
    expected_type: type[BaseException],
    expected_status: int,
    expected_message: str,
) -> None:
    def fail(_binding: object, _request: object) -> object:
        raise error

    async def afail(binding: object, request: object) -> object:
        return fail(binding, request)

    async def fallback() -> str:
        pytest.fail("fallback must not run")

    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    async def invoke() -> None:
        if asynchronous:
            await bridge.ainvoke(
                prepare=lambda: None, call=afail, fallback=fallback, adapt=str, error_context=context()
            )
        else:
            bridge.invoke(
                prepare=lambda: None,
                call=fail,
                fallback=lambda: pytest.fail("fallback must not run"),
                adapt=str,
                error_context=context(),
            )

    with pytest.raises(expected_type, match=expected_message) as caught:
        await invoke()

    assert type(caught.value) is expected_type
    assert caught.value.status_code == expected_status
    assert caught.value.llm_provider == "anthropic"
    assert caught.value.model == "model"
    assert caught.value.__cause__ is error


@pytest.mark.asyncio
async def test_async_upstream_failure_maps_to_api_error_without_fallback() -> None:
    async def fail(_binding: object, _request: object) -> object:
        raise RustUpstreamError(503, "overloaded")

    async def fallback() -> object:
        pytest.fail("fallback must not run")

    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    with pytest.raises(APIError, match="overloaded") as caught:
        await bridge.ainvoke(prepare=lambda: None, call=fail, fallback=fallback, adapt=str, error_context=context())

    assert caught.value.status_code == 503


def test_unknown_failure_is_preserved_without_fallback() -> None:
    error: Final = RuntimeError("unknown")
    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    with pytest.raises(RuntimeError, match="unknown") as caught:
        bridge.invoke(
            prepare=lambda: None,
            call=lambda _binding, _request: (_ for _ in ()).throw(error),
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            error_context=context(),
        )

    assert caught.value is error


@pytest.mark.parametrize(
    ("process_enabled", "binding_available", "declined", "expected_message"),
    (
        pytest.param(False, True, False, "native messages endpoint is disabled", id="disabled"),
        pytest.param(None, False, False, "native messages endpoint is unavailable", id="unavailable"),
        pytest.param(
            None,
            True,
            True,
            "native messages endpoint declined the request: unsupported",
            id="declined",
        ),
    ),
)
def test_require_explains_why_rust_did_not_handle_request(
    process_enabled: bool | None,
    binding_available: bool,
    declined: bool,
    expected_message: str,
) -> None:
    def call(_binding: object, _request: object) -> object:
        if declined:
            raise RustBridgeDeclined("unsupported")
        return object()

    bridge: Final = runtime.EndpointBinding(
        route="messages",
        load=object if binding_available else lambda: None,
        enabled=lambda: process_enabled is not False,
    )

    with pytest.raises(RuntimeError, match=f"^{expected_message}$"):
        bridge.require(
            prepare=lambda: None,
            call=call,
            adapt=str,
            error_context=context(),
        )


@pytest.mark.parametrize(
    ("state", "expected", "expected_events"),
    (
        pytest.param("disabled", False, (), id="disabled"),
        pytest.param("ineligible", False, (), id="ineligible"),
        pytest.param("unavailable", False, ("load",), id="unavailable"),
        pytest.param("available", True, ("load",), id="available"),
    ),
)
def test_can_attempt_only_enabled_available_requests(
    state: str,
    expected: bool,
    expected_events: tuple[str, ...],
) -> None:
    events: list[str] = []

    def load() -> object | None:
        events.append("load")
        return None if state == "unavailable" else object()

    bridge: Final = runtime.EndpointBinding(route="messages", load=load, enabled=lambda: state != "disabled")

    assert (
        bridge.can_attempt(
            eligible=state != "ineligible",
        )
        is expected
    )
    assert tuple(events) == expected_events


def test_native_endpoint_applies_partial_overrides_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    def native_sync() -> str:
        return "native"

    async def native_async() -> str:
        return "native async"

    def replacement_sync() -> str:
        return "replacement"

    monkeypatch.setattr(
        bindings,
        "get_native_bridge",
        lambda: SimpleNamespace(
            chat_completions=native_sync,
            achat_completions=native_async,
            ready_endpoints={"test": frozenset({"callbacks"})},
        ),
    )
    endpoint: Final[runtime.EndpointDispatch[object, object]] = runtime.EndpointDispatch.native(
        route="test",
        sync=lambda native: native.chat_completions,
        asynchronous=lambda native: native.achat_completions,
        enabled=enabled,
    )

    assert endpoint.sync.load() is native_sync
    assert endpoint.asynchronous.load() is native_async
    endpoint.override(sync=replacement_sync)
    assert endpoint.sync.load() is replacement_sync
    assert endpoint.asynchronous.load() is native_async
    endpoint.override(asynchronous=None)
    assert endpoint.sync.load() is replacement_sync
    assert endpoint.asynchronous.load() is None
    endpoint.reset()
    assert endpoint.sync.load() is native_sync
    assert endpoint.asynchronous.load() is native_async


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_response_adaptation_failure_never_authorizes_fallback(asynchronous: bool) -> None:
    def adapt(value: str) -> str:
        assert value == "provider response"
        raise RustBridgeDeclined("adapter failed after provider response")

    async def native(binding: object, request: object) -> str:
        return "provider response"

    async def fallback() -> str:
        pytest.fail("a received response must not be retried")

    bridge = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    async def invoke() -> None:
        if asynchronous:
            await bridge.ainvoke(
                prepare=lambda: None, call=native, fallback=fallback, adapt=adapt, error_context=context()
            )
        else:
            bridge.invoke(
                prepare=lambda: None,
                call=lambda binding, request: "provider response",
                fallback=lambda: pytest.fail("a received response must not be retried"),
                adapt=adapt,
                error_context=context(),
            )

    with pytest.raises(RustBridgeDeclined, match="adapter failed"):
        await invoke()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("available, accepted", ((False, False), (True, False), (True, True)))
async def test_preflight_runs_after_binding_selection_before_preparation(
    asynchronous: bool, available: bool, accepted: bool
) -> None:
    events: list[str] = []

    def load() -> object | None:
        events.append("load")
        return object() if available else None

    def preflight() -> runtime.PythonFallback | None:
        events.append("preflight")
        return None if accepted else runtime.PythonFallback(runtime.PythonFallbackReason.NATIVE_DECLINED)

    def prepare() -> int:
        events.append("prepare")
        return 7

    def call(binding: object, request: int) -> int:
        events.append("native")
        return request

    async def acall(binding: object, request: int) -> int:
        return call(binding, request)

    def fallback() -> str:
        events.append("python")
        return "3"

    async def afallback() -> str:
        return fallback()

    endpoint: Final = runtime.EndpointBinding(route="ocr", load=load, enabled=enabled)
    result: Final = (
        await endpoint.ainvoke(
            prepare=prepare, call=acall, fallback=afallback, adapt=str, error_context=context(), preflight=preflight
        )
        if asynchronous
        else endpoint.invoke(
            prepare=prepare, call=call, fallback=fallback, adapt=str, error_context=context(), preflight=preflight
        )
    )
    assert result == ("7" if available and accepted else "3")
    assert events == (
        ["load", "preflight", "prepare", "native"]
        if available and accepted
        else ["load", "preflight", "python"]
        if available
        else ["load", "python"]
    )


def test_preflight_failure_is_not_a_native_decline() -> None:
    endpoint: Final = runtime.EndpointBinding(route="ocr", load=object, enabled=enabled)

    def preflight() -> runtime.PythonFallback | None:
        raise ValueError("invalid acceptance contract")

    with pytest.raises(ValueError, match="invalid acceptance contract"):
        endpoint.invoke(
            prepare=lambda: pytest.fail("must not prepare"),
            call=lambda binding, request: pytest.fail("must not invoke"),
            fallback=lambda: pytest.fail("must not fall back"),
            adapt=str,
            error_context=context(),
            preflight=preflight,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    (
        "ocr",
        "chat_completions",
        "messages",
        "responses_websocket",
        "transcription",
    ),
)
@pytest.mark.parametrize("capabilities", (None, frozenset(), frozenset({"streaming_callbacks"})))
async def test_unready_routes_never_prepare_or_call_native(
    monkeypatch: pytest.MonkeyPatch, route: str, capabilities: frozenset[str] | None
) -> None:
    def unexpected(*_args: object) -> object:
        pytest.fail("unready native route must not prepare or execute")

    native: Final = SimpleNamespace(
        ready_endpoints={} if capabilities is None else {route: capabilities},
        chat_completions=unexpected,
    )
    monkeypatch.setattr(loader, "get_native_bridge", lambda: native)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    endpoint: Final = runtime.EndpointBinding.native(
        route=route, select=lambda native: native.chat_completions, enabled=runtime.always_enabled
    )
    arguments: Final = {
        "prepare": unexpected,
        "preflight": unexpected,
        "call": unexpected,
        "adapt": unexpected,
        "error_context": runtime.BridgeErrorContext(provider="test", model="test-model"),
    }
    assert not endpoint.can_attempt()
    assert endpoint.invoke(**arguments, fallback=lambda: "python") == "python"
    with pytest.raises(RuntimeError, match=f"native {route} endpoint is unavailable"):
        endpoint.require(**arguments)

    async def fallback() -> str:
        return "python"

    assert await endpoint.ainvoke(**arguments, fallback=fallback) == "python"
    with pytest.raises(RuntimeError, match=f"native {route} endpoint is unavailable"):
        await endpoint.arequire(**arguments)
