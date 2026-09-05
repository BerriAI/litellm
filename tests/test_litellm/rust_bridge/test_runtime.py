from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import bindings, runtime


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


def enabled(*, request_override: bool | None = None) -> bool:
    return request_override is not False


@dataclass(frozen=True, slots=True)
class FallbackCase:
    request_override: bool | None = None
    eligible: bool = True
    binding_available: bool = True
    declined: bool = False
    expected_events: tuple[str, ...] = ()


FALLBACK_CASES: Final = (
    pytest.param(
        FallbackCase(request_override=False, expected_events=("python",)),
        id="request-disabled",
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

    bridge: Final = runtime.EndpointBinding(route="messages", load=load, enabled=enabled)
    result: Final = bridge.invoke(
        prepare=lambda: events.append("prepare"),
        call=call,
        fallback=lambda: events.append("python") or "fallback",
        adapt=str,
        context=context(),
        request_override=case.request_override,
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

    bridge: Final = runtime.EndpointBinding(route="messages", load=load, enabled=enabled)
    result: Final = await bridge.ainvoke(
        prepare=lambda: events.append("prepare"),
        call=call,
        fallback=fallback,
        adapt=str,
        context=context(),
        request_override=case.request_override,
        eligible=case.eligible,
    )

    assert result == "fallback"
    assert tuple(events) == case.expected_events


def test_invoke_adapts_native_success_without_fallback() -> None:
    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    result: Final = bridge.invoke(
        prepare=lambda: None,
        call=lambda _binding, _request: 3,
        fallback=lambda: pytest.fail("fallback must not run"),
        adapt=lambda value: f"adapted-{value}",
        context=context(),
    )

    assert result == "adapted-3"


@pytest.mark.asyncio
async def test_ainvoke_adapts_native_success_without_fallback() -> None:
    async def call(_binding: object, _request: object) -> int:
        return 3

    async def fallback() -> str:
        pytest.fail("fallback must not run")

    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)
    result: Final = await bridge.ainvoke(
        prepare=lambda: None,
        call=call,
        fallback=fallback,
        adapt=lambda value: f"adapted-{value}",
        context=context(),
    )

    assert result == "adapted-3"


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_message"),
    (
        pytest.param(RustUpstreamError(429, "rate limited"), 429, "rate limited", id="provider-status"),
        pytest.param(RustUpstreamError(0, "connection reset"), 500, "connection reset", id="transport-failure"),
    ),
)
def test_upstream_failure_maps_to_api_error_without_fallback(
    error: RustUpstreamError,
    expected_status: int,
    expected_message: str,
) -> None:
    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    with pytest.raises(APIError, match=expected_message) as caught:
        bridge.invoke(
            prepare=lambda: None,
            call=lambda _binding, _request: (_ for _ in ()).throw(error),
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            context=context(),
        )

    assert caught.value.status_code == expected_status
    assert caught.value.llm_provider == "anthropic"
    assert caught.value.model == "model"


@pytest.mark.asyncio
async def test_async_upstream_failure_maps_to_api_error_without_fallback() -> None:
    async def fail(_binding: object, _request: object) -> object:
        raise RustUpstreamError(503, "overloaded")

    async def fallback() -> object:
        pytest.fail("fallback must not run")

    bridge: Final = runtime.EndpointBinding(route="messages", load=object, enabled=enabled)

    with pytest.raises(APIError, match="overloaded") as caught:
        await bridge.ainvoke(prepare=lambda: None, call=fail, fallback=fallback, adapt=str, context=context())

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
            context=context(),
        )

    assert caught.value is error


@pytest.mark.parametrize(
    ("request_override", "binding_available", "declined", "expected_message"),
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
    request_override: bool | None,
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
        enabled=enabled,
    )

    with pytest.raises(RuntimeError, match=f"^{expected_message}$"):
        bridge.require(
            prepare=lambda: None,
            call=call,
            adapt=str,
            context=context(),
            request_override=request_override,
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

    bridge: Final = runtime.EndpointBinding(route="messages", load=load, enabled=enabled)

    assert (
        bridge.can_attempt(
            request_override=False if state == "disabled" else None,
            eligible=state != "ineligible",
        )
        is expected
    )
    assert tuple(events) == expected_events


def test_native_endpoint_applies_partial_overrides_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    native_sync: Final = lambda: "native"

    async def native_async() -> str:
        return "native async"

    replacement_sync: Final = lambda: "replacement"
    monkeypatch.setattr(
        bindings,
        "get_native_bridge",
        lambda: SimpleNamespace(sync_route=native_sync, async_route=native_async),
    )
    monkeypatch.setattr(bindings, "native_route_ready", lambda _route, _capabilities: True)
    endpoint: Final[runtime.EndpointDispatch[object, object]] = runtime.EndpointDispatch.native(
        route="test",
        sync="sync_route",
        asynchronous="async_route",
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
