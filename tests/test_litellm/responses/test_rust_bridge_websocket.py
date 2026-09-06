from __future__ import annotations

import pytest

from litellm.llms.custom_httpx.llm_http_handler import _rust_responses_websocket_enabled
from litellm.rust_bridge import configuration, responses_websocket
from litellm.rust_bridge.runtime import Handled, NativeFailed, NativeSkipped, NativeSkipReason


class _FakeNativeConnection:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def recv_text(self) -> str:
        return "response.completed"

    async def close(self) -> None:
        self.closed = True


class _ClosedNativeConnection:
    async def recv_text(self) -> None:
        return None


class _FakeNativeBridge:
    @classmethod
    async def connect(
        cls,
        *,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> _FakeNativeConnection:
        return _FakeNativeConnection()


@pytest.fixture(autouse=True)
def reset_responses_websocket():
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()
    yield
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()


def test_rust_websocket_bridge_uses_process_enablement() -> None:
    configuration.rust(False)
    assert not _rust_responses_websocket_enabled("openai")
    configuration.rust(True)
    assert _rust_responses_websocket_enabled("openai")
    assert not _rust_responses_websocket_enabled("anthropic")


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket.ConnectionAdapter(_ClosedNativeConnection())

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration.rust(True)
    responses_websocket._RESPONSES_WEBSOCKET.override(None)

    assert await responses_websocket.connect(
        url="wss://example.test/responses",
        headers={},
        timeout=None,
    ) == NativeSkipped(NativeSkipReason.UNAVAILABLE)


@pytest.mark.asyncio
async def test_enabled_bridge_connects_and_adapts_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FakeNativeBridge)

    connection = await responses_websocket.connect(
        url="wss://example.test/responses",
        headers={"Authorization": "Bearer key"},
        timeout=1.0,
    )

    assert isinstance(connection, Handled)
    connection = connection.value
    await connection.send("response.create")
    assert await connection.recv() == "response.completed"
    await connection.close()


class _FailingNativeBridge:
    @classmethod
    async def connect(
        cls,
        *,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> _FakeNativeConnection:
        raise RuntimeError("connection failed")


@pytest.mark.asyncio
async def test_connection_failure_is_reported_to_orchestration() -> None:
    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FailingNativeBridge)
    result = await responses_websocket.connect(url="wss://example.test/responses", headers={}, timeout=None)
    assert isinstance(result, NativeFailed)
    assert str(result.error) == "connection failed"


@pytest.mark.asyncio
async def test_managed_connection_closes_native_socket_on_consumer_failure() -> None:
    configuration.rust(True)
    socket = _FakeNativeConnection()

    class Bridge:
        @classmethod
        async def connect(
            cls, *, url: str, headers: dict[str, str], timeout_seconds: float | None
        ) -> _FakeNativeConnection:
            return socket

    responses_websocket.set_rust_responses_websocket(connection=Bridge)
    result = await responses_websocket.managed_connect(url="wss://example.test/responses", headers={}, timeout=1.0)
    assert isinstance(result, Handled)

    async def use_connection() -> None:
        async with result.value as connection:
            await connection.send("hello")
            raise ValueError("consumer failed")

    with pytest.raises(ValueError, match="consumer failed"):
        await use_connection()
    assert socket.sent == ["hello"]
    assert socket.closed


@pytest.mark.asyncio
async def test_connection_failure_does_not_authorize_python_fallback() -> None:
    from contextlib import AbstractAsyncContextManager

    from litellm.rust_bridge.dispatch import anative_context, provider_errors

    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FailingNativeBridge)

    @anative_context(
        native=lambda: responses_websocket.managed_connect(
            url="wss://example.test/responses", headers={}, timeout=None
        ),
        route="responses_websocket",
        errors=lambda: provider_errors("openai", "responses websocket"),
    )
    def execute() -> AbstractAsyncContextManager[object]:
        pytest.fail("unknown native failures must not open a Python connection")

    async def run() -> None:
        async with execute():
            pytest.fail("connection must fail before entering its body")

    with pytest.raises(RuntimeError, match="connection failed"):
        await run()
