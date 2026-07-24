"""Tests for litellm/a2a_protocol/exception_mapping_utils.py."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.a2a_protocol import exception_mapping_utils as emu
from litellm.a2a_protocol.exceptions import A2ALocalhostURLError


def _localhost_error() -> A2ALocalhostURLError:
    return A2ALocalhostURLError(
        localhost_url="http://localhost:10001/",
        base_url="https://agent.example",
        original_error=ConnectionError("boom"),
    )


@pytest.mark.asyncio
async def test_localhost_retry_reuses_stashed_httpx_client():
    """The retry must reuse the httpx client LiteLLM attached at creation (it carries
    the agent's trace-id/auth headers), passing it straight into the new ClientConfig.
    """
    stashed_httpx_client = object()
    a2a_client = MagicMock()
    a2a_client._litellm_httpx_client = stashed_httpx_client
    new_client = MagicMock()

    captured = {}

    def fake_client_config(*, httpx_client, streaming):
        captured["httpx_client"] = httpx_client
        captured["streaming"] = streaming
        return MagicMock()

    with (
        patch.object(emu, "A2A_SDK_AVAILABLE", True),
        patch.object(emu, "set_agent_card_url") as mock_set_url,
        patch.object(emu, "ClientConfig", side_effect=fake_client_config),
        patch.object(
            emu, "create_client", new=AsyncMock(return_value=new_client)
        ) as mock_create,
    ):
        result = await emu.handle_a2a_localhost_retry(
            error=_localhost_error(),
            agent_card=MagicMock(),
            a2a_client=a2a_client,
            is_streaming=True,
        )

    assert result is new_client
    mock_set_url.assert_called_once()
    # The exact stashed client is threaded through, not a freshly built one.
    assert captured["httpx_client"] is stashed_httpx_client
    assert captured["streaming"] is True
    assert new_client._litellm_httpx_client is stashed_httpx_client
    assert mock_create.await_count == 1


@pytest.mark.asyncio
async def test_localhost_retry_raises_when_no_stashed_client():
    """An externally-supplied client has no LiteLLM httpx handle; the retry must fail
    with a clear error instead of excavating a2a-sdk internals."""
    a2a_client = MagicMock(spec=[])  # no _litellm_httpx_client attribute

    with (
        patch.object(emu, "A2A_SDK_AVAILABLE", True),
        patch.object(emu, "set_agent_card_url"),
        patch.object(emu, "create_client", new=AsyncMock()) as mock_create,
    ):
        with pytest.raises(RuntimeError, match="not created by create_a2a_client"):
            await emu.handle_a2a_localhost_retry(
                error=_localhost_error(),
                agent_card=MagicMock(),
                a2a_client=a2a_client,
                is_streaming=False,
            )

    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_localhost_retry_raises_when_agent_card_is_none():
    """With no agent card to rewrite, the retry must fail with a clear error instead
    of calling create_client(None, ...) and surfacing an opaque SDK TypeError."""
    a2a_client = MagicMock()
    a2a_client._litellm_httpx_client = MagicMock()

    with (
        patch.object(emu, "A2A_SDK_AVAILABLE", True),
        patch.object(emu, "set_agent_card_url") as mock_set_url,
        patch.object(emu, "create_client", new=AsyncMock()) as mock_create,
    ):
        with pytest.raises(RuntimeError, match="no agent card is available"):
            await emu.handle_a2a_localhost_retry(
                error=_localhost_error(),
                agent_card=None,
                a2a_client=a2a_client,
                is_streaming=False,
            )

    mock_set_url.assert_not_called()
    mock_create.assert_not_called()


def test_get_a2a_client_agent_card_reads_sdk_private_card():
    from litellm.a2a_protocol.main import _get_a2a_client_agent_card

    sdk_card = SimpleNamespace(name="Test Agent", url="http://localhost:10001/")
    a2a_client = SimpleNamespace(_card=sdk_card)

    assert _get_a2a_client_agent_card(a2a_client) is sdk_card


@pytest.mark.asyncio
async def test_stream_with_retry_raises_after_localhost_retries_exhausted():
    """Exhausted localhost retries must not return a silent empty stream."""
    from litellm.a2a_protocol.main import _execute_a2a_stream_with_retry

    localhost_err = _localhost_error()
    mock_request = MagicMock()
    mock_request.id = "req-1"
    mock_a2a_client = MagicMock()

    async def _always_fail_stream(a2a_client, request):
        raise localhost_err
        yield  # pragma: no cover - makes this an async generator

    with (
        patch(
            "litellm.a2a_protocol.main._stream_messages",
            new=_always_fail_stream,
        ),
        patch(
            "litellm.a2a_protocol.main.handle_a2a_localhost_retry",
            new=AsyncMock(return_value=mock_a2a_client),
        ),
    ):
        stream = _execute_a2a_stream_with_retry(
            a2a_client=mock_a2a_client,
            request=mock_request,
            agent_card=MagicMock(),
            card_url="http://localhost:10001/",
            api_base="https://agent.example",
            agent_name="test-agent",
        )
        with pytest.raises(
            RuntimeError,
            match="no response received after retry attempts",
        ):
            async for _chunk in stream:
                pytest.fail("expected retry exhaustion to raise before yielding")
