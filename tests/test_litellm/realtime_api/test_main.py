import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.realtime_api.main import (  # noqa: E402
    _is_openai_compatible_realtime_provider,
    _realtime_health_check,
    _realtime_health_check_connect,
)


def test_is_openai_compatible_realtime_provider_includes_hosted_vllm():
    assert _is_openai_compatible_realtime_provider("hosted_vllm") is True


def test_is_openai_compatible_realtime_provider_includes_vllm():
    assert _is_openai_compatible_realtime_provider("vllm") is True


def test_is_openai_compatible_realtime_provider_excludes_bedrock():
    assert _is_openai_compatible_realtime_provider("bedrock") is False


@pytest.mark.asyncio
async def test_realtime_health_check_connect_ws_url_has_no_ssl():
    class DummyAsyncContextManager:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, exc_type, exc, tb):
            return None

    with patch(
        "websockets.connect",
        return_value=DummyAsyncContextManager(AsyncMock()),
    ) as mock_ws_connect:
        await _realtime_health_check_connect(
            url="ws://localhost:8113/v1/realtime?model=test-model",
            headers={"Authorization": "Bearer test-key"},
        )

        mock_ws_connect.assert_called_once()
        assert mock_ws_connect.call_args.kwargs["ssl"] is None


@pytest.mark.asyncio
async def test_realtime_health_check_connect_wss_url_uses_ssl_context():
    class DummyAsyncContextManager:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, exc_type, exc, tb):
            return None

    shared_ssl_context = object()
    with (
        patch(
            "websockets.connect",
            return_value=DummyAsyncContextManager(AsyncMock()),
        ) as mock_ws_connect,
        patch(
            "litellm.llms.openai.realtime.handler.get_shared_realtime_ssl_context",
            return_value=shared_ssl_context,
        ),
    ):
        await _realtime_health_check_connect(
            url="wss://api.openai.com/v1/realtime?model=test-model",
            headers={"Authorization": "Bearer test-key"},
        )

        mock_ws_connect.assert_called_once()
        assert mock_ws_connect.call_args.kwargs["ssl"] is shared_ssl_context


@pytest.mark.asyncio
async def test_realtime_health_check_hosted_vllm_uses_openai_compatible_url():
    class DummyAsyncContextManager:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, exc_type, exc, tb):
            return None

    with patch(
        "websockets.connect",
        return_value=DummyAsyncContextManager(AsyncMock()),
    ) as mock_ws_connect:
        await _realtime_health_check(
            model="test-realtime-model",
            custom_llm_provider="hosted_vllm",
            api_key="test-key",
            api_base="http://127.0.0.1:8113",
        )

        called_url = mock_ws_connect.call_args.args[0]
        assert called_url.startswith("ws://127.0.0.1:8113/v1/realtime?")
        assert "model=test-realtime-model" in called_url
        assert mock_ws_connect.call_args.kwargs["ssl"] is None
        assert mock_ws_connect.call_args.kwargs["additional_headers"] == {
            "Authorization": "Bearer test-key",
        }


@pytest.mark.asyncio
async def test_realtime_health_check_openai_ws_url_has_no_ssl():
    class DummyAsyncContextManager:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, exc_type, exc, tb):
            return None

    with patch(
        "websockets.connect",
        return_value=DummyAsyncContextManager(AsyncMock()),
    ) as mock_ws_connect:
        await _realtime_health_check(
            model="test-model",
            custom_llm_provider="openai",
            api_key="test-key",
            api_base="http://127.0.0.1:8113",
        )

        called_url = mock_ws_connect.call_args.args[0]
        assert called_url.startswith("ws://127.0.0.1:8113/v1/realtime?")
        assert mock_ws_connect.call_args.kwargs["ssl"] is None
