import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.realtime_api import main as realtime_main  # noqa: E402
from litellm.realtime_api.main import (  # noqa: E402
    _arealtime,
    _is_openai_compatible_realtime_provider,
    _realtime_health_check,
    _realtime_health_check_connect,
)


class DummyAsyncContextManager:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return None


def test_is_openai_compatible_realtime_provider_includes_hosted_vllm():
    assert _is_openai_compatible_realtime_provider("hosted_vllm") is True


def test_is_openai_compatible_realtime_provider_includes_vllm():
    assert _is_openai_compatible_realtime_provider("vllm") is True


def test_is_openai_compatible_realtime_provider_excludes_bedrock():
    assert _is_openai_compatible_realtime_provider("bedrock") is False


@pytest.mark.asyncio
async def test_realtime_health_check_connect_ws_url_has_no_ssl():
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
async def test_realtime_health_check_vllm_uses_openai_compatible_url():
    with patch(
        "websockets.connect",
        return_value=DummyAsyncContextManager(AsyncMock()),
    ) as mock_ws_connect:
        await _realtime_health_check(
            model="qwen-realtime",
            custom_llm_provider="vllm",
            api_key="vllm-key",
            api_base="http://127.0.0.1:8113",
        )

        called_url = mock_ws_connect.call_args.args[0]
        assert called_url.startswith("ws://127.0.0.1:8113/v1/realtime?")
        assert "model=qwen-realtime" in called_url
        assert mock_ws_connect.call_args.kwargs["ssl"] is None
        assert mock_ws_connect.call_args.kwargs["additional_headers"] == {
            "Authorization": "Bearer vllm-key",
        }


@pytest.mark.asyncio
async def test_realtime_health_check_openai_compatible_api_key_falls_back_to_litellm_api_key(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "api_key", "global-litellm-key")
    monkeypatch.setattr(litellm, "openai_key", None)

    with patch(
        "websockets.connect",
        return_value=DummyAsyncContextManager(AsyncMock()),
    ) as mock_ws_connect:
        await _realtime_health_check(
            model="test-realtime-model",
            custom_llm_provider="hosted_vllm",
            api_key=None,
            api_base="http://127.0.0.1:8113",
        )

        assert mock_ws_connect.call_args.kwargs["additional_headers"] == {
            "Authorization": "Bearer global-litellm-key",
        }


@pytest.mark.asyncio
async def test_realtime_health_check_openai_ws_url_has_no_ssl():
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


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_llm_provider", ["hosted_vllm", "vllm"])
async def test_arealtime_openai_compatible_provider_routes_to_openai_realtime(monkeypatch, custom_llm_provider):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("test-realtime-model", custom_llm_provider, "dynamic-key", "http://127.0.0.1:8113")

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)

    await _arealtime(
        model=f"{custom_llm_provider}/test-realtime-model",
        websocket=MagicMock(),
        litellm_logging_obj=MagicMock(),
    )

    assert captured_kwargs["api_base"] == "http://127.0.0.1:8113"
    assert captured_kwargs["api_key"] == "dynamic-key"
    assert captured_kwargs["model"] == "test-realtime-model"


@pytest.mark.asyncio
async def test_arealtime_openai_compatible_uses_explicit_api_key_when_dynamic_key_missing(monkeypatch):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("test-realtime-model", "hosted_vllm", None, "http://127.0.0.1:8113")

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(litellm, "api_key", "should-not-use-yet")
    monkeypatch.setattr(litellm, "openai_key", None)

    await _arealtime(
        model="hosted_vllm/test-realtime-model",
        websocket=MagicMock(),
        api_key="explicit-api-key",
        litellm_logging_obj=MagicMock(),
    )

    assert captured_kwargs["api_key"] == "explicit-api-key"


@pytest.mark.asyncio
async def test_arealtime_openai_compatible_uses_litellm_params_api_key_fallback(monkeypatch):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    original_generic_litellm_params = realtime_main.GenericLiteLLMParams

    def litellm_params_with_api_key(**kwargs):
        params = original_generic_litellm_params(**kwargs)
        params.api_key = "params-api-key"
        params.api_base = "http://127.0.0.1:8113"
        return params

    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )
    monkeypatch.setattr(realtime_main, "GenericLiteLLMParams", litellm_params_with_api_key)

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("test-realtime-model", "hosted_vllm", None, None)

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(litellm, "api_key", "should-not-use-yet")
    monkeypatch.setattr(litellm, "openai_key", None)

    await _arealtime(
        model="hosted_vllm/test-realtime-model",
        websocket=MagicMock(),
        litellm_logging_obj=MagicMock(),
    )

    assert captured_kwargs["api_key"] == "params-api-key"
    assert captured_kwargs["api_base"] == "http://127.0.0.1:8113"


@pytest.mark.asyncio
async def test_arealtime_openai_compatible_uses_litellm_api_key_fallback(monkeypatch):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("test-realtime-model", "vllm", None, "http://127.0.0.1:8113")

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(litellm, "api_key", "global-litellm-key")
    monkeypatch.setattr(litellm, "openai_key", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    await _arealtime(
        model="vllm/test-realtime-model",
        websocket=MagicMock(),
        litellm_logging_obj=MagicMock(),
    )

    assert captured_kwargs["api_key"] == "global-litellm-key"


@pytest.mark.asyncio
async def test_arealtime_openai_compatible_api_base_falls_back_to_litellm_params(monkeypatch):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("test-realtime-model", "hosted_vllm", "test-key", None)

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(litellm, "api_base", "http://should-not-use.example.com")

    await _arealtime(
        model="hosted_vllm/test-realtime-model",
        websocket=MagicMock(),
        api_base="http://127.0.0.1:8113",
        api_key="test-key",
        litellm_logging_obj=MagicMock(),
    )

    assert captured_kwargs["api_base"] == "http://127.0.0.1:8113"
