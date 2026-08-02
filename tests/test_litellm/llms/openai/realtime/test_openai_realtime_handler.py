from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI, omit

from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context


class DummySDKConnectionManager:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return None


def make_realtime_sdk_client():
    connection = MagicMock()
    connection.send_raw = AsyncMock()
    connection.recv_bytes = AsyncMock()
    connection.close = AsyncMock()
    client = MagicMock(spec=AsyncOpenAI)
    client.realtime.connect = MagicMock(return_value=DummySDKConnectionManager(connection))
    return client


@pytest.mark.parametrize("api_base", ["https://api.openai.com/v1", "https://api.openai.com"])
def test_openai_realtime_handler_url_construction(api_base):
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    handler = OpenAIRealtime()
    url = handler._construct_url(
        api_base=api_base,
        query_params={
            "model": "gpt-4o-realtime-preview-2024-10-01",
        },
    )
    # Model parameter should be included in the URL
    assert url.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-realtime-preview-2024-10-01" in url


def test_openai_realtime_handler_url_with_extra_params():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/v1"
    query_params: RealtimeQueryParams = {
        "model": "gpt-4o-realtime-preview-2024-10-01",
        "intent": "chat",
    }
    url = handler._construct_url(api_base=api_base, query_params=query_params)
    # Both 'model' and other params should be included in the query string
    assert url.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-realtime-preview-2024-10-01" in url
    assert "intent=chat" in url


def test_openai_realtime_handler_model_parameter_inclusion():
    """
    Test that the model parameter is properly included in the WebSocket URL
    to prevent 'missing_model' errors from OpenAI.

    This test specifically verifies the fix for the issue where model parameter
    was being excluded from the query string, causing OpenAI to return
    invalid_request_error.missing_model errors.
    """
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/"

    # Test with just model parameter
    query_params_model_only: RealtimeQueryParams = {"model": "gpt-4o-mini-realtime-preview"}
    url = handler._construct_url(api_base=api_base, query_params=query_params_model_only)

    # Verify the URL structure
    assert url.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-mini-realtime-preview" in url

    # Test with model + additional parameters
    query_params_with_extras: RealtimeQueryParams = {
        "model": "gpt-4o-mini-realtime-preview",
        "intent": "chat",
    }
    url_with_extras = handler._construct_url(api_base=api_base, query_params=query_params_with_extras)

    # Verify both parameters are included
    assert url_with_extras.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-mini-realtime-preview" in url_with_extras
    assert "intent=chat" in url_with_extras

    # Verify the URL is properly formatted for OpenAI
    # Should match the pattern: wss://api.openai.com/v1/realtime?model=MODEL_NAME
    expected_pattern = "wss://api.openai.com/v1/realtime?model="
    assert expected_pattern in url
    assert expected_pattern in url_with_extras


@pytest.mark.asyncio
async def test_async_realtime_success():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/v1"
    api_key = "test-key"
    model = "gpt-4o-realtime-preview-2024-10-01"
    query_params: RealtimeQueryParams = {"model": model, "intent": "chat"}

    dummy_websocket = AsyncMock()
    dummy_logging_obj = MagicMock()
    sdk_client = make_realtime_sdk_client()
    with patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
            client=sdk_client,
        )

        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_realtime_url_contains_model():
    """
    Test that the async_realtime method properly constructs a URL with the model parameter
    when connecting to OpenAI, preventing 'missing_model' errors.
    """
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/"
    api_key = "test-key"
    model = "gpt-4o-mini-realtime-preview"
    query_params: RealtimeQueryParams = {"model": model}

    dummy_websocket = AsyncMock()
    dummy_logging_obj = MagicMock()
    sdk_client = make_realtime_sdk_client()
    with patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
            client=sdk_client,
        )

        sdk_client.realtime.connect.assert_called_once()
        called_kwargs = sdk_client.realtime.connect.call_args.kwargs
        assert called_kwargs["model"] == model
        additional_headers = called_kwargs["extra_headers"]
        assert additional_headers["Authorization"] == f"Bearer {api_key}"
        assert "OpenAI-Beta" not in additional_headers
        assert called_kwargs["max_retries"] == 0

        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_realtime_transcription_omits_sdk_model_query():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    handler = OpenAIRealtime()
    websocket = AsyncMock()
    logging_obj = MagicMock()
    sdk_client = make_realtime_sdk_client()
    with patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        mock_realtime_streaming.return_value.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model="gpt-live-transcribe",
            websocket=websocket,
            logging_obj=logging_obj,
            api_key="test-key",
            query_params={"intent": "transcription"},
            client=sdk_client,
        )

    called_kwargs = sdk_client.realtime.connect.call_args.kwargs
    assert called_kwargs["model"] is omit
    assert called_kwargs["extra_query"] == {"intent": "transcription"}


@pytest.mark.asyncio
async def test_async_realtime_forwards_openai_beta_header_when_client_sends_it():
    """Upstream WS gets OpenAI-Beta: realtime=v1 only when the client WebSocket included it."""
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/"
    api_key = "test-key"
    model = "gpt-4o-mini-realtime-preview"
    query_params: RealtimeQueryParams = {"model": model}

    dummy_websocket = MagicMock()
    dummy_websocket.scope = {
        "headers": [
            (b"openai-beta", b"realtime=v1"),
        ]
    }
    dummy_logging_obj = MagicMock()
    sdk_client = make_realtime_sdk_client()
    with patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
            client=sdk_client,
        )

        sdk_client.realtime.connect.assert_called_once()
        called_kwargs = sdk_client.realtime.connect.call_args.kwargs
        additional_headers = called_kwargs["extra_headers"]
        assert additional_headers["Authorization"] == f"Bearer {api_key}"
        assert additional_headers["OpenAI-Beta"] == "realtime=v1"


@pytest.mark.asyncio
async def test_async_realtime_uses_max_size_parameter():
    """
    Test that the async_realtime method uses the REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    constant for the max_size parameter to handle large base64 audio payloads.

    This verifies the fix for: https://github.com/BerriAI/litellm/issues/15747
    """
    from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/"
    api_key = "test-key"
    model = "gpt-4o-realtime-preview"
    query_params: RealtimeQueryParams = {"model": model}

    dummy_websocket = AsyncMock()
    dummy_logging_obj = MagicMock()
    sdk_client = make_realtime_sdk_client()
    with patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
            client=sdk_client,
        )

        sdk_client.realtime.connect.assert_called_once()
        called_kwargs = sdk_client.realtime.connect.call_args.kwargs
        connection_options = called_kwargs["websocket_connection_options"]
        assert connection_options["max_size"] is REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
        assert connection_options["ssl"] is not None
        assert connection_options["ssl"] is not False

        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_realtime_ws_url_has_no_ssl():
    """
    Test that when using http:// api_base (converted to ws://), the ssl argument
    is set to None. The websockets library doesn't accept ssl argument for ws:// URIs.

    This verifies the fix for: https://github.com/BerriAI/litellm/issues/19222
    """
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "http://localhost:8113"  # Non-SSL local server
    api_key = "test-key"
    model = "test-model"
    query_params: RealtimeQueryParams = {"model": model}

    dummy_websocket = AsyncMock()
    dummy_logging_obj = MagicMock()
    sdk_client = make_realtime_sdk_client()
    with patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
            client=sdk_client,
        )

        sdk_client.realtime.connect.assert_called_once()
        called_kwargs = sdk_client.realtime.connect.call_args.kwargs
        assert called_kwargs["model"] == model
        assert called_kwargs["websocket_connection_options"]["ssl"] is None


def test_translation_url_uses_dedicated_path():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    handler = OpenAIRealtime()
    url = handler._construct_url(
        api_base="https://api.openai.com/v1",
        query_params={"model": "gpt-realtime-translate"},
        realtime_mode="translation",
    )
    assert url == "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"


@pytest.mark.asyncio
async def test_translation_websocket_uses_direct_transport():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    backend = AsyncMock()

    class TranslationConnectionManager:
        async def __aenter__(self):
            return backend

        async def __aexit__(self, exc_type, exc, tb):
            return None

    websocket = MagicMock()
    websocket.scope = {"headers": []}
    websocket.close = AsyncMock()
    logging_obj = MagicMock()
    handler = OpenAIRealtime()

    with (
        patch("websockets.connect", return_value=TranslationConnectionManager()) as connect,
        patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as streaming,
    ):
        streaming.return_value.bidirectional_forward = AsyncMock()
        await handler.async_realtime(
            model="gpt-realtime-translate",
            websocket=websocket,
            logging_obj=logging_obj,
            api_base="https://api.openai.com/v1",
            api_key="sk-test",
            query_params={"model": "gpt-realtime-translate"},
            realtime_mode="translation",
        )

    connect.assert_called_once()
    assert connect.call_args.args[0] == ("wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate")
    assert streaming.call_args.kwargs["translation_session"] is True
