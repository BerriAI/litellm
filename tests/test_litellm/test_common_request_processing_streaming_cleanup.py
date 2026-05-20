from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging


def _mock_proxy_logging_obj():
    proxy_logging_obj = MagicMock(spec=ProxyLogging)
    proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(
        side_effect=lambda **kwargs: kwargs["response"]
    )
    proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    return proxy_logging_obj


@pytest.mark.asyncio
async def test_async_streaming_data_generator_closes_response_on_early_exit():
    mock_response = MagicMock()
    mock_response.aclose = AsyncMock()
    mock_proxy_logging_obj = _mock_proxy_logging_obj()

    async def mock_streaming_iterator(*args, **kwargs):
        yield {"content": "hello"}
        yield {"content": " world"}

    mock_proxy_logging_obj.async_post_call_streaming_iterator_hook = (
        mock_streaming_iterator
    )

    generator = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
        response=mock_response,
        user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
        request_data={"model": "gpt-3.5-turbo"},
        proxy_logging_obj=mock_proxy_logging_obj,
        serialize_chunk=lambda chunk: str(chunk),
        serialize_error=lambda proxy_exception: str(proxy_exception.to_dict()),
    )

    first_chunk = await generator.__anext__()
    assert first_chunk == "{'content': 'hello'}"

    await generator.aclose()

    mock_response.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_streaming_data_generator_closes_response_on_completion():
    mock_response = MagicMock()
    mock_response.aclose = AsyncMock()
    mock_proxy_logging_obj = _mock_proxy_logging_obj()

    async def mock_streaming_iterator(*args, **kwargs):
        yield {"content": "hello"}

    mock_proxy_logging_obj.async_post_call_streaming_iterator_hook = (
        mock_streaming_iterator
    )

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=mock_response,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            request_data={"model": "gpt-3.5-turbo"},
            proxy_logging_obj=mock_proxy_logging_obj,
            serialize_chunk=lambda chunk: str(chunk),
            serialize_error=lambda proxy_exception: str(proxy_exception.to_dict()),
        )
    ]

    assert chunks == ["{'content': 'hello'}"]
    mock_response.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_streaming_data_generator_closes_response_on_error():
    mock_response = MagicMock()
    mock_response.aclose = AsyncMock()
    mock_proxy_logging_obj = _mock_proxy_logging_obj()

    async def mock_streaming_iterator(*args, **kwargs):
        yield {"content": "hello"}
        raise RuntimeError("upstream connection reset")

    mock_proxy_logging_obj.async_post_call_streaming_iterator_hook = (
        mock_streaming_iterator
    )

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=mock_response,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            request_data={"model": "gpt-3.5-turbo"},
            proxy_logging_obj=mock_proxy_logging_obj,
            serialize_chunk=lambda chunk: str(chunk),
            serialize_error=lambda proxy_exception: str(proxy_exception.to_dict()),
        )
    ]

    assert chunks[0] == "{'content': 'hello'}"
    assert "upstream connection reset" in chunks[1]
    mock_proxy_logging_obj.post_call_failure_hook.assert_awaited_once()
    mock_response.aclose.assert_awaited_once()
