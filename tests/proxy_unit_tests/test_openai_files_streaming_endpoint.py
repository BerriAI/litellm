from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from litellm.proxy._types import ProxyException
from litellm.proxy.openai_files_endpoints.files_endpoints import get_file_content_streaming


async def _fake_stream_iterator():
    yield b"hello "
    yield b"world"


class _FakeProxyLogging:
    async def update_request_status(self, litellm_call_id: str, status: str):
        return None

    async def post_call_failure_hook(
        self, user_api_key_dict, original_exception: Exception, request_data
    ):
        return None


async def _fake_common_processing_pre_call_logic(self, **kwargs):
    return self.data, MagicMock()


@pytest.fixture
def request_obj() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v2/files/file-123/content",
        "headers": [],
        "query_string": b"",
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_get_file_content_streaming_returns_streaming_response(monkeypatch, request_obj):
    async def _mock_afile_content_streaming(**kwargs):
        return _fake_stream_iterator()

    monkeypatch.setattr("litellm.afile_content_streaming", _mock_afile_content_streaming)
    monkeypatch.setattr(
        "litellm.proxy.openai_files_endpoints.files_endpoints.ProxyBaseLLMRequestProcessing.common_processing_pre_call_logic",
        _fake_common_processing_pre_call_logic,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        _FakeProxyLogging(),
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {},
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_config",
        None,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.version",
        "test-version",
    )

    response = await get_file_content_streaming(
        request=request_obj,
        fastapi_response=Response(),
        file_id="file-123",
        provider="openai",
        user_api_key_dict=MagicMock(),
    )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "application/octet-stream"

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert chunks == [b"hello ", b"world"]