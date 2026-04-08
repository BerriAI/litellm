from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
from litellm.proxy.openai_files_endpoints import files_endpoints


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


class _FakeManagedFiles(BaseFileEndpoints):
    def __init__(self):
        self.calls = []

    async def acreate_file(self, *args, **kwargs):
        raise NotImplementedError

    async def afile_retrieve(self, *args, **kwargs):
        raise NotImplementedError

    async def afile_list(self, *args, **kwargs):
        raise NotImplementedError

    async def afile_delete(self, *args, **kwargs):
        raise NotImplementedError

    async def afile_content(self, *args, **kwargs):
        raise NotImplementedError

    async def afile_content_streaming(
        self,
        file_id: str,
        litellm_parent_otel_span,
        llm_router,
        chunk_size: int = 1024 * 1024,
        **data,
    ):
        self.calls.append(
            {
                "file_id": file_id,
                "litellm_parent_otel_span": litellm_parent_otel_span,
                "llm_router": llm_router,
                "chunk_size": chunk_size,
                "data": data,
            }
        )
        return _fake_stream_iterator()


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

    response = await files_endpoints.get_file_content_streaming(
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


@pytest.mark.asyncio
async def test_get_file_content_streaming_uses_managed_files_hook(monkeypatch, request_obj):
    fake_managed_files = _FakeManagedFiles()
    proxy_logging = MagicMock()
    proxy_logging.get_proxy_hook.return_value = fake_managed_files
    proxy_logging.update_request_status = AsyncMock(return_value=None)
    proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)

    async def _fake_common_processing_pre_call_logic(self, **kwargs):
        return self.data, MagicMock()

    monkeypatch.setattr(
        "litellm.proxy.openai_files_endpoints.files_endpoints.ProxyBaseLLMRequestProcessing.common_processing_pre_call_logic",
        _fake_common_processing_pre_call_logic,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", MagicMock())
    monkeypatch.setattr(
        "litellm.proxy.openai_files_endpoints.files_endpoints._is_base64_encoded_unified_file_id",
        lambda file_id: file_id,
    )

    response = await files_endpoints.get_file_content_streaming(
        request=request_obj,
        fastapi_response=Response(),
        file_id="file-123",
        provider="openai",
        user_api_key_dict=MagicMock(),
    )

    assert isinstance(response, StreamingResponse)

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert chunks == [b"hello ", b"world"]
    assert fake_managed_files.calls[0]["file_id"].startswith("file-123")