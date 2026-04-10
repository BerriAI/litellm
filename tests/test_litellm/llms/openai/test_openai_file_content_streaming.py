import pytest
from typing import AsyncIterator, cast

from litellm.files import main as files_main
from litellm.files.streaming import FileContentStreamingResult
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


@pytest.mark.asyncio
async def test_afile_content_streaming_routes_to_openai_streaming_handler(
    monkeypatch,
):
    captured_kwargs = {}

    async def _mock_stream():
        yield b"hello "
        yield b"world"

    def _mock_file_content_streaming(**kwargs):
        captured_kwargs.update(kwargs)
        return FileContentStreamingResult(
            stream_iterator=_mock_stream(),
            headers={"content-length": "11"},
        )

    monkeypatch.setattr(
        files_main.openai_files_instance,
        "file_content_streaming",
        _mock_file_content_streaming,
    )

    stream_result = await files_main.afile_content_streaming(
        file_id="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        organization="org-123",
        chunk_size=8,
    )

    async_stream_iterator = cast(AsyncIterator[bytes], stream_result.stream_iterator)
    chunks = [chunk async for chunk in async_stream_iterator]

    assert chunks == [b"hello ", b"world"]
    assert stream_result.headers["content-length"] == "11"
    assert captured_kwargs["_is_async"] is True
    assert captured_kwargs["file_content_request"]["file_id"] == "file-abc123"
    assert captured_kwargs["api_key"] == "sk-test"
    assert captured_kwargs["api_base"] == "https://api.openai.com/v1"
    assert captured_kwargs["organization"] == "org-123"
    assert captured_kwargs["chunk_size"] == 8


@pytest.mark.asyncio
async def test_afile_content_streaming_builds_standard_logging_object_on_completion(
    monkeypatch,
):
    captured_standard_logging_object = None

    async def _mock_stream():
        yield b"hello"

    def _mock_file_content_streaming(**kwargs):
        return FileContentStreamingResult(
            stream_iterator=_mock_stream(),
            headers={"content-length": "5"},
        )

    async def _mock_async_success_handler(
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        nonlocal captured_standard_logging_object
        captured_standard_logging_object = kwargs.get("standard_logging_object")
        self.model_call_details["standard_logging_object"] = captured_standard_logging_object

    monkeypatch.setattr(
        files_main.openai_files_instance,
        "file_content_streaming",
        _mock_file_content_streaming,
    )
    monkeypatch.setattr(
        LiteLLMLoggingObj,
        "async_success_handler",
        _mock_async_success_handler,
    )
    monkeypatch.setattr(
        LiteLLMLoggingObj,
        "handle_sync_success_callbacks_for_async_calls",
        lambda self, result, start_time, end_time, cache_hit=None: None,
    )

    stream_result = await files_main.afile_content_streaming(
        file_id="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
    )

    async_stream_iterator = cast(AsyncIterator[bytes], stream_result.stream_iterator)
    chunks = [chunk async for chunk in async_stream_iterator]

    assert chunks == [b"hello"]
    assert stream_result.headers["content-length"] == "5"
    assert captured_standard_logging_object is not None
    assert captured_standard_logging_object["call_type"] == "afile_content_streaming"
    assert captured_standard_logging_object["custom_llm_provider"] == "openai"
    assert captured_standard_logging_object["response"]["id"] == "file-abc123"
    assert (
        captured_standard_logging_object["hidden_params"]["api_base"]
        == "https://api.openai.com/v1"
    )
