import pytest
from typing import AsyncIterator, Iterator, cast

from litellm.files import main as files_main
from litellm.files.streaming import FileContentStreamingResponse
from litellm.files.types import FileContentStreamingResult
from litellm.llms.openai.openai import OpenAIFilesAPI
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


@pytest.mark.asyncio
async def test_afile_content_with_stream_routes_to_openai_streaming_handler(
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

    stream_result = cast(
        FileContentStreamingResult,
        await files_main.afile_content(
        file_id="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        organization="org-123",
        chunk_size=8,
        stream=True,
        ),
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
    assert captured_kwargs["client"] is None


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

    stream_result = cast(
        FileContentStreamingResult,
        await files_main.afile_content(
        file_id="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        stream=True,
        ),
    )

    async_stream_iterator = cast(AsyncIterator[bytes], stream_result.stream_iterator)
    chunks = [chunk async for chunk in async_stream_iterator]

    assert chunks == [b"hello"]
    assert stream_result.headers["content-length"] == "5"
    assert captured_standard_logging_object is not None
    assert captured_standard_logging_object["call_type"] == "afile_content"
    assert captured_standard_logging_object["custom_llm_provider"] == "openai"
    assert captured_standard_logging_object["response"]["id"] == "file-abc123"
    assert (
        captured_standard_logging_object["hidden_params"]["api_base"]
        == "https://api.openai.com/v1"
    )


@pytest.mark.asyncio
async def test_afile_content_streaming_shim_sets_stream_flag(
    monkeypatch,
):
    captured_kwargs = {}

    def _mock_file_content_streaming(**kwargs):
        captured_kwargs.update(kwargs)
        return FileContentStreamingResult(
            stream_iterator=iter(()),
            headers={},
        )

    monkeypatch.setattr(
        files_main.openai_files_instance,
        "file_content_streaming",
        _mock_file_content_streaming,
    )

    await files_main.afile_content(
        file_id="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        stream=True,
    )

    assert captured_kwargs["_is_async"] is True


@pytest.mark.asyncio
async def test_file_content_streaming_response_aclose_closes_underlying_async_generator():
    close_called = False

    async def _mock_stream():
        nonlocal close_called
        try:
            yield b"hello"
            yield b"world"
        finally:
            close_called = True

    stream = FileContentStreamingResponse(
        stream_iterator=_mock_stream(),
        file_id="file-abc123",
        model="gpt-4o",
        custom_llm_provider="openai",
        logging_obj=None,
    )

    assert await stream.__anext__() == b"hello"

    await stream.aclose()

    assert close_called is True


@pytest.mark.asyncio
async def test_afile_content_streaming_populates_hidden_params_before_iteration(
    monkeypatch,
):
    async def _mock_stream():
        yield b"hello"

    def _mock_file_content_streaming(**kwargs):
        return FileContentStreamingResult(
            stream_iterator=_mock_stream(),
            headers={"content-length": "5"},
        )

    monkeypatch.setattr(
        files_main.openai_files_instance,
        "file_content_streaming",
        _mock_file_content_streaming,
    )

    stream_result = cast(
        FileContentStreamingResult,
        await files_main.afile_content(
        file_id="file-abc123",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        stream=True,
        ),
    )

    stream_iterator = cast(FileContentStreamingResponse, stream_result.stream_iterator)

    assert stream_iterator._hidden_params["api_base"] == "https://api.openai.com/v1"
    assert stream_iterator._hidden_params["litellm_model_name"] is None


@pytest.mark.asyncio
async def test_afile_content_streaming_passes_exception_to_context_manager_exit():
    class MockAsyncResponse:
        headers = {"content-length": "1"}

        async def iter_bytes(self, chunk_size: int):
            yield b"a"
            raise RuntimeError("stream failed")

    class MockAsyncResponseContextManager:
        def __init__(self):
            self.exc_info = None

        async def __aenter__(self):
            return MockAsyncResponse()

        async def __aexit__(self, exc_type, exc, tb):
            self.exc_info = (exc_type, exc, tb)

    class MockAsyncFiles:
        def __init__(self, response_cm):
            self.with_streaming_response = self
            self._response_cm = response_cm

        def content(self, **kwargs):
            return self._response_cm

    class MockAsyncOpenAIClient:
        def __init__(self, response_cm):
            self.files = MockAsyncFiles(response_cm)

    response_cm = MockAsyncResponseContextManager()
    api = OpenAIFilesAPI()

    stream_result = await api.afile_content_streaming(
        file_content_request={"file_id": "file-abc123"},
        openai_client=MockAsyncOpenAIClient(response_cm),  # type: ignore[arg-type]
        chunk_size=1,
    )
    stream_iterator = cast(AsyncIterator[bytes], stream_result.stream_iterator)

    assert await stream_iterator.__anext__() == b"a"

    with pytest.raises(RuntimeError, match="stream failed") as exc_info:
        await stream_iterator.__anext__()

    assert response_cm.exc_info is not None
    assert response_cm.exc_info[0] is RuntimeError
    assert response_cm.exc_info[1] is exc_info.value
    assert response_cm.exc_info[2] is not None


def test_file_content_streaming_passes_exception_to_context_manager_exit():
    class MockSyncResponse:
        headers = {"content-length": "1"}

        def iter_bytes(self, chunk_size: int) -> Iterator[bytes]:
            yield b"a"
            raise RuntimeError("stream failed")

    class MockSyncResponseContextManager:
        def __init__(self):
            self.exc_info = None

        def __enter__(self):
            return MockSyncResponse()

        def __exit__(self, exc_type, exc, tb):
            self.exc_info = (exc_type, exc, tb)

    class MockSyncFiles:
        def __init__(self, response_cm):
            self.with_streaming_response = self
            self._response_cm = response_cm

        def content(self, **kwargs):
            return self._response_cm

    class MockSyncOpenAIClient:
        def __init__(self, response_cm):
            self.files = MockSyncFiles(response_cm)

    response_cm = MockSyncResponseContextManager()
    api = OpenAIFilesAPI()

    stream_result = api.file_content_streaming(
        _is_async=False,
        file_content_request={"file_id": "file-abc123"},
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        timeout=60,
        max_retries=None,
        organization=None,
        chunk_size=1,
        client=MockSyncOpenAIClient(response_cm),  # type: ignore[arg-type]
    )
    stream_iterator = cast(Iterator[bytes], stream_result.stream_iterator)

    assert next(stream_iterator) == b"a"

    with pytest.raises(RuntimeError, match="stream failed") as exc_info:
        next(stream_iterator)

    assert response_cm.exc_info is not None
    assert response_cm.exc_info[0] is RuntimeError
    assert response_cm.exc_info[1] is exc_info.value
    assert response_cm.exc_info[2] is not None
