"""
Tests for litellm.litellm_core_utils.logging_utils — base64 truncation helpers.
"""

import asyncio, datetime, importlib, litellm, os, pytest_asyncio
import sys
import threading
from itertools import chain
from types import FrameType
from unittest.mock import MagicMock

import pytest

from litellm.litellm_core_utils import logging_utils
from litellm.litellm_core_utils.logging_utils import (
    assemble_complete_response_from_streaming_chunks,
    _set_duration_in_model_call_details,
    _truncate_base64_in_string,
    format_base64_size,
    truncate_base64_in_messages,
    truncate_base64_in_messages_async,
)
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from datetime import datetime as datetime_assemble_streaming
from litellm import(
    Choices,
    ModelResponse,
    ModelResponseStream,
    TextChoices,
    TextCompletionResponse,
)
from litellm.constants import LOGGING_WORKER_MAX_TIME_PER_COROUTINE
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.llms.openai import AllMessageValues
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from typing import Final


class TestSetDurationInModelCallDetails:
    def test_records_provider_attempt_windows_in_shared_metadata(self):
        metadata = {"request_id": "test"}
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"litellm_params": {"metadata": metadata}}
        first_start = datetime.datetime(2025, 1, 1, 0, 0, 0)
        first_end = first_start + datetime.timedelta(milliseconds=300)
        second_start = datetime.datetime(2025, 1, 1, 0, 0, 1)
        second_end = second_start + datetime.timedelta(milliseconds=700)

        _set_duration_in_model_call_details(logging_obj, first_start, first_end)
        _set_duration_in_model_call_details(logging_obj, second_start, second_end)

        assert metadata["llm_api_timing_windows"] == (
            (first_start.timestamp(), first_end.timestamp()),
            (second_start.timestamp(), second_end.timestamp()),
        )
        assert logging_obj.model_call_details["llm_api_duration_ms"] == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# format_base64_size
# ---------------------------------------------------------------------------


class TestFormatBase64Size:
    def test_bytes_range(self):
        assert format_base64_size(4) == "3B"

    def test_kb_range(self):
        # 2000 base64 chars ~ 1500 bytes ~ 1.5KB
        assert "KB" in format_base64_size(2000)

    def test_mb_range(self):
        # 2_000_000 base64 chars ~ 1.5MB
        result = format_base64_size(2_000_000)
        assert "MB" in result


# ---------------------------------------------------------------------------
# _truncate_base64_in_string
# ---------------------------------------------------------------------------


class TestTruncateBase64InString:
    def test_short_data_uri_not_truncated(self):
        uri = "data:image/png;base64,AAAA"
        assert _truncate_base64_in_string(uri) == uri

    def test_long_data_uri_truncated(self):
        payload = "A" * 200
        uri = f"data:application/pdf;base64,{payload}"
        result = _truncate_base64_in_string(uri)
        assert "base64_data truncated" in result
        assert "application/pdf" in result
        assert payload not in result

    def test_multiple_data_uris(self):
        payload = "B" * 200
        text = f"first: data:image/png;base64,{payload} second: data:image/jpeg;base64,{payload}"
        result = _truncate_base64_in_string(text)
        assert result.count("base64_data truncated") == 2

    @pytest.mark.timeout(10)
    @pytest.mark.parametrize(
        "text",
        [
            'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n' * 50_000 + ";base64,",
            "data:" * 200_000 + ";base64,",
        ],
        ids=["sse_lines", "whitespace_free_prefixes"],
    )
    def test_repeated_data_prefixes_without_data_uris_are_scanned_in_linear_time(self, text: str):
        assert _truncate_base64_in_string(text) == text

    def test_no_data_uri(self):
        text = "hello world, no base64 here"
        assert _truncate_base64_in_string(text) == text


# ---------------------------------------------------------------------------
# truncate_base64_in_messages
# ---------------------------------------------------------------------------


class TestTruncateBase64InMessages:
    def test_none_input(self):
        assert truncate_base64_in_messages(None) is None

    def test_string_messages(self):
        payload = "C" * 200
        msg = f"Look at data:image/png;base64,{payload}"
        result = truncate_base64_in_messages(msg)
        assert isinstance(result, str)
        assert "base64_data truncated" in result

    def test_openai_vision_format(self):
        """Typical OpenAI multimodal message with image_url containing base64."""
        payload = "D" * 500
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{payload}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]
        result = truncate_base64_in_messages(messages)
        # Original must not be mutated
        assert payload in messages[0]["content"][1]["image_url"]["url"]
        # Result should be truncated
        url = result[0]["content"][1]["image_url"]["url"]
        assert "base64_data truncated" in url
        assert payload not in url
        # Non-base64 parts preserved
        assert result[0]["content"][0]["text"] == "What is in this image?"

    def test_multiple_images(self):
        """Two base64 images in one message."""
        payload1 = "E" * 300
        payload2 = "F" * 400
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{payload1}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:application/pdf;base64,{payload2}"},
                    },
                ],
            }
        ]
        result = truncate_base64_in_messages(messages)
        for part in result[0]["content"]:
            assert "base64_data truncated" in part["image_url"]["url"]

    def test_does_not_mutate_original(self):
        payload = "G" * 200
        messages = [{"role": "user", "content": f"data:image/png;base64,{payload}"}]
        truncate_base64_in_messages(messages)
        # Original unchanged
        assert payload in messages[0]["content"]

    def test_dict_messages(self):
        payload = "H" * 200
        messages = {"prompt": f"data:image/png;base64,{payload}"}
        result = truncate_base64_in_messages(messages)
        assert "base64_data truncated" in result["prompt"]

    def test_preserves_short_base64(self):
        """Short base64 under threshold should not be truncated."""
        short = "AAAA"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{short}"},
                    }
                ],
            }
        ]
        result = truncate_base64_in_messages(messages)
        assert result[0]["content"][0]["image_url"]["url"] == f"data:image/png;base64,{short}"


# ---------------------------------------------------------------------------
# truncate_base64_in_messages_async
# ---------------------------------------------------------------------------


def _image_messages(payload: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
            ],
        }
    ]


@pytest.fixture
def scan_threads(monkeypatch):
    """Record the thread that runs every base64 regex scan."""
    threads: list[int] = []
    original = logging_utils._truncate_base64_in_string

    def recording_scan(value: str) -> str:
        threads.append(threading.get_ident())
        return original(value)

    monkeypatch.setattr(logging_utils, "_truncate_base64_in_string", recording_scan)
    return threads


def _run_without_yielding(coroutine: Coroutine[object, object, object]) -> object:
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise AssertionError("the coroutine yielded to the event loop")


def _regex_calls_while(run: Callable[[], object]) -> tuple[object, tuple[str, ...]]:
    calls: Final[list[str]] = []  # mutable-ok: sys.setprofile reports each call to a callback that has to record it
    previous: Final = sys.getprofile()

    def record(frame: FrameType, event: str, arg: object) -> None:
        name: Final = getattr(arg, "__qualname__", "")
        if event == "c_call" and name.startswith("Pattern."):
            calls.append(name)

    sys.setprofile(record)
    try:
        result: Final = run()
    finally:
        sys.setprofile(previous)
    return result, tuple(calls)


def _tool_result_messages(tool_output: str) -> list[AllMessageValues]:  # mutable-ok: the truncator only walks lists
    return [
        {"role": "user", "content": "Summarize the rows the tool returned."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "dump_rows", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_output},
    ]


_PLAIN_TOOL_OUTPUT_CHARS: Final = 400 * 1024


class TestTruncateBase64InMessagesAsync:
    @pytest.mark.asyncio
    async def test_large_payload_is_scanned_off_the_event_loop(self, monkeypatch, scan_threads):
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_CHARS", 1_000)
        payload = "I" * 20_000
        messages = _image_messages(payload)

        result = await truncate_base64_in_messages_async(messages)
        offload_threads = tuple(scan_threads)

        assert result == truncate_base64_in_messages(messages)
        assert payload not in result[0]["content"][1]["image_url"]["url"]
        assert payload in messages[0]["content"][1]["image_url"]["url"]
        assert offload_threads
        assert threading.get_ident() not in offload_threads

    @pytest.mark.asyncio
    async def test_small_payload_stays_on_the_calling_thread(self, monkeypatch, scan_threads):
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_CHARS", 1_000)
        messages = _image_messages("J" * 200)

        result = await truncate_base64_in_messages_async(messages)

        assert result == truncate_base64_in_messages(messages)
        assert scan_threads
        assert set(scan_threads) == {threading.get_ident()}

    @pytest.mark.parametrize(
        "row",
        [
            '{"id": 7, "name": "widget", "status": "ok"}\n',
            "data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n",
            "const total = rows.map((r) => r.price).reduce((a, b) => a + b, 0);\n",
            'data: {"choices": [{"delta": {"content": "a; b"}}]}\n\n',
            "Content-Type: text/plain;base64,aGVsbG8=\n",
        ],
        ids=["json_rows", "sse_lines", "code_with_semicolons", "sse_lines_with_semicolons", "base64_marker_without_data"],
    )
    def test_tool_message_without_data_uri_is_copied_without_a_regex_scan(self, row: str):
        tool_output: Final = (row * (_PLAIN_TOOL_OUTPUT_CHARS // len(row) + 1))[:_PLAIN_TOOL_OUTPUT_CHARS]
        messages: Final = _tool_result_messages(tool_output)

        result, regex_calls = _regex_calls_while(lambda: truncate_base64_in_messages(messages))

        assert result == messages
        assert result is not messages
        assert regex_calls == ()

    def test_small_data_uri_is_truncated_inline_with_a_regex_scan(self):
        payload: Final = "L" * 2_000
        messages: Final = _image_messages(payload)

        result, regex_calls = _regex_calls_while(
            lambda: _run_without_yielding(truncate_base64_in_messages_async(messages))
        )

        assert regex_calls
        assert payload not in result[0]["content"][1]["image_url"]["url"]
        assert "base64_data truncated" in result[0]["content"][1]["image_url"]["url"]

    def test_large_plain_text_is_copied_inline_without_rescanning_or_a_thread_hop(
        self, monkeypatch: pytest.MonkeyPatch, scan_threads: Sequence[int]
    ):
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_CHARS", 1_000)
        messages: Final = _tool_result_messages("plain tool output\n" * 100)

        result: Final = _run_without_yielding(truncate_base64_in_messages_async(messages))

        assert result == messages
        assert result is not messages
        assert scan_threads == []

    @pytest.mark.asyncio
    async def test_long_plain_text_history_is_copied_off_the_event_loop(
        self, monkeypatch: pytest.MonkeyPatch, scan_threads: Sequence[int]
    ):
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_CHARS", 1_000)
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_STRINGS", 10)
        messages: Final = list(chain.from_iterable(_tool_result_messages(f"rows {turn}\n" * 100) for turn in range(5)))

        result: Final = await truncate_base64_in_messages_async(messages)
        offload_threads: Final = tuple(scan_threads)

        assert result == messages
        assert result is not messages
        assert offload_threads
        assert threading.get_ident() not in offload_threads

    @pytest.mark.asyncio
    async def test_plain_text_over_the_inline_limit_is_copied_off_the_event_loop(
        self, monkeypatch: pytest.MonkeyPatch, scan_threads: Sequence[int]
    ):
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_CHARS", 1_000)
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_INLINE_COPY_MAX_CHARS", 10_000)
        messages: Final = _tool_result_messages('data: {"choices": [{"delta": {"content": "a; b"}}]}\n\n' * 400)

        result: Final = await truncate_base64_in_messages_async(messages)
        offload_threads: Final = tuple(scan_threads)

        assert result == messages
        assert result is not messages
        assert offload_threads
        assert threading.get_ident() not in offload_threads

    @pytest.mark.asyncio
    async def test_none_and_disabled_truncation_short_circuit(self, monkeypatch, scan_threads):
        assert await truncate_base64_in_messages_async(None) is None
        monkeypatch.setattr(logging_utils, "MAX_BASE64_LENGTH_FOR_LOGGING", 0)
        messages = _image_messages("K" * 20_000)
        assert await truncate_base64_in_messages_async(messages) is messages
        assert scan_threads == []


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest_asyncio.fixture(loop_scope="function")
async def drain_logging_worker(isolate_litellm_state: None) -> AsyncIterator[None]:
    yield
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS)

LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS: Final = LOGGING_WORKER_MAX_TIME_PER_COROUTINE + 5.0

@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm state to the true defaults captured at conftest import time,
    then restores after the test. This prevents module-level mutations (e.g.
    `litellm.num_retries = 3` at the top of test_langfuse_e2e_test.py) from
    leaking across tests within the same xdist worker.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging
    from litellm.proxy.management_helpers import audit_logs as ll_audit_logs

    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])

_LIST_ATTRS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
    "pre_call_rules",
    "post_call_rules",
)

_SCALAR_ATTRS = (
    "set_verbose",
    "cache",
    "num_retries",
    "num_retries_per_request",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "s3_audit_callback_params",
    "datadog_params",
    "vector_store_registry",
)

_DEFAULTS: dict = {}

@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield

@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_1(is_async):
    """
    Test 1 - ModelResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
    """

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    list_streaming_chunks = []
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-5-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    complete_streaming_response = assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime_assemble_streaming.now(),
        end_time=datetime_assemble_streaming.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("list_streaming_chunks", list_streaming_chunks)
    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert len(list_streaming_chunks) == 1
    assert list_streaming_chunks[0] == chunk

    # Add final chunk
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                finish_reason="stop",
                delta=litellm.utils.Delta(
                    content="end of response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-5-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    complete_streaming_response = assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime_assemble_streaming.now(),
        end_time=datetime_assemble_streaming.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks,
        is_async=is_async,
    )

    print("list_streaming_chunks", list_streaming_chunks)
    print("complete_streaming_response", complete_streaming_response)

    # this is the 2nd chunk - complete_streaming_response should not be None
    assert complete_streaming_response is not None
    assert len(list_streaming_chunks) == 2

    assert isinstance(complete_streaming_response, ModelResponse)
    assert isinstance(complete_streaming_response.choices[0], Choices)

    pass

@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_2(is_async):
    """
    Test 2 - TextCompletionResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
    """

    from litellm.utils import TextCompletionStreamWrapper

    _text_completion_stream_wrapper = TextCompletionStreamWrapper(completion_stream=None, model="test_model")

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    list_streaming_chunks = []
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-5-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    chunk = _text_completion_stream_wrapper.convert_to_text_completion_object(chunk)

    complete_streaming_response = assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime_assemble_streaming.now(),
        end_time=datetime_assemble_streaming.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("list_streaming_chunks", list_streaming_chunks)
    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert len(list_streaming_chunks) == 1
    assert list_streaming_chunks[0] == chunk

    # Add final chunk
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                finish_reason="stop",
                delta=litellm.utils.Delta(
                    content="end of response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-5-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    chunk = _text_completion_stream_wrapper.convert_to_text_completion_object(chunk)
    complete_streaming_response = assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime_assemble_streaming.now(),
        end_time=datetime_assemble_streaming.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks,
        is_async=is_async,
    )

    print("list_streaming_chunks", list_streaming_chunks)
    print("complete_streaming_response", complete_streaming_response)

    # this is the 2nd chunk - complete_streaming_response should not be None
    assert complete_streaming_response is not None
    assert len(list_streaming_chunks) == 2

    assert isinstance(complete_streaming_response, TextCompletionResponse)
    assert isinstance(complete_streaming_response.choices[0], TextChoices)

    pass

@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_3(is_async):

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    list_streaming_chunks_1 = []
    list_streaming_chunks_2 = []

    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-5-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    complete_streaming_response = assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime_assemble_streaming.now(),
        end_time=datetime_assemble_streaming.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks_1,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("list_streaming_chunks_1", list_streaming_chunks_1)
    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert len(list_streaming_chunks_1) == 1
    assert list_streaming_chunks_1[0] == chunk
    assert len(list_streaming_chunks_2) == 0

    # now add a chunk to the 2nd list

    complete_streaming_response = assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime_assemble_streaming.now(),
        end_time=datetime_assemble_streaming.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks_2,
        is_async=is_async,
    )

    print("list_streaming_chunks_2", list_streaming_chunks_2)
    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert len(list_streaming_chunks_2) == 1
    assert list_streaming_chunks_2[0] == chunk
    assert len(list_streaming_chunks_1) == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_4(is_async):
    """
    Test 4 - build a complete response when 1 chunk is poorly formatted

    - Assert complete_streaming_response is None
    - Assert list_streaming_chunks is not empty
    """

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    list_streaming_chunks = []

    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                finish_reason="stop",
                delta=litellm.utils.Delta(
                    content="end of response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-5-mini",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)

    # remove attribute id from chunk
    del chunk.object

    complete_streaming_response = assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime_assemble_streaming.now(),
        end_time=datetime_assemble_streaming.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks,
        is_async=is_async,
    )

    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None

    print("list_streaming_chunks", list_streaming_chunks)

    assert len(list_streaming_chunks) == 1
