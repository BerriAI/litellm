import asyncio, gzip, importlib, os
import io
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, Literal, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import FormData
from starlette.requests import Request


import litellm
import litellm.proxy.common_utils.http_parsing_utils as http_parsing_utils
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.http_parsing_utils import (
    _is_form_content_type,
    _read_request_body,
    _safe_get_request_headers,
    _safe_get_request_parsed_body,
    _safe_get_request_query_params,
    _safe_set_request_parsed_body,
    coerce_numeric_form_fields,
    get_form_data,
    get_request_body,
    get_tags_from_request_body,
    numeric_form_fields,
    populate_request_with_path_params,
    read_raw_json_body,
)
from fastapi import Request as Request_http_parsing
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.utils import _invalidate_model_cost_lowercase_map
from starlette.types import Message
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def _starlette_request(
    body: bytes,
    content_type: str,
    path: str = "/v1/messages",
    content_encoding: str = "",
    content_length: str = "",
) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [
            (b"content-type", content_type.encode()),
            (b"content-encoding", content_encoding.encode()),
            (b"content-length", content_length.encode()),
        ],
        "query_string": b"",
    }
    chunks = iter((body,))

    async def receive():
        return {"type": "http.request", "body": next(chunks, b""), "more_body": False}

    return Request(scope, receive)


@pytest.mark.asyncio
async def test_read_request_body_marks_body_received_once_with_its_size(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, dict[str, str | int]]] = []  # mutable-ok: recorder for the injected phase_event double

    def record(name: str, attributes: dict[str, str | int]) -> None:
        events.append((name, dict(attributes)))

    monkeypatch.setattr(http_parsing_utils, "phase_event", record)
    body: Final = orjson.dumps({"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x" * 4096}]})
    request: Final = _starlette_request(body, "application/json")

    assert await _read_request_body(request) == orjson.loads(body)
    assert await _read_request_body(request) == orjson.loads(body)

    assert events == [("litellm.request.body_received", {"litellm.request.body_bytes": len(body)})]


@pytest.mark.asyncio
async def test_read_request_body_marks_body_received_for_binary_and_form_bodies(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, dict[str, str | int] | None]] = []  # mutable-ok: recorder for the phase_event double

    def record(name: str, attributes: dict[str, str | int] | None) -> None:
        events.append((name, None if attributes is None else dict(attributes)))

    monkeypatch.setattr(http_parsing_utils, "phase_event", record)
    protobuf: Final = b"\x08\x96\x01" * 50
    form: Final = b"model=whisper-1&language=en"
    form_type: Final = "application/x-www-form-urlencoded"

    await _read_request_body(_starlette_request(protobuf, "application/x-protobuf"))
    await _read_request_body(_starlette_request(form, form_type, content_length=str(len(form))))
    await _read_request_body(_starlette_request(form, form_type))

    assert events == [
        ("litellm.request.body_received", {"litellm.request.body_bytes": len(protobuf)}),
        ("litellm.request.body_received", {"litellm.request.body_bytes": len(form)}),
        ("litellm.request.body_received", None),
    ]


@pytest.mark.asyncio
async def test_read_raw_json_body_returns_the_bytes_the_parsed_body_came_from():
    body = b'{"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}'
    request = _starlette_request(body, "application/json")

    assert await _read_request_body(request) == orjson.loads(body)
    assert await read_raw_json_body(request) == body


@pytest.mark.asyncio
async def test_read_raw_json_body_is_none_until_the_body_has_been_parsed():
    request = _starlette_request(b'{"model": "claude-sonnet-4-5"}', "application/json")

    assert await read_raw_json_body(request) is None
    assert await read_raw_json_body(None) is None


@pytest.mark.asyncio
async def test_read_raw_json_body_is_none_for_form_bodies():
    request = _starlette_request(b"model=claude-sonnet-4-5", "application/x-www-form-urlencoded")

    assert await _read_request_body(request) == {"model": "claude-sonnet-4-5"}
    assert await read_raw_json_body(request) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/x-protobuf", "application/protobuf; charset=binary"])
async def test_protobuf_body_is_not_parsed_as_json(content_type):
    # OTLP trace exports (POST /v1/traces) are binary protobuf; arbitrary bytes like these
    # used to hit the JSON surrogate-repair path and fail auth with a 400.
    body = b"\n\xa2\x01\n\x1c\n\x0cservice.name\x12\x0c\n\nswarm\xed\xa0\x80\xff"
    request = _starlette_request(body, content_type)

    assert await _read_request_body(request) == {}
    assert await request.body() == body  # body is still readable by the endpoint


@pytest.mark.asyncio
async def test_gzipped_json_trace_body_survives_auth_pre_read():
    body = gzip.compress(b'{"resourceSpans": []}')
    request = _starlette_request(body, "application/json", "/v1/traces", "gzip")
    assert await _read_request_body(request) == {}
    assert await request.body() == body


@pytest.mark.asyncio
async def test_read_raw_json_body_is_none_for_a_request_that_only_mocks_the_parsed_body_path():
    mock_request = MagicMock()

    assert await read_raw_json_body(mock_request) is None


@pytest.mark.asyncio
async def test_request_body_caching():
    """
    Test that the request body is cached after the first read and subsequent
    calls use the cached version instead of parsing again.
    """
    # Create a mock request with a JSON body
    mock_request = MagicMock()
    test_data = {"key": "value"}
    # Use AsyncMock for the body method
    mock_request.body = AsyncMock(return_value=orjson.dumps(test_data))
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # First call should parse the body
    result1 = await _read_request_body(mock_request)
    assert result1 == test_data
    assert "parsed_body" in mock_request.scope
    assert mock_request.scope["parsed_body"] == (("key",), {"key": "value"})

    # Verify the body was read once
    mock_request.body.assert_called_once()

    # Reset the mock to track the second call
    mock_request.body.reset_mock()

    # Second call should use the cached body
    result2 = await _read_request_body(mock_request)
    assert result2 == {"key": "value"}

    # Verify the body was not read again
    mock_request.body.assert_not_called()


@pytest.mark.asyncio
async def test_form_data_parsing():
    """
    Test that form data is correctly parsed from the request.
    """
    # Create a mock request with form data
    mock_request = MagicMock()
    test_data = {"name": "test_user", "message": "hello world"}

    # Mock the form method to return the test data as an awaitable
    mock_request.form = AsyncMock(return_value=FormData(test_data))
    mock_request.headers = {"content-type": "application/x-www-form-urlencoded"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the form data was correctly parsed
    assert result == test_data
    assert "parsed_body" in mock_request.scope
    assert mock_request.scope["parsed_body"] == (
        ("name", "message"),
        {"name": "test_user", "message": "hello world"},
    )

    # Verify form() was called
    mock_request.form.assert_called_once()

    # The body method should not be called for form data
    assert not hasattr(mock_request, "body") or not mock_request.body.called


@pytest.mark.asyncio
async def test_form_data_with_json_metadata():
    """
    Test that form data with a JSON-encoded metadata field is correctly parsed.

    When form data includes a 'metadata' field, it comes as a JSON string that needs
    to be parsed into a Python dictionary (lines 42-43 of http_parsing_utils.py).
    """
    # Create a mock request with form data containing JSON metadata
    mock_request = MagicMock()

    # Metadata is sent as a JSON string in form data
    metadata_json_string = json.dumps(
        {
            "user_id": "12345",
            "request_type": "audio_transcription",
            "tags": ["urgent", "production"],
            "custom_field": {"nested": "value"},
        }
    )

    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": metadata_json_string,  # This is a JSON string, not a dict
    }

    # Mock the form method to return the test data as an awaitable
    mock_request.form = AsyncMock(return_value=FormData(test_data))
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata was parsed from JSON string to dict
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["user_id"] == "12345"
    assert result["metadata"]["request_type"] == "audio_transcription"
    assert result["metadata"]["tags"] == ["urgent", "production"]
    assert result["metadata"]["custom_field"] == {"nested": "value"}

    # Verify other fields remain unchanged
    assert result["model"] == "whisper-1"
    assert result["file"] == "audio.mp3"

    # Verify form() was called
    mock_request.form.assert_called_once()


@pytest.mark.asyncio
async def test_form_data_with_invalid_json_metadata():
    """
    Test that form data with invalid JSON in metadata field raises an exception.

    This tests error handling when the metadata field contains malformed JSON.
    """
    # Create a mock request with form data containing invalid JSON metadata
    mock_request = MagicMock()

    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": '{"invalid": json}',  # Invalid JSON - unquoted value
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=FormData(test_data))
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    # Should raise JSONDecodeError when trying to parse invalid JSON metadata
    with pytest.raises(json.JSONDecodeError):
        await _read_request_body(mock_request)


@pytest.mark.asyncio
async def test_form_data_without_metadata():
    """
    Test that form data without metadata field works correctly.

    Ensures the metadata parsing logic doesn't break when metadata is absent.
    """
    # Create a mock request with form data without metadata
    mock_request = MagicMock()

    test_data = {"model": "whisper-1", "file": "audio.mp3", "language": "en"}

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=FormData(test_data))
    mock_request.headers = {"content-type": "application/x-www-form-urlencoded"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify all fields are preserved as-is
    assert result == test_data
    assert "metadata" not in result
    assert result["model"] == "whisper-1"
    assert result["file"] == "audio.mp3"
    assert result["language"] == "en"


@pytest.mark.asyncio
async def test_form_data_with_empty_metadata():
    """
    Test that form data with empty JSON object in metadata field is parsed correctly.
    """
    # Create a mock request with form data containing empty metadata
    mock_request = MagicMock()

    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": "{}",  # Empty JSON object as string
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=FormData(test_data))
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata was parsed to an empty dict
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
    assert result["metadata"] == {}
    assert result["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_form_data_with_dict_metadata():
    """
    Test that form data with metadata already as a dict is not parsed again.

    This handles edge cases where metadata might already be a dictionary
    (shouldn't happen in normal form data, but defensive coding).
    """
    # Create a mock request with form data where metadata is already a dict
    mock_request = MagicMock()

    metadata_dict = {"user_id": "12345", "tags": ["test"]}

    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": metadata_dict,  # Already a dict, not a string
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=FormData(test_data))
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata remains as a dict and is not parsed
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
    assert result["metadata"] == metadata_dict
    assert result["metadata"]["user_id"] == "12345"
    assert result["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_form_data_with_none_metadata():
    """
    Test that form data with None metadata value is handled gracefully.
    """
    # Create a mock request with form data where metadata is None
    mock_request = MagicMock()

    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": None,  # None value
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=FormData(test_data))
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata remains None (not parsed)
    assert "metadata" in result
    assert result["metadata"] is None
    assert result["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_empty_request_body():
    """
    Test handling of empty request bodies.
    """
    # Create a mock request with an empty body
    mock_request = MagicMock()
    mock_request.body = AsyncMock(return_value=b"")  # Empty bytes as an awaitable
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # Parse the empty body
    result = await _read_request_body(mock_request)

    # Verify an empty dict is returned
    assert result == {}
    assert "parsed_body" in mock_request.scope
    assert mock_request.scope["parsed_body"] == ((), {})

    # Verify the body was read
    mock_request.body.assert_called_once()


@pytest.mark.asyncio
async def test_circular_reference_handling():
    """
    Test that cached request body isn't modified when the returned result is modified.
    Demonstrates the mutable dictionary reference issue.
    """
    # Create a mock request with initial data
    mock_request = MagicMock()
    initial_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    mock_request.body = AsyncMock(return_value=orjson.dumps(initial_body))
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # First parse
    result = await _read_request_body(mock_request)

    # Verify initial parse
    assert result["model"] == "gpt-4"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]

    # Modify the result by adding proxy_server_request
    result["proxy_server_request"] = {
        "url": "http://0.0.0.0:4000/v1/chat/completions",
        "method": "POST",
        "headers": {"content-type": "application/json"},
        "body": result,  # Creates circular reference
    }

    # Second parse using the same request - will use the modified cached value
    result2 = await _read_request_body(mock_request)
    assert "proxy_server_request" not in result2  # This will pass, showing the cache pollution


@pytest.mark.asyncio
async def test_json_parsing_error_handling():
    """
    Test that JSON parsing errors are properly handled and raise ProxyException
    with appropriate error messages.
    """
    # Test case 1: Trailing comma error
    mock_request = MagicMock()
    invalid_json_with_trailing_comma = b"""{
        "model": "gpt-4o",
        "tools": [
            {
                "type": "mcp",
                "server_label": "litellm",
                "headers": {
                    "x-litellm-api-key": "Bearer sk-9876",
                }
            }
        ],
        "input": "Run available tools"
    }"""

    mock_request.body = AsyncMock(return_value=invalid_json_with_trailing_comma)
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # Should raise ProxyException for trailing comma
    with pytest.raises(ProxyException) as exc_info:
        await _read_request_body(mock_request)

    assert exc_info.value.code == "400"
    assert "Invalid JSON payload" in exc_info.value.message
    assert "trailing comma" in exc_info.value.message

    # Test case 2: Unquoted property name error
    mock_request2 = MagicMock()
    invalid_json_unquoted_property = b"""{
        "model": "gpt-4o",
        "tools": [
            {
                type: "mcp",
                "server_label": "litellm"
            }
        ],
        "input": "Run available tools"
    }"""

    mock_request2.body = AsyncMock(return_value=invalid_json_unquoted_property)
    mock_request2.headers = {"content-type": "application/json"}
    mock_request2.scope = {}

    # Should raise ProxyException for unquoted property
    with pytest.raises(ProxyException) as exc_info2:
        await _read_request_body(mock_request2)

    assert exc_info2.value.code == "400"
    assert "Invalid JSON payload" in exc_info2.value.message

    # Test case 3: Valid JSON should work normally
    mock_request3 = MagicMock()
    valid_json = b"""{
        "model": "gpt-4o",
        "tools": [
            {
                "type": "mcp",
                "server_label": "litellm",
                "headers": {
                    "x-litellm-api-key": "Bearer sk-9876"
                }
            }
        ],
        "input": "Run available tools"
    }"""

    mock_request3.body = AsyncMock(return_value=valid_json)
    mock_request3.headers = {"content-type": "application/json"}
    mock_request3.scope = {}

    # Should parse successfully
    result = await _read_request_body(mock_request3)
    assert result["model"] == "gpt-4o"
    assert result["input"] == "Run available tools"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "mcp"


def _make_json_request(body: bytes) -> MagicMock:
    mock_request = MagicMock()
    mock_request.body = AsyncMock(return_value=body)
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}
    return mock_request


@pytest.mark.asyncio
async def test_surrogate_repair_skipped_above_size_limit(monkeypatch):
    """
    The surrogate-repair fallback runs two full-body re.sub passes that block the
    event loop on multi-MB malformed bodies. Above MAX_REQUEST_BODY_SIZE_TO_REPAIR_MB
    the repair must be skipped and the existing 400 raised immediately, while bodies
    at or below the limit still get repaired.

    `NaN` is rejected by orjson and accepted by the json fallback, so a body containing
    it is only salvaged when the repair path runs.
    """
    import litellm.proxy.common_utils.http_parsing_utils as http_parsing_utils

    # Cap the repair at ~100 bytes so the test stays fast and independent of the default.
    monkeypatch.setattr(http_parsing_utils, "MAX_REQUEST_BODY_SIZE_TO_REPAIR_MB", 100 / (1024 * 1024))

    small_body = b'{"model":"gpt-4o","x":NaN}'
    assert len(small_body) <= 100
    repaired = await _read_request_body(_make_json_request(small_body))
    assert repaired["model"] == "gpt-4o"

    padding = "a" * 200
    large_body = b'{"model":"gpt-4o","pad":"' + padding.encode() + b'","x":NaN}'
    assert len(large_body) > 100
    with pytest.raises(ProxyException) as exc_info:
        await _read_request_body(_make_json_request(large_body))
    assert exc_info.value.code == "400"
    assert "Invalid JSON payload" in exc_info.value.message

    # Disabling the cap (0) restores repair for the same large body, proving the cap
    # — not the malformed content — is what short-circuits the repair.
    monkeypatch.setattr(http_parsing_utils, "MAX_REQUEST_BODY_SIZE_TO_REPAIR_MB", 0)
    repaired_large = await _read_request_body(_make_json_request(large_body))
    assert repaired_large["model"] == "gpt-4o"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"say ok \\ud83d", id="lone-high-surrogate"),
        pytest.param(b"say ok \\ude00", id="lone-low-surrogate"),
        pytest.param(b"\\ud83d\\ud83d\\ude00", id="lone-high-before-valid-pair"),
    ],
)
async def test_lone_surrogate_escape_is_rejected_with_400(content: bytes):
    """
    orjson rejects a lone surrogate escape, and the json fallback accepts it, so the
    parsed body used to carry a code point no provider request can UTF-8 encode. That
    surfaced as a 500 from the provider handler instead of a 400 for the bad input.
    """
    body = b'{"model":"gpt-4o","messages":[{"role":"user","content":"' + content + b'"}]}'
    with pytest.raises(ProxyException) as exc_info:
        await _read_request_body(_make_json_request(body))
    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"
    assert "Invalid JSON payload" in exc_info.value.message

    paired = body.replace(content, b"say ok \\ud83d\\ude00")
    parsed = await _read_request_body(_make_json_request(paired))
    assert parsed["messages"][0]["content"] == "say ok \U0001f600"


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ["application/x-protobuf", "application/protobuf", "application/octet-stream"])
async def test_json_body_under_a_binary_content_type_is_still_parsed(media_type: str):
    request = _starlette_request(b'{"model": "claude-sonnet-5"}', media_type)
    assert await _read_request_body(request) == {"model": "claude-sonnet-5"}


@pytest.mark.asyncio
async def test_get_form_data():
    """
    A repeated `foo[]` key is how the OpenAI SDKs send a list, so every value has to
    survive. `FormData`, not a dict: a dict cannot even hold the duplicate key.
    """
    mock_request = MagicMock()
    mock_request.form = AsyncMock(
        return_value=FormData(
            [
                ("file", "file_object"),
                ("model", "gpt-4o-transcribe"),
                ("include[]", "logprobs"),
                ("language", "en"),
                ("prompt", "Transcribe this audio file"),
                ("response_format", "json"),
                ("stream", "false"),
                ("temperature", "0.2"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ]
        )
    )

    result = await get_form_data(mock_request)

    assert result["file"] == "file_object"
    assert result["model"] == "gpt-4o-transcribe"
    assert result["language"] == "en"
    assert result["prompt"] == "Transcribe this audio file"
    assert result["response_format"] == "json"
    assert result["stream"] == "false"
    assert result["temperature"] == "0.2"
    assert result["include"] == ["logprobs"]
    assert result["timestamp_granularities"] == ["word", "segment"]


def test_get_tags_from_request_body_with_metadata_tags():
    """
    Test that tags are correctly extracted from request body metadata.
    """
    request_body = {"model": "gpt-4", "metadata": {"tags": ["tag1", "tag2", "tag3"]}}

    result = get_tags_from_request_body(request_body=request_body)

    assert result == ["tag1", "tag2", "tag3"]


def test_get_tags_from_request_body_with_litellm_metadata_tags():
    """
    Test that tags are correctly extracted from request body when using litellm_metadata.
    """
    request_body = {
        "model": "gpt-4",
        "litellm_metadata": {"tags": ["tag1", "tag2", "tag3"]},
    }

    result = get_tags_from_request_body(request_body=request_body)

    assert result == ["tag1", "tag2", "tag3"]


def test_get_tags_from_request_body_with_root_tags():
    """
    Test that tags are correctly extracted from root level of request body.
    """
    request_body = {"model": "gpt-4", "tags": ["tag1", "tag2"]}

    result = get_tags_from_request_body(request_body=request_body)

    assert result == ["tag1", "tag2"]


def test_get_tags_from_request_body_with_combined_tags():
    """
    Test that tags from both metadata and root level are combined.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": {"tags": ["tag1", "tag2"]},
        "tags": ["tag3", "tag4"],
    }

    result = get_tags_from_request_body(request_body=request_body)

    assert result == ["tag1", "tag2", "tag3", "tag4"]


def test_get_tags_from_request_body_filters_non_strings():
    """
    Test that non-string values in tags list are filtered out.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": {"tags": ["tag1", 123, "tag2", None, "tag3", {"nested": "dict"}]},
    }

    result = get_tags_from_request_body(request_body=request_body)

    assert result == ["tag1", "tag2", "tag3"]


def test_get_tags_from_request_body_no_tags():
    """
    Test that empty list is returned when no tags are present.
    """
    request_body = {"model": "gpt-4", "metadata": {}}

    result = get_tags_from_request_body(request_body=request_body)

    assert result == []


def test_get_tags_from_request_body_with_dict_tags():
    """
    Test that function handles dict tags gracefully without crashing.
    When tags is a dict instead of a list, it should be ignored and return empty list.
    """
    request_body = {
        "model": "aws/anthropic/bedrock-claude-3-5-sonnet-v1",
        "messages": [{"role": "user", "content": "aloha"}],
        "metadata": {
            "tags": {
                "litellm_id": "litellm_ratelimit_test",
                "llm_id": "llmid_ratelimit_test",
            }
        },
    }

    result = get_tags_from_request_body(request_body=request_body)

    assert result == []
    assert isinstance(result, list)


def test_get_tags_from_request_body_with_null_metadata():
    """
    Test that function handles null metadata gracefully without crashing.

    This is a regression test for https://github.com/BerriAI/litellm/issues/17263
    When metadata is explicitly set to null/None, the function should return
    an empty list instead of raising AttributeError.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": None,  # OpenAI API accepts metadata: null
    }

    result = get_tags_from_request_body(request_body=request_body)

    assert result == []
    assert isinstance(result, list)


def test_populate_request_with_path_params_adds_query_params():
    """
    Test that populate_request_with_path_params correctly adds query parameters
    like organization_id to the request data.
    """
    # Create a mock request with query parameters
    mock_request = MagicMock()
    # Mock query_params as a dict-like object that can be converted to dict
    mock_request.query_params = {"organization_id": "org-123", "user_id": "user-456"}
    mock_request.path_params = {}
    # Mock url.path to avoid errors in _add_vector_store_id_from_path
    mock_request.url.path = "/v1/chat/completions"

    # Initial request data without query params
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Call the function
    result = populate_request_with_path_params(request_data, mock_request)

    # Verify query params were added
    assert result["organization_id"] == "org-123"
    assert result["user_id"] == "user-456"
    # Verify original data is preserved
    assert result["model"] == "gpt-4"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


def test_populate_request_with_path_params_does_not_overwrite_existing_values():
    """
    Test that populate_request_with_path_params does not overwrite existing values
    in request_data when query params contain the same keys.
    """
    # Create a mock request with query parameters
    mock_request = MagicMock()
    # Mock query_params as a dict-like object that can be converted to dict
    mock_request.query_params = {
        "organization_id": "org-query-param",
        "model": "gpt-3.5-turbo",
    }
    mock_request.path_params = {}
    # Mock url.path to avoid errors in _add_vector_store_id_from_path
    mock_request.url.path = "/v1/chat/completions"

    # Initial request data with existing values
    request_data = {
        "model": "gpt-4",  # This should NOT be overwritten
        "organization_id": "org-existing",  # This should NOT be overwritten
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Call the function
    result = populate_request_with_path_params(request_data, mock_request)

    # Verify existing values were NOT overwritten
    assert result["model"] == "gpt-4"  # Should keep original, not "gpt-3.5-turbo"
    assert result["organization_id"] == "org-existing"  # Should keep original, not "org-query-param"
    # Verify other data is preserved
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_request_body_with_html_script_tags():
    """
    Test that JSON request bodies containing HTML tags like <script> are
    parsed correctly without being blocked or modified.

    Regression test for GitHub issue #20441:
    https://github.com/BerriAI/litellm/issues/20441

    LLM message content frequently contains HTML/code snippets.
    The HTTP parsing layer must not interfere with such content.
    """
    test_messages = [
        {
            "role": "user",
            "content": "<script>alert('hello')</script>",
        },
        {
            "role": "user",
            "content": "<script> test </script>",
        },
        {
            "role": "user",
            "content": "Can you explain what <script> tags do in HTML?",
        },
        {
            "role": "user",
            "content": "Here is code: <div><script src='app.js'></script></div>",
        },
        {
            "role": "user",
            "content": "<img onerror='alert(1)' src='x'>",
        },
        {
            "role": "user",
            "content": "<iframe src='https://example.com'></iframe>",
        },
    ]

    for msg in test_messages:
        test_payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                msg,
            ],
        }

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=orjson.dumps(test_payload))
        mock_request.headers = {"content-type": "application/json"}
        mock_request.scope = {}

        result = await _read_request_body(mock_request)

        assert result["model"] == "gpt-4o"
        assert len(result["messages"]) == 3
        assert result["messages"][2]["content"] == msg["content"], (
            f"Message content with HTML was modified during parsing: "
            f"expected={msg['content']!r}, got={result['messages'][2]['content']!r}"
        )


def test_safe_get_request_headers_caches_on_request_state():
    """
    Test that _safe_get_request_headers caches the result on request.state
    and returns the same object on subsequent calls.
    """
    mock_request = MagicMock()
    mock_request.headers = {
        "content-type": "application/json",
        "authorization": "Bearer sk-123",
    }
    mock_request.state = MagicMock(spec=[])  # empty spec so getattr returns default

    # First call — should create and cache
    result1 = _safe_get_request_headers(mock_request)
    assert result1 == {
        "content-type": "application/json",
        "authorization": "Bearer sk-123",
    }
    assert mock_request.state._cached_headers is result1

    # Second call — should return the cached object (same identity)
    result2 = _safe_get_request_headers(mock_request)
    assert result2 is result1


def test_safe_get_request_headers_none_request():
    """
    Test that _safe_get_request_headers returns empty dict for None request.
    """
    result = _safe_get_request_headers(None)
    assert result == {}


def test_safe_get_request_headers_copy_protects_cache():
    """
    Test that callers using .copy() before mutation do not corrupt the cache.
    """
    mock_request = MagicMock()
    mock_request.headers = {"authorization": "Bearer sk-123", "host": "localhost"}
    mock_request.state = MagicMock(spec=[])

    original = _safe_get_request_headers(mock_request)

    # Simulate what mutation call sites do: copy then pop
    mutable = _safe_get_request_headers(mock_request).copy()
    mutable.pop("authorization", None)

    # Cache must be unaffected
    assert "authorization" in _safe_get_request_headers(mock_request)
    assert _safe_get_request_headers(mock_request) is original


def test_safe_get_request_headers_state_unavailable():
    """
    Test that _safe_get_request_headers still returns headers when
    request.state rejects attribute writes (the except path on the cache-write).
    """

    class ReadOnlyState:
        """State object that allows reads but raises on writes."""

        def __setattr__(self, name, value):
            raise AttributeError("read-only state")

        def __getattr__(self, name):
            return None  # _cached_headers not found → triggers fresh read

    mock_request = MagicMock()
    mock_request.headers = {"content-type": "application/json"}
    mock_request.state = ReadOnlyState()

    result = _safe_get_request_headers(mock_request)
    assert result == {"content-type": "application/json"}


class TestGetTagsFromRequestBodyStringCoerce:
    """Regression: the auth-time tag helper used `metadata.get("tags", ...)`
    directly, which raised AttributeError when metadata arrived as a JSON
    string (multipart/form-data or extra_body). That turned into a DoS at
    auth time and potentially bypassed tag-based RBAC if the caller caught
    the exception and fell through with empty tags.
    """

    def test_json_string_metadata_is_coerced_to_dict(self):
        from litellm.proxy.common_utils.http_parsing_utils import (
            get_tags_from_request_body,
        )

        metadata_json = json.dumps({"tags": ["a", "b"]})
        # Must not raise
        tags = get_tags_from_request_body({"metadata": metadata_json})
        assert tags == ["a", "b"]

    def test_unparseable_string_metadata_is_ignored(self):
        from litellm.proxy.common_utils.http_parsing_utils import (
            get_tags_from_request_body,
        )

        # Must not raise; must yield no metadata tags but keep root tags
        tags = get_tags_from_request_body({"metadata": "not-json", "tags": ["root-only"]})
        assert tags == ["root-only"]

    def test_dict_metadata_still_works(self):
        from litellm.proxy.common_utils.http_parsing_utils import (
            get_tags_from_request_body,
        )

        tags = get_tags_from_request_body({"metadata": {"tags": ["x"]}})
        assert tags == ["x"]


class TestIsFormContentType:
    @pytest.mark.parametrize(
        "content_type",
        [
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "multipart/form-data; boundary=----WebKitFormBoundary",
            "Application/X-WWW-Form-Urlencoded",
            "  multipart/form-data  ",
            "application/x-www-form-urlencoded; charset=utf-8",
        ],
    )
    def test_form_types_match(self, content_type):
        assert _is_form_content_type(content_type) is True

    @pytest.mark.parametrize(
        "content_type",
        [
            "",
            "application/json",
            "application/json; charset=utf-8",
            "application/form-json",
            "multiform/anything",
            "application/json; xform=1",
            "application/xml-with-form-data-but-not-actually",
            "text/plain",
            "form",
        ],
    )
    def test_non_form_types_rejected(self, content_type):
        assert _is_form_content_type(content_type) is False


class TestReadRequestBodyNonCanonicalContentType:
    """A JSON body with a ``"form"``-substring Content-Type must parse as JSON."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content_type",
        [
            "application/form-json",
            "application/json; xform=1",
            "multiform/anything",
        ],
    )
    async def test_json_body_with_formlike_content_type_parses_as_json(self, content_type):
        payload = {"user_config": {"model_list": []}, "model": "x"}

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=orjson.dumps(payload))
        mock_request.form = AsyncMock(return_value=FormData({}))
        mock_request.headers = {"content-type": content_type}
        mock_request.scope = {}

        result = await _read_request_body(mock_request)
        assert result == payload
        mock_request.form.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_form_post_still_parsed_as_form(self):
        mock_request = MagicMock()
        mock_request.form = AsyncMock(return_value=FormData({"k": "v"}))
        mock_request.body = AsyncMock(return_value=b"")
        mock_request.headers = {"content-type": "application/x-www-form-urlencoded"}
        mock_request.scope = {}

        result = await _read_request_body(mock_request)
        assert result == {"k": "v"}
        mock_request.form.assert_awaited_once()


class TestReadRequestBodyFormParseFailure:
    """
    A failed ``request.form()`` parse (e.g. multipart with missing boundary)
    must surface as a 400, not silently return ``{}`` — otherwise the
    auth-time pre-read sees an empty body while a later raw-body re-read
    sees the original payload, defeating every banned-param check.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised_exception",
        [
            ValueError("Missing boundary in multipart."),
            AssertionError("malformed chunk"),
            RuntimeError("form parser exploded"),
        ],
    )
    async def test_form_parse_failure_raises_400(self, raised_exception):
        mock_request = MagicMock()
        mock_request.form = AsyncMock(side_effect=raised_exception)
        mock_request.headers = {"content-type": "multipart/form-data"}
        mock_request.scope = {}

        with pytest.raises(ProxyException) as exc_info:
            await _read_request_body(mock_request)
        assert str(exc_info.value.code) == "400"


class TestGetRequestBody:
    @pytest.mark.asyncio
    async def test_json_with_charset_param_parses_as_json(self):
        payload = {"k": "v"}
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=orjson.dumps(payload))
        mock_request.headers = {"content-type": "application/json; charset=utf-8"}
        mock_request.scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}

        result = await get_request_body(mock_request)
        assert result == payload

    @pytest.mark.asyncio
    async def test_form_post_routes_to_form_data(self):
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "multipart/form-data; boundary=x"}
        mock_request.form = AsyncMock(return_value=FormData({"k": "v"}))
        mock_request.scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}

        result = await get_request_body(mock_request)
        assert result == {"k": "v"}

    @pytest.mark.asyncio
    async def test_substring_match_no_longer_accepted(self):
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/form-json"}
        mock_request.scope = {}

        with pytest.raises(ValueError, match="Unsupported content type"):
            await get_request_body(mock_request)

    @pytest.mark.asyncio
    async def test_non_post_returns_empty(self):
        mock_request = MagicMock()
        mock_request.method = "GET"
        assert await get_request_body(mock_request) == {}


class TestNumericFormFields:
    def test_image_edit_schema_yields_only_n(self):
        from litellm.types.images.main import ImageEditRequestParams

        assert dict(numeric_form_fields(get_type_hints(ImageEditRequestParams))) == {"n": int}

    def test_qualifiers_and_optionality_are_unwrapped(self):
        from typing import Optional

        from typing_extensions import Annotated, NotRequired, ReadOnly, Required, TypedDict

        class Schema(TypedDict, total=False):
            plain: int
            optional: Optional[int]
            piped: int | None
            read_only: ReadOnly[int | None]
            not_required: NotRequired[ReadOnly[int]]
            required: Required[ReadOnly[Annotated[float, "meta"]]]
            read_only_not_required: ReadOnly[NotRequired[int]]
            read_only_required: ReadOnly[Required[float]]

        assert dict(numeric_form_fields(get_type_hints(Schema))) == {
            "plain": int,
            "optional": int,
            "piped": int,
            "read_only": int,
            "not_required": int,
            "required": float,
            "read_only_not_required": int,
            "read_only_required": float,
        }

    def test_qualifiers_are_unwrapped_when_get_type_hints_keeps_extras(self):
        from typing_extensions import Annotated, NotRequired, ReadOnly, Required, TypedDict

        class Schema(TypedDict, total=False):
            annotated: ReadOnly[Annotated[int, "meta"]]
            not_required: NotRequired[ReadOnly[int]]
            required: Required[ReadOnly[Annotated[float, "meta"]]]

        assert dict(numeric_form_fields(get_type_hints(Schema, include_extras=True))) == {
            "annotated": int,
            "not_required": int,
            "required": float,
        }

    def test_non_scalar_and_bool_fields_are_skipped(self):
        from typing import Any, Literal, Optional, Union

        from typing_extensions import TypedDict

        class Schema(TypedDict, total=False):
            flag: bool
            optional_flag: Optional[bool]
            text: str
            choice: Optional[Literal["high", "low"]]
            numbers: list[int]
            mapping: Optional[dict[str, Any]]
            ambiguous: Union[int, str]

        assert dict(numeric_form_fields(get_type_hints(Schema))) == {}


class TestCoerceNumericFormFields:
    numeric_fields = {"n": int, "temperature": float}

    def test_numeric_strings_are_parsed(self):
        assert coerce_numeric_form_fields(
            parsed_body={"n": "2", "temperature": "0.5"},
            numeric_fields=self.numeric_fields,
        ) == {"n": 2, "temperature": 0.5}

    def test_other_fields_keep_their_string_values(self):
        result = coerce_numeric_form_fields(
            parsed_body={"size": "1024x1024", "prompt": "2", "quality": "high"},
            numeric_fields=self.numeric_fields,
        )
        assert result == {"size": "1024x1024", "prompt": "2", "quality": "high"}

    def test_unparseable_value_is_left_for_the_provider_to_reject(self):
        assert coerce_numeric_form_fields(
            parsed_body={"n": "two", "temperature": ""},
            numeric_fields=self.numeric_fields,
        ) == {"n": "two", "temperature": ""}

    def test_already_typed_and_non_string_values_pass_through(self):
        buffer = io.BytesIO(b"png")
        result = coerce_numeric_form_fields(
            parsed_body={"n": 3, "temperature": None, "image": buffer},
            numeric_fields=self.numeric_fields,
        )
        assert result == {"n": 3, "temperature": None, "image": buffer}


@pytest.mark.parametrize(
    "kind,settings,cli,path,body,expected",
    [
        ("completion", {"completion_model": "default"}, "cli", "path", "body", "default"),
        ("completion", {}, "cli", "path", "body", "cli"),
        ("completion", {}, None, "path", "body", "path"),
        ("completion", {}, None, None, "body", "body"),
        (
            "image_generation",
            {"completion_model": "text", "image_generation_model": "image"},
            None,
            None,
            "body",
            "image",
        ),
        ("image_generation", {"image_generation_model": "image"}, "cli", "path", "body", "cli"),
        ("image_generation", {"image_generation_model": "image"}, None, "path", "body", "path"),
        ("image_edit", {"completion_model": "text", "image_generation_model": "image"}, None, None, "body", "text"),
        ("image_edit", {"image_generation_model": "image"}, None, "path", "body", "path"),
        ("image_edit", {"image_generation_model": "image"}, None, None, "body", "image"),
        ("moderation", {"moderation_model": "mod"}, "cli", None, "body", "cli"),
        ("speech", {"completion_model": "text"}, None, None, "body", "body"),
        ("body", {"completion_model": "text"}, "cli", None, "body", "body"),
        ("path", {"completion_model": "text"}, "cli", "path", "body", "path"),
    ],
)
def test_shared_inference_model_selection_preserves_handler_precedence(
    kind: Literal["completion", "image_generation", "image_edit", "moderation", "speech", "body", "path"],
    settings: Mapping[str, object],
    cli: str | None,
    path: str | None,
    body: str,
    expected: str,
) -> None:
    from litellm.proxy.common_utils.http_parsing_utils import resolve_inference_model

    assert resolve_inference_model(body, settings, cli, path, kind=kind) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path,skip_parse",
    [
        ("POST", "/v1/traces", True),
        ("POST", "/v1/logs", True),
        ("GET", "/v1/logs", False),
        ("GET", "/v1/traces", False),
        ("POST", "/v1/messages", False),
        ("POST", "/v1/traces/other", False),
    ],
)
@pytest.mark.parametrize("root_path", ["", "/tenant-a"])
async def test_only_trace_ingest_skips_json_body(method: str, path: str, skip_parse: bool, root_path: str) -> None:
    body: Final = b'{"key":"value"}'
    receive: Final = AsyncMock(return_value={"type": "http.request", "body": body, "more_body": False})
    request: Final = Request(
        {
            "type": "http",
            "method": method,
            "path": root_path + path,
            "root_path": root_path,
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    parsed: Final = await _read_request_body(request)
    if skip_parse:
        assert parsed == {}
        receive.assert_not_awaited()
    else:
        assert parsed == {"key": "value"}
        receive.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type, encoding",
    [
        ("application/json", ""),
        ("application/x-protobuf", ""),
        ("application/json", "gzip"),
    ],
)
async def test_otlp_auth_does_not_consume_chunked_bodies_before_the_receiver_limit(content_type, encoding):
    from litellm.constants import OTLP_MAX_BODY_BYTES
    from litellm.tracing import Tenant, TraceReceiver, TracingPayloadTooLargeError

    received = []
    chunk = b"x" * (OTLP_MAX_BODY_BYTES // 2 + 1)

    async def receive():
        received.append(1)
        assert len(received) <= 2, "receiver must reject without consuming subsequent chunks"
        return {"type": "http.request", "body": chunk, "more_body": True}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/traces",
            "headers": [
                (b"content-type", content_type.encode()),
                (b"content-encoding", encoding.encode()),
            ],
        },
        receive,
    )
    assert await _read_request_body(request) == {}
    assert received == []
    storage = MagicMock()
    storage.ingest = AsyncMock()
    with pytest.raises(TracingPayloadTooLargeError):
        await TraceReceiver(storage).ingest(request.stream(), content_type, encoding, Tenant("team", "key"))
    assert len(received) == 2
    storage.ingest.assert_not_awaited()


@pytest.mark.asyncio
async def test_auth_body_read_and_trace_handler_leave_stream_for_receiver_limit() -> None:
    from litellm.constants import OTLP_MAX_BODY_BYTES
    from litellm.proxy import tracing_endpoints
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _read_request_body_deferring_parse_failure
    from litellm.tracing import TraceReceiver

    chunk: Final = b"x" * (OTLP_MAX_BODY_BYTES // 2 + 1)
    receive: Final = AsyncMock(side_effect=[{"type": "http.request", "body": chunk, "more_body": True}] * 2)
    request: Final = Request(
        {"type": "http", "method": "POST", "path": "/v1/traces", "headers": [(b"content-type", b"application/json")]},
        receive,
    )
    storage: Final = MagicMock()
    storage.ingest = AsyncMock()
    context: Final = await tracing_endpoints.provide_trace_access(
        auth=UserAPIKeyAuth(token="key", team_id="team"), tracing=TraceReceiver(storage), log_team_lookup=AsyncMock()
    )

    parsed, parse_error = await _read_request_body_deferring_parse_failure(request)
    assert parsed == {}
    assert parse_error is None
    receive.assert_not_awaited()

    response: Final = await tracing_endpoints.ingest_otlp_traces(request, context)
    assert response.status_code == 413
    assert receive.await_count == 2
    storage.ingest.assert_not_awaited()


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}

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

def _request(receive: Callable[[], Awaitable[Message]]) -> Request_http_parsing:
    return Request_http_parsing(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

def _request_with_body(body: bytes) -> Request_http_parsing:
    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    return _request(receive)

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_read_request_body_valid_json():
    result = await _read_request_body(_request_with_body(b'{"key": "value"}'))
    assert result == {"key": "value"}

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_read_request_body_empty_body():
    result = await _read_request_body(_request_with_body(b""))
    assert result == {}

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_read_request_body_invalid_json():
    with pytest.raises(ProxyException):
        await _read_request_body(_request_with_body(b'{"key": value}'))

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_read_request_body_large_payload():
    large_payload = '{"key":' + '"a"' * 10**6 + "}"
    with pytest.raises(ProxyException):
        await _read_request_body(_request_with_body(large_payload.encode()))

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_read_request_body_unexpected_error():
    async def receive() -> Message:
        raise ValueError("Unexpected error")

    result = await _read_request_body(_request(receive))
    assert result == {}
