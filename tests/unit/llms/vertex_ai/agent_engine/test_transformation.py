"""
Tests for Vertex AI Agent Engine transformation.

Tests the request transformation and streaming chunk parsing without making real API calls.
"""

from copy import deepcopy
from typing import Final, Literal

import pytest

from litellm import BadRequestError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.vertex_ai.agent_engine.sse_iterator import (
    VertexAgentEngineResponseIterator,
)
from litellm.llms.vertex_ai.agent_engine.transformation import VertexAgentEngineConfig, VertexAgentEngineError
from litellm.types.llms.openai import AllMessageValues, OpenAIMessageContent, OpenAIMessageContentListBlock
from litellm.types.llms.vertex_ai import ContentType


class TestVertexAgentEngineTransformRequest:
    """Tests for transform_request method."""

    def test_transform_request_basic(self):
        """
        Test that transform_request correctly formats messages into Vertex Agent Engine payload.
        """
        config = VertexAgentEngineConfig()

        messages = [{"role": "user", "content": "Hello, what can you do?"}]
        optional_params = {"user_id": "test-user-123"}
        litellm_params = {}

        result = config.transform_request(
            model="agent_engine/123456789",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers={},
        )

        assert result["class_method"] == "stream_query"
        assert result["input"]["message"] == "Hello, what can you do?"
        assert result["input"]["user_id"] == "test-user-123"
        assert "session_id" not in result["input"]

    def test_transform_request_with_session_id(self):
        """
        Test that transform_request includes session_id when provided.
        """
        config = VertexAgentEngineConfig()

        messages = [{"role": "user", "content": "Follow up question"}]
        optional_params = {
            "user_id": "test-user-123",
            "session_id": "session-abc-456",
        }
        litellm_params = {}

        result = config.transform_request(
            model="agent_engine/123456789",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers={},
        )

        assert result["class_method"] == "stream_query"
        assert result["input"]["message"] == "Follow up question"
        assert result["input"]["user_id"] == "test-user-123"
        assert result["input"]["session_id"] == "session-abc-456"


class TestVertexAgentEngineChunkParser:
    """Tests for the streaming chunk parser."""

    def test_chunk_parser_with_text_content(self):
        """
        Test that chunk_parser correctly extracts text from Vertex Agent Engine response format.
        """
        iterator = VertexAgentEngineResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        chunk = {
            "content": {
                "parts": [{"text": "Hello! I can help you with financial analysis."}],
                "role": "model",
            },
            "finish_reason": "STOP",
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 50,
                "total_token_count": 150,
            },
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].delta.content == "Hello! I can help you with financial analysis."
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    def test_chunk_parser_without_finish_reason(self):
        """
        Test that chunk_parser handles chunks without finish_reason (intermediate chunks).
        """
        iterator = VertexAgentEngineResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        chunk = {
            "content": {
                "parts": [{"text": "Partial response..."}],
                "role": "model",
            },
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].delta.content == "Partial response..."
        assert result.choices[0].finish_reason is None
        assert result.usage is None


@pytest.mark.parametrize(
    "content, expected",
    [
        ([], ""),
        ([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "onetwo"),
        (
            [{"type": "image_url", "image_url": "gs://bucket/image.png"}],
            {"role": "user", "parts": [{"file_data": {"file_uri": "gs://bucket/image.png", "mime_type": "image/png"}}]},
        ),
        (
            [{"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg?signature=123"}}],
            {
                "role": "user",
                "parts": [
                    {
                        "file_data": {
                            "file_uri": "https://example.com/photo.jpg?signature=123",
                            "mime_type": "image/jpeg",
                        }
                    }
                ],
            },
        ),
        (
            [{"type": "image_url", "image_url": {"url": "gs://bucket/no-extension", "format": "image/webp"}}],
            {
                "role": "user",
                "parts": [{"file_data": {"file_uri": "gs://bucket/no-extension", "mime_type": "image/webp"}}],
            },
        ),
        (
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
            {"role": "user", "parts": [{"inline_data": {"data": "AA==", "mime_type": "image/png"}}]},
        ),
        (
            [{"type": "file", "file": {"file_data": "gs://bucket/report.pdf"}}],
            {
                "role": "user",
                "parts": [{"file_data": {"file_uri": "gs://bucket/report.pdf", "mime_type": "application/pdf"}}],
            },
        ),
        (
            [{"type": "file", "file": {"file_id": "gs://bucket/report.pdf"}}],
            {
                "role": "user",
                "parts": [{"file_data": {"file_uri": "gs://bucket/report.pdf", "mime_type": "application/pdf"}}],
            },
        ),
        (
            [{"type": "file", "file": {"file_data": "JVBERi0xLjQ=", "filename": "report.pdf"}}],
            {"role": "user", "parts": [{"inline_data": {"data": "JVBERi0xLjQ=", "mime_type": "application/pdf"}}]},
        ),
        (
            [{"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0xLjQ="}}],
            {"role": "user", "parts": [{"inline_data": {"data": "JVBERi0xLjQ=", "mime_type": "application/pdf"}}]},
        ),
        (
            [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "after"},
            ],
            {
                "role": "user",
                "parts": [
                    {"text": "before"},
                    {"inline_data": {"data": "AA==", "mime_type": "image/png"}},
                    {"text": "after"},
                ],
            },
        ),
    ],
)
def test_attachment_parts_reach_agent_engine_without_changing_text_path(
    content: OpenAIMessageContent, expected: str | ContentType
) -> None:
    config: Final = VertexAgentEngineConfig()
    original: Final = deepcopy(content)
    messages: Final[list[AllMessageValues]] = [{"role": "user", "content": content}]
    result: Final = config.transform_request(
        model="agent_engine/123",
        messages=messages,
        optional_params={"user_id": "user-123", "session_id": "session-456"},
        litellm_params={},
        headers={},
    )
    assert result == {
        "class_method": "stream_query",
        "input": {
            "message": expected,
            "user_id": "user-123",
            "session_id": "session-456",
        },
    }
    assert messages == [{"role": "user", "content": original}]


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
        {"type": "video_url", "video_url": {"url": "gs://bucket/movie.mp4"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        {"type": "image_url", "image_url": {"url": "data:image/png,raw"}},
        {"type": "image_url", "image_url": {"url": "file:///tmp/image.png"}},
        {"type": "image_url", "image_url": {"url": "gs://bucket/no-extension"}},
        {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,AA=="}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "format": "image/jpeg"}},
        {"type": "image_url", "image_url": {}},
        {"type": "image_url", "image_url": {"url": "https://[invalid/image.png"}},
        {"type": "image_url", "image_url": {"url": "gs://bucket"}},
        {"type": "image_url", "image_url": {"url": "https:///image.png"}},
        {"type": "file", "file": {"file_id": "file-123"}},
        {"type": "file", "file": {"file_data": "AA=="}},
        {"type": "file", "file": {"file_data": "gs://bucket/a.pdf", "file_id": "gs://bucket/b.pdf"}},
        {"type": "file", "file": {"file_data": "AA==", "format": "not-a-mime-type"}},
        {"type": "file", "file": {}},
        {"type": "text"},
        {"type": "unknown", "data": "AA=="},
    ],
)
def test_unsupported_or_invalid_parts_raise_instead_of_disappearing(part: OpenAIMessageContentListBlock) -> None:
    config: Final = VertexAgentEngineConfig()
    messages: Final[list[AllMessageValues]] = [
        {"role": "user", "content": [{"type": "text", "text": "attachment"}, part]},
    ]
    with pytest.raises(VertexAgentEngineError) as error:
        config.transform_request(
            model="agent_engine/123",
            messages=messages,
            optional_params={"user_id": "user"},
            litellm_params={},
            headers={},
        )
    assert error.value.status_code == 400
    assert "Agent Engine" in error.value.message


@pytest.mark.parametrize("role, following", [("assistant", False), ("user", True)])
def test_media_is_rejected_when_it_cannot_be_forwarded(role: Literal["assistant", "user"], following: bool) -> None:
    config: Final = VertexAgentEngineConfig()
    media: Final[AllMessageValues] = {
        "role": role,
        "content": [
            {"type": "image_url", "image_url": {"url": "gs://bucket/image.png"}},
        ],
    }
    messages: Final[list[AllMessageValues]] = [media, {"role": "user", "content": "next"}] if following else [media]
    with pytest.raises(VertexAgentEngineError) as error:
        config.transform_request(
            model="agent_engine/123",
            messages=messages,
            optional_params={"user_id": "user"},
            litellm_params={},
            headers={},
        )
    assert error.value.status_code == 400


def test_invalid_media_maps_to_public_bad_request_error() -> None:
    config: Final = VertexAgentEngineConfig()
    messages: Final[list[AllMessageValues]] = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
            ],
        }
    ]
    with pytest.raises(VertexAgentEngineError) as provider_error:
        config.transform_request(
            model="agent_engine/123",
            messages=messages,
            optional_params={"user_id": "user"},
            litellm_params={},
            headers={},
        )
    with pytest.raises(BadRequestError) as public_error:
        exception_type(
            model="agent_engine/123",
            original_exception=provider_error.value,
            custom_llm_provider="vertex_ai",
            completion_kwargs={},
            extra_kwargs={},
        )
    assert public_error.value.status_code == 400
    assert "Agent Engine" in public_error.value.message


def test_text_history_keeps_the_last_message_string_path() -> None:
    config: Final = VertexAgentEngineConfig()
    messages: Final[list[AllMessageValues]] = [
        {"role": "user", "content": [{"type": "text", "text": "earlier"}]},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "last"},
    ]
    result: Final = config.transform_request(
        model="agent_engine/123", messages=messages, optional_params={"user_id": "user"}, litellm_params={}, headers={}
    )
    assert result == {"class_method": "stream_query", "input": {"message": "last", "user_id": "user"}}


def test_assistant_without_content_preserves_empty_string_path() -> None:
    config: Final = VertexAgentEngineConfig()
    messages: Final[list[AllMessageValues]] = [{"role": "assistant", "content": None}]
    result: Final = config.transform_request(
        model="agent_engine/123", messages=messages, optional_params={"user_id": "user"}, litellm_params={}, headers={}
    )
    assert result == {"class_method": "stream_query", "input": {"message": "", "user_id": "user"}}


@pytest.mark.parametrize(
    "messages, expected_error",
    [
        ([], "Agent Engine requires at least one message"),
        (
            [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "gs://bucket"}}]},
                {"role": "user", "content": "next"},
            ],
            "Agent Engine requires a valid gs:// or https:// media URI",
        ),
    ],
)
def test_invalid_messages_fail_before_forwarding_the_last_text(
    messages: list[AllMessageValues], expected_error: str
) -> None:
    config: Final = VertexAgentEngineConfig()
    with pytest.raises(VertexAgentEngineError) as error:
        config.transform_request(
            model="agent_engine/123",
            messages=messages,
            optional_params={"user_id": "user"},
            litellm_params={},
            headers={},
        )
    assert error.value.status_code == 400
    assert error.value.message == expected_error
