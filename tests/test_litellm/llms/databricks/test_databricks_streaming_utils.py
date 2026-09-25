"""
Regression test for the databricks streaming chunk parser.

OpenAI-compatible servers (e.g. Vertex AI Model Garden vLLM endpoints) send a final
usage-only chunk with an empty `choices` list when `stream_options.include_usage` is
set. `chunk_parser` previously did `choices[0]` unconditionally, raising
`IndexError` -> `MidStreamFallbackError` and crashing the stream.
"""

from typing import cast

from litellm.llms.databricks.streaming_utils import ModelResponseIterator, _usage_to_chat_completion_block
from litellm.types.utils import Usage


def test_chunk_parser_handles_empty_choices_usage_chunk():
    """A usage-only final chunk (empty choices) must not raise IndexError."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    usage_only_chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }

    result = iterator.chunk_parser(chunk=usage_only_chunk)

    assert result["text"] == ""
    assert result["is_finished"] is False
    assert result["usage"] is not None
    assert result["usage"]["prompt_tokens"] == 20
    assert result["usage"]["completion_tokens"] == 8


def test_chunk_parser_empty_choices_without_usage():
    """An empty-choices chunk with no usage block returns usage=None, no error."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result["text"] == ""
    assert result["usage"] is None


def test_chunk_parser_preserves_prompt_tokens_details_on_usage_only_chunk():
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    usage_only_chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {
            "prompt_tokens": 9888,
            "completion_tokens": 1,
            "total_tokens": 9889,
            "prompt_tokens_details": {"cached_tokens": 6752, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }

    result = iterator.chunk_parser(chunk=usage_only_chunk)

    assert result["usage"] is not None
    assert result["usage"]["prompt_tokens"] == 9888
    assert result["usage"]["completion_tokens"] == 1
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 6752
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


def test_chunk_parser_preserves_prompt_tokens_details_on_finish_chunk():
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 9888,
            "completion_tokens": 1,
            "total_tokens": 9889,
            "prompt_tokens_details": {"cached_tokens": 6752},
        },
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result["is_finished"] is True
    assert result["usage"] is not None
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 6752


def test_usage_to_block_returns_none_when_usage_missing():
    assert _usage_to_chat_completion_block(None) is None


def test_usage_to_block_preserves_prompt_and_completion_details():
    usage = Usage(
        prompt_tokens=9888,
        completion_tokens=1,
        total_tokens=9889,
        prompt_tokens_details={"cached_tokens": 6752, "audio_tokens": 0},
        completion_tokens_details={"reasoning_tokens": 0},
    )

    result = _usage_to_chat_completion_block(usage)

    assert result is not None
    assert result["prompt_tokens"] == 9888
    assert result["completion_tokens"] == 1
    assert result["total_tokens"] == 9889
    assert result["prompt_tokens_details"] is not None
    assert result["prompt_tokens_details"]["cached_tokens"] == 6752
    assert result["completion_tokens_details"] is not None
    assert result["completion_tokens_details"]["reasoning_tokens"] == 0


def test_usage_to_block_coerces_missing_counts_and_drops_absent_details():
    result = _usage_to_chat_completion_block(Usage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    assert result is not None
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0
    assert result["prompt_tokens_details"] is None
    assert result["completion_tokens_details"] is None


def test_usage_to_block_drops_non_dict_token_details():
    class _UsageDump:
        prompt_tokens = 3
        completion_tokens = 1
        total_tokens = 4

        def model_dump(self) -> dict[str, object]:
            return {"prompt_tokens_details": ["not-a-dict"], "completion_tokens_details": 0}

    result = _usage_to_chat_completion_block(cast(Usage, _UsageDump()))

    assert result is not None
    assert result["prompt_tokens"] == 3
    assert result["prompt_tokens_details"] is None
    assert result["completion_tokens_details"] is None


def test_chunk_parser_maps_zero_token_usage_on_finish_chunk():
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result["is_finished"] is True
    assert result["usage"] is not None
    assert result["usage"]["prompt_tokens"] == 0
    assert result["usage"]["completion_tokens"] == 0
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


def test_chunk_parser_normal_content_chunk_still_works():
    """A regular content chunk is unaffected by the empty-choices guard."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result["text"] == "hi"
