"""
Tests for synchronous streaming methods added to AnthropicAdapter and GoogleGenAIAdapter.

These tests verify that `sync_translate_completion_output_params_streaming` returns
a synchronous Iterator[bytes] (not an AsyncIterator) for both adapters.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class MockSyncCompletionStream:
    """A synchronous mock completion stream that yields ModelResponseStream chunks."""

    def __init__(self):
        self.responses = [
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content="Hello"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=" World"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=""), index=0, finish_reason="stop"
                    )
                ],
            ),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


# --- AnthropicAdapter sync streaming tests ---


def test_anthropic_sync_translate_returns_iterator():
    """
    Test that AnthropicAdapter.sync_translate_completion_output_params_streaming
    returns a synchronous iterator of bytes.
    """
    adapter = AnthropicAdapter()
    result = adapter.sync_translate_completion_output_params_streaming(
        completion_stream=MockSyncCompletionStream(),
        model="claude-3",
    )

    assert result is not None
    # Should be a generator / iterator, not a coroutine or async iterator
    assert hasattr(result, "__iter__")
    assert not hasattr(result, "__aiter__")

    chunks: list[bytes] = list(result)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, bytes)


def test_anthropic_sync_translate_produces_valid_sse():
    """
    Test that AnthropicAdapter.sync_translate_completion_output_params_streaming
    produces valid SSE-formatted bytes with event and data lines.
    """
    adapter = AnthropicAdapter()
    result = adapter.sync_translate_completion_output_params_streaming(
        completion_stream=MockSyncCompletionStream(),
        model="claude-3",
    )

    assert result is not None
    first_chunk = next(iter(result))
    chunk_str = first_chunk.decode("utf-8")

    lines = chunk_str.split("\n")
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    assert "message_start" in chunk_str


def test_anthropic_sync_translate_with_tool_name_mapping():
    """
    Test that tool_name_mapping parameter is accepted by
    sync_translate_completion_output_params_streaming.
    """
    adapter = AnthropicAdapter()
    result = adapter.sync_translate_completion_output_params_streaming(
        completion_stream=MockSyncCompletionStream(),
        model="claude-3",
        tool_name_mapping={"short_name": "original_long_tool_name"},
    )

    assert result is not None
    chunks = list(result)
    assert len(chunks) > 0


# --- GoogleGenAIAdapter sync streaming tests ---


class MockGoogleGenAIDictStream:
    """A synchronous mock stream that yields pre-transformed Google GenAI dict chunks.

    This simulates what GoogleGenAIStreamWrapper.__next__ produces after transforming
    ModelResponseStream chunks into Google GenAI format dicts.
    """

    def __init__(self):
        self.responses = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello"}],
                            "role": "model",
                        },
                        "index": 0,
                        "safetyRatings": [],
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": " World"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                        "safetyRatings": [],
                    }
                ]
            },
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def test_google_genai_sync_translate_returns_iterator():
    """
    Test that GoogleGenAIAdapter.sync_translate_completion_output_params_streaming
    returns a synchronous iterator of bytes.
    """
    adapter = GoogleGenAIAdapter()
    result = adapter.sync_translate_completion_output_params_streaming(
        completion_stream=MockGoogleGenAIDictStream(),
    )

    assert result is not None
    assert hasattr(result, "__iter__")
    assert not hasattr(result, "__aiter__")

    chunks: list[bytes] = list(result)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, bytes)


def test_google_genai_sync_translate_produces_valid_sse():
    """
    Test that GoogleGenAIAdapter.sync_translate_completion_output_params_streaming
    produces valid SSE-formatted bytes with 'data:' prefix.
    """
    adapter = GoogleGenAIAdapter()
    result = adapter.sync_translate_completion_output_params_streaming(
        completion_stream=MockGoogleGenAIDictStream(),
    )

    assert result is not None
    chunks = list(result)
    assert len(chunks) > 0

    for chunk in chunks:
        chunk_str = chunk.decode("utf-8")
        # Google GenAI SSE format: "data: {...}\n\n" for dict chunks
        assert chunk_str.startswith("data: ")
        assert chunk_str.endswith("\n\n")


def test_google_genai_sync_translate_sse_contains_candidates():
    """
    Test that the Google GenAI sync streaming output contains
    properly formatted candidate data in SSE events.
    """
    adapter = GoogleGenAIAdapter()
    result = adapter.sync_translate_completion_output_params_streaming(
        completion_stream=MockGoogleGenAIDictStream(),
    )

    assert result is not None
    found_candidates = False
    for chunk in result:
        chunk_str = chunk.decode("utf-8")
        if chunk_str.startswith("data: "):
            json_str = chunk_str[len("data: ") :].strip()
            if json_str:
                data = json.loads(json_str)
                if "candidates" in data:
                    found_candidates = True
                    candidate = data["candidates"][0]
                    assert "content" in candidate
                    assert "parts" in candidate["content"]

    assert found_candidates, "Expected at least one SSE chunk with 'candidates' data"
