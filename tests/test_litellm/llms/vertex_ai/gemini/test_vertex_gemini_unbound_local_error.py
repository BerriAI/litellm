"""
Tests for Gemini responses where content is present but "parts" is missing.

Gemini can return content: {"role": "model"} with no "parts" field when
finishReason is STOP. This is a known Gemini quirk (e.g. gemini-2.5-flash-lite
in long-running agentic tasks).

LiteLLM must handle this gracefully: return a valid response with
content=None and finish_reason="stop" rather than raising an exception.

Ref: https://github.com/BerriAI/litellm/issues/24442
"""

import pytest

import litellm
from litellm import ModelResponse
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)


def test_process_candidates_empty_content_no_parts_returns_valid_response():
    """
    When a candidate has content: {"role": "model"} with no "parts",
    _process_candidates must not raise and must produce a choice with
    content=None and finish_reason="stop".
    """
    candidates = [
        {
            "content": {
                "role": "model"
                # "parts" intentionally absent — the Gemini quirk under test
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ]
    model_response = ModelResponse()
    model_response.choices = []

    VertexGeminiConfig._process_candidates(
        _candidates=candidates,
        model_response=model_response,
        standard_optional_params={},
        cumulative_tool_call_index=0,
    )

    assert len(model_response.choices) == 1
    choice = model_response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content is None


def test_process_candidates_empty_content_no_parts_no_finish_reason():
    """
    When a candidate has content: {"role": "model"} with no "parts" and
    no finishReason, _process_candidates must not raise and must produce
    a choice with content=None.
    """
    candidates = [
        {
            "content": {"role": "model"},
            "index": 0,
        }
    ]
    model_response = ModelResponse()
    model_response.choices = []

    VertexGeminiConfig._process_candidates(
        _candidates=candidates,
        model_response=model_response,
        standard_optional_params={},
        cumulative_tool_call_index=0,
    )

    assert len(model_response.choices) == 1
    choice = model_response.choices[0]
    assert choice.message.content is None


def test_process_candidates_unbound_local_error_fix():
    """
    Regression test: _process_candidates must not raise UnboundLocalError
    when content is present but "parts" is missing.
    """
    candidates = [
        {
            "content": {
                "role": "model"
                # "parts" is missing intentionally to trigger the issue
            },
            "finishReason": "STOP",
        }
    ]
    model_response = ModelResponse()

    # Must not raise UnboundLocalError
    VertexGeminiConfig._process_candidates(
        _candidates=candidates,
        model_response=model_response,
        standard_optional_params={},
        cumulative_tool_call_index=0,
    )
