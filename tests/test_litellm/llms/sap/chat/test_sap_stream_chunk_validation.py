"""
Tests for SAP orchestration stream chunk normalization (_StreamParser._validate_chunk)
and the descriptive error raised when no orchestration deployment exists.

Regression tests for:
- TypeError: 'MockValSer' object is not an instance of 'SchemaSerializer'
  raised from nested model_dump() when the raw openai-SDK usage object
  (a deferred-build pydantic model) was attached to ModelResponseStream.
- IndexError: list index out of range raised from deployment_url when the
  configured resource group contains no orchestration deployment.
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.sap.chat.handler import GenAIHubOrchestrationError, _StreamParser
from litellm.types.utils import ModelResponseStream, Usage


def _final_chunk_payload() -> dict:
    """OpenAI-shaped final chunk as sent by the SAP orchestration service:
    every choice carries an empty `logprobs` ({}) and the last chunk carries usage."""
    return {
        "id": "chatcmpl-sap-final",
        "object": "chat.completion.chunk",
        "created": 1761319270,
        "model": "anthropic--claude-4.7-opus",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "logprobs": {},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "completion_tokens": 206,
            "prompt_tokens": 62322,
            "total_tokens": 62528,
        },
    }


def test_validate_chunk_drops_empty_logprobs():
    chunk = _StreamParser._validate_chunk(_final_chunk_payload())
    assert chunk.choices[0].logprobs is None


def test_validate_chunk_preserves_real_logprobs():
    payload = _final_chunk_payload()
    payload["choices"][0]["logprobs"] = {
        "content": [{"token": "Hello", "logprob": -0.1, "bytes": None, "top_logprobs": []}]
    }
    chunk = _StreamParser._validate_chunk(payload)
    assert chunk.choices[0].logprobs is not None
    assert chunk.choices[0].logprobs.content[0].token == "Hello"


def test_validate_chunk_converts_usage_to_litellm_usage():
    chunk = _StreamParser._validate_chunk(_final_chunk_payload())
    assert isinstance(chunk.usage, Usage)
    assert chunk.usage.completion_tokens == 206
    assert chunk.usage.prompt_tokens == 62322
    assert chunk.usage.total_tokens == 62528


def test_validate_chunk_without_usage_keeps_none():
    payload = _final_chunk_payload()
    del payload["usage"]
    chunk = _StreamParser._validate_chunk(payload)
    assert chunk.usage is None


def test_validated_usage_survives_nested_model_dump():
    """The original crash: the raw openai-SDK usage object attached to a
    ModelResponseStream made model_dump() raise
    "TypeError: 'MockValSer' object is not an instance of 'SchemaSerializer'"."""
    chunk = _StreamParser._validate_chunk(_final_chunk_payload())

    model_response = ModelResponseStream()
    setattr(model_response, "usage", chunk.usage)

    dumped = model_response.model_dump()  # must not raise
    assert dumped["usage"]["total_tokens"] == 62528


def test_deployment_url_raises_404_when_no_orchestration_deployment():
    from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

    config = GenAIHubOrchestrationConfig()
    config.token_creator = lambda: "Bearer FAKE_TOKEN"
    config._base_url = "https://api.ai.mock-sap.com/v2"
    config._resource_group = "fake-group"

    mock_client = MagicMock()
    mock_client.get.return_value.json.return_value = {"resources": []}

    with patch("litellm.module_level_client", mock_client):
        with pytest.raises(GenAIHubOrchestrationError) as exc_info:
            _ = config.deployment_url

    assert exc_info.value.status_code == 404
    assert "fake-group" in exc_info.value.message
