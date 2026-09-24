"""
Tests for AnthropicFilesConfig.transform_file_content_response

Batch results requests (URLs ending in /results) are translated from Anthropic
result JSONL into OpenAI batch output JSONL; regular file content passes through.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig


@pytest.fixture
def config():
    return AnthropicFilesConfig()


def _results_line():
    return json.dumps(
        {
            "custom_id": "req-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        }
    )


def test_results_response_is_translated_to_openai_jsonl(config):
    raw = httpx.Response(
        status_code=200,
        content=_results_line().encode("utf-8"),
        request=httpx.Request("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_abc/results"),
    )
    result = config.transform_file_content_response(raw_response=raw, logging_obj=MagicMock(), litellm_params={})
    lines = [line for line in result.response.content.decode("utf-8").split("\n") if line]
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["custom_id"] == "req-1"
    assert out["response"]["status_code"] == 200
    assert out["response"]["body"]["object"] == "chat.completion"
    assert out["response"]["body"]["usage"]["prompt_tokens"] == 3


def test_non_batch_file_content_passes_through_raw(config):
    content = b"plain file bytes"
    raw = httpx.Response(
        status_code=200,
        content=content,
        request=httpx.Request("GET", "https://api.anthropic.com/v1/files/file-abc/content"),
    )
    result = config.transform_file_content_response(raw_response=raw, logging_obj=MagicMock(), litellm_params={})
    assert result.response is raw
    assert result.response.content == content
