"""
Tests for Anthropic batch JSONL translation.
Ref: https://github.com/BerriAI/litellm/issues/27944
"""
import pytest
from litellm.batches.batch_utils import _translate_anthropic_batch_item_to_openai


def test_translate_succeeded_item():
    item = {
        "custom_id": "req-1",
        "result": {
            "type": "succeeded",
            "message": {
                "model": "claude-opus-4-7",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        },
    }
    result = _translate_anthropic_batch_item_to_openai(item)
    assert result["response"]["status_code"] == 200
    body = result["response"]["body"]
    assert body["model"] == "claude-opus-4-7"
    assert body["usage"]["prompt_tokens"] == 100
    assert body["usage"]["completion_tokens"] == 50
    assert body["usage"]["total_tokens"] == 150
    assert result["custom_id"] == "req-1"


def test_translate_failed_item_passes_through():
    item = {
        "custom_id": "req-2",
        "result": {"type": "errored", "error": {"type": "server_error"}},
    }
    result = _translate_anthropic_batch_item_to_openai(item)
    # failed items pass through unchanged; no response.status_code == 200
    assert result == item


def test_translate_zero_usage():
    item = {
        "custom_id": "req-3",
        "result": {
            "type": "succeeded",
            "message": {"model": "claude-haiku-4-5", "usage": {}},
        },
    }
    result = _translate_anthropic_batch_item_to_openai(item)
    assert result["response"]["body"]["usage"]["prompt_tokens"] == 0
    assert result["response"]["body"]["usage"]["completion_tokens"] == 0
    assert result["response"]["body"]["usage"]["total_tokens"] == 0
