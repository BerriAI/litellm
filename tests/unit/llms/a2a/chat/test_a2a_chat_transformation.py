"""Tests for litellm/llms/a2a/chat/transformation.py response transform."""

from unittest.mock import MagicMock

import pytest

from litellm.llms.a2a.chat.transformation import A2AConfig
from litellm.types.utils import ModelResponse


def _raw_response(text: str) -> MagicMock:
    raw = MagicMock()
    raw.status_code = 200
    raw.headers = {}
    raw.json.return_value = {
        "jsonrpc": "2.0",
        "id": "resp-1",
        "result": {
            "kind": "message",
            "parts": [{"kind": "text", "text": text}],
        },
    }
    return raw


def test_transform_response_sets_usage():
    """Regression: A2AConfig.transform_response must populate usage so per-token
    pricing computes real cost and callers don't get usage 0/0/0."""
    result = A2AConfig().transform_response(
        model="a2a/test-agent",
        raw_response=_raw_response("hello from the agent"),
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == (result.usage.prompt_tokens + result.usage.completion_tokens)


def test_transform_request_asks_the_agent_for_a_blocking_send():
    """Chat completions need the final answer in one response. Microsoft Foundry agents default to a
    non-blocking send that returns a submitted task, so the request must opt into blocking."""
    request = A2AConfig().transform_request(
        model="a2a/test-agent",
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["method"] == "message/send"
    assert request["params"]["configuration"] == {"blocking": True}


def test_transform_request_streams_without_a_send_configuration():
    request = A2AConfig().transform_request(
        model="a2a/test-agent",
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params={"stream": True},
        litellm_params={},
        headers={},
    )

    assert request["method"] == "message/stream"
    assert "configuration" not in request["params"]


@pytest.mark.parametrize("optional_params", [{}, {"stream": True}])
def test_transform_request_tags_the_message_with_its_kind(optional_params: dict):
    """A2A 0.3 messages carry a `kind` discriminator; Microsoft Foundry rejects a message without it as
    missing a required property, so both send methods must tag the message."""
    request = A2AConfig().transform_request(
        model="a2a/test-agent",
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert request["params"]["message"]["kind"] == "message"
