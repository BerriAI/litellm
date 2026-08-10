"""Tests for litellm/llms/a2a/chat/transformation.py response transform."""

from unittest.mock import MagicMock

from litellm.llms.a2a.chat.transformation import A2AConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
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


def test_stream_param_never_reaches_the_jsonrpc_body():
    """Regression: A2A servers validate JSON-RPC strictly and reject a top-level
    `stream` field with -32600 "Invalid Request. Extra fields: {'stream'}", answering
    application/json instead of an SSE stream; the caller then sees an empty stream
    with no error. Streaming is carried by the message/stream method alone."""
    config = A2AConfig()
    request_body = config.transform_request(
        model="a2a/test-agent",
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params={"stream": True},
        litellm_params={},
        headers={},
    )

    body = BaseLLMHTTPHandler()._add_stream_param_to_request_body(
        data=request_body,
        provider_config=config,
        fake_stream=False,
    )

    assert body["method"] == "message/stream"
    assert "stream" not in body
