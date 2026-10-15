"""Tests for litellm/llms/a2a/chat/transformation.py response transform."""

import uuid
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator
from litellm.llms.a2a.chat.transformation import A2AConfig
from litellm.llms.a2a.common_utils import A2AError
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


def test_get_model_response_iterator_parses_the_agent_stream():
    iterator = A2AConfig().get_model_response_iterator(
        streaming_response=iter(
            [
                '{"jsonrpc":"2.0","id":"1","result":{"kind":"task","status":{"state":"completed"},'
                '"artifacts":[{"parts":[{"kind":"text","text":"7"}]}]}}'
            ]
        ),
        sync_stream=True,
    )

    chunk = next(iterator)

    assert isinstance(iterator, A2AModelResponseIterator)
    assert chunk["text"] == "7"
    assert chunk["finish_reason"] == "stop"


def test_resolve_agent_config_from_registry_returns_the_explicit_headers_object_untouched():
    headers: dict[str, object] = {"X-Test": "value"}
    optional_params: dict[str, object] = {"stream": True}

    resolved = A2AConfig.resolve_agent_config_from_registry(
        agent_name="test-agent",
        api_base="http://explicit.example",
        api_key="explicit-key",
        headers=headers,
        optional_params=optional_params,
    )

    assert resolved[2] is headers
    assert resolved[:2] == ("http://explicit.example", "explicit-key")
    assert optional_params == {"stream": True}


def _transform(payload: object, headers: dict[str, str] | None = None) -> ModelResponse:
    return A2AConfig().transform_response(
        model="a2a/test-agent",
        raw_response=httpx.Response(200, json=payload, headers=headers),
        model_response=ModelResponse(),
        logging_obj=None,
        request_data={},
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


@pytest.mark.parametrize(
    ("result", "expected_text"),
    [
        ({"kind": "message", "parts": [{"kind": "text", "text": "direct"}]}, "direct"),
        ({"message": {"parts": [{"kind": "text", "text": "nested"}]}}, "nested"),
        ({"kind": "task", "status": {"message": {"parts": [{"kind": "text", "text": "status"}]}}}, "status"),
        ({"kind": "task", "artifacts": [{"parts": [{"kind": "text", "text": "artifact"}]}]}, "artifact"),
        ({"kind": "task", "status": {"state": "submitted"}}, ""),
        (None, ""),
        ("plain text", ""),
        ([{"kind": "text", "text": "not a result object"}], ""),
    ],
)
def test_transform_response_extracts_the_text_of_each_result_shape(result: object, expected_text: str):
    response = _transform({"jsonrpc": "2.0", "id": "resp-1", "result": result, "extra": None})

    assert response.choices[0].message.content == expected_text
    assert response.choices[0].finish_reason == "stop"
    assert response.id == "resp-1"
    assert response.model == "a2a/test-agent"


@pytest.mark.parametrize("payload", [{}, {"jsonrpc": "2.0", "result": {"parts": [{"kind": "text", "text": "4"}]}}])
def test_transform_response_generates_an_id_when_the_agent_sends_none(payload: dict[str, object]):
    response = _transform(payload)

    assert str(uuid.UUID(response.id)) == response.id


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        ({"code": -32600, "message": "Invalid Request"}, "A2A error: Invalid Request"),
        ({"code": -32600}, "A2A error: Unknown error"),
        ({}, "A2A error: Unknown error"),
        ({"message": None}, "A2A error: None"),
        ({"message": 429}, "A2A error: 429"),
        ({"message": {"detail": "nested"}}, "A2A error: {'detail': 'nested'}"),
    ],
)
def test_transform_response_raises_the_agent_error_with_its_message(error: dict[str, object], expected_message: str):
    payload = {"jsonrpc": "2.0", "id": "1", "error": error, "result": {"parts": [{"kind": "text", "text": "4"}]}}

    with pytest.raises(A2AError) as exc_info:
        _transform(payload, headers={"x-agent": "billing"})

    assert exc_info.value.message == expected_message
    assert exc_info.value.status_code == 200
    assert exc_info.value.headers["x-agent"] == "billing"


@pytest.mark.parametrize("error", [None, "429 Too Many Requests", 429, False, ["Request timed out"]])
def test_transform_response_rejects_an_error_member_that_is_not_an_object(error: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform({"jsonrpc": "2.0", "id": "1", "error": error, "result": {"parts": []}})

    assert [(detail["type"], detail["loc"]) for detail in exc_info.value.errors()] == [("dict_type", ())]
    assert "input_value" not in str(exc_info.value)


@pytest.mark.parametrize("payload", [[], [{"result": {"parts": []}}], ["rate limit"], "upstream unavailable"])
def test_transform_response_rejects_a_body_that_is_not_an_object(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert [(detail["type"], detail["loc"]) for detail in exc_info.value.errors()] == [("dict_type", ())]
    assert "input_value" not in str(exc_info.value)


def test_transform_response_reports_an_undecodable_body_as_an_agent_error():
    with pytest.raises(A2AError) as exc_info:
        A2AConfig().transform_response(
            model="a2a/test-agent",
            raw_response=httpx.Response(502, content=b"<html>Bad Gateway</html>"),
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[{"role": "user", "content": "hi there agent"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.message.startswith("Failed to parse A2A response: ")
