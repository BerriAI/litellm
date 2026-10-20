import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_TOKEN: Final = "synthetic-bedrock-bearer"
_CONTEXT_BETA: Final = "context-1m-2025-08-07"
_THINKING_BETA: Final = "interleaved-thinking-2025-05-14"
_REPLY_TEXT: Final = "beta header control"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_INVOKE_REPLY: Final = json.dumps(
    {
        "id": "msg_beta_header",
        "type": "message",
        "role": "assistant",
        "model": _BACKEND,
        "content": [{"type": "text", "text": _REPLY_TEXT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }
).encode()
_CONVERSE_REPLY: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": _REPLY_TEXT}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
        "metrics": {"latencyMs": 1},
    }
).encode()


def _invoke_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == f"/model/{_BACKEND}/invoke", request.target
    assert request.headers["authorization"] == f"Bearer {_TOKEN}"
    return Reply(body=_INVOKE_REPLY)


def _converse_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/converse"), request.target
    assert request.headers["authorization"] == f"Bearer {_TOKEN}"
    return Reply(body=_CONVERSE_REPLY)


def _sent_betas(outbound: dict[str, JsonValue]) -> tuple[str, ...] | None:
    betas: Final = outbound.get("anthropic_beta")
    if betas is None:
        return None
    assert isinstance(betas, list), outbound
    return tuple(sorted(str(beta) for beta in betas))


def _beta_header(value: str | None) -> dict[str, str]:
    return {} if value is None else {"anthropic-beta": value}


@pytest.mark.parametrize(
    ("header", "expected"),
    (
        pytest.param(json.dumps([_CONTEXT_BETA, _THINKING_BETA]), (_CONTEXT_BETA, _THINKING_BETA), id="json-array"),
        pytest.param(f'[ " {_CONTEXT_BETA} " ]', (_CONTEXT_BETA,), id="json-array-padded-entry"),
        pytest.param("[1, 2]", ("1", "2"), id="json-array-of-integers"),
        pytest.param("[null]", ("None",), id="json-array-holding-null"),
        pytest.param(f"[{_CONTEXT_BETA}]", (f"[{_CONTEXT_BETA}]",), id="bracketed-but-not-json"),
        pytest.param(f"{_CONTEXT_BETA}, {_THINKING_BETA}", (_CONTEXT_BETA, _THINKING_BETA), id="comma-separated"),
        pytest.param("[]", None, id="empty-json-array"),
        pytest.param(None, None, id="header-absent"),
    ),
)
def test_bedrock_invoke_chat_sends_the_client_anthropic_beta_header_as_a_body_list(
    gateway: Gateway,
    header: str | None,
    expected: tuple[str, ...] | None,
) -> None:
    with wire_server(_invoke_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{_BACKEND}",
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "synthetic beta request"}], "max_tokens": 16},
            headers=_beta_header(header),
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"][0]["message"]["content"] == _REPLY_TEXT, response.text
        assert payload["usage"]["prompt_tokens"] == 12 and payload["usage"]["completion_tokens"] == 5, response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        outbound: Final = _JSON_OBJECT.validate_json(requests[0].body)
        assert outbound["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "synthetic beta request"}]}
        ], outbound
        assert _sent_betas(outbound) == (None if expected is None else tuple(sorted(expected))), outbound


def test_bedrock_invoke_messages_sends_a_comma_separated_anthropic_beta_header_as_a_body_list(
    gateway: Gateway,
) -> None:
    with wire_server(_invoke_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{_BACKEND}",
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "messages": [{"role": "user", "content": "synthetic beta request"}], "max_tokens": 16},
            headers=_beta_header(f"{_CONTEXT_BETA}, {_THINKING_BETA}"),
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["content"] == [{"type": "text", "text": _REPLY_TEXT}], response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        assert _sent_betas(_JSON_OBJECT.validate_json(requests[0].body)) == (_CONTEXT_BETA,), requests[0].body


def test_bedrock_converse_chat_sends_a_comma_separated_anthropic_beta_header_as_additional_model_fields(
    gateway: Gateway,
) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{_BACKEND}",
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "synthetic beta request"}], "max_tokens": 16},
            headers=_beta_header(f"{_CONTEXT_BETA}, {_THINKING_BETA}"),
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"][0]["message"]["content"] == _REPLY_TEXT, response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        outbound: Final = _JSON_OBJECT.validate_json(requests[0].body)
        assert outbound["additionalModelRequestFields"] == {"anthropic_beta": [_CONTEXT_BETA]}, outbound
