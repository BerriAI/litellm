import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON: Final = TypeAdapter(dict[str, JsonValue])
_ANSWER: Final = "sampling params dropped"
_SAMPLING_SPELLINGS: Final = ("top_k", "top_p", "topK", "topP")
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}


@dataclass(frozen=True, slots=True)
class Backend:
    model: str
    path_suffix: str
    credentials: Mapping[str, JsonValue]


_BEDROCK_INVOKE: Final = Backend("bedrock/invoke/us.anthropic.claude-opus-4-8", "/invoke", _AWS)
_ANTHROPIC: Final = Backend("anthropic/claude-opus-4-8", "/v1/messages", {})
_BEDROCK_CONVERSE: Final = Backend("bedrock/converse/us.anthropic.claude-opus-4-8", "/converse", _AWS)
_NATIVE_BACKENDS: Final = (
    pytest.param(_BEDROCK_INVOKE, id="bedrock_invoke"),
    pytest.param(_ANTHROPIC, id="anthropic"),
)


def _answer(request: Request) -> Reply:
    if request.target.endswith("/converse"):
        return Reply(
            body=json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": _ANSWER}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
        )
    return Reply(
        body=json.dumps(
            {
                "id": "msg_dropped_sampling_params",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": _ANSWER}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
        ).encode()
    )


def _messages(gateway: Gateway, model: str, **sampling: JsonValue) -> tuple[int, str]:
    response: Final = gateway.request(
        "POST",
        "/v1/messages",
        {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}], **sampling},
    )
    return response.status_code, response.text


def _answered(status: int, text: str) -> None:
    assert status == 200, text
    content: Final = list_value(_JSON.validate_json(text)["content"])
    assert object_value(content[0])["text"] == _ANSWER, text


def _sent_without_sampling_params(wire: Wire, backend: Backend) -> None:
    (request,) = wire.drain()
    assert request.target.endswith(backend.path_suffix), request.target
    sent: Final = request.body.decode()
    assert not any(spelling in sent for spelling in _SAMPLING_SPELLINGS), sent


@pytest.mark.parametrize(
    "backend", (*_NATIVE_BACKENDS, pytest.param(_BEDROCK_CONVERSE, id="bedrock_converse_chat_bridge"))
)
def test_messages_drops_the_sampling_params_a_deployment_lists_on_a_model_without_them(
    gateway: Gateway, backend: Backend
) -> None:
    with wire_server(_answer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=backend.model,
            api_base=wire.url,
            additional_drop_params=["top_k", "top_p"],
            **backend.credentials,
        )
        _answered(*_messages(gateway, model, top_k=5, top_p=0.9))
        _sent_without_sampling_params(wire, backend)


@pytest.mark.parametrize("backend", _NATIVE_BACKENDS)
def test_messages_rejects_a_sampling_param_the_deployment_does_not_list_before_calling_the_provider(
    gateway: Gateway, backend: Backend
) -> None:
    with wire_server(_answer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=backend.model,
            api_base=wire.url,
            additional_drop_params=["top_k"],
            **backend.credentials,
        )
        rejected_status, rejected_text = _messages(gateway, model, top_k=5, top_p=0.9)
        assert rejected_status == 400, rejected_text
        assert "does not support top_p=0.9" in rejected_text, rejected_text
        _answered(*_messages(gateway, model, top_k=5))
        _sent_without_sampling_params(wire, backend)
