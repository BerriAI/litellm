import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.anthropic_sse import user_prompt
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_SERVED_MODEL: Final = "Qwen/Qwen2.5-0.5B-Instruct"
_TEXT: Final = "Hello"
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_PROMPT_TOKENS: Final = 5
_COMPLETION_TOKENS: Final = 3
_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_FINGERPRINT: Final = "fp-vllm-wire"


@dataclass(frozen=True, slots=True)
class Case:
    target: str
    request: Mapping[str, JsonValue]
    relayed_stream_options: JsonValue | None
    client_sees_usage: bool


_CASES: Final = {
    "chat_non_stream": Case("/v1/chat/completions", {}, None, True),
    "chat_stream": Case("/v1/chat/completions", {"stream": True}, {"include_usage": True}, False),
    "chat_stream_usage_requested": Case(
        "/v1/chat/completions",
        {"stream": True, "stream_options": {"include_usage": True}},
        {"include_usage": True},
        True,
    ),
    "completions_non_stream": Case("/v1/completions", {}, None, True),
    "completions_stream": Case("/v1/completions", {"stream": True}, None, True),
}


def _data_frame(frame: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def _vllm_stream(identity: str, head: Mapping[str, JsonValue], content: JsonValue, finish: JsonValue) -> Reply:
    chunk: Final = {"id": identity, "created": 1, "model": _SERVED_MODEL, **head}
    usage_frame: Final = _data_frame({**chunk, "choices": [], "usage": _USAGE, "system_fingerprint": _FINGERPRINT})
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _data_frame({**chunk, "choices": [content]}),
            _data_frame({**chunk, "choices": [finish]}),
            usage_frame + b"data: [DONE]\n\n",
        ),
    )


def _chat_reply(identity: str, body: Mapping[str, object]) -> Reply:
    if body.get("stream") is not True:
        return chat_reply(identity, _SERVED_MODEL, _TEXT, stream=False)
    return _vllm_stream(
        identity,
        {"object": "chat.completion.chunk"},
        {"index": 0, "delta": {"content": _TEXT}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    )


def _completion_reply(identity: str, body: Mapping[str, object]) -> Reply:
    if body.get("stream") is True:
        return _vllm_stream(
            identity,
            {"object": "text_completion"},
            {"index": 0, "text": _TEXT, "finish_reason": None},
            {"index": 0, "text": "", "finish_reason": "stop"},
        )
    completion: Final = {
        "id": identity,
        "object": "text_completion",
        "created": 1,
        "model": _SERVED_MODEL,
        "choices": [{"index": 0, "text": _TEXT, "finish_reason": "stop"}],
        "usage": _USAGE,
    }
    return Reply(body=json.dumps(completion).encode())


def _vllm_upstream(identity: str, marker: str, case: Case) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", case.target), request
        body: Final = JSON_OBJECT.validate_json(request.body)
        if case.target.endswith("/chat/completions"):
            assert user_prompt(body) == marker, body
            return _chat_reply(identity, body)
        assert body.get("prompt") == marker, body
        return _completion_reply(identity, body)

    return answering_model_discovery(respond)


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            "SELECT status, call_type, model, model_group, custom_llm_provider, prompt_tokens, completion_tokens, spend "
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )


@pytest.mark.parametrize("name", list(_CASES))
def test_vllm_passthrough_to_a_router_model_writes_one_spend_row_with_the_upstream_usage(
    gateway: Gateway, name: str
) -> None:
    case: Final = _CASES[name]
    marker: Final = "vllm-spend-" + uuid.uuid4().hex
    identity: Final = f"cmpl-{marker}"
    prompt: Final[Mapping[str, JsonValue]] = (
        {"messages": [{"role": "user", "content": marker}]} if "chat" in case.target else {"prompt": marker}
    )
    with wire_server(_vllm_upstream(identity, marker, case)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"hosted_vllm/{_SERVED_MODEL}",
            api_base=wire.url + "/v1",
            input_cost_per_token=_INPUT_COST,
            output_cost_per_token=_OUTPUT_COST,
        )
        response: Final = gateway.request(
            "POST", f"/vllm{case.target[len('/v1') :]}", {"model": model, **prompt, **case.request}
        )
        assert response.status_code == 200, response.text
        assert identity in response.text, response.text
        assert ('"usage"' in response.text) is case.client_sees_usage, response.text
        if case.request.get("stream") is True:
            assert _FINGERPRINT in response.text, response.text

        posted: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert tuple(request.target for request in posted) == (case.target,)
        assert JSON_OBJECT.validate_json(posted[0].body).get("stream_options") == case.relayed_stream_options

        rows: Final = _spend_rows(identity)
        assert len(rows) == 1, rows
        assert rows[0] == {
            "status": "success",
            "call_type": "allm_passthrough_route",
            "model": f"hosted_vllm/{_SERVED_MODEL}",
            "model_group": model,
            "custom_llm_provider": "hosted_vllm",
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "spend": pytest.approx(_PROMPT_TOKENS * _INPUT_COST + _COMPLETION_TOKENS * _OUTPUT_COST),
        }, rows
