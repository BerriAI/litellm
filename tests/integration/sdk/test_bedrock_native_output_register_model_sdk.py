import json
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import ModelResponse, Router
from litellm.utils import _invalidate_model_cost_lowercase_map  # pyright: ignore[reportPrivateUsage]  # required cache invalidation

_OPUS_4_8: Final = "us.anthropic.claude-opus-4-8"
_JSON_TOOL: Final = "json_tool_call"
_MARKER: Final = re.compile(r"probe ([0-9a-f]{32})")
_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, object]])
_OPTIONS: Final[dict[str, JsonValue]] = {
    "max_tokens": 2048,
    "thinking": {"type": "adaptive"},
    "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": _SCHEMA}},
    "num_retries": 0,
}


def _native(body: Mapping[str, JsonValue], invoke: bool) -> bool:
    output_config: Final = body.get("output_config" if invoke else "outputConfig")
    return isinstance(output_config, dict) and ("format" if invoke else "textFormat") in output_config


def _invoke_reply(answer: dict[str, JsonValue], native: bool) -> Reply:
    block: Final[dict[str, JsonValue]] = (
        {"type": "text", "text": json.dumps(answer)}
        if native
        else {"type": "tool_use", "id": "toolu_scripted", "name": _JSON_TOOL, "input": answer}
    )
    message: Final[dict[str, JsonValue]] = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": "claude-scripted",
        "content": [block],
        "stop_reason": "end_turn" if native else "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    return Reply(body=json.dumps(message).encode())


def _converse_reply(answer: dict[str, JsonValue], native: bool) -> Reply:
    block: Final[dict[str, JsonValue]] = (
        {"text": json.dumps(answer)}
        if native
        else {"toolUse": {"toolUseId": "tooluse_scripted", "name": _JSON_TOOL, "input": answer}}
    )
    converse: Final[dict[str, JsonValue]] = {
        "output": {"message": {"role": "assistant", "content": [block]}},
        "stopReason": "end_turn" if native else "tool_use",
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
        "metrics": {"latencyMs": 1},
    }
    return Reply(body=json.dumps(converse).encode())


def _bedrock(request: Request) -> Reply:
    body: Final = _JSON.validate_json(request.body)
    found: Final = _MARKER.search(json.dumps(body))
    assert found is not None, body
    invoke: Final = "/invoke" in request.target
    reply: Final[Callable[[dict[str, JsonValue], bool], Reply]] = _invoke_reply if invoke else _converse_reply
    return reply({"city": found.group(1)}, _native(body, invoke))


@pytest.fixture
def opus_4_8_opted_in_for_the_process(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "model_cost", _COST_MAP.validate_python(litellm.model_cost))  # pyright: ignore[reportUnknownMemberType]  # the cost map is untyped
    litellm.register_model(
        {_OPUS_4_8: {"supports_bedrock_native_structured_output": True}}, persist_across_reloads=False
    )
    yield
    monkeypatch.undo()
    _invalidate_model_cost_lowercase_map()


def _model(route: str) -> str:
    return f"bedrock/{route}/{_OPUS_4_8}"


def _connection(wire: Wire) -> dict[str, JsonValue]:
    return {"api_base": wire.url, "api_key": "synthetic-bedrock-bearer", "aws_region_name": "us-east-1"}


def _sent_native(wire: Wire, route: str, call: Callable[[list[dict[str, str]]], object]) -> bool:
    marker_hex: Final = uuid.uuid4().hex
    response: Final = call([{"role": "user", "content": f"Name a city. probe {marker_hex}"}])
    assert isinstance(response, ModelResponse), response
    assert json.loads(str(response.choices[0].message.content)) == {"city": marker_hex}, response  # pyright: ignore[reportAttributeAccessIssue]  # choices also types streaming chunks
    (request,) = wire.drain()
    return _native(_JSON.validate_json(request.body), route == "invoke")


@pytest.mark.parametrize("route", ["converse", "invoke"])
@pytest.mark.usefixtures("opus_4_8_opted_in_for_the_process")
def test_a_register_model_native_output_opt_in_reaches_every_call_but_a_deployment_that_opts_out(route: str) -> None:
    with wire_server(_bedrock) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "opted-out",
                    "litellm_params": {"model": _model(route), **_connection(wire)},
                    "model_info": {"supports_bedrock_native_structured_output": False},
                },
                {"model_name": "inherits", "litellm_params": {"model": _model(route), **_connection(wire)}},
            ],
            num_retries=0,
        )
        assert _sent_native(
            wire,
            route,
            lambda messages: router.completion(model="inherits", messages=messages, **_OPTIONS),  # pyright: ignore[reportUnknownMemberType]  # Router.completion leaves **kwargs untyped
        )
        assert not _sent_native(
            wire,
            route,
            lambda messages: router.completion(model="opted-out", messages=messages, **_OPTIONS),  # pyright: ignore[reportUnknownMemberType]  # Router.completion leaves **kwargs untyped
        )
        assert _sent_native(
            wire,
            route,
            lambda messages: litellm.completion(_model(route), messages, **_connection(wire), **_OPTIONS),
        )
