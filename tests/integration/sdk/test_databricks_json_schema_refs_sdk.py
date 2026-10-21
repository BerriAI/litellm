import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, Field, JsonValue

import litellm

_QWEN_BACKEND: Final = "databricks-qwen35-122b-a10b"
_CLAUDE_BACKEND: Final = "databricks-claude-haiku-4-5"
_API_KEY: Final = "synthetic-databricks-key"
_ORDER_JSON: Final = '{"buyer":{"name":"Ada"},"seller":{"name":"Bo"}}'
_MESSAGES: Final[tuple[Mapping[str, str], ...]] = ({"role": "user", "content": "Return the order as JSON."},)


class Person(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str


class Order(BaseModel):
    model_config = ConfigDict(frozen=True)
    buyer: Person
    seller: Person


class _StrictObject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    additional_properties: bool | None = Field(default=None, alias="additionalProperties")


class _OrderSchema(_StrictObject):
    defs: Mapping[str, _StrictObject] = Field(alias="$defs")


class _JsonSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    name: str
    strict: bool | None = None
    json_schema: dict[str, JsonValue] = Field(alias="schema")


class _ResponseFormat(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    type: str
    json_schema: _JsonSchema


class _Function(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    name: str
    parameters: dict[str, JsonValue]


class _Tool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    function: _Function


class _Outbound(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    model: str
    response_format: _ResponseFormat | None = None
    tools: tuple[_Tool, ...] = ()


class _CallerMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    content: str


class _CallerChoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    message: _CallerMessage


class _CallerCompletion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    choices: tuple[_CallerChoice, ...]


def _completion(backend: str) -> Reply:
    claude: Final = "claude" in backend
    tool_call: Final[dict[str, JsonValue]] = {
        "id": "json-tool-call",
        "type": "function",
        "function": {"name": "json_tool_call", "arguments": _ORDER_JSON},
    }
    message: Final[dict[str, JsonValue]] = (
        {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        if claude
        else {"role": "assistant", "content": _ORDER_JSON}
    )
    body: Final[dict[str, JsonValue]] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": backend,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if claude else "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16},
    }
    return Reply(body=json.dumps(body).encode())


def _refs(node: JsonValue) -> Iterator[str]:
    if isinstance(node, list):
        for item in node:
            yield from _refs(item)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str):
            yield value
        else:
            yield from _refs(value)


def _sent_schema(backend: str, received: tuple[Request, ...]) -> dict[str, JsonValue]:
    assert [(request.method, request.target) for request in received] == [("POST", "/chat/completions")]
    outbound: Final = _Outbound.model_validate_json(received[0].body)
    assert outbound.model == backend
    if "claude" in backend:
        assert outbound.response_format is None
        assert [tool.function.name for tool in outbound.tools] == ["json_tool_call"]
        return outbound.tools[0].function.parameters
    assert outbound.response_format is not None
    assert (outbound.response_format.type, outbound.response_format.json_schema.name) == ("json_schema", "Order")
    assert outbound.response_format.json_schema.strict is True
    return outbound.response_format.json_schema.json_schema


def _assert_strict_order_schema(schema: dict[str, JsonValue]) -> None:
    assert set(_refs(schema)) == {"#/$defs/Person"}, schema
    parsed: Final = _OrderSchema.model_validate(schema)
    assert (parsed.additional_properties, parsed.defs["Person"].additional_properties) == (False, False), schema


def _caller_order(response: litellm.ModelResponse) -> Order:
    return Order.model_validate_json(
        _CallerCompletion.model_validate_json(response.model_dump_json()).choices[0].message.content
    )


_BACKENDS: Final = (
    pytest.param(_QWEN_BACKEND, id="qwen"),
    pytest.param(_CLAUDE_BACKEND, id="claude"),
)
_EXPECTED_ORDER: Final = Order(buyer=Person(name="Ada"), seller=Person(name="Bo"))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_databricks_pydantic_response_format_sends_strict_defs_refs(backend: str) -> None:
    with wire_server(lambda _request: _completion(backend)) as wire:
        response: Final = litellm.completion(
            model=f"databricks/{backend}",
            api_base=wire.url,
            api_key=_API_KEY,
            messages=[dict(message) for message in _MESSAGES],
            response_format=Order,
            num_retries=0,
        )
        received: Final = wire.drain()
    _assert_strict_order_schema(_sent_schema(backend, received))
    assert isinstance(response, litellm.ModelResponse)
    assert _caller_order(response) == _EXPECTED_ORDER


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_databricks_pydantic_response_format_sends_strict_defs_refs_async(backend: str) -> None:
    with wire_server(lambda _request: _completion(backend)) as wire:
        response: Final = await litellm.acompletion(
            model=f"databricks/{backend}",
            api_base=wire.url,
            api_key=_API_KEY,
            messages=[dict(message) for message in _MESSAGES],
            response_format=Order,
            num_retries=0,
        )
        received: Final = wire.drain()
    _assert_strict_order_schema(_sent_schema(backend, received))
    assert isinstance(response, litellm.ModelResponse)
    assert _caller_order(response) == _EXPECTED_ORDER
