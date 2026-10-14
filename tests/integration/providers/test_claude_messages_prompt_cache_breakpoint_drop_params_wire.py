import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

import anthropic
import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    BEDROCK_MODEL,
    EPHEMERAL,
    PROVIDER_KEY,
    anthropic_peer,
    bedrock_labels,
    bedrock_peer,
    new_marker,
)
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(120)

Route: TypeAlias = Literal["anthropic", "bedrock_invoke", "bedrock_converse"]

_NATIVE_ROUTES: Final[tuple[Route, ...]] = ("anthropic", "bedrock_invoke")
_ROUTE: Final = TypeAdapter[Route](Route)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_TARGETS: Final[Mapping[Route, str]] = {
    "anthropic": "/v1/messages",
    "bedrock_invoke": f"/model/{BEDROCK_MODEL}/invoke",
    "bedrock_converse": f"/model/{BEDROCK_MODEL}/converse",
}
_EXTRA_INPUT: Final = "system.0.prompt_cache_breakpoint: Extra inputs are not permitted"


def _strict_peer(route: Route, request: Request) -> Reply:
    if route == "bedrock_converse":
        return bedrock_peer(request)
    if b"prompt_cache_breakpoint" not in request.body:
        return anthropic_peer(request)
    match route:
        case "anthropic":
            return Reply(
                status=400,
                body=json.dumps(
                    {"type": "error", "error": {"type": "invalid_request_error", "message": _EXTRA_INPUT}}
                ).encode(),
            )
        case "bedrock_invoke":
            return Reply(
                status=400,
                headers={"x-amzn-errortype": "ValidationException"},
                body=json.dumps({"message": _EXTRA_INPUT}).encode(),
            )
        case _:
            assert_never(route)


def _litellm_params(route: Route, wire: Wire) -> Mapping[str, JsonValue]:
    match route:
        case "anthropic":
            return {"model": f"anthropic/{ANTHROPIC_MODEL}", "api_base": wire.url, "api_key": PROVIDER_KEY}
        case "bedrock_invoke":
            return {
                "model": f"bedrock/invoke/{BEDROCK_MODEL}",
                "api_key": _BEDROCK_TOKEN,
                "aws_region_name": "us-east-1",
                "api_base": wire.url,
                "aws_bedrock_runtime_endpoint": wire.url,
            }
        case "bedrock_converse":
            return {
                "model": f"bedrock/converse/{BEDROCK_MODEL}",
                "api_key": PROVIDER_KEY,
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire.url,
            }
        case _:
            assert_never(route)


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire
    route: Route
    strict: str
    dropping: str

    def only_request(self) -> Request:
        received: Final = self.wire.drain()
        assert len(received) == 1, [request.target for request in received]
        assert received[0].target == _TARGETS[self.route], received[0].target
        return received[0]

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(base_url=str(self.gateway.client.base_url), api_key=self.gateway.key, max_retries=0)


@pytest.fixture(scope="module", params=_NATIVE_ROUTES)
def rig(request: pytest.FixtureRequest) -> Iterator[_Rig]:
    with _rig(_ROUTE.validate_python(request.param)) as built:  # pyright: ignore[reportAny]  # pytest types param as Any, validated here
        yield built


@pytest.fixture(scope="module")
def converse_rig() -> Iterator[_Rig]:
    with _rig("bedrock_converse") as built:
        yield built


@contextmanager
def _rig(route: Route) -> Iterator[_Rig]:
    with (
        wire_server(lambda received: _strict_peer(route, received)) as wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
    ):
        yield _Rig(
            gateway,
            wire,
            route,
            scenario.model(model_info=None, **_litellm_params(route, wire)),
            scenario.model(model_info=None, **_litellm_params(route, wire), drop_params=True),
        )


def _breakpoint(marked: bool) -> Mapping[str, JsonValue]:
    return {"prompt_cache_breakpoint": {"mode": "explicit"}} if marked else {}


def _system(breakpoint: bool) -> Sequence[JsonValue]:
    return [
        {
            "type": "text",
            "text": "Answer in one word.",
            **_breakpoint(breakpoint),
        }
    ]


def _messages(marker: str, breakpoint: bool) -> Sequence[JsonValue]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Stable context. marker-{marker}",
                    **_breakpoint(breakpoint),
                },
                {"type": "text", "text": "What is the weather?", "cache_control": EPHEMERAL},
            ],
        }
    ]


def _tools(breakpoint: bool) -> Sequence[JsonValue]:
    return [
        {
            "name": "lookup_weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            **_breakpoint(breakpoint),
        }
    ]


def _wire_json(request: Request) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _create(
    rig: _Rig, model: str, marker: str, *, breakpoint: bool, request_drop_params: bool = False
) -> anthropic.types.Message:
    with rig.client() as client:
        return client.messages.create(
            model=model,
            max_tokens=64,
            system=_system(breakpoint),  # pyright: ignore[reportArgumentType]  # the OpenAI key is the point of the test
            messages=_messages(marker, breakpoint),  # pyright: ignore[reportArgumentType]  # the OpenAI key is the point of the test
            tools=_tools(breakpoint),  # pyright: ignore[reportArgumentType]  # the OpenAI key is the point of the test
            extra_body={"cache": {"no-cache": True}, **({"drop_params": True} if request_drop_params else {})},
        )


def test_breakpoint_reaches_the_wire_verbatim_and_the_provider_400_reaches_the_caller(rig: _Rig) -> None:
    marker: Final = new_marker()

    with pytest.raises(anthropic.BadRequestError) as raised:
        _create(rig, rig.strict, marker, breakpoint=True)

    assert raised.value.status_code == 400, raised.value
    assert "Extra inputs are not permitted" in str(raised.value), raised.value
    body: Final = _wire_json(rig.only_request())
    assert body["system"] == _system(breakpoint=True), body
    assert body["messages"] == _messages(marker, breakpoint=True), body
    assert body["tools"] == _tools(breakpoint=True), body


def test_deployment_drop_params_strips_the_breakpoint_from_every_block(rig: _Rig) -> None:
    marker: Final = new_marker()

    message: Final = _create(rig, rig.dropping, marker, breakpoint=True)

    assert message.id == f"msg_{marker}", message
    body: Final = _wire_json(rig.only_request())
    assert body["system"] == _system(breakpoint=False), body
    assert body["messages"] == _messages(marker, breakpoint=False), body
    assert body["tools"] == _tools(breakpoint=False), body


def test_request_drop_params_strips_the_breakpoint_from_every_block(rig: _Rig) -> None:
    marker: Final = new_marker()

    message: Final = _create(rig, rig.strict, marker, breakpoint=True, request_drop_params=True)

    assert message.id == f"msg_{marker}", message
    body: Final = _wire_json(rig.only_request())
    assert body["system"] == _system(breakpoint=False), body
    assert body["messages"] == _messages(marker, breakpoint=False), body
    assert body["tools"] == _tools(breakpoint=False), body


def test_drop_params_sends_a_cache_control_only_request_byte_identical(rig: _Rig) -> None:
    marker: Final = new_marker()

    _create(rig, rig.strict, marker, breakpoint=False)
    strict_body: Final = rig.only_request().body
    _create(rig, rig.dropping, marker, breakpoint=False)
    dropping_body: Final = rig.only_request().body

    assert dropping_body == strict_body


def test_converse_refuses_the_breakpoint_before_the_wire_without_drop_params(converse_rig: _Rig) -> None:
    with pytest.raises(anthropic.BadRequestError) as raised:
        _create(converse_rig, converse_rig.strict, new_marker(), breakpoint=True)

    assert raised.value.status_code == 400, raised.value
    assert "prompt_cache_breakpoint" in str(raised.value) and "drop_params" in str(raised.value), raised.value
    assert converse_rig.wire.drain() == (), "the refused request must not reach Bedrock"


def test_converse_drop_params_strips_the_breakpoint_instead_of_translating_it(converse_rig: _Rig) -> None:
    _create(converse_rig, converse_rig.dropping, new_marker(), breakpoint=True)

    request: Final = converse_rig.only_request()
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert bedrock_labels(request) == ["user:text:What is the weather?"], request.body
