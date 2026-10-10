import json
from collections.abc import Callable
from typing import Final

import litellm
import pytest
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-4-6"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_COUNT_TARGET: Final = (
    f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/count-tokens:rawPredict"
)
_PEER_COUNT: Final = 4242
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final[list[dict[str, str]]] = [{"role": "user", "content": "Count this message"}]
_SYSTEM: Final = "You are a terse assistant that answers in one sentence"
_TOOLS: Final[list[dict[str, JsonValue]]] = [
    {
        "name": "get_weather",
        "description": "Look up the current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]


def _peer(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/_oauth/token":
            token: Final = {"access_token": "scripted-token", "token_type": "Bearer", "expires_in": 3600}
            return Reply(body=json.dumps(token).encode())
        if status == 200:
            return Reply(body=json.dumps({"input_tokens": _PEER_COUNT}).encode())
        rejection: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": "scripted"}}
        return Reply(status=status, body=json.dumps(rejection).encode())

    return respond


def _count_requests(requests: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in requests if "count-tokens" in request.target)


@pytest.fixture
def vertex_environment(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def configure(token_url: str) -> None:
        monkeypatch.setenv("VERTEXAI_PROJECT", _PROJECT)
        monkeypatch.setenv("VERTEXAI_LOCATION", _LOCATION)
        monkeypatch.setenv("VERTEXAI_CREDENTIALS", service_account_json(_PROJECT, token_url))

    return configure


async def test_acount_tokens_forwards_system_and_tools_to_the_partner_peer(
    vertex_environment: Callable[[str], None],
) -> None:
    with wire_server(_peer(200)) as wire:
        vertex_environment(wire.url)
        counted: Final = await litellm.acount_tokens(
            model=f"vertex_ai/{_BACKEND}", messages=_MESSAGES, tools=_TOOLS, system=_SYSTEM, api_base=wire.url
        )
        (sent,) = _count_requests(wire.drain())
        assert (sent.method, sent.target, sent.headers["authorization"]) == (
            "POST",
            _COUNT_TARGET,
            "Bearer scripted-token",
        )
        assert _JSON_OBJECT.validate_json(sent.body) == {
            "model": _BACKEND,
            "messages": _MESSAGES,
            "system": _SYSTEM,
            "tools": _TOOLS,
        }
        assert (counted.total_tokens, counted.tokenizer_type) == (_PEER_COUNT, "vertex_ai_partner_models"), counted


async def test_acount_tokens_falls_back_to_the_local_tokenizer_when_the_peer_rejects(
    vertex_environment: Callable[[str], None],
) -> None:
    with wire_server(_peer(400)) as wire:
        vertex_environment(wire.url)
        counted: Final = await litellm.acount_tokens(
            model=f"vertex_ai/{_BACKEND}", messages=_MESSAGES, tools=_TOOLS, system=_SYSTEM, api_base=wire.url
        )
        assert len(_count_requests(wire.drain())) == 1
        assert counted.tokenizer_type == "local_tokenizer", counted
        assert counted.total_tokens > 0 and counted.total_tokens != _PEER_COUNT, counted
