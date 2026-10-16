import json
from collections.abc import Callable, Mapping
from typing import Final
from urllib.parse import unquote

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

CLAUDE_ID: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
MANTLE_CLAUDE_ID: Final = "anthropic.claude-haiku-4-5"
TOKEN: Final = "synthetic-bedrock-bearer"
DANGEROUS_TOOL_USE_BETA: Final = "dangerous-tool-use-2026-09-03"
KEPT_BETA: Final = "context-1m-2025-08-07"
SAFEGUARDS: Final[JsonValue] = [{"type": "dangerous_tool_use"}]
NO_CACHE: Final[JsonValue] = {"no-cache": True}
INVOKE_TARGET: Final = f"/model/{CLAUDE_ID}/invoke"
MANTLE_TARGET: Final = "/anthropic/v1/messages"
REPLY: Final = json.dumps(
    {
        "id": "msg_scripted",
        "type": "message",
        "role": "assistant",
        "model": CLAUDE_ID,
        "content": [{"type": "text", "text": "scripted reply"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
).encode()
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BETAS: Final = TypeAdapter(tuple[str, ...])


def _peer(request: Request) -> Reply:
    if unquote(request.target) not in (INVOKE_TARGET, MANTLE_TARGET):
        return Reply(status=404, body=json.dumps({"error": f"unscripted target {request.target}"}).encode())
    assert request.headers["authorization"] == f"Bearer {TOKEN}", request.headers
    return Reply(body=REPLY)


def _invoke_body_betas(request: Request) -> frozenset[str]:
    return frozenset(_BETAS.validate_python(_OBJECT.validate_json(request.body).get("anthropic_beta") or ()))


def _mantle_header_betas(request: Request) -> frozenset[str]:
    return frozenset(beta.strip() for beta in request.headers.get("anthropic-beta", "").split(",") if beta.strip())


def _delivered(gateway: Gateway, wire: Wire, path: str, body: Mapping[str, JsonValue], client_betas: str) -> Request:
    response: Final = gateway.request("POST", path, body, headers={"anthropic-beta": client_betas})
    assert response.status_code == 200, response.text
    (sent,) = wire.drain()
    return sent


def _messages_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "cache": NO_CACHE,
        **extra,
    }


def _chat_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], "cache": NO_CACHE, **extra}


def _responses_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": "hi", "cache": NO_CACHE, **extra}


def test_invoke_messages_without_safeguards_sends_the_betas_of_a_request_that_never_had_the_beta(
    gateway: Gateway,
) -> None:
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{CLAUDE_ID}", api_key=TOKEN, aws_region_name="us-east-1", api_base=wire.url
        )
        with_beta: Final = _invoke_body_betas(
            _delivered(gateway, wire, "/v1/messages", _messages_body(model), f"{KEPT_BETA},{DANGEROUS_TOOL_USE_BETA}")
        )
        without_beta: Final = _invoke_body_betas(
            _delivered(gateway, wire, "/v1/messages", _messages_body(model), KEPT_BETA)
        )
    assert KEPT_BETA in without_beta, without_beta
    assert with_beta == without_beta, (with_beta, without_beta)


def test_invoke_messages_with_safeguards_keeps_the_dangerous_beta(gateway: Gateway) -> None:
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{CLAUDE_ID}", api_key=TOKEN, aws_region_name="us-east-1", api_base=wire.url
        )
        sent: Final = _invoke_body_betas(
            _delivered(
                gateway,
                wire,
                "/v1/messages",
                _messages_body(model, safeguards=SAFEGUARDS),
                f"{KEPT_BETA},{DANGEROUS_TOOL_USE_BETA}",
            )
        )
    assert {KEPT_BETA, DANGEROUS_TOOL_USE_BETA} <= sent, sent


@pytest.mark.parametrize(
    ("path", "build"),
    [("/v1/chat/completions", _chat_body), ("/v1/responses", _responses_body)],
    ids=["chat_completions", "responses"],
)
def test_invoke_chat_shaped_routes_without_safeguards_drop_only_the_dangerous_beta(
    gateway: Gateway, path: str, build: Callable[[str], dict[str, JsonValue]]
) -> None:
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{CLAUDE_ID}", api_key=TOKEN, aws_region_name="us-east-1", api_base=wire.url
        )
        with_beta: Final = _invoke_body_betas(
            _delivered(gateway, wire, path, build(model), f"{KEPT_BETA},{DANGEROUS_TOOL_USE_BETA}")
        )
        without_beta: Final = _invoke_body_betas(_delivered(gateway, wire, path, build(model), KEPT_BETA))
    assert KEPT_BETA in without_beta, without_beta
    assert with_beta == without_beta, (with_beta, without_beta)


def test_mantle_messages_without_safeguards_keeps_the_dangerous_beta(gateway: Gateway) -> None:
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock_mantle/{MANTLE_CLAUDE_ID}", api_key=TOKEN, aws_region_name="us-east-1", api_base=wire.url
        )
        sent: Final = _mantle_header_betas(
            _delivered(gateway, wire, "/v1/messages", _messages_body(model), f"{KEPT_BETA},{DANGEROUS_TOOL_USE_BETA}")
        )
    assert DANGEROUS_TOOL_USE_BETA in sent, sent
