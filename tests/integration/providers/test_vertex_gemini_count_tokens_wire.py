import asyncio
import json
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from types import MappingProxyType
from typing import Final
from uuid import uuid4

import anthropic
import httpx
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.process import graceful_stop_seconds
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "gemini-4-argon"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_COUNT: Final = 777
_UNIT: Final = " red blue green yellow orange purple black white silver gold"
_USER: Final = f"<filler>{_UNIT * 100}</filler>\nReturn exactly OK."
_MESSAGES: Final[JsonValue] = [{"role": "user", "content": _USER}]
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])
_REPLY: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "OK"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 11},
}


class _GeminiPeer:
    def __init__(self) -> None:
        self.status: int = 200
        self.requests: tuple[Request, ...] = ()
        self._lock: Final = threading.Lock()

    def set_status(self, status: int) -> None:
        self.status = status

    def __call__(self, request: Request) -> Reply:
        with self._lock:
            self.requests = (*self.requests, request)
            status: Final = self.status
        if request.target.endswith(":countTokens"):
            return Reply(status=status, body=json.dumps({"totalTokens": _COUNT}).encode())
        if request.target.endswith(":generateContent"):
            return Reply(body=json.dumps(_REPLY).encode())
        return Reply(status=404, body=json.dumps({"error": request.target}).encode())

    def drain(self) -> tuple[Request, ...]:
        with self._lock:
            requests: Final = self.requests
            self.requests = ()
        return requests


def _deployment(gateway: Gateway, scenario: Scenario, api_base: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{_MODEL}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    )


def _messages(model: str, system: JsonValue | None, tools: JsonValue | None) -> Mapping[str, JsonValue]:
    return {
        "model": model,
        "messages": _MESSAGES,
        **({"system": system} if system is not None else {}),
        **({"tools": tools} if tools is not None else {}),
    }


def _json_body(request: Request) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _count_requests(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.target.endswith(":countTokens"))


def _count_tokens_tools(generated_tools: JsonValue) -> JsonValue:
    return _JSON_LIST.validate_python(
        [
            {
                {
                    "function_declarations": "functionDeclarations",
                    "url_context": "urlContext",
                    "code_execution": "codeExecution",
                }.get(key, key): value
                for key, value in tool.items()
            }
            for tool in TypeAdapter(list[Mapping[str, JsonValue]]).validate_python(generated_tools)
        ]
    )


def _count_tokens_request_from_generate_content(
    generated_body: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    system_instruction: Final = generated_body.get("systemInstruction", generated_body.get("system_instruction"))
    generated_tools: Final = generated_body.get("tools")
    return MappingProxyType(
        {
            "contents": generated_body["contents"],
            **({"systemInstruction": system_instruction} if system_instruction is not None else {}),
            **({"tools": _count_tokens_tools(generated_tools)} if generated_tools is not None else {}),
        }
    )


def _tools(description: str) -> JsonValue:
    return _JSON_LIST.validate_python(
        [
            {
                "name": "noop",
                "description": description,
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    )


@pytest.mark.parametrize(
    ("large_system", "large_tool"),
    ((False, False), (True, False), (False, True), (True, True)),
    ids=("baseline", "large-system", "large-tool", "large-system-and-tool"),
)
def test_messages_and_count_tokens_send_matching_gemini_context(
    gateway: Gateway, large_system: bool, large_tool: bool
) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        system: Final = (
            f"<cache-buster>{uuid4().hex}</cache-buster>\n"
            "This is a synthetic context-budget diagnostic. Return exactly OK." + (_UNIT * 100 if large_system else "")
            if large_system
            else None
        )
        tools: Final = _tools(_UNIT * 100) if large_tool else None
        body: Final = _messages(model, system, tools)
        generated: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 16})
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body)
        assert generated.status_code == 200, generated.text
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _COUNT, counted.text
        requests: Final = peer.drain()
        generated_body: Final = _json_body(
            next(request for request in requests if request.target.endswith(":generateContent"))
        )
        count_body: Final = _json_body(next(request for request in requests if request.target.endswith(":countTokens")))
        assert count_body == _count_tokens_request_from_generate_content(generated_body)


def test_count_tokens_keeps_each_text_block_in_system_instruction(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        system: Final[JsonValue] = [
            {"type": "text", "text": "First instruction"},
            {"type": "text", "text": "Second instruction"},
        ]
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, system, None))
        assert response.status_code == 200, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        assert _json_body(requests[0]) == {
            "contents": [{"role": "user", "parts": [{"text": _USER}]}],
            "systemInstruction": {"parts": [{"text": "First instruction"}, {"text": "Second instruction"}]},
        }


def test_count_tokens_without_system_or_tools_sends_contents_only(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, None, None))
        assert response.status_code == 200, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        assert _json_body(requests[0]) == {"contents": [{"role": "user", "parts": [{"text": _USER}]}]}


_SDK_COUNT_ARGS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "messages": [{"role": "user", "content": _USER}],
        "system": "Follow the system instruction.",
        "tools": [
            {
                "name": "noop",
                "description": "Return no output",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }
)


def _sync_sdk_count_tokens(gateway: Gateway, model: str) -> int:
    client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
    return client.messages.count_tokens(model=model, **_SDK_COUNT_ARGS).input_tokens  # pyright: ignore[reportArgumentType]  # SDK TypedDicts reject a JsonValue mapping


async def _async_sdk_count_tokens(gateway: Gateway, model: str) -> int:
    client: Final = anthropic.AsyncAnthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
    return (await client.messages.count_tokens(model=model, **_SDK_COUNT_ARGS)).input_tokens  # pyright: ignore[reportArgumentType]  # SDK TypedDicts reject a JsonValue mapping


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync-sdk", "async-sdk"))
def test_anthropic_sdk_count_tokens_forwards_system_and_tools(gateway: Gateway, asynchronous: bool) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        input_tokens: Final = (
            asyncio.run(_async_sdk_count_tokens(gateway, model))
            if asynchronous
            else _sync_sdk_count_tokens(gateway, model)
        )
        assert input_tokens == _COUNT
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["systemInstruction"] == {"parts": [{"text": "Follow the system instruction."}]}
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {"name": "noop", "description": "Return no output", "parameters": {"type": "object"}}
                ]
            }
        ]


def test_google_count_tokens_forwards_system_and_tools_to_vertex(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        body: Final[Mapping[str, JsonValue]] = {
            "contents": [
                {"role": "user", "parts": [{"text": _USER}]},
                {"role": "model", "parts": [{"functionCall": {"name": "noop", "args": {"query": "value"}}}]},
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "noop", "response": {"result": "ok"}}}],
                },
            ],
            "systemInstruction": {"parts": [{"text": "Follow this instruction."}]},
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "noop",
                            "description": "Return no output",
                            "parameters": {"type": "OBJECT", "properties": {}},
                        }
                    ]
                }
            ],
        }
        response: Final = gateway.request("POST", f"/v1beta/models/{model}:countTokens", body)
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["totalTokens"] == _COUNT, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["contents"] == body["contents"]
        assert request_body["systemInstruction"] == body["systemInstruction"]
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "noop",
                        "description": "Return no output",
                        "parameters": {"type": "object"},
                    }
                ]
            }
        ]


def test_responses_input_tokens_maps_flat_function_tools_to_vertex(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        body: Final[Mapping[str, JsonValue]] = {
            "model": model,
            "instructions": "Follow the system instruction.",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_lookup",
                    "name": "lookup",
                    "arguments": '{"query":"value"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_lookup",
                    "output": "result",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        response: Final = gateway.request("POST", "/v1/responses/input_tokens", body)
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNT, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "lookup",
                        "description": "Look up a value",
                        "parameters": {"type": "object"},
                    }
                ]
            }
        ]


def test_utils_token_counter_maps_openai_function_tools_to_vertex(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        tool_schema: Final[JsonValue] = {"type": "object", "properties": {}}
        body: Final[Mapping[str, JsonValue]] = {
            "model": model,
            "messages": _MESSAGES,
            "system": "Follow the system instruction.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Return no output",
                        "parameters": tool_schema,
                    },
                }
            ],
        }
        response: Final = gateway.request("POST", "/utils/token_counter", body, params={"call_endpoint": "true"})
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["total_tokens"] == _COUNT, response.text
        requests: Final = _count_requests(wire)
        assert len(requests) == 1
        request_body: Final = _json_body(requests[0])
        assert request_body["systemInstruction"] == {"parts": [{"text": "Follow the system instruction."}]}
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "noop",
                        "description": "Return no output",
                        "parameters": {"type": "object"},
                    }
                ]
            }
        ]


def test_count_tokens_surfaces_peer_errors_and_rejects_non_text_system(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    peer.set_status(400)
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, None, None))
        assert response.status_code == 400, response.text
        assert len(_count_requests(wire)) == 1

    invalid_peer: Final = _GeminiPeer()
    with wire_server(invalid_peer) as invalid_wire, gateway.scenario() as scenario:
        invalid_model: Final = _deployment(gateway, scenario, invalid_wire.url)
        invalid_response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            _messages(
                invalid_model,
                [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
                None,
            ),
        )
        assert invalid_response.status_code == 400, invalid_response.text
        assert _count_requests(invalid_wire) == ()


@pytest.mark.parametrize(
    "tools",
    (
        [{"name": "noop", "input_schema": {"type": "object", "properties": {"query": "not-a-schema"}}}],
        [{"name": "noop", "input_schema": "not-a-schema"}],
    ),
    ids=("malformed-property", "string-schema"),
)
def test_count_tokens_rejects_unconvertible_tools_without_request(gateway: Gateway, tools: JsonValue) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", _messages(model, None, tools))
        assert response.status_code == 400, response.text
        assert "tools" in response.text
        assert _count_requests(wire) == ()


def _count_wave(clients: Sequence[httpx.Client], body: Mapping[str, JsonValue]) -> tuple[tuple[int, JsonValue], ...]:
    def count(client: httpx.Client) -> tuple[int, JsonValue]:
        response: Final = client.post("/v1/messages/count_tokens", json=dict(body))
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        return response.status_code, payload.get("input_tokens")

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        return tuple(executor.map(count, clients))


def test_concurrent_count_waves_recover_after_vertex_count_outage(gateway: Gateway) -> None:
    peer: Final = _GeminiPeer()
    with wire_server(peer) as wire, gateway.scenario() as scenario, ExitStack() as stack:
        model: Final = _deployment(gateway, scenario, wire.url)
        clients: Final = tuple(
            stack.enter_context(
                httpx.Client(
                    base_url=str(gateway.client.base_url),
                    headers={"Authorization": f"Bearer {gateway.key}"},
                    timeout=30,
                    trust_env=False,
                )
            )
            for _ in range(3)
        )
        body: Final = _messages(model, None, None)
        assert _count_wave(clients, body) == ((200, _COUNT), (200, _COUNT), (200, _COUNT))
        peer.set_status(503)
        assert _count_wave(clients, body) == ((503, None), (503, None), (503, None))
        peer.set_status(200)
        assert _count_wave(clients, body) == ((200, _COUNT), (200, _COUNT), (200, _COUNT))
        requests: Final = _count_requests(wire)
        assert len(requests) == 9
        assert (
            tuple(_json_body(request) for request in requests)
            == ({"contents": [{"role": "user", "parts": [{"text": _USER}]}]},) * 9
        )
