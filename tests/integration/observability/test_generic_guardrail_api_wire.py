import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway, eventually
from integration._support.openai_wire import chat_reply
from integration._support.ports import allocate_port
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

GEMINI_MODEL: Final = "gemini/gemini-3.8-flash"
ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-5-5"
OPENAI_MODEL: Final = "openai/gpt-5.6"
ECHO_GEMINI_TOOLS: Final = "echo-gemini-tools"
NONE_CHAT_ROWS: Final = "none-chat-rows"
NONE_CHAT_ROWS_DEFAULT: Final = "none-chat-rows-default"
NONE_ANTHROPIC_IMAGE: Final = "none-anthropic-image"
ECHO_ANTHROPIC_MESSAGES: Final = "echo-anthropic-messages"
FORWARDS_MESSAGES_AS_SENT: Final = frozenset({NONE_CHAT_ROWS, NONE_ANTHROPIC_IMAGE, ECHO_ANTHROPIC_MESSAGES})
FAIL_OPEN_CONNECT_ERROR: Final = "fail-open-connect-error"
FAIL_OPEN_TIMEOUT: Final = "fail-open-timeout"
FAIL_OPEN_HTTP_STATUS: Final = "fail-open-http-status"
FAIL_OPEN_LINE: Final = "Generic Guardrail API error (fail-open)"
PNG: Final = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR4nGO4IydHEmIY1TCqYfhqAACaMxgQdGu1YAAAAABJRU5ErkJggg=="
)
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])


def _verdict_none(_request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _verdict_echo(request: Request) -> Reply:
    body: Final = _JSON.validate_json(request.body)
    echoed: Final = {key: body[key] for key in ("structured_messages", "tools") if body.get(key)}
    return Reply(body=json.dumps({"action": "NONE", **echoed}).encode())


def _unavailable(_request: Request) -> Reply:
    return Reply(status=503, body=b'{"error": "guardrail unavailable"}')


def _gemini_reply(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
            }
        ).encode()
    )


def _anthropic_reply(_request: Request) -> Reply:
    return Reply(body=message_json("msg_guardrail_wire", "claude-sonnet-5-5", "ok"))


def _openai_reply(_request: Request) -> Reply:
    return chat_reply("chatcmpl-guardrail-wire", "gpt-5.6", "ok", stream=False)


def _guardrail(name: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": False,
            "api_base": api_base,
            "api_key": "synthetic-guardrail-key",
            **params,
        },
    }


@dataclass(frozen=True, slots=True)
class GuardrailRig:
    owned: OwnedProxy
    policies: Mapping[str, Wire]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[GuardrailRig]:
    directory: Final = tmp_path_factory.mktemp("generic-guardrail-wire")
    release: Final = threading.Event()

    def held_until_teardown(request: Request) -> Reply:
        release.wait()
        return _verdict_none(request)

    with ExitStack() as stack:
        policies: Final = MappingProxyType(
            {
                ECHO_GEMINI_TOOLS: stack.enter_context(wire_server(_verdict_echo)),
                NONE_CHAT_ROWS: stack.enter_context(wire_server(_verdict_none)),
                NONE_CHAT_ROWS_DEFAULT: stack.enter_context(wire_server(_verdict_none)),
                NONE_ANTHROPIC_IMAGE: stack.enter_context(wire_server(_verdict_none)),
                ECHO_ANTHROPIC_MESSAGES: stack.enter_context(wire_server(_verdict_echo)),
            }
        )
        stalled: Final = stack.enter_context(wire_server(held_until_teardown))
        stack.callback(release.set)
        unavailable: Final = stack.enter_context(wire_server(_unavailable))
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "generic-guardrail-wire.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "guardrails": [
                        *(
                            _guardrail(name, wire.url, forward_messages_as_sent=name in FORWARDS_MESSAGES_AS_SENT)
                            for name, wire in policies.items()
                        ),
                        _guardrail(FAIL_OPEN_CONNECT_ERROR, f"http://127.0.0.1:{allocate_port()}", fail_on_error=False),
                        _guardrail(FAIL_OPEN_TIMEOUT, stalled.url, fail_on_error=False, timeout=1),
                        _guardrail(FAIL_OPEN_HTTP_STATUS, unavailable.url, fail_on_error=False),
                    ],
                }
            )
        )
        owned: Final = stack.enter_context(owned_proxy_process(session_gateway, directory, {}, config=path))
        yield GuardrailRig(owned, policies)


def _bodies(wire: Wire) -> list[dict[str, JsonValue]]:
    return [_JSON.validate_json(request.body) for request in wire.drain()]


def _fail_open_lines(log: Path, guardrail: str) -> list[str]:
    return [
        line
        for line in log.read_text(errors="replace").splitlines()
        if FAIL_OPEN_LINE in line and f"guardrail_name={guardrail} " in line
    ]


def test_gemini_native_tools_reach_the_guardrail_and_gemini_as_sent(rig: GuardrailRig) -> None:
    tools: Final = [
        {"function_declarations": [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}]},
        {"googleSearch": {}},
    ]
    with wire_server(_gemini_reply) as gemini, rig.owned.gateway.scenario() as scenario:
        model: Final = scenario.model(model=GEMINI_MODEL, api_base=gemini.url, api_key="synthetic-gemini-key")
        response: Final = rig.owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": tools,
                "guardrails": [ECHO_GEMINI_TOOLS],
            },
        )
        assert response.status_code == 200, response.text
        assert [body["tools"] for body in _bodies(rig.policies[ECHO_GEMINI_TOOLS])] == [tools]
        sent: Final = _bodies(gemini)
        assert len(sent) == 1, sent
        assert {"googleSearch": {}} in sent[0]["tools"], sent
        assert [tool for tool in sent[0]["tools"] if "type" in tool] == [], sent


CHAT_ROWS: Final = (
    {"role": "user", "name": "pat", "content": "weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
)
OPENAI_CHAT_ROWS: Final = (
    {"role": "user", "content": "weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
)


@pytest.mark.parametrize(
    ("guardrail", "seen"),
    [(NONE_CHAT_ROWS, CHAT_ROWS), (NONE_CHAT_ROWS_DEFAULT, OPENAI_CHAT_ROWS)],
    ids=["forward_messages_as_sent", "default"],
)
def test_guardrail_sees_chat_rows_as_sent_only_when_configured(
    rig: GuardrailRig, guardrail: str, seen: tuple[dict[str, JsonValue], ...]
) -> None:
    messages: Final = list(CHAT_ROWS)
    with wire_server(_openai_reply) as openai_peer, rig.owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=OPENAI_MODEL, api_base=f"{openai_peer.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = rig.owned.gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": messages, "guardrails": [guardrail]}
        )
        assert response.status_code == 200, response.text
        assert [body["structured_messages"] for body in _bodies(rig.policies[guardrail])] == [list(seen)]
        assert len(openai_peer.drain()) == 1


def test_guardrail_sees_an_anthropic_image_block_on_chat_completions_as_sent(rig: GuardrailRig) -> None:
    image: Final = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]
    with wire_server(_anthropic_reply) as anthropic_peer, rig.owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=ANTHROPIC_MODEL, api_base=anthropic_peer.url, api_key="synthetic-anthropic-key"
        )
        rig.owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": image}, {"role": "user", "content": "what color?"}],
                "guardrails": [NONE_ANTHROPIC_IMAGE],
            },
        )
        seen: Final = _bodies(rig.policies[NONE_ANTHROPIC_IMAGE])
        assert [body["structured_messages"][0]["content"] for body in seen] == [image], seen


def test_echoing_guardrail_leaves_the_anthropic_messages_request_unchanged(rig: GuardrailRig) -> None:
    messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this square? Also check the weather in Paris."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Trip notes."}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny, 21C"},
                {"type": "text", "text": "Answer both in one sentence."},
            ],
        },
    ]
    tools: Final = [
        {
            "name": "get_weather",
            "description": "Weather for a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    with wire_server(_anthropic_reply) as anthropic_peer, rig.owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=ANTHROPIC_MODEL, api_base=anthropic_peer.url, api_key="synthetic-anthropic-key"
        )
        response: Final = rig.owned.gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": messages,
                "tools": tools,
                "guardrails": [ECHO_ANTHROPIC_MESSAGES],
            },
        )
        assert response.status_code == 200, response.text
        assert len(rig.policies[ECHO_ANTHROPIC_MESSAGES].drain()) == 1
        assert [body["messages"] for body in _bodies(anthropic_peer)] == [messages]


@pytest.mark.parametrize(
    ("guardrail", "tokens"),
    [
        (FAIL_OPEN_CONNECT_ERROR, ("error_type=ConnectError",)),
        (FAIL_OPEN_TIMEOUT, ("error_type=Timeout",)),
        (FAIL_OPEN_HTTP_STATUS, ("error_type=HTTPStatusError", "http_status_code=503")),
    ],
    ids=["connect_error", "timeout", "http_status"],
)
def test_fail_open_log_line_names_the_error_type(rig: GuardrailRig, guardrail: str, tokens: tuple[str, ...]) -> None:
    with wire_server(_openai_reply) as openai_peer, rig.owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=OPENAI_MODEL, api_base=f"{openai_peer.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = rig.owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hi"}], "guardrails": [guardrail]},
        )
        assert response.status_code == 200, response.text
        assert len(openai_peer.drain()) == 1
    lines: Final = eventually(lambda: _fail_open_lines(rig.owned.log, guardrail), bool, seconds=30)
    assert [line for line in lines if not all(token in line for token in tokens)] == [], lines
