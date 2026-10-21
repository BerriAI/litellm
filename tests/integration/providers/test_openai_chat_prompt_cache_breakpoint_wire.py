from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import openai
import pytest
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, gateway_from_environment, string_value
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(120)

_SYSTEM_AND_USER_WITH_CACHE_CONTROL: Final[list[JsonValue]] = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "Answer briefly"},
            {"type": "text", "text": "Stable rules", "cache_control": {"type": "ephemeral"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": "and nothing else"},
        ],
    },
]


def _reply_ok(request: Request) -> Reply:
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    return chat_reply("chatcmpl-1", string_value(body["model"]), "ok", stream=body.get("stream") is True)


@dataclass(frozen=True, slots=True)
class _Proxy:
    gateway: Gateway
    openai: Wire
    gpt_5_6_with_prompt_cache_options: str
    gpt_5_6_without_prompt_cache_options: str

    def openai_receives(
        self, model: str, messages: Sequence[JsonValue], *, stream: bool = False, tools: Sequence[JsonValue] = ()
    ) -> dict[str, JsonValue]:
        _ = self.openai.drain()
        request: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": list(messages),
            "stream": stream,
            "cache": {"no-cache": True},
        }
        response: Final = self.gateway.request(
            "POST", "/v1/chat/completions", {**request, "tools": list(tools)} if tools else request
        )
        assert response.status_code == 200, response.text
        return self.received_by_openai()

    def received_by_openai(self) -> dict[str, JsonValue]:
        posts: Final = tuple(request for request in self.openai.drain() if request.method == "POST")
        assert [post.target for post in posts] == ["/v1/chat/completions"], posts
        return rv.JSON_OBJECT.validate_json(posts[0].body)


@pytest.fixture(scope="module")
def proxy() -> Iterator[_Proxy]:
    with (
        wire_server(answering_model_discovery(_reply_ok)) as openai_upstream,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
    ):
        api_base: Final = f"{openai_upstream.url}/v1"
        yield _Proxy(
            gateway,
            openai_upstream,
            gpt_5_6_with_prompt_cache_options=scenario.model(
                model="openai/gpt-5.6", api_base=api_base, prompt_cache_options={"mode": "explicit"}
            ),
            gpt_5_6_without_prompt_cache_options=scenario.model(model="openai/gpt-5.6", api_base=api_base),
        )


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_cache_control_on_a_part_becomes_prompt_cache_breakpoint_on_that_part(proxy: _Proxy, stream: bool) -> None:
    received: Final = proxy.openai_receives(
        proxy.gpt_5_6_with_prompt_cache_options, _SYSTEM_AND_USER_WITH_CACHE_CONTROL, stream=stream
    )

    assert received["messages"] == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Answer briefly"},
                {"type": "text", "text": "Stable rules", "prompt_cache_breakpoint": {"mode": "explicit"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this", "prompt_cache_breakpoint": {"mode": "explicit"}},
                {"type": "text", "text": "and nothing else"},
            ],
        },
    ]
    assert received["prompt_cache_options"] == {"mode": "explicit"}
    assert received.get("stream", False) is stream
    assert "cache_control" not in str(received)


def test_cache_control_on_a_string_message_sent_with_the_openai_sdk(proxy: _Proxy) -> None:
    with openai.OpenAI(
        api_key=proxy.gateway.key, base_url=f"{str(proxy.gateway.client.base_url).rstrip('/')}/v1", max_retries=0
    ) as sdk:
        completion: Final = sdk.chat.completions.create(
            model=proxy.gpt_5_6_with_prompt_cache_options,
            messages=[
                {"role": "system", "content": "Answer briefly", "cache_control": {"type": "ephemeral"}},  # pyright: ignore[reportArgumentType]  # Anthropic mark the SDK forwards verbatim
                {"role": "user", "content": "Say hi"},
            ],
            extra_body={"cache": {"no-cache": True}},
        )
    assert completion.choices[0].message.content == "ok"

    assert proxy.received_by_openai()["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Answer briefly", "prompt_cache_breakpoint": {"mode": "explicit"}}],
        },
        {"role": "user", "content": "Say hi"},
    ]


def test_cache_control_on_a_tool_result_is_kept_and_on_a_tool_definition_is_dropped(proxy: _Proxy) -> None:
    received: Final = proxy.openai_receives(
        proxy.gpt_5_6_with_prompt_cache_options,
        [
            {"role": "user", "content": "Look it up"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "found it", "cache_control": {"type": "ephemeral"}}],
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {}},
                "cache_control": {"type": "ephemeral"},
            }
        ],
    )

    assert received["messages"] == [
        {"role": "user", "content": "Look it up"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "found it", "prompt_cache_breakpoint": {"mode": "explicit"}}],
        },
    ], "a breakpoint on a tool message part writes the cache on live gpt-5.6 Chat Completions (2026-10-12)"
    assert received["tools"] == [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    assert "cache_control" not in str(received)


def test_cache_control_on_an_assistant_turn_is_dropped(proxy: _Proxy) -> None:
    received: Final = proxy.openai_receives(
        proxy.gpt_5_6_with_prompt_cache_options,
        [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Earlier answer", "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": "Second question"},
        ],
    )

    assert received["messages"] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": [{"type": "text", "text": "Earlier answer"}]},
        {"role": "user", "content": "Second question"},
    ]
    assert "cache_control" not in str(received)
    assert "prompt_cache_breakpoint" not in str(received)


def test_custom_api_base_without_prompt_cache_options_forwards_cache_control_unchanged(proxy: _Proxy) -> None:
    received: Final = proxy.openai_receives(
        proxy.gpt_5_6_without_prompt_cache_options, _SYSTEM_AND_USER_WITH_CACHE_CONTROL
    )

    assert received["messages"] == _SYSTEM_AND_USER_WITH_CACHE_CONTROL
    assert "prompt_cache_options" not in received
    assert "prompt_cache_breakpoint" not in str(received)
