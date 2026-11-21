"""Contract: where prompt caching marks land when a chat request is sent to Anthropic.

Anthropic caches a prompt prefix up to each `cache_control` block, at most four per request. On chat the caller can
mark a block two ways. Anthropic's own `cache_control` (`{"type": "ephemeral"}`, optionally `"ttl": "1h"`) is copied
onto the translated block by `add_cache_control_to_content` (`litellm/litellm_core_utils/prompt_templates/
factory.py:2281`) for user and assistant parts, assistant tool calls and tool results (`factory.py:1973`), and onto
tools by `_map_tool_helper` (`litellm/llms/anthropic/chat/transformation.py:1317`). OpenAI's newer
`prompt_cache_breakpoint` (`{"mode": "explicit"}`, optional `ttl` of `30m` or `12h`) on a user content part is
translated to `cache_control` first (`AnthropicCacheControlHook.prompt_cache_breakpoints_as_capped_cache_control`,
`litellm/integrations/anthropic_cache_control_hook.py:1293`, and `prompt_cache_breakpoint_as_cache_control`,
`litellm/litellm_core_utils/prompt_templates/prompt_cache_breakpoint_utils.py:103`). A ttl Anthropic cannot honour,
a malformed marker, or more than four breakpoints after translation is a 400 before any provider call unless
`drop_params` is on, in which case the ttl is dropped, the malformed marker is dropped, or the middle marks are
dropped.

Round 1 pins a cache_control on a user part and on a system message, round 2 a 1h cache_control on a tool; this
file covers the rest. Every upstream is a loopback peer.
"""

from __future__ import annotations

from typing import Final

from integration._support.client import Gateway
from integration.providers.dec1_anthropic_support import (
    chat,
    content_blocks,
    deployment,
    error_message,
    marker,
    peer,
    sent_message,
    sent_messages,
    served,
    tool,
)
from pydantic import JsonValue

_EPHEMERAL: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
_ONE_HOUR: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
_EXPLICIT: Final[dict[str, JsonValue]] = {"mode": "explicit"}


def _user_parts(*parts: dict[str, JsonValue]) -> list[JsonValue]:
    return [{"role": "user", "content": [*parts, {"type": "text", "text": marker()}]}]


def _text(text: str, **marks: JsonValue) -> dict[str, JsonValue]:
    return {"type": "text", "text": text, **marks}


def _first_user_blocks(messages: list[JsonValue], gateway: Gateway, **extra: JsonValue) -> list[dict[str, JsonValue]]:
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=messages, **extra))
        call: Final = provider.only()
        assert b"prompt_cache_breakpoint" not in call.raw
        return content_blocks(sent_message(call, 0))


def _tool_turns(**marks: JsonValue) -> list[JsonValue]:
    return [
        {"role": "user", "content": marker()},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_dec1a",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    **marks,
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_dec1a", "content": "18C", **marks},
    ]


def test_a_prompt_cache_breakpoint_on_a_user_part_becomes_an_ephemeral_cache_control(gateway: Gateway) -> None:
    """The caller marks a user text part `prefix` with `prompt_cache_breakpoint: {"mode": "explicit"}`.

    The upstream user block `prefix` must carry `cache_control: {"type": "ephemeral"}` and the raw body must not
    contain `prompt_cache_breakpoint`, a key Anthropic would reject (`prompt_cache_breakpoint_as_cache_control`).
    """
    blocks: Final = _first_user_blocks(_user_parts(_text("prefix", prompt_cache_breakpoint=_EXPLICIT)), gateway)
    assert blocks[0] == {"type": "text", "text": "prefix", "cache_control": _EPHEMERAL}
    assert "cache_control" not in blocks[1]


def test_a_prompt_cache_breakpoint_ttl_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """The caller marks a user part with `prompt_cache_breakpoint: {"mode": "explicit", "ttl": "30m"}` and no
    `drop_params`.

    Anthropic's caches last 5m or 1h, so the caller must get 400 whose message says the ttl `has no equivalent`, and
    the peer must not be called (`_cache_control_for_prompt_cache_breakpoint`, `prompt_cache_breakpoint_utils.py:
    136`). Silently sending a 5m cache would bill the caller for a cache that expires earlier than asked.
    """
    messages: Final = _user_parts(_text("prefix", prompt_cache_breakpoint={"mode": "explicit", "ttl": "30m"}))
    with peer() as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider), messages=messages)
        assert response.status_code == 400, response.text
        assert "has no equivalent" in error_message(response)
        assert provider.posts() == ()


def test_a_prompt_cache_breakpoint_ttl_is_sent_as_a_five_minute_cache_under_drop_params(gateway: Gateway) -> None:
    """Same `ttl: 30m` marker with `drop_params: true` on the request.

    The upstream block must carry exactly `cache_control: {"type": "ephemeral"}`, with no ttl
    (`prompt_cache_breakpoint_utils.py:145`).
    """
    messages: Final = _user_parts(_text("prefix", prompt_cache_breakpoint={"mode": "explicit", "ttl": "30m"}))
    blocks: Final = _first_user_blocks(messages, gateway, drop_params=True)
    assert blocks[0].get("cache_control") == _EPHEMERAL


def test_a_malformed_prompt_cache_breakpoint_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """The caller marks a user part with `prompt_cache_breakpoint: {"mode": "auto"}` and no `drop_params`.

    The caller must get 400 whose message starts the explanation with `prompt_cache_breakpoint must be`, and the
    peer must not be called (`prompt_cache_breakpoint_utils.py:124`).
    """
    messages: Final = _user_parts(_text("prefix", prompt_cache_breakpoint={"mode": "auto"}))
    with peer() as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider), messages=messages)
        assert response.status_code == 400, response.text
        assert "prompt_cache_breakpoint must be" in error_message(response)
        assert provider.posts() == ()


def test_a_malformed_prompt_cache_breakpoint_is_dropped_under_drop_params(gateway: Gateway) -> None:
    """Same `{"mode": "auto"}` marker with `drop_params: true`.

    The request must reach the provider with the block `prefix` carrying neither `cache_control` nor the marker
    (`_cache_control_for_prompt_cache_breakpoint` returns None and the marker key is removed).
    """
    blocks: Final = _first_user_blocks(
        _user_parts(_text("prefix", prompt_cache_breakpoint={"mode": "auto"})), gateway, drop_params=True
    )
    assert blocks[0] == {"type": "text", "text": "prefix"}


def test_a_client_cache_control_wins_over_a_prompt_cache_breakpoint_on_the_same_block(gateway: Gateway) -> None:
    """The caller puts both `cache_control: {"type": "ephemeral", "ttl": "1h"}` and `prompt_cache_breakpoint:
    {"mode": "explicit"}` on one user part.

    The upstream block must keep the caller's 1h `cache_control` unchanged (`prompt_cache_breakpoint_utils.py:112`),
    not a 5m one derived from the breakpoint.
    """
    part: Final = _text("prefix", cache_control=_ONE_HOUR, prompt_cache_breakpoint=_EXPLICIT)
    blocks: Final = _first_user_blocks(_user_parts(part), gateway)
    assert blocks[0].get("cache_control") == _ONE_HOUR


def test_five_translated_breakpoints_are_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """The caller marks five user parts with `prompt_cache_breakpoint` and no `drop_params`.

    Anthropic takes at most four cache breakpoints (`MAX_CACHE_CONTROL_BLOCKS`, `anthropic_cache_control_hook.py:
    77`), so the caller must get 400 whose message names `at most 4 cache breakpoints`, and the peer must not be
    called (`anthropic_cache_control_hook.py:1320`).
    """
    parts: Final = [_text(f"part {n}", prompt_cache_breakpoint=_EXPLICIT) for n in range(1, 6)]
    with peer() as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider), messages=_user_parts(*parts))
        assert response.status_code == 400, response.text
        assert "at most 4 cache breakpoints" in error_message(response)
        assert provider.posts() == ()


def test_five_translated_breakpoints_keep_the_first_and_last_four_under_drop_params(gateway: Gateway) -> None:
    """Same five marked parts with `drop_params: true`.

    Exactly four upstream blocks may carry `cache_control`, and the middle one (`part 3`) must be the one left
    unmarked: the first and last translated marks fill the slots so the stable prefix and the newest turns stay
    cached (docstring of `prompt_cache_breakpoints_as_capped_cache_control`, `anthropic_cache_control_hook.py:1299`).
    """
    parts: Final = [_text(f"part {n}", prompt_cache_breakpoint=_EXPLICIT) for n in range(1, 6)]
    blocks: Final = _first_user_blocks(_user_parts(*parts), gateway, drop_params=True)
    marked: Final = [block["text"] for block in blocks if "cache_control" in block]
    assert marked == ["part 1", "part 2", "part 4", "part 5"]


def test_a_prompt_cache_breakpoint_on_an_assistant_part_is_not_sent(gateway: Gateway) -> None:
    """The history has an assistant text part `earlier answer` marked with `prompt_cache_breakpoint`.

    Assistant marks are left untranslated (`_may_carry_prompt_cache_breakpoint`, `anthropic_cache_control_hook.py:
    701`) and the factory copies only `cache_control`, so the upstream assistant block must carry no `cache_control`
    and the raw body no `prompt_cache_breakpoint`. UNVERIFIED: whether a future change should translate it instead.
    """
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": [_text("earlier answer", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": "next"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=messages))
        call: Final = provider.only()
        assert b"prompt_cache_breakpoint" not in call.raw
        assistant: Final = sent_message(call, 1)
        assert assistant["role"] == "assistant"
        assert all("cache_control" not in block for block in content_blocks(assistant))


def test_a_one_hour_cache_control_on_a_user_part_is_sent_with_its_ttl(gateway: Gateway) -> None:
    """The caller marks a user part with `cache_control: {"type": "ephemeral", "ttl": "1h"}`.

    The upstream block must carry exactly that object, ttl included (`ChatCompletionCachedContent(**param)`,
    `factory.py:2292`); losing the ttl turns a 1h write into a 5m one.
    """
    blocks: Final = _first_user_blocks(_user_parts(_text("prefix", cache_control=_ONE_HOUR)), gateway)
    assert blocks[0].get("cache_control") == _ONE_HOUR


def test_a_cache_control_inside_a_function_definition_is_moved_onto_the_anthropic_tool(gateway: Gateway) -> None:
    """The caller's function tool carries `cache_control: {"type": "ephemeral"}` inside `function`, the shape some
    OpenAI SDK users send.

    The upstream tool `get_weather` must carry `cache_control: {"type": "ephemeral"}` at the top level
    (`_cache_control_function`, `transformation.py:1329`), so the tool list is cached.
    """
    weather: Final = tool("get_weather", city={"type": "string"})
    function: Final = weather["function"]
    assert isinstance(function, dict)
    marked: Final = {**weather, "function": {**function, "cache_control": _EPHEMERAL}}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[marked]))
        tools: Final = provider.only().body["tools"]
        assert isinstance(tools, list) and len(tools) == 1
        sent: Final = tools[0]
        assert isinstance(sent, dict)
        assert (sent["name"], sent.get("cache_control")) == ("get_weather", _EPHEMERAL)


def test_a_cache_control_on_a_tool_message_lands_on_the_tool_result_block(gateway: Gateway) -> None:
    """The history's tool message `18C` carries a message-level `cache_control: {"type": "ephemeral"}`.

    The upstream `tool_result` block for `toolu_dec1a` must carry that `cache_control` (`convert_to_anthropic_tool_
    result`, `factory.py:1973`), which is how agents cache a long tool output.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=_tool_turns(cache_control=_EPHEMERAL),
                tools=[tool("get_weather", city={"type": "string"})],
            )
        )
        results: Final = [
            block
            for message in sent_messages(provider.only())
            if isinstance(message, dict)
            for block in content_blocks(message)
            if block.get("type") == "tool_result"
        ]
        assert [(block["tool_use_id"], block.get("cache_control")) for block in results] == [
            ("toolu_dec1a", _EPHEMERAL)
        ]


def test_a_cache_control_on_an_assistant_tool_call_lands_on_the_tool_use_block(gateway: Gateway) -> None:
    """The history's assistant tool call `toolu_dec1a` carries `cache_control: {"type": "ephemeral"}`.

    The upstream `tool_use` block must carry that `cache_control` (`convert_to_anthropic_tool_invoke`,
    `factory.py:2268`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=_tool_turns(cache_control=_EPHEMERAL),
                tools=[tool("get_weather", city={"type": "string"})],
            )
        )
        uses: Final = [
            block for block in content_blocks(sent_message(provider.only(), 1)) if block.get("type") == "tool_use"
        ]
        assert [(block["id"], block.get("cache_control")) for block in uses] == [("toolu_dec1a", _EPHEMERAL)]


def test_a_message_level_cache_control_on_an_assistant_string_lands_on_its_text_block(gateway: Gateway) -> None:
    """The history has an assistant message with string content `earlier answer` and a message-level
    `cache_control: {"type": "ephemeral"}`.

    The upstream assistant turn must be one text block `earlier answer` carrying that `cache_control`
    (`factory.py:3409`), so a cached conversation prefix can end on an assistant turn.
    """
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": "earlier answer", "cache_control": _EPHEMERAL},
        {"role": "user", "content": "next"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=messages))
        assert content_blocks(sent_message(provider.only(), 1)) == [
            {"type": "text", "text": "earlier answer", "cache_control": _EPHEMERAL}
        ]
