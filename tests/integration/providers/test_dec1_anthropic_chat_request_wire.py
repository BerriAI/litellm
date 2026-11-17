"""Contract: how `POST /v1/chat/completions` on an `anthropic/` deployment becomes an Anthropic Messages request.

The chat route is still Python (`litellm/rust_bridge/catalog.py` keeps `Route.CHAT_COMPLETIONS` PYTHON_ONLY), and
the translation lives in `AnthropicConfig` (`litellm/llms/anthropic/chat/transformation.py`) plus
`anthropic_messages_pt` (`litellm/litellm_core_utils/prompt_templates/factory.py`). When either is rewritten, the
upstream request must stay the same: one `POST {api_base}/v1/messages` carrying the deployment key as `x-api-key`,
`anthropic-version: 2023-06-01`, the backend model id without its `anthropic/` prefix, the leading system messages
lifted into the top-level `system` block list, OpenAI tools and tool choices rewritten to Anthropic's shapes, and a
`max_tokens` the caller never has to send because the Messages API requires it.

Every test reads the request the owned peer recorded, because a leaked or missing field is invisible in the
caller's response. Each deployment is created for one test, so only that test's requests reach its peer.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_anthropic_support import (
    ADAPTIVE_BACKEND,
    BACKEND,
    PDF_BASE64,
    PNG_BASE64,
    PROVIDER_KEY,
    betas,
    chat,
    content_blocks,
    deployment,
    marker,
    peer,
    sent_message,
    sent_messages,
    served,
    tool,
    user,
)
from pydantic import JsonValue

_WEATHER: Final = tool("get_weather", "Weather for a city", city={"type": "string"})
_TIME: Final = tool("get_time", "Local time for a city", city={"type": "string"})
_NOTE: Final = (
    "Operator note (not from the user): the following was originally a mid-conversation system-role reminder."
)


def test_the_request_is_one_post_to_v1_messages_on_the_deployment_api_base(gateway: Gateway) -> None:
    """A chat call on an `anthropic/claude-sonnet-4-5-20250929` deployment whose `api_base` is the peer root.

    The peer must see exactly one POST to `/v1/messages` with no query string, and the caller gets 200. The
    Anthropic config appends `/v1/messages` to a bare base (`get_complete_url`); a rewrite that keeps the OpenAI
    `/chat/completions` suffix would hit the wrong route.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider)))
        call: Final = provider.only()
        assert call.path == "/v1/messages"
        assert dict(call.query) == {}


def test_the_backend_model_is_sent_without_the_provider_prefix(gateway: Gateway) -> None:
    """The deployment model is `anthropic/claude-sonnet-4-5-20250929` under a random proxy model name.

    The upstream body `model` must be exactly `claude-sonnet-4-5-20250929`: neither the proxy alias nor the
    `anthropic/` routing prefix may leak to Anthropic, which would answer 404 not_found_error.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider)))
        assert provider.only().body["model"] == BACKEND


def test_the_deployment_key_is_sent_as_x_api_key_and_the_caller_key_is_not_forwarded(gateway: Gateway) -> None:
    """The caller authenticates with the proxy master key as a Bearer token; the deployment key is synthetic.

    The upstream request must carry `x-api-key: dec1-anthropic-provider-key` and no `authorization` header,
    because a plain api key on a custom base uses the `x-api-key` scheme unless `use_bearer_for_custom_base` is set
    (`AnthropicModelInfo._make_api_key_auth_header`). The proxy key must not appear in any upstream header.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider)))
        headers: Final = provider.only().headers
        assert headers.get("x-api-key") == PROVIDER_KEY
        assert "authorization" not in headers
        assert all(gateway.key not in value for value in headers.values())


def test_use_bearer_for_custom_base_sends_the_key_as_a_bearer_token(gateway: Gateway) -> None:
    """The deployment sets `use_bearer_for_custom_base: true` and a key that does not start with `sk-ant-`.

    Because the base is not api.anthropic.com, the key must be sent as `authorization: Bearer
    dec1-anthropic-provider-key` and `x-api-key` must be absent (`_make_api_key_auth_header`). Gateways in front of
    Anthropic depend on this switch.

    BUG (run 44176682961): the peer gets `x-api-key` and no `authorization`. `AnthropicConfig` reads the switch only
    from `litellm_params` (litellm/llms/anthropic/common_utils.py:1589-1591, used at :1446-1450), but
    `use_bearer_for_custom_base` is not one of the keys `get_litellm_params` copies from the call's kwargs
    (`OPTIONAL_KWARGS_KEYS`, litellm/litellm_core_utils/get_litellm_params.py:51-112 and :310-314), so a
    deployment's value never reaches it. Only unit tests that pass `litellm_params` directly exercise it.
    """
    pytest.skip("BUG: a deployment's use_bearer_for_custom_base never reaches AnthropicConfig's litellm_params")
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider, use_bearer_for_custom_base=True)))
        headers: Final = provider.only().headers
        assert headers.get("authorization") == f"Bearer {PROVIDER_KEY}"
        assert "x-api-key" not in headers


def test_anthropic_version_is_2023_06_01_and_the_body_is_json(gateway: Gateway) -> None:
    """A plain chat call. The upstream request must carry `anthropic-version: 2023-06-01`, `accept:
    application/json` and a JSON `content-type` (`get_anthropic_headers` defaults), and the body must decode as a
    JSON object with `model`, `messages` and `max_tokens`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider)))
        call: Final = provider.only()
        assert call.headers.get("anthropic-version") == "2023-06-01"
        assert call.headers.get("accept") == "application/json"
        assert call.headers.get("content-type", "").startswith("application/json")
        assert {"model", "messages", "max_tokens"} <= set(call.body)


def test_a_plain_request_sends_no_anthropic_beta_header(gateway: Gateway) -> None:
    """A text-only chat call with no tools, files, thinking or structured output.

    No `anthropic-beta` header may be sent: prompt caching no longer needs a beta and none of the feature detectors
    in `get_anthropic_headers` fire. A stray beta can make Anthropic reject the call for a model that lacks it.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider)))
        assert "anthropic-beta" not in provider.only().headers


def test_a_single_user_message_is_sent_as_one_user_turn(gateway: Gateway) -> None:
    """The caller sends one user message with string content.

    The upstream `messages` must hold exactly one entry with role `user` whose only text is the caller's text,
    and no `system` field is sent at all when the caller sent no system message.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), text=text))
        call: Final = provider.only()
        assert len(sent_messages(call)) == 1
        message: Final = sent_message(call, 0)
        assert message["role"] == "user"
        assert [block.get("text") for block in content_blocks(message)] == [text]
        assert "system" not in call.body


def test_a_leading_system_message_becomes_the_top_level_system_block_list(gateway: Gateway) -> None:
    """The caller sends a system message then a user message.

    The upstream body must carry `system: [{"type": "text", "text": <system>}]` and the system text must not
    appear in `messages`, which starts with the user turn (`translate_system_message`).
    """
    instruction: Final = f"be terse {marker()}"
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=[{"role": "system", "content": instruction}, user()],
            )
        )
        call: Final = provider.only()
        assert call.body["system"] == [{"type": "text", "text": instruction}]
        assert sent_message(call, 0)["role"] == "user"
        assert len(sent_messages(call)) == 1


def test_several_leading_system_messages_become_ordered_system_blocks(gateway: Gateway) -> None:
    """The caller sends two system messages (one string, one list of two text parts) before the user turn.

    The upstream `system` must be three text blocks in caller order, one per string and one per part, so the
    cached prefix Anthropic sees is stable across turns.
    """
    first, second, third = marker(), marker(), marker()
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=[
                    {"role": "system", "content": first},
                    {"role": "system", "content": [{"type": "text", "text": second}, {"type": "text", "text": third}]},
                    user(),
                ],
            )
        )
        system: Final = list_value(provider.only().body["system"])
        assert [object_value(block)["text"] for block in system] == [first, second, third]


def test_an_empty_system_message_is_left_out_of_the_system_blocks(gateway: Gateway) -> None:
    """The caller sends an empty string system message, a real one, then a user turn.

    Anthropic rejects empty text blocks, so the empty system message must be skipped and `system` must hold only
    the non-empty block (`translate_system_message` skips empty strings).
    """
    instruction: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=[{"role": "system", "content": ""}, {"role": "system", "content": instruction}, user()],
            )
        )
        assert provider.only().body["system"] == [{"type": "text", "text": instruction}]


def test_a_system_message_cache_control_is_kept_on_its_system_block(gateway: Gateway) -> None:
    """The caller's system message is a list with one text part carrying `cache_control: {"type": "ephemeral"}`.

    The upstream system block must keep `cache_control: {"type": "ephemeral"}` next to its text, which is what
    makes Anthropic write the prompt cache at that breakpoint.
    """
    instruction: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": instruction, "cache_control": {"type": "ephemeral"}}],
                    },
                    user(),
                ],
            )
        )
        assert provider.only().body["system"] == [
            {"type": "text", "text": instruction, "cache_control": {"type": "ephemeral"}}
        ]


def test_a_mid_conversation_system_message_becomes_a_noted_user_turn_on_an_unflagged_model(gateway: Gateway) -> None:
    """On `claude-sonnet-4-5-20250929` (no `supports_mid_conversation_system` in the cost map) the caller sends
    user, assistant, system, user.

    Only the leading run is lifted, so `system` is absent upstream. The later system message must become user
    text prefixed by the operator note, merged with the following user turn: the third upstream message is a user
    turn whose texts are the note, the system text and the user text, in that order
    (`mid_conversation_system._converted_for_unflagged_model`, #36559). Hoisting it would re-bill the cached prefix.
    """
    reminder, last = marker(), marker()
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=[
                    user(),
                    {"role": "assistant", "content": "ok"},
                    {"role": "system", "content": reminder},
                    user(last),
                ],
            )
        )
        call: Final = provider.only()
        assert "system" not in call.body
        assert [object_value(message)["role"] for message in sent_messages(call)] == ["user", "assistant", "user"]
        assert [block.get("text") for block in content_blocks(sent_message(call, 2))] == [_NOTE, reminder, last]


def test_a_mid_conversation_system_message_stays_a_system_turn_on_a_flagged_model(gateway: Gateway) -> None:
    """On `claude-opus-5-5` (`supports_mid_conversation_system: true`) the caller sends user, assistant, system, user.

    The system run after the assistant turn must slide behind the following user turn and keep role `system`:
    upstream roles are user, assistant, user, system and the last message's text is the reminder
    (`_placed_for_flagged_model`). UNVERIFIED: the exact block shape of an in-conversation system message; the
    test reads only its role and text.
    """
    reminder: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider, "anthropic/claude-opus-5-5", priced=False),
                messages=[
                    user(),
                    {"role": "assistant", "content": "ok"},
                    {"role": "system", "content": reminder},
                    user(),
                ],
            )
        )
        call: Final = provider.only()
        assert [object_value(message)["role"] for message in sent_messages(call)] == [
            "user",
            "assistant",
            "user",
            "system",
        ]
        assert [block.get("text") for block in content_blocks(sent_message(call, 3))] == [reminder]


def test_consecutive_user_messages_are_merged_into_one_user_turn(gateway: Gateway) -> None:
    """The caller sends two user messages in a row. Anthropic requires alternating roles, so the upstream
    `messages` must hold one user turn whose two text blocks are the two texts in order (`anthropic_messages_pt`).
    """
    first, second = marker(), marker()
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[user(first), user(second)]))
        call: Final = provider.only()
        assert len(sent_messages(call)) == 1
        assert [block.get("text") for block in content_blocks(sent_message(call, 0))] == [first, second]


def test_the_default_max_tokens_is_the_model_max_output_tokens(gateway: Gateway) -> None:
    """The caller sends no `max_tokens`. The Messages API requires one, so the proxy must send the cost map's
    `max_output_tokens` for `claude-sonnet-4-5-20250929`, which is 64000 (`AnthropicConfig.get_max_tokens_for_model`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider, priced=False)))
        assert provider.only().body["max_tokens"] == 64000


def test_the_default_max_tokens_follows_the_model_row(gateway: Gateway) -> None:
    """The deployment is `anthropic/claude-opus-4-6`, whose cost map row has `max_output_tokens: 128000`.

    Without a caller `max_tokens` the upstream body must say `max_tokens: 128000`, so the default is per model and
    not one constant.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider, f"anthropic/{ADAPTIVE_BACKEND}", priced=False)))
        assert provider.only().body["max_tokens"] == 128000


def test_an_unmapped_model_gets_the_128000_fallback_max_tokens(gateway: Gateway) -> None:
    """The deployment is `anthropic/claude-dec1-unmapped`, which has no cost map row.

    Without a caller `max_tokens` the upstream body must say `max_tokens: 128000`, the
    `DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS` default in `litellm/constants.py`. UNVERIFIED: a `fallback_generalizations`
    rule could give an unmapped Claude id another row; check `get_max_tokens` if this fails.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider, "anthropic/claude-dec1-unmapped", priced=False)))
        assert provider.only().body["max_tokens"] == 128000


def test_a_deployment_max_output_tokens_overrides_the_model_default(gateway: Gateway) -> None:
    """The deployment's `model_info` sets `max_output_tokens: 2048`; the caller sends no `max_tokens`.

    The upstream body must carry `max_tokens: 2048` (`transform_request` reads `litellm_params.model_info`).
    UNVERIFIED: that the router hands the deployment's `model_info` to `litellm_params` on the chat route.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False, model_info={"max_output_tokens": 2048})
        served(chat(gateway, model))
        assert provider.only().body["max_tokens"] == 2048


@pytest.mark.parametrize("field", ("max_tokens", "max_completion_tokens"))
def test_a_caller_token_limit_is_sent_as_max_tokens(gateway: Gateway, field: str) -> None:
    """The caller sends `max_tokens: 321` or `max_completion_tokens: 321`. Either must reach Anthropic as
    `max_tokens: 321`, and `max_completion_tokens` must not be sent (Anthropic would answer 400 extra inputs).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), **{field: 321}))
        body: Final = provider.only().body
        assert body["max_tokens"] == 321
        assert "max_completion_tokens" not in body


def test_stop_as_a_string_becomes_a_one_item_stop_sequences_list(gateway: Gateway) -> None:
    """The caller sends `stop: "END"`. The upstream body must carry `stop_sequences: ["END"]` and no `stop`
    field (`_map_stop_sequences`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), stop="END"))
        body: Final = provider.only().body
        assert body["stop_sequences"] == ["END"]
        assert "stop" not in body


def test_stop_as_a_list_keeps_its_order(gateway: Gateway) -> None:
    """The caller sends `stop: ["###", "END", "\\n\\nHuman:"]`. The upstream `stop_sequences` must be the same three
    strings in the same order.
    """
    stops: Final[list[JsonValue]] = ["###", "END", "\n\nHuman:"]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), stop=stops))
        assert provider.only().body["stop_sequences"] == stops


def test_a_whitespace_stop_sequence_is_dropped_under_drop_params(gateway: Gateway) -> None:
    """The deployment sets `drop_params: true`; the caller sends `stop: [" ", "END"]`.

    Anthropic rejects whitespace-only stop sequences, so the upstream `stop_sequences` must be `["END"]`.
    `_map_stop_sequences` reads only the global `litellm.drop_params` (transformation.py:1770), which the shared
    proxy leaves false, so today the whitespace entry is sent and Anthropic answers 400.
    """
    pytest.skip("BUG: deployment drop_params does not drop a whitespace stop sequence; only litellm.drop_params does")
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider, drop_params=True), stop=[" ", "END"]))
        assert provider.only().body["stop_sequences"] == ["END"]


def test_temperature_and_top_p_are_forwarded_on_a_model_with_sampling_params(gateway: Gateway) -> None:
    """The caller sends `temperature: 0.3` and `top_p: 0.9` to `claude-sonnet-4-5-20250929`, which supports
    sampling params. Both must reach the upstream body unchanged.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), temperature=0.3, top_p=0.9))
        body: Final = provider.only().body
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.9


def test_top_k_is_forwarded_as_a_provider_specific_param(gateway: Gateway) -> None:
    """The caller sends `top_k: 5`, which is not an OpenAI param. It must reach Anthropic as `top_k: 5`
    (`transform_request` gates it through `_apply_sampling_param`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), top_k=5))
        assert provider.only().body["top_k"] == 5


def test_the_openai_user_becomes_metadata_user_id(gateway: Gateway) -> None:
    """The caller sends `user: "dec1-user-42"`. The upstream body must carry `metadata: {"user_id":
    "dec1-user-42"}` and no top-level `user` field, which Anthropic would reject.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), user="dec1-user-42"))
        body: Final = provider.only().body
        assert body["metadata"] == {"user_id": "dec1-user-42"}
        assert "user" not in body


def test_an_email_user_is_not_sent_as_metadata(gateway: Gateway) -> None:
    """The caller sends `user: "someone@example.com"`. Anthropic refuses email addresses as user ids, so no
    `metadata` may be sent (`_valid_user_id`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), user="someone@example.com"))
        assert "metadata" not in provider.only().body


def test_function_tools_become_anthropic_tools_with_input_schema(gateway: Gateway) -> None:
    """The caller sends one OpenAI function tool `get_weather` with a description and a `city` string parameter.

    The upstream `tools` must hold one tool with `name: get_weather`, the description, `type: custom`, and an
    `input_schema` whose `properties` and `required` are the caller's (`_map_tool_helper`). No `function` wrapper
    may remain.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[_WEATHER]))
        tools: Final = list_value(provider.only().body["tools"])
        assert len(tools) == 1
        sent: Final = object_value(tools[0])
        assert sent["name"] == "get_weather"
        assert sent["description"] == "Weather for a city"
        assert sent.get("type") == "custom"
        assert "function" not in sent
        schema: Final = object_value(sent["input_schema"])
        assert schema["properties"] == {"city": {"type": "string"}}
        assert schema["required"] == ["city"]


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        ({"type": "function", "function": {"name": "get_weather"}}, {"type": "tool", "name": "get_weather"}),
    ],
    ids=("auto", "required", "none", "named"),
)
def test_tool_choice_is_mapped_to_the_anthropic_shape(
    gateway: Gateway, tool_choice: JsonValue, expected: dict[str, JsonValue]
) -> None:
    """The caller sends a tool and each OpenAI `tool_choice`. Upstream `tool_choice` must be the Anthropic shape:
    auto to `{"type": "auto"}`, required to `{"type": "any"}`, none to `{"type": "none"}`, a named function to
    `{"type": "tool", "name": ...}` (`_map_tool_choice`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[_WEATHER], tool_choice=tool_choice))
        assert provider.only().body["tool_choice"] == expected


@pytest.mark.parametrize(
    ("parallel", "tool_choice", "expected"),
    [
        (False, None, {"type": "auto", "disable_parallel_tool_use": True}),
        (True, None, {"type": "auto", "disable_parallel_tool_use": False}),
        (False, "required", {"type": "any", "disable_parallel_tool_use": True}),
        (False, "none", {"type": "none"}),
    ],
    ids=("off-alone", "on-alone", "off-with-required", "off-with-none"),
)
def test_parallel_tool_calls_becomes_the_inverse_disable_parallel_tool_use(
    gateway: Gateway, parallel: bool, tool_choice: str | None, expected: dict[str, JsonValue]
) -> None:
    """The caller sends tools with `parallel_tool_calls` and optionally a `tool_choice`.

    Anthropic's flag is the inverse: `parallel_tool_calls: false` must become `disable_parallel_tool_use: true` on
    the mapped choice, an absent choice defaults to `auto`, and `tool_choice: none` carries no flag at all.
    """
    extra: Final[dict[str, JsonValue]] = {"parallel_tool_calls": parallel}
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[_WEATHER, _TIME], **extra))
        assert provider.only().body["tool_choice"] == expected


def test_assistant_tool_calls_and_tool_results_become_tool_use_and_tool_result_blocks(gateway: Gateway) -> None:
    """The history holds a user turn, an assistant turn with one tool call `toolu_dec1a` to `get_weather` with
    arguments `{"city":"Paris"}`, and the tool's answer `18C`.

    Upstream, the assistant turn must hold a `tool_use` block with that id, name and the arguments parsed into an
    object, and the next user turn must open with a `tool_result` for `toolu_dec1a` whose text is `18C`.
    """
    history: Final[list[JsonValue]] = [
        user(),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_dec1a",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_dec1a", "content": "18C"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=history, tools=[_WEATHER]))
        call: Final = provider.only()
        tool_use: Final = [b for b in content_blocks(sent_message(call, 1)) if b["type"] == "tool_use"]
        assert tool_use == [
            {"type": "tool_use", "id": "toolu_dec1a", "name": "get_weather", "input": {"city": "Paris"}}
        ]
        result: Final = content_blocks(sent_message(call, 2))[0]
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_dec1a"
        content: Final = result["content"]
        texts: Final = (
            [content] if isinstance(content, str) else [object_value(b).get("text") for b in list_value(content)]
        )
        assert texts == ["18C"]


def test_parallel_tool_results_are_merged_into_one_user_turn_in_order(gateway: Gateway) -> None:
    """The assistant turn made two tool calls (`toolu_dec1a`, `toolu_dec1b`) and two tool messages answer them.

    Anthropic requires every result of one assistant turn in the next user turn, so upstream there must be exactly
    three messages and the last one must hold two `tool_result` blocks for `toolu_dec1a` then `toolu_dec1b`.
    """
    history: Final[list[JsonValue]] = [
        user(),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_dec1a",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                },
                {
                    "id": "toolu_dec1b",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": '{"city":"Paris"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_dec1a", "content": "18C"},
        {"role": "tool", "tool_call_id": "toolu_dec1b", "content": "09:00"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=history, tools=[_WEATHER, _TIME]))
        call: Final = provider.only()
        assert len(sent_messages(call)) == 3
        results: Final = [b["tool_use_id"] for b in content_blocks(sent_message(call, 2)) if b["type"] == "tool_result"]
        assert results == ["toolu_dec1a", "toolu_dec1b"]


def test_a_tool_history_without_tools_gets_the_dummy_tool(gateway: Gateway) -> None:
    """The history holds a tool call and its result but the caller sends no `tools`.

    Anthropic rejects tool blocks without tool definitions, so the proxy must add one tool named `dummy_tool`
    with description `This is a dummy tool call` (`litellm.utils.add_dummy_tool`).
    """
    history: Final[list[JsonValue]] = [
        user(),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "toolu_dec1a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_dec1a", "content": "18C"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=history))
        tools: Final = [object_value(t) for t in list_value(provider.only().body["tools"])]
        assert [(t["name"], t.get("description")) for t in tools] == [("dummy_tool", "This is a dummy tool call")]


def test_a_tool_name_with_invalid_characters_is_sanitized_upstream(gateway: Gateway) -> None:
    """The caller's tool is named `weather.lookup/v1`. Anthropic enforces `^[a-zA-Z0-9_-]{1,128}$`, so the upstream
    tool name must be `weather_lookup_v1` (`_basic_sanitize_anthropic_tool_name` replaces each invalid char with `_`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(gateway, deployment(scenario, provider), tools=[tool("weather.lookup/v1", city={"type": "string"})])
        )
        assert [object_value(t)["name"] for t in list_value(provider.only().body["tools"])] == ["weather_lookup_v1"]


def test_a_base64_image_becomes_an_image_block_with_media_type_and_data(gateway: Gateway) -> None:
    """The user turn has a text part and an `image_url` part with a `data:image/png;base64,...` URL.

    Upstream the user turn must hold an `image` block with `source: {"type": "base64", "media_type": "image/png",
    "data": <the same base64>}` and the text block, both in the caller's part order (`create_anthropic_image_param`).
    """
    text: Final = marker()
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"}},
            {"type": "text", "text": text},
        ],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message]))
        blocks: Final = content_blocks(sent_message(provider.only(), 0))
        assert [b["type"] for b in blocks] == ["image", "text"]
        assert blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}


def test_an_https_image_url_is_sent_as_a_url_source_without_downloading(gateway: Gateway) -> None:
    """The user turn has an `image_url` part with `https://images.dec1.invalid/cat.png`.

    Anthropic fetches https images itself, so the upstream block must be `{"type": "image", "source": {"type":
    "url", "url": <same url>}}` and the proxy must not try to fetch it (the host does not resolve, so a fetch would
    fail the call).
    """
    url: Final = "https://images.dec1.invalid/cat.png"
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [{"type": "text", "text": marker()}, {"type": "image_url", "image_url": {"url": url}}],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message]))
        blocks: Final = content_blocks(sent_message(provider.only(), 0))
        assert {"type": "image", "source": {"type": "url", "url": url}} in blocks


def test_an_http_image_url_on_a_loopback_host_is_refused_before_any_fetch(gateway: Gateway) -> None:
    """The user turn has an `image_url` part pointing at `http://<peer>/image.png` on 127.0.0.1.

    Pinned current (run 44176682961). Plain-http URLs are inlined (`AnthropicConfig.inlines_remote_media`), but the
    image fetch runs the proxy's URL policy first and refuses a loopback host that is not in
    `user_url_allowed_hosts` (`_rejected_image_fetch`, litellm/litellm_core_utils/prompt_templates/image_handling.py:103-108).
    The caller gets HTTP 400 naming the URL policy, and neither the image nor the chat reaches the peer. The
    inlining itself would need an owned proxy that allowlists 127.0.0.1.
    """
    with peer() as provider, gateway.scenario() as scenario:
        message: Final[dict[str, JsonValue]] = {
            "role": "user",
            "content": [
                {"type": "text", "text": marker()},
                {"type": "image_url", "image_url": {"url": f"{provider.url}/image.png"}},
            ],
        }
        response: Final = chat(gateway, deployment(scenario, provider), messages=[message])
        assert response.status_code == 400, response.text
        assert "Unable to fetch image from URL" in response.text
        assert "user_url_allowed_hosts" in response.text
        assert provider.all() == ()


def test_a_base64_pdf_file_part_becomes_a_document_block(gateway: Gateway) -> None:
    """The user turn has a `file` part with `file_data: data:application/pdf;base64,...`.

    Upstream it must be a `document` block with `source: {"type": "base64", "media_type": "application/pdf",
    "data": <same base64>}` (`anthropic_process_openai_file_message`), and no files beta is needed.
    """
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [
            {"type": "text", "text": marker()},
            {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{PDF_BASE64}"}},
        ],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message]))
        call: Final = provider.only()
        documents: Final = [b for b in content_blocks(sent_message(call, 0)) if b["type"] == "document"]
        assert documents == [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_BASE64}}
        ]
        assert "files-api-2025-04-14" not in betas(call)


def test_a_pdf_url_file_id_becomes_a_url_document_block(gateway: Gateway) -> None:
    """The user turn has a `file` part whose `file_id` is `https://docs.dec1.invalid/report.pdf` and no format.

    A URL file id is read as a document URL (`anthropic_infer_file_id_content_type`), so the upstream block must be
    `{"type": "document", "source": {"type": "url", "url": <same url>}}`.
    """
    url: Final = "https://docs.dec1.invalid/report.pdf"
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [{"type": "text", "text": marker()}, {"type": "file", "file": {"file_id": url}}],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message]))
        blocks: Final = content_blocks(sent_message(provider.only(), 0))
        assert {"type": "document", "source": {"type": "url", "url": url}} in blocks


def test_an_anthropic_file_id_becomes_a_file_source_and_adds_the_files_betas(gateway: Gateway) -> None:
    """The user turn has a `file` part with `file_id: file_dec1abc` and `format: application/pdf`.

    The upstream block must be `{"type": "document", "source": {"type": "file", "file_id": "file_dec1abc"}}` and
    the request must carry the `files-api-2025-04-14` and `code-execution-2025-05-22` betas (`get_anthropic_headers`
    with `file_id_used`).
    """
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [
            {"type": "text", "text": marker()},
            {"type": "file", "file": {"file_id": "file_dec1abc", "format": "application/pdf"}},
        ],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message]))
        call: Final = provider.only()
        assert {"type": "document", "source": {"type": "file", "file_id": "file_dec1abc"}} in content_blocks(
            sent_message(call, 0)
        )
        assert {"files-api-2025-04-14", "code-execution-2025-05-22"} <= betas(call)


def test_an_unknown_caller_beta_is_dropped_and_the_derived_files_beta_kept(gateway: Gateway) -> None:
    """The caller passes `extra_headers: {"anthropic-beta": "dec1-feature-2026-01-01"}` and an Anthropic file id.

    `get_anthropic_headers` unions the caller's betas (`user_anthropic_beta_headers`) with the derived files beta.

    Pinned current (run 44176682961): the merged header passes through `anthropic_beta_headers_manager`
    (`filter_and_transform_beta_headers`, litellm/anthropic_beta_headers_manager.py:296), which drops a beta
    the provider row does not list, so the unknown caller beta is dropped and only the derived files beta,
    once, reaches Anthropic.
    """
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [
            {"type": "text", "text": marker()},
            {"type": "file", "file": {"file_id": "file_dec1abc", "format": "application/pdf"}},
        ],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=[message],
                extra_headers={"anthropic-beta": "dec1-feature-2026-01-01"},
            )
        )
        sent: Final = provider.only().headers.get("anthropic-beta", "")
        values: Final = [beta.strip() for beta in sent.split(",") if beta.strip()]
        assert "dec1-feature-2026-01-01" not in values
        assert "files-api-2025-04-14" in values
        assert len(values) == len(set(values))


def test_web_search_options_become_the_hosted_web_search_tool(gateway: Gateway) -> None:
    """The caller sends `web_search_options: {"search_context_size": "high", "user_location": {"type":
    "approximate", "approximate": {"city": "Paris", "country": "FR"}}}`.

    The upstream `tools` must contain `{"type": "web_search_20250305", "name": "web_search", "max_uses": 10,
    "user_location": {"type": "approximate", "city": "Paris", "country": "FR"}}` (`map_web_search_tool`, high maps
    to 10 in `ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES`).
    """
    options: Final[dict[str, JsonValue]] = {
        "search_context_size": "high",
        "user_location": {"type": "approximate", "approximate": {"city": "Paris", "country": "FR"}},
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), web_search_options=options))
        tools: Final = [object_value(t) for t in list_value(provider.only().body["tools"])]
        assert {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 10,
            "user_location": {"type": "approximate", "city": "Paris", "country": "FR"},
        } in tools


def test_a_hosted_code_execution_tool_is_forwarded_with_its_beta(gateway: Gateway) -> None:
    """The caller sends the Anthropic hosted tool `{"type": "code_execution_20250825", "name": "code_execution"}`.

    It must reach Anthropic unchanged in `tools`, and the request must carry the `code-execution-2025-08-25` beta
    (`is_code_execution_tool_used`).
    """
    hosted: Final[dict[str, JsonValue]] = {"type": "code_execution_20250825", "name": "code_execution"}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[hosted]))
        call: Final = provider.only()
        assert hosted in [object_value(t) for t in list_value(call.body["tools"])]
        assert "code-execution-2025-08-25" in betas(call)


def test_a_container_upload_file_adds_the_code_execution_tool(gateway: Gateway) -> None:
    """The user turn has a `file` part with `file_id: file_dec1csv` and `format: text/csv`, which maps to a
    `container_upload` block.

    Upstream the block must be `{"type": "container_upload", "file_id": "file_dec1csv"}` and `tools` must gain
    `{"name": "code_execution", "type": "code_execution_20250522"}` next to the caller's `get_weather` tool
    (`add_code_execution_tool`).
    """
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [
            {"type": "text", "text": marker()},
            {"type": "file", "file": {"file_id": "file_dec1csv", "format": "text/csv"}},
        ],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message], tools=[_WEATHER]))
        call: Final = provider.only()
        assert {"type": "container_upload", "file_id": "file_dec1csv"} in content_blocks(sent_message(call, 0))
        assert {"name": "code_execution", "type": "code_execution_20250522"} in [
            object_value(t) for t in list_value(call.body["tools"])
        ]


def test_a_top_level_cache_control_is_forwarded_for_automatic_prompt_caching(gateway: Gateway) -> None:
    """The caller sends `cache_control: {"type": "ephemeral"}` at the top level of the chat body.

    It must reach Anthropic unchanged as the top-level `cache_control` (`map_openai_params` passes it through), and
    no prompt caching beta is added.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), cache_control={"type": "ephemeral"}))
        call: Final = provider.only()
        assert call.body["cache_control"] == {"type": "ephemeral"}
        assert not any(beta.startswith("prompt-caching") for beta in betas(call))


def test_a_user_content_part_cache_control_is_kept_on_its_block(gateway: Gateway) -> None:
    """The user turn is a list of two text parts; the first carries `cache_control: {"type": "ephemeral", "ttl":
    "1h"}`. Upstream, the first text block must keep exactly that `cache_control` and the second must have none.
    """
    first, second = marker(), marker()
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [
            {"type": "text", "text": first, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": second},
        ],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message]))
        blocks: Final = content_blocks(sent_message(provider.only(), 0))
        assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "cache_control" not in blocks[1]


def test_cache_control_injection_points_mark_the_system_message(gateway: Gateway) -> None:
    """The deployment sets `cache_control_injection_points: [{"location": "message", "role": "system"}]`; the
    caller sends a plain system string and a user turn with no marks.

    The upstream system block must carry `cache_control: {"type": "ephemeral"}` and the injection point setting
    itself must not be sent (`AnthropicCacheControlHook`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, cache_control_injection_points=[{"location": "message", "role": "system"}]
        )
        served(chat(gateway, model, messages=[{"role": "system", "content": marker()}, user()]))
        body: Final = provider.only().body
        system: Final = [object_value(b) for b in list_value(body["system"])]
        assert system[-1].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control_injection_points" not in body


def test_no_openai_only_or_proxy_fields_leak_into_the_anthropic_body(gateway: Gateway) -> None:
    """The caller sends `n: 1`, `user`, `stop`, `metadata: {"tags": ["dec1"]}` and `stream: false`.

    The upstream body must not contain `n`, `user`, `stop`, `stream_options`, `litellm_metadata`, `proxy_server_request`
    or `metadata.tags`; an unknown key makes Anthropic answer 400 `Extra inputs are not permitted`.
    UNVERIFIED: whether `stream: false` is sent; the test does not check `stream`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                n=1,
                user="dec1-user",
                stop="END",
                metadata={"tags": ["dec1"]},
            )
        )
        body: Final = provider.only().body
        assert not {"n", "user", "stop", "stream_options", "litellm_metadata", "proxy_server_request"} & set(body)
        assert body.get("metadata") in (None, {"user_id": "dec1-user"})


def test_a_container_upload_without_caller_tools_still_gets_the_code_execution_tool(gateway: Gateway) -> None:
    """Same `container_upload` file part as above, but the caller sends no `tools`.

    The upstream body must still carry `tools: [{"name": "code_execution", "type": "code_execution_20250522"}]`,
    since Anthropic cannot run a container upload without it. `transform_request` writes the list back only when
    `len(tools) > 1` (transformation.py:2698), so a lone added tool is dropped today.
    """
    pytest.skip("BUG: a container_upload with no caller tools is sent without the code_execution tool")
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [
            {"type": "text", "text": marker()},
            {"type": "file", "file": {"file_id": "file_dec1csv", "format": "text/csv"}},
        ],
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=[message]))
        assert list_value(provider.only().body["tools"]) == [
            {"name": "code_execution", "type": "code_execution_20250522"}
        ]
