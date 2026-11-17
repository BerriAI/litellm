"""Contract: how a non-streaming chat answer surfaces Anthropic blocks and usage fields that round 1 and 2 left out.

`AnthropicConfig.extract_response_content` (`litellm/llms/anthropic/chat/transformation.py:2877`) sorts the
provider's content blocks: `tool_use` and `server_tool_use` become tool calls (server ones are then filtered out
of `message.tool_calls`), every `*_tool_result` lands in `tool_results` (web search and web fetch results in
`web_search_results`), `thinking` and `redacted_thinking` become `thinking_blocks`, and `compaction` blocks are kept.
Any other block type, such as `mcp_tool_use`, matches no branch and is dropped. `_build_provider_specific_fields`
(`transformation.py:3237`) then puts them on `message.provider_specific_fields`, with `input_transformations` and
the `compaction` stop reason handled on the way. `calculate_usage` (`transformation.py:3075`) turns Anthropic usage
into OpenAI usage, including the five-minute and one-hour cache write split, compaction `iterations`, the tool
search count and `inference_geo`.

Every upstream is a loopback peer that answers a canned Anthropic message.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_anthropic2_support import provider_fields, replying, usage_of
from integration.providers.dec1_anthropic3_support import (
    MCP_TOOL_RESULT,
    MCP_TOOL_USE,
    ONE_HOUR_PRICES,
    REDACTED_DATA,
    spend_of,
)
from integration.providers.dec1_anthropic_support import (
    SIGNATURE,
    chat,
    deployment,
    first_choice,
    first_message,
    peer,
    served,
    usage_block,
)
from pydantic import JsonValue

_MCP_TURN: Final[list[JsonValue]] = [
    dict(MCP_TOOL_USE),
    dict(MCP_TOOL_RESULT),
    {"type": "text", "text": "widgets ship in blue"},
]


def _answer(gateway: Gateway, content: list[JsonValue] | None = None, **fields: JsonValue) -> dict[str, JsonValue]:
    with peer(replying(content, **fields)) as provider, gateway.scenario() as scenario:
        return served(chat(gateway, deployment(scenario, provider)))


def test_an_mcp_tool_use_block_is_not_returned_as_a_client_tool_call(gateway: Gateway) -> None:
    """The provider answers an `mcp_tool_use` block, its `mcp_tool_result` and the text `widgets ship in blue`.

    The caller's message must have no `tool_calls`, content `widgets ship in blue` and finish reason `stop`: the MCP
    call ran on Anthropic's side, and `extract_response_content` only turns `tool_use` and `server_tool_use` into
    calls (`transformation.py:2900`). A port that maps every `*_tool_use` to a tool call would make the client run it.
    """
    body: Final = _answer(gateway, _MCP_TURN)
    message: Final = first_message(body)
    assert not message.get("tool_calls")
    assert message["content"] == "widgets ship in blue"
    assert first_choice(body)["finish_reason"] == "stop"


def test_an_mcp_tool_result_is_returned_in_tool_results_and_the_mcp_call_itself_nowhere(gateway: Gateway) -> None:
    """Same MCP answer. `mcp_tool_result` ends in `_tool_result`, so it lands in `provider_specific_fields.
    tool_results` exactly as sent, with `code_interpreter_results: []` beside it (`transformation.py:3276`).

    The `mcp_tool_use` block matches no branch, so no provider field may carry it and `server_tool_calls` must be
    absent: the caller sees the MCP result but not the call that produced it, which a port should keep or fix on
    purpose.
    """
    fields: Final = provider_fields(_answer(gateway, _MCP_TURN))
    assert fields["tool_results"] == [dict(MCP_TOOL_RESULT)]
    assert fields["code_interpreter_results"] == []
    assert "server_tool_calls" not in fields
    assert "mcptoolu_dec1" not in str({key: value for key, value in fields.items() if key != "tool_results"})


def test_a_redacted_thinking_block_is_returned_with_its_data_and_adds_no_reasoning_text(gateway: Gateway) -> None:
    """The provider answers a `redacted_thinking` block, a signed `thinking` block `plan` and the text `done`.

    `message.thinking_blocks` must hold both in provider order, the first `{"type": "redacted_thinking", "data":
    <data>}` and the second carrying the signature, and `reasoning_content` must be exactly `plan`, since a redacted
    block has no readable text (`transformation.py:2945`).
    """
    content: Final[list[JsonValue]] = [
        {"type": "redacted_thinking", "data": REDACTED_DATA},
        {"type": "thinking", "thinking": "plan", "signature": SIGNATURE},
        {"type": "text", "text": "done"},
    ]
    message: Final = first_message(_answer(gateway, content))
    blocks: Final = [object_value(block) for block in list_value(message["thinking_blocks"])]
    assert [(block["type"], block.get("data"), block.get("signature")) for block in blocks] == [
        ("redacted_thinking", REDACTED_DATA, None),
        ("thinking", None, SIGNATURE),
    ]
    assert message["reasoning_content"] == "plan"
    assert message["content"] == "done"


def test_a_compaction_block_is_returned_in_compaction_blocks(gateway: Gateway) -> None:
    """The provider answers a `compaction` block `Earlier turns: widgets.` and the text `ok`.

    `provider_specific_fields.compaction_blocks` must be `[<the compaction block>]` (`transformation.py:2931`) and
    the content must be only `ok`; the summary is not text the user should see.
    """
    compaction: Final[dict[str, JsonValue]] = {"type": "compaction", "content": "Earlier turns: widgets."}
    body: Final = _answer(gateway, [compaction, {"type": "text", "text": "ok"}])
    assert provider_fields(body)["compaction_blocks"] == [compaction]
    assert first_message(body)["content"] == "ok"


def test_a_compaction_stop_reason_finishes_with_length(gateway: Gateway) -> None:
    """The provider answers text with `stop_reason: compaction`.

    The caller's finish reason must be `length` (`_FINISH_REASON_MAP`, `litellm/litellm_core_utils/
    core_helpers.py:439`), and since `compaction` is not one of the raw-kept stop reasons (`refusal`, `pause_turn`,
    `common_utils.py:2972`) no `stop_reason` provider field is added.
    """
    body: Final = _answer(gateway, [{"type": "text", "text": "partial"}], stop_reason="compaction")
    assert first_choice(body)["finish_reason"] == "length"
    assert "stop_reason" not in provider_fields(body)


def test_input_transformations_are_returned_in_provider_specific_fields(gateway: Gateway) -> None:
    """The provider's message carries top-level `input_transformations: [{"type": "dec1_trimmed"}]`.

    `provider_specific_fields.input_transformations` must equal that list (`transformation.py:3257`). UNVERIFIED:
    Anthropic's exact payload shape; the proxy relays any value as sent.
    """
    transformations: Final[list[JsonValue]] = [{"type": "dec1_trimmed"}]
    body: Final = _answer(gateway, input_transformations=transformations)
    assert provider_fields(body)["input_transformations"] == transformations


def test_cache_usage_splits_prompt_tokens_into_cached_written_and_text_tokens(gateway: Gateway) -> None:
    """The provider reports `input_tokens: 5`, `cache_read_input_tokens: 7`, `cache_creation_input_tokens: 3`.

    `usage.prompt_tokens_details` must report `cached_tokens: 7`, `cache_creation_tokens: 3` and `text_tokens: 5`,
    the raw uncached input (`PromptTokensDetailsWrapper`, `transformation.py:3143`). Round 1 pins only the totals.
    """
    usage: Final = usage_of(
        _answer(gateway, usage=usage_block(5, 7, cache_read_input_tokens=7, cache_creation_input_tokens=3))
    )
    details: Final = object_value(usage["prompt_tokens_details"])
    assert (details["cached_tokens"], details["cache_creation_tokens"], details["text_tokens"]) == (7, 3, 5)


def test_a_one_hour_cache_write_breakdown_is_returned_on_the_chat_usage(gateway: Gateway) -> None:
    """The provider reports `cache_creation_input_tokens: 6` with `cache_creation: {"ephemeral_5m_input_tokens": 2,
    "ephemeral_1h_input_tokens": 4}`.

    `usage.prompt_tokens_details.cache_creation_token_details` must be `{"ephemeral_5m_input_tokens": 2,
    "ephemeral_1h_input_tokens": 4}` (`_resolve_cache_creation_token_details`, `transformation.py:3061`), so a client
    can tell how much of its write will live for an hour.
    """
    usage: Final = usage_of(
        _answer(
            gateway,
            usage=usage_block(
                cache_creation_input_tokens=6,
                cache_creation={"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4},
            ),
        )
    )
    details: Final = object_value(object_value(usage["prompt_tokens_details"])["cache_creation_token_details"])
    assert (details.get("ephemeral_5m_input_tokens"), details.get("ephemeral_1h_input_tokens")) == (2, 4)


def test_cache_writes_missing_from_the_breakdown_bill_at_the_five_minute_rate(gateway: Gateway) -> None:
    """The provider reports `cache_creation_input_tokens: 6` but a breakdown of only `ephemeral_1h_input_tokens: 4`,
    on a deployment priced 0.001 input, 0.002 output, 0.003 per base write and 0.006 per one-hour write.

    The SpendLogs row must be 11 * 0.001 + 2 * 0.003 + 4 * 0.006 + 7 * 0.002 = 0.055: the two writes no lifetime
    covers bill at the base write rate (`calculate_cache_writing_cost`, `litellm/litellm_core_utils/llm_cost_calc/
    utils.py:1037`), never at zero and never at the one-hour rate.
    """
    usage: Final = usage_block(cache_creation_input_tokens=6, cache_creation={"ephemeral_1h_input_tokens": 4})
    with peer(replying(usage=usage)) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False, **ONE_HOUR_PRICES)
        served(chat(gateway, model))
        assert spend_of(gateway, model) == pytest.approx(0.055)


def test_compaction_iterations_replace_the_top_level_token_counts(gateway: Gateway) -> None:
    """The provider reports top-level `input_tokens: 999`, `output_tokens: 999` and `iterations: [{input 5, output 3,
    cache_read 2}, {input 6, output 4}]`, as a compacted turn does.

    The caller's usage must be summed over the iterations: `prompt_tokens: 13` (5 + 6 + 2 cached), `completion_tokens:
    7`, `total_tokens: 20` (`calculate_usage`, `transformation.py:3103`); the top-level counts are ignored.
    """
    usage_in: Final = usage_block(
        999,
        999,
        iterations=[
            {"input_tokens": 5, "output_tokens": 3, "cache_read_input_tokens": 2},
            {"input_tokens": 6, "output_tokens": 4},
        ],
    )
    usage: Final = usage_of(_answer(gateway, usage=usage_in))
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (13, 7, 20)


def test_tool_search_requests_are_counted_from_the_content_when_usage_omits_them(gateway: Gateway) -> None:
    """The provider answers a `server_tool_use` named `tool_search_tool_regex`, its `tool_search_tool_result` and
    text, with no `server_tool_use` counter in usage.

    `usage.server_tool_use.tool_search_requests` must be 1, counted from the content blocks whose name contains
    `tool_search` (`transformation.py:3127`), because Anthropic does not always report it.
    """
    content: Final[list[JsonValue]] = [
        {"type": "server_tool_use", "id": "srvtoolu_dec1ts", "name": "tool_search_tool_regex", "input": {"q": "w"}},
        {
            "type": "tool_search_tool_result",
            "tool_use_id": "srvtoolu_dec1ts",
            "content": {"type": "tool_search_tool_search_result", "tool_references": []},
        },
        {"type": "text", "text": "none found"},
    ]
    usage: Final = usage_of(_answer(gateway, content))
    assert object_value(usage["server_tool_use"])["tool_search_requests"] == 1


def test_inference_geo_is_returned_on_the_chat_usage(gateway: Gateway) -> None:
    """The provider reports `inference_geo: "us"` in usage.

    The caller's `usage.inference_geo` must be `us` (`calculate_usage`, `transformation.py:3093`).
    """
    usage: Final = usage_of(_answer(gateway, usage=usage_block(inference_geo="us")))
    assert usage.get("inference_geo") == "us"
