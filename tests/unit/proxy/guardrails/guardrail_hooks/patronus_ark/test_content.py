from typing import Final

import pytest
from pydantic import JsonValue

from litellm.proxy.guardrails.guardrail_hooks.patronus_ark.content import rewrite_message
from litellm.proxy.guardrails.guardrail_hooks.patronus_ark.patronus_ark import _redact
from litellm.types.proxy.guardrails.guardrail_hooks.patronus_ark import Source


@pytest.mark.parametrize(
    "offsets, expected",
    [
        ((), "abcdefghij"),
        (((2, 6), (4, 8)), "ab[REDACTED]ij"),
        (((2, 8), (3, 5)), "ab[REDACTED]ij"),
        (((4, 6), (2, 4), (2, 4)), "ab[REDACTED]ghij"),
        (((7, 9), (1, 3)), "a[REDACTED]defg[REDACTED]j"),
    ],
)
def test_redaction_merges_overlapping_adjacent_and_unsorted_evidence(offsets, expected) -> None:
    spans: Final = tuple({"start_char": start, "end_char": end} for start, end in offsets)
    assert _redact("abcdefghij", spans) == expected


@pytest.mark.parametrize("start, end", [(-1, 2), (1, 1), (4, 2), (0, 11)])
def test_invalid_evidence_cannot_rewrite_unrelated_text(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="invalid evidence offsets"):
        _redact("abcdefghij", ({"start_char": start, "end_char": end},))


async def mask_secret(text: str, source: Source, pi_only: bool) -> str:
    return text if pi_only else text.replace("secret", "[REDACTED]")


async def test_tool_json_redacts_keys_values_and_numbers_without_changing_metadata() -> None:
    async def visit(text: str, source: Source, pi_only: bool) -> str:
        assert source == "tool_call"
        return text if pi_only else text.replace("secret", "[REDACTED]").replace("12345", "[REDACTED]")

    result: Final = await rewrite_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"secret":["secret",12345,true,null],"nested":{"safe":2}}',
                    },
                }
            ],
        },
        visit,
    )
    assert result == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"[REDACTED]":["[REDACTED]","[REDACTED]",true,null],"nested":{"safe":2}}',
                },
            }
        ],
    }


@pytest.mark.parametrize("arguments", ['{"secret":1,"[REDACTED]":2}', '{"same":1,"same":2}'])
async def test_duplicate_tool_keys_are_refused_before_a_value_can_disappear(arguments: str) -> None:
    with pytest.raises(ValueError, match="uplicate"):
        await rewrite_message(
            {"role": "assistant", "content": None, "function_call": {"name": "lookup", "arguments": arguments}},
            mask_secret,
        )


async def test_non_text_parts_and_cache_control_survive_redaction() -> None:
    image: Final = {"type": "image_url", "image_url": {"url": "https://example.org/secret.png"}}
    result: Final = await rewrite_message(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "secret", "cache_control": {"type": "ephemeral"}},
                image,
            ],
        },
        mask_secret,
    )
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "[REDACTED]", "cache_control": {"type": "ephemeral"}},
            image,
        ],
    }


@pytest.mark.parametrize(
    "message, error, reason",
    [
        (None, TypeError, "Malformed chat message"),
        ({"role": "unknown", "content": "secret"}, ValueError, "Unsupported chat role"),
        ({"role": "user", "content": [{"type": "text", "text": 42}]}, TypeError, "Malformed text part"),
        ({"role": "assistant", "tool_calls": {}}, ValueError, "Malformed tool calls"),
        ({"role": "assistant", "tool_calls": [{"type": "custom"}]}, ValueError, "Unsupported tool call"),
        ({"role": "assistant", "function_call": "invalid"}, TypeError, "Malformed tool function"),
        (
            {"role": "assistant", "function_call": {"name": "lookup", "arguments": {}}},
            TypeError,
            "Tool arguments must be a JSON string",
        ),
    ],
)
async def test_malformed_messages_are_rejected_before_rewriting(
    message: JsonValue, error: type[Exception], reason: str
) -> None:
    with pytest.raises(error, match=reason):
        await rewrite_message(message, mask_secret)


async def test_null_user_content_survives_without_becoming_text() -> None:
    assert await rewrite_message({"role": "user", "content": None}, mask_secret) == {
        "role": "user",
        "content": None,
    }
