"""Tests for input-token counting shared across the reservation path's models."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.spend_tracking.input_tokens import (
    TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS,
    count_input_tokens,
    count_input_tokens_for_model,
)

CL100K_MODEL: Final = "gpt-4"


@pytest.mark.asyncio
async def test_large_input_is_still_counted() -> None:
    request_body: Final = {
        "model": CL100K_MODEL,
        "messages": [{"role": "user", "content": "x" * (TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS + 1)}],
    }

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=None, models=(CL100K_MODEL,))

    assert counts[CL100K_MODEL] == count_input_tokens_for_model(request_body=request_body, model=CL100K_MODEL)
    assert isinstance(counts, MappingProxyType)


def test_count_input_tokens_for_model_counts_anthropic_auto_tool_choice() -> None:
    model: Final = "claude-sonnet-5-5"
    body_without_tool_choice: Final = {
        "model": model,
        "messages": [{"role": "user", "content": "weather in pune?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
    }
    body_with_tool_choice: Final = body_without_tool_choice | {"tool_choice": {"type": "auto"}}

    expected: Final = count_input_tokens_for_model(request_body=body_without_tool_choice, model=model)
    actual: Final = count_input_tokens_for_model(request_body=body_with_tool_choice, model=model)

    assert isinstance(expected, int)
    assert actual == expected


_COMPACTION_ITEM: Final = {
    "type": "compaction",
    "id": "cmp_1",
    "encrypted_content": json.dumps({"type": "compaction", "content": "Turn one planned Tokyo."}),
}
_TURN_AFTER_COMPACTION: Final = {"role": "user", "content": "And Osaka?"}


@pytest.mark.parametrize(
    "earlier_item",
    [
        pytest.param({"role": "user", "content": "Plan a Tokyo trip. " * 500}, id="user_turn"),
        pytest.param({"role": ["system"], "content": "Plan a Tokyo trip."}, id="list_role"),
        pytest.param({"role": {"name": "system"}, "content": "Plan a Tokyo trip."}, id="dict_role"),
    ],
)
@pytest.mark.parametrize("model", [CL100K_MODEL, "claude-sonnet-5-5"])
@pytest.mark.asyncio
async def test_history_before_a_compaction_item_is_not_counted(model: str, earlier_item: dict[str, object]) -> None:
    compacted_body: Final = {"model": model, "input": [_COMPACTION_ITEM, _TURN_AFTER_COMPACTION]}
    full_body: Final = {"model": model, "input": [earlier_item, _COMPACTION_ITEM, _TURN_AFTER_COMPACTION]}

    full_counts: Final = await count_input_tokens(
        request_body=full_body, raw_body=json.dumps(full_body).encode(), models=(model,)
    )
    compacted_counts: Final = await count_input_tokens(
        request_body=compacted_body, raw_body=json.dumps(compacted_body).encode(), models=(model,)
    )

    assert full_counts[model] == compacted_counts[model]
    assert count_input_tokens_for_model(request_body=full_body, model=model) == compacted_counts[model]


def _compaction_item_carrying(extra_block_fields: dict[str, str]) -> dict[str, str]:
    return {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": json.dumps(
            {"type": "compaction", "content": "Turn one planned Tokyo.", **extra_block_fields}
        ),
    }


@pytest.mark.parametrize("model", [CL100K_MODEL, "claude-sonnet-5-5"])
def test_a_compaction_item_counts_only_its_summary(model: str) -> None:
    plain_body: Final = {"model": model, "input": [_compaction_item_carrying({}), _TURN_AFTER_COMPACTION]}
    padded_body: Final = {
        "model": model,
        "input": [_compaction_item_carrying({"opaque_state": "x" * 4000}), _TURN_AFTER_COMPACTION],
    }

    assert count_input_tokens_for_model(request_body=padded_body, model=model) == count_input_tokens_for_model(
        request_body=plain_body, model=model
    )
