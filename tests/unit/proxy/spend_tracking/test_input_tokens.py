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


_COMPACTION_ITEM: Final = {
    "type": "compaction",
    "id": "cmp_1",
    "encrypted_content": json.dumps({"type": "compaction", "content": "Turn one planned Tokyo."}),
}
_TURN_AFTER_COMPACTION: Final = {"role": "user", "content": "And Osaka?"}


@pytest.mark.parametrize("model", [CL100K_MODEL, "claude-sonnet-5-5"])
@pytest.mark.asyncio
async def test_history_before_a_compaction_item_is_not_counted(model: str) -> None:
    compacted_body: Final = {"model": model, "input": [_COMPACTION_ITEM, _TURN_AFTER_COMPACTION]}
    full_body: Final = {
        "model": model,
        "input": [{"role": "user", "content": "Plan a Tokyo trip. " * 500}, _COMPACTION_ITEM, _TURN_AFTER_COMPACTION],
    }

    full_counts: Final = await count_input_tokens(
        request_body=full_body, raw_body=json.dumps(full_body).encode(), models=(model,)
    )
    compacted_counts: Final = await count_input_tokens(
        request_body=compacted_body, raw_body=json.dumps(compacted_body).encode(), models=(model,)
    )

    assert full_counts[model] == compacted_counts[model]
    assert count_input_tokens_for_model(request_body=full_body, model=model) == compacted_counts[model]
