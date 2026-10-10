"""Tests for input-token counting shared across the reservation path's models."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Final

import pytest

import litellm
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


def _responses_body(image_url: str) -> dict:
    return {
        "model": CL100K_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this screenshot?"},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "image_url",
    (
        "data:image/png;base64," + "iVBORw0KGgo" * 100_000,
        "data:image/jpeg;base64," + "/9j/4AAQSkZJRg" * 100_000,
    ),
    ids=("png", "jpeg"),
)
def test_inline_base64_payload_is_not_counted_as_text(image_url: str) -> None:
    """A 1 MB inline image adds no more to the reservation than its `data:` prefix."""
    prefix: Final = image_url[: image_url.index(",") + 1]

    counted: Final = count_input_tokens_for_model(request_body=_responses_body(image_url), model=CL100K_MODEL)

    assert counted == count_input_tokens_for_model(request_body=_responses_body(prefix), model=CL100K_MODEL)
    assert counted is not None and counted < 100


@pytest.mark.parametrize(
    "text",
    (
        "https://example.com/cat.png",
        "data:text/plain,not base64 so it is text",
        "see data:image/png;base64,AAAA in the middle",
    ),
    ids=("remote_url", "data_url_without_base64", "data_url_not_at_start"),
)
def test_strings_that_are_not_inline_base64_are_counted_in_full(text: str) -> None:
    item: Final = {"text": text}

    counted: Final = count_input_tokens_for_model(request_body={"input": [item, text]}, model=CL100K_MODEL)

    assert counted == litellm.token_counter(model=CL100K_MODEL, text=json.dumps(item)) + litellm.token_counter(
        model=CL100K_MODEL, text=text
    )
