"""
Regression test for https://github.com/BerriAI/litellm/issues/28735

When stream_options={"include_usage": True} is set, the final usage chunk
must have choices: [] per the OpenAI Chat Completions streaming spec.
Previously model_response_creator() was setting a default choice on this
synthetic chunk, violating the spec.
"""
import sys
import time
from typing import Optional

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from litellm.utils import ModelResponseListIterator


def _make_usage_chunk(
    chunk_id: str = "chatcmpl-test",
    model: str = "gpt-4o-mini",
    created: int = 1742056047,
) -> ModelResponseStream:
    """Return a synthetic usage chunk with choices=[] (OpenAI spec compliance)."""
    return ModelResponseStream(
        id=chunk_id,
        created=created,
        model=model,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[],
        provider_specific_fields={},
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        ),
    )


def _make_content_chunk(
    chunk_id: str,
    content: str,
    index: int = 0,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> ModelResponseStream:
    """Return a normal content chunk."""
    choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=index,
            delta=Delta(
                content=content,
                role=role,
            ),
            logprobs=None,
        )
    ]
    return ModelResponseStream(
        id=chunk_id,
        created=1742056047,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=choices,
        provider_specific_fields={},
        usage=None,
    )


@pytest.fixture
def logging_obj():
    return Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "say GM"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
async def test_openai_stream_options_usage_chunk_choices_empty(
    sync_mode: bool, logging_obj: Logging
):
    """
    Verify the usage chunk has choices: [] per the OpenAI spec.

    Uses mocked model responses — no real network calls.
    """
    # Build a realistic stream: content chunks → usage chunk
    chunks = [
        _make_content_chunk("chatcmpl-1", "G", role="assistant"),
        _make_content_chunk("chatcmpl-2", "M"),
        _make_content_chunk("chatcmpl-3", "", finish_reason="stop"),
        _make_usage_chunk("chatcmpl-4"),
    ]

    completion_stream = ModelResponseListIterator(model_responses=chunks)

    wrapper = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="gpt-4o-mini",
        custom_llm_provider="openai",
        logging_obj=logging_obj,
        stream_options={"include_usage": True},
    )

    collected = []
    if sync_mode:
        for chunk in wrapper:
            collected.append(chunk)
    else:
        async for chunk in wrapper:
            collected.append(chunk)

    usage_chunks = [c for c in collected if getattr(c, "usage", None) is not None]
    assert usage_chunks, "No usage chunk found in collected stream"

    last_usage_chunk = usage_chunks[-1]
    assert last_usage_chunk.choices == [], (
        f"Usage chunk choices must be [] but got {last_usage_chunk.choices!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
async def test_usage_chunk_sent_once_per_stream(sync_mode: bool, logging_obj: Logging):
    """
    Ensure the synthetic usage chunk is sent exactly once and no extra chunks
    follow it.
    """
    chunks = [
        _make_content_chunk("chatcmpl-1", "hi", role="assistant"),
        _make_content_chunk("chatcmpl-2", "", finish_reason="stop"),
        _make_usage_chunk("chatcmpl-3"),
    ]

    completion_stream = ModelResponseListIterator(model_responses=chunks)

    wrapper = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="gpt-4o-mini",
        custom_llm_provider="openai",
        logging_obj=logging_obj,
        stream_options={"include_usage": True},
    )

    collected = []
    if sync_mode:
        for chunk in wrapper:
            collected.append(chunk)
    else:
        async for chunk in wrapper:
            collected.append(chunk)

    usage_chunks = [c for c in collected if getattr(c, "usage", None) is not None]
    assert len(usage_chunks) == 1, (
        f"Expected exactly 1 usage chunk, got {len(usage_chunks)}"
    )

    # No content after usage chunk
    last_chunk = collected[-1]
    assert last_chunk.choices == [], "Last chunk should be the usage chunk"