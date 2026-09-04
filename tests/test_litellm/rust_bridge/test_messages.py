from __future__ import annotations

from collections.abc import Generator
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages_handler,
)
from litellm.rust_bridge import messages as bridge


@pytest.fixture(autouse=True)
def reset_bridge() -> Generator[None]:
    bridge.set_rust_messages(messages=None, amessages=None)
    yield
    bridge.set_rust_messages(messages=None, amessages=None)


@pytest.mark.asyncio
async def test_public_messages_dispatches_before_provider_transformation() -> None:
    calls: list[dict[str, object]] = []

    async def native(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hello from rust"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    bridge.set_rust_messages(amessages=native)
    pending: Final = anthropic_messages_handler(
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-sonnet-4-5",
        is_async=True,
        rust=True,
        litellm_logging_obj=MagicMock(),
    )
    result: Final = await pending

    assert result["content"][0]["text"] == "hello from rust"
    assert len(calls) == 1
    assert calls[0]["api_base"] is None
    assert calls[0]["body"] == {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "stream": False,
    }
