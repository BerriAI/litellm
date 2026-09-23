from __future__ import annotations

from typing import Final

import pytest
from e2e_config import provider_edge_base, unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatStreamOptions, LiteLLMParamsBody, Usage
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.replayable]


class _Delta(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    index: int
    delta: _Delta
    finish_reason: str | None = None


class _Chunk(BaseModel):
    choices: tuple[_Choice, ...]
    usage: Usage | None = None


class TestChatStreamContract:
    @pytest.mark.covers("llm.chat_completions.openai.basic.stream.works")
    def test_chat_stream_is_sse_and_ends_with_done(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model: Final = f"e2e-chat-stream-{unique_marker()}"
        base: Final = provider_edge_base("openai")
        model_id: Final = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-5.6",
                api_key="os.environ/OPENAI_API_KEY",
                api_base=f"{base}/v1" if base else None,
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key: Final = resources.key()
        expected: Final = "The amber kite crosses the quiet lake."
        result: Final = proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[
                    ChatMessage(
                        role="user", content=f"Repeat exactly this sentence, with no additional text: {expected}"
                    )
                ],
                stream=True,
                stream_options=ChatStreamOptions(include_usage=True),
                max_completion_tokens=256,
                reasoning_effort="none",
            ),
        )
        require_successful_call(result)
        assert result.is_streaming, f"expected SSE content-type, got {result.content_type!r}"
        assert result.stream_events, "stream returned no data events"
        assert not result.stream_error, f"stream errored: {result.stream_error}"
        assert result.stream_done, "stream must terminate with [DONE]"
        assert result.stream_done_positions == (len(result.stream_events),), "[DONE] must occur once after all events"
        chunks: Final = tuple(_Chunk.model_validate_json(event) for event in result.stream_events)
        text_positions: Final = tuple(
            i for i, chunk in enumerate(chunks) if any(c.delta.content for c in chunk.choices)
        )
        terminal_positions: Final = tuple(
            i for i, chunk in enumerate(chunks) if any(c.finish_reason is not None for c in chunk.choices)
        )
        assert text_positions, "stream completed without meaningful text"
        assert len(terminal_positions) == 1, "expected exactly one terminal choice"
        assert text_positions[0] < terminal_positions[0], "meaningful text must arrive before termination"
        assert text_positions[-1] <= terminal_positions[0], "text arrived after termination"
        assert all(c.index == 0 for chunk in chunks for c in chunk.choices)
        assert tuple(c.finish_reason for c in chunks[terminal_positions[0]].choices) == ("stop",)
        text: Final = "".join(c.delta.content or "" for chunk in chunks for c in chunk.choices)
        assert text.strip() == expected, f"streamed answer was altered or incomplete: {text!r}"
        usage_positions: Final = tuple(i for i, chunk in enumerate(chunks) if chunk.usage is not None)
        assert usage_positions == (len(chunks) - 1,), "expected one final usage chunk"
        assert terminal_positions[0] < usage_positions[0], "usage must follow the terminal choice"
        usage: Final = chunks[-1].usage
        assert usage is not None
        assert usage.prompt_tokens is not None and usage.prompt_tokens > 0
        assert usage.completion_tokens is not None and usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
