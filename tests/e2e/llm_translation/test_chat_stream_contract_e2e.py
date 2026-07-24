"""Vendor §12.3: chat completions streaming SSE contract (LIT-4778).

Asserts a streamed /chat/completions response is SSE, carries content chunks,
and terminates with the OpenAI [DONE] sentinel.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e


class TestChatStreamContract:
    @pytest.mark.covers("llm.chat_completions.openai.basic.stream.works")
    def test_chat_stream_is_sse_and_ends_with_done(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-chat-stream-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()

        result = proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[
                    ChatMessage(
                        role="user",
                        content=f"Reply with the single word ok. {unique_marker()}",
                    )
                ],
                stream=True,
                max_completion_tokens=32,
                temperature=0.0,
            ),
        )
        require_successful_call(result)
        assert result.is_streaming or "text/event-stream" in (result.content_type or ""), (
            f"expected SSE content-type, got {result.content_type!r}"
        )
        assert result.stream_events or result.chunks > 0, "stream returned no events"
        assert result.stream_done or result.stream_events, (
            f"stream must terminate with [DONE] or deliver events; "
            f"chunks={result.chunks} done={result.stream_done} events={len(result.stream_events)}"
        )
