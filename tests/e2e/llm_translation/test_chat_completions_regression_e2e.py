"""Live regression net for /chat/completions across the configured providers.

GH #28991 broke /chat/completions (and /responses) for most models on some
releases: a clean 200 came back but with no real completion. A status check
alone would not have caught it, so each case here asserts the product promise -
a non-empty assistant message and a real model name in the body - across the
providers wired into the gateway config (OpenAI, Anthropic, Gemini, Azure
OpenAI). A regression that empties the completion for any provider fails that
provider's row here. The Azure OpenAI streaming case applies the same standard
to the SSE path: every data event must parse as a chat.completion.chunk and the
deltas must reassemble into real text, not just count as a 200 with chunks.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from models import ChatBody, ChatMessage, ChatStreamChunk
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

AZURE_CHAT_MODEL = "azure-gpt-5.4-mini"

CHAT_MODELS: tuple[tuple[str, str], ...] = (
    ("gpt-5.5", "openai"),
    ("claude-haiku-4-5", "anthropic"),
    ("gemini-2.5-flash", "gemini"),
    (AZURE_CHAT_MODEL, "azure_openai"),
)


class TestChatCompletionsRegression:
    @pytest.mark.parametrize(
        ("model", "route"),
        CHAT_MODELS,
        ids=[f"{model}-{route}" for model, route in CHAT_MODELS],
    )
    @pytest.mark.covers(
        "llm.chat_completions.openai.basic.nonstream.works",
        "llm.chat_completions.anthropic.basic.nonstream.works",
        "llm.chat_completions.vertex.basic.nonstream.works",
        "llm.chat_completions.azure_openai.basic.nonstream.works",
        exercised_on=[],
    )
    def test_chat_returns_real_completion(
        self, client: PassthroughClient, scoped_key: str, model: str, route: str
    ) -> None:
        response = unwrap(
            client.gateway.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"reply with one word {unique_marker()}",
                        )
                    ],
                    max_tokens=512,
                ),
            )
        )

        assert (
            response.model
        ), f"{model} ({route}): response carried no model name: {response}"
        assert (
            response.choices
        ), f"{model} ({route}): response had no choices: {response}"
        message = response.choices[0].message
        assert (
            message is not None and message.content and message.content.strip()
        ), f"{model} ({route}): 200 with an empty completion (#28991): {response}"

    @pytest.mark.covers("llm.chat_completions.azure_openai.basic.stream.works")
    def test_azure_openai_stream_returns_real_completion(
        self, client: PassthroughClient, scoped_key: str
    ) -> None:
        result = client.gateway.chat_stream(
            scoped_key,
            ChatBody(
                model=AZURE_CHAT_MODEL,
                messages=[
                    ChatMessage(
                        role="user",
                        content=f"reply with one word {unique_marker()}",
                    )
                ],
                max_tokens=512,
                stream=True,
            ),
        )

        assert result.ok, (
            f"{AZURE_CHAT_MODEL}: stream failed with status "
            f"{result.status_code}: {result.body[:300]}"
        )
        assert result.is_streaming, (
            f"{AZURE_CHAT_MODEL}: expected text/event-stream, got "
            f"{result.content_type}: {result.body[:300]}"
        )
        assert result.stream_error is None, (
            f"{AZURE_CHAT_MODEL}: 200 stream carried an error event: "
            f"{result.stream_error}"
        )
        assert result.events, f"{AZURE_CHAT_MODEL}: stream carried no SSE data events"
        assert result.events[-1] == "[DONE]", (
            f"{AZURE_CHAT_MODEL}: stream did not terminate with [DONE]: "
            f"{result.events[-1][:200]}"
        )

        chunks = [
            ChatStreamChunk.model_validate_json(event) for event in result.events[:-1]
        ]
        assert chunks, f"{AZURE_CHAT_MODEL}: stream held only the [DONE] sentinel"
        assert all(
            chunk.object == "chat.completion.chunk" for chunk in chunks
        ), f"{AZURE_CHAT_MODEL}: malformed chunk object types: {result.events[:5]}"
        assert any(
            chunk.model for chunk in chunks
        ), f"{AZURE_CHAT_MODEL}: no chunk carried a model name: {result.events[:5]}"

        content = "".join(
            choice.delta.content or ""
            for chunk in chunks
            for choice in chunk.choices
            if choice.delta is not None
        )
        assert content.strip(), (
            f"{AZURE_CHAT_MODEL}: stream chunks reassembled to an empty "
            f"completion (#28991): {result.events[:5]}"
        )
