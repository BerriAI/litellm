"""End-to-end (live) tests for the Eden AI provider.

These hit the real Eden AI API and are skipped automatically when
EDENAI_API_KEY is not set, so CI without the secret stays green.
Moved here from tests/test_litellm/llms/openai_like/test_edenai.py per
review feedback: live tests belong in tests/e2e/, not the unit suite.
"""

import os

import pytest

import litellm


class TestEdenAILiveIntegration:
    """Live integration tests against the real Eden AI API.

    Skipped automatically when EDENAI_API_KEY is not set, so CI without the
    secret stays green. When the key is present (local dev, or a dedicated
    integration runner) these validate that each endpoint flagged `true` in
    provider_endpoints_support.json actually routes a request through to
    api.edenai.run/v3 end-to-end. Endpoints flagged `false` (`/responses`,
    `/embeddings`, `/image/generations`, `/audio/*`, `/moderations`,
    `/batches`, `/rerank`) intentionally have no live test here; their
    handlers ship in follow-up PRs and the tests come with them.
    """

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY"), reason="EDENAI_API_KEY not set"
    )
    def test_edenai_live_chat_completions(self):
        """`/v1/chat/completions` via litellm.completion() — the openai_like
        loader's primary surface."""
        response = litellm.completion(
            model="edenai/openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with 'PING' and nothing else.",
                }
            ],
            max_tokens=8,
        )

        assert response is not None
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.total_tokens > 0

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY"), reason="EDENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_edenai_live_anthropic_messages(self):
        """`/v1/messages` (Anthropic-format) via litellm.anthropic_messages().
        LiteLLM translates the Anthropic-shape request to OpenAI shape
        internally and routes through Eden AI's openai-compatible endpoint;
        the response is translated back to Anthropic shape before being
        returned to the caller."""
        response = await litellm.anthropic_messages(
            model="edenai/anthropic/claude-opus-4-6",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with 'MSG' and nothing else.",
                }
            ],
            max_tokens=8,
        )

        assert response is not None
        # anthropic_messages returns a dict (Anthropic-shape response)
        assert response.get("type") == "message"
        assert response.get("role") == "assistant"
        content = response.get("content", [])
        assert content and content[0].get("type") == "text"
        assert content[0].get("text")
        usage = response.get("usage", {})
        assert usage.get("input_tokens", 0) > 0
