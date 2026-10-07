"""
A compacted history on the Vertex AI partner count-tokens endpoint needs the compaction beta: Vertex
validates the signed compaction block exactly like Anthropic and answers 400 without the header, which
the proxy then degrades into the local tokenizer.
"""

import pytest

import litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler as handler_mod
from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
    VertexAIPartnerModelsTokenCounter,
)
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES

COMPACTION_BETA = ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_09_04.value
COMPACTED_HISTORY = [
    {
        "role": "assistant",
        "content": [
            {
                "type": "compaction",
                "content": "The user is building a recipe app and asked for one-sentence class descriptions.",
                "signature": "EqQBCkYIBRgCKkBjZ2xhc3M" * 40,
            }
        ],
    },
    {"role": "user", "content": "Now do the same for Ingredient."},
]
PLAIN_HISTORY = [{"role": "user", "content": "Now do the same for Ingredient."}]


@pytest.fixture
def posted_headers(monkeypatch):
    captured = {}

    async def fake_ensure_access_token(self, credentials, project_id, custom_llm_provider):
        return "fake-token", "fake-project"

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"input_tokens": 404}

    class FakeClient:
        async def post(self, url, headers=None, json=None, **kwargs):
            captured.update(headers or {})
            return FakeResponse()

    monkeypatch.setattr(VertexAIPartnerModelsTokenCounter, "_ensure_access_token_async", fake_ensure_access_token)
    monkeypatch.setattr(handler_mod, "get_async_httpx_client", lambda **kwargs: FakeClient())
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "beta_expected"),
    [(COMPACTED_HISTORY, COMPACTION_BETA), (PLAIN_HISTORY, None)],
)
async def test_compaction_beta_is_sent_only_with_a_compacted_history(posted_headers, messages, beta_expected):
    result = await VertexAIPartnerModelsTokenCounter().handle_count_tokens_request(
        model="claude-sonnet-5-5",
        request_data={"model": "claude-sonnet-5-5", "messages": messages},
        litellm_params={"vertex_location": "us-east5"},
    )

    assert result["input_tokens"] == 404
    assert posted_headers["Authorization"] == "Bearer fake-token"
    assert posted_headers.get("anthropic-beta") == beta_expected
