"""Eden AI `/v3/embeddings`: OpenAI's embeddings API served by Eden's gateway, which reports the
real per-request cost at the top level of the body."""

import json

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.edenai.embedding.transformation import EdenAIEmbeddingConfig
from litellm.types.utils import EmbeddingResponse, LlmProviders
from litellm.utils import ProviderConfigManager

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_EMBEDDINGS_URL = f"{EDEN_BASE}/embeddings"
EDEN_REPORTED_COST = 0.0042
MODEL = "edenai/openai/text-embedding-3-small"
SELLER_MODEL = "openai/text-embedding-3-small"
VECTOR = [0.016754150390625, -0.055755615234375]


def _eden_embedding(cost: float | None = EDEN_REPORTED_COST) -> dict:
    """Live `/v3/embeddings` body: OpenAI shape plus Eden's top-level `cost`, `provider` and `status`."""
    body = {
        "status": "success",
        "model": "text-embedding-3-small",
        "data": [{"embedding": VECTOR, "index": 0, "object": "embedding"}],
        "object": "list",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "provider": "openai",
    }
    return body if cost is None else {**body, "cost": cost}


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


class TestRegistration:
    def test_eden_is_a_native_embedding_provider(self):
        config = ProviderConfigManager.get_provider_embedding_config(model=SELLER_MODEL, provider=LlmProviders.EDENAI)

        assert isinstance(config, EdenAIEmbeddingConfig)


class TestAuthentication:
    def test_missing_key_is_an_authentication_error_before_any_request(self, no_eden_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            litellm.embedding(model=MODEL, input="hello")
        assert not respx_mock.calls


class TestEmbedding:
    def test_posts_to_eden_with_the_bearer_key_and_the_seller_model_id(self, eden_key, respx_mock):
        respx_mock.post(EDEN_EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=_eden_embedding()))

        response = litellm.embedding(model=MODEL, input="hello", dimensions=2)

        assert isinstance(response, EmbeddingResponse)
        assert response.data[0]["embedding"] == VECTOR
        request = respx_mock.calls.last.request
        assert request.headers["Authorization"] == f"Bearer {eden_key}"
        assert request.headers["Content-Type"] == "application/json"
        body = _request_body(respx_mock)
        assert (body["model"], body["input"], body["dimensions"]) == (SELLER_MODEL, "hello", 2)

    def test_eden_reported_cost_beats_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=_eden_embedding()))

        response = litellm.embedding(model=MODEL, input="hello")

        assert get_response_cost_from_hidden_params(response._hidden_params) == EDEN_REPORTED_COST
        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST

    def test_a_body_without_cost_leaves_pricing_to_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=_eden_embedding(cost=None)))

        response = litellm.embedding(model=MODEL, input="hello")

        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    def test_extra_body_forwards_eden_only_fields(self, eden_key, respx_mock):
        respx_mock.post(EDEN_EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=_eden_embedding()))

        litellm.embedding(model=MODEL, input="hello", extra_body={"metadata": {"trace": "abc"}})

        assert _request_body(respx_mock)["metadata"] == {"trace": "abc"}

    @pytest.mark.asyncio
    async def test_async_call_tracks_the_same_cost(self, eden_key, httpx_transport, respx_mock):
        respx_mock.post(EDEN_EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=_eden_embedding()))

        response = await litellm.aembedding(model=MODEL, input=["hello", "world"])

        assert _request_body(respx_mock)["input"] == ["hello", "world"]
        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST


class TestErrors:
    def test_middleware_401_maps_to_authentication_error(self, eden_key, respx_mock):
        respx_mock.post(EDEN_EMBEDDINGS_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token."}))

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            litellm.embedding(model=MODEL, input="hello")

    def test_429_maps_to_rate_limit_error(self, eden_key, respx_mock):
        respx_mock.post(EDEN_EMBEDDINGS_URL).mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limit exceeded", "type": "rate_limit"}})
        )

        with pytest.raises(litellm.RateLimitError):
            litellm.embedding(model=MODEL, input="hello")
