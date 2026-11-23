"""deAPI `/v1/embeddings`: the gateway reports the charge in `usage.cost` and `X-DeAPI-Cost`, and leaves
`prompt_tokens` at 0 because the model behind it does not count tokens."""

import json

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.deapi.embedding.transformation import DeapiEmbeddingConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

DEAPI_EMBEDDINGS_URL = "https://oai.deapi.ai/v1/embeddings"
MODEL = "deapi/Bge_M3_FP16"
SELLER_MODEL = "Bge_M3_FP16"
REPORTED_COST = 0.00000017


def _embedding(cost: float | None = REPORTED_COST) -> httpx.Response:
    usage = {"prompt_tokens": 0, "total_tokens": 0}
    body = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.01, -0.02, 0.03]}],
        "model": SELLER_MODEL,
        "usage": usage if cost is None else {**usage, "cost": cost},
    }
    return httpx.Response(200, json=body, headers={} if cost is None else {"X-DeAPI-Cost": f"{cost:.8f}"})


class TestRegistration:
    def test_deapi_is_a_native_embedding_provider(self):
        config = ProviderConfigManager.get_provider_embedding_config(model=SELLER_MODEL, provider=LlmProviders.DEAPI)

        assert isinstance(config, DeapiEmbeddingConfig)

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_deapi_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="DEAPI_API_KEY"):
            litellm.embedding(model=MODEL, input="hi")
        assert not respx_mock.calls


class TestEmbedding:
    def test_posts_to_deapi_with_the_bearer_key_and_the_model_id(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_EMBEDDINGS_URL).mock(return_value=_embedding())

        response = litellm.embedding(model=MODEL, input=["hi"], encoding_format="float")

        assert response.data[0]["embedding"] == [0.01, -0.02, 0.03]
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {deapi_key}"
        assert json.loads(respx_mock.calls.last.request.content) == {
            "model": SELLER_MODEL,
            "input": ["hi"],
            "encoding_format": "float",
        }

    def test_the_reported_cost_is_the_response_cost(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_EMBEDDINGS_URL).mock(return_value=_embedding())

        response = litellm.embedding(model=MODEL, input="hi")

        assert get_response_cost_from_hidden_params(response._hidden_params) == REPORTED_COST

    def test_an_answer_without_cost_leaves_pricing_to_the_price_map(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_EMBEDDINGS_URL).mock(return_value=_embedding(cost=None))

        response = litellm.embedding(model=MODEL, input="hi")

        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    @pytest.mark.asyncio
    async def test_async_call_logs_the_reported_cost_as_spend(
        self, deapi_key, httpx_transport, spend_capture, respx_mock
    ):
        respx_mock.post(DEAPI_EMBEDDINGS_URL).mock(return_value=_embedding())

        await litellm.aembedding(model=MODEL, input="hi", litellm_call_id=spend_capture.call_id)
        await spend_capture.settle()

        assert spend_capture.costs == [REPORTED_COST]


class TestErrors:
    def test_wrong_dimensions_maps_to_bad_request(self, deapi_key, respx_mock):
        envelope = {
            "error": {
                "message": "Bge_M3_FP16 returns 1024-dimensional vectors.",
                "type": "invalid_request_error",
                "param": "dimensions",
                "code": "dimensions_not_supported",
            }
        }
        respx_mock.post(DEAPI_EMBEDDINGS_URL).mock(return_value=httpx.Response(400, json=envelope))

        with pytest.raises(litellm.BadRequestError, match="1024-dimensional"):
            litellm.embedding(model=MODEL, input="hi", dimensions=256)
