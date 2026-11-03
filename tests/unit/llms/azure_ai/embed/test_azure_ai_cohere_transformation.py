import json
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.azure_ai.embed.cohere_transformation import AzureAICohereConfig
from litellm.types.utils import EmbeddingResponse, Usage


@pytest.fixture
def native_azure_embedding_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clients: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", clients)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "api_key", None)
    for key in ("AZURE_AI_API_KEY", "AZURE_API_KEY", "AZURE_AD_TOKEN", "AZURE_CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "unrelated-cohere-key")
    yield
    clients.flush_cache()


@pytest.mark.usefixtures("native_azure_embedding_transport", "local_model_cost_map")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("model", ["Cohere-Embed-V5-Fast", "Cohere-Embed-V5-Pro"])
@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.parametrize("auth", [{"api_key": "azure-key"}, {"azure_ad_token": "entra-token"}])
@pytest.mark.parametrize(("text_tokens", "image_tokens"), [(5, 0), (0, 6), (3, 6)])
async def test_native_foundry_embeddings_keep_azure_auth_and_price_billed_usage(
    respx_mock: respx.MockRouter,
    model: str,
    sync_mode: bool,
    auth: dict[str, str],
    text_tokens: int,
    image_tokens: int,
) -> None:
    route: Final = respx_mock.post("https://foundry.services.ai.azure.com/providers/cohere/v2/embed").mock(
        return_value=httpx.Response(
            200,
            json={
                "embeddings": {"float": [[0.125, -0.375]]},
                "meta": {"billed_units": {"input_tokens": text_tokens, "image_tokens": image_tokens}},
            },
        )
    )
    input_value: Final = "data:image/png;base64,aGVsbG8=" if image_tokens else "hello world"
    params: Final = {
        "model": f"azure_ai/{model}",
        "api_base": "https://foundry.services.ai.azure.com",
        "input": [input_value],
        "encoding_format": "float",
        "dimensions": 256,
        **auth,
    }

    response: Final = litellm.embedding(**params) if sync_mode else await litellm.aembedding(**params)

    assert json.loads(route.calls.last.request.content) == {
        "model": model,
        "images" if image_tokens else "texts": [input_value],
        "input_type": "image" if image_tokens else litellm.COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
        "embedding_types": ["float"],
        "output_dimension": 256,
    }
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {next(iter(auth.values()))}"
    assert response.data == [{"object": "embedding", "index": 0, "embedding": [0.125, -0.375]}]
    assert response.usage is not None
    assert response.usage.prompt_tokens == text_tokens + image_tokens
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.text_tokens == text_tokens
    assert response.usage.prompt_tokens_details.image_tokens == (image_tokens or None)
    prices: Final = litellm.get_model_info(f"azure_ai/{model}")
    assert response._hidden_params["custom_llm_provider"] == "azure_ai"
    assert response._hidden_params["response_cost"] == pytest.approx(
        text_tokens * prices["input_cost_per_token"] + image_tokens * prices["input_cost_per_image_token"]
    )


@pytest.mark.usefixtures("native_azure_embedding_transport")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize(
    ("model", "base_path"),
    [
        ("Cohere-Embed-V5-Fast", "/models"),
        ("Cohere-Embed-V5-Fast", "/"),
        ("custom-embed-deployment", "/providers/cohere"),
        ("custom-embed-deployment", "/providers/cohere/v2/"),
        ("custom-embed-deployment", "/providers/cohere/v2/embed/"),
    ],
)
async def test_native_foundry_embeddings_accept_deployment_names_and_endpoint_forms(
    respx_mock: respx.MockRouter, model: str, base_path: str
) -> None:
    route: Final = respx_mock.post(
        "https://foundry.services.ai.azure.com/providers/cohere/v2/embed?api-version=preview"
    ).mock(return_value=httpx.Response(200, json={"embeddings": {"float": [[0.5]]}}))

    response: Final = await litellm.aembedding(
        model=f"azure_ai/{model}",
        input=["query"],
        api_base=f"https://foundry.services.ai.azure.com{base_path}?api-version=preview",
        api_key="azure-key",
        input_type="search_query",
        dimensions=256,
    )

    assert json.loads(route.calls.last.request.content) == {
        "model": model,
        "texts": ["query"],
        "input_type": "search_query",
        "output_dimension": 256,
    }
    assert response.data == [{"object": "embedding", "index": 0, "embedding": [0.5]}]


def test_azure_ai_cohere_request_splits_images_from_text_and_keeps_order():
    config: Final = AzureAICohereConfig()
    image_request, text_request, image_indexes = config.transform_request(
        input=[
            "describe this image",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
        ],
        optional_params={},
        model="Cohere-embed-v3-multilingual",
    )

    assert image_request == {"input": [{"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}]}
    assert text_request == {"input": ["describe this image"], "model": "Cohere-embed-v3-multilingual"}
    assert image_indexes == [1]


def test_azure_ai_cohere_response_maps_model_group_and_usage_headers():
    config: Final = AzureAICohereConfig()
    response: Final = EmbeddingResponse(
        model="",
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        data=[],
        hidden_params={
            "additional_headers": {
                "llm_provider-num_tokens": "7",
                "llm_provider-azureml-model-group": "offer-cohere-embed-multili-paygo",
            }
        },
    )

    result: Final = config.transform_response(response)

    assert result.model == "Cohere-embed-v3-multilingual"
    assert result.usage.prompt_tokens == 7
