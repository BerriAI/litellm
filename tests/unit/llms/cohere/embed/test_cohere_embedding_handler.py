from typing import Final

import httpx
import pytest
import respx

import litellm

BEDROCK_COHERE_INVOKE_URL: Final = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/cohere.embed-english-v3/invoke"
)
COHERE_EMBED_RESPONSE: Final = {"embeddings": [[0.1, 0.2]], "meta": {"billed_units": {"input_tokens": 1}}}


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_bedrock_cohere_embedding_sends_the_caller_timeout(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, sync_mode: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(BEDROCK_COHERE_INVOKE_URL).mock(
        return_value=httpx.Response(200, json=COHERE_EMBED_RESPONSE)
    )
    request: Final = {
        "model": "bedrock/cohere.embed-english-v3",
        "input": ["hello"],
        "timeout": 45,
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "mockaws_access_key_id",
        "aws_secret_access_key": "mockaws_secret_access_key",
    }

    response: Final = litellm.embedding(**request) if sync_mode else await litellm.aembedding(**request)

    assert route.calls.last.request.extensions["timeout"]["read"] == 45
    assert response.data[0]["embedding"] == [0.1, 0.2]

COHERE_DEFAULT_EMBED_URL: Final = "https://api.cohere.ai/v2/embed"


async def test_cohere_embedding_honors_cohere_api_base_env(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`COHERE_API_BASE` must be honored for embeddings, like the rerank path and
    the databricks/hosted_vllm embedding branches already do (#49411 family)."""
    gateway: Final = "https://gateway.example.com/v1/embed"
    monkeypatch.setattr(litellm, 'disable_aiohttp_transport', True)
    monkeypatch.setenv("COHERE_API_BASE", gateway)
    route: Final = respx_mock.post(gateway).mock(
        return_value=httpx.Response(200, json=COHERE_EMBED_RESPONSE)
    )

    response: Final = await litellm.aembedding(
        model="cohere/embed-english-v3.0", input=["hello"], api_key="mock-key"
    )

    assert route.calls.call_count == 1
    assert response.data[0]["embedding"] == [0.1, 0.2]


async def test_cohere_embedding_keeps_the_default_url_without_the_env(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, 'disable_aiohttp_transport', True)
    monkeypatch.delenv("COHERE_API_BASE", raising=False)
    route: Final = respx_mock.post(COHERE_DEFAULT_EMBED_URL).mock(
        return_value=httpx.Response(200, json=COHERE_EMBED_RESPONSE)
    )

    response: Final = await litellm.aembedding(
        model="cohere/embed-english-v3.0", input=["hello"], api_key="mock-key"
    )

    assert route.calls.call_count == 1
    assert response.data[0]["embedding"] == [0.1, 0.2]
