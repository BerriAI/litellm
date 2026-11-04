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
