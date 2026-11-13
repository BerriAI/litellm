import json
from typing import Final

import httpx
import pytest
import respx

import litellm


def test_mistral_embedding_uses_mistral_embeddings_endpoint(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.mistral.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "embd-migration",
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "mistral-embed",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )

    response: Final = litellm.embedding(
        model="mistral/mistral-embed",
        input=["hello"],
        api_key="test-mistral-key",
    )

    assert str(route.calls.last.request.url) == "https://api.mistral.ai/v1/embeddings"
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-mistral-key"
    assert json.loads(route.calls.last.request.read()) == {"model": "mistral-embed", "input": ["hello"]}
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (2, 2)


@pytest.mark.parametrize(
    "metadata,proxy_request,expected",
    [
        ({"label": "qa", "nested": {"id": 7}}, False, {"label": "qa", "nested": {"id": 7}}),
        ({}, False, {}),
        (None, False, None),
        (
            {
                "user_api_key": "internal",
                "requester_metadata": {
                    "label": "qa",
                    "headers": {"authorization": "internal"},
                    "used_client_oauth_token": False,
                },
            },
            True,
            {"label": "qa"},
        ),
        ({"user_api_key": "internal"}, True, None),
    ],
)
def test_mistral_embedding_forwards_only_requester_metadata(
    respx_mock: respx.MockRouter,
    metadata: dict[str, object] | None,
    proxy_request: bool,
    expected: dict[str, object] | None,
) -> None:
    route: Final = respx_mock.post("https://api.mistral.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "mistral-embed",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )
    litellm.embedding(
        model="mistral/mistral-embed",
        input=["hello"],
        api_key="test-mistral-key",
        metadata=metadata,
        litellm_metadata={"user_api_key": "internal"},
        **({"proxy_server_request": {"url": "http://localhost/v1/embeddings"}} if proxy_request else {}),
    )
    assert json.loads(route.calls.last.request.read()) == {
        "model": "mistral-embed",
        "input": ["hello"],
        **({"metadata": expected} if expected is not None else {}),
    }
