import json
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_mock
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.embedding.handler import OpenAILikeEmbeddingHandler
from litellm.types.utils import EmbeddingResponse


def _mock_openai_like_embedding_route(
    respx_mock: respx.MockRouter, base_url: str = "https://custom-openai-like.com/v1"
) -> respx.Route:
    return respx_mock.post(f"{base_url}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "custom-model",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )


def test_openai_like_embedding_sends_headers_as_http_headers_not_body(
    respx_mock: respx.MockRouter,
) -> None:
    mock_route: Final = _mock_openai_like_embedding_route(respx_mock)

    response = litellm.embedding(
        model="openai_like/custom-model",
        input=["hello world"],
        api_base="https://custom-openai-like.com/v1",
        api_key="sk-test-key",
        headers={"X-Custom-Header": "custom-value", "X-Trace-Id": "trace-123"},
    )

    assert mock_route.called is True
    last_request: Final = mock_route.calls.last.request
    assert last_request.headers["x-custom-header"] == "custom-value"
    assert last_request.headers["x-trace-id"] == "trace-123"
    assert last_request.headers["authorization"] == "Bearer sk-test-key"

    request_body: Final = json.loads(last_request.read())
    assert "extra_headers" not in request_body
    assert request_body["model"] == "custom-model"
    assert request_body["input"] == ["hello world"]
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_openai_like_embedding_extra_headers_kwarg_merged_into_http_headers(
    respx_mock: respx.MockRouter,
) -> None:
    mock_route: Final = _mock_openai_like_embedding_route(respx_mock)

    response = litellm.embedding(
        model="openai_like/custom-model",
        input=["hello world"],
        api_base="https://custom-openai-like.com/v1",
        api_key="sk-test-key",
        extra_headers={"X-Extra-Header": "extra-val"},
    )

    assert mock_route.called is True
    last_request: Final = mock_route.calls.last.request
    assert last_request.headers["x-extra-header"] == "extra-val"
    assert last_request.headers["authorization"] == "Bearer sk-test-key"

    request_body: Final = json.loads(last_request.read())
    assert "extra_headers" not in request_body
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_openai_like_embedding_both_headers_and_extra_headers_merge(
    respx_mock: respx.MockRouter,
) -> None:
    mock_route: Final = _mock_openai_like_embedding_route(respx_mock)

    response = litellm.embedding(
        model="openai_like/custom-model",
        input=["hello world"],
        api_base="https://custom-openai-like.com/v1",
        api_key="sk-test-key",
        headers={"X-Main-Header": "main-val"},
        extra_headers={"X-Extra-Header": "extra-val"},
    )

    assert mock_route.called is True
    last_request: Final = mock_route.calls.last.request
    assert last_request.headers["x-main-header"] == "main-val"
    assert last_request.headers["x-extra-header"] == "extra-val"

    request_body: Final = json.loads(last_request.read())
    assert "extra_headers" not in request_body
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("provider_prefix", ["llamafile", "lm_studio"])
def test_openai_like_aliases_forward_headers(respx_mock: respx.MockRouter, provider_prefix: str) -> None:
    mock_route: Final = _mock_openai_like_embedding_route(respx_mock)

    litellm.embedding(
        model=f"{provider_prefix}/custom-model",
        input=["testing aliases"],
        api_base="https://custom-openai-like.com/v1",
        api_key="sk-alias-key",
        headers={"X-Alias-Header": "alias-val"},
    )

    assert mock_route.called is True
    last_request: Final = mock_route.calls.last.request
    assert last_request.headers["x-alias-header"] == "alias-val"
    assert last_request.headers["authorization"] == "Bearer sk-alias-key"

    request_body: Final = json.loads(last_request.read())
    assert "extra_headers" not in request_body


def test_openai_like_embedding_empty_headers_works_cleanly(
    respx_mock: respx.MockRouter,
) -> None:
    mock_route: Final = _mock_openai_like_embedding_route(respx_mock)

    response = litellm.embedding(
        model="openai_like/custom-model",
        input=["hello world"],
        api_base="https://custom-openai-like.com/v1",
        api_key="sk-test-key",
        headers={},
    )

    assert mock_route.called is True
    last_request: Final = mock_route.calls.last.request
    assert last_request.headers["authorization"] == "Bearer sk-test-key"

    request_body: Final = json.loads(last_request.read())
    assert "extra_headers" not in request_body
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_direct_handler_pops_extra_headers_from_optional_params(
    respx_mock: respx.MockRouter,
) -> None:
    mock_route: Final = _mock_openai_like_embedding_route(respx_mock)
    handler = OpenAILikeEmbeddingHandler()

    class _StubLogging:
        def pre_call(self, *args, **kwargs):
            pass

        def post_call(self, *args, **kwargs):
            pass

    response = handler.embedding(
        model="custom-model",
        input=["direct call"],
        timeout=10.0,
        logging_obj=_StubLogging(),
        api_key="sk-direct-key",
        api_base="https://custom-openai-like.com/v1",
        optional_params={"extra_headers": {"X-Direct-Header": "direct-val"}, "user": "direct-user"},
        model_response=EmbeddingResponse(),
    )

    assert mock_route.called is True
    last_request: Final = mock_route.calls.last.request
    assert last_request.headers["x-direct-header"] == "direct-val"
    assert last_request.headers["authorization"] == "Bearer sk-direct-key"

    request_body: Final = json.loads(last_request.read())
    assert "extra_headers" not in request_body
    assert request_body["user"] == "direct-user"
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_openai_like_aembedding_sends_headers_as_http_headers_not_body(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": "custom-model",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    mock_post = mocker.patch.object(
        AsyncHTTPHandler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    )

    response = await litellm.aembedding(
        model="openai_like/custom-model",
        input=["hello async world"],
        api_base="https://custom-openai-like.com/v1",
        api_key="sk-test-key",
        headers={"X-Async-Header": "async-value"},
    )

    assert mock_post.called is True
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["X-Async-Header"] == "async-value"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"

    request_body: Final = json.loads(kwargs["data"])
    assert "extra_headers" not in request_body
    assert request_body["model"] == "custom-model"
    assert request_body["input"] == ["hello async world"]
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
