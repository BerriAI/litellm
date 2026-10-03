import json
from unittest.mock import MagicMock, Mock

import httpx
import pytest

import litellm
from litellm.llms.cloudflare.chat.transformation import CloudflareError
from litellm.llms.cloudflare.rerank.transformation import CloudflareRerankConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.rerank import RerankResponse
from litellm.utils import ProviderConfigManager


class _RecordingHTTPHandler(HTTPHandler):
    def __init__(self, response):
        super().__init__()
        self.response = response
        self.requests = []

    def post(self, url: str, **kwargs):
        self.requests.append({"url": url, **kwargs})
        return self.response



def test_provider_config_manager_returns_cloudflare_rerank_config():
    config = ProviderConfigManager.get_provider_rerank_config(
        model="@cf/baai/bge-reranker-base",
        provider=litellm.LlmProviders.CLOUDFLARE,
        api_base=None,
        present_version_params=[],
    )

    assert isinstance(config, CloudflareRerankConfig)


def test_get_complete_url_uses_native_workers_ai_endpoint(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    config = CloudflareRerankConfig()

    url = config.get_complete_url(
        api_base=None,
        model="@cf/baai/bge-reranker-base",
    )

    assert url == ("https://api.cloudflare.com/client/v4/accounts/account-id/ai/run/%40cf/baai/bge-reranker-base")


def test_get_complete_url_rewrites_openai_compatible_base():
    config = CloudflareRerankConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/account-id/ai/v1",
        model="@cf/baai/bge-reranker-base",
    )

    assert url.endswith("/accounts/account-id/ai/run/%40cf/baai/bge-reranker-base")


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run/%40cf/baai/bge-reranker-base",
            "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run/%40cf/baai/bge-reranker-base",
        ),
        (
            "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run",
            "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run/%40cf/baai/bge-reranker-base",
        ),
        (
            "https://example.com",
            "https://example.com/ai/run/%40cf/baai/bge-reranker-base",
        ),
    ],
)
def test_get_complete_url_handles_supported_base_shapes(api_base, expected):
    config = CloudflareRerankConfig()

    assert config.get_complete_url(api_base, "@cf/baai/bge-reranker-base") == expected


def test_get_complete_url_requires_account_id(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    config = CloudflareRerankConfig()

    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        config.get_complete_url(None, "@cf/baai/bge-reranker-base")


@pytest.mark.parametrize(
    "model,error_match",
    (
        ("../graphql", "cannot be a dot path segment"),
        ("@cf/baai/../graphql", "cannot be a dot path segment"),
        ("/@cf/baai/bge-reranker-base", "model is required"),
    ),
)
def test_get_complete_url_rejects_path_traversal(model, error_match):
    config = CloudflareRerankConfig()

    with pytest.raises(ValueError, match=error_match):
        config.get_complete_url(
            "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run",
            model,
        )


def test_get_complete_url_encodes_model_path_segments():
    config = CloudflareRerankConfig()

    url = config.get_complete_url(
        "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run",
        "@cf/baai/model?debug=1",
    )

    assert url.endswith("/ai/run/%40cf/baai/model%3Fdebug%3D1")


def test_validate_environment_and_supported_params():
    config = CloudflareRerankConfig()

    headers = config.validate_environment(
        headers={"X-Test": "value"},
        model="@cf/baai/bge-reranker-base",
        api_key="cf-key",
    )

    assert headers["Authorization"] == "Bearer cf-key"
    assert headers["X-Test"] == "value"
    assert config.get_supported_cohere_rerank_params("@cf/baai/bge-reranker-base") == (
        "query",
        "documents",
        "top_n",
        "return_documents",
    )


def test_validate_environment_requires_api_key(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    config = CloudflareRerankConfig()

    with pytest.raises(ValueError, match="Cloudflare API Key"):
        config.validate_environment({}, "@cf/baai/bge-reranker-base")


def test_map_cohere_rerank_params_without_top_n():
    config = CloudflareRerankConfig()

    assert config.map_cohere_rerank_params(
        non_default_params={},
        model="@cf/baai/bge-reranker-base",
        drop_params=False,
        query="query",
        documents=("document",),
    ) == {
        "query": "query",
        "documents": ("document",),
        "return_documents": True,
    }


@pytest.mark.parametrize(
    ("top_n", "return_documents", "expected"),
    [
        (None, None, {"query": "query", "documents": ("document",)}),
        (
            1,
            None,
            {"query": "query", "documents": ("document",), "top_n": 1},
        ),
    ],
)
def test_map_cohere_rerank_params_handles_optional_values(top_n, return_documents, expected):
    config = CloudflareRerankConfig()

    assert (
        config.map_cohere_rerank_params(
            non_default_params={},
            model="@cf/baai/bge-reranker-base",
            drop_params=False,
            query="query",
            documents=("document",),
            top_n=top_n,
            return_documents=return_documents,
        )
        == expected
    )


def test_transform_rerank_request():
    config = CloudflareRerankConfig()

    request = config.transform_rerank_request(
        model="@cf/baai/bge-reranker-base",
        optional_rerank_params={
            "query": "Which animal is faster?",
            "documents": ["A cheetah can sprint.", {"text": "A turtle walks."}],
            "top_n": 1,
        },
        headers={},
    )

    assert request == {
        "query": "Which animal is faster?",
        "contexts": (
            {"text": "A cheetah can sprint."},
            {"text": "A turtle walks."},
        ),
        "top_k": 1,
    }


@pytest.mark.parametrize(
    "params,error_match",
    (
        ({"documents": ("document",)}, "query is required for Cloudflare rerank"),
        ({"query": "query"}, "documents is required for Cloudflare rerank"),
        (
            {"query": "query", "documents": "document"},
            "documents is required for Cloudflare rerank",
        ),
        ({"query": "query", "documents": ()}, "documents is required for Cloudflare rerank"),
    ),
)
def test_transform_rerank_request_validates_required_params(params, error_match):
    config = CloudflareRerankConfig()

    with pytest.raises(ValueError, match=error_match):
        config.transform_rerank_request(
            model="@cf/baai/bge-reranker-base",
            optional_rerank_params=params,
            headers={},
        )


def test_transform_rerank_request_without_top_n():
    config = CloudflareRerankConfig()

    request = config.transform_rerank_request(
        model="@cf/baai/bge-reranker-base",
        optional_rerank_params={
            "query": "query",
            "documents": ({"title": "LiteLLM"},),
        },
        headers={},
    )

    assert request == {
        "query": "query",
        "contexts": ({"text": '{"title": "LiteLLM"}'},),
    }


def test_transform_rerank_request_rejects_invalid_document():
    config = CloudflareRerankConfig()

    with pytest.raises(TypeError, match="strings or dictionaries"):
        config.transform_rerank_request(
            model="@cf/baai/bge-reranker-base",
            optional_rerank_params={
                "query": "query",
                "documents": (123,),
            },
            headers={},
        )


def test_transform_rerank_response_handles_rest_envelope():
    config = CloudflareRerankConfig()
    logging_obj = MagicMock()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": [
                    {"id": 1, "score": 0.91},
                    {"id": 0, "score": 0.42},
                ]
            },
            "success": True,
            "errors": [],
            "messages": [],
        },
    )

    response = config.transform_rerank_response(
        model="@cf/baai/bge-reranker-base",
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=logging_obj,
        request_data={"query": "query", "contexts": [{"text": "a"}, {"text": "b"}]},
    )

    assert response.results == [
        {"index": 1, "relevance_score": 0.91},
        {"index": 0, "relevance_score": 0.42},
    ]


@pytest.mark.parametrize("return_documents", [True, False])
def test_transform_rerank_response_honors_return_documents(return_documents):
    config = CloudflareRerankConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={"result": {"response": [{"id": 0, "score": 0.91}]}},
    )

    response = config.transform_rerank_response(
        model="@cf/baai/bge-reranker-base",
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
        optional_params={
            "documents": ("LiteLLM is an LLM gateway.",),
            "return_documents": return_documents,
        },
    )

    assert response.results is not None
    if return_documents:
        assert response.results[0]["document"] == {"text": "LiteLLM is an LLM gateway."}
    else:
        assert "document" not in response.results[0]


@pytest.mark.parametrize(
    ("response_json", "error_type"),
    [
        ({"success": False, "errors": ("failed",)}, CloudflareError),
        ({"result": {}}, CloudflareError),
        ({"result": {"response": ["invalid"]}}, TypeError),
        ({"result": {"response": [{"id": "0", "score": 1}]}}, TypeError),
    ],
)
def test_transform_rerank_response_rejects_invalid_responses(response_json, error_type):
    config = CloudflareRerankConfig()
    raw_response = httpx.Response(status_code=400, json=response_json)

    with pytest.raises(error_type):
        config.transform_rerank_response(
            model="@cf/baai/bge-reranker-base",
            raw_response=raw_response,
            model_response=RerankResponse(),
            logging_obj=MagicMock(),
        )


@pytest.mark.parametrize(
    "raw_response",
    [
        httpx.Response(status_code=500, text="not json"),
        httpx.Response(status_code=200, json=("not", "an", "object")),
    ],
)
def test_transform_rerank_response_rejects_invalid_json(raw_response):
    config = CloudflareRerankConfig()

    with pytest.raises(CloudflareError):
        config.transform_rerank_response(
            model="@cf/baai/bge-reranker-base",
            raw_response=raw_response,
            model_response=RerankResponse(),
            logging_obj=MagicMock(),
        )


def test_get_error_class():
    error = CloudflareRerankConfig().get_error_class("failed", 400, {})

    assert isinstance(error, CloudflareError)
    assert error.status_code == 400


def test_litellm_rerank_sends_cloudflare_request():
    response_json = {
        "result": {"response": [{"id": 0, "score": 0.98}]},
        "success": True,
    }
    raw_response = Mock()
    raw_response.status_code = 200
    raw_response.json.return_value = response_json
    raw_response.text = json.dumps(response_json)
    client = _RecordingHTTPHandler(raw_response)

    response = litellm.rerank(
        model="cloudflare/@cf/baai/bge-reranker-base",
        query="What is LiteLLM?",
        documents=["LiteLLM is an LLM gateway.", "A recipe for soup."],
        top_n=1,
        api_key="test-key",
        api_base="https://api.cloudflare.com/client/v4/accounts/account-id/ai/run",
        client=client,
    )

    request = client.requests[0]
    assert request["url"].endswith("/ai/run/%40cf/baai/bge-reranker-base")
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert json.loads(request["data"]) == {
        "query": "What is LiteLLM?",
        "contexts": [
            {"text": "LiteLLM is an LLM gateway."},
            {"text": "A recipe for soup."},
        ],
        "top_k": 1,
    }
    assert response.results == [
        {
            "index": 0,
            "relevance_score": 0.98,
            "document": {"text": "LiteLLM is an LLM gateway."},
        }
    ]
