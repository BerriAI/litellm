import json
from unittest.mock import MagicMock, Mock, patch

import httpx

import litellm
from litellm.llms.cloudflare.rerank.transformation import CloudflareRerankConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.rerank import RerankResponse
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_cloudflare_rerank_config():
    config = ProviderConfigManager.get_provider_rerank_config(
        model="@cf/baai/bge-reranker-base",
        provider=litellm.LlmProviders.CLOUDFLARE,
        api_base=None,
        present_version_params=[],
    )

    assert isinstance(config, CloudflareRerankConfig)


def test_get_complete_url_uses_native_workers_ai_endpoint():
    config = CloudflareRerankConfig()

    with patch(
        "litellm.llms.cloudflare.rerank.transformation.get_secret_str",
        return_value="account-id",
    ):
        url = config.get_complete_url(
            api_base=None,
            model="@cf/baai/bge-reranker-base",
        )

    assert url == ("https://api.cloudflare.com/client/v4/accounts/account-id/ai/run/@cf/baai/bge-reranker-base")


def test_get_complete_url_rewrites_openai_compatible_base():
    config = CloudflareRerankConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/account-id/ai/v1",
        model="@cf/baai/bge-reranker-base",
    )

    assert url.endswith("/accounts/account-id/ai/run/@cf/baai/bge-reranker-base")


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
        "contexts": [
            {"text": "A cheetah can sprint."},
            {"text": "A turtle walks."},
        ],
        "top_k": 1,
    }


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


def test_litellm_rerank_sends_cloudflare_request():
    client = HTTPHandler()
    response_json = {
        "result": {"response": [{"id": 0, "score": 0.98}]},
        "success": True,
    }
    raw_response = Mock()
    raw_response.status_code = 200
    raw_response.json.return_value = response_json
    raw_response.text = json.dumps(response_json)

    with patch.object(HTTPHandler, "post", return_value=raw_response) as mock_post:
        response = litellm.rerank(
            model="cloudflare/@cf/baai/bge-reranker-base",
            query="What is LiteLLM?",
            documents=["LiteLLM is an LLM gateway.", "A recipe for soup."],
            top_n=1,
            api_key="test-key",
            api_base="https://api.cloudflare.com/client/v4/accounts/account-id/ai/run",
            client=client,
        )

    request = mock_post.call_args.kwargs
    assert request["url"].endswith("/ai/run/@cf/baai/bge-reranker-base")
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert json.loads(request["data"]) == {
        "query": "What is LiteLLM?",
        "contexts": [
            {"text": "LiteLLM is an LLM gateway."},
            {"text": "A recipe for soup."},
        ],
        "top_k": 1,
    }
    assert response.results == [{"index": 0, "relevance_score": 0.98}]
