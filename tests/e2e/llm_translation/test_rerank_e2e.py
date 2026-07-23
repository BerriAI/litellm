"""Live e2e: POST /v1/rerank ranks documents by relevance.

Registers a Cohere rerank deployment at runtime and asserts the endpoint returns
scored results within the requested top_n. No official OpenAI/Anthropic SDK
covers /v1/rerank, so the call rides the shared typed transport via
ProxyClient.rerank.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, RerankBody, RerankResponse
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

DOCUMENTS = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean.",
    "Washington, D.C. is the capital of the United States.",
    "Capital punishment has existed in the United States since before it was a country.",
]
QUERY = "What is the capital of the United States?"


def _assert_top_n_scored(response: RerankResponse) -> None:
    assert response.results, f"/rerank returned no results: {response!r}"
    assert len(response.results) <= 3, f"top_n=3 not honored: {response!r}"
    assert response.results[0].relevance_score is not None, (
        f"top rerank result has no relevance_score: {response!r}"
    )


class TestRerank:
    @pytest.mark.covers("llm.rerank.cohere.basic.nonstream.works")
    def test_rerank_scores_top_n(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-rerank-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model="cohere/rerank-v3.5", api_key="os.environ/COHERE_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            proxy.rerank(key, RerankBody(model=model, query=QUERY, documents=DOCUMENTS, top_n=3))
        )
        _assert_top_n_scored(response)

    @pytest.mark.covers("llm.rerank.bedrock.basic.nonstream.works", exercised_on=["rerank"])
    def test_bedrock_rerank_scores_top_n(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = f"e2e-bedrock-rerank-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="bedrock/amazon.rerank-v1:0",
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            proxy.rerank(key, RerankBody(model=model, query=QUERY, documents=DOCUMENTS, top_n=3))
        )
        _assert_top_n_scored(response)
