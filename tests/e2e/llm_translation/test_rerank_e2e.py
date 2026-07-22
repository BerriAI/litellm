"""Live e2e: POST /v1/rerank ranks documents by relevance.

Registers a Cohere rerank deployment at runtime and asserts the endpoint returns
scored results within the requested top_n. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, RerankResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

DOCUMENTS = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean.",
    "Washington, D.C. is the capital of the United States.",
    "Capital punishment has existed in the United States since before it was a country.",
]


class TestRerank:
    @pytest.mark.covers("llm.rerank.cohere.basic.nonstream.works")
    def test_rerank_scores_top_n(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-rerank-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="cohere/rerank-v3.5", api_key="os.environ/COHERE_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.rerank(
            key, model, "What is the capital of the United States?", DOCUMENTS, top_n=3
        )
        require_successful_call(result)
        parsed = RerankResult.model_validate_json(result.body)
        assert parsed.results, f"/rerank returned no results: {result.body[:300]}"
        assert len(parsed.results) <= 3, f"top_n=3 not honored: {result.body[:300]}"
        assert parsed.results[0].relevance_score is not None, (
            f"top rerank result has no relevance_score: {result.body[:300]}"
        )
