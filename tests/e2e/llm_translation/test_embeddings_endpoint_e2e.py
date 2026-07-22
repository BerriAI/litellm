"""Live e2e: POST /embeddings returns a real vector across OpenAI, Bedrock, Vertex.

Each test registers the deployment it needs at runtime (deleted on teardown) and
asserts a non-empty, non-zero vector came back. The LIT-3167 guard in
tests/e2e/embeddings/ covers the Gemini embedding path; embeddings cost tracking is
covered by tests/e2e/quota_management/spend_tracking/.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EmbeddingsResult, EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestEmbeddingsEndpoint:
    @pytest.mark.covers("llm.embeddings.openai.basic.nonstream.works")
    def test_embeddings_returns_vector(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-embeddings-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/text-embedding-3-small", api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.embeddings(key, model, "Say this is a test!")
        require_successful_call(result)
        parsed = EmbeddingsResult.model_validate_json(result.body)
        assert parsed.first_vector, f"/embeddings returned no vector: {result.body[:300]}"
        assert any(component != 0.0 for component in parsed.first_vector), (
            f"embedding vector is all zeros: {result.body[:300]}"
        )

    @pytest.mark.covers("llm.embeddings.bedrock.basic.nonstream.works")
    def test_bedrock_embeddings_returns_vector(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-embeddings-bedrock-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="bedrock/amazon.titan-embed-text-v2:0", aws_region_name="us-west-2"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.embeddings(key, model, "Say this is a test!")
        require_successful_call(result)
        parsed = EmbeddingsResult.model_validate_json(result.body)
        assert parsed.first_vector, f"/embeddings returned no vector: {result.body[:300]}"
        assert any(component != 0.0 for component in parsed.first_vector), (
            f"embedding vector is all zeros: {result.body[:300]}"
        )

    @pytest.mark.covers("llm.embeddings.vertex.basic.nonstream.works")
    def test_vertex_embeddings_returns_vector(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-embeddings-vertex-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="vertex_ai/gemini-embedding-2",
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location="us-central1",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.embeddings(key, model, "Say this is a test!")
        require_successful_call(result)
        parsed = EmbeddingsResult.model_validate_json(result.body)
        assert parsed.first_vector, f"/embeddings returned no vector: {result.body[:300]}"
        assert any(component != 0.0 for component in parsed.first_vector), (
            f"embedding vector is all zeros: {result.body[:300]}"
        )
