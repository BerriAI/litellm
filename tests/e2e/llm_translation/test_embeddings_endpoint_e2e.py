"""Live e2e: POST /embeddings returns a real vector.

Registers embedding deployments at runtime (OpenAI + Vertex) and asserts a
non-empty, non-zero vector came back.
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

    @pytest.mark.covers("llm.embeddings.vertex.basic.nonstream.works")
    def test_vertex_embeddings_returns_vector(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-embeddings-vertex-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="vertex_ai/text-embedding-005",
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location="us-central1",
                vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.embeddings(key, model, "Say this is a test!")
        require_successful_call(result)
        parsed = EmbeddingsResult.model_validate_json(result.body)
        assert parsed.first_vector, f"vertex /embeddings returned no vector: {result.body[:300]}"
        assert any(component != 0.0 for component in parsed.first_vector), (
            f"vertex embedding vector is all zeros: {result.body[:300]}"
        )
