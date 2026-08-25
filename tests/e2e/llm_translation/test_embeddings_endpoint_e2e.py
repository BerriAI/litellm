"""Live e2e: POST /embeddings returns a real vector across OpenAI, Bedrock, Vertex.

Each test registers the deployment it needs at runtime (deleted on teardown) and
asserts a non-empty, non-zero vector came back. The LIT-3167 guard in
tests/e2e/embeddings/ covers the Gemini embedding path; embeddings cost tracking is
covered by tests/e2e/quota_management/spend_tracking/.
"""

from __future__ import annotations

import pytest
from e2e_config import provider_edge_base, unique_marker
from e2e_http import (
    assert_client_error,
    require_successful_call,
)
from endpoints_client import EmbeddingsResult, EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


class _OptionalEmbeddingsBody(BaseModel):
    model: str | None = None
    input: str | list[str] | None = None


def _openai_embeddings_params() -> LiteLLMParamsBody:
    """The OpenAI embeddings deployment, wired through the record/replay edge when a
    fixture mode is active and straight at OpenAI otherwise (LIT-5974). Bedrock and
    Vertex stay live: SigV4 signs the Host header, and neither has an edge mount."""
    base = provider_edge_base("openai")
    return LiteLLMParamsBody(
        model="openai/text-embedding-3-small",
        api_key="os.environ/OPENAI_API_KEY",
        api_base=None if base is None else f"{base}/v1",
    )


class TestEmbeddingsEndpoint:
    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.basic.nonstream.works")
    def test_embeddings_returns_vector(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-embeddings-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            _openai_embeddings_params(),
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
                model="bedrock/amazon.titan-embed-text-v2:0",
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
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

    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.basic.nonstream.works")
    def test_array_input_returns_vectors(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-embeddings-array-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            _openai_embeddings_params(),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()
        result = endpoints_client.proxy.transport.send(
            "/embeddings",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalEmbeddingsBody(model=model, input=["Hello", "World", "Test"]),
        )
        require_successful_call(result)
        parsed = EmbeddingsResult.model_validate_json(result.body)
        assert len(parsed.data) == 3, f"expected 3 vectors: {result.body[:300]}"

    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.input_validation.nonstream.works")
    def test_missing_model_returns_client_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        key = resources.key()
        result = endpoints_client.proxy.transport.send(
            "/embeddings",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalEmbeddingsBody(input="hello"),
        )
        assert_client_error(result, "embeddings missing model")

    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.input_validation.nonstream.works")
    def test_missing_input_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-embeddings-missin-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            _openai_embeddings_params(),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()
        result = endpoints_client.proxy.transport.send(
            "/embeddings",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalEmbeddingsBody(model=model),
        )
        assert_client_error(result, "embeddings missing input")
