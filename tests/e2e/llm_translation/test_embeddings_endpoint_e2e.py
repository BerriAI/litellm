"""Live e2e: POST /embeddings returns a real vector across OpenAI, Bedrock, Vertex, Cohere.

Each test registers the deployment it needs at runtime (deleted on teardown),
drives the endpoint with the real OpenAI SDK (LIT-4577), and asserts a
non-empty, non-zero vector came back. The LIT-3167 guard in
tests/e2e/embeddings/ covers the Gemini embedding path; embeddings cost tracking
is covered by tests/e2e/quota_management/spend_tracking/. Malformed bodies the
SDK refuses to build stay on the shared transport.
"""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import provider_edge_base, unique_marker
from e2e_http import assert_client_error
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel
from sdk_clients import NO_PROXY_CACHE, SdkClients, response_header

pytestmark = pytest.mark.e2e

VERTEX_TEXT_EMBEDDING: Final = "vertex_ai/text-embedding-005"
VERTEX_MULTIMODAL_EMBEDDING: Final = "vertex_ai/multimodalembedding@001"


class _OptionalEmbeddingsBody(BaseModel):
    model: str | None = None
    input: str | list[str] | None = None


class _TokenEmbeddingsBody(BaseModel):
    model: str
    input: list[list[int]]


def _vertex_params(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=model, vertex_project="os.environ/VERTEXAI_PROJECT", vertex_location="us-central1")


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


def _register(
    proxy: ProxyClient, resources: ResourceManager, prefix: str, params: LiteLLMParamsBody
) -> tuple[str, str]:
    model = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _assert_embedding_vector(
    proxy: ProxyClient,
    resources: ResourceManager,
    sdk: SdkClients,
    prefix: str,
    params: LiteLLMParamsBody,
) -> None:
    model, key = _register(proxy, resources, prefix, params)
    client = sdk.openai(key)

    embeddings = client.embeddings.create(model=model, input="Say this is a test!", extra_body=NO_PROXY_CACHE)
    assert embeddings.data, f"/embeddings returned no data: {embeddings!r}"
    vector = embeddings.data[0].embedding
    assert vector, f"/embeddings returned no vector: {embeddings!r}"
    assert any(component != 0.0 for component in vector), "embedding vector is all zeros"


class TestEmbeddingsEndpoint:
    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.basic.nonstream.works")
    def test_embeddings_returns_vector(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        _assert_embedding_vector(proxy, resources, sdk, "e2e-embeddings", _openai_embeddings_params())

    @pytest.mark.covers("llm.embeddings.bedrock.basic.nonstream.works")
    def test_bedrock_embeddings_returns_vector(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        _assert_embedding_vector(
            proxy,
            resources,
            sdk,
            "e2e-embeddings-bedrock",
            LiteLLMParamsBody(
                model="bedrock/amazon.titan-embed-text-v2:0",
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )

    @pytest.mark.covers("llm.embeddings.cohere.basic.nonstream.works")
    def test_cohere_embeddings_returns_vector(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        _assert_embedding_vector(
            proxy,
            resources,
            sdk,
            "e2e-embeddings-cohere",
            LiteLLMParamsBody(model="cohere/embed-v4.0", api_key="os.environ/COHERE_API_KEY"),
        )

    @pytest.mark.covers("llm.embeddings.vertex.basic.nonstream.works")
    def test_vertex_embeddings_returns_vector(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        _assert_embedding_vector(
            proxy,
            resources,
            sdk,
            "e2e-embeddings-vertex",
            LiteLLMParamsBody(
                model="vertex_ai/text-embedding-005",
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location="us-central1",
            ),
        )

    def test_mistral_embeddings_returns_vector(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        _assert_embedding_vector(
            proxy,
            resources,
            sdk,
            "e2e-embeddings-mistral",
            LiteLLMParamsBody(model="mistral/mistral-embed", api_key="os.environ/MISTRAL_API_KEY"),
        )

    @pytest.mark.covers("llm.embeddings.vertex.basic.nonstream.works")
    def test_vertex_embeddings_honor_requested_dimensions(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, "e2e-embeddings-vertex-dims", _vertex_params(VERTEX_TEXT_EMBEDDING))

        embeddings = sdk.openai(key).embeddings.create(
            model=model,
            input="Say this is a test!",
            dimensions=8,
            extra_body={**NO_PROXY_CACHE, "task_type": "RETRIEVAL_QUERY", "auto_truncate": True},
        )
        assert len(embeddings.data[0].embedding) == 8, f"dimensions=8 was not honored: {embeddings!r}"
        assert embeddings.usage.prompt_tokens > 0, f"vertex embeddings reported no prompt usage: {embeddings.usage!r}"

    def test_vertex_multimodal_embeddings_honor_dimensions_and_are_costed(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(
            proxy, resources, "e2e-embeddings-vertex-mm", _vertex_params(VERTEX_MULTIMODAL_EMBEDDING)
        )

        raw = sdk.openai(key).embeddings.with_raw_response.create(
            model=model, input="Say this is a test!", dimensions=128, extra_body=NO_PROXY_CACHE
        )
        embeddings = raw.parse()
        assert len(embeddings.data[0].embedding) == 128, f"dimensions=128 was not honored: {embeddings!r}"
        cost = response_header(raw.headers, "x-litellm-response-cost")
        assert cost is not None and float(cost) > 0, f"multimodal embedding was not costed: {cost!r}"

    def test_bedrock_titan_rejects_token_array_input_as_bad_request(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register(
            proxy,
            resources,
            "e2e-embeddings-titan-tokens",
            LiteLLMParamsBody(
                model="bedrock/amazon.titan-embed-text-v2:0",
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )
        result = proxy.transport.send(
            "/embeddings",
            headers=proxy.transport.bearer(key),
            json=_TokenEmbeddingsBody(model=model, input=[[1]]),
        )
        assert result.status_code == 400, (
            f"titan cannot embed token arrays, so the caller must get a 400, got {result.status_code}: "
            f"{result.body[:300]}"
        )

    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.basic.nonstream.works")
    def test_array_input_returns_vectors(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register(proxy, resources, "e2e-embeddings-array", _openai_embeddings_params())
        embeddings = sdk.openai(key).embeddings.create(
            model=model, input=["Hello", "World", "Test"], extra_body=NO_PROXY_CACHE
        )
        assert len(embeddings.data) == 3, f"expected 3 vectors: {embeddings!r}"

    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.input_validation.nonstream.works")
    def test_missing_model_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key = resources.key()
        result = proxy.transport.send(
            "/embeddings",
            headers=proxy.transport.bearer(key),
            json=_OptionalEmbeddingsBody(input="hello"),
        )
        assert_client_error(result, "embeddings missing model")

    @pytest.mark.replayable
    @pytest.mark.covers("llm.embeddings.openai.input_validation.nonstream.works")
    def test_missing_input_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources, "e2e-embeddings-missin", _openai_embeddings_params())
        result = proxy.transport.send(
            "/embeddings",
            headers=proxy.transport.bearer(key),
            json=_OptionalEmbeddingsBody(model=model),
        )
        assert_client_error(result, "embeddings missing input")
