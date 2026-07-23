"""Live e2e: POST /embeddings returns a real vector across OpenAI, Bedrock, Vertex.

Each test registers the deployment it needs at runtime (deleted on teardown),
drives the endpoint with the real OpenAI SDK (LIT-4577), and asserts a
non-empty, non-zero vector came back. The LIT-3167 guard in
tests/e2e/embeddings/ covers the Gemini embedding path; embeddings cost tracking
is covered by tests/e2e/quota_management/spend_tracking/.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e


def _assert_embedding_vector(
    proxy: ProxyClient,
    resources: ResourceManager,
    sdk: SdkClients,
    prefix: str,
    params: LiteLLMParamsBody,
) -> None:
    model = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    client = sdk.openai(resources.key())

    embeddings = client.embeddings.create(model=model, input="Say this is a test!")
    assert embeddings.data, f"/embeddings returned no data: {embeddings!r}"
    vector = embeddings.data[0].embedding
    assert vector, f"/embeddings returned no vector: {embeddings!r}"
    assert any(component != 0.0 for component in vector), "embedding vector is all zeros"


class TestEmbeddingsEndpoint:
    @pytest.mark.covers("llm.embeddings.openai.basic.nonstream.works")
    def test_embeddings_returns_vector(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        _assert_embedding_vector(
            proxy,
            resources,
            sdk,
            "e2e-embeddings",
            LiteLLMParamsBody(
                model="openai/text-embedding-3-small", api_key="os.environ/OPENAI_API_KEY"
            ),
        )

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
                model="bedrock/amazon.titan-embed-text-v2:0", aws_region_name="us-west-2"
            ),
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
                model="vertex_ai/gemini-embedding-2",
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location="us-central1",
            ),
        )
