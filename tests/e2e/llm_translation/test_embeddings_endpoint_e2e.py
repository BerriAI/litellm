"""Live e2e: POST /embeddings returns a real vector.

Registers an OpenAI embedding deployment at runtime and asserts a non-empty,
non-zero vector came back. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py; the LIT-3167 guard in
tests/e2e/embeddings/ covers the Gemini embedding path.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call
from endpoints_client import EmbeddingsResult, EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


def _assert_real_vector(body: str) -> None:
    parsed = EmbeddingsResult.model_validate_json(body)
    assert parsed.first_vector, f"/embeddings returned no vector: {body[:300]}"
    assert any(component != 0.0 for component in parsed.first_vector), (
        f"embedding vector is all zeros: {body[:300]}"
    )


class TestEmbeddingsEndpoint:
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
        _assert_real_vector(result.body)

    @pytest.mark.covers("llm.embeddings.bedrock.basic.nonstream.works", exercised_on=["embeddings"])
    def test_bedrock_embeddings_returns_vector(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = f"e2e-bedrock-embeddings-{unique_marker()}"
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
        _assert_real_vector(result.body)
