from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from litellm.rag.ingestion.s3_vectors_ingestion import S3VectorsRAGIngestion


def _s3_vectors_ingest_options(**overrides):
    vector_store = {
        "custom_llm_provider": "s3_vectors",
        "vector_bucket_name": "test-vector-bucket",
        "aws_region_name": "us-east-1",
    }
    vector_store.update(overrides.pop("vector_store", {}))
    return {
        "vector_store": vector_store,
        **overrides,
    }


@pytest.mark.asyncio
async def test_s3_vectors_ingestion_uses_vector_store_embedding_model():
    ingestion = S3VectorsRAGIngestion(
        ingest_options=_s3_vectors_ingest_options(
            vector_store={"embedding_model": "text-embedding-3-large"}
        )
    )

    with patch(
        "litellm.rag.ingestion.s3_vectors_ingestion.litellm.aembedding",
        new_callable=AsyncMock,
    ) as mock_aembedding:
        mock_aembedding.return_value = SimpleNamespace(
            data=[{"embedding": [0.1, 0.2, 0.3]}]
        )

        embeddings = await ingestion.embed(["hello"])

        mock_aembedding.assert_awaited_once_with(
            model="text-embedding-3-large", input=["hello"]
        )
        assert embeddings == [[0.1, 0.2, 0.3]]

        # Scope the next assertion to dimension auto-detection. It must call
        # aembedding with the selected model instead of returning the default
        # S3 vector dimension without probing the embedding model.
        mock_aembedding.reset_mock()
        mock_aembedding.return_value = SimpleNamespace(
            data=[{"embedding": [0.1, 0.2, 0.3]}]
        )

        dimension = await ingestion._get_dimension_from_embedding_request()

    mock_aembedding.assert_awaited_once_with(
        model="text-embedding-3-large", input=["test"]
    )
    assert dimension == 3


def test_s3_vectors_ingestion_prefers_top_level_embedding_config():
    ingestion = S3VectorsRAGIngestion(
        ingest_options=_s3_vectors_ingest_options(
            embedding={"model": "text-embedding-3-small"},
            vector_store={"embedding_model": "text-embedding-3-large"},
        )
    )

    assert ingestion.embedding_config == {"model": "text-embedding-3-small"}
