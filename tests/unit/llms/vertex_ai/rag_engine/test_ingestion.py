import asyncio
from types import SimpleNamespace

import litellm
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.llms.vertex_ai.files.transformation import VertexAIFilesConfig
from litellm.llms.vertex_ai.rag_engine.ingestion import VertexAIRAGIngestion


def _ingestion_for_bucket(bucket: str) -> VertexAIRAGIngestion:
    return VertexAIRAGIngestion(
        {
            "vector_store": {
                "custom_llm_provider": "vertex_ai",
                "vector_store_id": "corpus-123",
                "vertex_project": "test-project",
                "gcs_bucket": bucket,
            }
        }
    )


def test_upload_lands_in_the_corpus_bucket_when_batch_bucket_env_is_set(monkeypatch):
    monkeypatch.setenv("GCS_BATCH_BUCKET_NAME", "batch-bucket")
    monkeypatch.setenv("GCS_BUCKET_NAME", "logging-bucket")
    resolver = VertexAIFilesConfig()

    async def acreate_file_through_real_bucket_resolver(**kwargs):
        bucket = resolver._get_configured_bucket_name(get_litellm_params(**kwargs))
        return SimpleNamespace(id=f"gs://{bucket}/{kwargs['file'][0]}")

    monkeypatch.setattr(litellm, "acreate_file", acreate_file_through_real_bucket_resolver)

    uri = asyncio.run(_ingestion_for_bucket("rag-bucket")._upload_file_to_gcs(b"doc", "doc.txt", "text/plain"))

    assert uri == "gs://rag-bucket/doc.txt"
