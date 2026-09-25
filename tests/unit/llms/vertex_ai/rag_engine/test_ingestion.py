import asyncio
import sys
from types import ModuleType, SimpleNamespace

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
                "vertex_location": "us-central1",
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


def _vertexai_sdk_stub(import_calls: list[dict[str, object]]) -> ModuleType:
    rag = ModuleType("vertexai.rag")
    rag.TransformationConfig = lambda chunking_config: chunking_config
    rag.ChunkingConfig = lambda chunk_size, chunk_overlap: (chunk_size, chunk_overlap)

    def import_files(**kwargs):
        import_calls.append(kwargs)
        return SimpleNamespace(imported_rag_files_count=1)

    rag.import_files = import_files
    vertexai = ModuleType("vertexai")
    vertexai.init = lambda project, location: None
    vertexai.rag = rag
    return vertexai


def test_ingest_runs_end_to_end_through_the_base_pipeline(monkeypatch):
    monkeypatch.setenv("GCS_BATCH_BUCKET_NAME", "batch-bucket")
    resolver = VertexAIFilesConfig()
    import_calls: list[dict[str, object]] = []
    stub = _vertexai_sdk_stub(import_calls)
    monkeypatch.setitem(sys.modules, "vertexai", stub)
    monkeypatch.setitem(sys.modules, "vertexai.rag", stub.rag)

    async def acreate_file_through_real_bucket_resolver(**kwargs):
        bucket = resolver._get_configured_bucket_name(get_litellm_params(**kwargs))
        return SimpleNamespace(id=f"gs://{bucket}/{kwargs['file'][0]}")

    monkeypatch.setattr(litellm, "acreate_file", acreate_file_through_real_bucket_resolver)

    result = asyncio.run(_ingestion_for_bucket("rag-bucket").ingest(file_data=("doc.txt", b"doc", "text/plain")))

    assert (result["status"], result["vector_store_id"], result["file_id"]) == ("completed", "corpus-123", "gs://rag-bucket/doc.txt")
    assert [(c["corpus_name"], c["paths"]) for c in import_calls] == [
        ("projects/test-project/locations/us-central1/ragCorpora/corpus-123", ["gs://rag-bucket/doc.txt"])
    ]
