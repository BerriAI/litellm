import asyncio
from unittest.mock import AsyncMock, patch

from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.rag.ingestion.openai_ingestion import OpenAIRAGIngestion


def test_openai_ingest_existing_file_id_attaches_without_uploading():
    asyncio.run(_run_openai_existing_file_id_attach_test())


async def _run_openai_existing_file_id_attach_test():
    ingestion = OpenAIRAGIngestion(
        {
            "chunking_strategy": {"type": "auto"},
            "vector_store": {
                "custom_llm_provider": "openai",
                "vector_store_id": "vs_existing",
            },
        }
    )

    with (
        patch(
            "litellm.rag.ingestion.openai_ingestion.vector_store_file_acreate",
            new_callable=AsyncMock,
        ) as mock_attach,
        patch(
            "litellm.rag.ingestion.openai_ingestion.litellm.acreate_file",
            new_callable=AsyncMock,
        ) as mock_upload,
    ):
        response = await ingestion.ingest(file_id="file_existing")

    assert response["status"] == "completed"
    assert response["vector_store_id"] == "vs_existing"
    assert response["file_id"] == "file_existing"
    mock_upload.assert_not_called()
    mock_attach.assert_awaited_once_with(
        vector_store_id="vs_existing",
        file_id="file_existing",
        custom_llm_provider="openai",
        chunking_strategy={"type": "auto"},
        api_key=None,
        api_base=None,
    )


def test_openai_ingest_existing_file_id_requires_vector_store_id():
    asyncio.run(_run_openai_existing_file_id_requires_vector_store_id_test())


async def _run_openai_existing_file_id_requires_vector_store_id_test():
    ingestion = OpenAIRAGIngestion({"vector_store": {"custom_llm_provider": "openai"}})

    with (
        patch(
            "litellm.rag.ingestion.openai_ingestion.vector_store_acreate",
            new_callable=AsyncMock,
        ) as mock_create_vector_store,
        patch(
            "litellm.rag.ingestion.openai_ingestion.vector_store_file_acreate",
            new_callable=AsyncMock,
        ) as mock_attach,
    ):
        response = await ingestion.ingest(file_id="file_existing")

    assert response["status"] == "failed"
    assert "vector_store_id is required" in response["error"]
    mock_create_vector_store.assert_not_called()
    mock_attach.assert_not_called()


class UnsupportedExistingFileIngestion(BaseRAGIngestion):
    async def store(
        self,
        file_content: bytes | None,
        filename: str | None,
        content_type: str | None,
        chunks: list[str],
        embeddings: list[list[float]] | None,
        existing_file_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        raise AssertionError("store should not be called for unsupported file_id")


def test_existing_file_id_fails_for_unsupported_ingestion_provider():
    asyncio.run(_run_unsupported_existing_file_id_test())


async def _run_unsupported_existing_file_id_test():
    ingestion = UnsupportedExistingFileIngestion(
        {"vector_store": {"custom_llm_provider": "unsupported"}}
    )

    response = await ingestion.ingest(file_id="file_existing")

    assert response["status"] == "failed"
    assert "does not support ingesting an existing file_id" in response["error"]
