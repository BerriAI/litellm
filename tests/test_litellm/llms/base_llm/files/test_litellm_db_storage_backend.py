from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Base64
from prisma.errors import RecordNotFoundError

from litellm.llms.base_llm.files.litellm_db_storage_backend import (
    LITELLM_DB_STORAGE_URL_PREFIX,
    LiteLLMDbStorageBackend,
    storage_url_to_row_id,
)


def _backend_with_table():
    table = MagicMock(create=AsyncMock(), find_unique=AsyncMock(), delete=AsyncMock())
    tables = MagicMock(litellm_managedfilecontenttable=table)
    prisma_client = MagicMock(db=tables, replica_db=tables)
    return LiteLLMDbStorageBackend(prisma_client), table


@pytest.mark.asyncio
async def test_upload_stores_bytes_and_returns_prefixed_row_id():
    backend, table = _backend_with_table()
    table.create.return_value = SimpleNamespace(id="row-1")
    content = b"\x00\x01binary jsonl\n"

    storage_url = await backend.upload_file(file_content=content, filename="input.jsonl", content_type="text/plain")

    assert storage_url == f"{LITELLM_DB_STORAGE_URL_PREFIX}row-1"
    stored = table.create.await_args.kwargs["data"]["content"]
    assert isinstance(stored, Base64)
    assert stored.decode() == content


@pytest.mark.asyncio
async def test_download_returns_exact_bytes_of_the_row():
    backend, table = _backend_with_table()
    content = b'{"custom_id": "1"}\n'
    table.find_unique.return_value = SimpleNamespace(id="row-1", content=Base64.encode(content))

    downloaded = await backend.download_file(f"{LITELLM_DB_STORAGE_URL_PREFIX}row-1")

    assert downloaded == content
    table.find_unique.assert_awaited_once_with(where={"id": "row-1"})


@pytest.mark.asyncio
async def test_download_missing_row_raises_value_error_naming_the_url():
    backend, table = _backend_with_table()
    table.find_unique.return_value = None
    storage_url = f"{LITELLM_DB_STORAGE_URL_PREFIX}missing"

    with pytest.raises(ValueError, match="missing"):
        await backend.download_file(storage_url)


@pytest.mark.asyncio
async def test_download_rejects_url_without_prefix_before_touching_the_db():
    backend, table = _backend_with_table()

    with pytest.raises(ValueError, match="https://elsewhere/blob"):
        await backend.download_file("https://elsewhere/blob")

    table.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_removes_the_parsed_row():
    backend, table = _backend_with_table()

    await backend.delete_file(f"{LITELLM_DB_STORAGE_URL_PREFIX}row-1")

    table.delete.assert_awaited_once_with(where={"id": "row-1"})


@pytest.mark.asyncio
async def test_delete_tolerates_a_row_that_is_already_gone():
    backend, table = _backend_with_table()
    table.delete.side_effect = RecordNotFoundError({"user_facing_error": {"message": "gone"}})

    await backend.delete_file(f"{LITELLM_DB_STORAGE_URL_PREFIX}row-1")

    table.delete.assert_awaited_once_with(where={"id": "row-1"})


def test_storage_url_to_row_id_round_trips():
    assert storage_url_to_row_id(f"{LITELLM_DB_STORAGE_URL_PREFIX}abc-123") == "abc-123"


def test_storage_url_to_row_id_rejects_foreign_urls():
    with pytest.raises(ValueError, match="s3://bucket/key"):
        storage_url_to_row_id("s3://bucket/key")
