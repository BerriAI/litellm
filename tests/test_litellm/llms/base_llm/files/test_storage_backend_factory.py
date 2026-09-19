from unittest.mock import MagicMock

import pytest

from litellm.llms.base_llm.files.litellm_db_storage_backend import (
    LITELLM_DB_STORAGE_BACKEND_NAME,
    LiteLLMDbStorageBackend,
)
from litellm.llms.base_llm.files.storage_backend_factory import get_storage_backend


def test_litellm_db_backend_is_built_on_the_given_prisma_client():
    prisma_client = MagicMock()

    backend = get_storage_backend(LITELLM_DB_STORAGE_BACKEND_NAME, prisma_client=prisma_client)

    assert isinstance(backend, LiteLLMDbStorageBackend)
    assert backend._table is prisma_client.db.litellm_managedfilecontenttable


def test_litellm_db_backend_without_a_database_is_rejected():
    with pytest.raises(ValueError, match="database-connected proxy"):
        get_storage_backend(LITELLM_DB_STORAGE_BACKEND_NAME)


def test_unknown_backend_is_still_rejected():
    with pytest.raises(ValueError, match="Unsupported storage backend type: nope"):
        get_storage_backend("nope", prisma_client=MagicMock())
