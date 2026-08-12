import os
import sys
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.openai_files_endpoints.common_utils import (
    apply_unified_file_ids,
    map_raw_file_ids_to_unified,
)
from litellm.types.utils import LiteLLMBatch


def _batch(input_file_id, output_file_id, error_file_id) -> LiteLLMBatch:
    return LiteLLMBatch(
        id="batch-1",
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id=input_file_id,
        object="batch",
        status="cancelled",
        output_file_id=output_file_id,
        error_file_id=error_file_id,
    )


@pytest.mark.asyncio
async def test_map_raw_file_ids_to_unified_empty_ids_skips_db():
    prisma_client = MagicMock()

    assert await map_raw_file_ids_to_unified(frozenset(), prisma_client) == {}

    prisma_client.db.litellm_managedfiletable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_map_raw_file_ids_to_unified_no_prisma_client_returns_empty():
    assert await map_raw_file_ids_to_unified(frozenset({"file-raw-1"}), None) == {}


@pytest.mark.asyncio
async def test_map_raw_file_ids_to_unified_bulk_queries_and_filters_to_requested_ids():
    row_a = MagicMock(
        unified_file_id="unified-a",
        flat_model_file_ids=["file-raw-a", "file-raw-other"],
    )
    row_b = MagicMock(unified_file_id="unified-b", flat_model_file_ids=["file-raw-b"])
    prisma_client = MagicMock()
    prisma_client.db.litellm_managedfiletable.find_many = AsyncMock(return_value=[row_a, row_b])

    mapping = await map_raw_file_ids_to_unified(
        frozenset({"file-raw-b", "file-raw-a", "file-raw-missing"}), prisma_client
    )

    prisma_client.db.litellm_managedfiletable.find_many.assert_awaited_once_with(
        where={"flat_model_file_ids": {"hasSome": ["file-raw-a", "file-raw-b", "file-raw-missing"]}}
    )
    assert dict(mapping) == {"file-raw-a": "unified-a", "file-raw-b": "unified-b"}


def test_apply_unified_file_ids_swaps_only_mapped_ids():
    batch = _batch(input_file_id="file-raw-in", output_file_id="file-raw-out", error_file_id=None)

    apply_unified_file_ids(batch, MappingProxyType({"file-raw-out": "unified-out"}))

    assert batch.input_file_id == "file-raw-in"
    assert batch.output_file_id == "unified-out"
    assert batch.error_file_id is None


def test_apply_unified_file_ids_swaps_all_three_ids():
    batch = _batch(
        input_file_id="file-raw-in",
        output_file_id="file-raw-out",
        error_file_id="file-raw-err",
    )

    apply_unified_file_ids(
        batch,
        MappingProxyType(
            {
                "file-raw-in": "unified-in",
                "file-raw-out": "unified-out",
                "file-raw-err": "unified-err",
            }
        ),
    )

    assert (batch.input_file_id, batch.output_file_id, batch.error_file_id) == (
        "unified-in",
        "unified-out",
        "unified-err",
    )
