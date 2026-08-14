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


# =========================================================================== #
# add_internal_model_credentials_for_batch - the snapshot that lets a completed
# batch's output file be read, and therefore its cost be recorded
# =========================================================================== #


def test_add_internal_model_credentials_attaches_an_immutable_snapshot():
    """Cost accounting for a completed batch reads its output file, and Bedrock resolves
    that bucket only from this snapshot. It must be immutable so nothing downstream can
    redirect the bucket that managed file ids are validated against."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        add_internal_model_credentials_for_batch,
    )

    router = MagicMock()
    router.get_deployment_credentials_with_provider = MagicMock(
        return_value={"s3_bucket_name": "configured-bucket", "aws_region_name": "us-east-1"}
    )
    data = {"batch_id": "unified-batch-id"}

    add_internal_model_credentials_for_batch(data=data, llm_router=router, model_id="deployment-1")

    snapshot = data["_litellm_internal_model_credentials"]
    assert snapshot["s3_bucket_name"] == "configured-bucket"
    assert isinstance(snapshot, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot["s3_bucket_name"] = "attacker-bucket"
    router.get_deployment_credentials_with_provider.assert_called_once_with(model_id="deployment-1")


@pytest.mark.parametrize(
    "model_id, credentials",
    [(None, {"s3_bucket_name": "b"}), ("deployment-1", None)],
    ids=["no-model-id", "deployment-has-no-credentials"],
)
def test_add_internal_model_credentials_is_a_noop_without_a_resolvable_deployment(model_id, credentials):
    """An unroutable batch must be left alone rather than given an empty snapshot, which
    would look like a configured bucket of nothing."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        add_internal_model_credentials_for_batch,
    )

    router = MagicMock()
    router.get_deployment_credentials_with_provider = MagicMock(return_value=credentials)
    data = {"batch_id": "unified-batch-id"}

    add_internal_model_credentials_for_batch(data=data, llm_router=router, model_id=model_id)

    assert "_litellm_internal_model_credentials" not in data


def test_add_internal_model_credentials_survives_a_failing_deployment_lookup():
    """The snapshot only enables cost accounting, so a batch whose deployment no longer
    resolves, which happens when a model group is removed while batches are in flight,
    must still be retrievable rather than failing the request on the lookup."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        add_internal_model_credentials_for_batch,
    )

    router = MagicMock()
    router.get_deployment_credentials_with_provider = MagicMock(side_effect=KeyError("deployment-gone"))
    data = {"batch_id": "unified-batch-id"}

    add_internal_model_credentials_for_batch(data=data, llm_router=router, model_id="deployment-gone")

    assert data == {"batch_id": "unified-batch-id"}
