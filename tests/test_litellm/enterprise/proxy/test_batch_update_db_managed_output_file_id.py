"""Regression: update_batch_in_database must not persist raw provider output_file_id."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.common_utils import update_batch_in_database
from litellm.types.utils import LiteLLMBatch


@pytest.mark.asyncio
async def test_update_batch_in_database_stores_unified_output_file_id():
    raw_output_file_id = "file-rawoutput789"
    unified_output_file_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    batch_id = "batch_managed_ids_test"
    unified_batch_id = (
        "litellm_proxy;model_id:my-model;llm_batch_id:batch_managed_ids_test"
    )

    response = LiteLLMBatch(
        id=batch_id,
        object="batch",
        status="completed",
        endpoint="/v1/chat/completions",
        input_file_id="file-input123",
        output_file_id=raw_output_file_id,
        completion_window="24h",
        created_at=1234567890,
    )
    response._hidden_params = {  # type: ignore[attr-defined]
        "model_id": "my-model",
        "model_name": "openai/gpt-4o",
    }

    mock_managed_files = MagicMock()
    mock_managed_files.get_unified_output_file_id = MagicMock(
        return_value=unified_output_file_id
    )
    mock_managed_files.store_unified_file_id = AsyncMock()

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    mock_prisma.db.litellm_managedobjecttable.update = AsyncMock()

    mock_logger = MagicMock()

    await update_batch_in_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma,
        verbose_proxy_logger=mock_logger,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    stored = json.loads(
        mock_prisma.db.litellm_managedobjecttable.update.call_args.kwargs["data"][
            "file_object"
        ]
    )
    assert stored["output_file_id"] == unified_output_file_id
    assert stored["output_file_id"] != raw_output_file_id
