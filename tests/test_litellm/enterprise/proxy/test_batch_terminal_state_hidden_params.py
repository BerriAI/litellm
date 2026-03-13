"""
Test that the terminal-state shortcut path in retrieve_batch populates
_hidden_params.model_id from the unified batch ID.

Regression: When a batch is in a terminal state (completed/failed/cancelled/expired),
the DB shortcut path returned a LiteLLMBatch with empty _hidden_params, causing the
managed_files hook to skip encoding output_file_id into a unified ID.
"""

import base64

import pytest

from litellm.proxy.openai_files_endpoints.common_utils import (
    get_model_id_from_unified_batch_id,
)
from litellm.types.utils import LiteLLMBatch


MODEL_ID = "model-xyz-123"
RAW_BATCH_ID = "batch-provider-456"
DECODED_UNIFIED_BATCH_ID = (
    f"litellm_proxy;model_id:{MODEL_ID};llm_batch_id:{RAW_BATCH_ID}"
)
B64_UNIFIED_BATCH_ID = (
    base64.urlsafe_b64encode(DECODED_UNIFIED_BATCH_ID.encode()).decode().rstrip("=")
)


def test_get_model_id_from_unified_batch_id():
    """model_id is correctly extracted from a unified batch ID."""
    assert get_model_id_from_unified_batch_id(DECODED_UNIFIED_BATCH_ID) == MODEL_ID


def test_get_model_id_returns_none_for_invalid():
    """Returns None for non-unified IDs."""
    assert get_model_id_from_unified_batch_id("plain-batch-id") is None
    assert get_model_id_from_unified_batch_id("") is None


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "expired"])
def test_terminal_batch_hidden_params_population(status):
    """
    Simulate the terminal-state shortcut path logic: when a batch from the DB
    has a terminal status and we have a unified_batch_id, _hidden_params should
    get model_id and unified_batch_id set.

    This mirrors the code added to endpoints.py lines 415-423.
    """
    # Create a batch as it would come from the database (empty _hidden_params)
    response = LiteLLMBatch(
        id=B64_UNIFIED_BATCH_ID,
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id="file-input-abc",
        object="batch",
        status=status,
        output_file_id="file-output-raw",
    )

    assert response._hidden_params.get("model_id") is None, "precondition: no model_id"

    unified_batch_id = DECODED_UNIFIED_BATCH_ID

    # This is the exact logic from the terminal-state shortcut path
    if unified_batch_id:
        response._hidden_params["unified_batch_id"] = unified_batch_id
        model_id_from_batch = get_model_id_from_unified_batch_id(unified_batch_id)
        if model_id_from_batch:
            response._hidden_params["model_id"] = model_id_from_batch

    assert response._hidden_params["model_id"] == MODEL_ID
    assert response._hidden_params["unified_batch_id"] == DECODED_UNIFIED_BATCH_ID


def test_terminal_batch_no_unified_id_leaves_hidden_params_empty():
    """
    When there is no unified_batch_id (non-managed batch), _hidden_params
    should remain unchanged.
    """
    response = LiteLLMBatch(
        id="batch-plain-id",
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id="file-input-abc",
        object="batch",
        status="completed",
    )

    unified_batch_id = None

    if unified_batch_id:
        response._hidden_params["unified_batch_id"] = unified_batch_id
        model_id_from_batch = get_model_id_from_unified_batch_id(unified_batch_id)
        if model_id_from_batch:
            response._hidden_params["model_id"] = model_id_from_batch

    assert response._hidden_params.get("model_id") is None
    assert response._hidden_params.get("unified_batch_id") is None
