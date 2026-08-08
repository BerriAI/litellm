"""
require_managed_files enforcement for litellm/proxy/vector_store_files_endpoints/endpoints.py

Every vector-store file route (create, retrieve, content, update, delete) resolves its
caller-supplied file id through _update_request_data_with_managed_file_id before the
provider call, so the guard lives there once and covers all five.

A raw provider file id has no ownership row, so without the guard it is attached to a
vector store or read back under the shared provider credentials with no tenant check.
"""

import base64
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from fastapi import HTTPException

import litellm
from litellm.proxy.vector_store_files_endpoints.endpoints import (
    _update_request_data_with_managed_file_id,
)
from litellm.types.utils import SpecialEnums

RAW_FILE_ID = "file-victim-abc123"


def _unified_file_id() -> str:
    unified = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json", "victim-unified-id", "gpt-4o-mini", RAW_FILE_ID, "gpt-4o-mini-id"
    )
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


def _resolve(file_id: str):
    return _update_request_data_with_managed_file_id(
        data={"vector_store_id": "vs-test", "file_id": file_id},
        file_id=file_id,
        request=MagicMock(headers={}, query_params={}),
        llm_router=None,
    )


def test_raw_file_id_rejected_when_managed_files_required():
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            _resolve(RAW_FILE_ID)

    assert exc.value.status_code == 400


def test_model_encoded_file_id_rejected_when_managed_files_required():
    """encode_file_id_with_model output is client-forgeable and carries no ownership
    row, so it is not a managed file id."""
    from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

    encoded = encode_file_id_with_model(RAW_FILE_ID, "gpt-4o-mini", id_type="file")

    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            _resolve(encoded)

    assert exc.value.status_code == 400


def test_unified_file_id_allowed_when_managed_files_required():
    with patch.object(litellm, "require_managed_files", True):
        data, original = _resolve(_unified_file_id())

    assert original == _unified_file_id()
    assert data["file_id"] == RAW_FILE_ID


def test_raw_file_id_allowed_when_managed_files_not_required():
    with patch.object(litellm, "require_managed_files", False):
        data, original = _resolve(RAW_FILE_ID)

    assert original is None
    assert data["file_id"] == RAW_FILE_ID
