"""
require_managed_files enforcement for litellm/proxy/vector_store_files_endpoints/endpoints.py

Every vector-store file route (create, retrieve, content, update, delete) resolves its
caller-supplied file id through _update_request_data_with_managed_file_id before the
provider call, so the guard lives there once and covers all five.

A raw or forged managed-looking file id has no ownership row, so without the guard it
is attached to a vector store or read back under shared provider credentials.
"""

import base64
from dataclasses import dataclass
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest


from fastapi import HTTPException

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.vector_store_files_endpoints.endpoints import (
    _update_request_data_with_managed_file_id,
)
from litellm.types.utils import SpecialEnums

RAW_FILE_ID = "file-victim-abc123"
CALLER = UserAPIKeyAuth(api_key="sk-test", user_id="attacker-user", team_id="team-b")


@dataclass(frozen=True)
class ManagedResourceAccessCheckerStub:
    file_access: Literal["allow", "deny", "missing"]

    async def can_user_call_unified_file_id(
        self,
        unified_file_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        if self.file_access == "missing":
            raise HTTPException(status_code=404, detail=f"File not found: {unified_file_id}")
        return self.file_access == "allow"

    async def can_user_call_unified_object_id(
        self,
        unified_object_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        return False


def _unified_file_id() -> str:
    unified = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json", "victim-unified-id", "gpt-4o-mini", RAW_FILE_ID, "gpt-4o-mini-id"
    )
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


async def _resolve(
    file_id: str,
    file_access: Literal["allow", "deny", "missing"] = "allow",
):
    return await _update_request_data_with_managed_file_id(
        data={"vector_store_id": "vs-test", "file_id": file_id},
        file_id=file_id,
        request=MagicMock(headers={}, query_params={}),
        user_api_key_dict=CALLER,
        managed_files_obj=ManagedResourceAccessCheckerStub(file_access=file_access),
        llm_router=None,
    )


@pytest.mark.asyncio
async def test_raw_file_id_rejected_when_managed_files_required():
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _resolve(RAW_FILE_ID)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_model_encoded_file_id_rejected_when_managed_files_required():
    """encode_file_id_with_model output is client-forgeable and carries no ownership
    row, so it is not a managed file id."""
    from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

    encoded = encode_file_id_with_model(RAW_FILE_ID, "gpt-4o-mini", id_type="file")

    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _resolve(encoded)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_forged_unified_file_id_rejected_without_ownership_record():
    forged_id = _unified_file_id()
    data = {"vector_store_id": "vs-test", "file_id": forged_id}

    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _update_request_data_with_managed_file_id(
                data=data,
                file_id=forged_id,
                request=MagicMock(headers={}, query_params={}),
                user_api_key_dict=CALLER,
                managed_files_obj=ManagedResourceAccessCheckerStub(file_access="missing"),
                llm_router=None,
            )

    assert exc.value.status_code == 404
    assert data["file_id"] == forged_id


@pytest.mark.asyncio
async def test_other_teams_unified_file_id_rejected():
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _resolve(_unified_file_id(), file_access="deny")

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_owned_unified_file_id_allowed_when_managed_files_required():
    with patch.object(litellm, "require_managed_files", True):
        data, original = await _resolve(_unified_file_id())

    assert original == _unified_file_id()
    assert data["file_id"] == RAW_FILE_ID


@pytest.mark.asyncio
async def test_raw_file_id_allowed_when_managed_files_not_required():
    with patch.object(litellm, "require_managed_files", False):
        data, original = await _resolve(RAW_FILE_ID)

    assert original is None
    assert data["file_id"] == RAW_FILE_ID
