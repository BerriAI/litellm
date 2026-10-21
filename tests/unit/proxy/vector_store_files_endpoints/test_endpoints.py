"""
require_managed_files enforcement for litellm/proxy/vector_store_files_endpoints/endpoints.py

Every vector-store file route (create, retrieve, content, update, delete) resolves its
caller-supplied file id through _update_request_data_with_managed_file_id before the
provider call, so the guard lives there once and covers all five.

A raw or forged managed-looking file id has no ownership row, so without the guard it
is attached to a vector store or read back under shared provider credentials.
"""

import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from fastapi import HTTPException

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.vector_store_files_endpoints.endpoints import (
    _update_request_data_with_managed_file_id,
    _with_managed_file_list_ids,
    _with_provider_file_id_cursors,
)
from litellm.types.utils import SpecialEnums
from litellm.types.vector_store_files import (
    VectorStoreFileListResponse,
    VectorStoreFileObject,
    VectorStoreFileStatus,
)

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


@dataclass(frozen=True)
class ManagedFileIdResolverStub:
    resolver: AsyncMock

    async def get_unified_file_ids_for_provider_file_ids(
        self,
        provider_file_ids: Sequence[str],
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Mapping[str, str]:
        return await self.resolver(
            provider_file_ids=provider_file_ids,
            user_api_key_dict=user_api_key_dict,
        )


def _unified_file_id(provider_file_id: str = RAW_FILE_ID) -> str:
    unified = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json",
        "victim-unified-id",
        "gpt-4o-mini",
        provider_file_id,
        "gpt-4o-mini-id",
    )
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


def _vector_store_file_row(file_id: str) -> VectorStoreFileObject:
    return {
        "id": file_id,
        "object": "vector_store.file",
        "created_at": 1700000000,
        "usage_bytes": 100,
        "vector_store_id": "vs-test",
        "status": VectorStoreFileStatus.COMPLETED,
        "last_error": None,
        "chunking_strategy": {"type": "auto"},
        "attributes": {"source": "test"},
    }


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


@pytest.mark.parametrize(
    "provider_ids",
    [
        (RAW_FILE_ID, "file-unmanaged-123"),
        ("file-unmanaged-123", RAW_FILE_ID),
    ],
)
@pytest.mark.asyncio
async def test_vector_store_file_list_maps_owned_ids_and_preserves_raw_ids(
    provider_ids: tuple[str, str],
) -> None:
    managed_file_id: Final = _unified_file_id()
    expected_provider_ids: Final = tuple(
        managed_file_id if provider_file_id == RAW_FILE_ID else provider_file_id
        for provider_file_id in provider_ids
    )
    provider_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row(provider_file_id)
            for provider_file_id in provider_ids
        ],
        "first_id": provider_ids[0],
        "last_id": provider_ids[1],
        "has_more": True,
    }
    original_response: Final = deepcopy(provider_response)
    resolver: Final = AsyncMock(return_value={RAW_FILE_ID: managed_file_id})
    managed_files_obj: Final = ManagedFileIdResolverStub(resolver=resolver)

    response: Final = await _with_managed_file_list_ids(
        response=provider_response,
        managed_files_obj=managed_files_obj,
        user_api_key_dict=CALLER,
    )

    expected_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row(provider_file_id)
            for provider_file_id in expected_provider_ids
        ],
        "first_id": expected_provider_ids[0],
        "last_id": expected_provider_ids[1],
        "has_more": True,
    }
    assert response == expected_response
    assert provider_response == original_response
    resolver.assert_awaited_once_with(
        provider_file_ids=tuple(dict.fromkeys(provider_ids)),
        user_api_key_dict=CALLER,
    )


@pytest.mark.asyncio
async def test_vector_store_file_list_only_maps_round_trippable_ids() -> None:
    managed_file_id: Final = _unified_file_id("file-model-a")
    provider_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row("file-model-a"),
            _vector_store_file_row("file-model-b"),
        ],
        "first_id": "file-model-a",
        "last_id": "file-model-b",
        "has_more": False,
    }
    resolver: Final = AsyncMock(
        return_value={
            "file-model-a": managed_file_id,
            "file-model-b": managed_file_id,
        }
    )
    managed_files_obj: Final = ManagedFileIdResolverStub(resolver=resolver)

    response: Final = await _with_managed_file_list_ids(
        response=provider_response,
        managed_files_obj=managed_files_obj,
        user_api_key_dict=CALLER,
    )

    expected_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row(managed_file_id),
            _vector_store_file_row("file-model-b"),
        ],
        "first_id": managed_file_id,
        "last_id": "file-model-b",
        "has_more": False,
    }
    assert response == expected_response


def test_vector_store_file_list_translates_managed_cursors_and_preserves_raw_after() -> (
    None
):
    managed_file_id: Final = _unified_file_id()

    assert _with_provider_file_id_cursors(
        {"after": managed_file_id, "before": managed_file_id}
    ) == {"after": RAW_FILE_ID, "before": RAW_FILE_ID}
    assert _with_provider_file_id_cursors({"after": RAW_FILE_ID}) == {
        "after": RAW_FILE_ID
    }


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
