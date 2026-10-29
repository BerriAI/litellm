from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException
from litellm_enterprise.proxy.hooks.managed_vector_stores import PROXY_LiteLLMManagedVectorStores

from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.vector_stores import VectorStoreCreateResponse

_OWNER: Final = "vector-store-owner"
_PROVIDER_ID: Final = "vs_provider_store"
_READ_CALL_TYPES: Final = ("avector_store_search", "avector_store_retrieve")
_WRITE_CALL_TYPES: Final = ("avector_store_update", "avector_store_delete")
_ROUTED_CALL_TYPES: Final = _READ_CALL_TYPES + _WRITE_CALL_TYPES


@dataclass(frozen=True, slots=True)
class _StoreRow:
    created_by: str

    def model_dump(self) -> dict[str, str | None]:
        return {"created_by": self.created_by, "team_id": None}


_OWNER_ROW: Final = _StoreRow(created_by=_OWNER)


@dataclass(frozen=True, slots=True)
class _StoreTable:
    row: _StoreRow | None

    async def find_first(self, where: Mapping[str, object]) -> _StoreRow | None:
        return self.row


@dataclass(frozen=True, slots=True)
class _Database:
    litellm_managedvectorstoretable: _StoreTable


@dataclass(frozen=True, slots=True)
class _PrismaClient:
    db: _Database


def _hook_and_unified_id(
    target_model_names: tuple[str, ...] = ("model-a",),
    stored_row: _StoreRow | None = _OWNER_ROW,
) -> tuple[PROXY_LiteLLMManagedVectorStores, str]:
    hook: Final = PROXY_LiteLLMManagedVectorStores(
        DualCache(), prisma_client=_PrismaClient(_Database(_StoreTable(stored_row)))
    )
    created: Final = VectorStoreCreateResponse(
        id=_PROVIDER_ID,
        object="vector_store",
        created_at=1700000000,
        name="managed-store",
        bytes=0,
        file_counts={"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
        status="completed",
        expires_after=None,
        expires_at=None,
        last_active_at=None,
        metadata={},
    )
    unified_id: Final = hook.generate_unified_resource_id(
        resource_objects=[created], target_model_names_list=list(target_model_names)
    )
    return hook, unified_id


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", _ROUTED_CALL_TYPES)
async def test_owner_request_goes_to_provider_store_on_its_model(call_type: str) -> None:
    hook, unified_id = _hook_and_unified_id()

    routed: Final = await hook.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(user_id=_OWNER),
        cache=DualCache(),
        data={"vector_store_id": unified_id},
        call_type=call_type,
    )

    assert routed == {"vector_store_id": _PROVIDER_ID, "model": "model-a"}


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", _READ_CALL_TYPES)
async def test_reading_store_created_on_several_models_goes_to_first_target_model(call_type: str) -> None:
    hook, unified_id = _hook_and_unified_id(("model-a", "model-b"))

    routed: Final = await hook.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(user_id=_OWNER),
        cache=DualCache(),
        data={"vector_store_id": unified_id},
        call_type=call_type,
    )

    assert routed == {"vector_store_id": _PROVIDER_ID, "model": "model-a"}


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", _WRITE_CALL_TYPES)
async def test_changing_store_created_on_several_models_is_refused_instead_of_changing_one_copy(
    call_type: str,
) -> None:
    hook, unified_id = _hook_and_unified_id(("model-a", "model-b"))

    with pytest.raises(HTTPException) as refused:
        await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(user_id=_OWNER),
            cache=DualCache(),
            data={"vector_store_id": unified_id},
            call_type=call_type,
        )

    assert refused.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", _ROUTED_CALL_TYPES)
async def test_other_user_is_refused_before_reaching_provider(call_type: str) -> None:
    hook, unified_id = _hook_and_unified_id()

    with pytest.raises(HTTPException) as refused:
        await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(user_id="someone-else"),
            cache=DualCache(),
            data={"vector_store_id": unified_id},
            call_type=call_type,
        )

    assert refused.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", _ROUTED_CALL_TYPES)
async def test_provider_store_id_passes_through_unchanged(call_type: str) -> None:
    hook, _ = _hook_and_unified_id()

    routed: Final = await hook.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(user_id="someone-else"),
        cache=DualCache(),
        data={"vector_store_id": "vs_raw_store"},
        call_type=call_type,
    )

    assert routed == {"vector_store_id": "vs_raw_store"}


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", _ROUTED_CALL_TYPES)
async def test_deleted_managed_store_answers_not_found_instead_of_access_denied(call_type: str) -> None:
    hook, unified_id = _hook_and_unified_id(stored_row=None)

    with pytest.raises(HTTPException) as missing:
        await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(user_id=_OWNER),
            cache=DualCache(),
            data={"vector_store_id": unified_id},
            call_type=call_type,
        )

    assert missing.value.status_code == 404
