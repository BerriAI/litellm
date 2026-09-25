from itertools import chain
from types import SimpleNamespace
from typing import Annotated, Final
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from prisma import Json
from prisma.types import LiteLLM_SCIMSourceCreateInput
from pydantic import BaseModel

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, hash_token
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import PrismaClient
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.types.proxy.management_endpoints.scim_agent_provisioning import (
    SCIMSourceConfig,
    SCIMSourceCreate,
    SCIMSourceResponse,
)

router: Final = APIRouter(prefix="/sources")


class _AdminPolicy(BaseModel):
    user_role: str | None = None
    allowed_routes: tuple[str, ...] | None = None


def _require_source_admin(auth: UserAPIKeyAuth) -> None:
    policy: Final = _AdminPolicy.model_validate(auth, from_attributes=True)
    if policy.user_role != LitellmUserRoles.PROXY_ADMIN or "/scim/*" in (policy.allowed_routes or ()):
        raise HTTPException(403, "Provisioning configuration requires an unrestricted proxy administrator")


def source_response(source: object) -> SCIMSourceResponse:
    return SCIMSourceResponse.model_validate(source, from_attributes=True)


async def _client() -> PrismaClient:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "Provisioning configuration requires a database")
    return prisma_client


@router.get("", response_model=tuple[SCIMSourceResponse, ...])
async def list_sources(auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> tuple[SCIMSourceResponse, ...]:
    _require_source_admin(auth)
    client: Final = await _client()
    async with client.tx() as tx:
        sources: Final = await tx.litellm_scimsource.find_many(order={"display_name": "asc"})
    return tuple(source_response(source) for source in sources)


@router.post("", response_model=SCIMSourceResponse, status_code=201)
async def create_source(
    data: SCIMSourceCreate, auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
) -> SCIMSourceResponse:
    _require_source_admin(auth)
    client: Final = await _client()
    token_hash: Final = hash_token(data.provisioning_token.get_secret_value())
    async with client.tx() as tx:
        key: Final = await VerificationTokenRepository(SimpleNamespace(db=tx)).table.find_unique(
            where={"token": token_hash}
        )
        if key is None or key.allowed_routes != ["/scim/*"]:
            raise HTTPException(400, "Select a dedicated token restricted to /scim/*")
        if await tx.litellm_scimsource.find_unique(where={"key_hash": token_hash}) is not None:
            raise HTTPException(409, "This token already belongs to a provisioning source")
        group_ids: Final = tuple(
            frozenset(chain.from_iterable(mapping.access_group_ids for mapping in data.group_mappings))
        )
        groups: Final = await tx.litellm_accessgrouptable.find_many(where={"access_group_id": {"in": list(group_ids)}})
        if frozenset(group.access_group_id for group in groups) != frozenset(group_ids):
            raise HTTPException(400, "A mapped access group does not exist")
        create_data: Final[LiteLLM_SCIMSourceCreateInput] = LiteLLM_SCIMSourceCreateInput(
            source_id=str(uuid4()),
            display_name=data.display_name,
            tenant_id=str(data.tenant_id),
            key_hash=token_hash,
            enabled=data.enabled,
            group_mappings=Json([mapping.model_dump(mode="json") for mapping in data.group_mappings]),
        )
        source: Final = await tx.litellm_scimsource.create(data=create_data)
    return source_response(source)


@router.put("/{source_id}", response_model=SCIMSourceResponse)
async def update_source(
    source_id: str, data: SCIMSourceConfig, auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
) -> SCIMSourceResponse:
    _require_source_admin(auth)
    client: Final = await _client()
    async with client.tx() as tx:
        source: Final = await tx.litellm_scimsource.find_unique(where={"source_id": source_id})
        if source is None:
            raise HTTPException(404, "Provisioning source not found")
        if str(data.tenant_id) != source.tenant_id:
            raise HTTPException(409, "A provisioning source's tenant is immutable")
        group_ids: Final = tuple(
            frozenset(chain.from_iterable(mapping.access_group_ids for mapping in data.group_mappings))
        )
        groups: Final = await tx.litellm_accessgrouptable.find_many(where={"access_group_id": {"in": list(group_ids)}})
        if frozenset(group.access_group_id for group in groups) != frozenset(group_ids):
            raise HTTPException(400, "A mapped access group does not exist")
        updated: Final = await tx.litellm_scimsource.update(
            where={"source_id": source_id},
            data={
                "display_name": data.display_name,
                "enabled": data.enabled,
                "group_mappings": Json([mapping.model_dump(mode="json") for mapping in data.group_mappings]),
            },
        )
    return source_response(updated)
