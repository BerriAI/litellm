from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, hash_token
from litellm.proxy.management_endpoints.scim.source_endpoints import create_source, list_sources, update_source
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.scim_agent_provisioning import SCIMSourceConfig, SCIMSourceCreate

TENANT: Final = "11111111-1111-4111-8111-111111111111"
ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth",
    [
        UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER),
        UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, allowed_routes=["/scim/*"]),
    ],
)
async def test_provisioning_credentials_cannot_manage_sources(auth: UserAPIKeyAuth) -> None:
    with pytest.raises(HTTPException) as failure:
        await list_sources(auth)
    assert failure.value.status_code == 403


def source_database(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

    client: Final = MagicMock(spec=PrismaClient)
    tx: Final = client.tx.return_value.__aenter__.return_value
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    tx.litellm_scimsource.find_unique = AsyncMock(return_value=None)
    tx.litellm_verificationtoken.find_unique = AsyncMock(return_value=SimpleNamespace(allowed_routes=["/scim/*"]))
    tx.litellm_accessgrouptable.find_many = AsyncMock(return_value=[])
    tx.litellm_scimsource.create = AsyncMock()
    return tx


@pytest.mark.asyncio
@pytest.mark.parametrize("routes", [[], ["/scim/*", "openai_routes"], ["/user/*"]])
async def test_source_token_must_be_restricted_to_scim_only(routes: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    tx: Final = source_database(monkeypatch)
    tx.litellm_verificationtoken.find_unique.return_value = SimpleNamespace(allowed_routes=routes)
    request: Final = SCIMSourceCreate(display_name="Source", tenant_id=TENANT, provisioning_token="test-token")
    with pytest.raises(HTTPException) as failure:
        await create_source(request, ADMIN)
    assert failure.value.status_code == 400
    tx.litellm_scimsource.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_token_cannot_be_reused_for_another_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    tx: Final = source_database(monkeypatch)
    tx.litellm_scimsource.find_unique.return_value = SimpleNamespace(source_id="existing-source")
    request: Final = SCIMSourceCreate(display_name="Source", tenant_id=TENANT, provisioning_token="test-token")
    with pytest.raises(HTTPException) as failure:
        await create_source(request, ADMIN)
    assert failure.value.status_code == 409
    tx.litellm_scimsource.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_creation_stores_a_hash_and_never_returns_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    tx: Final = source_database(monkeypatch)
    tx.litellm_scimsource.create.return_value = SimpleNamespace(
        source_id="source",
        display_name="Source",
        tenant_id=TENANT,
        enabled=True,
        group_mappings=[],
    )
    request: Final = SCIMSourceCreate(display_name="Source", tenant_id=TENANT, provisioning_token="test-token")
    result: Final = await create_source(request, ADMIN)
    stored: Final = tx.litellm_scimsource.create.call_args.kwargs["data"]
    assert stored["key_hash"] == hash_token("test-token")
    assert "test-token" not in str(stored)
    assert result.source_id == "source"
    assert "key_hash" not in result.model_dump()
    assert "provisioning_token" not in result.model_dump()


@pytest.mark.asyncio
async def test_source_update_rejects_tenant_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    tx: Final = source_database(monkeypatch)
    tx.litellm_scimsource.find_unique.return_value = SimpleNamespace(source_id="source", tenant_id=TENANT)
    request: Final = SCIMSourceConfig(display_name="Renamed", tenant_id="22222222-2222-4222-8222-222222222222")
    with pytest.raises(HTTPException) as failure:
        await update_source("source", request, ADMIN)
    assert failure.value.status_code == 409
    tx.litellm_scimsource.update.assert_not_called()


@pytest.mark.asyncio
async def test_source_creation_rejects_a_mapping_to_missing_access_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    tx: Final = source_database(monkeypatch)
    request: Final = SCIMSourceCreate(
        display_name="Source",
        tenant_id=TENANT,
        provisioning_token="test-token",
        group_mappings=[{"external_group_id": TENANT, "access_group_ids": ["missing"]}],
    )
    with pytest.raises(HTTPException) as failure:
        await create_source(request, ADMIN)
    assert failure.value.status_code == 400
    tx.litellm_scimsource.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_source_update_preserves_tenant_and_maps_existing_access_groups(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    tx: Final = source_database(monkeypatch)
    tx.litellm_scimsource.find_unique.return_value = SimpleNamespace(source_id="source", tenant_id=TENANT)
    tx.litellm_accessgrouptable.find_many.return_value = [SimpleNamespace(access_group_id="group")]
    request: Final = SCIMSourceConfig(
        display_name="Renamed",
        tenant_id=TENANT,
        enabled=enabled,
        group_mappings=[{"external_group_id": TENANT, "access_group_ids": ["group"]}],
    )
    tx.litellm_scimsource.update = AsyncMock(
        return_value=SimpleNamespace(source_id="source", **request.model_dump(mode="json"))
    )
    result: Final = await update_source("source", request, ADMIN)
    assert result.enabled is enabled
    assert result.group_mappings[0].access_group_ids == ("group",)
    stored: Final = tx.litellm_scimsource.update.call_args.kwargs["data"]
    assert stored["enabled"] is enabled
    assert "tenant_id" not in stored and "key_hash" not in stored


@pytest.mark.asyncio
async def test_missing_source_update_is_not_an_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    tx: Final = source_database(monkeypatch)
    request: Final = SCIMSourceConfig(display_name="Source", tenant_id=TENANT)
    with pytest.raises(HTTPException) as failure:
        await update_source("missing", request, ADMIN)
    assert failure.value.status_code == 404
    tx.litellm_scimsource.update.assert_not_called()


@pytest.mark.asyncio
async def test_source_list_returns_public_configuration_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    tx: Final = source_database(monkeypatch)
    tx.litellm_scimsource.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(
                source_id="source",
                display_name="Source",
                tenant_id=TENANT,
                enabled=True,
                group_mappings=[],
                key_hash="private-hash",
            )
        ]
    )
    result: Final = await list_sources(ADMIN)
    assert len(result) == 1 and result[0].source_id == "source"
    assert "private-hash" not in str(result)
    assert "key_hash" not in result[0].model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["list", "create", "update"])
@pytest.mark.parametrize("routes", [["/user/*"], ["openai_routes"], ["/scim/v2/sources"]])
async def test_any_scoped_administrator_is_denied_source_configuration(operation: str, routes: list[str]) -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, allowed_routes=routes)
    with pytest.raises(HTTPException) as failure:
        if operation == "list":
            await list_sources(auth)
        elif operation == "create":
            await create_source(SCIMSourceCreate(display_name="Source", tenant_id=TENANT, provisioning_token="test-token"), auth)
        else:
            await update_source("source", SCIMSourceConfig(display_name="Source", tenant_id=TENANT), auth)
    assert failure.value.status_code == 403
