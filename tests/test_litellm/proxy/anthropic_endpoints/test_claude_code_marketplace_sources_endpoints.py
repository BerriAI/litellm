"""
Unit tests for claude_code_marketplace_sources.py.

Covers the register/list/get/sync/delete marketplace-source routes and their
proxy-admin gating. resolve_and_sync itself is exercised separately in
test_claude_code_marketplace_sync.py, so here it's replaced with a stub that
reports success - these tests are only about the endpoint/DB-orchestration
layer built on top of it.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints import (
    claude_code_marketplace_sources as sources_module,
)
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace_sources import (
    delete_marketplace,
    get_marketplace_source,
    list_marketplaces,
    register_marketplace,
    sync_marketplace,
)
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace_sync import (
    SyncResult,
)
from litellm.types.proxy.claude_code_endpoints import RegisterMarketplaceRequest


class _FakeTable:
    def __init__(self):
        self._rows: dict = {}

    @staticmethod
    def _matches(record, where):
        return all(getattr(record, k, None) == v for k, v in where.items())

    async def find_unique(self, where):
        for record in self._rows.values():
            if self._matches(record, where):
                return record
        return None

    async def find_many(self, where=None):
        if not where:
            return list(self._rows.values())
        return [r for r in self._rows.values() if self._matches(r, where)]

    async def count(self, where=None):
        return len(await self.find_many(where))

    async def create(self, data):
        # Mirrors the LiteLLM_SkillMarketplaceTable / LiteLLM_ClaudeCodePluginTable
        # prisma schema defaults, since this fake has no DB layer to apply them.
        record_data = {
            "display_name": None,
            "branch": "main",
            "enabled": True,
            "sync_error": None,
            "skipped_count": 0,
            "last_synced_at": None,
            "created_at": None,
            "updated_at": None,
            "description": None,
            "version": None,
            "marketplace_id": None,
            **data,
        }
        record_data.setdefault("id", str(uuid.uuid4()))
        record = SimpleNamespace(**record_data)
        self._rows[record.id] = record
        return record

    async def update(self, where, data):
        record = await self.find_unique(where)
        if record is None:
            raise ValueError(f"no record matching {where}")
        for k, v in data.items():
            setattr(record, k, v)
        return record

    async def update_many(self, where, data):
        matched = await self.find_many(where)
        for record in matched:
            for k, v in data.items():
                setattr(record, k, v)
        return len(matched)


def _make_fake_prisma_client():
    client = SimpleNamespace()
    client.db = SimpleNamespace(
        litellm_skillmarketplacetable=_FakeTable(),
        litellm_claudecodeplugintable=_FakeTable(),
    )
    return client


_ADMIN = UserAPIKeyAuth(
    user_role=LitellmUserRoles.PROXY_ADMIN,
    api_key="sk-admin",
    user_id="admin-user",
)

_NON_ADMIN = UserAPIKeyAuth(
    user_role=LitellmUserRoles.INTERNAL_USER,
    api_key="sk-regular",
    user_id="regular-user",
)


@pytest.fixture(autouse=True)
def prisma_client(monkeypatch):
    client = _make_fake_prisma_client()
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", client)
    monkeypatch.setattr(litellm.proxy.proxy_server, "master_key", "sk-admin")
    monkeypatch.setattr(
        sources_module,
        "resolve_and_sync",
        AsyncMock(return_value=SyncResult(status="success", error=None, plugin_count=3)),
    )
    return client


@pytest.mark.asyncio
async def test_register_marketplace_success(prisma_client):
    request = RegisterMarketplaceRequest(source="anthropics/skills", name="anthropic-skills")

    response = await register_marketplace(request=request, user_api_key_dict=_ADMIN)

    assert response.status == "success"
    assert response.marketplace.name == "anthropic-skills"
    assert response.marketplace.source_ref == "anthropics/skills"
    assert response.marketplace.plugin_count == 3

    stored = await prisma_client.db.litellm_skillmarketplacetable.find_unique(where={"name": "anthropic-skills"})
    assert stored is not None


@pytest.mark.asyncio
async def test_register_marketplace_derives_name_from_source(prisma_client):
    request = RegisterMarketplaceRequest(source="anthropics/skills")

    response = await register_marketplace(request=request, user_api_key_dict=_ADMIN)

    assert response.marketplace.name == "skills"


@pytest.mark.asyncio
async def test_register_marketplace_duplicate_name_rejected(prisma_client):
    request = RegisterMarketplaceRequest(source="anthropics/skills", name="dup-skills")
    await register_marketplace(request=request, user_api_key_dict=_ADMIN)

    with pytest.raises(HTTPException) as exc_info:
        await register_marketplace(request=request, user_api_key_dict=_ADMIN)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_marketplaces(prisma_client):
    await register_marketplace(
        request=RegisterMarketplaceRequest(source="anthropics/skills", name="anthropic-skills"),
        user_api_key_dict=_ADMIN,
    )
    await register_marketplace(
        request=RegisterMarketplaceRequest(source="vercel-labs/skills", name="vercel-skills"),
        user_api_key_dict=_ADMIN,
    )

    response = await list_marketplaces(user_api_key_dict=_ADMIN)

    assert response.count == 2
    assert {m.name for m in response.marketplaces} == {"anthropic-skills", "vercel-skills"}


@pytest.mark.asyncio
async def test_get_marketplace_source(prisma_client):
    await register_marketplace(
        request=RegisterMarketplaceRequest(source="anthropics/skills", name="anthropic-skills"),
        user_api_key_dict=_ADMIN,
    )
    marketplace = await prisma_client.db.litellm_skillmarketplacetable.find_unique(where={"name": "anthropic-skills"})
    for plugin_name in ["anthropic-skills--doc", "anthropic-skills--example", "anthropic-skills--api"]:
        await prisma_client.db.litellm_claudecodeplugintable.create(
            data={"name": plugin_name, "marketplace_id": marketplace.id}
        )

    response = await get_marketplace_source(marketplace_name="anthropic-skills", user_api_key_dict=_ADMIN)

    assert response.name == "anthropic-skills"
    assert response.source_ref == "anthropics/skills"
    assert response.plugin_count == 3


@pytest.mark.asyncio
async def test_get_marketplace_source_not_found(prisma_client):
    with pytest.raises(HTTPException) as exc_info:
        await get_marketplace_source(marketplace_name="does-not-exist", user_api_key_dict=_ADMIN)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_sync_marketplace_reinvokes_resolve_and_sync(prisma_client):
    await register_marketplace(
        request=RegisterMarketplaceRequest(source="anthropics/skills", name="anthropic-skills"),
        user_api_key_dict=_ADMIN,
    )
    assert sources_module.resolve_and_sync.call_count == 1

    response = await sync_marketplace(marketplace_name="anthropic-skills", user_api_key_dict=_ADMIN)

    assert response.status == "success"
    assert sources_module.resolve_and_sync.call_count == 2


@pytest.mark.asyncio
async def test_sync_marketplace_not_found(prisma_client):
    with pytest.raises(HTTPException) as exc_info:
        await sync_marketplace(marketplace_name="does-not-exist", user_api_key_dict=_ADMIN)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_marketplace_disables_marketplace_and_its_plugins(prisma_client):
    await register_marketplace(
        request=RegisterMarketplaceRequest(source="anthropics/skills", name="anthropic-skills"),
        user_api_key_dict=_ADMIN,
    )
    marketplace = await prisma_client.db.litellm_skillmarketplacetable.find_unique(where={"name": "anthropic-skills"})
    await prisma_client.db.litellm_claudecodeplugintable.create(
        data={"name": "anthropic-skills--doc", "enabled": True, "marketplace_id": marketplace.id}
    )

    response = await delete_marketplace(marketplace_name="anthropic-skills", user_api_key_dict=_ADMIN)

    assert response["status"] == "success"

    refreshed_marketplace = await prisma_client.db.litellm_skillmarketplacetable.find_unique(
        where={"name": "anthropic-skills"}
    )
    assert refreshed_marketplace.enabled is False

    refreshed_plugin = await prisma_client.db.litellm_claudecodeplugintable.find_unique(
        where={"name": "anthropic-skills--doc"}
    )
    assert refreshed_plugin.enabled is False


@pytest.mark.asyncio
async def test_delete_marketplace_not_found(prisma_client):
    with pytest.raises(HTTPException) as exc_info:
        await delete_marketplace(marketplace_name="does-not-exist", user_api_key_dict=_ADMIN)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_non_admin_forbidden_on_all_marketplace_routes(prisma_client):
    """Regression test for the admin-gating decision: a real non-admin
    UserAPIKeyAuth (not a mock) must be rejected by every marketplace-source
    route, both before and after a marketplace exists."""
    request = RegisterMarketplaceRequest(source="anthropics/skills", name="anthropic-skills")

    with pytest.raises(HTTPException) as exc_info:
        await register_marketplace(request=request, user_api_key_dict=_NON_ADMIN)
    assert exc_info.value.status_code == 403

    # Seed a marketplace (as an admin) so the remaining routes have something
    # to act on - a 404 short-circuit before the role check would falsely
    # pass this test.
    await register_marketplace(request=request, user_api_key_dict=_ADMIN)

    with pytest.raises(HTTPException) as exc_info:
        await list_marketplaces(user_api_key_dict=_NON_ADMIN)
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        await get_marketplace_source(marketplace_name="anthropic-skills", user_api_key_dict=_NON_ADMIN)
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        await sync_marketplace(marketplace_name="anthropic-skills", user_api_key_dict=_NON_ADMIN)
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        await delete_marketplace(marketplace_name="anthropic-skills", user_api_key_dict=_NON_ADMIN)
    assert exc_info.value.status_code == 403

    # None of the non-admin calls should have mutated anything.
    stored = await prisma_client.db.litellm_skillmarketplacetable.find_unique(where={"name": "anthropic-skills"})
    assert stored.enabled is True
