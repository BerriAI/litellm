"""Session-scoped async ASGI client for HTTP-boundary behavior tests."""

import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx
import pytest_asyncio
import yaml
from prisma import Json

from litellm.proxy.utils import hash_token

MASTER_KEY = "sk-1234"
SCRATCH_PREFIX = "scratch-"


def _write_minimal_proxy_config() -> str:
    config = {
        "general_settings": {"master_key": MASTER_KEY},
        "litellm_settings": {},
    }
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        config["general_settings"]["database_url"] = database_url
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, f)
    f.close()
    return f.name


@pytest_asyncio.fixture(scope="session")
async def proxy_app():
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import (
        app,
        cleanup_router_config_variables,
        initialize,
        proxy_startup_event,
    )

    cleanup_router_config_variables()
    config_path = _write_minimal_proxy_config()

    # proxy_startup_event re-reads master_key from LITELLM_MASTER_KEY and
    # unconditionally overwrites the global, even when initialize() already
    # set it from the config YAML. Force (not setdefault) both vars: an
    # ambient LITELLM_MASTER_KEY with a different value would make the proxy
    # authenticate on that key while the tests still send MASTER_KEY.
    os.environ["LITELLM_MASTER_KEY"] = MASTER_KEY
    os.environ["CONFIG_FILE_PATH"] = config_path

    await initialize(config=config_path)

    # /key/regenerate is gated behind premium_user; flipping it lets the matrix
    # pin authz behavior instead of the licensing gate.
    proxy_server.premium_user = True

    async with proxy_startup_event(app):
        proxy_server.premium_user = True  # lifespan re-runs _license_check
        # The lifespan fires check_view_exists() as a background task; on a
        # fresh DB the first auth call races it and resolves user_id=None.
        if proxy_server.prisma_client is not None:
            await proxy_server.prisma_client.check_view_exists()
        yield app


@pytest_asyncio.fixture(scope="session")
async def proxy_client(proxy_app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=proxy_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def prisma(proxy_app):
    from litellm.proxy import proxy_server

    assert proxy_server.prisma_client is not None
    return proxy_server.prisma_client


@pytest_asyncio.fixture(scope="session")
async def world(prisma):
    from .actors import seed_world

    return await seed_world(prisma)


@dataclass(frozen=True)
class Scratch:
    prefix: str

    def tag(self, suffix: str = "") -> str:
        return f"{self.prefix}-{suffix}" if suffix else self.prefix


async def create_scratch_key(
    proxy_client,
    seeder_cleartext: str,
    scratch_prefix: str,
    *,
    user_id: str,
    team_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    key_alias: Optional[str] = None,
) -> str:
    """Seed a scratch-tagged key via /key/generate; returns its cleartext.

    Shared by the write-scenario matrices (key update/regenerate/delete).
    key_alias defaults to scratch_prefix; pass a distinct scratch-prefixed
    alias when a single scenario needs more than one key (/key/generate
    enforces unique aliases).
    """
    body: Dict[str, Any] = {
        "key_alias": key_alias or scratch_prefix,
        "user_id": user_id,
    }
    if team_id is not None:
        body["team_id"] = team_id
    if organization_id is not None:
        body["organization_id"] = organization_id
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder_cleartext}"},
        json=body,
    )
    assert resp.status_code == 200, f"setup failed: {resp.text}"
    return resp.json()["key"]


async def create_scratch_team(
    prisma,
    team_id: str,
    *,
    organization_id: Optional[str] = None,
    admin_user_ids: Optional[list] = None,
    member_user_ids: Optional[list] = None,
    team_member_permissions: Optional[list] = None,
    models: Optional[list] = None,
    max_budget: Optional[float] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Raw-seed a scratch-tagged team row; returns its team_id.

    The target team for the team write matrices (update / member_*). Raw
    prisma (not POST /team/new) avoids creation side effects — no creator
    auto-add, no membership rows written onto the world's users — so seeding
    never mutates the immutable read-world. The authz gates read the team's
    members_with_roles JSON, so a raw-seeded team exercises them exactly as
    a /team/new-created team would. team_id must start with the scratch
    prefix so the `scratch` fixture reclaims the row.

    team_member_permissions / models seed the matching raw columns — needed
    by the team-key-permission and team-model matrices.

    max_budget / tpm_limit / rpm_limit / metadata seed the team's own limit
    columns (Phase 4 F1+F3) — they live directly on LiteLLM_TeamTable, no
    budget-table relation needed.
    """
    admin_user_ids = list(admin_user_ids or [])
    member_user_ids = list(member_user_ids or [])
    members_with_roles = [
        {"user_id": uid, "role": "admin"} for uid in admin_user_ids
    ] + [{"user_id": uid, "role": "user"} for uid in member_user_ids]
    data: Dict[str, Any] = {
        "team_id": team_id,
        "team_alias": team_id,
        "admins": admin_user_ids,
        "members": admin_user_ids + member_user_ids,
        "members_with_roles": Json(members_with_roles),
    }
    if organization_id is not None:
        data["organization_id"] = organization_id
    if team_member_permissions is not None:
        data["team_member_permissions"] = team_member_permissions
    if models is not None:
        data["models"] = models
    if max_budget is not None:
        data["max_budget"] = max_budget
    if tpm_limit is not None:
        data["tpm_limit"] = tpm_limit
    if rpm_limit is not None:
        data["rpm_limit"] = rpm_limit
    if metadata is not None:
        data["metadata"] = Json(metadata)
    await prisma.db.litellm_teamtable.create(data=data)
    return team_id


async def create_scratch_org(
    prisma,
    scratch_prefix: str,
    *,
    max_budget: Optional[float] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    models: Optional[list] = None,
    metadata: Optional[dict] = None,
    suffix: str = "org",
) -> str:
    """Seed a scratch-tagged org + its own budget row; returns organization_id.

    The org's `budget_id` points at a fresh `litellm_budgettable` row that
    carries the per-org limits (`_check_org_key_limits` and the team budget
    helpers read `org_table.litellm_budget_table.<limit>`, not columns on the
    org row itself). Both rows share the scratch prefix so the teardown
    reclaims them — budget by `budget_id` prefix (already swept), org by
    `organization_id` prefix (added in this PR to the `scratch` fixture).

    models / metadata seed the matching org columns; `_check_org_team_limits`
    (F3) reads `org_table.models`, and the org metadata mirror of
    model_rpm_limit / model_tpm_limit is what F1's model-specific org guard
    consults.
    """
    org_id = f"{scratch_prefix}-{suffix}"
    budget_id = f"{scratch_prefix}-{suffix}-budget"
    budget_data: Dict[str, Any] = {
        "budget_id": budget_id,
        "created_by": "phase4-scratch",
        "updated_by": "phase4-scratch",
    }
    if max_budget is not None:
        budget_data["max_budget"] = max_budget
    if tpm_limit is not None:
        budget_data["tpm_limit"] = tpm_limit
    if rpm_limit is not None:
        budget_data["rpm_limit"] = rpm_limit
    await prisma.db.litellm_budgettable.create(data=budget_data)

    org_data: Dict[str, Any] = {
        "organization_id": org_id,
        "organization_alias": org_id,
        "budget_id": budget_id,
        "created_by": "phase4-scratch",
        "updated_by": "phase4-scratch",
    }
    if models is not None:
        org_data["models"] = models
    if metadata is not None:
        org_data["metadata"] = Json(metadata)
    await prisma.db.litellm_organizationtable.create(data=org_data)
    return org_id


@dataclass(frozen=True)
class SeededActor:
    user_id: str
    cleartext: str
    hashed: str


async def create_scratch_actor(
    prisma,
    scratch_prefix: str,
    *,
    user_role: str,
    org_admin_of: tuple = (),
    organization_id: Optional[str] = None,
    suffix: str = "actor",
) -> SeededActor:
    """Mint a scratch-prefixed user + verification token (+ org memberships).

    Reclaimed by the existing `scratch` teardown, which sweeps
    litellm_usertable, litellm_verificationtoken, and
    litellm_organizationmembership by scratch prefix — no bespoke cleanup
    needed. Does NOT write litellm_teammembership against world teams: the
    teardown reclaims that table only by team_id prefix, so a scratch actor
    needing team membership must join a scratch team instead. The cleartext
    is hashed with the real hash_token so the key authenticates end-to-end;
    models=[] satisfies LiteLLM_VerificationTokenView.
    """
    user_id = f"{scratch_prefix}-{suffix}"
    cleartext = "sk-" + uuid.uuid4().hex
    hashed = hash_token(cleartext)
    await prisma.db.litellm_usertable.create(
        data={
            "user_id": user_id,
            "user_role": user_role,
            "organization_id": organization_id,
        }
    )
    token_data: Dict[str, Any] = {
        "token": hashed,
        "key_name": f"{scratch_prefix}-{suffix}-key",
        "key_alias": f"{scratch_prefix}-{suffix}-alias",
        "user_id": user_id,
        "models": [],
    }
    if organization_id is not None:
        token_data["organization_id"] = organization_id
    await prisma.db.litellm_verificationtoken.create(data=token_data)
    for org_id in org_admin_of:
        await prisma.db.litellm_organizationmembership.create(
            data={
                "user_id": user_id,
                "organization_id": org_id,
                "user_role": "org_admin",
            }
        )
    return SeededActor(user_id=user_id, cleartext=cleartext, hashed=hashed)


@pytest_asyncio.fixture
async def scratch(prisma):
    handle = Scratch(prefix=f"{SCRATCH_PREFIX}{uuid.uuid4().hex[:12]}")
    try:
        yield handle
    finally:
        # Children before parents to avoid FK violations.
        await prisma.db.litellm_verificationtoken.delete_many(
            where={
                "OR": [
                    {"key_alias": {"startswith": handle.prefix}},
                    {"key_name": {"startswith": handle.prefix}},
                ]
            }
        )
        await prisma.db.litellm_teammembership.delete_many(
            where={"team_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_organizationmembership.delete_many(
            where={"user_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_teamtable.delete_many(
            where={"team_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_usertable.delete_many(
            where={"user_id": {"startswith": handle.prefix}}
        )
        # F1+F3 seed scratch orgs via create_scratch_org; the world seeder is
        # the only other writer of LiteLLM_OrganizationTable and uses the
        # behavior-pin- prefix, so a scratch-prefixed sweep here cannot
        # collide with the read-world. Org must be reclaimed BEFORE its
        # budget — org.budget_id → budget.budget_id, so deleting the parent
        # first would FK-violate on any still-attached scratch org.
        await prisma.db.litellm_organizationtable.delete_many(
            where={"organization_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_budgettable.delete_many(
            where={"budget_id": {"startswith": handle.prefix}}
        )
        # /team/member_add writes LiteLLM_UserTable.teams; the available-team
        # self-join writes it on a world actor whose row must survive. Strip
        # dangling scratch-team refs so the read-world stays immutable.
        polluted = await prisma.db.litellm_usertable.find_many(
            where={"teams": {"isEmpty": False}}
        )
        for user in polluted:
            cleaned = [t for t in user.teams if not t.startswith(handle.prefix)]
            if cleaned != list(user.teams):
                await prisma.db.litellm_usertable.update(
                    where={"user_id": user.user_id},
                    data={"teams": {"set": cleaned}},
                )
