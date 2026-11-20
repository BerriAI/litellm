from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_ObjectPermissionBase, NewOrganizationRequest
from litellm.proxy.management_helpers.mcp_auth_profile_grants import (
    enforce_auth_profile_grant_write,
)
from litellm.proxy.mcp_server.db import LOADABLE_MCP_SERVER_FILTER
from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile, MCPServer
from tests.unit.proxy.mcp_prisma_fake import FakeRow

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _server(server_id: str, *profile_ids: str) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=server_id,
        url=f"https://{server_id}.example/mcp",
        transport=MCPTransport.http,
        additional_auth=tuple(
            MCPAuthProfile(id=profile_id, auth_type=MCPAuth.bearer_token, authentication_token="t")
            for profile_id in profile_ids
        ),
    )


class _PermissionTable:
    def __init__(self, stored: dict[str, list[str]], granted_rows: tuple[SimpleNamespace, ...] = ()) -> None:
        self.stored = stored
        self.granted_rows = granted_rows
        self.grant_queries: tuple[object, ...] = ()

    async def find_unique(self, where: dict[str, str]) -> SimpleNamespace | None:
        grants: Final = self.stored.get(where["object_permission_id"])
        return None if grants is None else SimpleNamespace(mcp_auth_profiles=grants)

    async def find_many(self, where: object, include: object) -> tuple[SimpleNamespace, ...]:
        self.grant_queries = (*self.grant_queries, where)
        return self.granted_rows


class _ProfileTable:
    def __init__(self, rows: tuple[SimpleNamespace, ...] = ()) -> None:
        self.rows = rows
        self.lookups: tuple[object, ...] = ()

    async def find_many(self, where: dict[str, object]) -> tuple[SimpleNamespace, ...]:
        self.lookups = (*self.lookups, where)
        wanted: Final = where["profile_id_normalized"]["in"]
        return tuple(row for row in self.rows if row.profile_id_normalized in wanted)


def _prisma(
    table: _PermissionTable,
    profiles: _ProfileTable | None = None,
    replica: _PermissionTable | None = None,
    profile_replica: _ProfileTable | None = None,
) -> SimpleNamespace:
    profile_table: Final = profiles or _ProfileTable()
    return SimpleNamespace(
        db=SimpleNamespace(
            litellm_objectpermissiontable=replica or table, litellm_mcpauthprofiletable=profile_replica or profile_table
        ),
        writer_db=SimpleNamespace(litellm_objectpermissiontable=table, litellm_mcpauthprofiletable=profile_table),
    )


def _stored_profile(profile_id: str, server_id: str) -> SimpleNamespace:
    return SimpleNamespace(profile_id=profile_id, profile_id_normalized=profile_id.lower(), server_id=server_id)


@pytest.fixture(autouse=True)
def registered_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    for server in (
        _server("analytics", "ana-bots", "ana-ci"),
        _server("github", "gh-bots"),
        _server("news", "News-Bots"),
    ):
        monkeypatch.setitem(global_mcp_server_manager.config_mcp_servers, server.server_id, server)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, None),
        ([], ()),
        (["ANA-Bots", "gh-bots"], ("ana-bots", "gh-bots")),
        (["ana-bots", "ANA-BOTS"], ("ana-bots",)),
    ],
)
async def test_an_admin_grant_is_stored_as_the_registered_profile_ids(
    requested: list[str] | None, expected: tuple[str, ...] | None
) -> None:
    permission: Final = {} if requested is None else {"mcp_auth_profiles": requested}
    assert await enforce_auth_profile_grant_write(permission, None, True, None) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "named"),
    [
        (["nope"], "nope"),
        (["ana-bots", "ana-ci"], "analytics (ana-bots, ana-ci)"),
        ("ana-bots", "list of MCP auth profile ids"),
    ],
)
async def test_an_unknown_id_or_two_profiles_of_one_server_is_refused(requested: object, named: str) -> None:
    with pytest.raises(HTTPException) as refused:
        await enforce_auth_profile_grant_write({"mcp_auth_profiles": requested}, None, True, None)

    assert refused.value.status_code == 400
    assert named in refused.value.detail["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "requested"), [([], []), (["ana-bots"], ["ANA-BOTS"])], ids=["resends-empty", "resends-stored"]
)
async def test_anyone_may_resend_the_stored_grant_unchanged(stored: list[str], requested: list[str]) -> None:
    prisma: Final = _prisma(_PermissionTable({"perm-key": stored}))
    assert await enforce_auth_profile_grant_write({"mcp_auth_profiles": requested}, "perm-key", False, prisma) == tuple(
        stored
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "requested"),
    [([], ["ana-bots"]), (["ana-bots"], []), (["ana-bots"], ["ana-ci"])],
    ids=["adds", "clears", "switches"],
)
async def test_only_a_proxy_admin_changes_a_grant(stored: list[str], requested: list[str]) -> None:
    prisma: Final = _prisma(_PermissionTable({"perm-key": stored}))
    with pytest.raises(HTTPException) as refused:
        await enforce_auth_profile_grant_write({"mcp_auth_profiles": requested}, "perm-key", False, prisma)
    assert refused.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested",
    [["no-such-profile"], ["slack-bots"], ["ana-bots", "ana-ci"]],
    ids=["unknown", "stored-elsewhere", "two-of-one-server"],
)
async def test_a_non_admin_change_is_refused_alike_whatever_the_ids_name(requested: list[str]) -> None:
    profiles: Final = _ProfileTable((_stored_profile("Slack-Bots", "slack"),))

    with pytest.raises(HTTPException) as refused:
        await enforce_auth_profile_grant_write(
            {"mcp_auth_profiles": requested},
            "perm-key",
            False,
            cast("PrismaClient", _prisma(_PermissionTable({"perm-key": []}), profiles)),
        )

    assert (refused.value.status_code, refused.value.detail, profiles.lookups) == (
        403,
        {"error": "Only a proxy admin can change mcp_auth_profiles."},
        (),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_proxy_admin", [True, False])
async def test_a_grant_naming_too_many_ids_is_refused_before_any_read(is_proxy_admin: bool) -> None:
    profiles: Final = _ProfileTable()

    with pytest.raises(HTTPException) as refused:
        await enforce_auth_profile_grant_write(
            {"mcp_auth_profiles": [f"bots-{index}" for index in range(257)]},
            None,
            is_proxy_admin,
            cast("PrismaClient", _prisma(_PermissionTable({}), profiles)),
        )

    assert (refused.value.status_code, profiles.lookups) == (400, ())
    assert "at most 256" in str(refused.value.detail)


@pytest.mark.asyncio
async def test_a_non_admin_resend_is_compared_with_the_writer_not_a_lagging_replica() -> None:
    revoked_on_the_writer: Final = _prisma(
        _PermissionTable({"perm-key": []}), replica=_PermissionTable({"perm-key": ["ana-bots"]})
    )

    with pytest.raises(HTTPException) as refused:
        await enforce_auth_profile_grant_write(
            {"mcp_auth_profiles": ["ana-bots"]}, "perm-key", False, revoked_on_the_writer
        )

    assert refused.value.status_code == 403


@pytest.mark.asyncio
async def test_key_writes_store_the_canonical_grant() -> None:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_VerificationToken, UpdateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import _validate_mcp_servers_for_key_update

    validated: Final = await _validate_mcp_servers_for_key_update(
        data=UpdateKeyRequest(
            key="sk-nightly-bot", object_permission=LiteLLM_ObjectPermissionBase(mcp_auth_profiles=["ANA-Bots"])
        ),
        team_obj=None,
        existing_key_row=LiteLLM_VerificationToken(token="hashed-bot-key"),
        prisma_client=None,
        user_api_key_cache=DualCache(),
        is_proxy_admin=True,
    )
    assert validated == {"mcp_auth_profiles": ["ana-bots"]}


@pytest.mark.asyncio
async def test_an_organization_cannot_carry_a_grant() -> None:
    from litellm.proxy.management_endpoints.organization_endpoints import _set_object_permission

    with pytest.raises(HTTPException) as refused:
        await _set_object_permission(
            NewOrganizationRequest(
                organization_alias="payments-org",
                object_permission=LiteLLM_ObjectPermissionBase(mcp_auth_profiles=["ana-bots"]),
            ),
            None,
        )
    assert refused.value.status_code == 400


@pytest.mark.asyncio
async def test_team_writes_store_the_canonical_grant() -> None:
    from litellm.proxy.management_endpoints.team_endpoints import _normalize_team_auth_profile_grants

    team_write: Final[dict[str, object]] = {"object_permission": {"mcp_auth_profiles": ["GH-Bots"]}}
    await _normalize_team_auth_profile_grants(team_write, None, True, None)
    assert team_write == {"object_permission": {"mcp_auth_profiles": ["gh-bots"]}}


@pytest.mark.asyncio
async def test_a_team_edit_by_a_non_admin_resending_the_stored_grant_leaves_the_grant_unwritten() -> None:
    from litellm.proxy.management_endpoints.team_endpoints import _normalize_team_auth_profile_grants

    team_write: Final[dict[str, object]] = {
        "object_permission": {"mcp_servers": ["github"], "mcp_auth_profiles": ["GH-Bots"]}
    }
    stored: Final = _prisma(_PermissionTable({"perm-team": ["gh-bots"]}))

    await _normalize_team_auth_profile_grants(team_write, "perm-team", False, stored)

    assert team_write == {"object_permission": {"mcp_servers": ["github"]}}


@pytest.mark.asyncio
@pytest.mark.parametrize(("requested", "status_code"), [(["nope"], 400), (["ana-bots", "ana-ci"], 400)])
async def test_agent_writes_refuse_a_bad_grant(requested: list[str], status_code: int) -> None:
    from litellm.proxy.agent_endpoints.agent_registry import _permission_write

    with pytest.raises(HTTPException) as refused:
        await _permission_write(
            {"object_permission": {"mcp_auth_profiles": requested}}, None, _prisma(_PermissionTable({}))
        )
    assert refused.value.status_code == status_code


class _AgentTable:
    def __init__(self) -> None:
        self.writes: tuple[str, ...] = ()

    async def create(self, **kwargs: object) -> None:
        self.writes = (*self.writes, "create")

    async def update(self, **kwargs: object) -> None:
        self.writes = (*self.writes, "update")

    async def find_unique(self, **kwargs: object) -> FakeRow:
        return FakeRow(agent_id="nightly-agent", agent_name="nightly-agent", object_permission_id=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "patch"])
async def test_a_refused_agent_grant_writes_no_agent_row(operation: str) -> None:
    from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry

    agents: Final = _AgentTable()
    prisma: Final = _prisma(_PermissionTable({}))
    prisma.db.litellm_agentstable = agents
    agent: Final = {"agent_name": "nightly-agent", "object_permission": {"mcp_auth_profiles": ["nope"]}}

    write: Final = (
        AgentRegistry().add_agent_to_db(agent, prisma, created_by="admin")
        if operation == "create"
        else AgentRegistry().patch_agent_in_db("nightly-agent", agent, prisma, updated_by="admin")
    )
    with pytest.raises(HTTPException) as refused:
        await write

    assert (refused.value.status_code, agents.writes) == (400, ())


@pytest.mark.asyncio
async def test_agent_writes_store_the_canonical_grant() -> None:
    from litellm.proxy.agent_endpoints.agent_registry import _permission_write

    write: Final = await _permission_write(
        {"object_permission": {"mcp_auth_profiles": ["ANA-Bots"]}}, None, _prisma(_PermissionTable({}))
    )
    assert write is not None and write["create"]["mcp_auth_profiles"] == ["ana-bots"]


@pytest.mark.asyncio
async def test_a_key_edit_by_its_owner_may_resend_the_stored_grant_which_stays_unwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_VerificationToken, UpdateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import _validate_mcp_servers_for_key_update

    validated: Final = await _validate_mcp_servers_for_key_update(
        data=UpdateKeyRequest(
            key="sk-nightly-bot", object_permission=LiteLLM_ObjectPermissionBase(mcp_auth_profiles=["ana-bots"])
        ),
        team_obj=None,
        existing_key_row=LiteLLM_VerificationToken(token="hashed-bot-key", object_permission_id="perm-key"),
        prisma_client=_prisma(_PermissionTable({"perm-key": ["ana-bots"]})),
        user_api_key_cache=DualCache(),
        is_proxy_admin=False,
    )
    assert validated == {}


@pytest.mark.asyncio
async def test_a_grant_is_checked_against_the_profile_table_not_only_this_workers_registry() -> None:
    profiles: Final = _ProfileTable((_stored_profile("Slack-Bots", "slack"), _stored_profile("other", "slack")))

    granted: Final = await enforce_auth_profile_grant_write(
        {"mcp_auth_profiles": ["slack-bots", "gh-bots"]}, None, True, _prisma(_PermissionTable({}), profiles)
    )

    assert granted == ("Slack-Bots", "gh-bots")
    assert profiles.lookups == (
        {"profile_id_normalized": {"in": ["slack-bots"]}, "server": {"is": LOADABLE_MCP_SERVER_FILTER}},
    )


@pytest.mark.asyncio
async def test_a_profile_created_moments_ago_can_be_granted_before_the_replica_has_it() -> None:
    created_on_the_writer: Final = _ProfileTable((_stored_profile("New-Bots", "tickets"),))

    granted: Final = await enforce_auth_profile_grant_write(
        {"mcp_auth_profiles": ["new-bots"]},
        None,
        True,
        _prisma(_PermissionTable({}), created_on_the_writer, profile_replica=_ProfileTable()),
    )

    assert granted == ("New-Bots",)


@pytest.mark.asyncio
async def test_a_profile_of_a_server_the_registry_does_not_load_cannot_be_granted() -> None:
    with pytest.raises(HTTPException) as refused:
        await enforce_auth_profile_grant_write(
            {"mcp_auth_profiles": ["pending-bots"]}, None, True, _prisma(_PermissionTable({}), _ProfileTable())
        )

    assert (refused.value.status_code, refused.value.detail) == (
        400,
        {"error": "Unknown MCP auth profile ids in mcp_auth_profiles: pending-bots"},
    )


@pytest.mark.asyncio
async def test_a_grant_naming_only_config_profiles_reads_no_profile_rows() -> None:
    profiles: Final = _ProfileTable()

    await enforce_auth_profile_grant_write(
        {"mcp_auth_profiles": ["ANA-Bots"]}, None, True, _prisma(_PermissionTable({}), profiles)
    )

    assert profiles.lookups == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["customer", "project"])
async def test_customers_and_projects_cannot_carry_a_grant(carrier: str) -> None:
    from litellm.proxy.management_helpers.mcp_auth_profile_grants import refuse_auth_profile_grants

    with pytest.raises(HTTPException) as refused:
        refuse_auth_profile_grants(carrier, {"mcp_auth_profiles": ["ana-bots"]})

    assert refused.value.detail == {
        "error": f"mcp_auth_profiles can be set on keys, teams and agents, not on a {carrier}."
    }


def test_a_grant_serialized_as_json_is_still_refused_on_a_user() -> None:
    from litellm.proxy.management_helpers.mcp_auth_profile_grants import refuse_auth_profile_grants

    with pytest.raises(HTTPException) as refused:
        refuse_auth_profile_grants("user", '{"mcp_auth_profiles": ["ana-bots"]}')

    assert refused.value.status_code == 400
