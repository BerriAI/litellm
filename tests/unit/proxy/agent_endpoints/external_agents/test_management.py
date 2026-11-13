import sqlite3
from collections.abc import Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from prisma import Prisma

from litellm.proxy.agent_endpoints.external_agents.cache import GovernanceMetadata
from litellm.proxy.agent_endpoints.external_agents.governance import ExternalAgentStore
from litellm.proxy.agent_endpoints.external_agents.management import ExternalAgentManagement
from litellm.proxy.agent_endpoints.external_agents.members import sync_members
from litellm.proxy.agent_endpoints.external_agents.models import (
    ActorMapping,
    ActorsPut,
    MappedUser,
    Policy,
    PolicyPut,
    Source,
    SourceCreate,
    SourcePatch,
    HookFailure,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

SOURCE: Final = Source(source_id="source", display_name="Bots", signing_secret="encrypted", tenant_id="tenant")


def database_with_rows(rows: Sequence[object]) -> MagicMock:
    db: Final = MagicMock(spec=Prisma)
    db.query_raw = AsyncMock(side_effect=rows)
    db.execute_raw = AsyncMock(return_value=1)
    db.tx.return_value.__aenter__ = AsyncMock(return_value=db)
    db.tx.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


@pytest.mark.asyncio
async def test_source_patch_refreshes_cached_authentication_and_mode() -> None:
    cache: Final = GovernanceMetadata()
    cache.sources.put("source", SOURCE)
    updated: Final = SOURCE.model_copy(update={"mode": "enforce", "signing_secret": "rotated"})
    row: Final = updated.model_dump() | {"signing_secret": "rotated"}
    db: Final = database_with_rows([[row], [row]])
    saved: Final = await ExternalAgentManagement(db, cache).patch_source("source", SourcePatch(mode="enforce"), "admin")
    assert not isinstance(saved, HookFailure)
    assert saved.mode == "enforce"
    assert await ExternalAgentStore(db, cache).source("source") == updated


@pytest.mark.asyncio
async def test_policy_write_invalidates_cached_allow_all_policy() -> None:
    cache: Final = GovernanceMetadata()
    cache.policies.put(("source", None, None), None)
    policy: Final = Policy(policy_id="policy", source_id="source", blocked_tools=("Shell",))
    db: Final = database_with_rows([[{"source_id": "source"}], [], [policy.model_dump()], [policy.model_dump()]])
    assert (
        await ExternalAgentManagement(db, cache).put_policy(PolicyPut(source_id="source", blocked_tools=("Shell",)))
        == policy
    )
    assert await ExternalAgentStore(db, cache).policy(SOURCE, None) == policy


@pytest.mark.asyncio
async def test_actor_write_invalidates_cached_unmapped_actor() -> None:
    cache: Final = GovernanceMetadata()
    cache.actors.put(("source", "actor"), None)
    user: Final = MappedUser(user_id="user", teams=("team",))
    db: Final = database_with_rows([[{"source_id": "source"}], [{"user_id": "user"}], [user.model_dump()]])
    assert (
        await ExternalAgentManagement(db, cache).actors(
            ActorsPut(source_id="source", mappings=(ActorMapping(external_actor_id="actor", user_id="user"),))
        )
        == 1
    )
    assert await ExternalAgentStore(db, cache).actor("source", "actor") == user


@pytest.mark.asyncio
@pytest.mark.parametrize("populated", (False, True))
async def test_summary_counts_enforced_denials_and_distinct_unmapped_members(populated: bool) -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE "LiteLLM_ExternalAgent" (
                source_id TEXT, team_id TEXT, owner_user_id TEXT, owner_inferred BOOLEAN,
                last_actor_id TEXT, last_seen_at TEXT, external_agent_id TEXT DEFAULT 'bot'
            );
            CREATE TABLE "LiteLLM_ExternalAgentPolicy" (source_id TEXT, team_id TEXT);
            CREATE TABLE "LiteLLM_ExternalAgentActor" (source_id TEXT, external_actor_id TEXT, user_id TEXT);
            CREATE TABLE "LiteLLM_UserTable" (user_id TEXT);
            CREATE TABLE "LiteLLM_ExternalAgentDecision" (
                source_id TEXT, actor_id TEXT, verdict TEXT, would_verdict TEXT, enforced BOOLEAN, created_at TEXT,
                external_agent_id TEXT DEFAULT 'bot'
            );
        """)
        if populated:
            connection.executescript("""
                INSERT INTO "LiteLLM_ExternalAgent" (source_id,team_id,owner_user_id,owner_inferred,last_actor_id,last_seen_at) VALUES
                    ('source', NULL, NULL, false, 'same-member', NULL),
                    ('source', NULL, NULL, false, 'same-member', NULL);
                INSERT INTO "LiteLLM_UserTable" VALUES ('user');
                INSERT INTO "LiteLLM_ExternalAgentActor" VALUES
                    ('source', 'mapped', 'user'), ('source', 'deleted-user', 'missing-user');
                INSERT INTO "LiteLLM_ExternalAgentDecision" (source_id,actor_id,verdict,would_verdict,enforced,created_at) VALUES
                    ('source', 'same-member', 'allow', 'deny', false, '2026-01-02'),
                    ('source', 'same-member', 'deny', 'deny', true, '2026-01-02'),
                    ('source', 'older-member', 'deny', 'deny', true, '2025-12-31'),
                    ('source', 'mapped', 'allow', 'allow', true, '2026-01-02'),
                    ('source', 'deleted-user', 'allow', 'allow', false, '2026-01-02'),
                    ('source', NULL, 'allow', 'allow', false, '2026-01-02'),
                    ('other-source', 'other-member', 'deny', 'deny', true, '2026-01-02');
            """)

        async def query_rows(query: str, source_id: str) -> tuple[Mapping[str, object], ...]:
            statement: Final = query.replace("::int", "").replace("NOW()-INTERVAL '24 hours'", "'2026-01-01'")
            return tuple(dict(row) for row in connection.execute(statement, {"1": source_id}))

        db: Final = database_with_rows(())
        db.query_raw = AsyncMock(side_effect=query_rows)
        summary: Final = await ExternalAgentManagement(db, GovernanceMetadata()).summary("source")
        assert summary.model_dump() == {
            "discovered": 2 if populated else 0,
            "covered_by_team_policy": 0,
            "needs_owner": 2 if populated else 0,
            "unmapped_actors": 3 if populated else 0,
            "denied_24h": 1 if populated else 0,
            "last_event_at": None,
            "needs_attention": 2 if populated else 0,
        }


class MemberClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response

    async def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        assert url == "https://api.cursor.com/teams/members"
        assert headers == {"Authorization": "Basic dGVzdC1rZXk6"}
        assert timeout == 5.0
        return self.response


@pytest.mark.asyncio
async def test_member_sync_matches_email_and_preserves_manual_mappings() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE "LiteLLM_UserTable" (user_id TEXT, user_email TEXT);
            CREATE TABLE "LiteLLM_ExternalAgentActor" (
                id TEXT, source_id TEXT, external_actor_id TEXT, user_id TEXT, UNIQUE(source_id,external_actor_id)
            );
            INSERT INTO "LiteLLM_UserTable" VALUES ('match', 'Member@Example.test'), ('removed', 'removed@example.test');
            INSERT INTO "LiteLLM_ExternalAgentActor" VALUES ('manual','source','existing','manual-user');
        """)

        async def query_rows(query: str, *args: object) -> tuple[Mapping[str, object], ...]:
            parameters: Final = {str(index + 1): value for index, value in enumerate(args)}
            return tuple(dict(row) for row in connection.execute(query, parameters))

        async def execute(query: str, *args: object) -> int:
            parameters: Final = {str(index + 1): value for index, value in enumerate(args)}
            return connection.execute(query, parameters).rowcount

        db: Final = database_with_rows(())
        db.query_raw = AsyncMock(side_effect=query_rows)
        db.execute_raw = AsyncMock(side_effect=execute)
        response: Final = httpx.Response(
            200,
            request=httpx.Request("GET", "https://api.cursor.com/teams/members"),
            json={
                "teamMembers": [
                    {"id": "actor", "email": "member@example.TEST"},
                    {"id": "unknown", "email": "unknown@example.test"},
                    {"id": "existing", "email": "member@example.test"},
                    {"id": "removed", "email": "removed@example.test", "isRemoved": True},
                ],
            },
        )
        result: Final = await sync_members(db, MemberClient(response), "source", "test-key")
        assert not isinstance(result, HookFailure)
        assert result.model_dump() == {"matched": 1, "unmatched": 1, "already_mapped": 1}
        assert tuple(
            tuple(row)
            for row in connection.execute(
                'SELECT external_actor_id,user_id FROM "LiteLLM_ExternalAgentActor" ORDER BY external_actor_id'
            )
        ) == (("actor", "match"), ("existing", "manual-user"))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (200, 401))
async def test_member_sync_empty_response_or_failed_credentials(status: int) -> None:
    client: Final = MemberClient(
        httpx.Response(
            status, request=httpx.Request("GET", "https://api.cursor.com/teams/members"), json={"teamMembers": []}
        )
    )
    result: Final = await sync_members(database_with_rows(()), client, "source", "test-key")
    if status == 401:
        assert result == HookFailure(
            status=502, reason="Member sync failed. Check the GrokBot Admin API key and retry."
        )
        return
    assert not isinstance(result, HookFailure)
    assert result.model_dump() == {"matched": 0, "unmatched": 0, "already_mapped": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("creating", (False, True))
async def test_saving_admin_key_automatically_syncs_members(creating: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "external-agent-unit-test-key")
    row: Final = SOURCE.model_dump() | {"admin_api_key": encrypt_value_helper("test-key")}
    db: Final = database_with_rows(([row], [row], []))
    client: Final = MemberClient(
        httpx.Response(
            200,
            request=httpx.Request("GET", "https://api.cursor.com/teams/members"),
            json={
                "teamMembers": [{"id": "member", "email": "member@example.test"}],
            },
        )
    )
    service: Final = ExternalAgentManagement(db, client=client)
    result: Final = (
        await service.create_source(SourceCreate(admin_api_key="test-key"), "admin")
        if creating
        else await service.patch_source("source", SourcePatch(admin_api_key="test-key"), "admin")
    )
    assert not isinstance(result, HookFailure)
    assert result.sync_result is not None
    assert result.sync_result.model_dump() == {"matched": 1, "unmatched": 0, "already_mapped": 0}
