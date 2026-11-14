import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from mcp.types import ListToolsRequest

from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _team_membership_inflight,
    get_key_object,
    get_object_permission,
    get_team_membership,
    get_team_object,
    get_user_object,
)
from litellm.proxy.common_utils.strict_cache_refill import strict_cache_refill
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server import catalog, operations
from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler, UnloadableEntitlementError
from litellm.models.mcp_policy import PolicySubject, ReachableObjects
from litellm.proxy.mcp_server.policy_freshness import (
    RequestFreshness,
    ValidatedPolicyMarks,
    cached_policy_is_current,
    cached_policy_keys,
    current_listing,
    policy_subject_for,
    validated_listing,
    validated_policy_marks,
)
from litellm.proxy.mcp_server import policy_freshness as policy_freshness_module
from litellm.repositories.mcp_policy_repository import MCPPolicyRepository
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

KEY: Final = PolicySubject(key_hash="key-hash")
CACHE_TTL: Final = 600
CACHED: Final = {"cached": True}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def fingerprint_row(
    digest: str,
    *,
    uses_toolsets: bool = False,
    team_ids: tuple[str, ...] = ("team",),
    access_group_ids: tuple[str, ...] = (),
) -> Mapping[str, object]:
    return {
        "digest": digest,
        "uses_toolsets": uses_toolsets,
        "reachable": {
            "user_id": "owner",
            "project_id": None,
            "team_ids": list(team_ids),
            "organization_ids": [],
            "permission_ids": ["permission"],
            "access_group_ids": list(access_group_ids),
        },
    }


def database(*rows: Mapping[str, object]) -> MagicMock:
    client: Final = MagicMock()
    client.writer_db.query_raw = AsyncMock(side_effect=[[row] for row in rows])
    client.db.query_raw = AsyncMock(side_effect=AssertionError("a read replica must not serve the policy hash"))
    return client


@pytest.fixture(autouse=True)
def settled_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_freshness_module, "_SETTLE_SECONDS", 0.0)


@pytest.fixture
def cache() -> UserApiKeyCache:
    return UserApiKeyCache(default_in_memory_ttl=CACHE_TTL)


async def is_current(
    client: MagicMock, cache: UserApiKeyCache, marks: ValidatedPolicyMarks, subject: PolicySubject | None = KEY
) -> bool:
    return await cached_policy_is_current(subject, prisma_client=client, cache=cache, marks=marks)


def test_marks_match_only_the_recorded_digest_and_stay_bounded() -> None:
    marks: Final = ValidatedPolicyMarks(capacity=2)
    first: Final = PolicySubject(key_hash="a")
    second: Final = PolicySubject(key_hash="b")
    third: Final = PolicySubject(key_hash="c")
    assert not marks.matches(first, "d1", max_age=60)
    marks.record(first, "d1")
    marks.record(second, "d2")
    assert marks.matches(first, "d1", max_age=60)
    assert not marks.matches(first, "other", max_age=60)
    marks.record(third, "d3")
    assert marks.phase(second, "d2") == "other"
    assert marks.matches(first, "d1", max_age=60)
    assert marks.matches(third, "d3", max_age=60)


def test_a_mark_is_trusted_only_for_its_max_age_but_its_digest_is_still_recorded() -> None:
    clock: Final = Clock()
    marks: Final = ValidatedPolicyMarks(clock=clock)
    marks.record(KEY, "d")
    clock.now += 299
    assert marks.matches(KEY, "d", max_age=300)
    clock.now += 2
    assert not marks.matches(KEY, "d", max_age=300)
    assert marks.phase(KEY, "d") == "settled"


@pytest.mark.parametrize(
    "other",
    [
        PolicySubject(user_id="u", pinned_team_id="t2", pinned_org_id="o"),
        PolicySubject(user_id="u", pinned_team_id="t", pinned_org_id="o2"),
        PolicySubject(user_id="u2", pinned_team_id="t", pinned_org_id="o"),
    ],
)
def test_a_keyless_mark_depends_on_the_user_team_and_organization(other: PolicySubject) -> None:
    marks: Final = ValidatedPolicyMarks()
    marks.record(PolicySubject(user_id="u", pinned_team_id="t", pinned_org_id="o"), "digest")
    assert marks.matches(PolicySubject(user_id="u", pinned_team_id="t", pinned_org_id="o"), "digest", max_age=60)
    assert not marks.matches(other, "digest", max_age=60)


def test_every_cache_family_is_evicted_including_ids_whose_rows_are_gone() -> None:
    reachable: Final = ReachableObjects(
        user_id="owner",
        project_id="project",
        team_ids=("live-team", "deleted-team"),
        organization_ids=("org",),
        permission_ids=("permission",),
        access_group_ids=("live-group", "deleted-group"),
    )
    assert set(cached_policy_keys(KEY, reachable)) == {
        "key-hash",
        "owner",
        "user_object_permission_id:owner",
        "team_id:live-team",
        "team_id:deleted-team",
        "live-team_owner",
        "deleted-team_owner",
        "team_membership:owner:live-team",
        "team_membership:owner:deleted-team",
        "org_id:org",
        "org_id:org:with_budget",
        "org_id:org:with_budget:last_known",
        "project_id:project",
        "object_permission_id:permission",
        "access_group_id:live-group",
        "access_group_id:deleted-group",
    }


def test_a_keyless_subject_evicts_no_key_entry() -> None:
    reachable: Final = ReachableObjects()
    assert cached_policy_keys(PolicySubject(user_id="u"), reachable) == ("u", "user_object_permission_id:u")


@pytest.mark.asyncio
async def test_entries_written_by_the_real_readers_are_gone_after_a_changed_digest(cache: UserApiKeyCache) -> None:
    permission_lookup: Final = AsyncMock(return_value=LiteLLM_ObjectPermissionTable(object_permission_id="permission"))
    user_lookup: Final = AsyncMock(return_value=LiteLLM_UserTable(user_id="owner"))
    reader: Final = MagicMock()
    reader.db.litellm_objectpermissiontable.find_unique = permission_lookup
    reader.db.litellm_usertable.find_unique = user_lookup
    await get_object_permission("permission", reader, cache)
    await get_user_object("owner", reader, cache, user_id_upsert=False)
    await get_object_permission("permission", reader, cache)
    await get_user_object("owner", reader, cache, user_id_upsert=False)
    assert (permission_lookup.await_count, user_lookup.await_count) == (1, 1)
    async with validated_listing():
        await is_current(database(fingerprint_row("v1"), fingerprint_row("v1")), cache, ValidatedPolicyMarks())
    await get_object_permission("permission", reader, cache)
    await get_user_object("owner", reader, cache, user_id_upsert=False)
    assert (permission_lookup.await_count, user_lookup.await_count) == (2, 2)


@pytest.mark.asyncio
async def test_the_fingerprint_is_read_from_the_writer_with_the_subject_bound() -> None:
    client: Final = database(fingerprint_row("abc", team_ids=("t",), access_group_ids=("g",)))
    subject: Final = PolicySubject(key_hash="h", user_id="u", pinned_team_id="t", pinned_org_id="o")
    result: Final = await MCPPolicyRepository(client).get_fingerprint(subject)
    assert (result.digest, result.uses_toolsets) == ("abc", False)
    assert result.reachable.team_ids == ("t",)
    assert result.reachable.access_group_ids == ("g",)
    assert client.writer_db.query_raw.await_args.args[1:] == ("h", "u", "t", "o")


@pytest.mark.asyncio
async def test_the_fingerprint_accepts_a_json_text_reachable_column() -> None:
    reachable_text: Final = (
        '{"user_id": "u", "project_id": null, "team_ids": ["t"], "organization_ids": [], '
        '"permission_ids": ["p"], "access_group_ids": []}'
    )
    client: Final = database({"digest": "abc", "uses_toolsets": False, "reachable": reachable_text})
    assert (await MCPPolicyRepository(client).get_fingerprint(KEY)).reachable.permission_ids == ("p",)


@pytest.mark.asyncio
async def test_a_subject_that_cannot_be_fingerprinted_reads_the_database(cache: UserApiKeyCache) -> None:
    client: Final = database(fingerprint_row("v1"))
    async with validated_listing():
        assert not await is_current(client, cache, ValidatedPolicyMarks(), subject=None)
    client.writer_db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_check_outside_a_listing_reads_the_database(cache: UserApiKeyCache) -> None:
    client: Final = database(fingerprint_row("v1"))
    marks: Final = ValidatedPolicyMarks()
    marks.record(KEY, "v1")
    assert not await is_current(client, cache, marks)
    client.writer_db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_missing_database_reads_the_database(cache: UserApiKeyCache) -> None:
    async with validated_listing():
        assert not await cached_policy_is_current(KEY, prisma_client=None, cache=cache, marks=ValidatedPolicyMarks())


@pytest.mark.asyncio
async def test_a_failing_fingerprint_fails_closed_and_leaves_the_cache_alone(cache: UserApiKeyCache) -> None:
    client: Final = MagicMock()
    client.writer_db.query_raw = AsyncMock(side_effect=RuntimeError("db down"))
    marks: Final = ValidatedPolicyMarks()
    marks.record(KEY, "v1")
    await cache.async_set_cache(key="team_id:team", value=CACHED)
    async with validated_listing():
        assert not await is_current(client, cache, marks)
    assert await cache.async_get_cache(key="team_id:team") == CACHED


@pytest.mark.asyncio
async def test_a_first_listing_evicts_the_callers_entries_and_a_completed_one_validates_the_digest(
    cache: UserApiKeyCache,
) -> None:
    marks: Final = ValidatedPolicyMarks()
    client: Final = database(*(fingerprint_row("v1"),) * 5)
    await cache.async_set_cache(key="team_id:team", value=CACHED)
    await cache.async_set_cache(key="team_id:someone-else", value=CACHED)
    async with validated_listing():
        assert not await is_current(client, cache, marks)
        assert not strict_cache_refill()
    assert await cache.async_get_cache(key="team_id:team") is None
    assert await cache.async_get_cache(key="team_id:someone-else") == CACHED
    async with validated_listing():
        assert not await is_current(client, cache, marks)
    async with validated_listing():
        assert await is_current(client, cache, marks)
        assert strict_cache_refill()
    assert not strict_cache_refill()


@pytest.mark.asyncio
async def test_a_refill_that_lands_after_the_first_eviction_is_evicted_again_before_the_cache_is_trusted(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy_freshness_module, "_SETTLE_SECONDS", 5.0)
    clock: Final = Clock()
    marks: Final = ValidatedPolicyMarks(clock=clock)
    client: Final = database(*(fingerprint_row("v1"),) * 6)
    async with validated_listing():
        assert not await is_current(client, cache, marks)
    await cache.async_set_cache(key="team_id:team", value=CACHED)
    clock.now += 1
    async with validated_listing():
        assert not await is_current(client, cache, marks)
        assert not strict_cache_refill()
    assert await cache.async_get_cache(key="team_id:team") == CACHED
    clock.now += 5
    async with validated_listing():
        assert not await is_current(client, cache, marks)
        assert not strict_cache_refill()
    assert await cache.async_get_cache(key="team_id:team") is None
    async with validated_listing():
        assert await is_current(client, cache, marks)


@pytest.mark.asyncio
async def test_a_listing_that_fails_does_not_validate_the_digest(cache: UserApiKeyCache) -> None:
    marks: Final = ValidatedPolicyMarks()
    client: Final = database(fingerprint_row("v1"), fingerprint_row("v1"), fingerprint_row("v1"))

    async def failing_listing() -> None:
        async with validated_listing():
            assert not await is_current(client, cache, marks)
            raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError):
        await failing_listing()
    async with validated_listing():
        assert not await is_current(client, cache, marks)


@pytest.mark.asyncio
async def test_a_change_made_during_the_reload_is_not_marked_valid(cache: UserApiKeyCache) -> None:
    marks: Final = ValidatedPolicyMarks()
    client: Final = database(fingerprint_row("v1"), fingerprint_row("v2"), fingerprint_row("v1"))
    async with validated_listing():
        assert not await is_current(client, cache, marks)
    assert marks.phase(KEY, "v1") == "other"
    async with validated_listing():
        assert not await is_current(client, cache, marks)


@pytest.mark.asyncio
async def test_a_changed_digest_evicts_again_and_stops_trusting_the_cache(cache: UserApiKeyCache) -> None:
    marks: Final = ValidatedPolicyMarks()
    marks.record(KEY, "v1")
    client: Final = database(fingerprint_row("v2"))
    await cache.async_set_cache(key="team_id:team", value=CACHED)
    async with validated_listing():
        assert not await is_current(client, cache, marks)
        assert not strict_cache_refill()
    assert await cache.async_get_cache(key="team_id:team") is None


@pytest.mark.asyncio
async def test_an_expired_validation_reads_the_database_but_keeps_the_shared_entries(cache: UserApiKeyCache) -> None:
    clock: Final = Clock()
    marks: Final = ValidatedPolicyMarks(clock=clock)
    marks.record(KEY, "v1")
    client: Final = database(*(fingerprint_row("v1"),) * 4)
    await cache.async_set_cache(key="team_id:team", value=CACHED)
    clock.now += 299
    async with validated_listing():
        assert await is_current(client, cache, marks)
    clock.now += 2
    async with validated_listing():
        assert not await is_current(client, cache, marks)
    assert await cache.async_get_cache(key="team_id:team") == CACHED
    async with validated_listing():
        assert await is_current(client, cache, marks)


@pytest.mark.asyncio
async def test_access_group_entries_are_evicted_with_the_rest(cache: UserApiKeyCache) -> None:
    client: Final = database(fingerprint_row("v1", access_group_ids=("group",)), fingerprint_row("v1"))
    await cache.async_set_cache(key="access_group_id:group", value=CACHED)
    async with validated_listing():
        await is_current(client, cache, ValidatedPolicyMarks())
    assert await cache.async_get_cache(key="access_group_id:group") is None


@pytest.mark.asyncio
async def test_toolset_permissions_keep_the_database_read(cache: UserApiKeyCache) -> None:
    marks: Final = ValidatedPolicyMarks()
    marks.record(KEY, "v1")
    client: Final = database(fingerprint_row("v1", uses_toolsets=True), fingerprint_row("v1", uses_toolsets=True))
    await cache.async_set_cache(key="team_id:team", value=CACHED)
    async with validated_listing():
        assert not await is_current(client, cache, marks)
        assert not strict_cache_refill()
    assert await cache.async_get_cache(key="team_id:team") == CACHED


@pytest.mark.asyncio
async def test_a_failed_eviction_fails_closed_and_validates_nothing(cache: UserApiKeyCache) -> None:
    redis: Final = MagicMock()
    redis.delete_cache_keys = AsyncMock(side_effect=RuntimeError("cache down"))
    cache.attach_redis_cache(redis)
    marks: Final = ValidatedPolicyMarks()
    client: Final = database(fingerprint_row("v1"), fingerprint_row("v1"))
    async with validated_listing():
        assert not await is_current(client, cache, marks)
    assert marks.phase(KEY, "v1") == "other"


@pytest.mark.asyncio
async def test_a_task_started_inside_a_listing_sees_it_end_and_a_listing_there_opens_its_own() -> None:
    gate: Final = asyncio.Event()

    async def read_after_the_listing() -> RequestFreshness | None:
        await gate.wait()
        return current_listing()

    async def list_after_the_listing() -> bool:
        await gate.wait()
        async with validated_listing() as listing:
            return listing is current_listing()

    async with validated_listing():
        reader: Final = asyncio.create_task(read_after_the_listing())
        lister: Final = asyncio.create_task(list_after_the_listing())
    gate.set()
    assert await reader is None
    assert await lister is True


def caller(
    *,
    jwt: bool = False,
    user_id: str | None = "someone",
    project_id: str | None = None,
    agent_id: str | None = None,
    end_user_id: str | None = None,
    toolset_id: str | None = None,
) -> UserAPIKeyAuth:
    auth: Final = UserAPIKeyAuth(
        api_key="key-hash",
        user_id=user_id,
        team_id="team",
        org_id="org",
        project_id=project_id,
        agent_id=agent_id,
        end_user_id=end_user_id,
        jwt_claims={"sub": "someone"} if jwt else None,
    )
    auth.mcp_toolset_id = toolset_id
    return auth


def test_a_virtual_key_is_validated_by_its_hash() -> None:
    subject: Final = policy_subject_for(caller(), keyless_user=False, virtual_key=True, jwt_teams_from_claim_list=False)
    assert subject == PolicySubject(key_hash="key-hash")


def test_a_keyless_user_is_validated_by_user_and_pinned_team() -> None:
    subject: Final = policy_subject_for(caller(), keyless_user=True, virtual_key=False, jwt_teams_from_claim_list=False)
    assert subject == PolicySubject(user_id="someone", pinned_team_id="team")


def test_a_jwt_caller_is_validated_by_user_team_and_organization() -> None:
    subject: Final = policy_subject_for(
        caller(jwt=True), keyless_user=False, virtual_key=False, jwt_teams_from_claim_list=False
    )
    assert subject == PolicySubject(user_id="someone", pinned_team_id="team", pinned_org_id="org")


@pytest.mark.parametrize("keyless_user", [True, False])
@pytest.mark.parametrize(
    "scoped",
    [
        caller(project_id="project"),
        caller(agent_id="agent"),
        caller(end_user_id="customer"),
        caller(toolset_id="toolset"),
    ],
)
def test_callers_scoped_beyond_the_hash_keep_the_database_read(scoped: UserAPIKeyAuth, keyless_user: bool) -> None:
    assert (
        policy_subject_for(
            scoped, keyless_user=keyless_user, virtual_key=not keyless_user, jwt_teams_from_claim_list=False
        )
        is None
    )


@pytest.mark.parametrize(
    ("auth", "claim_list"),
    [
        (caller(jwt=True), True),
        (caller(jwt=True, user_id=None), False),
        (caller(), False),
    ],
)
def test_callers_the_hash_cannot_describe_keep_the_database_read(auth: UserAPIKeyAuth, claim_list: bool) -> None:
    assert policy_subject_for(auth, keyless_user=False, virtual_key=False, jwt_teams_from_claim_list=claim_list) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [True, False])
async def test_a_cache_refill_during_a_validated_listing_fails_closed(cache: UserApiKeyCache, missing: bool) -> None:
    marks: Final = ValidatedPolicyMarks()
    marks.record(KEY, "v1")
    lookup: Final = AsyncMock(return_value=None, side_effect=None if missing else RuntimeError("unavailable"))
    failing: Final = MagicMock()
    failing.writer_db.litellm_objectpermissiontable.find_unique = lookup
    failing.db.litellm_objectpermissiontable.find_unique = lookup
    async with validated_listing():
        assert await is_current(database(fingerprint_row("v1")), cache, marks)
        with pytest.raises(HTTPException if missing else RuntimeError):
            await get_object_permission("referenced", failing, cache)
    assert await get_object_permission("referenced", failing, cache) is None


@pytest.mark.asyncio
async def test_a_user_ceiling_that_cannot_be_read_denies_during_a_validated_listing(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    marks: Final = ValidatedPolicyMarks()
    marks.record(KEY, "v1")
    failing: Final = MagicMock()
    failing.db.litellm_usertable.find_unique = AsyncMock(side_effect=RuntimeError("unavailable"))
    async with validated_listing():
        assert await is_current(database(fingerprint_row("v1")), cache, marks)
        with pytest.raises(HTTPException) as denied:
            await MCPRequestHandler._user_object_permission_id("someone", failing)
    assert denied.value.status_code == 503
    assert await MCPRequestHandler._user_object_permission_id("someone", failing) is None


def admitted_user(user_id: str) -> UserAPIKeyAuth:
    auth: Final = UserAPIKeyAuth(user_id=user_id)
    auth.mcp_admitted_user_subject = True
    return auth


def serve_policy(
    monkeypatch: pytest.MonkeyPatch, cache: UserApiKeyCache, user_id: str, *rows: Mapping[str, object]
) -> tuple[MagicMock, AsyncMock]:
    user_lookup: Final = AsyncMock(return_value=LiteLLM_UserTable(user_id=user_id, user_role="internal_user"))
    client: Final = MagicMock()
    client.writer_db.litellm_usertable.find_unique = user_lookup
    client.db.litellm_usertable.find_unique = user_lookup
    client.writer_db.query_raw = AsyncMock(side_effect=[[row] for row in rows])
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    return client, user_lookup


async def forced_fresh(auth: UserAPIKeyAuth) -> bool | None:
    async with validated_listing():
        refreshed: Final = await MCPRequestHandler.refresh_catalog_authority(auth)
    return None if refreshed is None else refreshed.requires_fresh_policy


@pytest.mark.asyncio
async def test_a_listing_reads_the_admitted_user_from_the_cache_once_its_policy_is_validated(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, user_lookup = serve_policy(monkeypatch, cache, "wired-user", *(fingerprint_row("v1"),) * 6)
    flags: Final = (
        await forced_fresh(admitted_user("wired-user")),
        await forced_fresh(admitted_user("wired-user")),
        await forced_fresh(admitted_user("wired-user")),
        await forced_fresh(admitted_user("wired-user")),
    )
    assert flags == (True, True, False, False)
    assert user_lookup.await_count == 4


@pytest.mark.asyncio
async def test_a_policy_change_between_listings_sends_the_next_one_to_the_database(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows: Final = (*(fingerprint_row("v1"),) * 5, *(fingerprint_row("v2"),) * 2)
    serve_policy(monkeypatch, cache, "changing-user", *rows)
    flags: Final = (
        await forced_fresh(admitted_user("changing-user")),
        await forced_fresh(admitted_user("changing-user")),
        await forced_fresh(admitted_user("changing-user")),
        await forced_fresh(admitted_user("changing-user")),
    )
    assert flags == (True, True, False, True)


@pytest.mark.asyncio
async def test_a_jwt_caller_is_trusted_to_the_cache_once_its_policy_is_validated(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_policy(monkeypatch, cache, "jwt-user", *(fingerprint_row("v1"),) * 5)

    def jwt_caller() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_id="jwt-user", team_id="jwt-team", jwt_claims={"sub": "jwt-user"})

    flags: Final = [await forced_fresh(jwt_caller()) for _ in range(3)]
    assert flags == [True, True, False]


@pytest.mark.asyncio
async def test_a_project_scoped_caller_never_reaches_the_policy_hash(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = serve_policy(monkeypatch, cache, "scoped-user", fingerprint_row("v1"))
    scoped: Final = UserAPIKeyAuth(user_id="scoped-user", project_id="project", jwt_claims={"sub": "scoped-user"})
    assert await forced_fresh(scoped) is True
    client.writer_db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_jwt_caller_whose_teams_come_from_a_claim_list_keeps_the_database_read(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        proxy_server.jwt_handler, "litellm_jwtauth", LiteLLM_JWTAuth(team_ids_jwt_field="teams"), raising=False
    )
    client, _ = serve_policy(monkeypatch, cache, "claim-list-user", fingerprint_row("v1"))
    claim_list_caller: Final = UserAPIKeyAuth(user_id="claim-list-user", jwt_claims={"sub": "claim-list-user"})
    assert await forced_fresh(claim_list_caller) is True
    client.writer_db.query_raw.assert_not_awaited()


@asynccontextmanager
async def trusted_listing(cache: UserApiKeyCache) -> AsyncGenerator[None]:
    marks: Final = ValidatedPolicyMarks()
    marks.record(KEY, "v1")
    async with validated_listing():
        assert await is_current(database(fingerprint_row("v1")), cache, marks)
        yield


def unreadable_client() -> MagicMock:
    failing: Final = AsyncMock(side_effect=RuntimeError("unavailable"))
    client: Final = MagicMock()
    for reader in (client.db, client.writer_db):
        for table in (reader.litellm_accessgrouptable, reader.litellm_teamtable, reader.litellm_mcpservertable):
            table.find_unique = failing
            table.find_many = failing
    return client


def install_proxy(monkeypatch: pytest.MonkeyPatch, cache: UserApiKeyCache, client: MagicMock) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)


@pytest.fixture
def unreadable_database(cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client: Final = unreadable_client()
    install_proxy(monkeypatch, cache, client)
    return client


@pytest.mark.asyncio
async def test_a_caller_that_already_requires_fresh_policy_stays_on_the_database_read(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_policy(monkeypatch, cache, "pinned-user", *(fingerprint_row("v1"),) * 6)
    flags: Final = [await forced_fresh(admitted_user("pinned-user")) for _ in range(3)]
    assert flags == [True, True, False]
    pinned: Final = admitted_user("pinned-user")
    pinned.requires_fresh_policy = True
    assert await forced_fresh(pinned) is True


@pytest.mark.asyncio
async def test_a_real_listing_validates_the_digest_and_later_ones_read_the_key_from_the_cache(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    grants: Final = LiteLLM_ObjectPermissionTable(object_permission_id="listing-grants", mcp_servers=["alpha"])
    get_data: Final = AsyncMock(return_value=UserAPIKeyAuth(object_permission_id="listing-grants"))
    tables: Final = SimpleNamespace(
        litellm_objectpermissiontable=SimpleNamespace(find_unique=AsyncMock(return_value=grants)),
        query_raw=AsyncMock(side_effect=[[fingerprint_row("v1")]] * 6),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=tables, writer_db=tables, get_data=get_data))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": []})
    monkeypatch.setitem(
        operations.global_mcp_server_manager.registry,
        "alpha",
        MCPServer(server_id="alpha", name="alpha", transport=MCPTransport.http, spec_path="/listing.yaml"),
    )
    caller: Final = UserAPIKeyAuth(
        api_key="listing-key-hash", object_permission=grants, object_permission_id="listing-grants"
    )
    caller.via_virtual_key = True
    for _ in range(4):
        await catalog.list_gateway_catalog(
            operations.prepare_context(user_api_key_auth=caller),
            ListToolsRequest(),
            log_list_tools_to_spendlogs=False,
        )
    assert get_data.await_count == 3
    assert validated_policy_marks.phase(PolicySubject(key_hash="listing-key-hash"), "v1") == "settled"


@pytest.mark.asyncio
async def test_unreadable_key_access_groups_deny_during_a_trusted_listing(
    cache: UserApiKeyCache, unreadable_database: MagicMock
) -> None:
    key: Final = UserAPIKeyAuth(api_key="group-key", access_group_ids=["group"])
    outside: Final = await MCPRequestHandler._key_access_group_tools_for_server(key, "alpha")
    assert outside.owned == () and not outside.owns_server
    async with trusted_listing(cache):
        with pytest.raises(UnloadableEntitlementError):
            await MCPRequestHandler._key_access_group_tools_for_server(key, "alpha")


@pytest.mark.asyncio
async def test_unreadable_team_access_groups_deny_during_a_trusted_listing(
    cache: UserApiKeyCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    client: Final = unreadable_client()
    team: Final = AsyncMock(return_value=LiteLLM_TeamTable(team_id="team", access_group_ids=["group"]))
    client.db.litellm_teamtable.find_unique = team
    client.writer_db.litellm_teamtable.find_unique = team
    install_proxy(monkeypatch, cache, client)
    member: Final = UserAPIKeyAuth(api_key="team-key", team_id="team")
    assert await MCPRequestHandler._team_access_group_tools_for_server(member, "alpha") == ()
    async with trusted_listing(cache):
        with pytest.raises(UnloadableEntitlementError):
            await MCPRequestHandler._team_access_group_tools_for_server(member, "alpha")


@pytest.mark.asyncio
async def test_an_unreadable_team_ceiling_is_not_cached_as_no_ceiling_during_a_trusted_listing(
    cache: UserApiKeyCache,
) -> None:
    client: Final = unreadable_client()
    team: Final = AsyncMock(return_value=LiteLLM_TeamTable(team_id="ceiling-team", object_permission_id="ceiling"))
    client.db.litellm_teamtable.find_unique = team
    client.writer_db.litellm_teamtable.find_unique = team
    failing: Final = AsyncMock(side_effect=RuntimeError("unavailable"))
    client.db.litellm_objectpermissiontable.find_unique = failing
    client.writer_db.litellm_objectpermissiontable.find_unique = failing
    async with trusted_listing(cache):
        with pytest.raises(HTTPException):
            await get_team_object(team_id="ceiling-team", prisma_client=client, user_api_key_cache=cache)
    assert await cache.async_get_cache(key="team_id:ceiling-team") is None


@pytest.mark.asyncio
async def test_a_trusted_listing_does_not_join_a_membership_load_another_request_started(
    cache: UserApiKeyCache,
) -> None:
    shared: Final = LiteLLM_TeamMembership(user_id="member", team_id="shared-team")

    async def load_elsewhere() -> LiteLLM_TeamMembership:
        return shared

    started_elsewhere: Final = asyncio.ensure_future(load_elsewhere())
    _team_membership_inflight["team_membership:member:shared-team"] = started_elsewhere
    client: Final = MagicMock()
    lookup: Final = AsyncMock(return_value=None)
    client.db.litellm_teammembership.find_unique = lookup
    client.writer_db.litellm_teammembership.find_unique = lookup
    try:
        assert await get_team_membership("member", "shared-team", client, cache) == shared
        async with trusted_listing(cache):
            assert await get_team_membership("member", "shared-team", client, cache) is None
    finally:
        _team_membership_inflight.pop("team_membership:member:shared-team", None)


@pytest.mark.asyncio
async def test_unreadable_access_group_servers_fail_during_a_trusted_listing(
    cache: UserApiKeyCache, unreadable_database: MagicMock
) -> None:
    assert await MCPRequestHandler._get_db_server_ids_for_access_groups(unreadable_database, ["group"]) == set()
    assert await MCPRequestHandler.get_mcp_servers_from_access_groups(["group"]) == []
    async with trusted_listing(cache):
        with pytest.raises(RuntimeError):
            await MCPRequestHandler._get_db_server_ids_for_access_groups(unreadable_database, ["group"])
        with pytest.raises(RuntimeError):
            await MCPRequestHandler.get_mcp_servers_from_access_groups(["group"])


@pytest.mark.asyncio
async def test_an_unreadable_team_fails_during_a_trusted_listing(
    cache: UserApiKeyCache, unreadable_database: MagicMock
) -> None:
    member: Final = UserAPIKeyAuth(api_key="team-key", team_id="team")
    assert await MCPRequestHandler._allowed_mcp_servers_for_single_team("team", member) == []
    async with trusted_listing(cache):
        with pytest.raises(HTTPException) as unreadable:
            await MCPRequestHandler._allowed_mcp_servers_for_single_team("team", member)
    assert unreadable.value.status_code == 404


@pytest.mark.asyncio
async def test_a_key_whose_permission_cannot_be_read_fails_during_a_trusted_listing(cache: UserApiKeyCache) -> None:
    client: Final = MagicMock()
    client.get_data = AsyncMock(return_value=UserAPIKeyAuth(object_permission_id="permission"))
    client.db.litellm_objectpermissiontable.find_unique = AsyncMock(side_effect=RuntimeError("unavailable"))
    assert (await get_key_object("key-outside", client, cache)).object_permission_id == "permission"
    async with trusted_listing(cache):
        with pytest.raises(RuntimeError):
            await get_key_object("key-inside", client, cache)
