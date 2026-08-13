import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
from prisma.engine.errors import EngineConnectionError

sys.path.insert(0, os.path.abspath("../.."))

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_IN_MEMORY_TTL
from litellm.proxy.auth.cli_session_registry import (
    cli_session_id,
    is_cli_session_revoked,
    list_cli_sessions,
    record_cli_session,
    revoke_cli_session,
)
from litellm.proxy.utils import hash_token

SESSION_TOKEN = "cli-session-abc123"


class FakeRow:
    def __init__(self, data: dict):
        self._data = dict(data)

    def model_dump(self) -> dict:
        return dict(self._data)


class FakeCLISessionTable:
    """In-memory stand-in for the prisma table actions on LiteLLM_CLISessionTable."""

    def __init__(self, rows: dict | None = None):
        self.rows = dict(rows or {})
        self.find_unique_calls = 0
        self.created: list[dict] = []
        self.update_calls: list[dict] = []

    async def find_unique(self, where):
        self.find_unique_calls += 1
        row = self.rows.get(where["session_id"])
        return None if row is None else FakeRow(row)

    async def create(self, data):
        self.created.append(dict(data))
        row = {
            "created_at": datetime.now(timezone.utc),
            "revoked_at": None,
            "revoked_by": None,
            **dict(data),
        }
        self.rows[row["session_id"]] = row
        return FakeRow(row)

    async def update(self, where, data):
        self.update_calls.append(dict(data))
        row = self.rows.get(where["session_id"])
        if row is None:
            return None
        row.update(data)
        return FakeRow(row)

    async def find_many(self, where, order, skip, take):
        gt = where["expires_at"]["gt"]
        matching = [r for r in self.rows.values() if r["expires_at"] > gt]
        ordered = sorted(matching, key=lambda r: r["created_at"], reverse=True)
        return [FakeRow(r) for r in ordered[skip : skip + take]]

    async def count(self, where):
        gt = where["expires_at"]["gt"]
        return len([r for r in self.rows.values() if r["expires_at"] > gt])


class FakeDB:
    def __init__(self, table):
        self.litellm_clisessiontable = table


class FakePrismaClient:
    def __init__(self, table):
        self.db = FakeDB(table)


class FakeRedisCache:
    """Round-trips through JSON exactly as the real Redis backend does, so a value
    that only survives the in-memory path is caught."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def async_set_cache(self, key, value, **kwargs):
        self.store[key] = json.dumps(value)

    async def async_get_cache(self, key, parent_otel_span=None, **kwargs):
        raw = self.store.get(key)
        return None if raw is None else json.loads(raw)


def _session_row(*, revoked_at=None, expires_in_hours: float = 24.0, session_id=None, user_id="u-1"):
    now = datetime.now(timezone.utc)
    return {
        "session_id": session_id or hash_token(SESSION_TOKEN),
        "user_id": user_id,
        "team_id": "t-1",
        "created_at": now,
        "expires_at": now + timedelta(hours=expires_in_hours),
        "revoked_at": revoked_at,
        "revoked_by": None,
    }


def _cache() -> DualCache:
    return DualCache(in_memory_cache=InMemoryCache())


@pytest.mark.asyncio
async def test_unrevoked_session_is_not_revoked():
    table = FakeCLISessionTable({hash_token(SESSION_TOKEN): _session_row()})

    assert (
        await is_cli_session_revoked(
            session_token=SESSION_TOKEN,
            prisma_client=FakePrismaClient(table),
            user_api_key_cache=_cache(),
        )
        is False
    )


@pytest.mark.asyncio
async def test_revoked_session_is_revoked():
    table = FakeCLISessionTable({hash_token(SESSION_TOKEN): _session_row(revoked_at=datetime.now(timezone.utc))})

    assert (
        await is_cli_session_revoked(
            session_token=SESSION_TOKEN,
            prisma_client=FakePrismaClient(table),
            user_api_key_cache=_cache(),
        )
        is True
    )


@pytest.mark.asyncio
async def test_session_with_no_row_still_authenticates():
    """Sessions minted before this registry existed have no row. They must keep
    working until they expire rather than being locked out by the upgrade."""
    table = FakeCLISessionTable()

    assert (
        await is_cli_session_revoked(
            session_token=SESSION_TOKEN,
            prisma_client=FakePrismaClient(table),
            user_api_key_cache=_cache(),
        )
        is False
    )


@pytest.mark.asyncio
async def test_lookup_is_cached_for_one_cache_interval():
    """Without the cache every CLI request would cost a DB read. The TTL is also the
    bound this feature advertises on how fast a revoke reaches another replica."""
    table = FakeCLISessionTable({hash_token(SESSION_TOKEN): _session_row()})
    cache = _cache()
    prisma = FakePrismaClient(table)

    for _ in range(3):
        await is_cli_session_revoked(session_token=SESSION_TOKEN, prisma_client=prisma, user_api_key_cache=cache)

    assert table.find_unique_calls == 1
    expires_at = cache.in_memory_cache.ttl_dict[f"cli_session_revoked:{hash_token(SESSION_TOKEN)}"]
    assert expires_at - time.time() == pytest.approx(DEFAULT_IN_MEMORY_TTL, abs=1)


@pytest.mark.asyncio
async def test_revoking_replica_refuses_the_session_immediately():
    """The pod that served the revoke must not keep honouring the cached 'not
    revoked' answer it wrote earlier in the same interval."""
    session_id = hash_token(SESSION_TOKEN)
    table = FakeCLISessionTable({session_id: _session_row()})
    cache = _cache()
    prisma = FakePrismaClient(table)

    assert (
        await is_cli_session_revoked(session_token=SESSION_TOKEN, prisma_client=prisma, user_api_key_cache=cache)
        is False
    )

    await revoke_cli_session(
        prisma_client=prisma,
        user_api_key_cache=cache,
        session_id=session_id,
        revoked_by="admin-1",
    )

    calls_before = table.find_unique_calls
    assert (
        await is_cli_session_revoked(session_token=SESSION_TOKEN, prisma_client=prisma, user_api_key_cache=cache)
        is True
    )
    assert table.find_unique_calls == calls_before


@pytest.mark.asyncio
async def test_revocation_reaches_an_observing_replica_through_redis():
    """A second replica that primed its own 'not revoked' answer must pick the
    revocation up from Redis once its local entry lapses, without needing the DB."""
    session_id = hash_token(SESSION_TOKEN)
    table = FakeCLISessionTable({session_id: _session_row()})
    prisma = FakePrismaClient(table)
    redis = FakeRedisCache()
    revoking_replica = DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis)
    observing_replica = DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis)

    assert (
        await is_cli_session_revoked(
            session_token=SESSION_TOKEN, prisma_client=prisma, user_api_key_cache=observing_replica
        )
        is False
    )

    await revoke_cli_session(
        prisma_client=prisma,
        user_api_key_cache=revoking_replica,
        session_id=session_id,
        revoked_by="admin-1",
    )

    observing_replica.in_memory_cache.cache_dict.pop(f"cli_session_revoked:{session_id}")
    calls_before = table.find_unique_calls
    assert (
        await is_cli_session_revoked(
            session_token=SESSION_TOKEN, prisma_client=prisma, user_api_key_cache=observing_replica
        )
        is True
    )
    assert table.find_unique_calls == calls_before


@pytest.mark.asyncio
async def test_revoking_twice_keeps_the_first_revocation_time():
    session_id = hash_token(SESSION_TOKEN)
    first_revoked_at = datetime.now(timezone.utc) - timedelta(hours=2)
    table = FakeCLISessionTable({session_id: _session_row(revoked_at=first_revoked_at)})

    revoked = await revoke_cli_session(
        prisma_client=FakePrismaClient(table),
        user_api_key_cache=_cache(),
        session_id=session_id,
        revoked_by="admin-2",
    )

    assert revoked is not None
    assert revoked.revoked_at == first_revoked_at
    assert table.update_calls == []


@pytest.mark.asyncio
async def test_revoking_an_unknown_session_returns_none():
    revoked = await revoke_cli_session(
        prisma_client=FakePrismaClient(FakeCLISessionTable()),
        user_api_key_cache=_cache(),
        session_id="not-a-session",
        revoked_by="admin-1",
    )

    assert revoked is None


@pytest.mark.asyncio
async def test_recorded_session_is_keyed_by_the_hash_not_the_token():
    """The registry id is safe to hand to an operator and safe to log; the session
    token itself must never be persisted."""
    table = FakeCLISessionTable()

    recorded = await record_cli_session(
        prisma_client=FakePrismaClient(table),
        session_token=SESSION_TOKEN,
        user_id="u-1",
        team_id="t-1",
    )

    assert recorded.session_id == hash_token(SESSION_TOKEN)
    assert cli_session_id(SESSION_TOKEN) == hash_token(SESSION_TOKEN)
    assert SESSION_TOKEN not in json.dumps(table.created, default=str)
    assert table.created[0]["expires_at"] > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_no_db_connection_does_not_refuse_the_session():
    assert (
        await is_cli_session_revoked(session_token=SESSION_TOKEN, prisma_client=None, user_api_key_cache=_cache())
        is False
    )


class UnreachableCLISessionTable(FakeCLISessionTable):
    async def find_unique(self, where):
        raise EngineConnectionError("Could not connect to the query engine")


@pytest.mark.asyncio
async def test_db_outage_follows_the_proxy_wide_posture(monkeypatch):
    """The revocation lookup must not invent its own availability policy. An operator
    who opted into serving during a DB outage keeps serving CLI sessions; one who did
    not gets the same failure every other DB-backed auth read gives."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = FakePrismaClient(UnreachableCLISessionTable())

    monkeypatch.setattr(proxy_server, "general_settings", {"allow_requests_on_db_unavailable": True}, raising=False)
    assert (
        await is_cli_session_revoked(
            session_token=SESSION_TOKEN, prisma_client=prisma, user_api_key_cache=_cache()
        )
        is False
    )

    monkeypatch.setattr(proxy_server, "general_settings", {"allow_requests_on_db_unavailable": False}, raising=False)
    with pytest.raises(EngineConnectionError):
        await is_cli_session_revoked(session_token=SESSION_TOKEN, prisma_client=prisma, user_api_key_cache=_cache())


@pytest.mark.asyncio
async def test_listing_hides_expired_sessions():
    table = FakeCLISessionTable(
        {
            "live": _session_row(session_id="live", expires_in_hours=1),
            "dead": _session_row(session_id="dead", expires_in_hours=-1),
        }
    )

    listed = await list_cli_sessions(prisma_client=FakePrismaClient(table), page=1, page_size=50)

    assert listed.total_count == 1
    assert [s.session_id for s in listed.sessions] == ["live"]
