"""End-to-end tests for the Prisma-compatibility shim against SQLite.

These exercise the shim through its public surface so that breakage in
filter translation, mutations, batch_, or tx surfaces here rather than
in the proxy at runtime.

Run with::

    uv run pytest tests/test_litellm/proxy/db/sqlmodel_orm/test_compat.py -vv
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from litellm.proxy.db.sqlmodel import errors
from litellm.proxy.db.sqlmodel.compat import PrismaCompatClient
from litellm.proxy.db.sqlmodel.engine import LiteLLMDB
from litellm.proxy.db.sqlmodel.models import (
    LiteLLMConfig,
    LiteLLMSpendLogs,
    LiteLLMTeamMembership,
)


class _SQLiteLiteLLMDB(LiteLLMDB):
    """LiteLLMDB subclass pinned to in-memory SQLite for tests.

    The shipped LiteLLMDB pins ``pool_size``/``max_overflow`` which are
    NullPool-incompatible on SQLite. We override the engine builder for
    the test environment.
    """

    def _build_engine(self, url: str):  # type: ignore[override]
        return create_async_engine(url, future=True, echo=False)


@pytest_asyncio.fixture
async def db() -> AsyncIterator[_SQLiteLiteLLMDB]:
    db = _SQLiteLiteLLMDB("sqlite+aiosqlite:///:memory:")
    # Create the subset of tables we exercise in this test file.
    async with db.writer.begin() as conn:
        for cls in (LiteLLMConfig, LiteLLMSpendLogs, LiteLLMTeamMembership):
            await conn.run_sync(cls.__table__.create)
    yield db
    await db.disconnect()


@pytest_asyncio.fixture
async def client(db) -> PrismaCompatClient:
    return PrismaCompatClient(db)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_find_unique(client: PrismaCompatClient):
    await client.litellm_config.create(
        data={"param_name": "foo", "param_value": {"x": 1}}
    )
    row = await client.litellm_config.find_unique(where={"param_name": "foo"})
    assert row is not None
    assert row.param_name == "foo"
    assert row.param_value == {"x": 1}


@pytest.mark.asyncio
async def test_find_unique_returns_none_when_missing(client: PrismaCompatClient):
    row = await client.litellm_config.find_unique(where={"param_name": "absent"})
    assert row is None


@pytest.mark.asyncio
async def test_update_round_trip(client: PrismaCompatClient):
    await client.litellm_config.create(
        data={"param_name": "k", "param_value": {"v": 1}}
    )
    await client.litellm_config.update(
        where={"param_name": "k"}, data={"param_value": {"v": 2}}
    )
    row = await client.litellm_config.find_unique(where={"param_name": "k"})
    assert row is not None and row.param_value == {"v": 2}


@pytest.mark.asyncio
async def test_update_missing_raises_record_not_found(client: PrismaCompatClient):
    with pytest.raises(errors.RecordNotFoundError):
        await client.litellm_config.update(
            where={"param_name": "absent"}, data={"param_value": {}}
        )


@pytest.mark.asyncio
async def test_upsert_creates_then_updates(client: PrismaCompatClient):
    await client.litellm_config.upsert(
        where={"param_name": "u"},
        data={
            "create": {"param_name": "u", "param_value": {"v": 1}},
            "update": {"param_value": {"v": 2}},
        },
    )
    row = await client.litellm_config.find_unique(where={"param_name": "u"})
    assert row is not None and row.param_value == {"v": 1}

    await client.litellm_config.upsert(
        where={"param_name": "u"},
        data={
            "create": {"param_name": "u", "param_value": {"v": 99}},
            "update": {"param_value": {"v": 2}},
        },
    )
    row2 = await client.litellm_config.find_unique(where={"param_name": "u"})
    assert row2 is not None and row2.param_value == {"v": 2}


@pytest.mark.asyncio
async def test_delete_returns_row(client: PrismaCompatClient):
    await client.litellm_config.create(
        data={"param_name": "d", "param_value": {"v": 1}}
    )
    deleted = await client.litellm_config.delete(where={"param_name": "d"})
    assert deleted is not None and deleted.param_name == "d"
    assert (await client.litellm_config.find_unique(where={"param_name": "d"})) is None


# ---------------------------------------------------------------------------
# Bulk + count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_many_with_take_skip_order(client: PrismaCompatClient):
    for i in range(5):
        await client.litellm_config.create(
            data={"param_name": f"x{i}", "param_value": {"i": i}}
        )
    rows = await client.litellm_config.find_many(
        order={"param_name": "asc"}, take=2, skip=1
    )
    assert [r.param_name for r in rows] == ["x1", "x2"]


@pytest.mark.asyncio
async def test_count_with_where(client: PrismaCompatClient):
    for name in ("a", "b", "c"):
        await client.litellm_config.create(data={"param_name": name, "param_value": {}})
    assert (
        await client.litellm_config.count(where={"param_name": {"in": ["a", "b"]}})
    ) == 2


@pytest.mark.asyncio
async def test_update_many_returns_count(client: PrismaCompatClient):
    for name in ("a1", "a2", "a3"):
        await client.litellm_config.create(
            data={"param_name": name, "param_value": {"v": 0}}
        )
    res = await client.litellm_config.update_many(
        where={"param_name": {"in": ["a1", "a2"]}},
        data={"param_value": {"v": 9}},
    )
    assert int(res) == 2
    assert res == 2  # CountResult __eq__ with int


@pytest.mark.asyncio
async def test_delete_many_returns_count(client: PrismaCompatClient):
    for name in ("d1", "d2", "d3"):
        await client.litellm_config.create(data={"param_name": name, "param_value": {}})
    res = await client.litellm_config.delete_many(
        where={"param_name": {"in": ["d1", "d2"]}}
    )
    assert int(res) == 2
    assert (await client.litellm_config.count()) == 1


# ---------------------------------------------------------------------------
# Filter translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_or_and_not_filters(client: PrismaCompatClient):
    for name in ("alpha", "beta", "gamma"):
        await client.litellm_config.create(data={"param_name": name, "param_value": {}})
    rows = await client.litellm_config.find_many(
        where={"OR": [{"param_name": "alpha"}, {"param_name": "gamma"}]},
        order={"param_name": "asc"},
    )
    assert [r.param_name for r in rows] == ["alpha", "gamma"]

    rows = await client.litellm_config.find_many(
        where={"NOT": {"param_name": "beta"}},
        order={"param_name": "asc"},
    )
    assert [r.param_name for r in rows] == ["alpha", "gamma"]


@pytest.mark.asyncio
async def test_contains_insensitive(client: PrismaCompatClient):
    for name in ("FooBar", "BarBaz", "Quux"):
        await client.litellm_config.create(data={"param_name": name, "param_value": {}})
    rows = await client.litellm_config.find_many(
        where={"param_name": {"contains": "bar", "mode": "insensitive"}},
        order={"param_name": "asc"},
    )
    names = [r.param_name for r in rows]
    assert "FooBar" in names and "BarBaz" in names
    assert "Quux" not in names


# ---------------------------------------------------------------------------
# Increment / data translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_with_increment(client: PrismaCompatClient):
    await client.litellm_teammembership.create(
        data={"user_id": "u1", "team_id": "t1", "spend": 1.0, "total_spend": 0.0}
    )
    await client.litellm_teammembership.update_many(
        where={"user_id": "u1", "team_id": "t1"},
        data={"spend": {"increment": 4.0}, "total_spend": {"increment": 1.0}},
    )
    row = await client.litellm_teammembership.find_first(
        where={"user_id": "u1", "team_id": "t1"}
    )
    assert row is not None
    assert row.spend == 5.0
    assert row.total_spend == 1.0


# ---------------------------------------------------------------------------
# Raw SQL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_raw_positional_and_kw(client: PrismaCompatClient):
    await client.litellm_config.create(
        data={"param_name": "raw1", "param_value": {"v": 1}}
    )
    rows = await client.query_raw(
        'SELECT param_name FROM "LiteLLM_Config" WHERE param_name = $1', "raw1"
    )
    assert rows and rows[0]["param_name"] == "raw1"

    rows2 = await client.query_raw(
        query='SELECT param_name FROM "LiteLLM_Config" WHERE param_name = $1',
        *("raw1",),
    )
    assert rows2 and rows2[0]["param_name"] == "raw1"


@pytest.mark.asyncio
async def test_execute_raw_returns_rowcount(client: PrismaCompatClient):
    await client.litellm_config.create(data={"param_name": "ex1", "param_value": {}})
    n = await client.execute_raw(
        'DELETE FROM "LiteLLM_Config" WHERE param_name = $1', "ex1"
    )
    assert n == 1


# ---------------------------------------------------------------------------
# batch_ + tx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_commits_all_ops(client: PrismaCompatClient):
    await client.litellm_config.create(
        data={"param_name": "b1", "param_value": {"v": 0}}
    )
    batcher = client.batch_()
    batcher.litellm_config.update(
        where={"param_name": "b1"}, data={"param_value": {"v": 1}}
    )
    batcher.litellm_config.create(data={"param_name": "b2", "param_value": {"v": 2}})
    await batcher.commit()

    a = await client.litellm_config.find_unique(where={"param_name": "b1"})
    b = await client.litellm_config.find_unique(where={"param_name": "b2"})
    assert a is not None and a.param_value == {"v": 1}
    assert b is not None and b.param_value == {"v": 2}


@pytest.mark.asyncio
async def test_tx_rolls_back_on_exception(client: PrismaCompatClient):
    with pytest.raises(RuntimeError):
        async with client.tx() as tx:
            await tx.litellm_config.create(
                data={"param_name": "tx1", "param_value": {}}
            )
            raise RuntimeError("boom")
    assert (
        await client.litellm_config.find_unique(where={"param_name": "tx1"})
    ) is None


@pytest.mark.asyncio
async def test_tx_with_inner_batch(client: PrismaCompatClient):
    async with client.tx() as tx:
        batch = tx.batch_()
        batch.litellm_config.create(
            data={"param_name": "tx_b1", "param_value": {"v": 1}}
        )
        batch.litellm_config.create(
            data={"param_name": "tx_b2", "param_value": {"v": 2}}
        )
        await batch.commit()
    rows = await client.litellm_config.find_many(
        where={"param_name": {"in": ["tx_b1", "tx_b2"]}}
    )
    assert {r.param_name for r in rows} == {"tx_b1", "tx_b2"}
