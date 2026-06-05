import pytest

from litellm.proxy.auth.v2 import policy_store
from litellm.proxy.auth.v2.policy_store import (
    DEFAULT_POLICIES,
    load_policy_snapshot,
    reset_cache,
)


class _Row:
    def __init__(self, ptype, *values):
        self.ptype = ptype
        for i in range(6):
            setattr(self, f"v{i}", values[i] if i < len(values) else None)


class _DB:
    def __init__(self, rows):
        self.litellm_casbinrule = self
        self._rows = rows

    async def find_many(self):
        return self._rows


class _Prisma:
    def __init__(self, rows):
        self.db = _DB(rows)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()


def test_bootstrap_admin_policy_always_present():
    policies, _, _ = policy_store._split_rules([])
    assert ["role:proxy_admin", "*", "*", "*", "allow"] in policies
    assert DEFAULT_POLICIES[0] == ["role:proxy_admin", "*", "*", "*", "allow"]


def test_rows_split_by_ptype():
    rows = [
        _Row("p", "role:model_reader", "*", "model:*", "read", "allow"),
        _Row("g", "user:alice", "role:model_reader"),
        _Row("g2", "model:gpt-4o", "group:prod"),
    ]
    policies, groupings, resource_groupings = policy_store._split_rules(rows)
    assert ["role:model_reader", "*", "model:*", "read", "allow"] in policies
    assert ["user:alice", "role:model_reader"] in groupings
    # g2 must NOT leak into the g groupings.
    assert ["model:gpt-4o", "group:prod"] in resource_groupings
    assert ["model:gpt-4o", "group:prod"] not in groupings


def test_empty_trailing_columns_are_trimmed():
    policies, _, _ = policy_store._split_rules(
        [_Row("p", "role:x", "*", "model:*", "read", "allow")]
    )
    assert policies[-1] == ["role:x", "*", "model:*", "read", "allow"]


@pytest.mark.asyncio
async def test_load_snapshot_without_db_returns_only_bootstrap():
    policies, groupings, resource_groupings = await load_policy_snapshot(
        prisma_client=None
    )
    assert policies == [list(p) for p in DEFAULT_POLICIES]
    assert groupings == []
    assert resource_groupings == []


@pytest.mark.asyncio
async def test_load_snapshot_reads_db_rows():
    prisma = _Prisma(
        [
            _Row("p", "role:model_reader", "*", "model:*", "read", "allow"),
            _Row("g", "user:alice", "role:model_reader"),
            _Row("g2", "model:gpt-4o", "group:prod"),
        ]
    )
    policies, groupings, resource_groupings = await load_policy_snapshot(
        prisma_client=prisma
    )
    assert ["role:model_reader", "*", "model:*", "read", "allow"] in policies
    assert ["user:alice", "role:model_reader"] in groupings
    assert ["model:gpt-4o", "group:prod"] in resource_groupings
