"""
Tests for the gateway request (SGR) fold and its commit to
LiteLLM_DailyGatewayRequests.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from litellm.proxy.db.gateway_request_tracking import (
    GatewayRequestAccumulator,
    commit_gateway_requests_to_db,
    flush_gateway_requests,
)
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory
from litellm.types.proxy.gateway_requests import GatewayRequestCounts, GatewayRequestKey


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _record(accumulator: GatewayRequestAccumulator, status_code: int, **overrides) -> None:
    accumulator.record(
        category=overrides.get("category", BillableCategory.LLM),
        route=overrides.get("route", "/chat/completions"),
        status_code=status_code,
    )


# ── fold ──────────────────────────────────────────────────────────────────────


def test_folds_repeated_requests_into_one_key():
    acc = GatewayRequestAccumulator()
    for _ in range(3):
        _record(acc, 200)
    _record(acc, 500)

    snapshot = acc.drain()
    assert snapshot == {
        GatewayRequestKey(date=_today(), category="llm", route="/chat/completions"): (
            GatewayRequestCounts(successful_requests=3, failed_requests=1)
        )
    }


@pytest.mark.parametrize(
    "status_code, expected_successful, expected_failed",
    [(200, 1, 0), (201, 1, 0), (204, 1, 0), (299, 1, 0), (300, 0, 1), (400, 0, 1), (500, 0, 1)],
)
def test_success_boundary_is_2xx(status_code: int, expected_successful: int, expected_failed: int):
    acc = GatewayRequestAccumulator()
    _record(acc, status_code)
    counts = next(iter(acc.drain().values()))
    assert (counts.successful_requests, counts.failed_requests) == (expected_successful, expected_failed)


def test_distinct_dimensions_do_not_merge():
    acc = GatewayRequestAccumulator()
    _record(acc, 200, route="/chat/completions")
    _record(acc, 200, route="/embeddings")
    _record(acc, 200, category=BillableCategory.MCP, route="/mcp")
    assert len(acc.drain()) == 3


def test_drain_empties_the_fold():
    acc = GatewayRequestAccumulator()
    _record(acc, 200)
    assert len(acc.drain()) == 1
    assert acc.drain() == {}


def test_drain_snapshot_is_not_mutated_by_later_records():
    acc = GatewayRequestAccumulator()
    _record(acc, 200)
    snapshot = acc.drain()
    _record(acc, 200)
    assert next(iter(snapshot.values())).successful_requests == 1


# ── commit ────────────────────────────────────────────────────────────────────


class FakeTable:
    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert(self, *, where: dict, data: dict) -> None:
        self.upserts.append({"where": where, "data": data})


class FakeBatcher:
    def __init__(self, table: FakeTable) -> None:
        self.litellm_dailygatewayrequests = table

    async def __aenter__(self) -> "FakeBatcher":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False


class FakeDB:
    def __init__(self, table: FakeTable) -> None:
        self._table = table

    def batch_(self) -> FakeBatcher:
        return FakeBatcher(self._table)


class FakePrismaClient:
    def __init__(self) -> None:
        self.table = FakeTable()
        self.db = FakeDB(self.table)


def test_commit_upserts_one_incrementing_row_per_key():
    client = FakePrismaClient()
    snapshot = {
        GatewayRequestKey(date="2026-08-01", category="llm", route="/chat/completions"): (
            GatewayRequestCounts(successful_requests=7, failed_requests=2)
        )
    }

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    assert len(client.table.upserts) == 1
    written = client.table.upserts[0]
    assert written["where"] == {
        "date_category_route": {
            "date": "2026-08-01",
            "category": "llm",
            "route": "/chat/completions",
        }
    }
    assert written["data"]["update"] == {
        "successful_requests": {"increment": 7},
        "failed_requests": {"increment": 2},
    }
    assert written["data"]["create"]["successful_requests"] == 7


def test_commit_is_deterministically_ordered():
    """Concurrent writers must touch rows in the same order or they deadlock."""
    client = FakePrismaClient()
    keys = [
        GatewayRequestKey(date="2026-08-02", category="llm", route="/embeddings"),
        GatewayRequestKey(date="2026-08-01", category="mcp", route="/mcp"),
        GatewayRequestKey(date="2026-08-01", category="llm", route="/chat/completions"),
    ]
    snapshot = {key: GatewayRequestCounts(successful_requests=1, failed_requests=0) for key in keys}

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    written_order = [
        (row["where"]["date_category_route"]["date"], row["where"]["date_category_route"]["category"])
        for row in client.table.upserts
    ]
    assert written_order == [("2026-08-01", "llm"), ("2026-08-01", "mcp"), ("2026-08-02", "llm")]


def test_commit_skips_the_database_entirely_when_nothing_accumulated():
    client = FakePrismaClient()
    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot={}))
    assert client.table.upserts == []


# ── flush ─────────────────────────────────────────────────────────────────────


def test_flush_drains_and_commits():
    client = FakePrismaClient()
    acc = GatewayRequestAccumulator()
    _record(acc, 200)

    asyncio.run(flush_gateway_requests(client, acc))

    assert len(client.table.upserts) == 1
    assert acc.drain() == {}


class ExplodingDB:
    def batch_(self):
        raise RuntimeError("db gone")


class ExplodingClient:
    db = ExplodingDB()


def test_flush_swallows_commit_failure_so_the_scheduler_survives():
    acc = GatewayRequestAccumulator()
    _record(acc, 200)

    asyncio.run(flush_gateway_requests(ExplodingClient(), acc))


def test_failed_flush_keeps_counts_for_the_next_attempt():
    """A dropped flush would silently undercount the SGR source of truth."""
    acc = GatewayRequestAccumulator()
    _record(acc, 200)
    _record(acc, 500)

    asyncio.run(flush_gateway_requests(ExplodingClient(), acc))

    client = FakePrismaClient()
    asyncio.run(flush_gateway_requests(client, acc))

    assert client.table.upserts[0]["data"]["update"] == {
        "successful_requests": {"increment": 1},
        "failed_requests": {"increment": 1},
    }


def test_restored_counts_merge_with_requests_recorded_meanwhile():
    acc = GatewayRequestAccumulator()
    _record(acc, 200)
    asyncio.run(flush_gateway_requests(ExplodingClient(), acc))

    _record(acc, 200)
    client = FakePrismaClient()
    asyncio.run(flush_gateway_requests(client, acc))

    assert len(client.table.upserts) == 1
    assert client.table.upserts[0]["data"]["update"]["successful_requests"] == {"increment": 2}
