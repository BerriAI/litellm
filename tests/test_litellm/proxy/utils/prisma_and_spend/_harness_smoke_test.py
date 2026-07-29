"""Self-tests for the prisma_and_spend test harness fixtures.

Verifies the fixtures themselves do what their docstrings claim.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.utils import PrismaClient


def test_normalize_scrubs_volatile_keys() -> None:
    from tests.test_litellm.proxy.utils.prisma_and_spend.conftest import normalize

    out = normalize({"id": 1, "spend": 2.0, "team_id": "t1"})
    assert out == {"id": "<VOLATILE>", "spend": "<VOLATILE>", "team_id": "t1"}


def test_normalize_recurses_into_lists() -> None:
    from tests.test_litellm.proxy.utils.prisma_and_spend.conftest import normalize

    out = normalize([{"id": "x"}, {"team_id": "t"}])
    assert out == [{"id": "<VOLATILE>"}, {"team_id": "t"}]


def test_mock_prisma_client_has_common_tables(mock_prisma_client: Any) -> None:
    for table in (
        "litellm_verificationtoken",
        "litellm_teamtable",
        "litellm_usertable",
        "litellm_spendlogs",
        "litellm_config",
        "litellm_healthchecktable",
    ):
        assert hasattr(mock_prisma_client.db, table)


@pytest.mark.asyncio
async def test_mock_dual_cache_round_trip(mock_dual_cache: Any) -> None:
    await mock_dual_cache.async_set_cache("k", "v")
    assert await mock_dual_cache.async_get_cache("k") == "v"
    await mock_dual_cache.async_delete_cache("k")
    assert await mock_dual_cache.async_get_cache("k") is None


def test_prisma_client_fixture_is_a_real_prismaclient(
    prisma_client: PrismaClient,
) -> None:
    assert isinstance(prisma_client, PrismaClient)
    assert callable(prisma_client.hash_token)


@pytest.mark.asyncio
async def test_fake_clock_advances(fake_clock: Any) -> None:
    start = fake_clock.now
    await asyncio.sleep(2.5)
    assert fake_clock.now == start + 2.5
    assert fake_clock.sleep_calls == [2.5]


def test_make_spend_log_row_factory(make_spend_log_row: Any) -> None:
    row = make_spend_log_row(request_id="abc", spend=0.5)
    assert row["request_id"] == "abc"
    assert row["spend"] == 0.5


@pytest.mark.asyncio
async def test_in_memory_smtp_captures(in_memory_smtp: Any) -> None:
    factory = in_memory_smtp.server_factory()
    conn = factory("smtp.invalid", 25)
    with conn:
        conn.starttls()
        from email.message import EmailMessage

        m = EmailMessage()
        m["Subject"] = "S"
        m.set_content("<p>x</p>", subtype="html")
        conn.send_message(m, from_addr="a@b", to_addrs="c@d")
    assert len(in_memory_smtp.sent) == 1
