"""Pin ``PrismaClient`` health + spend-logs counter helpers.

Symbols pinned here:
  - ``PrismaClient.health_check``
  - ``PrismaClient._get_spend_logs_row_count``
  - ``PrismaClient._set_spend_logs_row_count_in_proxy_state``
  - ``PrismaClient._validate_response_time``
  - ``PrismaClient._clean_details``
  - ``PrismaClient.save_health_check_result``
  - ``PrismaClient.get_health_check_history``
  - ``PrismaClient.get_all_latest_health_checks``
  - ``PrismaClient._is_sha256_hex`` (a nested helper inside
    ``migrate_passwords_to_scrypt_async``; the pin list assigns it to this
    cluster as a documentation artifact)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
async def test_health_check_returns_query_raw_result(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_raw = AsyncMock(return_value=[{"?column?": 1}])
    result = await prisma_client.health_check()
    actual = {
        "result": result,
        "query_raw_called": prisma_client.db.query_raw.await_count,
        "query_sql": prisma_client.db.query_raw.await_args.args[0],
        "type": type(result).__name__,
    }
    assert actual == {
        "result": [{"?column?": 1}],
        "query_raw_called": 1,
        "query_sql": "SELECT 1",
        "type": "list",
    }


@pytest.mark.asyncio
async def test_health_check_raises_when_query_raw_fails(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_raw = AsyncMock(side_effect=RuntimeError("connection refused"))
    with pytest.raises(RuntimeError, match="connection refused"):
        await prisma_client.health_check()


@pytest.mark.asyncio
async def test_get_spend_logs_row_count_returns_int_from_pg_class(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_raw = AsyncMock(return_value=[{"reltuples": 12345}])
    result = await prisma_client._get_spend_logs_row_count()
    actual = {
        "result": result,
        "query_count": prisma_client.db.query_raw.await_count,
        "query_kwargs": prisma_client.db.query_raw.await_args.kwargs,
        "type": type(result).__name__,
    }
    assert actual == {
        "result": 12345,
        "query_count": 1,
        "query_kwargs": {
            "query": prisma_client.db.query_raw.await_args.kwargs["query"]
        },
        "type": "int",
    }


@pytest.mark.asyncio
async def test_get_spend_logs_row_count_error_falls_back_to_zero(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_raw = AsyncMock(side_effect=RuntimeError("perm denied"))
    assert await prisma_client._get_spend_logs_row_count() == 0


@pytest.mark.asyncio
async def test_set_spend_logs_row_count_in_proxy_state_writes_to_state(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_state = MagicMock()
    fake_state.set_proxy_state_variable = MagicMock()

    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(proxy_server_mod, "proxy_state", fake_state, raising=False)

    prisma_client._get_spend_logs_row_count = AsyncMock(return_value=99)
    await prisma_client._set_spend_logs_row_count_in_proxy_state()
    kwargs = fake_state.set_proxy_state_variable.call_args.kwargs
    assert kwargs == {"variable_name": "spend_logs_row_count", "value": 99}


@pytest.mark.asyncio
async def test_set_spend_logs_row_count_error_raises_through_backoff(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_state = MagicMock()
    fake_state.set_proxy_state_variable = MagicMock(side_effect=RuntimeError("boom"))
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(proxy_server_mod, "proxy_state", fake_state, raising=False)

    prisma_client._get_spend_logs_row_count = AsyncMock(return_value=1)
    with pytest.raises(RuntimeError, match="boom"):
        await prisma_client._set_spend_logs_row_count_in_proxy_state()


def test_validate_response_time_passes_finite_value(prisma_client: PrismaClient) -> None:
    inputs = {
        "ok": prisma_client._validate_response_time(123.45),
        "none": prisma_client._validate_response_time(None),
        "inf": prisma_client._validate_response_time(float("inf")),
        "neg_inf": prisma_client._validate_response_time(float("-inf")),
        "nan": prisma_client._validate_response_time(float("nan")),
    }
    assert inputs == {
        "ok": 123.45,
        "none": None,
        "inf": None,
        "neg_inf": None,
        "nan": None,
    }


def test_validate_response_time_invalid_string_returns_none(
    prisma_client: PrismaClient,
) -> None:
    """Non-numeric input is logged and returned as None. The name is the
    error hint; the input itself is invalid, not a thrown exception."""
    assert prisma_client._validate_response_time("not-a-float") is None


def test_clean_details_round_trips_json(prisma_client: PrismaClient) -> None:
    details = {"latency": 1.5, "ok": True, "error": None, "model": "gpt-4o"}
    cleaned = prisma_client._clean_details(details)
    pinned = {
        "cleaned": cleaned,
        "is_dict": isinstance(cleaned, dict),
        "none_for_non_dict": prisma_client._clean_details("oops"),  # type: ignore[arg-type]
        "none_for_none": prisma_client._clean_details(None),
    }
    assert pinned == {
        "cleaned": details,
        "is_dict": True,
        "none_for_non_dict": None,
        "none_for_none": None,
    }


def test_clean_details_invalid_payload_returns_none(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``safe_dumps`` itself blows up (e.g. an internal exception), the
    error path swallows it and returns None.
    """
    import litellm.proxy.utils as utils_mod

    def _explode(_: Any) -> str:
        raise RuntimeError("safe_dumps broken")

    monkeypatch.setattr(utils_mod, "safe_dumps", _explode)
    assert prisma_client._clean_details({"x": 1}) is None


@pytest.mark.asyncio
async def test_save_health_check_result_creates_record(
    prisma_client: PrismaClient,
) -> None:
    expected = MagicMock(name="HealthCheckRow")
    prisma_client.db.litellm_healthchecktable.create = AsyncMock(return_value=expected)
    result = await prisma_client.save_health_check_result(
        model_name="gpt-4o",
        status="healthy",
        healthy_count=3,
        unhealthy_count=0,
        response_time_ms=150.0,
        details={"latency": 1, "ok": True},
        checked_by="probe",
        model_id="m-1",
    )
    data = prisma_client.db.litellm_healthchecktable.create.await_args.kwargs["data"]
    pinned = {
        "returned": result,
        "model_name": data["model_name"],
        "status": data["status"],
        "healthy_count": data["healthy_count"],
        "response_time_ms": data["response_time_ms"],
        "details": data["details"],
        "checked_by": data["checked_by"],
        "model_id": data["model_id"],
    }
    assert pinned == {
        "returned": expected,
        "model_name": "gpt-4o",
        "status": "healthy",
        "healthy_count": 3,
        "response_time_ms": 150.0,
        "details": {"latency": 1, "ok": True},
        "checked_by": "probe",
        "model_id": "m-1",
    }


@pytest.mark.asyncio
async def test_save_health_check_result_db_failure_returns_none(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_healthchecktable.create = AsyncMock(
        side_effect=RuntimeError("db down")
    )
    result = await prisma_client.save_health_check_result(
        model_name="gpt-4o", status="healthy"
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_health_check_history_filters_by_model_and_status(
    prisma_client: PrismaClient,
) -> None:
    rows = [MagicMock(name=f"row-{i}") for i in range(2)]
    prisma_client.db.litellm_healthchecktable.find_many = AsyncMock(return_value=rows)
    result = await prisma_client.get_health_check_history(
        model_name="gpt-4o", limit=5, offset=10, status_filter="healthy"
    )
    kwargs = prisma_client.db.litellm_healthchecktable.find_many.await_args.kwargs
    actual = {
        "result_len": len(result),
        "where": kwargs["where"],
        "order": kwargs["order"],
        "take": kwargs["take"],
        "skip": kwargs["skip"],
    }
    assert actual == {
        "result_len": 2,
        "where": {"model_name": "gpt-4o", "status": "healthy"},
        "order": {"checked_at": "desc"},
        "take": 5,
        "skip": 10,
    }


@pytest.mark.asyncio
async def test_get_health_check_history_db_error_returns_empty_list(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_healthchecktable.find_many = AsyncMock(
        side_effect=RuntimeError("network down")
    )
    assert await prisma_client.get_health_check_history() == []


@pytest.mark.asyncio
async def test_get_all_latest_health_checks_uses_distinct(
    prisma_client: PrismaClient,
) -> None:
    rows = [MagicMock(name=f"row-{i}") for i in range(3)]
    prisma_client.db.litellm_healthchecktable.find_many = AsyncMock(return_value=rows)
    result = await prisma_client.get_all_latest_health_checks()
    kwargs = prisma_client.db.litellm_healthchecktable.find_many.await_args.kwargs
    actual = {
        "len": len(result),
        "distinct": kwargs["distinct"],
        "order_len": len(kwargs["order"]),
        "first_order": kwargs["order"][0],
    }
    assert actual == {
        "len": 3,
        "distinct": ["model_id", "model_name"],
        "order_len": 3,
        "first_order": {"model_id": "asc"},
    }


@pytest.mark.asyncio
async def test_get_all_latest_health_checks_db_error_returns_empty_list(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_healthchecktable.find_many = AsyncMock(
        side_effect=RuntimeError("oops")
    )
    assert await prisma_client.get_all_latest_health_checks() == []
