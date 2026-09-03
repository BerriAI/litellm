import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.health_check_latest import LatestHealthCheckRow
from litellm.proxy.health_endpoints._health_endpoints import (
    _aggregate_health_check_results,
    _build_model_param_to_info_mapping,
    _perform_health_check_and_save,
    _save_background_health_checks_to_db,
    _save_health_check_results_if_changed,
    _save_health_check_to_db,
    latest_health_checks_endpoint,
)
from litellm.proxy.utils import PrismaClient


@pytest.fixture
def mock_prisma():
    """Simplified mock PrismaClient with bound methods"""
    client = MagicMock()
    client.db.litellm_healthchecktable.create = AsyncMock(
        return_value={"id": "test-id"}
    )
    client.db.litellm_healthchecktable.find_many = AsyncMock(
        return_value=[{"id": "1", "model_name": "test"}]
    )

    # Bind actual methods
    import types

    for method in [
        "save_health_check_result",
        "_validate_response_time",
        "_clean_details",
        "get_health_check_history",
        "get_all_latest_health_checks",
    ]:
        setattr(client, method, types.MethodType(getattr(PrismaClient, method), client))

    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,healthy,unhealthy,should_succeed",
    [
        ("healthy", 1, 0, True),
        ("unhealthy", 0, 1, True),
        ("healthy", 1, 0, False),  # Database error case
    ],
)
async def test_save_health_check_result(
    mock_prisma, status, healthy, unhealthy, should_succeed
):
    """Test health check result saving with various scenarios"""
    if not should_succeed:
        mock_prisma.db.litellm_healthchecktable.create.side_effect = Exception(
            "DB Error"
        )

    result = await mock_prisma.save_health_check_result(
        model_name="test-model",
        status=status,
        healthy_count=healthy,
        unhealthy_count=unhealthy,
    )

    if should_succeed:
        mock_prisma.db.litellm_healthchecktable.create.assert_called_once()
    else:
        assert result is None


@pytest.mark.asyncio
async def test_get_health_check_history(mock_prisma):
    """Test health check history retrieval"""
    result = await mock_prisma.get_health_check_history(model_name="test", limit=50)
    mock_prisma.db.litellm_healthchecktable.find_many.assert_called_once()
    assert len(result) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "healthy_count,unhealthy_count,expected_status",
    [
        (1, 0, "healthy"),
        (0, 1, "unhealthy"),
        (2, 1, "healthy"),
    ],
)
async def test_save_health_check_to_db(healthy_count, unhealthy_count, expected_status):
    """Test _save_health_check_to_db function with different endpoint counts"""
    mock_client = MagicMock()
    mock_client.save_health_check_result = AsyncMock()

    healthy_endpoints = [{"model": "test"}] * healthy_count
    unhealthy_endpoints = [{"error": "test error"}] * unhealthy_count

    await _save_health_check_to_db(
        mock_client,
        "test-model",
        healthy_endpoints,
        unhealthy_endpoints,
        1234567890.0,
        "test-user",
    )

    call_args = mock_client.save_health_check_result.call_args[1]
    assert call_args["status"] == expected_status
    assert call_args["healthy_count"] == healthy_count
    assert call_args["unhealthy_count"] == unhealthy_count


@pytest.mark.asyncio
async def test_save_health_check_to_db_no_client():
    """Test graceful handling when no database client"""
    result = await _save_health_check_to_db(None, "test", [], [], 0.0, "user")
    assert result is None


# Tests for background health check functions


def test_build_model_param_to_info_mapping():
    """Test building model parameter to info mapping"""
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
        {
            "model_name": "gpt-4",
            "model_info": {"id": "model-456"},
            "litellm_params": {"model": "gpt-4"},
        },
        {
            "model_name": "gpt-3.5-turbo-alias",
            "model_info": {"id": "model-789"},
            "litellm_params": {"model": "gpt-3.5-turbo"},  # Same model param
        },
    ]

    result = _build_model_param_to_info_mapping(model_list)

    assert "gpt-3.5-turbo" in result
    assert "gpt-4" in result
    assert len(result["gpt-3.5-turbo"]) == 2  # Two models share same param
    assert len(result["gpt-4"]) == 1
    assert result["gpt-3.5-turbo"][0]["model_name"] == "gpt-3.5-turbo"
    assert result["gpt-3.5-turbo"][0]["model_id"] == "model-123"
    assert result["gpt-3.5-turbo"][1]["model_name"] == "gpt-3.5-turbo-alias"
    assert result["gpt-3.5-turbo"][1]["model_id"] == "model-789"


def test_build_model_param_to_info_mapping_no_model_name():
    """Test mapping skips models without model_name"""
    model_list = [
        {
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
    ]

    result = _build_model_param_to_info_mapping(model_list)
    assert len(result) == 0


def test_aggregate_health_check_results():
    """Test aggregating health check results per model"""
    model_param_to_info = {
        "gpt-3.5-turbo": [
            {"model_name": "gpt-3.5-turbo", "model_id": "model-123"},
        ],
        "gpt-4": [
            {"model_name": "gpt-4", "model_id": "model-456"},
        ],
    }

    healthy_endpoints = [
        {"model": "gpt-3.5-turbo"},
    ]
    unhealthy_endpoints = [
        {"model": "gpt-4", "error": "Rate limit exceeded"},
    ]

    result = _aggregate_health_check_results(
        model_param_to_info, healthy_endpoints, unhealthy_endpoints
    )

    # Check gpt-3.5-turbo is healthy
    gpt35_key = ("model-123", "gpt-3.5-turbo")
    assert gpt35_key in result
    assert result[gpt35_key]["healthy_count"] == 1
    assert result[gpt35_key]["unhealthy_count"] == 0
    assert result[gpt35_key]["error_message"] is None

    # Check gpt-4 is unhealthy
    gpt4_key = ("model-456", "gpt-4")
    assert gpt4_key in result
    assert result[gpt4_key]["healthy_count"] == 0
    assert result[gpt4_key]["unhealthy_count"] == 1
    assert "Rate limit" in result[gpt4_key]["error_message"]


def test_aggregate_health_check_results_multiple_endpoints():
    """Test aggregation with multiple endpoints for same model"""
    model_param_to_info = {
        "gpt-3.5-turbo": [
            {"model_name": "gpt-3.5-turbo", "model_id": "model-123"},
        ],
    }

    healthy_endpoints = [
        {"model": "gpt-3.5-turbo"},
        {"model": "gpt-3.5-turbo"},
    ]
    unhealthy_endpoints = []

    result = _aggregate_health_check_results(
        model_param_to_info, healthy_endpoints, unhealthy_endpoints
    )

    key = ("model-123", "gpt-3.5-turbo")
    assert result[key]["healthy_count"] == 2
    assert result[key]["unhealthy_count"] == 0


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_status_changed():
    """Test saving when status changes"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()

    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }

    # Latest check shows unhealthy, new result is healthy (status changed)
    latest_checks_map = {
        "model-123": MagicMock(
            status="unhealthy",
            checked_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        ),
    }

    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma,
        model_results,
        latest_checks_map,
        start_time,
        "background_health_check",
    )

    # Should save because status changed
    mock_prisma.save_health_check_result.assert_called_once()
    call_kwargs = mock_prisma.save_health_check_result.call_args[1]
    assert call_kwargs["status"] == "healthy"
    assert call_kwargs["model_name"] == "gpt-3.5-turbo"
    assert call_kwargs["checked_by"] == "background_health_check"


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_status_unchanged_recent():
    """Test skipping save when status unchanged and checked recently"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()

    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }

    # Latest check shows healthy, new result is healthy (status unchanged)
    # And checked recently (within 1 hour)
    latest_checks_map = {
        "model-123": MagicMock(
            status="healthy",
            checked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        ),
    }

    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma,
        model_results,
        latest_checks_map,
        start_time,
        "background_health_check",
    )

    # Should NOT save because status unchanged and checked recently
    mock_prisma.save_health_check_result.assert_not_called()


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_status_unchanged_old():
    """Test saving when status unchanged but last check is old (>1 hour)"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()

    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }

    # Latest check shows healthy, new result is healthy (status unchanged)
    # But checked >1 hour ago
    latest_checks_map = {
        "model-123": MagicMock(
            status="healthy",
            checked_at=datetime.now(timezone.utc) - timedelta(hours=2),
        ),
    }

    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma,
        model_results,
        latest_checks_map,
        start_time,
        "background_health_check",
    )

    # Should save because last check is old (>1 hour)
    mock_prisma.save_health_check_result.assert_called_once()


@pytest.mark.asyncio
async def test_save_health_check_results_if_changed_no_previous_check():
    """Test saving when there's no previous check"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()

    model_results = {
        ("model-123", "gpt-3.5-turbo"): {
            "model_name": "gpt-3.5-turbo",
            "model_id": "model-123",
            "healthy_count": 1,
            "unhealthy_count": 0,
            "error_message": None,
        },
    }

    # No previous check
    latest_checks_map = {}

    start_time = 1234567890.0
    await _save_health_check_results_if_changed(
        mock_prisma,
        model_results,
        latest_checks_map,
        start_time,
        "background_health_check",
    )

    # Should save because no previous check
    mock_prisma.save_health_check_result.assert_called_once()


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db():
    """Test the main background health check save function"""
    mock_prisma = MagicMock()
    mock_prisma.save_health_check_result = AsyncMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(return_value=[])

    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
    ]

    healthy_endpoints = [{"model": "gpt-3.5-turbo"}]
    unhealthy_endpoints = []

    start_time = 1234567890.0

    await _save_background_health_checks_to_db(
        mock_prisma,
        model_list,
        healthy_endpoints,
        unhealthy_endpoints,
        start_time,
        "background_health_check",
    )

    # Should call get_all_latest_health_checks and save_health_check_result
    mock_prisma.get_all_latest_health_checks.assert_called_once()
    mock_prisma.save_health_check_result.assert_called_once()

    call_kwargs = mock_prisma.save_health_check_result.call_args[1]
    assert call_kwargs["model_name"] == "gpt-3.5-turbo"
    assert call_kwargs["model_id"] == "model-123"
    assert call_kwargs["status"] == "healthy"
    assert call_kwargs["checked_by"] == "background_health_check"


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_no_prisma():
    """Test graceful handling when no prisma client"""
    result = await _save_background_health_checks_to_db(
        None, [], [], [], 0.0, "background_health_check"
    )
    assert result is None


@pytest.mark.asyncio
async def test_save_background_health_checks_to_db_exception_handling():
    """Test exception handling in background health check save"""
    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(
        side_effect=Exception("DB Error")
    )

    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "model_info": {"id": "model-123"},
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
    ]

    # Should not raise exception, should handle gracefully
    await _save_background_health_checks_to_db(
        mock_prisma, model_list, [], [], 0.0, "background_health_check"
    )

    # Function should complete without raising


def _raw_latest_row(model_name: str, model_id, checked_at: datetime) -> dict:
    return {
        "health_check_id": f"hc-{model_id or 'no-id'}-{model_name}",
        "model_name": model_name,
        "model_id": model_id,
        "status": "healthy",
        "healthy_count": 1,
        "unhealthy_count": 0,
        "error_message": None,
        "response_time_ms": 10.0,
        "details": None,
        "checked_by": "pod-1",
        "checked_at": checked_at.isoformat(),
        "created_at": checked_at.isoformat(),
        "updated_at": checked_at.isoformat(),
    }


@pytest.mark.asyncio
async def test_get_all_latest_health_checks_keeps_every_distinct_group_with_its_own_checked_at(mock_prisma):
    """
    Postgres owns the dedup. (id, name), (other id, name) and (NULL, name) are distinct groups and each row
    must arrive typed, with its own checked_at, for the 1h re-save compare and the id-or-name lookup key.
    """
    now = datetime.now(timezone.utc)
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[
            _raw_latest_row("gpt-3.5-turbo", "model-123", now - timedelta(minutes=1)),
            _raw_latest_row("gpt-3.5-turbo", "model-456", now - timedelta(minutes=5)),
            _raw_latest_row("gpt-4", "deployment-abc", now - timedelta(minutes=2)),
            _raw_latest_row("gpt-4", None, now - timedelta(minutes=3)),
        ]
    )

    result = await mock_prisma.get_all_latest_health_checks()

    assert {(check.model_id, check.model_name): check.checked_at for check in result} == {
        ("model-123", "gpt-3.5-turbo"): now - timedelta(minutes=1),
        ("model-456", "gpt-3.5-turbo"): now - timedelta(minutes=5),
        ("deployment-abc", "gpt-4"): now - timedelta(minutes=2),
        (None, "gpt-4"): now - timedelta(minutes=3),
    }


@pytest.mark.asyncio
async def test_save_background_health_checks_compares_raw_checked_at_against_utc_now(mock_prisma):
    """
    Raw rows carry ISO strings and the engine may omit the offset. A naive checked_at would TypeError
    inside the 1h compare, be swallowed, and silently stop every save; a stale row must still re-save.
    """
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None)
    fresh = datetime.now(timezone.utc) - timedelta(minutes=5)
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[
            _raw_latest_row("stale-model", "stale-id", stale),
            _raw_latest_row("fresh-model", "fresh-id", fresh),
        ]
    )
    mock_prisma.save_health_check_result = AsyncMock()
    model_list = [
        {"model_name": "stale-model", "model_info": {"id": "stale-id"}, "litellm_params": {"model": "openai/stale"}},
        {"model_name": "fresh-model", "model_info": {"id": "fresh-id"}, "litellm_params": {"model": "openai/fresh"}},
    ]

    await _save_background_health_checks_to_db(
        mock_prisma,
        model_list,
        [{"model": "openai/stale"}, {"model": "openai/fresh"}],
        [],
        time.time(),
        "pod-1",
    )
    await asyncio.sleep(0)

    assert [call.kwargs["model_id"] for call in mock_prisma.save_health_check_result.await_args_list] == ["stale-id"]


@pytest.mark.asyncio
async def test_latest_health_checks_endpoint_serialises_raw_rows(monkeypatch):
    row = LatestHealthCheckRow(
        health_check_id="hc-1",
        model_name="gpt-4",
        model_id="deployment-abc",
        status="healthy",
        healthy_count=1,
        unhealthy_count=0,
        error_message=None,
        response_time_ms=12.5,
        details='{"region": "eu"}',
        checked_by="pod-1",
        checked_at=datetime(2026, 8, 25),
        created_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    prisma = MagicMock()
    prisma.get_all_latest_health_checks = AsyncMock(return_value=(row,))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    response = await latest_health_checks_endpoint(user_api_key_dict=UserAPIKeyAuth())

    assert response == {
        "latest_health_checks": {
            "deployment-abc": {
                "health_check_id": "hc-1",
                "model_name": "gpt-4",
                "model_id": "deployment-abc",
                "status": "healthy",
                "healthy_count": 1,
                "unhealthy_count": 0,
                "error_message": None,
                "response_time_ms": 12.5,
                "details": {"region": "eu"},
                "checked_by": "pod-1",
                "checked_at": "2026-08-25T00:00:00+00:00",
                "created_at": "2026-08-25T00:00:00+00:00",
            }
        },
        "total_models": 1,
    }


@pytest.mark.asyncio
async def test_perform_health_check_and_save_passes_model_id_to_perform_health_check():
    """Test that _perform_health_check_and_save passes model_id to perform_health_check so health checks run by model id."""
    model_list = [
        {
            "model_name": "gpt-4",
            "model_info": {"id": "deployment-abc"},
            "litellm_params": {"model": "gpt-4"},
        },
    ]
    healthy = [{"model": "gpt-4"}]
    unhealthy = []

    async def mock_perform_health_check(
        model_list,
        model=None,
        cli_model=None,
        details=True,
        model_id=None,
        max_concurrency=None,
        **kwargs,
    ):
        return healthy, unhealthy, {}

    with patch(
        "litellm.proxy.health_endpoints._health_endpoints.perform_health_check",
        side_effect=mock_perform_health_check,
    ) as mock_perform:
        result = await _perform_health_check_and_save(
            model_list=model_list,
            target_model=None,
            cli_model=None,
            details=True,
            prisma_client=None,
            start_time=0.0,
            user_id="user-1",
            model_id="deployment-abc",
        )

    mock_perform.assert_called_once()
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["model_id"] == "deployment-abc"
    assert result["healthy_count"] == 1
    assert result["unhealthy_count"] == 0


@pytest.mark.asyncio
async def test_perform_health_check_and_save_forwards_skip_disabled_background_flag():
    """health_check_skip_disabled_background_models should reach perform_health_check."""
    model_list = [
        {
            "model_name": "gpt-4",
            "model_info": {"id": "deployment-abc"},
            "litellm_params": {"model": "gpt-4"},
        },
    ]

    async def mock_perform_health_check(**kwargs):
        return [], [], {}

    with patch(
        "litellm.proxy.health_endpoints._health_endpoints.perform_health_check",
        side_effect=mock_perform_health_check,
    ) as mock_perform:
        await _perform_health_check_and_save(
            model_list=model_list,
            target_model=None,
            cli_model=None,
            details=True,
            prisma_client=None,
            start_time=0.0,
            user_id="user-1",
            model_id=None,
            health_check_skip_disabled_background_models=True,
        )

    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["health_check_skip_disabled_background_models"] is True


def test_parse_background_health_check_model_groups_unset_returns_none():
    from litellm.proxy.health_check import parse_background_health_check_model_groups

    assert parse_background_health_check_model_groups(None) is None
    assert parse_background_health_check_model_groups({}) is None
    assert (
        parse_background_health_check_model_groups(
            {"background_health_check_model_groups": None}
        )
        is None
    )


def test_parse_background_health_check_model_groups_list_returns_frozenset():
    from litellm.proxy.health_check import parse_background_health_check_model_groups

    parsed = parse_background_health_check_model_groups(
        {"background_health_check_model_groups": ["prod-openai", "prod-claude"]}
    )
    assert parsed == frozenset({"prod-openai", "prod-claude"})


@pytest.mark.parametrize("bad_value", ["prod-openai", 42, {"a": 1}, [1, 2], [None]])
def test_parse_background_health_check_model_groups_malformed_raises(bad_value):
    from litellm.proxy.health_check import parse_background_health_check_model_groups

    with pytest.raises(ValueError, match="must be a list of model group names"):
        parse_background_health_check_model_groups(
            {"background_health_check_model_groups": bad_value}
        )


def test_filter_deployments_to_model_groups():
    from litellm.proxy.health_check import filter_deployments_to_model_groups

    model_list = [
        {"model_name": "prod-openai", "model_info": {"id": "a"}},
        {"model_name": "internal-claude", "model_info": {"id": "b"}},
        {"model_name": "prod-openai", "model_info": {"id": "c"}},
    ]

    assert filter_deployments_to_model_groups(model_list, None) == tuple(model_list)
    assert filter_deployments_to_model_groups(
        model_list, frozenset({"prod-openai"})
    ) == (model_list[0], model_list[2])
    assert filter_deployments_to_model_groups(model_list, frozenset()) == ()


if __name__ == "__main__":
    pytest.main([__file__])
