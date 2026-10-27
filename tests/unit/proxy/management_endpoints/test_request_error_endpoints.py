import os
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

from tests._master_key import MASTER_KEY

_THROWAWAY_ENV: Final = {"DATABASE_URL": "sqlite:///:memory:", "LITELLM_MASTER_KEY": MASTER_KEY}
_PRE_EXISTING_ENV: Final = {key: os.environ.get(key) for key in _THROWAWAY_ENV}
for _key, _value in _THROWAWAY_ENV.items():
    os.environ.setdefault(_key, _value)

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.request_error_endpoints import (
    _DateRow,
    _EntityRow,
    _EntityStatusRow,
    fold_by_date,
    fold_by_status_code,
    fold_entities,
    get_request_error_activity,
)

for _key, _previous in _PRE_EXISTING_ENV.items():
    if _previous is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _previous  # test-quality-ok: import-time restore of the pre-existing value

_DATE_ROWS: Final = (
    _DateRow(date="2026-10-07", successful_requests=90, failed_requests=10),
    _DateRow(date="2026-10-07", status_code=429, failed_requests=7),
    _DateRow(date="2026-10-07", status_code=500, failed_requests=2),
    _DateRow(date="2026-10-08", successful_requests=50, failed_requests=5),
    _DateRow(date="2026-10-08", status_code=429, failed_requests=5),
)


def test_fold_by_date_splits_failures_into_client_and_server_errors() -> None:
    by_date: Final = fold_by_date(_DATE_ROWS)
    assert [entry.model_dump(exclude={"by_status_code"}) for entry in by_date] == [
        {
            "date": "2026-10-07",
            "successful_requests": 90,
            "failed_requests": 10,
            "client_errors": 7,
            "server_errors": 2,
        },
        {"date": "2026-10-08", "successful_requests": 50, "failed_requests": 5, "client_errors": 5, "server_errors": 0},
    ]
    assert [entry.model_dump() for entry in by_date[0].by_status_code] == [
        {"status_code": 429, "failed_requests": 7},
        {"status_code": 500, "failed_requests": 2},
    ]
    assert [entry.model_dump() for entry in by_date[1].by_status_code] == [{"status_code": 429, "failed_requests": 5}]


def test_fold_by_status_code_sums_across_days_and_ranks_by_count() -> None:
    assert [entry.model_dump() for entry in fold_by_status_code(_DATE_ROWS)] == [
        {"status_code": 429, "failed_requests": 12},
        {"status_code": 500, "failed_requests": 2},
    ]


def test_fold_entities_attaches_top_status_and_keeps_kinds_apart() -> None:
    rows: Final = (
        _EntityRow(kind="key", id="hash-1", label="prod-key", api_requests=100, failed_requests=10),
        _EntityRow(kind="key", id="hash-2", label=None, api_requests=40, failed_requests=12),
        _EntityRow(kind="team", id="team-a", label="Team A", api_requests=140, failed_requests=22),
    )
    status_rows: Final = (
        _EntityStatusRow(kind="key", id="hash-1", status_code=429, failed_requests=6),
        _EntityStatusRow(kind="team", id="hash-1", status_code=401, failed_requests=99),
    )
    by_key: Final = fold_entities("key", rows, status_rows)
    assert [entry.model_dump() for entry in by_key] == [
        {
            "id": "hash-2",
            "label": None,
            "api_requests": 40,
            "failed_requests": 12,
            "top_status_code": None,
            "top_status_code_requests": 0,
        },
        {
            "id": "hash-1",
            "label": "prod-key",
            "api_requests": 100,
            "failed_requests": 10,
            "top_status_code": 429,
            "top_status_code_requests": 6,
        },
    ]
    assert [entry.id for entry in fold_entities("team", rows, status_rows)] == ["team-a"]
    assert fold_entities("user", rows, status_rows) == ()


def _auth(role: LitellmUserRoles) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_role=role)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM])
async def test_non_admin_roles_are_refused(role: LitellmUserRoles) -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.query_raw = AsyncMock()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma_client):
        with pytest.raises(HTTPException) as refused:
            await get_request_error_activity(user_api_key_dict=_auth(role))
    assert refused.value.status_code == 403
    prisma_client.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_database_is_a_500() -> None:
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        with pytest.raises(HTTPException) as refused:
            await get_request_error_activity(user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN))
    assert refused.value.status_code == 500


@pytest.mark.asyncio
async def test_admin_gets_totals_series_and_caller_breakdown() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.query_raw = AsyncMock(
        side_effect=[
            [row.model_dump() for row in _DATE_ROWS],
            [{"kind": "team", "id": "team-a", "label": "Team A", "api_requests": 140, "failed_requests": 15}],
            [{"kind": "team", "id": "team-a", "status_code": 429, "failed_requests": 12}],
        ]
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma_client):
        response: Final = await get_request_error_activity(
            user_api_key_dict=_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
            start_date="2026-10-01",
            end_date="2026-10-08",
        )
    assert response.total_successful_requests == 140
    assert response.total_failed_requests == 15
    assert [entry.date for entry in response.by_date] == ["2026-10-07", "2026-10-08"]
    assert response.by_status_code[0].status_code == 429
    assert response.by_team[0].top_status_code == 429
    assert response.by_key == ()
    for awaited in prisma_client.db.query_raw.await_args_list:
        assert awaited.args[1:] == ("2026-10-01", "2026-10-08")
