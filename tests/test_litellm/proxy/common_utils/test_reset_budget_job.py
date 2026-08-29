import asyncio
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import prisma
import pytest


from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.common_utils import reset_budget_job as reset_budget_job_module
from litellm.constants import (
    PROXY_BUDGET_RESCHEDULER_MIN_TIME,
    RESET_BUDGET_JOB_LOCK_TTL_SECONDS,
    RESET_BUDGET_JOB_NAME,
)
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings


# Mock classes for testing
class MockTable:
    """A single prisma table: records reads/writes and replays canned rows."""

    def __init__(self):
        self.find_many_calls: List[Dict[str, Any]] = []
        self.update_many_calls: List[Dict[str, Any]] = []
        self._find_many_results: List[Any] = []

    def set_find_many_results(self, results: List[Any]):
        self._find_many_results = results

    async def find_many(self, where: Dict[str, Any]) -> List[Any]:
        self.find_many_calls.append({"where": where})
        return self._find_many_results

    async def update_many(self, where: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        self.update_many_calls.append({"where": where, "data": data})
        return {"count": 1}


class MockBatcher:
    """Captures the writes queued on one `db.batch_()` and whether it committed.

    Mirrors prisma's batch ergonomics enough that the reset job's write helpers
    can run against the mock, and keeps `committed` so tests can prove a failed
    cascade persisted nothing.
    """

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.committed: bool = False

        class _Table:
            def __init__(_self, table_name: str, outer: "MockBatcher"):
                _self._table_name = table_name
                _self._outer = outer

            def _record(_self, op, where, data):
                _self._outer.calls.append({"table": _self._table_name, "op": op, "where": where, "data": data})

            def update(_self, where, data):
                _self._record("update", where, data)

            def update_many(_self, where, data):
                _self._record("update_many", where, data)

        self.litellm_verificationtoken = _Table("key", self)
        self.litellm_usertable = _Table("user", self)
        self.litellm_teamtable = _Table("team", self)
        self.litellm_budgettable = _Table("budget", self)
        self.litellm_teammembership = _Table("team_membership", self)
        self.litellm_organizationtable = _Table("org", self)
        self.litellm_tagtable = _Table("tag", self)
        self.litellm_endusertable = _Table("enduser", self)

    async def commit(self):
        self.committed = True
        return self.calls


class MockDB:
    def __init__(self):
        self.litellm_teammembership = MockTable()
        self.litellm_verificationtoken = MockTable()
        self.litellm_endusertable = MockTable()
        self.litellm_organizationtable = MockTable()
        self.litellm_tagtable = MockTable()
        self.batch_calls: List[Dict[str, Any]] = []
        self.batchers: List[MockBatcher] = []

    def batch_(self):
        batcher = MockBatcher()
        self.batchers.append(batcher)
        # Aggregate calls across all batches so tests can assert on cumulative
        # writes. Only committed batches contribute: an abandoned batch writes
        # nothing, exactly as prisma behaves.
        original_commit = batcher.commit

        async def _record_and_commit():
            self.batch_calls.extend(batcher.calls)
            return await original_commit()

        batcher.commit = _record_and_commit  # type: ignore[assignment]
        return batcher


class MockPrismaClient:
    def __init__(self):
        self.data: Dict[str, List[Any]] = {
            "key": [],
            "user": [],
            "team": [],
            "budget": [],
            "enduser": [],
        }
        self.updated_data: Dict[str, List[Any]] = {
            "key": [],
            "user": [],
            "team": [],
            "budget": [],
            "enduser": [],
        }
        self.get_data_calls: List[Dict[str, Any]] = []
        self.db = MockDB()

    async def get_data(self, table_name, query_type, **kwargs):
        self.get_data_calls.append({"table_name": table_name, "query_type": query_type, **kwargs})
        data = self.data.get(table_name, [])

        # Handle specific filtering for budget table queries
        if table_name == "budget" and query_type == "find_all" and "reset_at" in kwargs:
            # Return budgets that need to be reset (simulate expired budgets)
            return [item for item in data if hasattr(item, "budget_reset_at")]

        # Handle specific filtering for enduser table queries
        if table_name == "enduser" and query_type == "find_all" and "budget_id_list" in kwargs:
            budget_id_list = kwargs["budget_id_list"]
            # Return endusers that match the budget IDs
            return [
                item
                for item in data
                if hasattr(item, "litellm_budget_table")
                and hasattr(item.litellm_budget_table, "budget_id")
                and item.litellm_budget_table.budget_id in budget_id_list
            ]

        # Handle key queries with expires and reset_at
        if table_name == "key" and query_type == "find_all" and ("expires" in kwargs or "reset_at" in kwargs):
            return [item for item in data if hasattr(item, "budget_reset_at")]

        return data

    async def update_data(self, query_type, data_list, table_name):
        self.updated_data[table_name] = data_list
        return data_list


class MockProxyLogging:
    class MockServiceLogging:
        async def async_service_success_hook(self, **kwargs):
            pass

        async def async_service_failure_hook(self, **kwargs):
            pass

    def __init__(self):
        self.service_logging_obj = self.MockServiceLogging()


# Test fixtures
@pytest.fixture
def mock_prisma_client():
    return MockPrismaClient()


@pytest.fixture
def mock_proxy_logging():
    return MockProxyLogging()


@pytest.fixture
def reset_budget_job(mock_prisma_client, mock_proxy_logging):
    return ResetBudgetJob(proxy_logging_obj=mock_proxy_logging, prisma_client=mock_prisma_client)


# Helper function to run async tests
async def run_async_test(coro):
    return await coro


_ALREADY_EXPIRED = object()


def _budget_row(
    budget_id: str = "test-budget-1",
    budget_duration: Any = "7d",
    budget_reset_at: Any = _ALREADY_EXPIRED,
    max_budget: float = 10.0,
):
    """An expiring budget tier, shaped like the rows get_data() hands back."""
    now = datetime.now(timezone.utc)
    return type(
        "LiteLLM_BudgetTableFull",
        (),
        {
            "max_budget": max_budget,
            "budget_duration": budget_duration,
            "budget_reset_at": (now - timedelta(hours=1) if budget_reset_at is _ALREADY_EXPIRED else budget_reset_at),
            "budget_id": budget_id,
            "created_at": now - timedelta(days=30),
        },
    )


def _batch_writes(mock_prisma_client, table: str, op: str | None = None) -> List[Dict[str, Any]]:
    """Writes that were committed to the DB, optionally narrowed to one op."""
    return [
        call
        for call in mock_prisma_client.db.batch_calls
        if call["table"] == table and (op is None or call["op"] == op)
    ]


# Tests
def test_write_key_reset_updates_skips_none_token_and_still_writes_the_rest(reset_budget_job, mock_prisma_client):
    """A key with token=None must be skipped, not queued as where={"token": None}.

    Queueing a None token makes the prisma batch commit raise and aborts the
    whole batch, silently dropping every key reset that cycle (the #27730
    blast radius this write path exists to prevent).
    """
    reset_at = datetime.now(timezone.utc)
    keys = [
        LiteLLM_VerificationToken(token=None, budget_reset_at=reset_at),
        LiteLLM_VerificationToken(token="tok-ok", budget_reset_at=reset_at),
    ]

    asyncio.run(reset_budget_job._write_key_reset_updates(updated_keys=keys))

    assert _batch_writes(mock_prisma_client, "key") == [
        {
            "table": "key",
            "op": "update",
            "where": {"token": "tok-ok"},
            "data": {"spend": 0, "budget_reset_at": reset_at},
        }
    ]


def test_reset_budget_for_key(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)
    test_key = type(
        "LiteLLM_VerificationToken",
        (),
        {
            "spend": 100.0,
            "budget_duration": "30d",
            "budget_reset_at": now,
            "id": "test-key-1",
            "token": "tok-key-1",
        },
    )

    mock_prisma_client.data["key"] = [test_key]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_keys())

    # The reset writes only {spend, budget_reset_at} per row via batch_().
    # Full-row writes would re-detonate the Prisma DataError on rows carrying
    # object_permission_id / budget_limits (see #27730).
    key_writes = [c for c in mock_prisma_client.db.batch_calls if c["table"] == "key"]
    assert len(key_writes) == 1
    write = key_writes[0]
    assert write["where"] == {"token": "tok-key-1"}
    assert write["data"]["spend"] == 0
    assert write["data"]["budget_reset_at"] > now
    assert set(write["data"].keys()) == {"spend", "budget_reset_at"}


def test_reset_budget_for_key_honors_injected_reset_time(mock_prisma_client, mock_proxy_logging):
    """Injected BudgetResetSettings drives the written reset time end to end (DI, no globals).

    Before the configurable-reset-time change this wrote a midnight reset_at (hour 0);
    with noon injected it must write a noon reset_at.
    """
    job = ResetBudgetJob(
        proxy_logging_obj=mock_proxy_logging,
        prisma_client=mock_prisma_client,
        reset_settings=BudgetResetSettings(timezone="UTC", reset_time_of_day=dt_time(12, 0)),
    )
    now = datetime.now(timezone.utc)
    test_key = type(
        "LiteLLM_VerificationToken",
        (),
        {
            "spend": 100.0,
            "budget_duration": "1d",
            "budget_reset_at": now,
            "id": "test-key-noon",
            "token": "tok-noon",
        },
    )
    mock_prisma_client.data["key"] = [test_key]

    asyncio.run(job.reset_budget_for_litellm_keys())

    key_writes = [c for c in mock_prisma_client.db.batch_calls if c["table"] == "key"]
    assert len(key_writes) == 1
    reset_at = key_writes[0]["data"]["budget_reset_at"].astimezone(timezone.utc)
    assert reset_at.hour == 12
    assert reset_at.minute == 0


def test_reset_budget_for_user(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)
    test_user = type(
        "LiteLLM_UserTable",
        (),
        {
            "spend": 200.0,
            "budget_duration": "7d",
            "budget_reset_at": now,
            "id": "test-user-1",
            "user_id": "uid-1",
        },
    )

    mock_prisma_client.data["user"] = [test_user]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_users())

    user_writes = [c for c in mock_prisma_client.db.batch_calls if c["table"] == "user"]
    assert len(user_writes) == 1
    write = user_writes[0]
    assert write["where"] == {"user_id": "uid-1"}
    assert write["data"]["spend"] == 0
    assert write["data"]["budget_reset_at"] > now
    assert set(write["data"].keys()) == {"spend", "budget_reset_at"}


def test_reset_budget_for_team(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)
    test_team = type(
        "LiteLLM_TeamTable",
        (),
        {
            "spend": 500.0,
            "budget_duration": "1mo",
            "budget_reset_at": now,
            "id": "test-team-1",
            "team_id": "tid-1",
        },
    )

    mock_prisma_client.data["team"] = [test_team]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget_for_litellm_teams())

    team_writes = [c for c in mock_prisma_client.db.batch_calls if c["table"] == "team"]
    assert len(team_writes) == 1
    write = team_writes[0]
    assert write["where"] == {"team_id": "tid-1"}
    assert write["data"]["spend"] == 0
    assert write["data"]["budget_reset_at"] > now
    assert set(write["data"].keys()) == {"spend", "budget_reset_at"}


def test_reset_budget_for_enduser(reset_budget_job, mock_prisma_client):
    """End-user spend is zeroed and the tier's window advances, in one batch."""
    now = datetime.now(timezone.utc)
    test_budget = _budget_row(budget_id="test-budget-1", budget_duration="1d", budget_reset_at=now)

    test_enduser = type(
        "LiteLLM_EndUserTable",
        (),
        {
            "spend": 20.0,
            "litellm_budget_table": test_budget,
            "user_id": "test-enduser-1",
        },
    )

    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["enduser"] = [test_enduser]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    assert _batch_writes(mock_prisma_client, "enduser") == [
        {
            "table": "enduser",
            "op": "update_many",
            "where": {"user_id": {"in": ["test-enduser-1"]}},
            "data": {"spend": 0},
        }
    ]

    budget_writes = _batch_writes(mock_prisma_client, "budget")
    assert len(budget_writes) == 1
    assert budget_writes[0]["where"] == {"budget_id": "test-budget-1"}
    assert budget_writes[0]["data"]["budget_reset_at"] > now
    assert set(budget_writes[0]["data"].keys()) == {"budget_reset_at"}


def test_reset_budget_all(reset_budget_job, mock_prisma_client):
    # Setup test data with timezone-aware datetime
    now = datetime.now(timezone.utc)

    # Create test objects for all three types
    test_key = type(
        "LiteLLM_VerificationToken",
        (),
        {
            "spend": 100.0,
            "budget_duration": "30d",
            "budget_reset_at": now,
            "id": "test-key-1",
            "token": "tok-all-1",
        },
    )

    test_user = type(
        "LiteLLM_UserTable",
        (),
        {
            "spend": 200.0,
            "budget_duration": "7d",
            "budget_reset_at": now,
            "id": "test-user-1",
            "user_id": "uid-all-1",
        },
    )

    test_team = type(
        "LiteLLM_TeamTable",
        (),
        {
            "spend": 500.0,
            "budget_duration": "1mo",
            "budget_reset_at": now,
            "id": "test-team-1",
            "team_id": "tid-all-1",
        },
    )

    test_budget = type(
        "LiteLLM_BudgetTable",
        (),
        {
            "max_budget": 500.0,
            "budget_duration": "1d",
            "budget_reset_at": now,
            "budget_id": "test-budget-1",
        },
    )

    test_enduser = type(
        "LiteLLM_EndUserTable",
        (),
        {
            "spend": 20.0,
            "litellm_budget_table": test_budget,
            "user_id": "test-enduser-1",
        },
    )

    mock_prisma_client.data["key"] = [test_key]
    mock_prisma_client.data["user"] = [test_user]
    mock_prisma_client.data["team"] = [test_team]
    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["enduser"] = [test_enduser]

    # Run the test
    asyncio.run(reset_budget_job.reset_budget())

    # key/user/team rows are written via batch_().<table>.update — verify each
    # one fired exactly once with the narrow {spend, budget_reset_at} payload.
    for table_name, where in [
        ("key", {"token": "tok-all-1"}),
        ("user", {"user_id": "uid-all-1"}),
        ("team", {"team_id": "tid-all-1"}),
    ]:
        writes = _batch_writes(mock_prisma_client, table_name, op="update")
        assert len(writes) == 1, f"expected 1 {table_name} write, got {len(writes)}"
        assert writes[0]["where"] == where
        assert writes[0]["data"]["spend"] == 0
        assert set(writes[0]["data"].keys()) == {"spend", "budget_reset_at"}

    # The budget tier's cascade rides the same batch machinery.
    assert _batch_writes(mock_prisma_client, "enduser") == [
        {
            "table": "enduser",
            "op": "update_many",
            "where": {"user_id": {"in": ["test-enduser-1"]}},
            "data": {"spend": 0},
        }
    ]
    assert len(_batch_writes(mock_prisma_client, "budget")) == 1


_LINKED_TABLE_CASES = [
    ("team_membership", {"budget_id": {"in": ["7d-budget-tier"]}}),
    (
        "key",
        {
            "budget_id": {"in": ["7d-budget-tier"]},
            "budget_duration": None,
            "spend": {"gt": 0},
        },
    ),
    ("org", {"budget_id": {"in": ["7d-budget-tier"]}, "spend": {"gt": 0}}),
    ("tag", {"budget_id": {"in": ["7d-budget-tier"]}, "spend": {"gt": 0}}),
]


@pytest.mark.parametrize(
    "table, expected_where",
    _LINKED_TABLE_CASES,
    ids=[case[0] for case in _LINKED_TABLE_CASES],
)
def test_budget_table_reset_zeroes_spend_on_every_linked_table(
    reset_budget_job, mock_prisma_client, table, expected_where
):
    """One expiring tier zeroes spend on every row it gates.

    The filters carry real behavior: keys must be narrowed to
    `budget_duration: None` so keys with their own reset schedule aren't
    double-reset by reset_budget_for_litellm_keys(), and the payload must stay
    exactly {"spend": 0} because `total_spend` is a lifetime counter a reset
    may never touch.
    """
    mock_prisma_client.data["budget"] = [_budget_row(budget_id="7d-budget-tier", budget_duration="7d")]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    writes = _batch_writes(mock_prisma_client, table, op="update_many")
    assert len(writes) == 1, f"expected exactly 1 {table} write, got {writes}"
    assert writes[0]["where"] == expected_where
    assert writes[0]["data"] == {"spend": 0}


def test_budget_table_reset_writes_nothing_when_no_budget_is_due(reset_budget_job, mock_prisma_client):
    """Nothing due means no transaction is opened at all."""
    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    assert mock_prisma_client.db.batchers == []
    assert mock_prisma_client.db.batch_calls == []


def _run_reset_at_fixed_now(job, fixed_now):
    """Run the budget-table reset with `now` pinned for reset-time math."""
    from unittest.mock import patch

    with patch("litellm.proxy.common_utils.timezone_utils.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        asyncio.run(job.reset_budget_for_litellm_budget_table())


@pytest.mark.parametrize(
    "budget_duration, expected_day, expected_month",
    [
        ("30d", 1, 7),  # 30d → 1st of next month
        ("1mo", 1, 7),  # 1mo → 1st of next month
        ("1d", 16, 6),  # 1d → next midnight (same month)
    ],
    ids=["30d-calendar-month", "1mo-calendar-month", "1d-next-midnight"],
)
def test_budget_reset_at_written_is_calendar_aligned(
    reset_budget_job, mock_prisma_client, budget_duration, expected_day, expected_month
):
    """The advanced budget_reset_at is calendar-aligned, not a sliding
    now + duration offset."""
    fixed_now = datetime(2023, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    mock_prisma_client.data["budget"] = [
        _budget_row(
            budget_id="test-budget",
            budget_duration=budget_duration,
            budget_reset_at=fixed_now - timedelta(hours=1),
        )
    ]

    _run_reset_at_fixed_now(reset_budget_job, fixed_now)

    writes = _batch_writes(mock_prisma_client, "budget")
    assert len(writes) == 1
    written = writes[0]["data"]["budget_reset_at"]
    assert (written.day, written.month) == (expected_day, expected_month)
    assert (written.hour, written.minute, written.second) == (0, 0, 0)


def test_budget_reset_at_written_for_7d_is_next_monday(reset_budget_job, mock_prisma_client):
    """7d budgets advance to next Monday at midnight."""
    # 2023-06-14 is a Wednesday
    fixed_now = datetime(2023, 6, 14, 10, 30, 0, tzinfo=timezone.utc)
    mock_prisma_client.data["budget"] = [
        _budget_row(budget_id="test-budget", budget_duration="7d", budget_reset_at=fixed_now - timedelta(hours=1))
    ]

    _run_reset_at_fixed_now(reset_budget_job, fixed_now)

    written = _batch_writes(mock_prisma_client, "budget")[0]["data"]["budget_reset_at"]
    assert (written.day, written.month) == (19, 6)
    assert written.weekday() == 0
    assert written.hour == 0


def test_budget_with_no_duration_gets_no_reset_at_write(reset_budget_job, mock_prisma_client):
    """A tier without a duration has no next window, so its row is left alone
    rather than rewritten with an unchanged value."""
    mock_prisma_client.data["budget"] = [
        _budget_row(
            budget_id="no-duration", budget_duration=None, budget_reset_at=datetime(2023, 6, 20, tzinfo=timezone.utc)
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    assert _batch_writes(mock_prisma_client, "budget") == []


def test_budget_reset_at_written_when_previously_null(reset_budget_job, mock_prisma_client):
    """A tier whose budget_reset_at was never initialized still gets one."""
    fixed_now = datetime(2023, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    mock_prisma_client.data["budget"] = [
        _budget_row(budget_id="test-budget", budget_duration="30d", budget_reset_at=None)
    ]

    _run_reset_at_fixed_now(reset_budget_job, fixed_now)

    written = _batch_writes(mock_prisma_client, "budget")[0]["data"]["budget_reset_at"]
    assert (written.day, written.month) == (1, 7)


def test_reset_budget_resets_endusers_with_null_budget_id(reset_budget_job, mock_prisma_client):
    """
    When litellm.max_end_user_budget_id is configured and that budget is
    being reset, end users with budget_id=NULL should also have their spend
    reset.  These users were implicitly created and have no budget_id persisted,
    but are enforced against the default budget in-memory.
    """
    import litellm

    now = datetime.now(timezone.utc)
    default_budget_id = "default-enduser-budget"
    litellm.max_end_user_budget_id = default_budget_id

    # Budget that is due for reset — matches the default end user budget
    test_budget = type(
        "LiteLLM_BudgetTableFull",
        (),
        {
            "max_budget": 50.0,
            "budget_duration": "1d",
            "budget_reset_at": now - timedelta(hours=1),
            "budget_id": default_budget_id,
            "created_at": now - timedelta(days=1),
        },
    )

    # End user WITH explicit budget_id (found by the normal budget_id_list query)
    enduser_with_budget = type(
        "LiteLLM_EndUserTable",
        (),
        {
            "spend": 30.0,
            "litellm_budget_table": test_budget,
            "user_id": "enduser-explicit",
        },
    )

    # End user WITHOUT budget_id (NULL) — should also be reset
    enduser_no_budget_row = type(
        "EndUserRow",
        (),
        {
            "spend": 25.0,
            "user_id": "enduser-implicit",
            "budget_id": None,
            "alias": None,
            "allowed_model_region": None,
            "default_model": None,
            "blocked": False,
            "object_permission_id": None,
            "object_permission": None,
            "litellm_budget_table": None,
            "model_dump": lambda self=None: {
                "spend": 25.0,
                "user_id": "enduser-implicit",
                "blocked": False,
                "alias": None,
                "allowed_model_region": None,
                "default_model": None,
                "litellm_budget_table": None,
                "object_permission_id": None,
                "object_permission": None,
            },
        },
    )

    mock_prisma_client.data["budget"] = [test_budget]
    mock_prisma_client.data["enduser"] = [enduser_with_budget]

    # Set up the DB mock for NULL-budget-id end users
    mock_prisma_client.db.litellm_endusertable.set_find_many_results([enduser_no_budget_row])

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    # Both end users are zeroed by the same committed statement.
    enduser_writes = _batch_writes(mock_prisma_client, "enduser")
    assert len(enduser_writes) == 1, f"Expected a single enduser write, got {enduser_writes}"
    assert set(enduser_writes[0]["where"]["user_id"]["in"]) == {
        "enduser-explicit",
        "enduser-implicit",
    }
    assert enduser_writes[0]["data"] == {"spend": 0}

    # Verify find_many was called to fetch NULL-budget-id end users
    find_many_calls = mock_prisma_client.db.litellm_endusertable.find_many_calls
    assert len(find_many_calls) == 1
    assert find_many_calls[0]["where"] == {"budget_id": None, "spend": {"gt": 0}}

    litellm.max_end_user_budget_id = None


def test_reset_budget_skips_null_budget_id_endusers_when_default_not_configured(reset_budget_job, mock_prisma_client):
    """
    When litellm.max_end_user_budget_id is NOT configured, end users with
    budget_id=NULL should NOT be fetched or reset.
    """
    import litellm

    now = datetime.now(timezone.utc)
    litellm.max_end_user_budget_id = None

    test_budget = type(
        "LiteLLM_BudgetTableFull",
        (),
        {
            "max_budget": 50.0,
            "budget_duration": "1d",
            "budget_reset_at": now - timedelta(hours=1),
            "budget_id": "some-budget",
            "created_at": now - timedelta(days=1),
        },
    )

    mock_prisma_client.data["budget"] = [test_budget]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    # Should NOT have queried for NULL-budget-id end users
    find_many_calls = mock_prisma_client.db.litellm_endusertable.find_many_calls
    assert len(find_many_calls) == 0

    litellm.max_end_user_budget_id = None


def test_reset_budget_skips_null_budget_id_endusers_when_default_not_in_reset_list(
    reset_budget_job, mock_prisma_client
):
    """
    When litellm.max_end_user_budget_id IS configured but the corresponding
    budget is NOT in the budgets-to-reset list (not yet expired), end users
    with budget_id=NULL should NOT be reset.
    """
    import litellm

    now = datetime.now(timezone.utc)
    litellm.max_end_user_budget_id = "default-budget-not-expired"

    # A different budget that IS expiring (not the default one)
    test_budget = type(
        "LiteLLM_BudgetTableFull",
        (),
        {
            "max_budget": 50.0,
            "budget_duration": "1d",
            "budget_reset_at": now - timedelta(hours=1),
            "budget_id": "other-budget",
            "created_at": now - timedelta(days=1),
        },
    )

    mock_prisma_client.data["budget"] = [test_budget]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    # Should NOT have queried for NULL-budget-id end users
    find_many_calls = mock_prisma_client.db.litellm_endusertable.find_many_calls
    assert len(find_many_calls) == 0

    litellm.max_end_user_budget_id = None


# ---------------------------------------------------------------------------
# reset_budget_windows (per-key / per-team concurrent window resets)
# ---------------------------------------------------------------------------


def _make_reset_budget_windows_job(
    monkeypatch,
    key_rows: List[Dict[str, Any]],
    team_rows: List[Dict[str, Any]],
):
    """Build a ResetBudgetJob with a fully-mocked prisma client and a fake
    `litellm.proxy.proxy_server` module exposing a stub `spend_counter_cache`.

    Returns (job, prisma_client_mock, spend_counter_cache_mock).
    """
    prisma_client = MagicMock()

    async def fake_query_raw(query: str, *args, **kwargs):
        # Dispatch by table name in the SQL so a single stub covers both calls.
        if '"LiteLLM_VerificationToken"' in query:
            return key_rows
        if '"LiteLLM_TeamTable"' in query:
            return team_rows
        raise AssertionError(f"Unexpected query_raw call: {query}")

    prisma_client.db.query_raw = AsyncMock(side_effect=fake_query_raw)
    prisma_client.db.litellm_verificationtoken.update = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=None)

    # Stub out litellm.proxy.proxy_server so the in-function
    # `from litellm.proxy.proxy_server import spend_counter_cache` resolves
    # without importing the real (heavy) module.
    spend_counter_cache = MagicMock()
    spend_counter_cache.in_memory_cache.set_cache = MagicMock()
    spend_counter_cache.redis_cache = None  # skip the async redis branch

    fake_module = types.ModuleType("litellm.proxy.proxy_server")
    fake_module.spend_counter_cache = spend_counter_cache
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_module)

    job = ResetBudgetJob(proxy_logging_obj=MagicMock(), prisma_client=prisma_client)
    return job, prisma_client, spend_counter_cache


def test_reset_budget_windows_uses_is_not_null_filter(monkeypatch):
    """Regression guard for the Prisma client limitation documented in
    RobertCraigie/prisma-client-py#714: `{"not": None}` on a `Json?` column
    raises `MissingRequiredValueError`. We work around it by using `query_raw`
    with `IS NOT NULL`. If someone reverts to the ORM filter, this test fails.
    """
    job, prisma_client, _ = _make_reset_budget_windows_job(monkeypatch, key_rows=[], team_rows=[])

    asyncio.run(job.reset_budget_windows())

    queries = [call.args[0] for call in prisma_client.db.query_raw.await_args_list]
    assert len(queries) == 2, queries
    key_query, team_query = queries

    assert '"LiteLLM_VerificationToken"' in key_query
    assert "budget_limits IS NOT NULL" in key_query
    assert '"LiteLLM_TeamTable"' in team_query
    assert "budget_limits IS NOT NULL" in team_query


def test_reset_budget_windows_resets_expired_key_window(monkeypatch):
    """A key whose window's `reset_at` has passed gets an update with a new
    `reset_at` in the future, and the in-memory spend counter is cleared."""
    now = datetime.utcnow()
    expired = (now - timedelta(minutes=5)).isoformat() + "Z"

    key_rows = [
        {
            "token": "sk-expired",
            "budget_limits": [{"budget_duration": "1d", "reset_at": expired}],
        }
    ]
    job, prisma_client, spend_counter_cache = _make_reset_budget_windows_job(
        monkeypatch, key_rows=key_rows, team_rows=[]
    )

    asyncio.run(job.reset_budget_windows())

    # Update should have been called exactly once with the expired token.
    prisma_client.db.litellm_verificationtoken.update.assert_awaited_once()
    call_kwargs = prisma_client.db.litellm_verificationtoken.update.await_args.kwargs
    assert call_kwargs["where"] == {"token": "sk-expired"}

    # The `budget_limits` payload is re-serialized JSON with a bumped reset_at.
    written_windows = json.loads(call_kwargs["data"]["budget_limits"])
    assert len(written_windows) == 1
    new_reset_at = datetime.fromisoformat(written_windows[0]["reset_at"].replace("Z", "+00:00")).replace(tzinfo=None)
    assert new_reset_at > now

    # The spend counter for this key+window was cleared.
    spend_counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:key:sk-expired:window:1d", value=0.0)


def test_reset_budget_windows_skips_unexpired_key_window(monkeypatch):
    """If `reset_at` is in the future, no write should happen for that key."""
    now = datetime.utcnow()
    future = (now + timedelta(hours=1)).isoformat() + "Z"

    key_rows = [
        {
            "token": "sk-future",
            "budget_limits": [{"budget_duration": "1d", "reset_at": future}],
        }
    ]
    job, prisma_client, _ = _make_reset_budget_windows_job(monkeypatch, key_rows=key_rows, team_rows=[])

    asyncio.run(job.reset_budget_windows())

    prisma_client.db.litellm_verificationtoken.update.assert_not_awaited()


def test_reset_budget_windows_resets_expired_team_window(monkeypatch):
    """Same as the key test, but for teams."""
    now = datetime.utcnow()
    expired = (now - timedelta(minutes=1)).isoformat() + "Z"

    team_rows = [
        {
            "team_id": "team-expired",
            "budget_limits": [{"budget_duration": "30d", "reset_at": expired}],
        }
    ]
    job, prisma_client, spend_counter_cache = _make_reset_budget_windows_job(
        monkeypatch, key_rows=[], team_rows=team_rows
    )

    asyncio.run(job.reset_budget_windows())

    prisma_client.db.litellm_teamtable.update.assert_awaited_once()
    call_kwargs = prisma_client.db.litellm_teamtable.update.await_args.kwargs
    assert call_kwargs["where"] == {"team_id": "team-expired"}
    assert "budget_limits" in call_kwargs["data"]

    spend_counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:team:team-expired:window:30d", value=0.0)


def test_reset_budget_windows_handles_string_budget_limits(monkeypatch):
    """Defensive: if `query_raw` returns `budget_limits` as a JSON-encoded
    string (driver-dependent), the code still parses and resets it.
    """
    now = datetime.utcnow()
    expired = (now - timedelta(minutes=1)).isoformat() + "Z"

    key_rows = [
        {
            "token": "sk-string-limits",
            "budget_limits": json.dumps([{"budget_duration": "1d", "reset_at": expired}]),
        }
    ]
    job, prisma_client, _ = _make_reset_budget_windows_job(monkeypatch, key_rows=key_rows, team_rows=[])

    asyncio.run(job.reset_budget_windows())

    prisma_client.db.litellm_verificationtoken.update.assert_awaited_once()


def test_reset_budget_windows_skips_row_with_empty_budget_limits(monkeypatch):
    """A row whose `budget_limits` comes back as an empty/falsy payload
    (shouldn't happen given the WHERE filter, but we guard anyway) must not
    trigger an update or crash the loop."""
    key_rows = [
        {"token": "sk-empty-list", "budget_limits": []},
        {"token": "sk-empty-str", "budget_limits": ""},
    ]
    job, prisma_client, _ = _make_reset_budget_windows_job(monkeypatch, key_rows=key_rows, team_rows=[])

    asyncio.run(job.reset_budget_windows())

    prisma_client.db.litellm_verificationtoken.update.assert_not_awaited()


def test_reset_budget_windows_query_error_does_not_break_team_path(monkeypatch):
    """If the key query raises, the teams path still runs (and vice-versa).
    Each side has its own try/except; this locks that in."""
    now = datetime.utcnow()
    expired = (now - timedelta(minutes=1)).isoformat() + "Z"

    prisma_client = MagicMock()

    async def fake_query_raw(query: str, *args, **kwargs):
        if '"LiteLLM_VerificationToken"' in query:
            raise RuntimeError("boom")
        if '"LiteLLM_TeamTable"' in query:
            return [
                {
                    "team_id": "team-ok",
                    "budget_limits": [{"budget_duration": "1d", "reset_at": expired}],
                }
            ]
        raise AssertionError(query)

    prisma_client.db.query_raw = AsyncMock(side_effect=fake_query_raw)
    prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=None)

    spend_counter_cache = MagicMock()
    spend_counter_cache.in_memory_cache.set_cache = MagicMock()
    spend_counter_cache.redis_cache = None
    fake_module = types.ModuleType("litellm.proxy.proxy_server")
    fake_module.spend_counter_cache = spend_counter_cache
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_module)

    job = ResetBudgetJob(proxy_logging_obj=MagicMock(), prisma_client=prisma_client)

    asyncio.run(job.reset_budget_windows())  # must not raise

    prisma_client.db.litellm_teamtable.update.assert_awaited_once()


# ---------------------------------------------------------------------------
# Counter invalidation on budget reset
# ---------------------------------------------------------------------------


def _make_counter_invalidation_job(monkeypatch):
    """Stub spend_counter_cache (and user_api_key_cache) so we can observe
    invalidation calls.

    Both caches are looked up via ``from litellm.proxy.proxy_server import
    <name>`` inside the reset job, so we publish them on a fake module.
    """
    spend_counter_cache = MagicMock()
    spend_counter_cache.in_memory_cache.set_cache = MagicMock()
    spend_counter_cache.redis_cache = MagicMock()
    spend_counter_cache.redis_cache.async_set_cache = AsyncMock()

    user_api_key_cache = MagicMock()
    user_api_key_cache.async_delete_cache = AsyncMock()

    fake_module = types.ModuleType("litellm.proxy.proxy_server")
    fake_module.spend_counter_cache = spend_counter_cache
    fake_module.user_api_key_cache = user_api_key_cache
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_module)

    spend_counter_cache.user_api_key_cache = user_api_key_cache
    return spend_counter_cache


def test_reset_budget_for_keys_invalidates_redis_counter(reset_budget_job, mock_prisma_client, monkeypatch):
    """Key budget reset must clear the Redis spend counter."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)

    now = datetime.now(timezone.utc)
    mock_prisma_client.data["key"] = [
        type(
            "Key",
            (),
            {
                "spend": 100.0,
                "budget_duration": "30d",
                "budget_reset_at": now,
                "id": "key-1",
                "token": "sk-abc",
            },
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_keys())

    counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:key:sk-abc", value=0.0, ttl=60)


def test_reset_budget_for_users_invalidates_redis_counter(reset_budget_job, mock_prisma_client, monkeypatch):
    """User budget reset must clear the Redis spend counter."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)

    now = datetime.now(timezone.utc)
    mock_prisma_client.data["user"] = [
        type(
            "User",
            (),
            {
                "spend": 50.0,
                "budget_duration": "7d",
                "budget_reset_at": now,
                "id": "user-1",
                "user_id": "alice",
            },
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_users())

    counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:user:alice", value=0.0, ttl=60)


def test_reset_budget_for_proxy_budget_row_invalidates_global_spend_cache(
    reset_budget_job, mock_prisma_client, monkeypatch
):
    """Regression for LIT-4309: resetting the proxy-wide budget aggregate row
    ("litellm-proxy-budget") must also drop the cached global-spend
    accumulator ("{admin}:spend") that _global_proxy_budget_check enforces
    against. Without the invalidation, the cached value survives the DB reset
    and the global cap keeps blocking requests for the whole next window."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)

    now = datetime.now(timezone.utc)
    mock_prisma_client.data["user"] = [
        type(
            "User",
            (),
            {
                "spend": 150.0,
                "budget_duration": "30d",
                "budget_reset_at": now,
                "id": "row-1",
                "user_id": "litellm-proxy-budget",
            },
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_users())

    counter_cache.user_api_key_cache.async_delete_cache.assert_any_call(key="default_user_id:spend")


def test_reset_budget_for_ordinary_user_does_not_touch_global_spend_cache(
    reset_budget_job, mock_prisma_client, monkeypatch
):
    """The global-spend accumulator must only be dropped when the proxy
    budget aggregate row itself resets, not on every user reset."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)

    now = datetime.now(timezone.utc)
    mock_prisma_client.data["user"] = [
        type(
            "User",
            (),
            {
                "spend": 50.0,
                "budget_duration": "7d",
                "budget_reset_at": now,
                "id": "user-1",
                "user_id": "alice",
            },
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_users())

    assert not any(
        call.kwargs.get("key") == "default_user_id:spend"
        for call in counter_cache.user_api_key_cache.async_delete_cache.call_args_list
    )


def test_reset_budget_for_teams_invalidates_redis_counter(reset_budget_job, mock_prisma_client, monkeypatch):
    """Team budget reset must clear the Redis spend counter."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)

    now = datetime.now(timezone.utc)
    mock_prisma_client.data["team"] = [
        type(
            "Team",
            (),
            {
                "spend": 200.0,
                "budget_duration": "1mo",
                "budget_reset_at": now,
                "id": "team-1",
                "team_id": "team-x",
            },
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_teams())

    counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:team:team-x", value=0.0, ttl=60)


def test_reset_does_not_zero_counter_when_db_write_fails(monkeypatch):
    """
    Regression for #27730 (the bypass-half).

    If the DB write inside the reset job raises (e.g. Prisma DataError on a
    row carrying object_permission_id or budget_limits), the Redis spend
    counter MUST NOT be zeroed — that would let get_current_spend admit
    requests past the cap while the DB row still holds the over-budget
    spend.

    Pre-fix: _reset_budget_common pre-zeroed the counter before the DB
    write attempt, opening the bypass window.
    Post-fix: counter invalidation lives in the caller, AFTER the DB write
    commits. If the write raises, the post-write invalidation never runs.
    """
    counter_cache = _make_counter_invalidation_job(monkeypatch)

    now = datetime.now(timezone.utc)
    prisma_client = MagicMock()

    matching_key = type(
        "Key",
        (),
        {
            "spend": 100.0,
            "budget_duration": "30d",
            "budget_reset_at": now - timedelta(seconds=1),
            "token": "sk-failing",
        },
    )

    # get_data returns one key needing reset; the batched DB write then explodes.
    async def fake_get_data(table_name, query_type, **kwargs):
        if table_name == "key":
            return [matching_key]
        return []

    prisma_client.get_data = fake_get_data

    batcher = MagicMock()
    batcher.litellm_verificationtoken.update = MagicMock()

    async def failing_commit():
        raise RuntimeError("simulated Prisma DataError on update")

    batcher.commit = failing_commit
    prisma_client.db.batch_ = MagicMock(return_value=batcher)

    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=prisma_client)

    asyncio.run(job.reset_budget_for_litellm_keys())

    # CRITICAL: counter invalidation must NOT have been called at all —
    # the DB write raised before the post-write invalidation loop. Using
    # assert_not_called() instead of iterating call_args_list, because the
    # latter is vacuously true when the list is empty (would pass even if
    # the bypass were re-introduced via a different code path).
    counter_cache.in_memory_cache.set_cache.assert_not_called()


def test_reset_budget_for_keys_writes_only_spend_and_reset_at(reset_budget_job, mock_prisma_client):
    """
    Regression for #27730 (the trigger-half).

    The reset job must write only {spend, budget_reset_at} per row — never
    the full key object. Sending the full object via the old update_data
    batcher path made Prisma reject any row carrying object_permission_id
    or budget_limits (both became non-NULL on UI-created keys after v1.84.0).
    """
    now = datetime.now(timezone.utc)
    key_with_problematic_fields = type(
        "LiteLLM_VerificationToken",
        (),
        {
            "spend": 50.0,
            "budget_duration": "30d",
            "budget_reset_at": now,
            "token": "sk-problematic",
            "object_permission_id": "perm-abc",  # would be rejected on update
            "budget_limits": [{"max_budget": 5}],  # would be rejected on update
            "metadata": {"some": "thing"},
        },
    )
    mock_prisma_client.data["key"] = [key_with_problematic_fields]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_keys())

    key_writes = [c for c in mock_prisma_client.db.batch_calls if c["table"] == "key"]
    assert len(key_writes) == 1
    payload_keys = set(key_writes[0]["data"].keys())
    assert payload_keys == {"spend", "budget_reset_at"}, (
        f"reset payload must not include any field besides spend / budget_reset_at, "
        f"got: {payload_keys}. Any extra field (object_permission_id, budget_limits, etc.) "
        f"trips Prisma DataError and detonates the whole batch."
    )


_INVALIDATION_CASES = [
    (
        "litellm_teammembership",
        type("Membership", (), {"user_id": "alice", "team_id": "team-x", "budget_id": "budget-1"}),
        "spend:team_member:alice:team-x",
        {"team-x_alice"},
    ),
    (
        "litellm_verificationtoken",
        type("Key", (), {"token": "sk-linked"}),
        "spend:key:sk-linked",
        {"sk-linked"},
    ),
    (
        "litellm_organizationtable",
        type("Org", (), {"organization_id": "org-acme"}),
        "spend:org:org-acme",
        {"org_id:org-acme", "org_id:org-acme:with_budget"},
    ),
    (
        "litellm_tagtable",
        type("Tag", (), {"tag_name": "tenant-42"}),
        "spend:tag:tenant-42",
        {"tag:tenant-42"},
    ),
]


@pytest.mark.parametrize(
    "table_attr, linked_row, counter_key, cache_keys",
    _INVALIDATION_CASES,
    ids=["team_membership", "key", "org", "tag"],
)
def test_budget_table_reset_invalidates_counters_and_management_cache(
    reset_budget_job, mock_prisma_client, monkeypatch, table_attr, linked_row, counter_key, cache_keys
):
    """Every row the cascade zeroes gets its spend counter cleared and its
    management-cache entry dropped.

    Both matter. ``SpendCounterReseed.from_db`` returns None for tags, so once
    the counter expires the budget check falls back to the cached row's
    ``.spend``; and for keys, orgs and team memberships another pod's cached
    object can stay pinned above the zeroed DB row until its TTL. Team
    membership cache keys follow auth's ``{team_id}_{user_id}`` shape, and orgs
    carry both the plain and the ``:with_budget`` entry.
    """
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    mock_prisma_client.data["budget"] = [_budget_row(budget_id="budget-1")]
    getattr(mock_prisma_client.db, table_attr).set_find_many_results([linked_row])

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    counter_cache.in_memory_cache.set_cache.assert_any_call(key=counter_key, value=0.0, ttl=60)
    counter_cache.redis_cache.async_set_cache.assert_any_await(key=counter_key, value=0.0, ttl=60)
    deleted = {call.kwargs.get("key") for call in counter_cache.user_api_key_cache.async_delete_cache.await_args_list}
    assert cache_keys <= deleted


def test_budget_table_reset_invalidates_every_tag_not_just_the_first(reset_budget_job, mock_prisma_client, monkeypatch):
    """When several tags share the expiring tier, all of them are evicted."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    mock_prisma_client.data["budget"] = [_budget_row(budget_id="budget-1")]
    mock_prisma_client.db.litellm_tagtable.set_find_many_results(
        [type("Tag", (), {"tag_name": name}) for name in ("tenant-a", "tenant-b", "tenant-c")]
    )

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    deleted = {call.kwargs.get("key") for call in counter_cache.user_api_key_cache.async_delete_cache.await_args_list}
    assert deleted == {"tag:tenant-a", "tag:tenant-b", "tag:tenant-c"}


def test_budget_table_reset_commits_even_when_cache_eviction_fails(reset_budget_job, mock_prisma_client, monkeypatch):
    """Eviction runs after the commit, so a broken cache cannot undo the write."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    counter_cache.user_api_key_cache.async_delete_cache = AsyncMock(side_effect=RuntimeError("cache unavailable"))
    mock_prisma_client.data["budget"] = [_budget_row(budget_id="budget-1")]
    mock_prisma_client.db.litellm_tagtable.set_find_many_results([type("Tag", (), {"tag_name": "tenant-42"})])

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    assert len(_batch_writes(mock_prisma_client, "tag", op="update_many")) == 1
    assert mock_prisma_client.db.batchers[0].committed is True


# ---------------------------------------------------------------------------
# Atomicity of the budget-table cascade (LIT-5138)
# ---------------------------------------------------------------------------


class FailingCommitDB(MockDB):
    """Batches that blow up at commit, like a Postgres timeout mid-cascade."""

    def batch_(self):
        batcher = super().batch_()

        async def _fail():
            raise RuntimeError("simulated Postgres timeout mid-cascade")

        batcher.commit = _fail
        return batcher


class FailingTeamMembershipDB(MockDB):
    """Queueing the team-membership reset raises, i.e. the cascade breaks after
    earlier writes are already queued."""

    def batch_(self):
        batcher = super().batch_()

        def _fail(where, data):
            raise RuntimeError("simulated failure queueing the team-membership reset")

        batcher.litellm_teammembership.update_many = _fail
        return batcher


class OrderRecordingDB(MockDB):
    """Appends a marker to a shared list when a batch commits."""

    def __init__(self, events):
        super().__init__()
        self._events = events

    def batch_(self):
        batcher = super().batch_()
        wrapped = batcher.commit

        async def _record_commit():
            self._events.append("commit")
            return await wrapped()

        batcher.commit = _record_commit
        return batcher


def _job_with_expired_budget(db, proxy_logging=None):
    """A job with one due tier and a linked tag, so cache invalidation has
    something to invalidate and its absence is a real signal."""
    prisma_client = MockPrismaClient()
    prisma_client.db = db
    prisma_client.data["budget"] = [_budget_row(budget_id="budget-1", budget_duration="7d")]
    db.litellm_tagtable.set_find_many_results([type("Tag", (), {"tag_name": "tenant-42"})])
    job = ResetBudgetJob(
        proxy_logging_obj=proxy_logging or MockProxyLogging(),
        prisma_client=prisma_client,
    )
    return job, prisma_client


@pytest.mark.parametrize(
    "db_factory",
    [FailingCommitDB, FailingTeamMembershipDB],
    ids=["commit-fails", "queueing-fails"],
)
def test_budget_reset_at_is_not_advanced_when_the_cascade_fails(db_factory, monkeypatch):
    """Regression for LIT-5138.

    The old code committed the new budget_reset_at first and zeroed the
    dependent spend afterwards. A failure part-way through left the tier
    stamped for the next window, so every later tick skipped it and team
    member / enduser / org / tag spend stayed at the cap for the whole window.
    One transaction means a failure anywhere persists nothing and the tier is
    still due on the next tick.
    """
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    job, prisma_client = _job_with_expired_budget(db_factory())

    asyncio.run(job.reset_budget_for_litellm_budget_table())  # swallowed, retried next tick

    assert prisma_client.db.batch_calls == [], "a failed cascade must not persist any write"
    assert prisma_client.db.batchers[0].committed is False
    assert prisma_client.updated_data["budget"] == [], "budget_reset_at must not be advanced outside the transaction"
    counter_cache.in_memory_cache.set_cache.assert_not_called()
    counter_cache.user_api_key_cache.async_delete_cache.assert_not_awaited()


def test_budget_cascade_writes_land_in_a_single_transaction(reset_budget_job, mock_prisma_client, monkeypatch):
    """Dependent spend and the budget_reset_at advance ride one batch."""
    _make_counter_invalidation_job(monkeypatch)
    now = datetime.now(timezone.utc)
    budget = _budget_row(budget_id="budget-1", budget_duration="7d")
    mock_prisma_client.data["budget"] = [budget]
    mock_prisma_client.data["enduser"] = [
        type("EndUser", (), {"spend": 5.0, "litellm_budget_table": budget, "user_id": "enduser-1", "budget_id": "budget-1"})
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    assert len(mock_prisma_client.db.batchers) == 1, "the cascade must not be split across transactions"
    batcher = mock_prisma_client.db.batchers[0]
    assert batcher.committed is True
    assert {(call["table"], call["op"]) for call in batcher.calls} == {
        ("team_membership", "update_many"),
        ("key", "update_many"),
        ("org", "update_many"),
        ("tag", "update_many"),
        ("enduser", "update_many"),
        ("budget", "update_many"),
    }
    budget_write = next(call for call in batcher.calls if call["table"] == "budget")
    assert budget_write["data"]["budget_reset_at"] > now


def test_caches_are_invalidated_only_after_the_transaction_commits(monkeypatch):
    """A counter zeroed before the write lands would admit requests past the
    cap while the DB still holds the over-budget spend."""
    events = []
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    counter_cache.in_memory_cache.set_cache.side_effect = lambda **kwargs: events.append("counter")

    job, _ = _job_with_expired_budget(OrderRecordingDB(events))

    asyncio.run(job.reset_budget_for_litellm_budget_table())

    assert events == ["commit", "counter"]


def test_failed_cascade_is_logged_as_a_cascade_failure(monkeypatch):
    """The failure log has to name what actually broke. The old catch-all
    blamed end users even when the team-membership write was the failure."""
    from unittest.mock import patch

    _make_counter_invalidation_job(monkeypatch)
    job, _ = _job_with_expired_budget(FailingTeamMembershipDB())

    with patch("litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception") as mock_exception:
        asyncio.run(job.reset_budget_for_litellm_budget_table())

    assert mock_exception.call_count == 1
    message = mock_exception.call_args.args[0]
    assert "cascade" in message
    for mentioned in ("team member", "enduser", "org", "tag", "budget_reset_at"):
        assert mentioned in message, f"failure log should mention {mentioned}: {message}"


def _extract_reset_where(find_many_mock):
    """Return the ``where`` dict passed to a mocked repository ``find_many``."""
    assert find_many_mock.await_count == 1
    _, kwargs = find_many_mock.await_args
    return kwargs["where"]


def _asserts_null_reset_is_due(where):
    """A budget-reset ``find_many`` filter must select rows whose
    ``budget_reset_at`` is NULL but which have a ``budget_duration`` set, in
    addition to rows whose ``budget_reset_at`` is already in the past.

    Regression guard: a user/team seeded from ``default_internal_user_params``
    (or created via ``/user/new`` without an explicit ``budget_reset_at``) has
    ``budget_duration`` set but ``budget_reset_at = NULL``. A plain
    ``{"budget_reset_at": {"lt": now}}`` filter never matches NULL, so such rows
    would never be reset and their spend would accumulate for the lifetime of
    the row, silently exceeding ``max_budget``.
    """
    branches = where.get("OR")
    assert isinstance(branches, list), f"expected an OR filter, got {where!r}"

    has_null_branch = {"budget_reset_at": None} in branches
    has_expired_branch = any(isinstance(b, dict) and isinstance(b.get("budget_reset_at"), dict) for b in branches)
    assert has_null_branch, f"missing NULL-reset_at branch in {where!r}"
    assert has_expired_branch, f"missing expired-reset_at branch in {where!r}"
    assert where.get("NOT") == {"budget_duration": None}, f"NULL reset_at is only due with a duration: {where!r}"


_RESET_TABLE_ATTRS = {
    "user": "litellm_usertable",
    "team": "litellm_teamtable",
    "budget": "litellm_budgettable",
    "key": "litellm_verificationtoken",
}


def _run_reset_query(table_name, **extra):
    """Run ``get_data`` for one table's budget-reset query against a mocked
    prisma handle, and hand back the ``find_many`` mock it drove."""
    from litellm.proxy.utils import PrismaClient

    client = PrismaClient.__new__(PrismaClient)
    client.db = MagicMock()
    find_many = AsyncMock(return_value=[])
    setattr(getattr(client.db, _RESET_TABLE_ATTRS[table_name]), "find_many", find_many)

    now = datetime.now(timezone.utc)
    expires = {"expires": now} if table_name == "key" else {}
    asyncio.run(client.get_data(table_name=table_name, query_type="find_all", reset_at=now, **expires, **extra))
    return find_many


@pytest.mark.parametrize("table_name", ["user", "team", "budget", "key"])
def test_get_data_reset_query_applies_the_row_limit(table_name):
    """The reset job pages through due rows, so ``limit`` has to reach prisma as
    ``take``. Dropped, every worker goes back to pulling the entire expired set
    in one unbounded query at the same calendar boundary."""
    find_many = _run_reset_query(table_name, limit=7)

    assert find_many.await_args.kwargs["take"] == 7


@pytest.mark.parametrize("table_name", ["user", "team", "budget", "key"])
def test_get_data_reset_query_skips_rows_with_no_budget_duration(table_name):
    """A row with a past budget_reset_at but no budget_duration has no next
    window to move to, so it stays due forever. Fetching it means re-reading and
    re-zeroing it on every tick, and a full chunk of such rows makes the paged
    scan report no progress and starve the whole phase.
    """
    find_many = _run_reset_query(table_name)

    assert find_many.await_args.kwargs["where"]["NOT"] == {"budget_duration": None}


@pytest.mark.parametrize("table_name", ["user", "team", "budget", "key"])
def test_get_data_reset_query_is_unlimited_when_no_limit_is_passed(table_name):
    """Callers that pass no limit keep the old unbounded behaviour."""
    find_many = _run_reset_query(table_name)

    assert find_many.await_args.kwargs.get("take") is None


@pytest.mark.parametrize("table_name", ["user", "team"])
def test_get_data_reset_query_selects_null_budget_reset_at(table_name):
    """``PrismaClient.get_data(..., reset_at=...)`` for the user and team tables
    must select rows with a NULL ``budget_reset_at`` (and a non-NULL
    ``budget_duration``), matching the budget-table query. Without this, users
    auto-created from ``default_internal_user_params`` are never reset."""
    from litellm.proxy.utils import PrismaClient

    # Build a PrismaClient without running its heavy __init__; only .db is used.
    client = PrismaClient.__new__(PrismaClient)
    client.db = MagicMock()

    find_many = AsyncMock(return_value=[])
    table_attr = {
        "user": "litellm_usertable",
        "team": "litellm_teamtable",
    }[table_name]
    setattr(getattr(client.db, table_attr), "find_many", find_many)

    now = datetime.now(timezone.utc)
    asyncio.run(client.get_data(table_name=table_name, query_type="find_all", reset_at=now))

    _asserts_null_reset_is_due(_extract_reset_where(find_many))


def _key_row(token: str, budget_duration: Any = "30d"):
    """A key that is already due for a reset, shaped like a get_data() row."""
    now = datetime.now(timezone.utc)
    return type(
        "LiteLLM_VerificationToken",
        (),
        {
            "spend": 100.0,
            "budget_duration": budget_duration,
            "budget_reset_at": now - timedelta(hours=1),
            "token": token,
        },
    )


def _user_row(user_id: str, budget_duration: Any = "30d"):
    now = datetime.now(timezone.utc)
    return type(
        "LiteLLM_UserTable",
        (),
        {
            "spend": 100.0,
            "budget_duration": budget_duration,
            "budget_reset_at": now - timedelta(hours=1),
            "user_id": user_id,
        },
    )


def _team_row(team_id: str, budget_duration: Any = "30d"):
    now = datetime.now(timezone.utc)
    return type(
        "LiteLLM_TeamTable",
        (),
        {
            "spend": 100.0,
            "budget_duration": budget_duration,
            "budget_reset_at": now - timedelta(hours=1),
            "team_id": team_id,
        },
    )


# ---------------------------------------------------------------------------
# Chunked batches
# ---------------------------------------------------------------------------


class ChunkedPrismaClient(MockPrismaClient):
    """Replays a scripted sequence of get_data chunks per table.

    The last chunk repeats forever, so a phase that fails to terminate keeps
    seeing rows rather than quietly running out of data.
    """

    def __init__(self, chunks_by_table: Dict[str, List[List[Any]]]):
        super().__init__()
        self._chunks_by_table = chunks_by_table
        self.fetches_by_table: Dict[str, int] = {}

    async def get_data(self, table_name, query_type, **kwargs):
        self.get_data_calls.append({"table_name": table_name, "query_type": query_type, **kwargs})
        chunks = self._chunks_by_table.get(table_name)
        if not chunks:
            return []
        index = self.fetches_by_table.get(table_name, 0)
        self.fetches_by_table[table_name] = index + 1
        return chunks[min(index, len(chunks) - 1)]


def _chunked_job(chunks_by_table):
    client = ChunkedPrismaClient(chunks_by_table)
    return client, ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)


def _fetch_limits(client, table_name):
    return [call.get("limit") for call in client.get_data_calls if call["table_name"] == table_name]


def test_key_reset_walks_the_due_rows_one_chunk_at_a_time(monkeypatch):
    """Each chunk is fetched under a LIMIT and committed on its own batch, so a
    large backlog never becomes one giant transaction."""
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    client, job = _chunked_job({"key": [[_key_row("k1"), _key_row("k2")], [_key_row("k3")]]})

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.fetches_by_table["key"] == 2
    assert _fetch_limits(client, "key") == [2, 2]
    assert len(client.db.batchers) == 2
    assert all(batcher.committed for batcher in client.db.batchers)
    assert [len(batcher.calls) for batcher in client.db.batchers] == [2, 1]
    assert [w["where"]["token"] for w in _batch_writes(client, "key", op="update")] == ["k1", "k2", "k3"]


def test_key_reset_stops_after_a_chunk_shorter_than_the_batch_size(monkeypatch):
    """Fewer rows than the limit means the backlog is drained, so no follow-up
    query is worth issuing."""
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 5)
    client, job = _chunked_job({"key": [[_key_row("k1")]]})

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.fetches_by_table["key"] == 1


def test_key_reset_stops_when_a_full_chunk_advances_nothing(monkeypatch):
    """A key with no budget_duration keeps its past budget_reset_at, so the very
    same rows come back on the next fetch. Treating those writes as progress
    would re-read that chunk until the iteration cap, every tick."""
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    stuck_chunk = [_key_row("k1", budget_duration=None), _key_row("k2", budget_duration=None)]
    client, job = _chunked_job({"key": [stuck_chunk]})

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.fetches_by_table["key"] == 1


def test_key_reset_stops_when_the_fetch_fails(monkeypatch):
    """A phase whose query raises has made no progress; retrying it in a tight
    loop would just hammer a struggling database."""
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    client, job = _chunked_job({"key": [[_key_row("k1"), _key_row("k2")]]})

    async def _boom(table_name, query_type, **kwargs):
        client.get_data_calls.append({"table_name": table_name, "query_type": query_type, **kwargs})
        raise RuntimeError("db is down")

    client.get_data = _boom

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert len(client.get_data_calls) == 1


def test_key_reset_is_capped_at_max_chunks_per_run(monkeypatch):
    """Backstop against a phase that keeps making progress forever: the run ends
    and the leftovers wait for the next tick."""
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 1)
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN", 3)
    client, job = _chunked_job({"key": [[_key_row("k1")]]})

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.fetches_by_table["key"] == 3


@pytest.mark.parametrize(
    "phase, table_name, row_factory",
    [
        ("reset_budget_for_litellm_users", "user", lambda uid: _user_row(uid)),
        ("reset_budget_for_litellm_teams", "team", lambda tid: _team_row(tid)),
    ],
    ids=["users", "teams"],
)
def test_user_and_team_resets_are_chunked_too(monkeypatch, phase, table_name, row_factory):
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    client, job = _chunked_job({table_name: [[row_factory("a"), row_factory("b")], [row_factory("c")]]})

    asyncio.run(getattr(job, phase)())

    assert client.fetches_by_table[table_name] == 2
    assert _fetch_limits(client, table_name) == [2, 2]
    assert len(client.db.batchers) == 2
    assert len(_batch_writes(client, table_name, op="update")) == 3


def test_budget_table_reset_walks_chunks_until_it_runs_dry(monkeypatch):
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    client, job = _chunked_job({"budget": [[_budget_row("b1"), _budget_row("b2")], [_budget_row("b3")]]})

    asyncio.run(job.reset_budget_for_litellm_budget_table())

    assert client.fetches_by_table["budget"] == 2
    assert _fetch_limits(client, "budget") == [2, 2]
    assert len(client.db.batchers) == 2
    assert all(batcher.committed for batcher in client.db.batchers)
    assert [w["where"]["budget_id"] for w in _batch_writes(client, "budget", op="update_many")] == ["b1", "b2", "b3"]


def test_budget_table_reset_stops_when_a_full_chunk_advances_no_window(monkeypatch):
    """A tier with no budget_duration has its linked spend zeroed but keeps its
    past budget_reset_at, so it stays due. Counting those spend writes as
    progress would re-read the same chunk until the cap."""
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    stuck_chunk = [_budget_row("b1", budget_duration=None), _budget_row("b2", budget_duration=None)]
    client, job = _chunked_job({"budget": [stuck_chunk]})

    asyncio.run(job.reset_budget_for_litellm_budget_table())

    assert client.fetches_by_table["budget"] == 1
    assert _batch_writes(client, "budget", op="update_many") == []


def test_budget_table_reset_stops_when_the_cascade_fails(monkeypatch):
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    client, job = _chunked_job({"budget": [[_budget_row("b1"), _budget_row("b2")]]})
    client.db = FailingCommitDB()

    asyncio.run(job.reset_budget_for_litellm_budget_table())

    assert client.fetches_by_table["budget"] == 1


# ---------------------------------------------------------------------------
# Progress means "no longer due", not "was written"
# ---------------------------------------------------------------------------


def test_key_reset_stops_when_the_new_reset_time_is_not_in_the_future(monkeypatch):
    """A "0s" budget_duration resolves to the current time, so the row is written
    and comes straight back on the next fetch. Treating a written row as progress
    burns the whole per-run chunk cap on rows that never move.
    """
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    stuck_chunk = [_key_row("k1", budget_duration="0s"), _key_row("k2", budget_duration="0s")]
    client, job = _chunked_job({"key": [stuck_chunk]})

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.fetches_by_table["key"] == 1
    assert len(_batch_writes(client, "key", op="update")) == 2


def test_budget_table_reset_stops_when_the_new_window_is_not_in_the_future(monkeypatch):
    """Same zero-length window on the budget tier: advancing it to now leaves it
    due, so the cascade must not report progress."""
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    stuck_chunk = [_budget_row("b1", budget_duration="0s"), _budget_row("b2", budget_duration="0s")]
    client, job = _chunked_job({"budget": [stuck_chunk]})

    asyncio.run(job.reset_budget_for_litellm_budget_table())

    assert client.fetches_by_table["budget"] == 1
    assert len(_batch_writes(client, "budget", op="update_many")) == 2


class PoisonRow:
    """A row the in-memory reset cannot write, like the DataError rows in #27730."""

    token = "poison"
    budget_duration = "30d"
    budget_reset_at = None

    def __setattr__(self, name: str, value: Any) -> None:
        raise RuntimeError("simulated failure resetting this row")


class RecordingServiceLogging:
    def __init__(self):
        self.success_calls: List[Dict[str, Any]] = []
        self.failure_calls: List[Dict[str, Any]] = []

    async def async_service_success_hook(self, **kwargs):
        self.success_calls.append(kwargs)

    async def async_service_failure_hook(self, **kwargs):
        self.failure_calls.append(kwargs)


class RecordingProxyLogging:
    def __init__(self):
        self.service_logging_obj = RecordingServiceLogging()


def _run_and_drain_hooks(make_coro):
    """The service hooks are fired as tasks; give them a turn before asserting."""

    async def _run():
        await make_coro()
        await asyncio.sleep(0.05)

    asyncio.run(_run())


def test_key_reset_keeps_paging_when_some_rows_in_a_chunk_fail(monkeypatch):
    """One row that cannot be reset must not cost the phase its remaining chunks:
    the rows that did reset are committed and are real progress, and the failure
    is reported instead of aborting the run.
    """
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    client = ChunkedPrismaClient({"key": [[PoisonRow(), _key_row("k1")], [_key_row("k2")]]})
    logging_obj = RecordingProxyLogging()
    job = ResetBudgetJob(proxy_logging_obj=logging_obj, prisma_client=client)

    _run_and_drain_hooks(job.reset_budget_for_litellm_keys)

    assert client.fetches_by_table["key"] == 2
    assert [w["where"]["token"] for w in _batch_writes(client, "key", op="update")] == ["k1", "k2"]
    assert [call["call_type"] for call in logging_obj.service_logging_obj.failure_calls] == ["reset_budget_keys"]
    assert set(logging_obj.service_logging_obj.failure_calls[0]["event_metadata"]) == {"num_keys_found"}
    assert [call["call_type"] for call in logging_obj.service_logging_obj.success_calls] == ["reset_budget_keys"]


@pytest.mark.parametrize(
    "phase, table_name, row_factory, call_type",
    [
        ("reset_budget_for_litellm_users", "user", _user_row, "reset_budget_users"),
        ("reset_budget_for_litellm_teams", "team", _team_row, "reset_budget_teams"),
    ],
    ids=["users", "teams"],
)
def test_user_and_team_chunks_report_progress_despite_a_failed_row(
    monkeypatch, phase, table_name, row_factory, call_type
):
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    client = ChunkedPrismaClient({table_name: [[PoisonRow(), row_factory("a")], [row_factory("b")]]})
    logging_obj = RecordingProxyLogging()
    job = ResetBudgetJob(proxy_logging_obj=logging_obj, prisma_client=client)

    _run_and_drain_hooks(getattr(job, phase))

    assert client.fetches_by_table[table_name] == 2
    assert len(_batch_writes(client, table_name, op="update")) == 2
    assert [call["call_type"] for call in logging_obj.service_logging_obj.failure_calls] == [call_type]


class FakePodLockManager:
    """Stands in for the redis-backed PodLockManager.

    Lets a test pick which of the three states a pod lands in: it wins the
    lease, another pod already holds it, or redis cannot answer at all.
    """

    def __init__(self, *, acquired: bool, held_by_other: bool = False, has_redis: bool = True):
        self.redis_cache = MagicMock() if has_redis else None
        if self.redis_cache is not None:
            self.redis_cache.async_get_cache = AsyncMock(return_value="another-pod" if held_by_other else None)
        self._acquired = acquired
        self.acquire_calls: List[Dict[str, str | int | None]] = []
        self.release_calls: List[str] = []

    @staticmethod
    def get_redis_lock_key(cronjob_id: str) -> str:
        return f"cronjob_lock:{cronjob_id}"

    async def acquire_lock(self, cronjob_id: str, ttl: int | None = None) -> bool:
        self.acquire_calls.append({"cronjob_id": cronjob_id, "ttl": ttl})
        return self._acquired

    async def release_lock(self, cronjob_id: str) -> None:
        self.release_calls.append(cronjob_id)


def _make_leader_election_job(monkeypatch, pod_lock_manager):
    """A ResetBudgetJob wired to one lock manager, with every read observable.

    `prisma_client.get_data_calls` plus `prisma_client.db.query_raw` together
    cover every read the sweep makes, so a pod that skipped the tick leaves
    both untouched.
    """
    prisma_client = MockPrismaClient()
    prisma_client.db.query_raw = AsyncMock(return_value=[])

    spend_counter_cache = MagicMock()
    spend_counter_cache.redis_cache = None
    fake_module = types.ModuleType("litellm.proxy.proxy_server")
    fake_module.spend_counter_cache = spend_counter_cache
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_module)

    job = ResetBudgetJob(
        proxy_logging_obj=MockProxyLogging(),
        prisma_client=prisma_client,
        pod_lock_manager=pod_lock_manager,
    )
    return job, prisma_client


def _swept(prisma_client) -> bool:
    return bool(prisma_client.get_data_calls) or prisma_client.db.query_raw.await_count > 0


def test_reset_budget_sweeps_and_releases_when_it_wins_the_lease(monkeypatch):
    """The elected pod does the work and hands the lease back, so the next tick
    can elect any pod rather than waiting out the TTL."""
    lock = FakePodLockManager(acquired=True)
    job, prisma_client = _make_leader_election_job(monkeypatch, lock)

    asyncio.run(job.reset_budget())

    assert _swept(prisma_client)
    assert [call["cronjob_id"] for call in lock.acquire_calls] == [RESET_BUDGET_JOB_NAME]
    assert lock.release_calls == [RESET_BUDGET_JOB_NAME]


def test_reset_budget_does_nothing_when_another_pod_holds_the_lease(monkeypatch):
    """The whole point of the lease: a fleet must not multiply one sweep by its
    replica count. A pod that loses the election issues no query at all, and
    must not release a lease it never took."""
    lock = FakePodLockManager(acquired=False, held_by_other=True)
    job, prisma_client = _make_leader_election_job(monkeypatch, lock)

    asyncio.run(job.reset_budget())

    assert not _swept(prisma_client)
    assert lock.release_calls == []


def test_reset_budget_sweeps_unguarded_when_redis_cannot_answer(monkeypatch):
    """acquire_lock reports contention and an unreachable redis identically, so
    reading a failed acquire as contention would strand every expired budget at
    its cap on every pod for as long as redis is down. No holder means sweep."""
    lock = FakePodLockManager(acquired=False, held_by_other=False)
    job, prisma_client = _make_leader_election_job(monkeypatch, lock)

    asyncio.run(job.reset_budget())

    assert _swept(prisma_client)
    assert lock.release_calls == []


def test_reset_budget_sweeps_when_the_deployment_has_no_redis(monkeypatch):
    """A single-pod or redis-less deployment keeps its pre-election behavior."""
    lock = FakePodLockManager(acquired=False, has_redis=False)
    job, prisma_client = _make_leader_election_job(monkeypatch, lock)

    asyncio.run(job.reset_budget())

    assert _swept(prisma_client)
    assert lock.acquire_calls == []
    assert lock.release_calls == []


def test_reset_budget_sweeps_when_no_lock_manager_is_injected(monkeypatch):
    """Callers that construct the job without a lock manager still sweep."""
    job, prisma_client = _make_leader_election_job(monkeypatch, None)

    asyncio.run(job.reset_budget())

    assert _swept(prisma_client)


def test_reset_budget_releases_the_lease_when_a_phase_raises(monkeypatch):
    """A crash mid-sweep must not hold the lease for its whole TTL, which would
    stop every pod resetting budgets until it expired."""
    lock = FakePodLockManager(acquired=True)
    job, _ = _make_leader_election_job(monkeypatch, lock)

    async def boom() -> None:
        raise RuntimeError("phase exploded")

    monkeypatch.setattr(job, "reset_budget_for_litellm_keys", boom)

    with pytest.raises(RuntimeError):
        asyncio.run(job.reset_budget())

    assert lock.release_calls == [RESET_BUDGET_JOB_NAME]


def test_reset_budget_lease_outlives_one_scheduler_tick(monkeypatch):
    """A lease shorter than the gap between ticks expires mid-sweep and lets a
    second pod start sweeping, which is the amplification the lease removes."""
    lock = FakePodLockManager(acquired=True)
    job, _ = _make_leader_election_job(monkeypatch, lock)

    asyncio.run(job.reset_budget())

    assert lock.acquire_calls[0]["ttl"] == RESET_BUDGET_JOB_LOCK_TTL_SECONDS
    assert RESET_BUDGET_JOB_LOCK_TTL_SECONDS > PROXY_BUDGET_RESCHEDULER_MIN_TIME


def _window_row(source_id_column: str, row_id: str, reset_at: datetime) -> Dict[str, Any]:
    return {
        source_id_column: row_id,
        "budget_limits": [{"budget_duration": "1h", "reset_at": reset_at.isoformat(), "max_budget": 10}],
    }


def _paginating_window_job(monkeypatch, pages_by_table: Dict[str, List[List[Dict[str, Any]]]]):
    """Serve each table a canned sequence of pages and record every query.

    Returns (job, calls) where calls is a list of (sql, cursor, limit).
    """
    prisma_client = MagicMock()
    remaining = {table: list(pages) for table, pages in pages_by_table.items()}
    calls: List[Dict[str, Any]] = []

    async def fake_query_raw(query: str, *args, **kwargs):
        table = "key" if '"LiteLLM_VerificationToken"' in query else "team"
        calls.append({"table": table, "sql": query, "cursor": args[0], "limit": args[1]})
        pages = remaining[table]
        return pages.pop(0) if pages else []

    prisma_client.db.query_raw = AsyncMock(side_effect=fake_query_raw)
    prisma_client.db.litellm_verificationtoken.update = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=None)

    spend_counter_cache = MagicMock()
    spend_counter_cache.redis_cache = None
    fake_module = types.ModuleType("litellm.proxy.proxy_server")
    fake_module.spend_counter_cache = spend_counter_cache
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_module)

    job = ResetBudgetJob(proxy_logging_obj=MagicMock(), prisma_client=prisma_client)
    return job, calls


def test_reset_budget_windows_pages_by_cursor_instead_of_reading_the_table(monkeypatch):
    """The window scan used to read every row carrying budget_limits in one
    statement, so its memory and its statement cost grew with the deployment's
    key count. It now walks pages, and each page resumes past the last row.
    """
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    past = datetime.utcnow() - timedelta(hours=2)
    job, calls = _paginating_window_job(
        monkeypatch,
        {
            "key": [
                [_window_row("token", "k1", past), _window_row("token", "k2", past)],
                [_window_row("token", "k3", past)],
            ],
            "team": [[]],
        },
    )

    asyncio.run(job.reset_budget_windows())

    key_calls = [call for call in calls if call["table"] == "key"]
    assert [call["cursor"] for call in key_calls] == ["", "k2"], "second page must resume past the last row read"
    assert {call["limit"] for call in key_calls} == {2}
    assert all("LIMIT $2" in call["sql"] for call in key_calls)
    # the short second page ends the scan; a third query would re-read forever
    assert len(key_calls) == 2


def test_reset_budget_windows_pages_to_the_end_of_a_large_table(monkeypatch):
    """The scan must reach the last row within one tick.

    Capping the pages per run would need a resume position, and that position
    cannot live in the process: the lease is released after every sweep, so a
    later tick can elect a pod whose position is unset, restart at the first
    row, and leave the tail pinned at its cap forever. Paging alone bounds the
    memory, so the walk runs to completion instead.
    """
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 1)
    # the table needs far more pages than any per-run cap would allow, so a
    # capped walk stops short and only an uncapped one reaches the last row
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN", 3)
    past = datetime.utcnow() - timedelta(hours=2)
    rows = [_window_row("token", f"k{i:03d}", past) for i in range(1, 26)]
    job, visited = _cursor_paginating_window_job(monkeypatch, rows)

    asyncio.run(job.reset_budget_windows())

    assert visited == [f"k{i:03d}" for i in range(1, 26)], visited


def test_reset_budget_windows_survives_one_table_failing(monkeypatch):
    """A broken key scan must not cost the team scan its sweep."""
    prisma_client = MagicMock()

    async def fake_query_raw(query: str, *args, **kwargs):
        if '"LiteLLM_VerificationToken"' in query:
            raise RuntimeError("key scan exploded")
        return []

    prisma_client.db.query_raw = AsyncMock(side_effect=fake_query_raw)
    spend_counter_cache = MagicMock()
    spend_counter_cache.redis_cache = None
    fake_module = types.ModuleType("litellm.proxy.proxy_server")
    fake_module.spend_counter_cache = spend_counter_cache
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_module)
    job = ResetBudgetJob(proxy_logging_obj=MagicMock(), prisma_client=prisma_client)

    asyncio.run(job.reset_budget_windows())

    queried = [call.args[0] for call in prisma_client.db.query_raw.await_args_list]
    assert any('"LiteLLM_TeamTable"' in sql for sql in queried)


def test_row_payloads_stay_out_of_reset_job_event_metadata(monkeypatch):
    """Every found and updated row used to be JSON-serialized into the service
    hook's metadata on every chunk, on the event loop, whether or not any
    consumer read it. Only the counts are reported now."""
    client = ChunkedPrismaClient({"key": [[_key_row("k1"), _key_row("k2")]]})
    logging_obj = RecordingProxyLogging()
    job = ResetBudgetJob(proxy_logging_obj=logging_obj, prisma_client=client)

    _run_and_drain_hooks(job.reset_budget_for_litellm_keys)

    metadata = logging_obj.service_logging_obj.success_calls[0]["event_metadata"]
    assert metadata["num_keys_found"] == 2
    assert metadata["num_keys_updated"] == 2
    assert {"keys_found", "keys_updated", "keys_failed"}.isdisjoint(metadata)
    assert all(isinstance(value, int) for value in metadata.values()), metadata


def test_debug_row_dump_is_deferred_until_a_record_is_emitted():
    """`logger.debug("%s", json.dumps(rows))` serializes before the logger drops
    the record, so the sweep paid for a full dump of every chunk at any log
    level. The wrapper defers the work to the formatter."""
    serialized = []

    class Tracked:
        def __repr__(self) -> str:
            serialized.append("serialized")
            return "tracked"

    lazy = reset_budget_job_module._LazyJson([Tracked()])
    assert serialized == [], "constructing the wrapper must not serialize"

    assert "tracked" in str(lazy)
    assert serialized == ["serialized"]


def _cursor_paginating_window_job(monkeypatch, key_rows: List[Dict[str, Any]]):
    """Serve real keyset pages out of one ordered table, honouring the cursor.

    Unlike the canned-page helper above, this models the database: a page is
    whatever rows sort after the cursor, so a scan that forgets its cursor
    genuinely re-reads the same prefix.
    """
    prisma_client = MagicMock()
    ordered = sorted(key_rows, key=lambda r: r["token"])
    visited: List[str] = []

    async def fake_query_raw(query: str, *args, **kwargs):
        if '"LiteLLM_TeamTable"' in query:
            return []
        cursor, limit = args[0], args[1]
        page = [row for row in ordered if row["token"] > cursor][:limit]
        visited.extend(row["token"] for row in page)
        return page

    prisma_client.db.query_raw = AsyncMock(side_effect=fake_query_raw)
    prisma_client.db.litellm_verificationtoken.update = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=None)

    spend_counter_cache = MagicMock()
    spend_counter_cache.redis_cache = None
    fake_module = types.ModuleType("litellm.proxy.proxy_server")
    fake_module.spend_counter_cache = spend_counter_cache
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_module)

    job = ResetBudgetJob(proxy_logging_obj=MagicMock(), prisma_client=prisma_client)
    return job, visited


def test_every_tick_sweeps_the_whole_window_table_whichever_pod_won(monkeypatch):
    """Coverage must not depend on which pod was elected.

    The lease is released after each sweep, so consecutive ticks routinely run
    on different pods. A scan carrying a resume position in process memory would
    have a fresh pod start over at the first row, so rows past one run's reach
    would never be swept by anyone. Two independent job instances, standing in
    for two pods, must each cover the table end to end.
    """
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_BATCH_SIZE", 2)
    monkeypatch.setattr(reset_budget_job_module, "RESET_BUDGET_JOB_MAX_CHUNKS_PER_RUN", 2)
    past = datetime.utcnow() - timedelta(hours=2)
    rows = [_window_row("token", f"k{i:03d}", past) for i in range(1, 12)]
    expected = [f"k{i:03d}" for i in range(1, 12)]

    pod_a, visited_a = _cursor_paginating_window_job(monkeypatch, rows)
    pod_b, visited_b = _cursor_paginating_window_job(monkeypatch, rows)

    asyncio.run(pod_a.reset_budget_windows())
    asyncio.run(pod_b.reset_budget_windows())

    assert visited_a == expected, visited_a
    assert visited_b == expected, visited_b


class FlakyPrismaClient(MockPrismaClient):
    """A client whose first N reads (or first N batch commits) fail with a
    transport error, and which records every reconnect attempt.
    """

    def __init__(self, *, read_failures: int = 0, commit_failures: int = 0, error: Exception | None = None):
        super().__init__()
        self.reconnect_reasons: List[str] = []
        self.read_attempts: int = 0
        self.commit_attempts: int = 0
        self._read_failures = read_failures
        self._commit_failures = commit_failures
        self._error = error or httpx.ConnectError("All connection attempts failed")

        outer = self
        original_batch = self.db.batch_

        def _batch_():
            batcher = original_batch()
            batch_commit = batcher.commit

            async def _maybe_failing_commit():
                outer.commit_attempts += 1
                if outer._commit_failures > 0:
                    outer._commit_failures -= 1
                    raise outer._error
                return await batch_commit()

            batcher.commit = _maybe_failing_commit
            return batcher

        self.db.batch_ = _batch_

    async def attempt_db_reconnect(self, *, reason, timeout_seconds=None, lock_timeout_seconds=None) -> bool:
        self.reconnect_reasons.append(reason)
        return True

    async def get_data(self, table_name, query_type, **kwargs):
        self.read_attempts += 1
        if self._read_failures > 0:
            self._read_failures -= 1
            raise self._error
        return await super().get_data(table_name, query_type, **kwargs)


def _due_row(table: str, identifier: str):
    now = datetime.now(timezone.utc)
    id_field = {"key": "token", "user": "user_id", "team": "team_id"}[table]
    return type(
        "Row",
        (),
        {
            "spend": _DUE_ROW_SPEND,
            "budget_duration": "30d",
            "budget_reset_at": now - timedelta(seconds=1),
            id_field: identifier,
        },
    )


@pytest.mark.parametrize(
    "phase, table_name, reason",
    [
        ("reset_budget_for_litellm_keys", "key", "reset_budget_read_keys_failure"),
        ("reset_budget_for_litellm_users", "user", "reset_budget_read_users_failure"),
        ("reset_budget_for_litellm_teams", "team", "reset_budget_read_teams_failure"),
    ],
    ids=["keys", "users", "teams"],
)
def test_transient_transport_error_on_read_reconnects_and_still_resets(phase, table_name, reason):
    """A dropped connection on the read must cost one reconnect-and-retry, not
    the whole tick (LIT-5372). Pre-fix the httpx.ConnectError was swallowed and
    the phase reset nothing until the next tick, 10 minutes later.
    """
    client = FlakyPrismaClient(read_failures=1)
    client.data[table_name] = [_due_row(table_name, "row-1")]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(getattr(job, phase)())

    assert client.reconnect_reasons == [reason]
    assert len(_batch_writes(client, table_name, op="update")) == 1


@pytest.mark.parametrize(
    "phase, table_name, reason",
    [
        ("reset_budget_for_litellm_keys", "key", "reset_budget_write_keys_failure"),
        ("reset_budget_for_litellm_users", "user", "reset_budget_write_users_failure"),
        ("reset_budget_for_litellm_teams", "team", "reset_budget_write_teams_failure"),
    ],
    ids=["keys", "users", "teams"],
)
def test_connect_error_on_write_reconnects_and_commits(phase, table_name, reason):
    """A ConnectError proves the commit never reached the database, so replaying
    it cannot double-apply anything: the rows still get reset on this tick.
    """
    client = FlakyPrismaClient(commit_failures=1)
    client.data[table_name] = [_due_row(table_name, "row-1")]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(getattr(job, phase)())

    assert client.reconnect_reasons == [reason]
    assert client.commit_attempts == 2
    assert len(_batch_writes(client, table_name, op="update")) == 1


@pytest.mark.parametrize("ambiguous_error_name", ["ReadError", "ReadTimeout"])
def test_ambiguous_transport_error_on_write_is_not_replayed(ambiguous_error_name):
    """A post-send transport error leaves the commit outcome unknown. Since the
    reset zeroes spend unconditionally, replaying it would erase spend accrued
    after a commit that actually landed, so only reads may retry these.
    """
    client = FlakyPrismaClient(
        commit_failures=1,
        error=getattr(httpx, ambiguous_error_name)("ambiguous"),
    )
    client.data["key"] = [_due_row("key", "tok-1")]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.reconnect_reasons == []
    assert client.commit_attempts == 1


@pytest.mark.parametrize("ambiguous_error_name", ["ReadError", "ReadTimeout"])
def test_ambiguous_transport_error_on_read_still_retries(ambiguous_error_name):
    """Reads have nothing to double-apply, so the full transport class retries."""
    client = FlakyPrismaClient(read_failures=1, error=getattr(httpx, ambiguous_error_name)("ambiguous"))
    client.data["key"] = [_due_row("key", "tok-1")]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.reconnect_reasons == ["reset_budget_read_keys_failure"]
    assert len(_batch_writes(client, "key", op="update")) == 1


def test_transport_error_on_budget_cascade_read_reconnects_and_commits():
    client = FlakyPrismaClient(read_failures=1)
    budget = _budget_row(budget_id="b-1", budget_duration="1d")
    client.data["budget"] = [budget]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(job.reset_budget_for_litellm_budget_table())

    assert client.reconnect_reasons == ["reset_budget_read_budgets_failure"]
    assert [w["where"]["budget_id"] for w in _batch_writes(client, "budget", op="update_many")] == ["b-1"]


def test_non_transport_error_still_surfaces_without_a_reconnect():
    """A UniqueViolationError means the DB is reachable and the statement was
    refused, so reconnecting would be pointless: the phase must fail as before.
    """
    client = FlakyPrismaClient(read_failures=1, error=prisma.errors.UniqueViolationError(MagicMock()))
    client.data["key"] = [_due_row("key", "tok-1")]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.reconnect_reasons == []
    assert client.read_attempts == 1
    assert _batch_writes(client, "key") == []


def test_transport_error_that_outlives_the_reconnect_is_not_retried_forever():
    client = FlakyPrismaClient(read_failures=2)
    client.data["key"] = [_due_row("key", "tok-1")]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.reconnect_reasons == ["reset_budget_read_keys_failure"]
    assert client.read_attempts == 2
    assert _batch_writes(client, "key") == []


def test_transport_error_on_window_read_reconnects_and_still_resets(monkeypatch):
    """The raw per-window queries are reads too, so a blip there must not cost
    the whole window-reset phase."""
    expired = (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "Z"
    key_rows = [{"token": "sk-expired", "budget_limits": [{"budget_duration": "1d", "reset_at": expired}]}]
    job, prisma_client, _ = _make_reset_budget_windows_job(monkeypatch, key_rows=key_rows, team_rows=[])
    reconnect_reasons: List[str] = []
    good_query_raw = prisma_client.db.query_raw

    async def failing_once_query_raw(query: str, *args, **kwargs):
        if '"LiteLLM_VerificationToken"' in query and not reconnect_reasons:
            raise httpx.ConnectError("All connection attempts failed")
        return await good_query_raw(query, *args, **kwargs)

    async def record_reconnect(*, reason, timeout_seconds=None, lock_timeout_seconds=None) -> bool:
        reconnect_reasons.append(reason)
        return True

    prisma_client.db.query_raw = AsyncMock(side_effect=failing_once_query_raw)
    prisma_client.attempt_db_reconnect = record_reconnect

    asyncio.run(job.reset_budget_windows())

    assert reconnect_reasons == ["reset_budget_read_key_windows_failure"]
    prisma_client.db.litellm_verificationtoken.update.assert_awaited_once()


def test_connect_error_on_window_write_reconnects_and_writes(monkeypatch):
    expired = (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "Z"
    team_rows = [{"team_id": "team-expired", "budget_limits": [{"budget_duration": "1d", "reset_at": expired}]}]
    job, prisma_client, _ = _make_reset_budget_windows_job(monkeypatch, key_rows=[], team_rows=team_rows)
    reconnect_reasons: List[str] = []

    async def failing_once_update(**kwargs) -> None:
        if not reconnect_reasons:
            raise httpx.ConnectError("All connection attempts failed")

    async def record_reconnect(*, reason, timeout_seconds=None, lock_timeout_seconds=None) -> bool:
        reconnect_reasons.append(reason)
        return True

    prisma_client.db.litellm_teamtable.update = AsyncMock(side_effect=failing_once_update)
    prisma_client.attempt_db_reconnect = record_reconnect

    asyncio.run(job.reset_budget_windows())

    assert reconnect_reasons == ["reset_budget_write_team_windows_failure"]
    assert prisma_client.db.litellm_teamtable.update.await_count == 2


_DUE_ROW_SPEND = 42.0
_SPEND_ACCRUED_AFTER_COMMIT = 7.5


class AmbiguousCommitClient(MockPrismaClient):
    """A client whose batch commit lands in the database and only then fails in
    transit, so the caller cannot tell whether it committed.

    The queued spend-zero is applied to `key_spend`, and fresh usage accrues in
    the window between that landed commit and any replay, so a replay is
    observable as erased spend rather than merely as an extra commit.
    """

    def __init__(self, *, error: Exception, spend_accrued_after_commit: float):
        super().__init__()
        self.key_spend: float = _DUE_ROW_SPEND
        self.commit_attempts: int = 0
        self.reconnect_reasons: list[str] = []

        outer = self
        original_batch = self.db.batch_

        def _batch_():
            batcher = original_batch()
            batch_commit = batcher.commit

            async def _commit_then_lose_the_response():
                outer.commit_attempts += 1
                result = await batch_commit()
                for call in batcher.calls:
                    if call["table"] == "key" and call["data"].get("spend") == 0:
                        outer.key_spend = 0.0
                if outer.commit_attempts > 1:
                    return result
                outer.key_spend += spend_accrued_after_commit
                raise error

            batcher.commit = _commit_then_lose_the_response
            return batcher

        self.db.batch_ = _batch_

    async def attempt_db_reconnect(self, *, reason, timeout_seconds=None, lock_timeout_seconds=None) -> bool:
        self.reconnect_reasons.append(reason)
        return True


@pytest.mark.parametrize(
    "error, expected_commits, expected_spend, expected_reconnects",
    [
        (httpx.ReadError("response lost in transit"), 1, _SPEND_ACCRUED_AFTER_COMMIT, []),
        (httpx.ReadTimeout("response lost in transit"), 1, _SPEND_ACCRUED_AFTER_COMMIT, []),
        (httpx.ConnectError("never left the client"), 2, 0.0, ["reset_budget_write_keys_failure"]),
    ],
    ids=["read_error", "read_timeout", "connect_error_erasure_control"],
)
def test_ambiguous_commit_replay_does_not_erase_newly_accrued_spend(
    error, expected_commits, expected_spend, expected_reconnects
):
    """A reset zeroes spend unconditionally, so replaying a commit that already
    landed erases every dollar spent since it landed (LIT-5372 review finding).

    The `connect_error` case is the control: it is the one error class allowed
    to replay, and driving it through this same land-then-fail harness proves
    the spend assertion can actually observe an erasure. In production a
    ConnectError means the statements never reached the database, so its replay
    has nothing to erase.
    """
    client = AmbiguousCommitClient(error=error, spend_accrued_after_commit=_SPEND_ACCRUED_AFTER_COMMIT)
    client.data["key"] = [_due_row("key", "tok-1")]
    job = ResetBudgetJob(proxy_logging_obj=MockProxyLogging(), prisma_client=client)

    asyncio.run(job.reset_budget_for_litellm_keys())

    assert client.key_spend == expected_spend
    assert client.commit_attempts == expected_commits
    assert client.reconnect_reasons == expected_reconnects


# ---------------------------------------------------------------------------
# Budget rollover (LIT-3085): overage beyond max_budget carries into the next
# window instead of being forgiven
# ---------------------------------------------------------------------------


@pytest.fixture
def rollover_enabled(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "budget_rollover", True)


@pytest.mark.parametrize(
    "run_phase, table, id_field, id_value, row_factory",
    [
        (
            lambda job: job.reset_budget_for_litellm_keys(),
            "key",
            "token",
            "tok-roll",
            lambda now: type(
                "Key",
                (),
                {
                    "spend": 150.0,
                    "max_budget": 100.0,
                    "budget_duration": "1d",
                    "budget_reset_at": now,
                    "token": "tok-roll",
                },
            ),
        ),
        (
            lambda job: job.reset_budget_for_litellm_users(),
            "user",
            "user_id",
            "user-roll",
            lambda now: type(
                "User",
                (),
                {
                    "spend": 150.0,
                    "max_budget": 100.0,
                    "budget_duration": "30d",
                    "budget_reset_at": now,
                    "user_id": "user-roll",
                },
            ),
        ),
        (
            lambda job: job.reset_budget_for_litellm_teams(),
            "team",
            "team_id",
            "team-roll",
            lambda now: type(
                "Team",
                (),
                {
                    "spend": 150.0,
                    "max_budget": 100.0,
                    "budget_duration": "1mo",
                    "budget_reset_at": now,
                    "team_id": "team-roll",
                },
            ),
        ),
    ],
)
def test_direct_reset_carries_overage_when_rollover_enabled(
    rollover_enabled, reset_budget_job, mock_prisma_client, monkeypatch, run_phase, table, id_field, id_value, row_factory
):
    """spend=150 against max_budget=100 must decrement by the cap (leaving 50)
    rather than zero the row, and the spend counter must be seeded with 50."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    now = datetime.now(timezone.utc)
    mock_prisma_client.data[table] = [row_factory(now)]

    asyncio.run(run_phase(reset_budget_job))

    writes = _batch_writes(mock_prisma_client, table)
    assert len(writes) == 1
    assert writes[0]["where"] == {id_field: id_value}
    assert writes[0]["data"]["spend"] == {"decrement": 100.0}
    assert writes[0]["data"]["budget_reset_at"] > now
    counter_prefix = {"key": "spend:key", "user": "spend:user", "team": "spend:team"}[table]
    counter_cache.in_memory_cache.set_cache.assert_any_call(key=f"{counter_prefix}:{id_value}", value=50.0, ttl=60)


def test_direct_reset_zeroes_under_budget_row_even_with_rollover(
    rollover_enabled, reset_budget_job, mock_prisma_client, monkeypatch
):
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    now = datetime.now(timezone.utc)
    mock_prisma_client.data["key"] = [
        type(
            "Key",
            (),
            {"spend": 40.0, "max_budget": 100.0, "budget_duration": "1d", "budget_reset_at": now, "token": "tok-under"},
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_keys())

    assert _batch_writes(mock_prisma_client, "key")[0]["data"]["spend"] == 0
    counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:key:tok-under", value=0.0, ttl=60)


def test_direct_reset_zeroes_row_without_max_budget_even_with_rollover(
    rollover_enabled, reset_budget_job, mock_prisma_client, monkeypatch
):
    """No cap means nothing to carry against: reset to zero as before."""
    _make_counter_invalidation_job(monkeypatch)
    now = datetime.now(timezone.utc)
    mock_prisma_client.data["key"] = [
        type(
            "Key",
            (),
            {"spend": 150.0, "max_budget": None, "budget_duration": "1d", "budget_reset_at": now, "token": "tok-nocap"},
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_keys())

    assert _batch_writes(mock_prisma_client, "key")[0]["data"]["spend"] == 0


def test_budget_cascade_carries_overage_per_tier_when_rollover_enabled(
    rollover_enabled, reset_budget_job, mock_prisma_client, monkeypatch
):
    """A team member 5 over the tier cap keeps a spend of 5 in the next window:
    the cascade decrements over-cap rows by the cap, zeroes the rest, and seeds
    the spend counter with the carried amount."""
    counter_cache = _make_counter_invalidation_job(monkeypatch)
    budget = _budget_row(budget_id="budget-roll", budget_duration="7d", max_budget=10.0)
    mock_prisma_client.data["budget"] = [budget]
    membership = type(
        "Membership",
        (),
        {"user_id": "member-1", "team_id": "team-1", "spend": 15.0, "budget_id": "budget-roll"},
    )
    mock_prisma_client.db.litellm_teammembership.set_find_many_results([membership])

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    membership_writes = _batch_writes(mock_prisma_client, "team_membership")
    assert {
        "table": "team_membership",
        "op": "update_many",
        "where": {"budget_id": "budget-roll", "spend": {"gt": 10.0}},
        "data": {"spend": {"decrement": 10.0}},
    } in membership_writes
    assert {
        "table": "team_membership",
        "op": "update_many",
        "where": {"budget_id": "budget-roll", "spend": {"gt": 0, "lte": 10.0}},
        "data": {"spend": 0},
    } in membership_writes
    counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:team_member:member-1:team-1", value=5.0, ttl=60)


def test_budget_cascade_carries_enduser_overage_when_rollover_enabled(
    rollover_enabled, reset_budget_job, mock_prisma_client, monkeypatch
):
    _make_counter_invalidation_job(monkeypatch)
    budget = _budget_row(budget_id="budget-roll", budget_duration="1d", max_budget=10.0)
    mock_prisma_client.data["budget"] = [budget]
    mock_prisma_client.data["enduser"] = [
        type(
            "EndUser",
            (),
            {"spend": 15.0, "litellm_budget_table": budget, "user_id": "enduser-roll", "budget_id": "budget-roll"},
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    enduser_writes = _batch_writes(mock_prisma_client, "enduser")
    assert {
        "table": "enduser",
        "op": "update_many",
        "where": {"user_id": {"in": ["enduser-roll"]}, "spend": {"gt": 10.0}},
        "data": {"spend": {"decrement": 10.0}},
    } in enduser_writes
    assert {
        "table": "enduser",
        "op": "update_many",
        "where": {"user_id": {"in": ["enduser-roll"]}, "spend": {"lte": 10.0}},
        "data": {"spend": 0},
    } in enduser_writes


def _replay_spend_writes(writes, spend):
    """Apply the queued update_many statements in order, the way the DB
    transaction executes them, and return the row's final spend."""
    for write in writes:
        condition = write["where"].get("spend")
        if isinstance(condition, dict):
            if "gt" in condition and not spend > condition["gt"]:
                continue
            if "lte" in condition and not spend <= condition["lte"]:
                continue
        payload = write["data"]["spend"]
        spend = payload if not isinstance(payload, dict) else spend - payload["decrement"]
    return spend


@pytest.mark.parametrize("table", ["team_membership", "enduser"])
def test_cascade_rollover_writes_survive_sequential_execution(
    rollover_enabled, reset_budget_job, mock_prisma_client, monkeypatch, table
):
    """The statements run one after another inside a transaction, so a
    decrement-then-zero order would re-match the decremented row (now in the
    0..cap range) and erase the carried spend. Replaying the writes in queue
    order must leave the overage, for any spend between cap and twice the cap."""
    _make_counter_invalidation_job(monkeypatch)
    budget = _budget_row(budget_id="budget-roll", budget_duration="7d", max_budget=10.0)
    mock_prisma_client.data["budget"] = [budget]
    membership = type(
        "Membership",
        (),
        {"user_id": "member-1", "team_id": "team-1", "spend": 15.0, "budget_id": "budget-roll"},
    )
    mock_prisma_client.db.litellm_teammembership.set_find_many_results([membership])
    mock_prisma_client.data["enduser"] = [
        type(
            "EndUser",
            (),
            {"spend": 15.0, "litellm_budget_table": budget, "user_id": "enduser-roll", "budget_id": "budget-roll"},
        )
    ]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    writes = _batch_writes(mock_prisma_client, table)
    assert _replay_spend_writes(writes, 15.0) == 5.0
    assert _replay_spend_writes(writes, 8.0) == 0
    assert _replay_spend_writes(writes, 25.0) == 15.0


def test_budget_cascade_zeroes_everything_when_rollover_disabled(reset_budget_job, mock_prisma_client, monkeypatch):
    """Control: with the flag off the cascade keeps the plain zeroing writes."""
    _make_counter_invalidation_job(monkeypatch)
    budget = _budget_row(budget_id="budget-off", budget_duration="7d", max_budget=10.0)
    mock_prisma_client.data["budget"] = [budget]

    asyncio.run(reset_budget_job.reset_budget_for_litellm_budget_table())

    membership_writes = _batch_writes(mock_prisma_client, "team_membership")
    assert membership_writes == [
        {
            "table": "team_membership",
            "op": "update_many",
            "where": {"budget_id": {"in": ["budget-off"]}},
            "data": {"spend": 0},
        }
    ]


def test_window_reset_carries_counter_overage_when_rollover_enabled(rollover_enabled, monkeypatch):
    """A per-window counter at 130 against a 100 cap restarts the window at 30."""
    now = datetime.utcnow()
    expired = (now - timedelta(minutes=5)).isoformat() + "Z"
    key_rows = [
        {
            "token": "sk-roll",
            "budget_limits": [{"budget_duration": "1d", "reset_at": expired, "max_budget": 100.0}],
        }
    ]
    job, prisma_client, spend_counter_cache = _make_reset_budget_windows_job(
        monkeypatch, key_rows=key_rows, team_rows=[]
    )
    spend_counter_cache.async_get_cache = AsyncMock(return_value=130.0)

    asyncio.run(job.reset_budget_windows())

    prisma_client.db.litellm_verificationtoken.update.assert_awaited_once()
    spend_counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:key:sk-roll:window:1d", value=30.0)


def test_window_reset_zeroes_counter_when_rollover_disabled(monkeypatch):
    now = datetime.utcnow()
    expired = (now - timedelta(minutes=5)).isoformat() + "Z"
    key_rows = [
        {
            "token": "sk-off",
            "budget_limits": [{"budget_duration": "1d", "reset_at": expired, "max_budget": 100.0}],
        }
    ]
    job, prisma_client, spend_counter_cache = _make_reset_budget_windows_job(
        monkeypatch, key_rows=key_rows, team_rows=[]
    )
    spend_counter_cache.async_get_cache = AsyncMock(return_value=130.0)

    asyncio.run(job.reset_budget_windows())

    spend_counter_cache.in_memory_cache.set_cache.assert_any_call(key="spend:key:sk-off:window:1d", value=0.0)
    spend_counter_cache.async_get_cache.assert_not_awaited()
