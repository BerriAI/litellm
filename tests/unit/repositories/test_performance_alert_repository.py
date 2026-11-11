from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.spend_tracking.performance_alerts import run_performance_alerts
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository, is_statement_timeout


def _raw_query_error(message: str) -> Exception:
    from prisma.errors import RawQueryError

    return RawQueryError({"user_facing_error": {"is_panic": False, "message": message, "meta": {}}})


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("Raw query failed. Code: `57014`. Message: `ERROR: canceling statement due to statement timeout`", True),
        ("Error querying the database: FATAL: sorry, too many clients already", False),
        (
            "Raw query failed. Code: `57P01`. Message: `FATAL: terminating connection due to administrator command`",
            False,
        ),
        ("Raw query failed. Code: `40P01`. Message: `ERROR: deadlock detected`", False),
    ),
)
def test_only_a_statement_timeout_marks_a_resource_unreadable(message: str, expected: bool) -> None:
    assert is_statement_timeout(_raw_query_error(message)) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("enabled", "used"), ((True, 5), (False, 0)))
async def test_stopped_monitoring_never_reads_usage_or_claims_work(enabled: bool, used: int) -> None:
    client: Final = MagicMock()
    transaction: Final = client.writer_db.tx.return_value.__aenter__.return_value
    transaction.query_raw = AsyncMock(side_effect=[[], [{"enabled": enabled, "alerts_used": used}]])
    transaction.execute_raw = AsyncMock()
    await run_performance_alerts(
        PerformanceAlertRepository(client), now=lambda: datetime(2026, 1, 9, tzinfo=timezone.utc)
    )
    queries: Final = tuple(call.args[0] for call in transaction.query_raw.await_args_list)
    assert len(queries) == 2
    assert 'FROM "LiteLLM_PerformanceAlertState" WHERE id = 1' in queries[-1]
    transaction.execute_raw.assert_not_awaited()
