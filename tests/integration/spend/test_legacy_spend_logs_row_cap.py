import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway
from integration._support.database import read_rows

LEGACY_SPEND_LOGS_ROW_CAP: Final = 10000


def _seed_spend_rows(user_id: str, count: int) -> tuple[str, ...]:
    started: Final = datetime.now(timezone.utc) - timedelta(days=1)
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" '
            '(request_id, call_type, "startTime", "endTime", "user", status) '
            "SELECT %s || '-' || n, 'acompletion', %s + n * interval '1 second', %s + n * interval '1 second', %s, "
            "'success' FROM generate_series(1, %s) AS n",
            (user_id, started, started, user_id, count),
        )
    return tuple(f"{user_id}-{n}" for n in range(1, count + 1))


def _delete_spend_rows(user_id: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE "user" = %s', (user_id,))


@pytest.mark.covers("spend.legacy_spend_logs.row_count_is_capped_at_the_most_recent_rows_and_flagged_truncated")
def test_legacy_spend_logs_returns_only_the_cap_of_most_recent_rows_and_flags_truncation(gateway: Gateway) -> None:
    user_id: Final = f"integration-cap-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(_delete_spend_rows, user_id)
        seeded: Final = _seed_spend_rows(user_id, LEGACY_SPEND_LOGS_ROW_CAP + 1)
        assert read_rows('SELECT count(*)::text AS total FROM "LiteLLM_SpendLogs" WHERE "user" = %s', (user_id,)) == [
            {"total": str(LEGACY_SPEND_LOGS_ROW_CAP + 1)}
        ]
        response: Final = gateway.request("GET", "/spend/logs", params={"user_id": user_id})
        assert response.status_code == 200, response.text
        returned: Final = tuple(row["request_id"] for row in response.json())
        assert len(returned) == LEGACY_SPEND_LOGS_ROW_CAP, f"{len(returned)} rows: {response.text[:300]}"
        assert returned == tuple(reversed(seeded[1:])), response.text[:300]
        assert response.headers.get("x-litellm-spend-logs-truncated") == "true", dict(response.headers)
