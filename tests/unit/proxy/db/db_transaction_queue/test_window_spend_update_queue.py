from datetime import datetime, timedelta, timezone


from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    build_window_spend_transaction,
    to_naive_utc,
)

WINDOW_A = datetime(2026, 8, 1, tzinfo=timezone.utc)
WINDOW_B = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _txn(
    entity_id: str,
    window_start: datetime,
    spend: float,
    duration: str = "30d",
    entity_type: str = "key",
    started_at: datetime | None = None,
):
    return build_window_spend_transaction(
        entity_type=entity_type,
        entity_id=entity_id,
        window_duration=duration,
        window_start=window_start,
        spend=spend,
        started_at=started_at,
    )


def test_build_window_spend_transaction_stores_naive_utc_iso():
    """window_start rides the Redis buffer as a string and lands in a naive-UTC
    TIMESTAMP(3) column, so a non-UTC input must be converted, not truncated."""
    non_utc = datetime(2026, 8, 1, 20, 0, tzinfo=timezone(timedelta(hours=-4)))

    assert _txn("k1", non_utc, 1.0) == {
        "entity_type": "key",
        "entity_id": "k1",
        "window_duration": "30d",
        "window_start": "2026-08-02T00:00:00.000000",
        "spend": 1.0,
        "started_at": None,
    }


def test_build_window_spend_transaction_stores_started_at_as_naive_utc_iso():
    """started_at is compared against LiteLLM_SpendLogs.startTime, which the
    spend log writer stores after converting the request start to UTC."""
    non_utc = datetime(2026, 8, 10, 8, 30, 15, 123456, tzinfo=timezone(timedelta(hours=-4)))

    assert _txn("k1", WINDOW_A, 1.0, started_at=non_utc)["started_at"] == "2026-08-10T12:30:15.123456"




def test_to_naive_utc_leaves_naive_values_alone():
    naive = datetime(2026, 8, 1, 12, 0)
    assert to_naive_utc(naive) == naive
