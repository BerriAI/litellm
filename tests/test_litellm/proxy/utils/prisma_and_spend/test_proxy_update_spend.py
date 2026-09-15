"""Pin ``ProxyUpdateSpend`` behavior.

Symbols pinned here:
  - ``ProxyUpdateSpend.update_end_user_spend``
  - ``ProxyUpdateSpend.update_spend_logs``
  - ``ProxyUpdateSpend.disable_spend_updates``
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Dict, List, TypeAlias
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.utils as utils_mod
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.db.spend_log_batching import spend_log_row_bytes
from litellm.proxy.utils import PrismaClient, ProxyUpdateSpend, enqueue_spend_logs


@pytest.fixture(autouse=True)
def reset_spend_log_queue_bytes() -> Iterator[None]:
    PrismaClient.spend_log_queue_bytes = 0
    yield
    PrismaClient.spend_log_queue_bytes = 0


class _AsyncCM:
    def __init__(self, target: Any) -> None:
        self.target = target

    async def __aenter__(self) -> Any:
        return self.target

    async def __aexit__(self, *exc: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_update_end_user_spend_upserts_each_end_user(
    mock_prisma_client: Any,
) -> None:
    batcher = MagicMock()
    batcher.litellm_endusertable.upsert = MagicMock()
    transaction = MagicMock()
    transaction.batch_ = lambda: _AsyncCM(batcher)
    mock_prisma_client.db.tx = lambda timeout: _AsyncCM(transaction)

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    end_user_costs: Dict[str, float] = {"u_b": 1.0, "u_a": 0.5}
    await ProxyUpdateSpend.update_end_user_spend(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        proxy_logging_obj=proxy_logging,
        end_user_list_transactions=end_user_costs,
    )
    calls = batcher.litellm_endusertable.upsert.call_args_list
    ordered_ids = [c.kwargs["where"]["user_id"] for c in calls]
    creates = [c.kwargs["data"]["create"] for c in calls]
    pinned = {
        "upsert_count": len(calls),
        "ordered_ids": ordered_ids,
        "first_create_keys": sorted(creates[0].keys()),
        "first_create_user_id": creates[0]["user_id"],
        "first_create_spend": creates[0]["spend"],
    }
    assert pinned == {
        "upsert_count": 2,
        "ordered_ids": ["u_a", "u_b"],
        "first_create_keys": sorted(["user_id", "spend", "blocked"]),
        "first_create_user_id": "u_a",
        "first_create_spend": 0.5,
    }


@pytest.mark.asyncio
async def test_update_end_user_spend_retries_on_connect_error(
    mock_prisma_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DB_RETRY_SAFE_ERROR_TYPES`` (ConnectError, statements provably never
    sent) retries with jittered backoff; once retries are exhausted the
    original exception bubbles up via ``_raise_failed_update_spend_exception``.
    """
    import httpx
    import litellm.proxy.utils as utils_mod

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)

    err = httpx.ConnectError("down")
    mock_prisma_client.db.tx = MagicMock(side_effect=err)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    with pytest.raises(httpx.ConnectError):
        await ProxyUpdateSpend.update_end_user_spend(
            n_retry_times=1,
            prisma_client=mock_prisma_client,
            proxy_logging_obj=proxy_logging,
            end_user_list_transactions={"u": 1.0},
        )
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] <= 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous_error_name", ["ReadTimeout", "ReadError"])
async def test_update_end_user_spend_does_not_retry_post_send_ambiguous_errors(
    mock_prisma_client: Any, ambiguous_error_name: str
) -> None:
    """Post-send errors are ambiguous and retrying can double-apply increments
    (see DB_RETRY_SAFE_ERROR_TYPES); they must raise on the first attempt."""
    import httpx

    err = getattr(httpx, ambiguous_error_name)("ambiguous")
    mock_prisma_client.db.tx = MagicMock(side_effect=err)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    with pytest.raises((httpx.ReadTimeout, httpx.ReadError)):
        await ProxyUpdateSpend.update_end_user_spend(
            n_retry_times=3,
            prisma_client=mock_prisma_client,
            proxy_logging_obj=proxy_logging,
            end_user_list_transactions={"u": 1.0},
        )
    mock_prisma_client.db.tx.assert_called_once()


@pytest.mark.asyncio
async def test_update_end_user_spend_non_connection_error_raises_immediately(
    mock_prisma_client: Any,
) -> None:
    mock_prisma_client.db.tx = MagicMock(side_effect=RuntimeError("unknown"))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    with pytest.raises(RuntimeError, match="unknown"):
        await ProxyUpdateSpend.update_end_user_spend(
            n_retry_times=3,
            prisma_client=mock_prisma_client,
            proxy_logging_obj=proxy_logging,
            end_user_list_transactions={"u": 1.0},
        )


def _end_user_deadlock_error() -> Exception:
    from prisma.errors import RawQueryError

    return RawQueryError(data={"user_facing_error": {"error_code": "P2034", "meta": {"table": "LiteLLM_EndUserTable"}}})


def _failing_tx(error: Exception) -> Any:
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(side_effect=error)
    tx.__aexit__ = AsyncMock(return_value=False)
    return tx


@pytest.mark.asyncio
async def test_update_end_user_spend_retries_on_deadlock_then_commits(
    mock_prisma_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for #27989: a Postgres deadlock (P2034/40P01) on the end-user
    spend batch is retried with jittered backoff and the increments land,
    instead of raising immediately and dropping the flushed spend."""
    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    batcher = MagicMock()
    batcher.litellm_endusertable.upsert = MagicMock()
    transaction = MagicMock()
    transaction.batch_ = lambda: _AsyncCM(batcher)
    mock_prisma_client.db.tx = MagicMock(side_effect=[_failing_tx(_end_user_deadlock_error()), _AsyncCM(transaction)])

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    await ProxyUpdateSpend.update_end_user_spend(
        n_retry_times=3,
        prisma_client=mock_prisma_client,
        proxy_logging_obj=proxy_logging,
        end_user_list_transactions={"end-user-1": 0.25},
    )

    assert mock_prisma_client.db.tx.call_count == 2
    batcher.litellm_endusertable.upsert.assert_called_once()
    assert batcher.litellm_endusertable.upsert.call_args.kwargs["where"] == {"user_id": "end-user-1"}
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] <= 2.0
    proxy_logging.failure_handler.assert_not_called()


@pytest.mark.asyncio
async def test_update_end_user_spend_raises_after_exhausting_deadlock_retries(
    mock_prisma_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prisma.errors import RawQueryError

    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))
    mock_prisma_client.db.tx = MagicMock(side_effect=lambda timeout: _failing_tx(_end_user_deadlock_error()))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    with pytest.raises(RawQueryError):
        await ProxyUpdateSpend.update_end_user_spend(
            n_retry_times=2,
            prisma_client=mock_prisma_client,
            proxy_logging_obj=proxy_logging,
            end_user_list_transactions={"end-user-1": 0.25},
        )

    assert mock_prisma_client.db.tx.call_count == 3


@pytest.mark.asyncio
async def test_update_spend_logs_writes_batches_via_create_many(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    logs = [make_spend_log_row(request_id=f"r{i}", spend=float(i)) for i in range(3)]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )
    kwargs = mock_prisma_client.db.litellm_spendlogs.create_many.await_args.kwargs
    pinned = {
        "calls": mock_prisma_client.db.litellm_spendlogs.create_many.await_count,
        "data_len": len(kwargs["data"]),
        "skip_duplicates": kwargs["skip_duplicates"],
        "first_request_id": kwargs["data"][0]["request_id"],
    }
    assert pinned == {
        "calls": 1,
        "data_len": 3,
        "skip_duplicates": True,
        "first_request_id": "r0",
    }


@pytest.mark.asyncio
async def test_update_spend_logs_bounds_each_statement_by_payload_bytes(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flush of prompt-carrying rows must reach Prisma as several small
    statements rather than one huge one: the query engine's resident memory is
    a high-water mark set by the largest statement it ever executes, and it
    never returns that memory to the OS."""
    monkeypatch.setattr(utils_mod, "SPEND_LOG_WRITE_BATCH_MAX_BYTES", 50_000)
    blob = json.dumps({"content": "x" * 10_000})
    logs = [make_spend_log_row(request_id=f"r{i}", messages=blob, response=blob) for i in range(50)]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )

    calls = mock_prisma_client.db.litellm_spendlogs.create_many.await_args_list
    written = [row["request_id"] for call in calls for row in call.kwargs["data"]]
    # Each statement is encoded whole rather than summed row by row, so the
    # assertion covers the collection framing the rows carry on the wire and
    # not only the payload the batcher adds up.
    largest_statement_bytes = max(len(json.dumps(list(call.kwargs["data"]), default=str)) for call in calls)
    assert written == [f"r{i}" for i in range(50)]
    assert [len(call.kwargs["data"]) for call in calls] == [2] * 25
    assert largest_statement_bytes <= 50_000


@pytest.mark.asyncio
async def test_update_spend_logs_uses_spend_logs_url_when_set(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEND_LOGS_URL", "http://writer.invalid")
    writer = MagicMock()
    writer.post = AsyncMock(return_value=MagicMock(status_code=200))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    logs = [make_spend_log_row(request_id="r1")]
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=writer,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )
    pinned = {
        "post_calls": writer.post.await_count,
        "url": writer.post.await_args.kwargs["url"],
        "headers": writer.post.await_args.kwargs["headers"],
        "create_many_calls": mock_prisma_client.db.litellm_spendlogs.create_many.await_count,
    }
    assert pinned == {
        "post_calls": 1,
        "url": "http://writer.invalid/spend/update",
        "headers": {"Content-Type": "application/json"},
        "create_many_calls": 0,
    }


@pytest.mark.asyncio
async def test_update_spend_logs_pops_logs_when_logs_to_process_is_none(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    mock_prisma_client.spend_log_transactions = [
        make_spend_log_row(request_id="a"),
        make_spend_log_row(request_id="b"),
    ]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )
    assert mock_prisma_client.spend_log_transactions == []
    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 1


@pytest.mark.asyncio
async def test_update_spend_logs_failure_raises_after_retries(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When all retries exhaust the underlying DB error, the helper raises
    via ``_raise_failed_update_spend_exception``.
    """
    import httpx
    import litellm.proxy.utils as utils_mod

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)

    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=httpx.ReadError("network blip"))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    with pytest.raises(httpx.ReadError):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=1,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=[make_spend_log_row(request_id="r1")],
        )


def _data_error(message: str) -> Any:
    from prisma.errors import DataError

    return DataError({"user_facing_error": {"message": message}})


@pytest.mark.asyncio
async def test_update_spend_logs_isolates_poison_row_and_persists_good_rows(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    """One row Postgres rejects (22P05) must not drop the whole batch.

    The good rows still persist and only the offending row is dropped, with no
    exception bubbling up. On the unfixed single-shot ``create_many`` the first
    write raises and the entire batch is lost.
    """
    poison_id = "r1"
    written: List[str] = []

    async def _create_many(*, data: Any, skip_duplicates: bool) -> None:
        ids = [row["request_id"] for row in data]
        if poison_id in ids:
            raise _data_error("Inconsistent column data: 22P05 invalid byte sequence for encoding UTF8: 0x00")
        written.extend(ids)

    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=_create_many)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    logs = [make_spend_log_row(request_id=f"r{i}") for i in range(4)]
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )
    assert sorted(written) == ["r0", "r2", "r3"]


@pytest.mark.asyncio
async def test_update_spend_logs_reraises_connection_masquerade_dataerror(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    """A P1001 "can't reach database server" outage that prisma mislabels as a
    ``DataError`` is transient, not a poison row: it must propagate so the batch
    is surfaced/retried rather than bisected into silent per-row drops.
    """
    err = _data_error("Can't reach database server at db-host:5432")
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=err)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    with pytest.raises(type(err)):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=0,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=[
                make_spend_log_row(request_id="a"),
                make_spend_log_row(request_id="b"),
            ],
        )


@pytest.mark.asyncio
async def test_update_spend_logs_retries_and_requeues_batch_on_db_outage(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A P1001 outage must be retried and, once retries exhaust, the batch goes
    back to the head of the queue so the next flush persists it. Before the fix
    prisma's ``DataError`` masquerade fell outside the retry clause, so the pod
    dropped every queued spend log for the duration of the outage.
    """

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(
        side_effect=_data_error("Can't reach database server at db-host:5432 (P1001)")
    )
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    logs = [make_spend_log_row(request_id="a"), make_spend_log_row(request_id="b")]
    queued_during_outage = make_spend_log_row(request_id="c")
    mock_prisma_client.spend_log_transactions = [queued_during_outage]

    with pytest.raises(type(_data_error("x"))):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=2,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=logs,
        )

    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 3
    assert [row["request_id"] for row in mock_prisma_client.spend_log_transactions] == ["a", "b", "c"]


def _deadlock_error() -> Exception:
    return _data_error(
        "Error occurred during query execution: ConnectorError(ConnectorError { user_facing_error: None, "
        'kind: QueryError(PostgresError { code: "40P01", message: "deadlock detected", severity: "ERROR" }) })'
    )


@pytest.mark.asyncio
async def test_update_spend_logs_retries_deadlock_and_keeps_every_row(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 40P01 deadlock aborts the whole insert, so the same rows succeed on replay.
    Before the fix the deadlock surfaced as a plain ``DataError`` and went through
    poison-row isolation, which bisected the batch and dropped every row the
    deadlock happened to hit as if Postgres had rejected it.
    """

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)
    create_many = AsyncMock(side_effect=[_deadlock_error(), _deadlock_error(), None])
    mock_prisma_client.db.litellm_spendlogs.create_many = create_many
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = []

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=2,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=[make_spend_log_row(request_id="a"), make_spend_log_row(request_id="b")],
    )

    attempts = tuple(tuple(row["request_id"] for row in call.kwargs["data"]) for call in create_many.await_args_list)
    assert attempts == (("a", "b"), ("a", "b"), ("a", "b"))
    assert mock_prisma_client.spend_log_transactions == []


@pytest.mark.asyncio
async def test_update_spend_logs_requeues_batch_once_deadlock_retries_exhaust(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If every retry deadlocks, the batch goes back to the head of the queue for
    the next flush instead of being dropped.
    """

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=_deadlock_error())
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = [make_spend_log_row(request_id="c")]

    with pytest.raises(type(_deadlock_error())):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=1,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=[make_spend_log_row(request_id="a"), make_spend_log_row(request_id="b")],
        )

    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 2
    assert [row["request_id"] for row in mock_prisma_client.spend_log_transactions] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_update_spend_logs_requeues_batch_on_non_transport_db_error(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    """A DB error that is neither transport nor deadlock (here P2021, the table is
    gone mid-migration) is not retried in place, but the dequeued batch must not
    be lost either: it goes back to the head of the queue so it lands once the
    DB is healthy again.
    """
    from prisma.errors import TableNotFoundError

    err = TableNotFoundError(
        {"user_facing_error": {"error_code": "P2021", "message": "The table does not exist", "meta": {}}}
    )
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=err)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = [make_spend_log_row(request_id="c")]

    with pytest.raises(TableNotFoundError):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=2,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=[make_spend_log_row(request_id="a"), make_spend_log_row(request_id="b")],
        )

    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 1
    assert [row["request_id"] for row in mock_prisma_client.spend_log_transactions] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_requeue_after_outage_drops_oldest_logs_past_the_byte_budget(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    """Requeueing must stay bounded by what the queue costs in memory, not by a
    row count: a row carries the whole prompt under
    ``store_prompts_in_spend_logs``, so a row cap that survives an outage of
    counter-only rows is an OOM once prompts are stored. Past the budget the
    oldest rows are the ones dropped.
    """
    budget = 3 * spend_log_row_bytes(make_spend_log_row(request_id="new0"))
    mock_prisma_client.spend_log_transactions = []
    await enqueue_spend_logs(mock_prisma_client, [make_spend_log_row(request_id="new0")], max_bytes=budget)

    await enqueue_spend_logs(
        mock_prisma_client,
        [make_spend_log_row(request_id=f"old{i}") for i in range(4)],
        at_head=True,
        max_bytes=budget,
    )

    assert [row["request_id"] for row in mock_prisma_client.spend_log_transactions] == ["old2", "old3", "new0"]


@pytest.mark.asyncio
async def test_enqueue_drops_oldest_logs_once_producers_fill_the_queue(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    """The budget has to govern the producer side too. While a flush retries
    against a dead DB, requests keep landing, so an append path that ignores the
    budget leaves the outage OOM open no matter how well the requeue trims.
    """
    budget = 2 * spend_log_row_bytes(make_spend_log_row(request_id="old0"))
    mock_prisma_client.spend_log_transactions = []
    await enqueue_spend_logs(
        mock_prisma_client,
        [make_spend_log_row(request_id=f"old{i}") for i in range(2)],
        max_bytes=budget,
    )

    await enqueue_spend_logs(mock_prisma_client, [make_spend_log_row(request_id="new0")], max_bytes=budget)

    assert [row["request_id"] for row in mock_prisma_client.spend_log_transactions] == ["old1", "new0"]


@pytest.mark.asyncio
async def test_flush_returns_the_bytes_it_took_off_the_queue(mock_prisma_client: Any, make_spend_log_row: Any) -> None:
    """A flush has to give its bytes back to the budget. Accounting that only
    ever grows would treat a healthy pod as permanently full and start dropping
    fresh spend logs after the queue has already drained.
    """
    budget = 2 * spend_log_row_bytes(make_spend_log_row(request_id="row0"))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = []
    await enqueue_spend_logs(
        mock_prisma_client,
        [make_spend_log_row(request_id=f"row{i}") for i in range(2)],
        max_bytes=budget,
    )

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )
    await enqueue_spend_logs(mock_prisma_client, [make_spend_log_row(request_id="row9")], max_bytes=budget)

    assert [row["request_id"] for row in mock_prisma_client.spend_log_transactions] == ["row9"]


@pytest.mark.asyncio
async def test_update_spend_logs_does_not_requeue_non_transport_failures(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    """Only DB failures are worth replaying. A row the proxy itself cannot
    serialize would fail the same way on every flush, so requeueing it would
    wedge the head of the queue forever.
    """
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=ValueError("bad payload"))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = []

    with pytest.raises(ValueError, match="bad payload"):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=1,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=[make_spend_log_row(request_id="a")],
        )

    assert mock_prisma_client.spend_log_transactions == []
    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 1


@pytest.mark.asyncio
async def test_update_spend_logs_caps_isolation_attempts_under_poison_flood(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flood of poisoned rows must not amplify one failed bulk insert into
    unbounded failed inserts. The per-batch failure budget hard-caps the number
    of failed ``create_many`` calls regardless of how many rows are poisoned, so
    the DB work stays bounded and well below the input row count, and the helper
    still completes without raising.
    """
    import litellm.proxy.utils as utils_mod

    attempt_cap = utils_mod.MAX_SPEND_LOG_ISOLATION_FAILURES_PER_BATCH
    # single create_many batch (< BATCH_SIZE) whose row count exceeds the attempt
    # cap, so the bound bites and attempts stay below the input row count
    n_rows = attempt_cap * 3
    # One statement, so this measures the isolation cap alone. The per-statement
    # floor the row budget adds is pinned separately below.
    monkeypatch.setattr(utils_mod, "SPEND_LOG_WRITE_BATCH_MAX_ROWS", n_rows)

    async def _always_poison(*, data: Any, skip_duplicates: bool) -> None:
        raise _data_error("invalid byte sequence for encoding UTF8: 0x00")

    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=_always_poison)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    logs = [make_spend_log_row(request_id=f"r{i}") for i in range(n_rows)]

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )

    attempts = mock_prisma_client.db.litellm_spendlogs.create_many.await_count
    assert attempts <= attempt_cap
    assert attempts < n_rows


@pytest.mark.asyncio
async def test_row_budget_costs_at_most_one_extra_attempt_per_statement(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting a flush into more statements must not buy the poison flood a
    fresh isolation budget each time. Every statement costs the one insert it
    takes to discover it is poisoned, and the shared budget caps everything
    above that, so the whole flush stays within the cap plus the statement
    count however finely it is split.
    """
    attempt_cap = utils_mod.MAX_SPEND_LOG_ISOLATION_FAILURES_PER_BATCH
    n_rows = attempt_cap * 3
    logs = [make_spend_log_row(request_id=f"r{i}") for i in range(n_rows)]
    split = {"max_bytes": 2_000_000, "max_rows": 100, "monkeypatch": monkeypatch}

    # A clean flush issues exactly one call per statement, so this is the observed
    # split count rather than an arithmetic one; asserting it is >1 is what proves
    # the row budget really divided the flush.
    statements = await _flush_and_count_create_many(mock_prisma_client, logs, poison=False, **split)
    attempts = await _flush_and_count_create_many(mock_prisma_client, logs, poison=True, **split)

    assert statements > 1
    assert attempts <= attempt_cap + statements
    assert attempts < n_rows


async def _flush_and_count_create_many(
    mock_prisma_client: Any,
    logs: List[Any],
    max_bytes: int,
    poison: bool,
    monkeypatch: pytest.MonkeyPatch,
    max_rows: int = 10_000,
) -> int:
    """Run one flush and return how many ``create_many`` calls it issued.

    ``max_rows`` defaults high enough not to bind so a caller varying
    ``max_bytes`` measures the byte budget alone.
    """

    async def _create_many(*, data: Any, skip_duplicates: bool) -> None:
        if poison:
            raise _data_error("invalid byte sequence for encoding UTF8: 0x00")

    monkeypatch.setattr(utils_mod, "SPEND_LOG_WRITE_BATCH_MAX_BYTES", max_bytes)
    monkeypatch.setattr(utils_mod, "SPEND_LOG_WRITE_BATCH_MAX_ROWS", max_rows)
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=_create_many)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )
    return int(mock_prisma_client.db.litellm_spendlogs.create_many.await_count)


@pytest.mark.asyncio
async def test_poison_flood_cost_does_not_grow_with_the_number_of_statements(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting a flush by payload bytes must not multiply what a poison flood
    costs. The same poisoned rows are flushed as one statement and as several;
    the split run may only pay the one unavoidable insert per extra statement,
    not a fresh isolation budget each time.
    """
    blob = json.dumps({"content": "x" * 2_000})
    logs = [make_spend_log_row(request_id=f"r{i}", messages=blob, response=blob) for i in range(300)]
    split_bytes = 250_000

    statements = await _flush_and_count_create_many(
        mock_prisma_client, logs, max_bytes=split_bytes, poison=False, monkeypatch=monkeypatch
    )
    split_attempts = await _flush_and_count_create_many(
        mock_prisma_client, logs, max_bytes=split_bytes, poison=True, monkeypatch=monkeypatch
    )
    single_attempts = await _flush_and_count_create_many(
        mock_prisma_client, logs, max_bytes=1_000_000_000, poison=True, monkeypatch=monkeypatch
    )

    # A clean flush issues exactly one call per statement, so this is the split
    # count; asserting it is >1 is what proves the split was really exercised.
    assert statements > 1
    assert single_attempts == utils_mod.MAX_SPEND_LOG_ISOLATION_FAILURES_PER_BATCH
    assert split_attempts <= single_attempts + statements


@pytest.mark.asyncio
async def test_clean_statement_is_still_written_after_a_poison_flood(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sharing the isolation budget must not let a poison flood in one
    statement silently drop the clean statements behind it. The budget bounds
    failed inserts only, so a later statement is still attempted and its rows
    persist.
    """
    blob = json.dumps({"content": "x" * 2_000})
    poisoned = [make_spend_log_row(request_id=f"bad{i}", messages=blob, response=blob) for i in range(300)]
    # Larger than the budget, so it is always yielded as its own statement and
    # the assertion cannot pass by riding along with a poisoned one.
    lone_blob = json.dumps({"content": "x" * 300_000})
    clean = [make_spend_log_row(request_id="good", messages=lone_blob, response=lone_blob)]
    written: List[str] = []

    async def _create_many(*, data: Any, skip_duplicates: bool) -> None:
        ids = [row["request_id"] for row in data]
        if any(request_id.startswith("bad") for request_id in ids):
            raise _data_error("invalid byte sequence for encoding UTF8: 0x00")
        written.extend(ids)

    monkeypatch.setattr(utils_mod, "SPEND_LOG_WRITE_BATCH_MAX_BYTES", 250_000)
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=_create_many)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=poisoned + clean,
    )

    assert written == ["good"]


def test_disable_spend_updates_reflects_general_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static method delegates to ``general_settings['disable_spend_updates']``;
    flipping that value toggles the helper's return.
    """
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(proxy_server_mod, "general_settings", {"disable_spend_updates": True})
    pinned = {
        "with_flag_true": ProxyUpdateSpend.disable_spend_updates(),
        "type_is_bool": isinstance(ProxyUpdateSpend.disable_spend_updates(), bool),
        "method_is_static": isinstance(ProxyUpdateSpend.__dict__["disable_spend_updates"], staticmethod),
    }
    assert pinned == {
        "with_flag_true": True,
        "type_is_bool": True,
        "method_is_static": True,
    }


def test_disable_spend_updates_default_false_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(proxy_server_mod, "general_settings", {})
    assert ProxyUpdateSpend.disable_spend_updates() is False


def test_disable_spend_updates_error_when_general_settings_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.delattr(proxy_server_mod, "general_settings", raising=False)
    with pytest.raises(ImportError):
        ProxyUpdateSpend.disable_spend_updates()


SpendLogRow: TypeAlias = dict[str, object]
"""One LiteLLM_SpendLogs row as the flush hands it to prisma."""

SpendLogRowFactory: TypeAlias = Callable[..., SpendLogRow]


class _SpendLogsTable:
    """``LiteLLM_SpendLogs`` as the flush sees it: ``create_many`` is ``INSERT ... ON CONFLICT
    DO NOTHING`` on ``request_id`` and reports how many rows landed, and ``query_raw`` answers the
    identity read-back with the stored ``(request_id, litellm_call_id)`` pairs."""

    def __init__(self, seeded: Sequence[SpendLogRow] | None = None) -> None:
        self.rows: dict[str, SpendLogRow] = {str(row["request_id"]): dict(row) for row in seeded or []}
        self.inserts: list[list[str]] = []
        self.query_raw_calls = 0

    async def create_many(self, *, data: Sequence[SpendLogRow], skip_duplicates: bool) -> int:
        assert skip_duplicates is True
        self.inserts.append([str(row["request_id"]) for row in data])
        inserted = 0
        for row in data:
            request_id = str(row["request_id"])
            if request_id in self.rows:
                continue
            self.rows[request_id] = dict(row)
            inserted += 1
        return inserted

    async def query_raw(self, sql: str, request_ids: Sequence[str]) -> list[SpendLogRow]:
        self.query_raw_calls += 1
        assert "WHERE request_id = ANY($1::text[])" in sql
        return [
            {"request_id": rid, "litellm_call_id": row.get("litellm_call_id")}
            for rid, row in self.rows.items()
            if rid in request_ids
        ]


def _wire_spend_logs_table(
    mock_prisma_client: MagicMock, seeded: Sequence[SpendLogRow] | None = None
) -> _SpendLogsTable:
    table = _SpendLogsTable(seeded)
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=table.create_many)
    mock_prisma_client.db.query_raw = AsyncMock(side_effect=table.query_raw)
    return table


def _wire_routed_db(mock_prisma_client: MagicMock, table: _SpendLogsTable) -> MagicMock:
    """``prisma_client.db`` as a read-replica router: top-level reads go to the reader, which never
    sees this flush's writes, and ``writer`` is the engine the rows landed on."""
    routed = MagicMock(spec=RoutingPrismaWrapper)
    routed.writer_unavailable = False
    routed.litellm_spendlogs = MagicMock()
    routed.litellm_spendlogs.create_many = AsyncMock(side_effect=table.create_many)
    routed.query_raw = AsyncMock(return_value=[])
    routed.writer = MagicMock()
    routed.writer.query_raw = AsyncMock(side_effect=table.query_raw)
    mock_prisma_client.db = routed
    return routed


def _inference_row(
    make_spend_log_row: SpendLogRowFactory, *, request_id: str, litellm_call_id: str, response_id: str | None = None
) -> SpendLogRow:
    """A row as ``get_logging_payload`` builds it for an inference call: ``metadata`` is a JSON
    string carrying the id the provider minted, which is the row's key unless it was re-keyed."""
    return make_spend_log_row(
        request_id=request_id,
        litellm_call_id=litellm_call_id,
        call_type="acompletion",
        metadata=json.dumps({"response_id": response_id or request_id}),
    )


async def _flush(mock_prisma_client: MagicMock, logs: list[SpendLogRow]) -> None:
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=0,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )


@pytest.mark.asyncio
async def test_update_spend_logs_rekeys_the_rows_a_reused_provider_response_id_would_drop(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory
) -> None:
    """Three requests answered with one provider id are three charged requests, so three rows
    must land: the first keeps the provider id, the other two are re-keyed on their own call id.
    Observed on a live proxy against a self-hosted server returning a fixed completion id: the
    duplicate-tolerant flush kept one row while key and daily spend counted all three."""
    table = _wire_spend_logs_table(mock_prisma_client)
    logs = [
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id=f"call-{i}")
        for i in range(3)
    ]

    await _flush(mock_prisma_client, logs)

    assert {rid: row["litellm_call_id"] for rid, row in table.rows.items()} == {
        "chatcmpl-static-1": "call-0",
        "call-1": "call-1",
        "call-2": "call-2",
    }
    assert table.inserts == [["chatcmpl-static-1"] * 3, ["call-1", "call-2"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [json.dumps({"response_id": "static-1"}), {"response_id": "static-1"}])
async def test_update_spend_logs_rekeys_a_row_carrying_the_id_the_provider_minted(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory, metadata: str | dict[str, str]
) -> None:
    """Whatever the route, a row whose key is the id the provider minted for that response is a
    charged request of its own once another row holds the same id."""
    table = _wire_spend_logs_table(mock_prisma_client)
    logs = [
        make_spend_log_row(
            request_id="static-1", litellm_call_id=f"call-{i}", call_type="aresponses", metadata=metadata
        )
        for i in range(2)
    ]

    await _flush(mock_prisma_client, logs)

    assert list(table.rows) == ["static-1", "call-1"]


@pytest.mark.asyncio
async def test_update_spend_logs_reads_stored_identities_back_from_the_writer(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory
) -> None:
    """The read-back must see the rows this flush just landed, so it goes to the writer: with a
    read replica configured, ``db.query_raw`` is routed to the reader, and a lagging reader would
    report every just-inserted row as missing and re-insert it under its call id."""
    table = _wire_spend_logs_table(mock_prisma_client)
    routed = _wire_routed_db(mock_prisma_client, table)
    logs = [
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id=f"call-{i}")
        for i in range(2)
    ]

    await _flush(mock_prisma_client, logs)

    routed.writer.query_raw.assert_awaited_once()
    routed.query_raw.assert_not_awaited()
    assert sorted(table.rows) == ["call-1", "chatcmpl-static-1"]


@pytest.mark.asyncio
async def test_update_spend_logs_rekeys_only_the_colliding_rows_of_a_mixed_batch(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory
) -> None:
    """Two providers reusing ids and one issuing unique ids share a flush: every row of the
    unique provider keeps its key and only the later rows of each reused id are re-keyed."""
    table = _wire_spend_logs_table(mock_prisma_client)
    logs = [
        _inference_row(make_spend_log_row, request_id="chatcmpl-unique-a", litellm_call_id="call-a"),
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id="call-b"),
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-2", litellm_call_id="call-c"),
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id="call-d"),
        _inference_row(make_spend_log_row, request_id="chatcmpl-unique-e", litellm_call_id="call-e"),
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-2", litellm_call_id="call-f"),
    ]

    await _flush(mock_prisma_client, logs)

    assert {rid: row["litellm_call_id"] for rid, row in table.rows.items()} == {
        "chatcmpl-unique-a": "call-a",
        "chatcmpl-static-1": "call-b",
        "chatcmpl-static-2": "call-c",
        "call-d": "call-d",
        "chatcmpl-unique-e": "call-e",
        "call-f": "call-f",
    }


@pytest.mark.asyncio
async def test_update_spend_logs_rekeys_a_row_whose_provider_id_an_earlier_flush_stored(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory
) -> None:
    """The colliding row usually landed in an earlier flush, or from another worker."""
    table = _wire_spend_logs_table(
        mock_prisma_client,
        seeded=[_inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id="call-old")],
    )

    await _flush(
        mock_prisma_client,
        [_inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id="call-new")],
    )

    assert {rid: row["litellm_call_id"] for rid, row in table.rows.items()} == {
        "chatcmpl-static-1": "call-old",
        "call-new": "call-new",
    }


@pytest.mark.asyncio
async def test_update_spend_logs_does_not_rekey_a_replay_of_a_stored_row(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory
) -> None:
    """A transport retry replays the whole batch; a row already stored under its own call id is
    the same request, not a collision, and must not become a second row."""
    table = _wire_spend_logs_table(
        mock_prisma_client,
        seeded=[_inference_row(make_spend_log_row, request_id="chatcmpl-1", litellm_call_id="call-1")],
    )

    await _flush(
        mock_prisma_client,
        [_inference_row(make_spend_log_row, request_id="chatcmpl-1", litellm_call_id="call-1")],
    )

    assert list(table.rows) == ["chatcmpl-1"]
    assert len(table.inserts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, request_id, metadata",
    [
        ("aretrieve_batch", "batch_abc_batch_cost", json.dumps({"response_id": None})),
        ("afile_retrieve", "file-abc", json.dumps({})),
        ("avector_store_retrieve", "vs_abc", {}),
        ("aget_responses", "resp_abc", "not json"),
        (None, "obj_abc", None),
    ],
)
async def test_update_spend_logs_keeps_object_keyed_rows_collapsed_on_their_object_id(
    mock_prisma_client: MagicMock,
    make_spend_log_row: SpendLogRowFactory,
    call_type: str | None,
    request_id: str,
    metadata: str | dict[str, str] | None,
) -> None:
    """Every poll of one batch shares its cost row by design, and every read of a stored object
    is keyed on that object's id: such a row carries no minted response id, so a duplicate is
    not a lost row and the identity read-back is not even issued."""
    table = _wire_spend_logs_table(mock_prisma_client)
    logs = [
        make_spend_log_row(request_id=request_id, litellm_call_id=f"call-{i}", call_type=call_type, metadata=metadata)
        for i in range(2)
    ]

    await _flush(mock_prisma_client, logs)

    assert list(table.rows) == [request_id]
    assert table.query_raw_calls == 0
    assert len(table.inserts) == 1


@pytest.mark.asyncio
async def test_update_spend_logs_drops_and_logs_a_row_already_keyed_on_its_call_id(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """A row whose key already is its call id (a client pinning x-litellm-call-id on a failure
    row, say) has nothing to fall back to: it stays dropped, but loudly."""
    table = _wire_spend_logs_table(
        mock_prisma_client,
        seeded=[_inference_row(make_spend_log_row, request_id="call-pinned", litellm_call_id="call-other")],
    )

    with caplog.at_level(logging.ERROR, logger=utils_mod.verbose_proxy_logger.name):
        await _flush(
            mock_prisma_client,
            [
                make_spend_log_row(
                    request_id="call-pinned", litellm_call_id="call-pinned", call_type="acompletion", metadata="{}"
                )
            ],
        )

    assert list(table.rows) == ["call-pinned"]
    assert len(table.inserts) == 1
    dropped = [record.getMessage() for record in caplog.records if "dropping spend log row" in record.getMessage()]
    assert dropped == [
        "Spend tracking - dropping spend log row whose request_id call-pinned is already taken and whose "
        "litellm_call_id call-pinned offers no other key"
    ]


@pytest.mark.asyncio
async def test_update_spend_logs_rekey_that_collides_again_stops(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory
) -> None:
    """A client pinning one x-litellm-call-id against a provider reusing one response id
    collides on both keys. The stored row already keyed on that call id reads as this row's
    replay, so the row stays skipped in the same pass, exactly as a pinned call id skips today."""
    table = _wire_spend_logs_table(
        mock_prisma_client,
        seeded=[
            _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id="call-old"),
            make_spend_log_row(
                request_id="call-pinned", litellm_call_id="call-pinned", call_type="acompletion", metadata="{}"
            ),
        ],
    )

    await _flush(
        mock_prisma_client,
        [_inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id="call-pinned")],
    )

    assert sorted(table.rows) == ["call-pinned", "chatcmpl-static-1"]
    assert table.inserts == [["chatcmpl-static-1"]]


@pytest.mark.asyncio
async def test_update_spend_logs_treats_a_replay_of_a_rekeyed_row_as_stored(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """A transport retry replays a whole batch, including rows an earlier attempt already re-keyed.
    Those are stored under their call id, so the read-back finds them there and nothing is re-keyed
    or warned about a second time."""
    table = _wire_spend_logs_table(mock_prisma_client)
    logs = [
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id=f"call-{i}")
        for i in range(2)
    ]
    await _flush(mock_prisma_client, logs)
    inserts_after_first_flush = len(table.inserts)
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=utils_mod.verbose_proxy_logger.name):
        await _flush(mock_prisma_client, logs)

    assert sorted(table.rows) == ["call-1", "chatcmpl-static-1"]
    assert len(table.inserts) == inserts_after_first_flush + 1
    assert [record for record in caplog.records if "re-keyed" in record.getMessage()] == []


@pytest.mark.asyncio
async def test_update_spend_logs_leaves_rows_skipped_when_the_read_back_fails_on_its_data(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """A read-back the database rejects (not a transport fault) must not requeue the whole flush
    behind it: the skipped rows stay skipped, which is what happened before, and the failure is
    logged."""
    table = _wire_spend_logs_table(mock_prisma_client)
    mock_prisma_client.db.query_raw = AsyncMock(side_effect=_data_error("invalid input syntax for type text[]"))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    logs = [
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id=f"call-{i}")
        for i in range(2)
    ]

    with caplog.at_level(logging.ERROR, logger=utils_mod.verbose_proxy_logger.name):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=0,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=logs,
        )

    assert list(table.rows) == ["chatcmpl-static-1"]
    assert len(table.inserts) == 1
    assert mock_prisma_client.spend_log_transactions == []
    assert any("could not read back" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_update_spend_logs_retries_the_flush_when_the_read_back_hits_a_transport_fault(
    mock_prisma_client: MagicMock, make_spend_log_row: SpendLogRowFactory
) -> None:
    """A transport fault during the read-back is the same outage as one during the insert, so
    the flush retries (and finally requeues) instead of leaving the rows behind."""
    _ = _wire_spend_logs_table(mock_prisma_client)
    mock_prisma_client.db.query_raw = AsyncMock(side_effect=httpx.ReadError("network blip"))
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    logs = [
        _inference_row(make_spend_log_row, request_id="chatcmpl-static-1", litellm_call_id=f"call-{i}")
        for i in range(2)
    ]

    with pytest.raises(httpx.ReadError):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=0,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=logs,
        )

    assert mock_prisma_client.spend_log_transactions == logs
