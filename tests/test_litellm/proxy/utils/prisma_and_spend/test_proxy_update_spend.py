"""Pin ``ProxyUpdateSpend`` behavior.

Symbols pinned here:
  - ``ProxyUpdateSpend.update_end_user_spend``
  - ``ProxyUpdateSpend.update_spend_logs``
  - ``ProxyUpdateSpend.disable_spend_updates``
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import ProxyUpdateSpend


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
async def test_update_end_user_spend_retries_on_connection_error(
    mock_prisma_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DB_CONNECTION_ERROR_TYPES`` failures should be retried with backoff;
    once retries are exhausted, ``_raise_failed_update_spend_exception`` is
    invoked and the original exception bubbles up.
    """
    import httpx
    import litellm.proxy.utils as utils_mod

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)

    err = httpx.ReadError("conn reset")
    mock_prisma_client.db.tx = MagicMock(side_effect=err)
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    with pytest.raises(httpx.ReadError):
        await ProxyUpdateSpend.update_end_user_spend(
            n_retry_times=1,
            prisma_client=mock_prisma_client,
            proxy_logging_obj=proxy_logging,
            end_user_list_transactions={"u": 1.0},
        )
    assert sleeps == [1.0]


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

    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(
        side_effect=httpx.ReadError("network blip")
    )
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
            raise _data_error(
                "Inconsistent column data: 22P05 invalid byte sequence for encoding UTF8: 0x00"
            )
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
async def test_update_spend_logs_caps_isolation_attempts_under_poison_flood(
    mock_prisma_client: Any, make_spend_log_row: Any
) -> None:
    """A flood of poisoned rows must not amplify one failed bulk insert into
    unbounded failed inserts. The per-batch attempt budget hard-caps the number
    of ``create_many`` calls regardless of how many rows are poisoned, so the DB
    work stays bounded and well below the input row count, and the helper still
    completes without raising.
    """
    import litellm.proxy.utils as utils_mod

    attempt_cap = utils_mod.MAX_SPEND_LOG_ISOLATION_ATTEMPTS_PER_BATCH
    # single create_many batch (< BATCH_SIZE) whose row count exceeds the attempt
    # cap, so the bound bites and attempts stay below the input row count
    n_rows = attempt_cap * 3

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


def test_disable_spend_updates_reflects_general_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static method delegates to ``general_settings['disable_spend_updates']``;
    flipping that value toggles the helper's return.
    """
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(
        proxy_server_mod, "general_settings", {"disable_spend_updates": True}
    )
    pinned = {
        "with_flag_true": ProxyUpdateSpend.disable_spend_updates(),
        "type_is_bool": isinstance(ProxyUpdateSpend.disable_spend_updates(), bool),
        "method_is_static": isinstance(
            ProxyUpdateSpend.__dict__["disable_spend_updates"], staticmethod
        ),
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
