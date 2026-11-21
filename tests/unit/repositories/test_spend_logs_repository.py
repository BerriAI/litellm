from collections.abc import Mapping, Sequence
from datetime import timedelta
from types import TracebackType
from typing import Final

import pytest

import litellm
from litellm.repositories import spend_logs_repository
from litellm.repositories.spend_log_commits import ACCOUNTING_FIELDS
from litellm.repositories.spend_logs_repository import SpendLogsRepository


class _Transaction:
    def __init__(self, accepted: frozenset[str], fail: bool) -> None:
        self.accepted: Final = accepted
        self.fail: Final = fail

    async def query_raw(self, sql: str, *args: object) -> list[dict[str, object]]:
        if self.fail:
            raise RuntimeError("connection dropped before commit")
        if sql.startswith('INSERT INTO "LiteLLM_SpendLogDelivery"'):
            return [{"delivery_id": delivery_id} for delivery_id in sorted(self.accepted)]
        return []

    async def execute_raw(self, sql: str, *args: object) -> int:
        return 0


class _Writer:
    def __init__(self, accepted: frozenset[str], fail: bool = False) -> None:
        self.transaction: Final = _Transaction(accepted, fail)
        self.committed = False

    def tx(self, *, timeout: timedelta) -> "_Writer":
        return self

    async def __aenter__(self) -> _Transaction:
        return self.transaction

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.committed = kind is None


class _Client:
    def __init__(self, writer: _Writer) -> None:
        self.writer_db: Final = writer


class _Observer:
    def __init__(self, writer: _Writer, fail: bool = False) -> None:
        self.writer: Final = writer
        self.fail: Final = fail
        self.heard: Final[list[tuple[bool, tuple[Mapping[str, object], ...]]]] = []  # mutable-ok: records calls

    def ledger_committed(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.heard.append((self.writer.committed, tuple(rows)))
        if self.fail:
            raise RuntimeError("lens is down")


def _row(name: str) -> dict[str, object]:
    return {
        "request_id": f"request-{name}",
        "accounting_delivery_id": f"delivery-{name}",
        "accounting_payload": {
            "request_id": f"request-{name}",
            "spend": 0.25,
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "api_key": "hash-a",
            "metadata": {"accounting_windows": []},
        },
        "spend": 9.0,
        "messages": "secret prompt",
        "endTime": "2026-11-20T10:00:01",
        "model_id": "model-a",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("observer_fails", (False, True), ids=("observer-ok", "observer-fails"))
async def test_observers_hear_only_accepted_deliveries_after_the_transaction_commits(
    monkeypatch: pytest.MonkeyPatch, observer_fails: bool
) -> None:
    writer: Final = _Writer(accepted=frozenset({"delivery-a"}))
    observer: Final = _Observer(writer, fail=observer_fails)
    monkeypatch.setattr(litellm, "callbacks", [observer])

    accepted: Final = await SpendLogsRepository(_Client(writer)).insert_with_deliveries(
        (_row("a"), _row("duplicate")), write_spend_logs=False
    )

    assert accepted == 1
    assert len(observer.heard) == 1
    committed, rows = observer.heard[0]
    assert committed is True
    assert rows == (
        {
            **dict.fromkeys(ACCOUNTING_FIELDS),
            "request_id": "request-a",
            "spend": 0.25,
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "api_key": "hash-a",
            "delivery_id": "delivery-a",
        },
    )


@pytest.mark.asyncio
async def test_a_failed_ledger_transaction_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    writer: Final = _Writer(accepted=frozenset({"delivery-a"}), fail=True)
    observer: Final = _Observer(writer)
    monkeypatch.setattr(litellm, "callbacks", [observer])

    with pytest.raises(RuntimeError, match="connection dropped"):
        await SpendLogsRepository(_Client(writer)).insert_with_deliveries((_row("a"),))

    assert observer.heard == []


@pytest.mark.asyncio
async def test_without_observers_the_ledger_count_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    writer: Final = _Writer(accepted=frozenset({"delivery-a", "delivery-b"}))
    monkeypatch.setattr(litellm, "callbacks", [])

    assert (
        await SpendLogsRepository(_Client(writer)).insert_with_deliveries(
            (_row("a"), _row("b")), write_spend_logs=False
        )
        == 2
    )


@pytest.mark.asyncio
async def test_a_report_that_fails_after_the_commit_keeps_the_exact_count_without_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer: Final = _Writer(accepted=frozenset({"delivery-a", "delivery-b"}))
    observer: Final = _Observer(writer)
    monkeypatch.setattr(litellm, "callbacks", [observer])
    calls: Final[list[int]] = []  # mutable-ok: counts projection attempts

    def broken(rows: object, accepted: object) -> tuple[dict[str, object], ...]:
        calls.append(1)
        raise ValueError("projection failed")

    monkeypatch.setattr(spend_logs_repository, "_committed_summaries", broken)

    accepted: Final = await SpendLogsRepository(_Client(writer)).insert_with_deliveries(
        (_row("a"), _row("b")), write_spend_logs=False
    )

    assert (accepted, writer.committed, len(calls), observer.heard) == (2, True, 1, [])
