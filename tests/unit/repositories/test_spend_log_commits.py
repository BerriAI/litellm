import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

import litellm
from litellm.repositories.spend_log_commits import (
    ACCOUNTING_FIELDS,
    MAX_FIELD_BYTES,
    MAX_SUMMARY_BYTES,
    NativeCommitReports,
    committed_summary,
    notify_ledger_commit,
    report_ledger_commit,
)


class _Observer:
    def __init__(self, fail: bool = False) -> None:
        self.fail: Final = fail
        self.heard: Final = asyncio.Queue[tuple[Mapping[str, object], ...]]()

    def ledger_committed(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.heard.put_nowait(tuple(rows))
        if self.fail:
            raise RuntimeError("observer failed")


def test_summaries_carry_only_the_canonical_accounting_projection() -> None:
    summary: Final = committed_summary(
        "delivery-a",
        {"request_id": "request-a", "spend": 0.5, "model": "billed-model", "metadata": {"windows": []}},
        "logged-request",
    )
    assert summary == {
        **dict.fromkeys(ACCOUNTING_FIELDS),
        "request_id": "request-a",
        "spend": 0.5,
        "delivery_id": "delivery-a",
    }
    assert committed_summary("delivery-b", {"spend": 1.0}, "logged-request")["request_id"] == "logged-request"


@pytest.mark.parametrize(
    "accounting",
    (
        pytest.param({"user": "u" * (MAX_FIELD_BYTES + 1)}, id="accounting-field"),
        pytest.param({"team_id": ["nested"]}, id="non-scalar"),
        pytest.param({"team_id": "界" * (MAX_FIELD_BYTES // 3 + 1)}, id="multibyte-over-the-byte-bound"),
        pytest.param({"api_key": "\ud800"}, id="unencodable-text"),
    ),
)
def test_oversized_summaries_become_countable_markers(accounting: dict[str, object]) -> None:
    assert committed_summary("delivery-a", {"request_id": "request-a", **accounting}, "request-a") == {
        "delivery_id": "delivery-a",
        "request_id": "request-a",
        "unreportable": True,
    }


def test_field_bounds_count_utf8_bytes_like_the_native_writer() -> None:
    fits: Final = "界" * (MAX_FIELD_BYTES // 3)
    over: Final = fits + "界"
    assert committed_summary("delivery-a", {"request_id": "request-a", "team_id": fits}, "request-a")["team_id"] == fits
    assert committed_summary(over, {"request_id": "request-a"}, "request-a") == {
        "delivery_id": fits,
        "request_id": "request-a",
        "unreportable": True,
    }


def test_report_failures_never_escape_after_a_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    observer: Final = _Observer()
    monkeypatch.setattr(litellm, "callbacks", [observer])

    def broken() -> tuple[Mapping[str, object], ...]:
        raise KeyError("accounting_payload")

    report_ledger_commit(broken)
    assert observer.heard.empty()


def test_one_failing_observer_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    failing: Final = _Observer(fail=True)
    healthy: Final = _Observer()
    monkeypatch.setattr(litellm, "callbacks", [failing, "not-an-observer", healthy, healthy])
    notify_ledger_commit(({"request_id": "request-a"},))
    assert failing.heard.get_nowait() == ({"request_id": "request-a"},)
    assert healthy.heard.get_nowait() == ({"request_id": "request-a"},)
    assert healthy.heard.empty()
    notify_ledger_commit(())
    assert healthy.heard.empty()


@pytest.mark.asyncio
async def test_native_reports_decode_on_the_loop_within_their_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    observer: Final = _Observer()
    monkeypatch.setattr(litellm, "callbacks", [observer])
    reports: Final = NativeCommitReports(asyncio.get_running_loop(), max_pending=3)

    reports.receive([json.dumps({"request_id": "a"}), "not json", "x" * (MAX_SUMMARY_BYTES + 1)])
    reports.receive([json.dumps({"request_id": "over-budget"})])
    assert (reports.pending, reports.dropped) == (3, 1)
    assert observer.heard.empty()

    assert await asyncio.wait_for(observer.heard.get(), timeout=5) == ({"request_id": "a"},)
    assert reports.pending == 0


@pytest.mark.asyncio
async def test_native_reports_are_ignored_without_observers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    reports: Final = NativeCommitReports(asyncio.get_running_loop())
    reports.receive([json.dumps({"request_id": "a"})])
    assert (reports.pending, reports.dropped) == (0, 0)


class _ClosingLoop:
    """A loop that reports open but is closed by the time a report is scheduled, then reopens."""

    def __init__(self) -> None:
        self.closing = True
        self.scheduled: Final[list[tuple[str, ...]]] = []  # mutable-ok: records scheduled reports

    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, callback: object, texts: tuple[str, ...]) -> None:
        if self.closing:
            self.closing = False
            raise RuntimeError("Event loop is closed")
        self.scheduled.append(texts)


def test_native_reports_on_a_closing_loop_are_dropped_and_release_their_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_Observer()])
    closed: Final = asyncio.new_event_loop()
    closed.close()
    on_closed: Final = NativeCommitReports(closed, max_pending=2)
    on_closed.receive(["a", "b"])
    assert (on_closed.pending, on_closed.dropped) == (0, 2)

    closing: Final = _ClosingLoop()
    reports: Final = NativeCommitReports(closing, max_pending=2)  # pyright: ignore[reportArgumentType]  # duck-typed loop
    reports.receive(["a", "b"])
    assert (reports.pending, reports.dropped) == (0, 2)
    reports.receive(["c", "d"])
    assert (reports.pending, reports.dropped, closing.scheduled) == (2, 2, [("c", "d")])
