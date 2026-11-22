from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.integrations.clickhouse.spend_log_types import SpendLogRecord
from litellm.integrations.lens.ledger import LedgerHandoff, Release, start_micros

_START: Final = datetime(2026, 11, 20, 10, 0, 0, 123456, tzinfo=timezone.utc)
_STARTED: Final = start_micros(_START)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _companion(request_id: str, api_key: str = "key-a", key_alias: str = "alias") -> SpendLogRecord:
    return SpendLogRecord(
        request_id=request_id,
        response_id=request_id,
        litellm_call_id="call",
        call_type="acompletion",
        api_key=api_key,
        key_alias=key_alias,
        team_id="team-a",
        team_alias="team alias",
        organization_id="",
        user="",
        end_user="",
        model="model",
        model_group="group",
        model_id="",
        custom_llm_provider="openai",
        api_base="",
        spend=99.0,
        prompt_tokens=99,
        completion_tokens=99,
        total_tokens=198,
        cache_read_tokens=0,
        cache_write_tokens=0,
        start_time=1,
        end_time=2,
        completion_start_time=None,
        status="success",
        error_str="",
        cache_hit=False,
        session_id="",
        trace_id="trace",
        span_id="span",
        request_tags=(),
        metadata="{}",
        messages="prompt",
        response="answer",
    )


_COMMITTED: Final = {
    "request_id": "request-a",
    "delivery_id": "delivery-a",
    "spend": 0.25,
    "prompt_tokens": 3,
    "completion_tokens": 4,
    "api_key": "key-a",
    "team_id": "team-a",
    "startTime": _START.isoformat(),
    "request_duration_ms": 1500,
}
_ACCOUNTING: Final = {
    "spend": 0.25,
    "prompt_tokens": 3,
    "completion_tokens": 4,
    "total_tokens": 7,
    "api_key": "key-a",
    "team_id": "team-a",
    "user": "",
    "organization_id": "",
    "end_user": "",
}


def test_a_record_waits_for_its_commit_and_takes_the_committed_accounting() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    assert handoff.hold("request-a", _STARTED, _companion("request-a"), 100) == Release()
    released: Final = handoff.commit((_COMMITTED,))
    assert released.dropped == 0
    assert released.ready == ({**_companion("request-a"), **_ACCOUNTING},)
    assert (len(handoff.held), handoff.held_bytes, len(handoff.committed)) == (0, 0, 0)


def test_a_commit_that_arrives_first_waits_for_its_record() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    assert handoff.commit((_COMMITTED,)) == Release()
    assert handoff.hold("request-a", _STARTED, _companion("request-a"), 100).ready == (
        {**_companion("request-a"), **_ACCOUNTING},
    )


def test_a_commit_without_accounting_cost_is_dropped_not_filled_from_the_callback() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    handoff.hold("request-a", _STARTED, _companion("request-a"), 100)
    assert handoff.commit(({**_COMMITTED, "spend": None},)) == Release(dropped=1)


def test_unmatched_records_and_commits_expire_as_counted_drops_without_export() -> None:
    clock: Final = _Clock()
    handoff: Final = LedgerHandoff(clock=clock, wait_seconds=10)
    handoff.hold("uncommitted", _STARTED, _companion("uncommitted"), 100)
    assert (
        handoff.commit((_COMMITTED, {**_COMMITTED, "request_id": "other-pod", "delivery_id": "d-other"})) == Release()
    )
    clock.now = 10
    released: Final = handoff.hold("later", _STARTED, _companion("later"), 100)
    assert released == Release(dropped=3)
    assert (tuple(key[0] for key in handoff.held), len(handoff.committed), handoff.waiting) == (("later",), 0, {})


def test_bounds_evict_the_oldest_records_and_drain_drops_everything_unmatched() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock(), max_rows=2, max_bytes=250)
    assert handoff.hold("a", _STARTED, _companion("a"), 100).dropped == 0
    assert handoff.hold("b", _STARTED, _companion("b"), 100).dropped == 0
    assert handoff.hold("c", _STARTED, _companion("c"), 100).dropped == 1
    assert handoff.hold("huge", _STARTED, _companion("huge"), 251).dropped == 1
    assert (tuple(key[0] for key in handoff.held), handoff.held_bytes) == (("b", "c"), 200)
    overflow: Final = handoff.commit(
        tuple({**_COMMITTED, "request_id": f"r{index}", "delivery_id": f"d{index}"} for index in range(3))
    )
    assert (overflow.ready, overflow.dropped, len(handoff.committed)) == ((), 1, 2)
    assert handoff.drain() == Release(dropped=4)
    assert (len(handoff.held), handoff.held_bytes, len(handoff.committed), handoff.waiting) == (0, 0, 0, {})


def test_missing_canonical_usage_and_identity_never_fall_back_to_the_callback_values() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    handoff.hold("request-a", _STARTED, _companion("request-a"), 100)
    committed: Final = {
        "request_id": "request-a",
        "delivery_id": "d",
        "spend": 0.5,
        "prompt_tokens": None,
        "api_key": "key-a",
        "team_id": "team-a",
        "startTime": _START.isoformat(),
    }
    merged: Final = handoff.commit((committed,)).ready[0]
    assert (merged["spend"], merged["prompt_tokens"], merged["completion_tokens"], merged["total_tokens"]) == (
        0.5,
        0,
        0,
        0,
    )
    assert (merged["organization_id"], merged["end_user"]) == ("", "")
    assert (merged["messages"], merged["trace_id"]) == ("prompt", "trace")


def _delivery(delivery_id: str, spend: float) -> dict[str, object]:
    return {**_COMMITTED, "delivery_id": delivery_id, "spend": spend}


def test_only_the_delivery_that_meets_the_callback_record_is_exported_when_the_callback_comes_first() -> None:
    clock: Final = _Clock()
    handoff: Final = LedgerHandoff(clock=clock, wait_seconds=10)
    handoff.hold("request-a", _STARTED, _companion("request-a"), 100)
    first: Final = handoff.commit((_delivery("d1", 0.25), _delivery("d2", 0.75), _delivery("d1", 0.25)))
    assert [(row["spend"], row["messages"]) for row in first.ready] == [(0.25, "prompt")]
    clock.now = 10
    assert handoff.hold("other", _STARTED, _companion("other"), 100) == Release(dropped=1)
    assert handoff.commit((_delivery("d2", 0.75),)) == Release()


def test_two_waiting_deliveries_that_cannot_be_told_apart_are_both_dropped() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    assert handoff.commit((_delivery("d1", 0.25), _delivery("d2", 0.75))) == Release(dropped=2)
    assert handoff.hold("request-a", _STARTED, _companion("request-a"), 100) == Release()
    assert handoff.drain() == Release(dropped=1)


def test_an_unreportable_delivery_is_counted_and_never_takes_a_record() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    handoff.hold("request-a", _STARTED, _companion("request-a"), 100)
    marker: Final = {"delivery_id": "d", "request_id": "request-a", "unreportable": True}
    assert handoff.commit((marker,)) == Release(dropped=1)
    assert handoff.drain() == Release(dropped=1)


def test_a_second_record_that_cannot_be_told_apart_drops_both_and_expiry_keeps_arrival_order() -> None:
    clock: Final = _Clock()
    handoff: Final = LedgerHandoff(clock=clock, wait_seconds=10)
    handoff.hold("a", _STARTED, _companion("a"), 100)
    clock.now = 5
    handoff.hold("b", _STARTED, _companion("b"), 100)
    clock.now = 6
    assert handoff.hold("a", _STARTED, _companion("a"), 100).dropped == 2
    assert handoff.commit((_COMMITTED | {"request_id": "a"},)) == Release()
    clock.now = 15
    assert handoff.hold("c", _STARTED, _companion("c"), 100).dropped == 1
    assert tuple(key[0] for key in handoff.held) == ("c",)


@pytest.mark.parametrize("commit_first", (False, True), ids=("record-first", "commit-first"))
def test_a_delivery_of_another_tenant_never_takes_a_record_with_the_same_request_id(commit_first: bool) -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    other_tenant: Final = {**_COMMITTED, "api_key": "key-b", "team_id": "team-b"}
    first, second = (
        (handoff.commit((other_tenant,)), handoff.hold("request-a", _STARTED, _companion("request-a"), 100))
        if commit_first
        else (handoff.hold("request-a", _STARTED, _companion("request-a"), 100), handoff.commit((other_tenant,)))
    )
    assert (first, second) == (Release(), Release())
    assert handoff.drain() == Release(dropped=2)


@pytest.mark.parametrize("commit_first", (False, True), ids=("records-first", "commits-first"))
def test_parallel_cache_hits_of_one_response_each_take_their_own_record(commit_first: bool) -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    later: Final = _START + timedelta(microseconds=1)
    commits: Final = (
        {**_COMMITTED, "request_id": "chatcmpl-1_cache_hit1795310982.45", "delivery_id": "d1", "spend": 0.0},
        {
            **_COMMITTED,
            "request_id": "chatcmpl-1_cache_hit1795310982.46",
            "delivery_id": "d2",
            "spend": 0.0,
            "startTime": later.isoformat(),
        },
    )
    records: Final = (
        ("chatcmpl-1", _STARTED, _companion("chatcmpl-1", key_alias="first")),
        ("chatcmpl-1", start_micros(later), _companion("chatcmpl-1", key_alias="second")),
    )
    early: Final = handoff.commit(commits) if commit_first else Release()
    held: Final = tuple(handoff.hold(request_id, started, row, 100) for request_id, started, row in records)
    late: Final = Release() if commit_first else handoff.commit(commits)
    ready: Final = (*early.ready, *(release.ready[0] for release in held if release.ready), *late.ready)
    assert sorted(str(row["key_alias"]) for row in ready) == ["first", "second"]
    assert handoff.drain() == Release()


@pytest.mark.parametrize(
    ("logged", "recorded", "pairs"),
    (
        pytest.param(_START.isoformat(), _START, True, id="same-instant"),
        pytest.param("2026-11-20T12:00:00.123456+02:00", _START, True, id="other-offset-same-instant"),
        pytest.param(_START.isoformat(), _START + timedelta(microseconds=1), False, id="one-microsecond-later"),
        pytest.param("2026-11-20T10:00:00+00:00", _START.replace(microsecond=0), True, id="whole-second"),
        pytest.param("not a time", _START, False, id="unparseable"),
    ),
)
def test_a_delivery_takes_a_record_only_for_the_exact_same_start_instant(
    logged: str, recorded: datetime, pairs: bool
) -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    handoff.hold("request-a", start_micros(recorded), _companion("request-a"), 100)
    released: Final = handoff.commit(({**_COMMITTED, "startTime": logged},))
    assert len(released.ready) == (1 if pairs else 0)


def test_a_request_id_that_already_ends_like_a_cache_hit_still_meets_its_delivery() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    request_id: Final = "chatcmpl-1_cache_hit1795310982.45"
    handoff.hold(request_id, _STARTED, _companion(request_id), 100)
    assert len(handoff.commit(({**_COMMITTED, "request_id": request_id},)).ready) == 1
