from typing import Final

from litellm.integrations.clickhouse.spend_log_types import SpendLogRecord
from litellm.integrations.lens.ledger import LedgerHandoff, Release


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _companion(request_id: str) -> SpendLogRecord:
    return SpendLogRecord(
        request_id=request_id,
        response_id=request_id,
        litellm_call_id="call",
        call_type="acompletion",
        api_key="callback-key",
        key_alias="alias",
        team_id="callback-team",
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
    "api_key": "committed-key",
    "team_id": "committed-team",
    "startTime": "2026-11-20T10:00:00",
    "request_duration_ms": 1500,
}
_ACCOUNTING: Final = {
    "spend": 0.25,
    "prompt_tokens": 3,
    "completion_tokens": 4,
    "total_tokens": 7,
    "api_key": "committed-key",
    "team_id": "committed-team",
    "user": "",
    "organization_id": "",
    "end_user": "",
}


def test_a_record_waits_for_its_commit_and_takes_the_committed_accounting() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    assert handoff.hold("request-a", _companion("request-a"), 100) == Release()
    released: Final = handoff.commit((_COMMITTED,))
    assert released.dropped == 0
    assert released.ready == ({**_companion("request-a"), **_ACCOUNTING},)
    assert (len(handoff.held), handoff.held_bytes, len(handoff.committed)) == (0, 0, 0)


def test_a_commit_that_arrives_first_waits_for_its_record() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    assert handoff.commit((_COMMITTED,)) == Release()
    assert handoff.hold("request-a", _companion("request-a"), 100).ready == (
        {**_companion("request-a"), **_ACCOUNTING},
    )


def test_a_commit_without_accounting_cost_is_dropped_not_filled_from_the_callback() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    handoff.hold("request-a", _companion("request-a"), 100)
    assert handoff.commit(({"request_id": "request-a", "delivery_id": "d"},)) == Release(dropped=1)


def test_unmatched_records_and_commits_expire_as_counted_drops_without_export() -> None:
    clock: Final = _Clock()
    handoff: Final = LedgerHandoff(clock=clock, wait_seconds=10)
    handoff.hold("uncommitted", _companion("uncommitted"), 100)
    assert (
        handoff.commit((_COMMITTED, {**_COMMITTED, "request_id": "other-pod", "delivery_id": "d-other"})) == Release()
    )
    clock.now = 10
    released: Final = handoff.hold("later", _companion("later"), 100)
    assert released == Release(dropped=3)
    assert (set(handoff.held), len(handoff.committed), handoff.waiting) == ({"later"}, 0, {})


def test_bounds_evict_the_oldest_records_and_drain_drops_everything_unmatched() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock(), max_rows=2, max_bytes=250)
    assert handoff.hold("a", _companion("a"), 100).dropped == 0
    assert handoff.hold("b", _companion("b"), 100).dropped == 0
    assert handoff.hold("c", _companion("c"), 100).dropped == 1
    assert handoff.hold("huge", _companion("huge"), 251).dropped == 1
    assert (tuple(handoff.held), handoff.held_bytes) == (("b", "c"), 200)
    overflow: Final = handoff.commit(tuple({**_COMMITTED, "delivery_id": f"d{index}"} for index in range(3)))
    assert (overflow.ready, overflow.dropped, len(handoff.committed)) == ((), 1, 2)
    assert handoff.drain() == Release(dropped=4)
    assert (len(handoff.held), handoff.held_bytes, len(handoff.committed), handoff.waiting) == (0, 0, 0, {})


def test_missing_canonical_usage_and_identity_never_fall_back_to_the_callback_values() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    handoff.hold("request-a", _companion("request-a"), 100)
    committed: Final = {"request_id": "request-a", "delivery_id": "d", "spend": 0.5, "prompt_tokens": None}
    merged: Final = handoff.commit((committed,)).ready[0]
    assert (merged["spend"], merged["prompt_tokens"], merged["completion_tokens"], merged["total_tokens"]) == (
        0.5,
        0,
        0,
        0,
    )
    assert (merged["api_key"], merged["team_id"], merged["user"]) == ("", "", "")
    assert (merged["messages"], merged["trace_id"]) == ("prompt", "trace")


def _delivery(delivery_id: str, spend: float) -> dict[str, object]:
    return {**_COMMITTED, "delivery_id": delivery_id, "spend": spend}


def test_only_the_delivery_that_meets_the_callback_record_is_exported_when_the_callback_comes_first() -> None:
    clock: Final = _Clock()
    handoff: Final = LedgerHandoff(clock=clock, wait_seconds=10)
    handoff.hold("request-a", _companion("request-a"), 100)
    first: Final = handoff.commit((_delivery("d1", 0.25), _delivery("d2", 0.75), _delivery("d1", 0.25)))
    assert [(row["spend"], row["messages"]) for row in first.ready] == [(0.25, "prompt")]
    clock.now = 10
    assert handoff.hold("other", _companion("other"), 100) == Release(dropped=1)
    assert handoff.commit((_delivery("d2", 0.75),)) == Release()


def test_only_the_delivery_that_meets_the_callback_record_is_exported_when_the_commits_come_first() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    assert handoff.commit((_delivery("d1", 0.25), _delivery("d2", 0.75))) == Release()
    assert [row["spend"] for row in handoff.hold("request-a", _companion("request-a"), 100).ready] == [0.25]
    assert handoff.drain() == Release(dropped=1)


def test_an_unreportable_delivery_is_counted_and_drops_its_waiting_record() -> None:
    handoff: Final = LedgerHandoff(clock=_Clock())
    handoff.hold("request-a", _companion("request-a"), 100)
    marker: Final = {"delivery_id": "d", "request_id": "request-a", "unreportable": True}
    assert handoff.commit((marker,)) == Release(dropped=1)
    assert (len(handoff.held), handoff.drain()) == (0, Release())


def test_expiry_follows_arrival_order_after_a_record_is_replaced() -> None:
    clock: Final = _Clock()
    handoff: Final = LedgerHandoff(clock=clock, wait_seconds=10)
    handoff.hold("a", _companion("a"), 100)
    clock.now = 5
    handoff.hold("b", _companion("b"), 100)
    clock.now = 6
    assert handoff.hold("a", _companion("a"), 100).dropped == 1
    clock.now = 15
    assert handoff.hold("c", _companion("c"), 100).dropped == 1
    assert tuple(handoff.held) == ("a", "c")
