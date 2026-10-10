import asyncio
import time
from typing import Final

import pytest

from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded, DBLookupStallTracker, bounded_db_lookup


async def _never_answers() -> None:
    await asyncio.Event().wait()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_bounded_db_lookup_fails_a_stalled_lookup_at_the_deadline_and_records_the_hit():
    tracker: Final = DBLookupStallTracker()
    started: Final = time.monotonic()

    with pytest.raises(DBLookupDeadlineExceeded) as exc_info:
        await bounded_db_lookup(_never_answers(), name="team", deadline_seconds=0.05, tracker=tracker)

    assert time.monotonic() - started < 2
    assert exc_info.value.lookup == "team"
    assert exc_info.value.deadline_seconds == 0.05
    assert str(exc_info.value) == "team lookup did not answer within 0.05s"
    assert isinstance(exc_info.value, asyncio.TimeoutError)
    assert tracker.stalled_within(30) is True


@pytest.mark.asyncio
async def test_bounded_db_lookup_returns_a_prompt_answer_without_recording_a_stall():
    tracker: Final = DBLookupStallTracker()

    async def answers() -> str:
        return "row"

    assert await bounded_db_lookup(answers(), name="key", deadline_seconds=0.05, tracker=tracker) == "row"
    assert tracker.stalled_within(30) is False


@pytest.mark.asyncio
async def test_bounded_db_lookup_fails_a_whole_stalled_burst_within_one_deadline():
    tracker: Final = DBLookupStallTracker()
    burst: Final = 200
    started: Final = time.monotonic()

    results: Final = await asyncio.gather(
        *(
            bounded_db_lookup(_never_answers(), name=f"key-{i}", deadline_seconds=0.1, tracker=tracker)
            for i in range(burst)
        ),
        return_exceptions=True,
    )

    assert time.monotonic() - started < 2
    assert len(results) == burst
    assert all(isinstance(result, DBLookupDeadlineExceeded) for result in results)
    assert tracker.stalled_within(30) is True


@pytest.mark.asyncio
async def test_bounded_db_lookup_fails_at_the_deadline_even_when_the_lookup_absorbs_the_cancel():
    tracker: Final = DBLookupStallTracker()
    absorbed: Final = asyncio.Event()
    let_go: Final = asyncio.Event()

    async def absorbs_the_cancel() -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            absorbed.set()
            await let_go.wait()
        return "late row"

    started: Final = time.monotonic()
    with pytest.raises(DBLookupDeadlineExceeded):
        await asyncio.wait_for(
            bounded_db_lookup(absorbs_the_cancel(), name="key", deadline_seconds=0.05, tracker=tracker),
            timeout=2,
        )

    assert time.monotonic() - started < 1
    assert tracker.stalled_within(30) is True
    await asyncio.wait_for(absorbed.wait(), timeout=1)
    let_go.set()
    await asyncio.sleep(0)


def test_stall_tracker_reports_a_stall_only_inside_the_window():
    clock: Final = _FakeClock()
    tracker: Final = DBLookupStallTracker(clock=clock)

    assert tracker.stalled_within(30) is False
    tracker.record_hit()
    assert tracker.stalled_within(30) is True
    assert tracker.stalled_within(0) is False
    clock.now += 29.9
    assert tracker.stalled_within(30) is True
    clock.now += 0.2
    assert tracker.stalled_within(30) is False
    tracker.record_hit()
    tracker.clear()
    assert tracker.stalled_within(30) is False
