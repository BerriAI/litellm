import threading
import time

from litellm.callbacks_v1.builtin.outbox import Outbox
from litellm.callbacks_v1.builtin.runtime import CallJoin
from tests.test_litellm.callbacks_v1.builtin.support import golden, golden_call


def test_put_returns_before_a_slow_delivery_finishes() -> None:
    release = threading.Event()
    delivered: list[tuple[int, ...]] = []

    def deliver(batch: tuple[int, ...]) -> None:
        release.wait(2)
        delivered.append(batch)

    outbox = Outbox("slow", deliver)
    started = time.monotonic()
    outbox.put(1)

    assert time.monotonic() - started < 0.5
    assert delivered == []
    release.set()
    assert outbox.flush()
    assert delivered == [(1,)]


def test_items_are_delivered_batch_size_at_a_time_and_flush_sends_the_rest() -> None:
    delivered: list[tuple[int, ...]] = []
    outbox = Outbox("batched", delivered.append, batch_size=2, flush_interval=60)

    for item in (1, 2, 3):
        outbox.put(item)

    assert outbox.flush()
    assert delivered == [(1, 2), (3,)]


def test_a_failed_delivery_is_dropped_and_the_next_one_still_goes_out() -> None:
    delivered: list[tuple[int, ...]] = []

    def deliver(batch: tuple[int, ...]) -> None:
        if batch == (1,):
            raise RuntimeError("vendor is down")
        delivered.append(batch)

    outbox = Outbox("flaky", deliver)
    outbox.put(1)
    outbox.put(2)

    assert outbox.flush()
    assert delivered == [(2,)]


def test_a_full_outbox_drops_the_new_item_instead_of_blocking_the_call() -> None:
    release = threading.Event()
    outbox = Outbox("full", lambda batch: release.wait(2) and None, capacity=1)
    outbox.put(1)
    time.sleep(0.05)
    outbox.put(2)

    started = time.monotonic()
    outbox.put(3)

    assert time.monotonic() - started < 0.5
    release.set()
    assert outbox.flush()


def test_the_join_returns_one_record_at_the_terminal_envelope_and_forgets_the_call() -> None:
    join = CallJoin()
    started, request, received, terminal = golden_call("call.succeeded")

    assert [join.accept(envelope) for envelope in (started, request, received)] == [None, None, None]
    record = join.accept(terminal)

    assert record is not None
    assert (record.call_id, record.call_type) == ("call-123", "ocr")
    assert (record.started, record.request, record.terminal) == (started["event"], request["event"], terminal["event"])
    assert join.open_calls() == 0


def test_a_call_that_failed_before_sending_has_no_request() -> None:
    record = CallJoin().accept(golden("call.failed"))
    assert record is not None and record.request is None and record.started is None


def test_calls_that_never_finish_are_bounded() -> None:
    join = CallJoin(capacity=2)
    started = golden("call.started")
    for call_id in ("a", "b", "c"):
        join.accept({**started, "call_id": call_id})

    assert join.open_calls() == 2
