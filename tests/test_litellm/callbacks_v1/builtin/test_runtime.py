"""What the in-process host gives a port, and what it spares it, whichever shape it declares.

`test_conventions.py` holds the product's ports to these rules. This holds the rules
themselves, on ports written for the purpose, so the looser shape is covered before any
built-in needs it: an `ExporterPort` reaches its vendor itself, and everything else about how
it is run -- the joined record, the thread that is not the call's, a failure that is dropped
rather than raised -- is what a `SinkPort` already gets.
"""

import threading
from dataclasses import dataclass, field
from typing import Final

from litellm.callbacks_v1.builtin.port import Batching, CallRecord, ExporterPort
from litellm.callbacks_v1.builtin.runtime import Exporter, Sink
from tests.test_litellm.callbacks_v1.builtin.support import OPENMETER, FakeTransport, golden_call


@dataclass
class Recording(ExporterPort):
    """A port that reaches its vendor itself, which here is writing down what it was handed
    and which thread handed it over."""

    size: int = 1
    refuse: bool = False
    batches: list[tuple[CallRecord, ...]] = field(default_factory=list)
    threads: set[int] = field(default_factory=set)

    @property
    def batching(self) -> Batching:
        return Batching(size=self.size, interval=0.05)

    def export(self, batch: tuple[CallRecord, ...], /) -> None:
        self.threads.add(threading.get_ident())
        self.batches.append(batch)
        if self.refuse:
            raise RuntimeError("the vendor said no")


def test_an_exporter_reaches_its_vendor_itself_and_never_on_the_call() -> None:
    """The restriction was never on doing I/O; it is on doing it where the caller waits."""
    port: Final = Recording()
    subscriber: Final = Exporter("recording", port)
    handling: Final = threading.get_ident()

    for envelope in golden_call("call.succeeded"):
        subscriber.on_event(envelope)

    assert subscriber.outbox.flush()
    assert port.batches, "the exporter was never reached"
    assert port.threads and handling not in port.threads, "the vendor was reached on the call's own thread"


def test_an_exporter_gets_the_call_joined_from_its_envelopes() -> None:
    port: Final = Recording()
    subscriber: Final = Exporter("recording", port)

    for envelope in golden_call("call.succeeded", user_api_key_user_id="user-1"):
        subscriber.on_event(envelope)

    assert subscriber.outbox.flush()
    ((record,),) = port.batches
    assert record.terminal["type"] == "call.succeeded"
    assert record.started is not None and record.started["metadata"]["user_api_key_user_id"] == "user-1"
    assert record.request is not None and record.request["model"]
    assert subscriber.open_calls() == 0, "a finished call was kept"


def test_an_exporter_that_raises_loses_that_batch_and_keeps_serving() -> None:
    """A vendor's failure is the outbox's to swallow, as it is for the legacy batch loggers;
    a port does not wrap its own `export` in `try` to stay alive."""
    port: Final = Recording(refuse=True)
    subscriber: Final = Exporter("recording", port)

    for _ in range(2):
        for envelope in golden_call("call.succeeded"):
            subscriber.on_event(envelope)
        assert subscriber.outbox.flush()

    assert len(port.batches) == 2, "a batch the vendor refused stopped the next one"


def test_a_sink_and_an_exporter_are_run_the_same_way() -> None:
    """The two shapes differ in what a port returns, not in how the host runs it."""
    exporter: Final = Exporter("recording", Recording())
    sink: Final = Sink("openmeter", OPENMETER, FakeTransport())

    assert exporter.events == sink.events
    assert (exporter.name, sink.name) == ("recording", "openmeter")
    assert (exporter.open_calls(), sink.open_calls()) == (0, 0)
