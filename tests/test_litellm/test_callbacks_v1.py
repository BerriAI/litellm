import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Final, get_args

import pytest
from pydantic import TypeAdapter

from litellm import callbacks_v1
from litellm.callbacks_v1 import EnvelopeV1, EventName, RequestFactsV1, SchemaVersion, WirePatchV1

ROOT: Final = Path(__file__).parents[2]
GOLDEN_PATH: Final = ROOT / "litellm-rust/crates/callbacks-v1/golden/v1"


def _clear_registry() -> None:
    for subscriber in callbacks_v1.snapshot():
        callbacks_v1.unregister(subscriber)


@pytest.fixture(autouse=True)
def clear_registry() -> Generator[None, None, None]:
    _clear_registry()
    yield
    _clear_registry()


def test_the_event_names_and_schemas_are_the_literal_aliases() -> None:
    assert callbacks_v1.EVENTS == frozenset(get_args(EventName))
    assert callbacks_v1.SUPPORTED_SCHEMAS == frozenset(get_args(SchemaVersion))


def test_v1_golden_envelopes_match_the_typed_contract() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Items .*ReadOnly.*", category=UserWarning)
        adapter: Final = TypeAdapter(callbacks_v1.EnvelopeV1)
        envelopes: Final = [adapter.validate_json(path.read_text(), strict=True) for path in GOLDEN_PATH.glob("*.json")]
    assert {envelope["event"]["type"] for envelope in envelopes} == callbacks_v1.EVENTS
    assert {envelope["schema"] for envelope in envelopes} == callbacks_v1.SUPPORTED_SCHEMAS


class SyncObserver:
    name: str = "valid"
    events: frozenset[EventName] = frozenset({"call.started"})

    def on_event(self, event: callbacks_v1.EnvelopeV1) -> None:
        del event


class AsyncObserver:
    name: str = "async-observer"
    events: frozenset[EventName] = frozenset({"call.started"})

    async def async_on_event(self, event: callbacks_v1.EnvelopeV1) -> None:
        del event


class SyncInterceptor:
    name: str = "sync-interceptor"
    events: frozenset[EventName] = frozenset()

    def before_send(self, request: callbacks_v1.RequestFactsV1) -> callbacks_v1.WirePatchV1 | None:
        del request
        return None


class AsyncInterceptor:
    name: str = "async-interceptor"
    events: frozenset[EventName] = frozenset()

    async def async_before_send(self, request: callbacks_v1.RequestFactsV1) -> callbacks_v1.WirePatchV1 | None:
        del request
        return None


class WrongSyncKind:
    name: str = "wrong-sync-kind"
    events: frozenset[EventName] = frozenset({"call.started"})

    async def on_event(self, event: callbacks_v1.EnvelopeV1) -> None:
        del event


@pytest.mark.parametrize(
    ("callback", "error"),
    [
        (
            type(
                "Unsupported",
                (),
                {"name": "unsupported", "schema": 2, "events": frozenset(), "before_send": lambda *_: None},
            )(),
            callbacks_v1.UnsupportedSchema,
        ),
        (
            type(
                "Unknown",
                (),
                {"name": "unknown", "schema": 1, "events": frozenset({"new.event"}), "on_event": lambda *_: None},
            )(),
            callbacks_v1.UnknownEvent,
        ),
        (
            type("Missing", (), {"name": "missing", "schema": 1, "events": frozenset()})(),
            callbacks_v1.MissingHandler,
        ),
        (
            type("Empty", (), {"name": "empty", "schema": 1, "events": frozenset(), "on_event": lambda *_: None})(),
            callbacks_v1.EmptyObserverEvents,
        ),
        (
            type(
                "AsyncFunction",
                (),
                {
                    "name": "async-function",
                    "schema": 1,
                    "events": frozenset({"call.started"}),
                    "async_on_event": lambda *_: None,
                },
            )(),
            callbacks_v1.HandlerKindMismatch,
        ),
        (WrongSyncKind(), callbacks_v1.HandlerKindMismatch),
    ],
    ids=["schema", "event", "missing", "empty-events", "async-kind", "coroutine-in-sync-slot"],
)
def test_register_rejects_invalid_callbacks(callback: object, error: type[ValueError]) -> None:
    with pytest.raises(error):
        callbacks_v1.register(callback)  # pyright: ignore[reportArgumentType]  # invalid callback shapes are the subject of the test


@pytest.mark.parametrize("callback", [SyncObserver(), AsyncObserver(), SyncInterceptor(), AsyncInterceptor()])
def test_register_accepts_each_handler_family(callback: callbacks_v1.Callback) -> None:
    callbacks_v1.register(callback)
    assert len(callbacks_v1.snapshot()) == 1


def test_register_rejects_duplicate_names() -> None:
    callbacks_v1.register(SyncObserver())
    with pytest.raises(callbacks_v1.DuplicateName):
        callbacks_v1.register(SyncObserver())


def test_snapshot_is_stable_across_later_registry_writes() -> None:
    first: Final = SyncObserver()
    second: Final = AsyncObserver()
    callbacks_v1.register(first)
    captured: Final = callbacks_v1.snapshot()
    callbacks_v1.register(second)
    callbacks_v1.unregister(first)
    assert callbacks_v1.snapshot() is not captured
    assert tuple(subscriber.name for subscriber in captured) == ("valid",)


def test_register_defaults_the_schema_when_the_callback_declares_none() -> None:
    callbacks_v1.register(SyncObserver())
    assert callbacks_v1.snapshot()[0].schema == callbacks_v1.SCHEMA


def test_register_still_rejects_a_declared_foreign_schema() -> None:
    class Foreign(SyncObserver):
        name: str = "foreign"
        schema: int = 2

    with pytest.raises(callbacks_v1.UnsupportedSchema):
        callbacks_v1.register(Foreign())


def test_on_event_registers_a_sync_function_in_the_sync_slot() -> None:
    @callbacks_v1.on_event("call.started", "call.succeeded")
    def record(event: EnvelopeV1) -> None:
        del event

    subscriber: Final = callbacks_v1.snapshot()[0]
    assert subscriber.on_event is record
    assert subscriber.async_on_event is None
    assert subscriber.events == frozenset({"call.started", "call.succeeded"})
    assert subscriber.name.endswith("record")
    assert record(  # the function is returned unchanged, so it stays directly callable
        {"schema": 1, "call_id": "c", "call_type": "ocr", "seq": 0, "event": {"type": "response.received", "body": ""}}
    ) is None


def test_on_event_routes_a_coroutine_function_to_the_async_slot() -> None:
    @callbacks_v1.on_event("call.failed", name="named-observer")
    async def record(event: EnvelopeV1) -> None:
        del event

    subscriber: Final = callbacks_v1.snapshot()[0]
    assert subscriber.async_on_event is record
    assert subscriber.on_event is None
    assert subscriber.name == "named-observer"


def test_before_send_routes_by_kind_and_subscribes_to_no_events() -> None:
    @callbacks_v1.before_send
    def sync_patch(request: RequestFactsV1) -> WirePatchV1 | None:
        del request
        return None

    @callbacks_v1.before_send
    async def async_patch(request: RequestFactsV1) -> WirePatchV1 | None:
        del request
        return None

    first, second = callbacks_v1.snapshot()
    assert (first.before_send, first.async_before_send) == (sync_patch, None)
    assert (second.before_send, second.async_before_send) == (None, async_patch)
    assert first.events == second.events == frozenset()


def test_on_event_rejects_an_unknown_event_at_decoration() -> None:
    with pytest.raises(callbacks_v1.UnknownEvent):

        @callbacks_v1.on_event("call.startd")  # pyright: ignore[reportArgumentType]  # the typo is the subject of the test
        def record(event: EnvelopeV1) -> None:
            del event


def test_on_event_rejects_an_empty_event_set_at_decoration() -> None:
    with pytest.raises(callbacks_v1.EmptyObserverEvents):

        @callbacks_v1.on_event()
        def record(event: EnvelopeV1) -> None:
            del event


def test_unregister_accepts_a_bare_name() -> None:
    @callbacks_v1.on_event("call.started", name="by-name")
    def record(event: EnvelopeV1) -> None:
        del event

    assert len(callbacks_v1.snapshot()) == 1
    callbacks_v1.unregister("by-name")
    assert callbacks_v1.snapshot() == ()
