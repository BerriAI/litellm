import inspect
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Final, cast

import pytest
from pydantic import BaseModel, TypeAdapter

from litellm.rust_bridge import callbacks_next as callbacks

ROOT: Final = Path(__file__).parents[3]
CONTRACT_PATH: Final = ROOT / "litellm-rust/crates/callbacks-next/python_contract.json"
GOLDEN_PATH: Final = ROOT / "litellm-rust/crates/callbacks-next/golden/v1"


def _clear_registry() -> None:
    for subscriber in callbacks.snapshot():
        callbacks.unregister(cast(callbacks.Callback, subscriber))


@pytest.fixture(autouse=True)
def clear_registry() -> Generator[None, None, None]:
    _clear_registry()
    yield
    _clear_registry()


def test_the_rust_contract_matches_the_shim_signatures() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())
    assert contract == {name: list(inspect.signature(getattr(callbacks, name)).parameters) for name in contract}


def test_v1_golden_envelopes_match_the_typed_contract() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Items .*ReadOnly.*", category=UserWarning)
        adapter: Final = TypeAdapter(callbacks.EnvelopeV1)
        events: Final = {
            adapter.validate_json(path.read_text(), strict=True)["event"]["type"] for path in GOLDEN_PATH.glob("*.json")
        }
    assert events == callbacks.EVENTS


class ValidCallback:
    name: str = "valid"
    schema: int = 1
    events: frozenset[str] = frozenset({"call.started"})

    def on_event(self, event: callbacks.EnvelopeV1) -> None:
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
            callbacks.UnsupportedSchema,
        ),
        (
            type(
                "Unknown",
                (),
                {"name": "unknown", "schema": 1, "events": frozenset({"new.event"}), "on_event": lambda *_: None},
            )(),
            callbacks.UnknownEvent,
        ),
        (
            type("Missing", (), {"name": "missing", "schema": 1, "events": frozenset()})(),
            callbacks.MissingHandler,
        ),
        (
            type("Empty", (), {"name": "empty", "schema": 1, "events": frozenset(), "on_event": lambda *_: None})(),
            callbacks.EmptyObserverEvents,
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
            callbacks.HandlerKindMismatch,
        ),
    ],
    ids=["schema", "event", "missing", "empty-events", "async-kind"],
)
def test_register_rejects_invalid_callbacks(callback: object, error: type[ValueError]) -> None:
    with pytest.raises(error):
        callbacks.register(callback)  # pyright: ignore[reportArgumentType]  # invalid callback shapes are the subject of the test


class WrongSyncKind:
    name: str = "wrong-sync-kind"
    schema: int = 1
    events: frozenset[str] = frozenset({"call.started"})

    async def on_event(self, event: callbacks.EnvelopeV1) -> None:
        del event


def test_register_rejects_coroutine_in_sync_slot() -> None:
    with pytest.raises(callbacks.HandlerKindMismatch):
        callbacks.register(cast(callbacks.Callback, WrongSyncKind()))


class AsyncObserver:
    name: str = "async-observer"
    schema: int = 1
    events: frozenset[str] = frozenset({"call.started"})

    async def async_on_event(self, event: callbacks.EnvelopeV1) -> None:
        del event


class SyncInterceptor:
    name: str = "sync-interceptor"
    schema: int = 1
    events: frozenset[str] = frozenset()

    def before_send(self, request: callbacks.RequestFactsV1) -> callbacks.WirePatchV1 | None:
        del request
        return None


class AsyncInterceptor:
    name: str = "async-interceptor"
    schema: int = 1
    events: frozenset[str] = frozenset()

    async def async_before_send(self, request: callbacks.RequestFactsV1) -> callbacks.WirePatchV1 | None:
        del request
        return None


@pytest.mark.parametrize("callback", [ValidCallback(), AsyncObserver(), SyncInterceptor(), AsyncInterceptor()])
def test_register_accepts_each_handler_family(callback: callbacks.Callback) -> None:
    callbacks.register(callback)
    assert len(callbacks.snapshot()) == 1


def test_register_rejects_duplicate_names() -> None:
    callbacks.register(cast(callbacks.Callback, ValidCallback()))
    with pytest.raises(callbacks.DuplicateName):
        callbacks.register(cast(callbacks.Callback, ValidCallback()))


def test_snapshot_is_stable_across_later_registry_writes() -> None:
    first: Final = ValidCallback()
    second: Final = AsyncObserver()
    callbacks.register(cast(callbacks.Callback, first))
    captured: Final = callbacks.snapshot()
    callbacks.register(cast(callbacks.Callback, second))
    callbacks.unregister(cast(callbacks.Callback, first))
    assert callbacks.snapshot() is not captured
    assert tuple(subscriber.name for subscriber in captured) == ("valid",)


class ResponseModel(BaseModel):
    value: int


def test_project_response_supports_pydantic_and_json_values() -> None:
    assert callbacks.project_response(ResponseModel(value=3)) == {"value": 3}
    original: Final = {"nested": [1, True, None]}
    assert callbacks.project_response(original) is original


def test_project_response_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError):
        callbacks.project_response(object())
