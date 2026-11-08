from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from typing import Final, cast  # noqa: TID251  # observation adapters preserve the native callable protocol

from pydantic import TypeAdapter

from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import _native
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.chat_completions.entrypoints import NATIVE_ACOMPLETION, NATIVE_COMPLETION
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR
from litellm.rust_bridge.public_call import NativeCall, optional_str
from litellm.rust_bridge.responses.entrypoints import NATIVE_ARESPONSES, NATIVE_RESPONSES
from tests.test_litellm_rust.support.recording_server import NativeEntry, RecordingServer

SyncEntry = Callable[[NativeCall], object]
AsyncEntry = Callable[[NativeCall], Awaitable[object]]


def _sync_observer(native: SyncEntry, entries: list[NativeEntry]) -> SyncEntry:
    def observed(request: NativeCall) -> object:
        try:
            result: Final = native(request)
        except _native.RustBridgeDeclined:
            entries.append(NativeEntry(optional_str(request.resolved.get("api_base")), True))
            raise
        except BaseException:
            entries.append(NativeEntry(optional_str(request.resolved.get("api_base")), False))
            raise
        entries.append(NativeEntry(optional_str(request.resolved.get("api_base")), False))
        return result

    return observed


def _async_observer(native: AsyncEntry, entries: list[NativeEntry]) -> AsyncEntry:
    async def observed(request: NativeCall) -> object:
        try:
            result: Final = await native(request)
        except _native.RustBridgeDeclined:
            entries.append(NativeEntry(optional_str(request.resolved.get("api_base")), True))
            raise
        except BaseException:
            entries.append(NativeEntry(optional_str(request.resolved.get("api_base")), False))
            raise
        entries.append(NativeEntry(optional_str(request.resolved.get("api_base")), False))
        return result

    return observed


def _observe_sync_binding(binding: NativeBinding[SyncEntry], entries: list[NativeEntry]) -> None:
    native: Final = binding.load()
    if native is not None:
        binding.override(_sync_observer(native, entries))


def _observe_async_binding(binding: NativeBinding[AsyncEntry], entries: list[NativeEntry]) -> None:
    native: Final = binding.load()
    if native is not None:
        binding.override(_async_observer(native, entries))


@contextmanager
def recording_native_entries() -> Generator[list[NativeEntry]]:
    entries: Final[list[NativeEntry]] = []
    synchronous: Final = cast(  # cast-ok: all sync route bindings accept NativeCall and return a public result
        tuple[NativeBinding[SyncEntry], ...],
        (NATIVE_MESSAGES, NATIVE_RESPONSES, NATIVE_COMPLETION, NATIVE_OCR),
    )
    asynchronous: Final = cast(  # cast-ok: all sync route bindings accept NativeCall and return a public result
        tuple[NativeBinding[AsyncEntry], ...],
        (NATIVE_AMESSAGES, NATIVE_ARESPONSES, NATIVE_ACOMPLETION, NATIVE_AOCR),
    )
    try:
        for binding in synchronous:
            _observe_sync_binding(binding, entries)
        for async_binding in asynchronous:
            _observe_async_binding(async_binding, entries)
        yield entries
    finally:
        for binding in (*synchronous, *asynchronous):
            binding.reset()


def assert_served_natively(server: RecordingServer, *responses: object, count: int = 1) -> None:
    assert len(server.requests) == (len(responses) or count)
    if not responses:
        accepted: Final = tuple(
            entry
            for entry in server.native_entries
            if not entry.declined and (entry.api_base is None or entry.api_base.startswith(server.base_url))
        )
        assert len(accepted) >= max(count, 1), "no matching native execution evidence"
    for response in responses:
        _assert_response_marker(response)


def _assert_response_marker(response: object) -> None:
    headers: Final = TypeAdapter(dict[str, object]).validate_python(
        get_hidden_params_dict(response).get("additional_headers")
    )
    assert headers.get("x-litellm-rust") == "true"
