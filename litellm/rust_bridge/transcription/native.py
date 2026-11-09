from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall


class RustTranscription(Protocol):
    def __call__(
        self,
        call: NativeCall,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAtranscription(Protocol):
    def __call__(
        self,
        call: NativeCall,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


def _sync_binding(value: object) -> RustTranscription | None:
    if not callable(value):
        return None
    return cast("RustTranscription", value)  # cast-ok: callable validated at the native binding boundary


def _async_binding(value: object) -> RustAtranscription | None:
    if not callable(value):
        return None
    return cast("RustAtranscription", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_TRANSCRIPTION: Final = NativeBinding("transcription", validate=_sync_binding)
NATIVE_ATRANSCRIPTION: Final = NativeBinding("atranscription", validate=_async_binding)
