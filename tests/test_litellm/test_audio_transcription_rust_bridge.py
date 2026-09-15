from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Iterator
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.errors import RustRouteUnavailableError
from litellm.rust_bridge.transcription import configure_rust_transcription
from litellm.rust_bridge.transcription.lifecycle import wrap_async, wrap_sync


class RustBridgeDeclined(Exception):
    pass


class RustBridgeUnavailable(Exception):
    pass


class RustHostCallbackError(Exception):
    pass


class RustUpstreamError(Exception):
    pass


NATIVE_EXCEPTIONS: Final = SimpleNamespace(
    RustBridgeDeclined=RustBridgeDeclined,
    RustBridgeUnavailable=RustBridgeUnavailable,
    RustHostCallbackError=RustHostCallbackError,
    RustUpstreamError=RustUpstreamError,
)


class RecordingSync:
    def __init__(self, result: object = "native") -> None:
        self.result: Final = result
        self.calls: Final[list[tuple[dict[str, object], tuple[object, ...], dict[str, object], object]]] = []

    def __call__(
        self,
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> object:
        self.calls.append((request, args, kwargs, host))
        return self.result


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    configuration.reset_rust_configuration()
    configure_rust_transcription(transcription=None, atranscription=None)
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: NATIVE_EXCEPTIONS)
    yield
    configuration.reset_rust_configuration()
    configure_rust_transcription(transcription=None, atranscription=None)


def sync_python(model: str, file: object, **kwargs: object) -> object:
    return model, file, kwargs


async def async_python(model: str, file: object, **kwargs: object) -> object:
    return model, file, kwargs


def test_public_boundary_enters_native_once_and_preserves_call_shape() -> None:
    rust: Final = RecordingSync()
    configure_rust_transcription(transcription=rust)
    wrapped: Final = wrap_sync(sync_python)
    audio: Final = ("audio.wav", b"audio", "audio/wav")

    assert wrapped("bedrock/model", audio, temperature=0) == "native"
    assert len(rust.calls) == 1
    request, args, kwargs, _ = rust.calls[0]
    assert args == ("bedrock/model", audio)
    assert kwargs == {"temperature": 0}
    assert request["model"] == "bedrock/model"
    assert request["audio"] == {"format": "wav"}
    assert inspect.signature(wrapped) == inspect.signature(sync_python)


def test_python_provider_never_enters_native() -> None:
    rust: Final = RecordingSync()
    configure_rust_transcription(transcription=rust)
    wrapped: Final = wrap_sync(sync_python)

    assert wrapped("openai/whisper-1", b"audio") == ("openai/whisper-1", b"audio", {})
    assert rust.calls == []


def test_bedrock_requires_native_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    with pytest.raises(RustRouteUnavailableError, match="bridge is unavailable"):
        wrap_sync(sync_python)("bedrock/model", b"audio")


@pytest.mark.asyncio
async def test_async_post_admission_decline_does_not_fall_back() -> None:
    error: Final = RustBridgeDeclined("execution")

    def native(
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> Awaitable[object]:
        async def result() -> object:
            await asyncio.sleep(0)
            raise error

        return result()

    configure_rust_transcription(atranscription=native)
    with pytest.raises(RustBridgeDeclined) as caught:
        await wrap_async(async_python)("bedrock/model", b"audio")
    assert caught.value is error
