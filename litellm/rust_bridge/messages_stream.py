"""Thin Python wrapper for the native Rust Anthropic Messages streaming bridge."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol

import httpx

from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustAmessagesStream(Protocol):
    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[object]:
        raise NotImplementedError


class RustMessagesStreamAdapter:
    """Adapts the native SSE-frame async iterator to ``AsyncIterator[bytes]``.

    ``aclose`` drops the native stream so the Rust side releases the upstream
    response body (aborting the provider request) instead of waiting for GC.
    """

    def __init__(self, native_stream: object) -> None:
        self._native: object | None = native_stream

    def __aiter__(self) -> RustMessagesStreamAdapter:
        return self

    async def __anext__(self) -> bytes:
        native = self._native
        if native is None:
            raise StopAsyncIteration
        return await native.__anext__()  # type: ignore[attr-defined,no-any-return]

    async def aclose(self) -> None:
        self._native = None

    async def __aenter__(self) -> RustMessagesStreamAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustMessagesStreamState:
    amessages_stream: RustAmessagesStream | None = None


_STATE: Final[_RustMessagesStreamState] = _RustMessagesStreamState()


def set_rust_amessages_stream(
    *,
    amessages_stream: RustAmessagesStream | None | _Unset = _UNSET,
) -> None:
    if not isinstance(amessages_stream, _Unset):
        _STATE.amessages_stream = amessages_stream


def load_rust_amessages_stream() -> RustAmessagesStream | None:
    if _STATE.amessages_stream is not None:
        return _STATE.amessages_stream
    from litellm.rust_bridge import get_native_bridge

    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    return getattr(native_bridge, "amessages_stream", None)  # type: ignore[return-value]


async def amessages_stream(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
) -> AsyncIterator[bytes] | None:
    """Start a native streaming call; returns an SSE-frame iterator, or ``None``.

    Unlike the non-streaming bridge, ``stream`` stays in the body: the native
    route forces it upstream. Each yielded item is one complete SSE frame
    (``event: ...\\ndata: {...}\\n\\n``) as ``bytes``.
    """
    rust_amessages_stream: Final = load_rust_amessages_stream()
    if rust_amessages_stream is None:
        return None
    native_stream: Final = await rust_amessages_stream(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout_seconds=timeout_to_seconds(timeout),
    )
    return RustMessagesStreamAdapter(native_stream)
