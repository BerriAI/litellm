"""
Streaming binary response wrapper for httpx.

Analogous to ``AiohttpResponseStream`` in ``aiohttp_transport.py`` but for
native httpx streaming responses.  Used by TTS / audio-speech endpoints when
``stream=True`` to yield audio bytes as they arrive from the upstream
provider without buffering the entire response in memory.
"""

import asyncio
from typing import Any, Optional

import httpx


class HttpxStreamHandler:
    """
    A streaming binary response that yields bytes as they arrive from the
    upstream provider, without buffering the entire response in memory.

    Returned by speech/TTS endpoints when ``stream=True`` is passed.  The 
    audio bytes are **not** available via a``.content`` property — they 
    must be consumed through ``aiter_bytes()`` (async) or ``iter_bytes()`` (sync).

    The caller is responsible for ensuring the stream is fully consumed or
    explicitly closed so that the underlying HTTP connection is released.
    ``aiter_bytes`` / ``iter_bytes`` handle this automatically via a
    ``finally`` block that invokes the cleanup callback.
    """

    _hidden_params: dict

    def __init__(
        self,
        response: httpx.Response,
        cleanup: Any = None,
    ) -> None:
        self.response = response
        self._hidden_params = {}
        self._cleanup = cleanup
        self._consumed = False

    async def aiter_bytes(
        self,
        chunk_size: Optional[int] = None,
    ) -> Any:  # AsyncIterator[bytes] — typed as Any for compatibility
        """Yield bytes from the upstream response as they arrive."""
        if self._consumed:
            raise RuntimeError("HttpxStreamHandler has already been consumed.")
        self._consumed = True
        try:
            async for chunk in self.response.aiter_bytes(chunk_size=chunk_size):
                yield chunk
        finally:
            await self.aclose()

    def iter_bytes(
        self,
        chunk_size: Optional[int] = None,
    ) -> Any:  # Iterator[bytes] — typed as Any for compatibility
        """Yield bytes from the upstream response as they arrive (sync)."""
        if self._consumed:
            raise RuntimeError("HttpxStreamHandler has already been consumed.")
        self._consumed = True
        try:
            for chunk in self.response.iter_bytes(chunk_size=chunk_size):
                yield chunk
        finally:
            self.close()

    async def aclose(self) -> None:
        """Release the underlying HTTP connection (async)."""
        try:
            await self.response.aclose()
        except Exception:
            pass
        if self._cleanup is not None:
            cleanup_result = self._cleanup()
            if asyncio.iscoroutine(cleanup_result):
                await cleanup_result

    def close(self) -> None:
        """Release the underlying HTTP connection (sync)."""
        try:
            self.response.close()
        except Exception:
            pass
        if self._cleanup is not None:
            result = self._cleanup()
            # If cleanup returns a coroutine in a sync context we cannot
            # await it — callers using sync streams should provide a sync
            # cleanup callback or None.
            if asyncio.iscoroutine(result):
                result.close()  # discard to avoid RuntimeWarning
