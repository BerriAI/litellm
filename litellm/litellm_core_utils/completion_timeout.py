"""Completion HTTP timeout resolution (kept out of ``main.py`` to limit import cycles)."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Final

import httpx
from openai import Timeout as SDKTimeout
from typing_extensions import assert_never

from litellm.constants import COMPLETION_HTTP_FALLBACK_SECONDS


class StreamKind(Enum):
    NOT_STREAMED = "not_streamed"
    LIVE = "live"
    FAKED = "faked"


class CompletionTimeout:
    """Resolves HTTP timeout for ``completion()`` from model vs global settings."""

    @staticmethod
    def normalize(timeout: httpx.Timeout | SDKTimeout) -> httpx.Timeout:
        if isinstance(timeout, httpx.Timeout):  # pyright: ignore[reportUnnecessaryIsInstance]  # SDK 2 aliases both classes.
            return timeout
        return httpx.Timeout(connect=timeout.connect, read=timeout.read, write=timeout.write, pool=timeout.pool)

    @staticmethod
    def _fallback_when_no_explicit_timeout(
        global_timeout: float | str | None,
    ) -> float:
        """
        Used when ``model_timeout`` and kwargs timeouts are all unset.

        ``global_timeout`` is the explicitly-configured ``litellm.request_timeout``
        (numeric / string) or ``None`` when it was never set. ``None`` falls back to
        :data:`~litellm.constants.COMPLETION_HTTP_FALLBACK_SECONDS`; any explicit value
        (including ``6000``) is honored.
        """
        if global_timeout is None:
            return COMPLETION_HTTP_FALLBACK_SECONDS
        return float(global_timeout)

    @staticmethod
    def _longer_limit(first: float | None, second: float | None) -> float | None:
        return None if first is None or second is None else max(first, second)

    @staticmethod
    def _longer(first: float | httpx.Timeout, second: float | httpx.Timeout) -> float | httpx.Timeout:
        if not isinstance(first, httpx.Timeout) and not isinstance(second, httpx.Timeout):
            return max(first, second)
        first_limits: Final = httpx.Timeout(first)
        second_limits: Final = httpx.Timeout(second)
        longer: Final = CompletionTimeout._longer_limit
        return httpx.Timeout(
            connect=longer(first_limits.connect, second_limits.connect),
            read=longer(first_limits.read, second_limits.read),
            write=longer(first_limits.write, second_limits.write),
            pool=longer(first_limits.pool, second_limits.pool),
        )

    @staticmethod
    def for_stream(
        stream_kind: StreamKind, request_timeout: float | httpx.Timeout, stream_timeout: float | httpx.Timeout | None
    ) -> float | httpx.Timeout:
        """
        ``stream_timeout`` bounds each read of a live stream, so it replaces the request timeout there. A fake
        stream is one non-streaming request whose first byte arrives when generation ends, so it takes the longer
        of each limit and never fails sooner than the same call would with either one alone.
        """
        if stream_timeout is None:
            return request_timeout
        match stream_kind:
            case StreamKind.NOT_STREAMED:
                return request_timeout
            case StreamKind.LIVE:
                return stream_timeout
            case StreamKind.FAKED:
                return CompletionTimeout._longer(request_timeout, stream_timeout)
            case _:
                assert_never(stream_kind)

    @staticmethod
    def as_limit(timeout: object) -> float | httpx.Timeout:
        if isinstance(timeout, (httpx.Timeout, SDKTimeout)):
            return CompletionTimeout.normalize(timeout)
        if isinstance(timeout, (float, int, str)):
            return float(timeout)
        raise TypeError(f"timeout must be a number or an httpx.Timeout, got {type(timeout).__name__}")

    @staticmethod
    def resolve(
        model_timeout: float | str | httpx.Timeout | SDKTimeout | None,
        kwargs: dict,
        custom_llm_provider: str,
        *,
        global_timeout: float | str | None,
        supports_httpx_timeout: Callable[[str], bool],
    ) -> float | httpx.Timeout:
        """
        Resolution order (first non-None wins):

        1. ``model_timeout`` (call argument / merged ``litellm_params``)
        2. ``kwargs["timeout"]``
        3. ``kwargs["request_timeout"]``
        4. ``global_timeout`` (the explicitly-configured ``litellm.request_timeout``),
           or 600 when nothing was configured.

        Coerce :class:`httpx.Timeout` when the provider does not support it.
        """
        resolved: float | str | httpx.Timeout | SDKTimeout
        if model_timeout is not None:
            resolved = model_timeout
        elif kwargs.get("timeout") is not None:
            resolved = kwargs["timeout"]
        elif kwargs.get("request_timeout") is not None:
            resolved = kwargs["request_timeout"]
        else:
            resolved = CompletionTimeout._fallback_when_no_explicit_timeout(global_timeout)

        if isinstance(resolved, (httpx.Timeout, SDKTimeout)):
            normalized: Final = CompletionTimeout.normalize(resolved)
            if supports_httpx_timeout(custom_llm_provider):
                return normalized
            read_timeout: Final = normalized.read
            return (
                float(read_timeout) if read_timeout is not None else COMPLETION_HTTP_FALLBACK_SECONDS
            )  # default 10 min timeout
        return float(resolved)
