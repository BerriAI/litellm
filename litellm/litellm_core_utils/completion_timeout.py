"""Completion HTTP timeout resolution (kept out of ``main.py`` to limit import cycles)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from enum import Enum
from functools import partial
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
    def _read_seconds(timeout: float | httpx.Timeout) -> float:
        if isinstance(timeout, httpx.Timeout):
            return math.inf if timeout.read is None else timeout.read
        return timeout

    @staticmethod
    def for_stream(
        stream_kind: StreamKind, request_timeout: float | httpx.Timeout, stream_timeout: float | httpx.Timeout | None
    ) -> float | httpx.Timeout:
        """
        ``stream_timeout`` bounds each read of a live stream, so it replaces the request timeout there. A fake
        stream is one non-streaming request whose first byte arrives when generation ends, so it takes the longer
        of the two and never fails sooner than the same call would with either one alone.
        """
        if stream_timeout is None:
            return request_timeout
        match stream_kind:
            case StreamKind.NOT_STREAMED:
                return request_timeout
            case StreamKind.LIVE:
                return stream_timeout
            case StreamKind.FAKED:
                return max((request_timeout, stream_timeout), key=CompletionTimeout._read_seconds)
            case _:
                assert_never(stream_kind)

    @staticmethod
    def _coerce(
        timeout: object, custom_llm_provider: str, supports_httpx_timeout: Callable[[str], bool]
    ) -> float | httpx.Timeout:
        if isinstance(timeout, (httpx.Timeout, SDKTimeout)):
            normalized: Final = CompletionTimeout.normalize(timeout)
            if supports_httpx_timeout(custom_llm_provider):
                return normalized
            read_timeout: Final = normalized.read
            return (
                float(read_timeout) if read_timeout is not None else COMPLETION_HTTP_FALLBACK_SECONDS
            )  # default 10 min timeout
        if isinstance(timeout, (float, int, str)):
            return float(timeout)
        raise TypeError(f"timeout must be a number or an httpx.Timeout, got {type(timeout).__name__}")

    @staticmethod
    def resolve(
        model_timeout: float | str | httpx.Timeout | SDKTimeout | None,
        kwargs: Mapping[str, object],
        custom_llm_provider: str,
        *,
        global_timeout: float | str | None,
        supports_httpx_timeout: Callable[[str], bool],
        stream_kind: StreamKind,
    ) -> float | httpx.Timeout:
        """
        Request timeout resolution order (first non-None wins):

        1. ``model_timeout`` (call argument / merged ``litellm_params``)
        2. ``kwargs["timeout"]``
        3. ``kwargs["request_timeout"]``
        4. ``global_timeout`` (the explicitly-configured ``litellm.request_timeout``),
           or 600 when nothing was configured.

        A stream then applies ``kwargs["stream_timeout"]`` per :meth:`for_stream`.
        Coerce :class:`httpx.Timeout` when the provider does not support it.
        """
        coerce: Final = partial(
            CompletionTimeout._coerce,
            custom_llm_provider=custom_llm_provider,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        non_stream_candidates: Final = (model_timeout, kwargs.get("timeout"), kwargs.get("request_timeout"))
        explicit: Final = next((candidate for candidate in non_stream_candidates if candidate is not None), None)
        request_timeout: Final = coerce(
            explicit if explicit is not None else CompletionTimeout._fallback_when_no_explicit_timeout(global_timeout)
        )
        stream_timeout: Final = None if stream_kind is StreamKind.NOT_STREAMED else kwargs.get("stream_timeout")
        return CompletionTimeout.for_stream(
            stream_kind, request_timeout, None if stream_timeout is None else coerce(stream_timeout)
        )
