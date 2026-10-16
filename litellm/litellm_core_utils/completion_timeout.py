"""Completion HTTP timeout resolution (kept out of ``main.py`` to limit import cycles)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import httpx
from openai import Timeout as SDKTimeout

from litellm.constants import COMPLETION_HTTP_FALLBACK_SECONDS


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
    def resolve(
        model_timeout: float | str | httpx.Timeout | SDKTimeout | None,
        kwargs: Mapping[str, object],
        custom_llm_provider: str,
        *,
        global_timeout: float | str | None,
        supports_httpx_timeout: Callable[[str], bool],
        reads_live_stream: bool,
    ) -> float | httpx.Timeout:
        """
        Resolution order (first non-None wins):

        1. ``kwargs["stream_timeout"]``, only when ``reads_live_stream``. It bounds each read of a live
           stream, and a fake stream is one non-streaming request whose first byte arrives when
           generation ends, so a fake stream skips it
        2. ``model_timeout`` (call argument / merged ``litellm_params``)
        3. ``kwargs["timeout"]``
        4. ``kwargs["request_timeout"]``
        5. ``global_timeout`` (the explicitly-configured ``litellm.request_timeout``),
           or 600 when nothing was configured.

        Coerce :class:`httpx.Timeout` when the provider does not support it.
        """
        candidates: Final = (
            kwargs.get("stream_timeout") if reads_live_stream else None,
            model_timeout,
            kwargs.get("timeout"),
            kwargs.get("request_timeout"),
        )
        explicit: Final = next((candidate for candidate in candidates if candidate is not None), None)
        resolved: Final = (
            explicit if explicit is not None else CompletionTimeout._fallback_when_no_explicit_timeout(global_timeout)
        )

        if isinstance(resolved, (httpx.Timeout, SDKTimeout)):
            normalized: Final = CompletionTimeout.normalize(resolved)
            if supports_httpx_timeout(custom_llm_provider):
                return normalized
            read_timeout: Final = normalized.read
            return (
                float(read_timeout) if read_timeout is not None else COMPLETION_HTTP_FALLBACK_SECONDS
            )  # default 10 min timeout
        if isinstance(resolved, (float, int, str)):
            return float(resolved)
        raise TypeError(f"timeout must be a number or an httpx.Timeout, got {type(resolved).__name__}")
