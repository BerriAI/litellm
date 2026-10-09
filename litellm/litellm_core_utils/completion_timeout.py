"""Completion HTTP timeout resolution (kept out of ``main.py`` to limit import cycles)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import httpx

from litellm.constants import COMPLETION_HTTP_FALLBACK_SECONDS


class CompletionTimeout:
    """Resolves HTTP timeout for ``completion()`` from model vs global settings."""

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
        model_timeout: float | str | httpx.Timeout | None,
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
        resolved: float | str | httpx.Timeout
        if model_timeout is not None:
            resolved = model_timeout
        elif kwargs.get("timeout") is not None:
            resolved = kwargs["timeout"]
        elif kwargs.get("request_timeout") is not None:
            resolved = kwargs["request_timeout"]
        else:
            resolved = CompletionTimeout._fallback_when_no_explicit_timeout(global_timeout)

        if isinstance(resolved, httpx.Timeout) and not supports_httpx_timeout(custom_llm_provider):
            read_timeout: Final = resolved.read
            resolved = (
                float(read_timeout) if read_timeout is not None else COMPLETION_HTTP_FALLBACK_SECONDS
            )  # default 10 min timeout
        elif not isinstance(resolved, httpx.Timeout):
            resolved = float(resolved)

        return resolved
