"""Completion HTTP timeout resolution (kept out of ``main.py`` to limit import cycles)."""

from __future__ import annotations

from typing import Callable, Optional, Union

import httpx

from litellm.constants import (
    COMPLETION_HTTP_FALLBACK_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)


class CompletionTimeout:
    """Resolves HTTP timeout for ``completion()`` from model vs global settings."""

    @staticmethod
    def _fallback_when_no_explicit_timeout(
        global_timeout: Optional[Union[float, str]],
    ) -> float:
        """
        Used when ``model_timeout`` and kwargs timeouts are all unset.

        ``global_timeout`` is :attr:`litellm.request_timeout` (numeric / string), not
        :class:`httpx.Timeout`.

        If it equals :data:`~litellm.constants.DEFAULT_REQUEST_TIMEOUT_SECONDS` (6000),
        return :data:`~litellm.constants.COMPLETION_HTTP_FALLBACK_SECONDS`. Same if
        ``None``. Otherwise return ``float(global_timeout)``.
        """
        if global_timeout is None:
            return COMPLETION_HTTP_FALLBACK_SECONDS
        if float(global_timeout) == float(DEFAULT_REQUEST_TIMEOUT_SECONDS):
            return COMPLETION_HTTP_FALLBACK_SECONDS
        return float(global_timeout)

    @staticmethod
    def resolve(
        model_timeout: Optional[Union[float, str, httpx.Timeout]],
        kwargs: dict,
        custom_llm_provider: str,
        *,
        global_timeout: Optional[Union[float, str]],
        supports_httpx_timeout: Callable[[str], bool],
    ) -> Union[float, httpx.Timeout]:
        """
        Resolution order (first non-None wins):

        1. ``model_timeout`` (call argument / merged ``litellm_params``)
        2. ``kwargs["timeout"]``
        3. ``kwargs["request_timeout"]``
        4. Fallback from ``global_timeout`` (:attr:`litellm.request_timeout`) — if it is
           the package default (6000), use 600 instead.

        Coerce :class:`httpx.Timeout` when the provider does not support it.
        Explicit ``6000`` on the model or in kwargs is kept as ``6000``.
        """
        resolved: Union[float, str, httpx.Timeout]
        if model_timeout is not None:
            resolved = model_timeout
        elif kwargs.get("timeout") is not None:
            resolved = kwargs["timeout"]
        elif kwargs.get("request_timeout") is not None:
            resolved = kwargs["request_timeout"]
        else:
            resolved = CompletionTimeout._fallback_when_no_explicit_timeout(
                global_timeout
            )

        if isinstance(resolved, httpx.Timeout) and not supports_httpx_timeout(
            custom_llm_provider
        ):
            read_timeout = resolved.read
            resolved = (
                float(read_timeout)
                if read_timeout is not None
                else COMPLETION_HTTP_FALLBACK_SECONDS
            )  # default 10 min timeout
        elif not isinstance(resolved, httpx.Timeout):
            resolved = float(resolved)  # type: ignore

        return resolved
