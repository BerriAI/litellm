"""Completion HTTP timeout resolution (kept out of ``main.py`` to limit import cycles)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import httpx
from openai import Timeout as SDKTimeout
from pydantic import TypeAdapter

from litellm.constants import COMPLETION_HTTP_FALLBACK_SECONDS
from litellm.types.litellm_params import ServiceTierTimeout

_TIER_TIMEOUTS: Final = TypeAdapter(Mapping[str, ServiceTierTimeout])


def get_service_tier_timeout(
    request: Mapping[str, object],
    deployment: Mapping[str, object],
    defaults: Mapping[str, object],
) -> float | None:
    tier: Final = request.get("service_tier", deployment.get("service_tier", defaults.get("service_tier")))
    if not isinstance(tier, str):
        return None
    stream_overrides: Final = deployment.get("stream_timeout_by_service_tier") if request.get("stream") else None
    stream_timeout: Final = (
        _TIER_TIMEOUTS.validate_python(stream_overrides).get(tier) if stream_overrides is not None else None
    )
    if stream_timeout is not None:
        return stream_timeout
    overrides: Final = deployment.get("timeout_by_service_tier")
    return _TIER_TIMEOUTS.validate_python(overrides).get(tier) if overrides is not None else None


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
