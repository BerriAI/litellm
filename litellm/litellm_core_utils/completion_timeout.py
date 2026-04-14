"""Completion HTTP timeout resolution (kept out of ``main.py`` to limit import cycles)."""

from __future__ import annotations

from typing import Callable, Optional, Union

import httpx

from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS


class CompletionTimeout:
    """Resolves HTTP timeout for ``completion()`` from model vs global settings."""

    @staticmethod
    def resolve(
        model_timeout: Optional[Union[float, str, httpx.Timeout]],
        kwargs: dict,
        custom_llm_provider: str,
        *,
        global_timeout: Optional[Union[float, str, httpx.Timeout]],
        supports_httpx_timeout: Callable[[str], bool],
    ) -> Union[float, httpx.Timeout]:
        """
        Order: ``model_timeout`` (call argument / merged ``litellm_params``), then
        ``kwargs["timeout"]``, ``kwargs["request_timeout"]``, then ``global_timeout``
        (e.g. :attr:`litellm.request_timeout` from proxy ``litellm_settings``), else ``600``.

        Coerce :class:`httpx.Timeout` when the provider does not support it. If the value
        came only from ``global_timeout`` (no model/kwargs timeout) and equals ``6000``
        (:data:`~litellm.constants.DEFAULT_REQUEST_TIMEOUT_SECONDS`), use ``600`` for
        completion so chat calls do not inherit the long package default; explicit
        ``model_timeout`` / kwargs values of ``6000`` are left unchanged.
        """
        timeout_from_global_only = False
        if model_timeout is not None:
            resolved: Union[float, str, httpx.Timeout] = model_timeout
        elif kwargs.get("timeout") is not None:
            resolved = kwargs["timeout"]
        elif kwargs.get("request_timeout") is not None:
            resolved = kwargs["request_timeout"]
        else:
            resolved = global_timeout if global_timeout is not None else 600
            timeout_from_global_only = True

        if isinstance(resolved, httpx.Timeout) and not supports_httpx_timeout(
            custom_llm_provider
        ):
            read_timeout = resolved.read
            resolved = (
                float(read_timeout) if read_timeout is not None else 600.0
            )  # default 10 min timeout
        elif not isinstance(resolved, httpx.Timeout):
            resolved = float(resolved)  # type: ignore

        if (
            timeout_from_global_only
            and not isinstance(resolved, httpx.Timeout)
            and float(resolved) == float(DEFAULT_REQUEST_TIMEOUT_SECONDS)
        ):
            resolved = 600.0

        return resolved
