"""
LIT-2577: install v3 rate-limit remaining-headers fallback on PrometheusLogger.

The legacy v1 ``parallel_request_limiter`` writes per-(key, model) remaining
values into request ``metadata`` under
    ``litellm-key-remaining-{requests,tokens}-{model_group}``

The active v3 limiter (``parallel_request_limiter_v3.py``) instead writes them
into ``response._hidden_params["additional_headers"]`` (mirrored into the
standard_logging_payload's ``hidden_params.additional_headers``) under
    ``x-ratelimit-{model_per_key,key}-remaining-{requests,tokens}``

``PrometheusLogger._set_virtual_key_rate_limit_metrics`` in
``litellm/integrations/prometheus.py`` only reads the v1 metadata keys, so
when v3 is in use the gauges fall through to ``sys.maxsize`` (~9.22e18) and
DataDog / Prometheus dashboards show nonsense values. This module wraps the
method so it reads the v3 ``additional_headers`` as a fallback before falling
through to ``sys.maxsize``; v1 metadata keys still take precedence so the
existing public contract is preserved.

This patch is installed at import time by ``litellm/integrations/__init__.py``.
A future cleanup can inline this logic directly into ``prometheus.py``; the
indirection here exists because the size of ``prometheus.py`` (~156 KB) blocks
the single-tool-call push path we have available today.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union


def _get_additional_headers_from_kwargs(kwargs: dict) -> Dict[str, Any]:
    """
    Pull ``additional_headers`` out of ``standard_logging_object.hidden_params``.

    Returns an empty dict on any missing/non-dict layer so callers can
    ``.get(...)`` freely without try/except.
    """
    standard_logging_payload = kwargs.get("standard_logging_object") or {}
    if not isinstance(standard_logging_payload, dict):
        return {}
    hidden_params = standard_logging_payload.get("hidden_params") or {}
    if not isinstance(hidden_params, dict):
        return {}
    additional_headers = hidden_params.get("additional_headers") or {}
    if not isinstance(additional_headers, dict):
        return {}
    return additional_headers


def _get_remaining_from_v3_headers(
    additional_headers: Dict[str, Any],
    rate_limit_type: str,
) -> Optional[Union[int, float]]:
    """
    Return the per-(key, model) remaining value emitted by the v3 rate limiter.

    The v3 post-call hook writes per-descriptor headers of the form
    ``x-ratelimit-{descriptor_key}-remaining-{rate_limit_type}``. For the
    Prometheus virtual-key gauges we want the per-(key, model) value so we
    prefer ``model_per_key`` (which already scopes by both the API key and the
    model_group) and fall back to ``key`` (per-key, all-models) if it is
    absent. Returns ``None`` if neither descriptor is present.
    """
    if not additional_headers:
        return None
    for descriptor in ("model_per_key", "key"):
        value = additional_headers.get(
            f"x-ratelimit-{descriptor}-remaining-{rate_limit_type}"
        )
        if value is not None:
            try:
                # v3 values are ints but tolerate strings/floats just in case
                return int(value)
            except (TypeError, ValueError):
                return value
    return None


def _install_patch() -> None:
    """
    Apply the v3 fallback to ``PrometheusLogger._set_virtual_key_rate_limit_metrics``.

    Idempotent — second/Nth call is a no-op.

    Strategy: wrap the existing method. If either of the v1 metadata keys
    (``litellm-key-remaining-{requests,tokens}-{model_group}``) is absent on
    the inbound ``metadata`` dict, look the value up in the v3
    ``additional_headers`` and inject it into a *shallow copy* of metadata
    before delegating to the original. The original's existing
    ``sys.maxsize`` fallback continues to handle the "neither limiter
    populated anything" case unchanged.
    """
    # NOTE: import lazily. Importing this module is done from
    # ``litellm/integrations/__init__.py``; resolving
    # ``litellm.integrations.prometheus`` here works because Python returns
    # the in-progress integrations package object for the parent reference
    # and then loads ``prometheus.py`` to completion as a submodule.
    from litellm.integrations import prometheus

    if getattr(prometheus.PrometheusLogger, "_lit2577_patched", False):
        return

    from litellm.proxy.common_utils.callback_utils import (
        get_model_group_from_litellm_kwargs,
    )

    original = prometheus.PrometheusLogger._set_virtual_key_rate_limit_metrics

    def patched(
        self,
        user_api_key,
        user_api_key_alias,
        kwargs,
        metadata,
        model_id=None,
    ):
        model_group = get_model_group_from_litellm_kwargs(kwargs)
        rk = f"litellm-key-remaining-requests-{model_group}"
        tk = f"litellm-key-remaining-tokens-{model_group}"

        # Only do work if at least one v1 key is missing
        v1_requests = metadata.get(rk) if isinstance(metadata, dict) else None
        v1_tokens = metadata.get(tk) if isinstance(metadata, dict) else None
        if v1_requests is not None and v1_tokens is not None:
            return original(
                self, user_api_key, user_api_key_alias, kwargs, metadata, model_id
            )

        headers = _get_additional_headers_from_kwargs(kwargs)
        if not headers:
            return original(
                self, user_api_key, user_api_key_alias, kwargs, metadata, model_id
            )

        # Build a shallow copy of metadata with v3-derived values filled in.
        # We never overwrite values the caller already provided.
        patched_metadata: Optional[dict] = None
        if v1_requests is None:
            v = _get_remaining_from_v3_headers(headers, "requests")
            if v is not None:
                patched_metadata = dict(metadata) if isinstance(metadata, dict) else {}
                patched_metadata[rk] = v
        if v1_tokens is None:
            v = _get_remaining_from_v3_headers(headers, "tokens")
            if v is not None:
                if patched_metadata is None:
                    patched_metadata = (
                        dict(metadata) if isinstance(metadata, dict) else {}
                    )
                patched_metadata[tk] = v

        if patched_metadata is not None:
            metadata = patched_metadata

        return original(
            self, user_api_key, user_api_key_alias, kwargs, metadata, model_id
        )

    prometheus.PrometheusLogger._set_virtual_key_rate_limit_metrics = patched
    prometheus.PrometheusLogger._lit2577_patched = True

    # Expose helpers as module-level attributes on ``prometheus`` so the
    # regression tests (which import them from
    # ``litellm.integrations.prometheus``) work without callers having to
    # know about this patch module.
    prometheus._get_additional_headers_from_kwargs = _get_additional_headers_from_kwargs
    prometheus._get_remaining_from_v3_headers = _get_remaining_from_v3_headers


# Apply on first import. The integrations package __init__ imports us, so this
# runs before any user code touches ``PrometheusLogger``.
_install_patch()
