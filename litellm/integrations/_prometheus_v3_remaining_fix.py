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

This module is imported by ``litellm/integrations/__init__.py``. To avoid a
circular import (``prometheus.py`` -> ``litellm.proxy._types`` -> ``litellm``
which is still mid-init), we DO NOT touch ``prometheus`` at import time.
Instead we install a one-shot ``sys.meta_path`` finder that intercepts the
first import of ``litellm.integrations.prometheus`` and installs the patch
after that module finishes loading. By that point the litellm package has
fully initialized, so the proxy helper import in the wrapped method works.

A future cleanup can inline this logic directly into ``prometheus.py``; the
indirection here exists because the size of ``prometheus.py`` (~156 KB)
blocks the single-tool-call push path we have available today.
"""
from __future__ import annotations

import importlib
import importlib.abc
import sys
from typing import Any, Dict, Optional, Union

_PROMETHEUS_FQN = "litellm.integrations.prometheus"


def _get_additional_headers_from_kwargs(kwargs: dict) -> Dict[str, Any]:
    """
    Pull ``additional_headers`` out of ``standard_logging_object.hidden_params``.

    Returns an empty dict on any missing / non-dict layer so callers can
    ``.get(...)`` freely.
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

    Prefers ``model_per_key`` (which already scopes by both the API key and
    the model_group) and falls back to ``key`` (per-key, all-models) if
    absent. Returns ``None`` if neither descriptor is present *or* the value
    cannot be coerced to a number. Returning ``None`` (rather than the raw
    value) keeps the caller's fallback chain intact -- a malformed header
    must NOT propagate into ``Gauge.set()`` which only accepts numbers and
    would raise ``TypeError`` mid-callback, breaking the whole logging hook.
    """
    if not additional_headers:
        return None
    for descriptor in ("model_per_key", "key"):
        value = additional_headers.get(
            f"x-ratelimit-{descriptor}-remaining-{rate_limit_type}"
        )
        if value is None:
            continue
        # Accept ints/floats directly; coerce numeric strings; reject anything
        # else (will fall through to the next descriptor / sys.maxsize).
        if isinstance(value, bool):
            # bool is a subclass of int; reject to avoid silently treating
            # True/False as 1/0 in a rate-limit gauge.
            continue
        if isinstance(value, (int, float)):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return float(value)
            except (TypeError, ValueError):
                # Non-numeric header value (e.g. an error string). Skip it
                # rather than crash the prometheus gauge.
                continue
    return None


def _install_patch_on_module(prometheus_module: Any) -> None:
    """
    Apply the v3 fallback wrap to ``PrometheusLogger._set_virtual_key_rate_limit_metrics``.

    Safe to call multiple times — second/Nth call is a no-op.
    """
    PrometheusLogger = getattr(prometheus_module, "PrometheusLogger", None)
    if PrometheusLogger is None:
        return
    if getattr(PrometheusLogger, "_lit2577_patched", False):
        return

    original = PrometheusLogger._set_virtual_key_rate_limit_metrics

    def patched(
        self,
        user_api_key,
        user_api_key_alias,
        kwargs,
        metadata,
        model_id=None,
    ):
        # NOTE: this import is intentionally lazy. Doing it at module load
        # creates a circular import because ``callback_utils`` imports from
        # ``litellm`` which is still mid-init when integrations is imported.
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        model_group = get_model_group_from_litellm_kwargs(kwargs)
        rk = f"litellm-key-remaining-requests-{model_group}"
        tk = f"litellm-key-remaining-tokens-{model_group}"

        v1_requests = metadata.get(rk) if isinstance(metadata, dict) else None
        v1_tokens = metadata.get(tk) if isinstance(metadata, dict) else None

        if v1_requests is None or v1_tokens is None:
            headers = _get_additional_headers_from_kwargs(kwargs)
            if headers:
                patched_metadata: Optional[dict] = None
                if v1_requests is None:
                    v = _get_remaining_from_v3_headers(headers, "requests")
                    if v is not None:
                        patched_metadata = (
                            dict(metadata) if isinstance(metadata, dict) else {}
                        )
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

    PrometheusLogger._set_virtual_key_rate_limit_metrics = patched
    PrometheusLogger._lit2577_patched = True

    # Expose the helpers as module-level attributes on ``prometheus`` so the
    # regression tests can ``from litellm.integrations.prometheus import
    # _get_additional_headers_from_kwargs, _get_remaining_from_v3_headers``
    # without needing to know about this patch module.
    prometheus_module._get_additional_headers_from_kwargs = (
        _get_additional_headers_from_kwargs
    )
    prometheus_module._get_remaining_from_v3_headers = (
        _get_remaining_from_v3_headers
    )


class _PrometheusPostImportHook(importlib.abc.MetaPathFinder):
    """
    One-shot meta-path finder: intercepts the first import of
    ``litellm.integrations.prometheus``, lets the normal loader run it to
    completion, then installs our patch.

    After firing it removes itself from ``sys.meta_path``.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _PROMETHEUS_FQN:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            # Already removed (e.g. nested import) — harmless, fall through.
            pass
        # Walk the remaining finders so we don't deadlock on ourselves.
        for finder in list(sys.meta_path):
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            original_loader = spec.loader
            if original_loader is None or not hasattr(
                original_loader, "exec_module"
            ):
                return spec

            class _WrappedLoader(importlib.abc.Loader):
                def create_module(self, spec_):
                    if hasattr(original_loader, "create_module"):
                        return original_loader.create_module(spec_)
                    return None

                def exec_module(self, module):
                    original_loader.exec_module(module)
                    try:
                        _install_patch_on_module(module)
                    except Exception:
                        # Patch failure must NOT break Prometheus logging.
                        # The original method still works; we just don't get
                        # the v3 fallback. Worse than a fix, better than a crash.
                        import logging

                        logging.getLogger(__name__).warning(
                            "LIT-2577: failed to install Prometheus v3 fallback patch",
                            exc_info=True,
                        )

            spec.loader = _WrappedLoader()
            return spec
        return None


# If prometheus is already imported (unusual: would mean something pulled it
# in before integrations.__init__), patch it directly. Otherwise install the
# meta-path hook for the first future import.
_existing = sys.modules.get(_PROMETHEUS_FQN)
if _existing is not None:
    _install_patch_on_module(_existing)
else:
    _hook = _PrometheusPostImportHook()
    if not any(
        isinstance(f, _PrometheusPostImportHook) for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _hook)
