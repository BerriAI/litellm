"""
Self-contained reactive surface fallback for Databricks **chat completions**.

The connector routes optimistically to the AI Gateway (no preflight probe). When a
workspace does not serve the gateway, the *real* request comes back with a
host-level "absent" signal (``404`` / ``501`` / ``endpoint_not_found``). This thin
wrapper catches that one case, records the host as gateway-absent (so every later
request for that host skips straight to serving-endpoints), and retries the call
once — which now resolves to ``<host>/serving-endpoints/chat/completions``.

Design notes:
- Entirely contained in the Databricks connector; it wraps the shared
  ``BaseLLMHTTPHandler.completion`` without modifying it (zero blast radius for
  other providers). The ``responses()`` path has its own surface-fallback chain.
- The retry re-runs the full handler flow, so the gateway→serving switch happens
  purely via the per-host cache + ``get_complete_url`` — no URL is threaded here.
- Only fires when the first attempt actually targeted the gateway (auto/optimistic
  or forced gateway); explicit ``/serving-endpoints`` users never pay a retry.
- ``optional_params`` is snapshotted and restored before the retry so values the
  first attempt popped (request tags, profile, user-agent) are reapplied.
- Streaming: a gateway-absent workspace returns the ``404`` at request
  establishment (inside ``completion``, before any chunk is produced), so the retry
  is safe. This wrapper only guards the ``completion`` call, NOT iteration of the
  returned stream, so a mid-stream error is intentionally never retried (no risk of
  duplicated/replayed chunks).
"""

import asyncio
import copy
from typing import Any, Optional

from ..ai_gateway import (
    is_gateway_absent_error,
    mark_gateway_absent,
    parse_use_ai_gateway_flag,
    resolve_surface,
    workspace_host_from_base,
)


def _safe_snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return dict(value) if isinstance(value, dict) else value


def _restore(target: dict, snapshot: Any) -> None:
    if isinstance(target, dict) and isinstance(snapshot, dict):
        target.clear()
        target.update(snapshot)


def _gateway_fallback_host(
    exc: Exception,
    api_base: Optional[str],
    optional_params: Any,
    litellm_params: Any,
) -> Optional[str]:
    """Return the workspace host to mark gateway-absent if ``exc`` is a host-level
    gateway-absence AND the first attempt targeted the gateway; else ``None``."""
    if not is_gateway_absent_error(exc):
        return None

    # Local import avoids a circular import at module load.
    from .transformation import DatabricksConfig

    try:
        host = workspace_host_from_base(DatabricksConfig()._get_api_base(api_base))
    except Exception:
        return None

    use_ai_gateway = parse_use_ai_gateway_flag(litellm_params, optional_params)
    surface = resolve_surface(api_base=api_base, use_ai_gateway=use_ai_gateway, host=host)
    return host if surface == "ai_gateway" else None


def databricks_chat_completion_with_surface_fallback(
    handler: Any,
    *,
    acompletion: bool,
    api_base: Optional[str],
    optional_params: dict,
    litellm_params: dict,
    **completion_kwargs: Any,
):
    """Call ``handler.completion``; on a host-level gateway-absent error, mark the
    host absent and retry once (resolving to serving-endpoints). Sync + async."""
    op_snapshot = _safe_snapshot(optional_params)

    def _call():
        return handler.completion(
            acompletion=acompletion,
            api_base=api_base,
            optional_params=optional_params,
            litellm_params=litellm_params,
            **completion_kwargs,
        )

    if acompletion:

        async def _acall():
            try:
                res = _call()
                if asyncio.iscoroutine(res):
                    res = await res
                return res
            except Exception as exc:
                host = _gateway_fallback_host(exc, api_base, op_snapshot, litellm_params)
                if host is None:
                    raise
                mark_gateway_absent(host)
                _restore(optional_params, op_snapshot)
                res = _call()
                if asyncio.iscoroutine(res):
                    res = await res
                return res

        return _acall()

    try:
        return _call()
    except Exception as exc:
        host = _gateway_fallback_host(exc, api_base, op_snapshot, litellm_params)
        if host is None:
            raise
        mark_gateway_absent(host)
        _restore(optional_params, op_snapshot)
        return _call()
