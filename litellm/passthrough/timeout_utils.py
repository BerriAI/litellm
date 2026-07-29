import sys
from typing import Optional

DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS = 600.0


def resolve_pass_through_request_timeout(
    endpoint_timeout: Optional[float] = None,
) -> float:
    """
    Resolve the upstream httpx timeout for pass_through_request.

    Precedence: per-endpoint timeout -> general_settings.pass_through_request_timeout -> 600s default.

    Uses sys.modules to read general_settings only when the proxy module is already
    loaded, avoiding a fastapi transitive import in pure SDK contexts.
    """
    if endpoint_timeout is not None:
        return float(endpoint_timeout)

    try:
        proxy_server = sys.modules.get("litellm.proxy.proxy_server")
        if proxy_server is not None:
            global_timeout = getattr(proxy_server, "general_settings", {}).get("pass_through_request_timeout")
            if global_timeout is not None:
                return float(global_timeout)
    except Exception:
        pass

    return DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS


def resolve_llm_passthrough_timeout(
    kwargs: Optional[dict] = None,
    litellm_params: Optional[dict] = None,
    router_timeout: Optional[float] = None,
) -> float:
    """
    Resolve upstream httpx timeout for SDK native passthrough (e.g. Bedrock /converse).

    Precedence: kwargs timeout/request_timeout -> litellm_params timeout/request_timeout
    -> router_timeout -> general_settings.pass_through_request_timeout -> 600s default.
    """
    kwargs = kwargs or {}
    litellm_params = litellm_params or {}

    for source in (kwargs, litellm_params):
        for key in ("timeout", "request_timeout"):
            val = source.get(key)
            if val is not None:
                return float(val)

    if router_timeout is not None:
        return float(router_timeout)

    return resolve_pass_through_request_timeout()
