import sys
from typing import Final

DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS: Final = 600.0


def resolve_pass_through_request_timeout(
    endpoint_timeout: float | None = None,
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
        proxy_server: Final = sys.modules.get("litellm.proxy.proxy_server")
        if proxy_server is not None:
            global_timeout: Final = getattr(proxy_server, "general_settings", {}).get("pass_through_request_timeout")
            if global_timeout is not None:
                return float(global_timeout)
    except Exception:
        pass

    return DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS


def resolve_llm_passthrough_timeout(
    kwargs: dict | None = None,
    litellm_params: dict | None = None,
    router_timeout: float | None = None,
    router_stream_timeout: float | None = None,
) -> float:
    """
    Resolve upstream httpx timeout for SDK native passthrough (e.g. Bedrock /converse,
    Anthropic /v1/messages).

    Non-streaming precedence: kwargs timeout/request_timeout -> litellm_params
    timeout/request_timeout -> router_timeout -> general_settings.pass_through_request_timeout
    -> 600s default.

    Streaming (``kwargs["stream"]`` truthy) additionally consults ``stream_timeout`` at each
    level before the non-streaming key, matching ``Router._get_stream_timeout`` on the
    completion route: kwargs stream_timeout -> kwargs timeout/request_timeout ->
    litellm_params stream_timeout -> litellm_params timeout/request_timeout ->
    router_stream_timeout -> router_timeout -> pass_through_request_timeout -> 600s.
    """
    kwargs = kwargs or {}
    litellm_params = litellm_params or {}
    is_stream: Final[bool] = bool(kwargs.get("stream", False))

    keys: Final[tuple[str, ...]] = (
        ("stream_timeout", "timeout", "request_timeout") if is_stream else ("timeout", "request_timeout")
    )
    for source in (kwargs, litellm_params):
        for key in keys:
            val = source.get(key)
            if val is not None:
                return float(val)

    if is_stream and router_stream_timeout is not None:
        return float(router_stream_timeout)
    if router_timeout is not None:
        return float(router_timeout)

    return resolve_pass_through_request_timeout()
