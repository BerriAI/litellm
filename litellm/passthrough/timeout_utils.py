import sys
from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict

DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS: Final = 600.0


class _TimeoutFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    stream: bool = False
    stream_timeout: float | None = None
    timeout: float | None = None
    request_timeout: float | None = None


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
    kwargs: Mapping[str, object] | None = None,
    litellm_params: Mapping[str, object] | None = None,
    router_timeout: float | str | None = None,
    router_stream_timeout: float | str | None = None,
) -> float:
    """
    Resolve upstream httpx timeout for SDK native passthrough (e.g. Bedrock /converse,
    Anthropic /v1/messages).

    Non-streaming precedence: kwargs timeout/request_timeout -> litellm_params
    timeout/request_timeout -> router_timeout -> general_settings.pass_through_request_timeout
    -> 600s default.

    Streaming (``kwargs["stream"]`` truthy) resolves ``stream_timeout`` at every level before
    any generic timeout, matching ``Router._get_stream_timeout`` on the completion route:
    kwargs stream_timeout -> litellm_params stream_timeout -> router_stream_timeout, then the
    non-streaming chain above.
    """
    request: Final = _TimeoutFields.model_validate(kwargs or {})
    deployment: Final = _TimeoutFields.model_validate(litellm_params or {})
    stream_candidates: Final = (
        (request.stream_timeout, deployment.stream_timeout, router_stream_timeout) if request.stream else ()
    )
    candidates: Final = (
        *stream_candidates,
        request.timeout,
        request.request_timeout,
        deployment.timeout,
        deployment.request_timeout,
        router_timeout,
    )
    resolved: Final = next((float(val) for val in candidates if val is not None), None)
    return resolved if resolved is not None else resolve_pass_through_request_timeout()
