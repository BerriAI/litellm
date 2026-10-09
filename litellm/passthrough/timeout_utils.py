import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.request_timeout_resolver import get_configured_request_timeout

DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS: Final = 600.0

_SECONDS: Final = TypeAdapter(float)
_NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


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
    timeout/request_timeout -> router_timeout -> litellm.request_timeout (litellm_settings.request_timeout,
    when explicitly set) -> general_settings.pass_through_request_timeout -> 600s default.

    Streaming (``kwargs["stream"]`` truthy) resolves ``stream_timeout`` at every level before
    any generic timeout, matching ``Router._get_stream_timeout`` on the completion route:
    kwargs stream_timeout -> litellm_params stream_timeout -> router_stream_timeout, then the
    non-streaming chain above.

    Only the first set value is validated as seconds, so a value in a lower-precedence
    field never fails the call.
    """
    request: Final = kwargs if kwargs is not None else _NO_PARAMS
    deployment: Final = litellm_params if litellm_params is not None else _NO_PARAMS
    stream_candidates: Final = (
        (request.get("stream_timeout"), deployment.get("stream_timeout"), router_stream_timeout)
        if request.get("stream")
        else ()
    )
    candidates: Final = (
        *stream_candidates,
        request.get("timeout"),
        request.get("request_timeout"),
        deployment.get("timeout"),
        deployment.get("request_timeout"),
        router_timeout,
        get_configured_request_timeout(),
    )
    winner: Final = next((val for val in candidates if val is not None), None)
    return resolve_pass_through_request_timeout() if winner is None else _SECONDS.validate_python(winner)
