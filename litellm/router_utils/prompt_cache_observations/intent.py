from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.integrations.anthropic_cache_control_hook import (
    GATEWAY_INJECTED_CACHE_METADATA_KEY,
    GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT,
)

_METADATA_BUCKETS: Final = ("litellm_metadata", "metadata")
_OBJECTS: Final = TypeAdapter(tuple[object, ...])


def _client_body(litellm_params: Mapping[str, object]) -> Mapping[str, object] | None:
    return as_str_mapping((as_str_mapping(litellm_params.get("proxy_server_request")) or {}).get("body"))


def _listed(value: object) -> tuple[object, ...] | None:
    return _OBJECTS.validate_python(value) if isinstance(value, list) else None


def _client_marks_cache(body: Mapping[str, object]) -> bool:
    messages: Final = _listed(body.get("messages", body.get("input")))
    in_messages: Final = AnthropicCacheControlHook.count_request_cache_breakpoints(messages or (), body.get("system"))
    outside_messages: Final = AnthropicCacheControlHook.count_external_cache_breakpoints(
        _listed(body.get("tools")), body.get("cache_control"), body
    )
    return in_messages + outside_messages > 0


def _gateway_marked(litellm_params: Mapping[str, object], deployment_id: str) -> bool:
    """`AnthropicCacheControlHook.record_gateway_injection` names the deployment it injected for, so a marker
    left by an earlier attempt on another deployment of the same request does not count for this one."""
    buckets: Final = (as_str_mapping(litellm_params.get(name)) for name in _METADATA_BUCKETS)
    marks: Final = (bucket.get(GATEWAY_INJECTED_CACHE_METADATA_KEY) for bucket in buckets if bucket is not None)
    return any(
        mark in (GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT, deployment_id) for mark in marks if isinstance(mark, str)
    )


def cache_requested(litellm_params: Mapping[str, object], deployment_id: str) -> bool:
    """Caching asked for before any transformation: breakpoints in the body the client sent the proxy, or
    breakpoints the gateway injected for this attempt's deployment."""
    body: Final = _client_body(litellm_params)
    return (body is not None and _client_marks_cache(body)) or _gateway_marked(litellm_params, deployment_id)
