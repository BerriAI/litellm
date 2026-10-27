"""Detectors for Python-only Anthropic Messages behavior."""

from __future__ import annotations

from typing import Final

import litellm
from litellm.llms.opencode.common_utils import is_opencode_api_base
from litellm.rust_bridge.detectors import api_base, overridden, present, streaming
from litellm.rust_bridge.public_call import NativeCall, optional_mapping, optional_sequence, optional_str


def provider_affinity(request: NativeCall) -> bool:
    return is_opencode_api_base(api_base(request))


def anthropic_workload_identity(request: NativeCall) -> bool:
    """Python mints an Anthropic federation token when no static credential is configured."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    from litellm.secret_managers.main import get_secret_str

    def configured(param: str, env: str) -> bool:
        return present(request, param) or get_secret_str(env) is not None

    return (
        AnthropicModelInfo.get_api_key(optional_str(request.resolved.get("api_key"))) is None
        and AnthropicModelInfo.get_auth_token() is None
        and configured("anthropic_federation_rule_id", "ANTHROPIC_FEDERATION_RULE_ID")
        and configured("anthropic_organization_id", "ANTHROPIC_ORGANIZATION_ID")
    )


_THINKING_BLOCKS: Final = frozenset({"thinking", "redacted_thinking"})


def _replays_thinking(message: object) -> bool:
    content: Final = optional_sequence((optional_mapping(message) or {}).get("content")) or ()
    return any((optional_mapping(block) or {}).get("type") in _THINKING_BLOCKS for block in content)


def thinking_signature_retry(request: NativeCall) -> bool:
    """Python may strip replayed thinking blocks and retry after a rejected signature."""
    return any(_replays_thinking(message) for message in optional_sequence(request.resolved.get("messages")) or ())


def stream_cost_injection(request: NativeCall) -> bool:
    """Python adds response cost to streaming usage when the global setting is truthy."""
    return streaming(request) and bool(litellm.include_cost_in_streaming_usage)


def pre_request_hook(_request: NativeCall) -> bool:
    """The async Messages handler runs callbacks that can rewrite the request."""
    return overridden("async_pre_request_hook", litellm.callbacks)


def websearch_interception(_request: NativeCall) -> bool:
    """The async Messages handler can answer a web-search request without an LLM call."""
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger

    return any(isinstance(callback, WebSearchInterceptionLogger) for callback in litellm.callbacks)


def _resolved_provider(request: NativeCall) -> str | None:
    from litellm.exceptions import BadRequestError
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    provider: Final = optional_str(request.resolved.get("custom_llm_provider"))
    try:
        return get_llm_provider(str(request.resolved["model"]), provider)[1]
    except BadRequestError:
        return provider


def intercepted(request: NativeCall) -> bool:
    """The async Messages handler delegates matching tools to a registered interceptor."""
    from litellm.llms.anthropic.messages.native.interceptors import get_messages_interceptors

    tools: Final = optional_sequence(request.resolved.get("tools"))
    if not tools:
        return False
    mappings: Final = tuple(mapping for mapping in map(optional_mapping, tools) if mapping is not None)
    provider: Final = _resolved_provider(request)
    return any(interceptor.can_handle(mappings, provider) for interceptor in get_messages_interceptors())


def safeguards_classifier(request: NativeCall) -> bool:
    """Python's proxy classifier evaluates dangerous tool use for non-Claude models."""
    from litellm.llms.anthropic.messages.safeguards import native_route_classifies

    return native_route_classifies(str(request.resolved["model"]), request.resolved.get("safeguards"))
