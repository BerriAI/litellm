"""Detectors for Python-only Anthropic Messages behavior."""

from __future__ import annotations

import litellm
from litellm.rust_bridge.detectors import overridden, present
from litellm.rust_bridge.public_call import NativeCall


def streamed_mock_response(request: NativeCall) -> bool:
    return present(request, "mock_response") and request.resolved.get("stream") is True


def pre_request_hook(_request: NativeCall) -> bool:
    """The async Messages handler runs callbacks that can rewrite the request."""
    return overridden("async_pre_request_hook", litellm.callbacks)


def websearch_interception(_request: NativeCall) -> bool:
    """The async Messages handler can answer a web-search request without an LLM call."""
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger

    return any(isinstance(callback, WebSearchInterceptionLogger) for callback in litellm.callbacks)


def safeguards_classifier(request: NativeCall) -> bool:
    """Python's proxy classifier evaluates dangerous tool use for non-Claude models."""
    from litellm.llms.anthropic.messages.safeguards import native_route_classifies

    return native_route_classifies(str(request.resolved["model"]), request.resolved.get("safeguards"))
