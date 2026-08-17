"""Shared helpers for the reliability e2e tests (fallbacks, timeouts, cache).

These are plain functions over the router suite's shared ProxyClient, not a
fixture/client class: the tests reuse the router `client` fixture and pass
`client.proxy`. Fallbacks and timeouts are driven by REAL deployments that all
point at the real `openai/gpt-5.5`; a bad base URL yields a real connection
error and a 1ms deadline yields a real timeout, and each test wires the
reroute per request through a `router_settings_override` in the /chat/completions
body, so a single long-lived proxy serves every reliability behavior.
"""

from __future__ import annotations

from pydantic import ValidationError

from proxy_client import ProxyClient
from e2e_http import StreamingResponse
from models import (
    ChatMessage,
    ChatResponse,
    LiteLLMParamsBody,
    ReliabilityChatBody,
    RouterSettingsOverride,
)

REAL_MODEL = "openai/gpt-5.5"
REAL_KEY = "os.environ/OPENAI_API_KEY"


def create_bad_base_deployment(proxy: ProxyClient, name: str) -> str:
    """Register a deployment pointing at an unreachable base, so every call to it
    fails with a real connection error the fallback can reroute around."""
    return proxy.create_model(
        name, LiteLLMParamsBody(model=REAL_MODEL, api_key=REAL_KEY, api_base="http://127.0.0.1:9/v1")
    )


def create_timeout_deployment(proxy: ProxyClient, name: str) -> str:
    """Register a deployment with a 1ms deadline the real backend always exceeds."""
    return proxy.create_model(name, LiteLLMParamsBody(model=REAL_MODEL, api_key=REAL_KEY, timeout=0.001))


def chat_override(
    proxy: ProxyClient,
    key: str,
    model: str,
    content: str,
    override: RouterSettingsOverride | None = None,
    stream: bool = False,
) -> StreamingResponse:
    """POST /chat/completions with an optional per-request router_settings_override,
    returning the raw outcome so tests read status, body, and reliability headers."""
    return proxy.transport.send(
        "/chat/completions",
        headers=proxy.transport.bearer(key),
        json=ReliabilityChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=content)],
            max_tokens=16,
            stream=stream,
            router_settings_override=override,
        ),
        stream=stream,
    )


def content_of(resp: StreamingResponse) -> str | None:
    """The assistant message content of a successful chat response, or None when the
    body is not a success shape (an error body, or an elided streamed body)."""
    try:
        parsed = ChatResponse.model_validate_json(resp.body)
    except ValidationError:
        return None
    if not parsed.choices:
        return None
    message = parsed.choices[0].message
    return message.content if message is not None else None
