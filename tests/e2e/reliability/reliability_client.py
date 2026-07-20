"""Client for the reliability e2e suite: the shared ProxyClient plus the mock
deployment creators and the per-request-override chat call the tests drive.

The whole suite is mock-based: deployments are registered via /model/new with a
`mock_response` (a litellm exception name raises that error; any other string is
returned as the completion content) or `mock_timeout`. Fallbacks and retries are
driven per request through a `router_settings_override` in the /chat/completions
body, so a single long-lived proxy serves every reliability behavior without any
static per-behavior config.
"""

from __future__ import annotations

from dataclasses import dataclass

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

MOCK_MODEL = "openai/gpt-4o-mini"


@dataclass(frozen=True, slots=True)
class ReliabilityClient:
    proxy: ProxyClient

    def create_mock(self, name: str, mock_response: str) -> str:
        """Register a mock deployment under `name`: `mock_response` that names a
        litellm exception raises it, any other string is returned as the content."""
        return self.proxy.create_model(name, LiteLLMParamsBody(model=MOCK_MODEL, mock_response=mock_response))

    def create_timeout_deployment(self, name: str, timeout: float = 1.0) -> str:
        """Register a deployment whose calls always time out after `timeout`s."""
        return self.proxy.create_model(name, LiteLLMParamsBody(model=MOCK_MODEL, mock_timeout=True, timeout=timeout))

    def chat_override(
        self,
        key: str,
        model: str,
        content: str,
        override: RouterSettingsOverride | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        """POST /chat/completions with an optional per-request router_settings_override,
        returning the raw outcome so tests read status, body, and reliability headers."""
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
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
