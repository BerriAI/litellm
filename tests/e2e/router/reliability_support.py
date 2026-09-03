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
    ModelInfoBody,
    ModelNewBody,
    ReliabilityChatBody,
    RouterSettingsOverride,
)

REAL_MODEL = "openai/gpt-5.5"
REAL_KEY = "os.environ/OPENAI_API_KEY"

# The smallest-context chat model OpenAI still serves (16385 tokens). A prompt
# past that limit comes back as a real `context_length_exceeded` 400, which is
# what litellm maps to ContextWindowExceededError.
SMALL_CONTEXT_MODEL = "openai/gpt-3.5-turbo"
SMALL_CONTEXT_LIMIT_TOKENS = 16385


def oversized_prompt(marker: str) -> str:
    """A prompt comfortably past SMALL_CONTEXT_MODEL's context limit, so the
    provider refuses it on length rather than answering a truncated version."""
    return f"{marker} " + ("token " * (SMALL_CONTEXT_LIMIT_TOKENS + 4000))


def create_bad_base_deployment(proxy: ProxyClient, name: str) -> str:
    """Register a deployment pointing at an unreachable base, so every call to it
    fails with a real connection error the fallback can reroute around."""
    return proxy.create_model(
        name, LiteLLMParamsBody(model=REAL_MODEL, api_key=REAL_KEY, api_base="http://127.0.0.1:9/v1")
    )


def create_timeout_deployment(proxy: ProxyClient, name: str) -> str:
    """Register a deployment with a 1ms deadline the real backend always exceeds."""
    return proxy.create_model(name, LiteLLMParamsBody(model=REAL_MODEL, api_key=REAL_KEY, timeout=0.001))


def create_small_context_deployment(proxy: ProxyClient, name: str) -> str:
    """Register a deployment on the smallest-context model OpenAI still serves, so an
    oversized prompt earns a real context-window refusal from the provider."""
    return proxy.create_model(name, LiteLLMParamsBody(model=SMALL_CONTEXT_MODEL, api_key=REAL_KEY))


def create_always_timing_out_deployment(proxy: ProxyClient, name: str) -> str:
    """The always-picked half of a retry pair: a 1ms deadline the backend always
    exceeds, all of the model group's shuffle weight, and a cooldown policy that
    benches it on its first Timeout so the retry cannot land on it again."""
    return proxy.register_model(
        ModelNewBody(
            model_name=name,
            litellm_params=LiteLLMParamsBody(model=REAL_MODEL, api_key=REAL_KEY, timeout=0.001, weight=1),
            model_info=ModelInfoBody(allowed_fails_policy={"TimeoutErrorAllowedFails": 0}),
        )
    )


def create_zero_weight_backup_deployment(proxy: ProxyClient, name: str) -> str:
    """The other half of a retry pair: healthy, but weight 0, so the weighted shuffle
    never opens on it. It is reachable only once its sibling is benched and the
    weighted pick falls through to a uniform one over what is left."""
    return proxy.register_model(
        ModelNewBody(
            model_name=name,
            litellm_params=LiteLLMParamsBody(model=REAL_MODEL, api_key=REAL_KEY, weight=0),
            model_info=ModelInfoBody(),
        )
    )


def chat_override(
    proxy: ProxyClient,
    key: str,
    model: str,
    content: str,
    override: RouterSettingsOverride | None = None,
    stream: bool = False,
    cache: dict[str, bool] | None = {"no-cache": True},
) -> StreamingResponse:
    """POST /chat/completions with an optional per-request router_settings_override,
    returning the raw outcome so tests read status, body, and reliability headers."""
    return proxy.transport.send(
        "/chat/completions",
        headers=proxy.transport.bearer(key),
        json=ReliabilityChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=content)],
            max_tokens=512,
            stream=stream,
            router_settings_override=override,
            cache=cache,
        ),
        stream=stream,
    )


def _parsed(resp: StreamingResponse) -> ChatResponse | None:
    try:
        return ChatResponse.model_validate_json(resp.body)
    except ValidationError:
        return None


def content_of(resp: StreamingResponse) -> str | None:
    """The assistant message content of a successful chat response, or None when the
    body is not a success shape (an error body, or an elided streamed body)."""
    parsed = _parsed(resp)
    if parsed is None or not parsed.choices:
        return None
    message = parsed.choices[0].message
    return message.content if message is not None else None


def finish_reason_of(resp: StreamingResponse) -> str | None:
    parsed = _parsed(resp)
    if parsed is None or not parsed.choices:
        return None
    return parsed.choices[0].finish_reason


def completion_tokens_of(resp: StreamingResponse) -> int | None:
    parsed = _parsed(resp)
    if parsed is None or parsed.usage is None:
        return None
    return parsed.usage.completion_tokens


def reasoning_tokens_of(resp: StreamingResponse) -> int | None:
    parsed = _parsed(resp)
    if parsed is None or parsed.usage is None or parsed.usage.completion_tokens_details is None:
        return None
    return parsed.usage.completion_tokens_details.reasoning_tokens
