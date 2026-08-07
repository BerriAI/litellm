"""Live e2e: a prompt too large for a deployment's context window is rerouted to
the configured context-window fallback.

The primary is a real `openai/gpt-4` deployment, chosen for the smallest context
window OpenAI still serves (8k), so a prompt can overflow it for a few cents;
`gpt-5.5`, the model the rest of this suite runs on, has a million-token window
that no affordable prompt reaches. The overflow is therefore real: OpenAI itself
rejects the call with a context-length error, which is the trigger
`context_window_fallbacks` exists for.

The same oversized prompt is sent twice. The first call carries no fallback and
must come back as a context-window rejection, which is what proves the prompt
genuinely overflows the primary instead of the deployment being broken some other
way. The second names `gpt-5.5` as the context-window fallback and must come back
200, served by a different deployment than the primary, with the proxy reporting
the attempted fallback.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import ChatMessage, LiteLLMParamsBody, ReliabilityChatBody, RouterSettingsOverride
from proxy_client import ProxyClient
from reliability_support import REAL_KEY, REAL_MODEL

pytestmark = pytest.mark.e2e

SMALL_CONTEXT_MODEL = "openai/gpt-4"
LARGE_CONTEXT_GROUP = "gpt-5.5"
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "
FILLER_REPEATS = 900
ANSWER_TOKENS = 256


def _oversized_prompt() -> str:
    """Roughly ten thousand tokens of filler: past gpt-4's 8k window, and unique per
    call so the proxy's response cache never answers it from an earlier run."""
    return f"{FILLER_SENTENCE * FILLER_REPEATS} summarize the text above [{unique_marker()}]"


def _summarize(
    proxy: ProxyClient, key: str, group: str, override: RouterSettingsOverride | None = None
) -> StreamingResponse:
    """Ask `group` to summarize an oversized prompt, optionally with a fallback wired
    for this one request. The answer budget is generous on purpose: gpt-5.5 spends
    tokens on reasoning before it writes anything, and a budget it cannot finish in
    is its own error, which would muddy the context-window one under test."""
    return proxy.transport.send(
        "/chat/completions",
        headers=proxy.transport.bearer(key),
        json=ReliabilityChatBody(
            model=group,
            messages=[ChatMessage(role="user", content=_oversized_prompt())],
            max_tokens=ANSWER_TOKENS,
            router_settings_override=override,
        ),
    )


class TestReliabilityContextWindowFallback:
    @pytest.mark.covers("reliability.fallback.context_window.routes_to_fallback")
    def test_context_window_overflow_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-ctx-{unique_marker()}"
        model_id = client.proxy.create_model(
            primary, LiteLLMParamsBody(model=SMALL_CONTEXT_MODEL, api_key=REAL_KEY)
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))

        rejected = _summarize(client.proxy, scoped_key, primary)
        assert rejected.status_code == 400, (
            f"the oversized prompt should be rejected by {SMALL_CONTEXT_MODEL} with a 400, got "
            f"{rejected.status_code}: {rejected.body[:300]}"
        )
        assert "context length" in rejected.body.lower(), (
            f"the rejection should name the context length, got: {rejected.body[:300]}"
        )

        rerouted = _summarize(
            client.proxy,
            scoped_key,
            primary,
            RouterSettingsOverride(context_window_fallbacks=[{primary: [LARGE_CONTEXT_GROUP]}]),
        )
        assert rerouted.status_code == 200, (
            f"the same prompt should be served by the context-window fallback, got "
            f"{rerouted.status_code}: {rerouted.body[:300]}"
        )
        served = rerouted.headers.get("x-litellm-model-id")
        assert served is not None and served != model_id, (
            f"the fallback answer came from the deployment that could not fit the prompt "
            f"({model_id}); x-litellm-model-id={served!r}"
        )
        assert rerouted.headers.get("x-litellm-model-name") == REAL_MODEL, (
            f"the fallback should have been served by a {REAL_MODEL} deployment, but "
            f"x-litellm-model-name is {rerouted.headers.get('x-litellm-model-name')!r}"
        )
        attempted = rerouted.headers.get("x-litellm-attempted-fallbacks")
        assert attempted is not None and int(attempted) >= 1, (
            f"the proxy should report at least one attempted fallback, got {attempted!r}"
        )
