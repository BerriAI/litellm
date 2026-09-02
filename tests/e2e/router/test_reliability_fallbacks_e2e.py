"""Live e2e: per-request fallbacks reroute a failing deployment's traffic to a
healthy one.

Each test registers a primary deployment that fails (an unreachable base URL, or
a 1ms deadline) and calls it with a `router_settings_override` mapping it to the
real `gpt-5.5`. The proof the fallback fired is twofold: the response is a
completion from `gpt-5.5`, and the proxy reports at least one attempted fallback
in the x-litellm-attempted-fallbacks header. Empty content is accepted only when
`finish_reason == "length"` and the response billed completion tokens, since
gpt-5.5 counts reasoning against max_tokens and can consume the whole budget
before emitting any text; a fallback that produced nothing at all still fails.

The context-window case is a different reroute from a plain failure: the provider
refuses the prompt on length, and `context_window_fallbacks` is the setting that
reroutes it, not `fallbacks`.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import RouterSettingsOverride
from reliability_support import (
    chat_override,
    completion_tokens_of,
    content_of,
    create_bad_base_deployment,
    create_small_context_deployment,
    create_timeout_deployment,
    finish_reason_of,
    oversized_prompt,
    reasoning_tokens_of,
)

pytestmark = pytest.mark.e2e


def _assert_served_by_fallback(resp: StreamingResponse) -> None:
    assert resp.status_code == 200, f"expected 200 after fallback, got {resp.status_code}: {resp.body[:300]}"
    content = content_of(resp)
    finish_reason = finish_reason_of(resp)
    completion_tokens = completion_tokens_of(resp) or 0
    reasoning_tokens = reasoning_tokens_of(resp) or 0
    assert isinstance(content, str), (
        f"the gpt-5.5 fallback should have returned a completion body, got content {content!r} "
        f"(body={resp.body[:300]})"
    )
    assert content or (finish_reason == "length" and completion_tokens > 0), (
        f"the gpt-5.5 fallback returned empty content with finish_reason={finish_reason!r}, "
        f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens}; empty "
        f"content is only acceptable when the budget was spent on non-visible reasoning "
        f"(body={resp.body[:300]})"
    )
    attempted = resp.headers.get("x-litellm-attempted-fallbacks")
    assert attempted is not None, "response is missing the x-litellm-attempted-fallbacks header"
    assert int(attempted) >= 1, f"x-litellm-attempted-fallbacks should be >= 1, got {attempted!r}"


class TestReliabilityFallbacks:
    @pytest.mark.covers("reliability.fallback.5xx.routes_to_fallback")
    def test_5xx_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-fail-{unique_marker()}"
        model_id = create_bad_base_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy, scoped_key, primary, f"say hi {unique_marker()}",
            override=RouterSettingsOverride(fallbacks=[{primary: ["gpt-5.5"]}]),
        )
        _assert_served_by_fallback(resp)

    @pytest.mark.covers("reliability.fallback.timeout.routes_to_fallback")
    def test_timeout_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-tofail-{unique_marker()}"
        model_id = create_timeout_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy, scoped_key, primary, f"say hi {unique_marker()}",
            override=RouterSettingsOverride(fallbacks=[{primary: ["gpt-5.5"]}]),
        )
        _assert_served_by_fallback(resp)

    @pytest.mark.covers("reliability.fallback.context_window.routes_to_fallback")
    def test_context_window_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-ctxfail-{unique_marker()}"
        model_id = create_small_context_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy, scoped_key, primary, oversized_prompt(unique_marker()),
            override=RouterSettingsOverride(context_window_fallbacks=[{primary: ["gpt-5.5"]}]),
        )
        _assert_served_by_fallback(resp)
