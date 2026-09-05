"""Live e2e: a conversation that wrote a provider-side prompt cache keeps landing
on the deployment holding that cache.

The group starts as a single Anthropic deployment. The first call carries a system
turn long enough to clear the provider's cache floor, marked `cache_control`, and
the provider reports it wrote the cache. Then a second deployment on another
provider joins the group with twenty times the shuffle weight, and every follow-up
with the same system turn still lands on the Anthropic deployment and reads the
cache back, which is the affinity the router's `prompt_caching` pre-call check
provides: it pins a cached conversation to its deployment before the shuffle runs.

The proxy has to run with `router_settings.optional_pre_call_checks:
["prompt_caching"]` for that check to exist, so the test reads GET /router/settings
first and fails, naming the missing setting, rather than reporting a routing bug.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import ChatMessage, LiteLLMParamsBody, ModelInfoBody, ModelNewBody
from reliability_support import (
    REAL_KEY,
    REAL_MODEL,
    cached_system_turn,
    chat_turns_override,
    create_caching_deployment,
    model_id_of,
    usage_of,
)

pytestmark = pytest.mark.e2e

FOLLOW_UPS = 3


class TestReliabilityPromptCachingAffinity:
    @pytest.mark.covers("reliability.cache.prompt_caching_model_select.returns_cached")
    def test_cached_conversation_stays_on_deployment_holding_its_cache(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        checks = client.proxy.router_settings().optional_pre_call_checks
        assert "prompt_caching" in checks, (
            f"the proxy runs with optional_pre_call_checks={checks}; this test needs "
            'router_settings.optional_pre_call_checks: ["prompt_caching"] in its config'
        )

        group = f"reliability-cache-{unique_marker()}"
        cached = create_caching_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(cached))
        system = cached_system_turn(unique_marker())

        first = chat_turns_override(
            client.proxy, scoped_key, group, [system, ChatMessage(role="user", content=f"say hi {unique_marker()}")]
        )
        assert first.status_code == 200, f"the cache-writing call failed with {first.status_code}: {first.body[:300]}"
        assert model_id_of(first) == cached
        written = usage_of(first)
        assert written is not None and (written.cache_creation_input_tokens or 0) > 0, (
            f"the provider should have written the prompt cache on the first call, usage={written}"
        )

        heavyweight = client.proxy.register_model(
            ModelNewBody(
                model_name=group,
                litellm_params=LiteLLMParamsBody(model=REAL_MODEL, api_key=REAL_KEY, weight=20),
                model_info=ModelInfoBody(),
            )
        )
        resources.defer(lambda: client.proxy.delete_model(heavyweight))

        for turn in range(FOLLOW_UPS):
            follow_up = chat_turns_override(
                client.proxy,
                scoped_key,
                group,
                [system, ChatMessage(role="user", content=f"follow-up {turn} {unique_marker()}")],
            )
            assert follow_up.status_code == 200, (
                f"follow-up {turn} failed with {follow_up.status_code}: {follow_up.body[:300]}"
            )
            assert model_id_of(follow_up) == cached, (
                f"follow-up {turn} landed on {model_id_of(follow_up)!r} instead of the deployment holding the "
                f"cache ({cached}), even though the heavier-weighted newcomer holds no cache for this conversation"
            )
            read = usage_of(follow_up)
            assert read is not None and (read.cache_read_input_tokens or 0) > 0, (
                f"follow-up {turn} stayed on {cached} but read nothing from the cache, usage={read}"
            )
