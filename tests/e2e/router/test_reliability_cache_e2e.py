"""An exact cache hit preserves the full choices and usage without another provider call.

Response IDs, creation timestamps and proxy headers are transport metadata;
compare every field within choices and usage, including provider extensions.
"""

from __future__ import annotations

from typing import Final

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import (
    FIXTURE_DIR,
    FIXTURE_MODE_RAW,
    PROVIDER_EDGE_ADVERTISE_HOST,
    PROVIDER_EDGE_BIND_HOST,
    REQUEST_TIMEOUT,
    unique_marker,
)
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody
from provider_edge import ProviderRequestObservation, observed_provider_edge
from pydantic import BaseModel, JsonValue

pytestmark = [pytest.mark.e2e, pytest.mark.replayable]


class _CacheChatBody(ChatBody):
    ttl: int = 600


class _CachedAnswer(BaseModel):
    model: str
    choices: tuple[dict[str, JsonValue], ...]
    usage: dict[str, JsonValue]


class TestReliabilityCache:
    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    def test_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-{marker}"
        prompt: Final = f"Reply with a short sentence about a blue lantern. Request marker: {marker}"
        observation: Final = ProviderRequestObservation(marker)

        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(
                model,
                LiteLLMParamsBody(
                    model="openai/gpt-5.6",
                    api_key="os.environ/OPENAI_API_KEY",
                    api_base=f"{edge.api_base('openai')}/v1",
                ),
            )
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)],
                max_completion_tokens=512,
                reasoning_effort="none",
                cache=None,
            )
            first: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            assert first.status_code == 200, f"first call should succeed, got {first.status_code}: {first.body[:300]}"
            assert "x-litellm-cache-key" not in first.headers, "first call must be a cache miss"
            answer: Final = ChatResponse.model_validate_json(first.body)
            assert len(answer.choices) == 1
            choice: Final = answer.choices[0]
            assert choice.message is not None and choice.message.role == "assistant"
            assert choice.message.content is not None and choice.message.content.strip(), "first answer is empty"
            assert choice.finish_reason == "stop"
            assert answer.usage is not None
            assert answer.usage.prompt_tokens is not None and answer.usage.prompt_tokens > 0
            assert answer.usage.completion_tokens is not None and answer.usage.completion_tokens > 0
            assert answer.usage.total_tokens == answer.usage.prompt_tokens + answer.usage.completion_tokens
            assert observation.count == 1, "first miss must invoke the provider exactly once"

            second: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            assert second.status_code == 200, f"second call should succeed, got {second.status_code}: {second.body[:300]}"
            assert second.headers.get("x-litellm-cache-key"), "identical request must hit the response cache"
            assert _CachedAnswer.model_validate_json(second.body) == _CachedAnswer.model_validate_json(first.body), (
                "cache hit changed the answer, finish reason or usage"
            )
            assert observation.count == 1, "two successful requests must invoke the provider exactly once"
