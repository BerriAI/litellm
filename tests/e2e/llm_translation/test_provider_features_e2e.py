"""Live e2e for model-specific request features: service_tier and prompt_cache_retention.

Each case asserts the feature took effect, not just a 200.

service_tier is an OpenAI concept. The proxy forwards it and the provider echoes
the tier back on the response, so sending a non-default tier ("priority") and
reading it back off ``service_tier`` proves the param was honored end to end;
litellm's own default injection (and service_tier="auto") both report "default",
so a "priority" echo can only come from the request being forwarded. "flex" is
avoided here because it is capacity-constrained and returns a transient 429 when
flex resources are unavailable. Bedrock and Vertex do not accept service_tier, so
that cell is OpenAI-only by design.

prompt_cache_retention is judged on the body OpenAI receives, captured by a live
provider edge, not on the reply: OpenAI echoes "24h" on gpt-5.5 even when the
request never carried the param, so an echo would pass with the param dropped.

Prompt caching hits and misses live in test_cache_control.py.
"""

from __future__ import annotations

import queue
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import pytest
from pydantic import BaseModel

from e2e_config import PROVIDER_EDGE_ADVERTISE_HOST, PROVIDER_EDGE_BIND_HOST, unique_marker
from e2e_http import unwrap
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from passthrough_client import PassthroughClient
from provider_edge import LiveEdge, start_provider_edge
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

SERVICE_TIER = "priority"
OPENAI_BACKEND: Final = "openai/gpt-5.5"
OPENAI_RESPONSES_BRIDGE_BACKEND: Final = "openai/responses/gpt-5.5"
PROMPT_CACHE_RETENTION: Final = "24h"


class TestServiceTier:
    @pytest.mark.covers(
        "llm.chat_completions.openai.service_tier.nonstream.works", exercised_on=[]
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.OPENAI,),
            models=(OPENAI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_openai_service_tier_is_echoed(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-service-tier-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="reply with one word")],
                    max_tokens=64,
                    service_tier=SERVICE_TIER,
                ),
            )
        )
        assert response.service_tier == SERVICE_TIER, (
            f"service_tier not honored: sent {SERVICE_TIER!r}, response reported "
            f"{response.service_tier!r} ({response})"
        )


class _OutboundRetention(BaseModel, frozen=True):
    prompt_cache_retention: str | None = None


@dataclass(frozen=True, slots=True)
class _OpenAIBodyCapture:
    """The raw bodies sent to one OpenAI path, seen by a live edge in front of
    api.openai.com. Pass ``observe`` as the edge's ``observe_request``."""

    path: str
    _bodies: queue.SimpleQueue[bytes] = field(default_factory=queue.SimpleQueue)

    def observe(self, url: str, headers: Mapping[str, str], body: bytes | None) -> None:
        if body is not None and url.split("?")[0].endswith(self.path):
            self._bodies.put(body)

    def retentions(self) -> tuple[str | None, ...]:
        bodies: Final = tuple(self._bodies.get_nowait() for _ in range(self._bodies.qsize()))
        return tuple(_OutboundRetention.model_validate_json(body).prompt_cache_retention for body in bodies)


def _openai_edge_api_base(resources: ResourceManager, capture: _OpenAIBodyCapture, *, reachable_from_proxy: bool) -> str:
    """The proxy may run in a container, so its edge binds where the proxy can
    reach it; an in-process SDK call reaches the edge on loopback."""
    edge: Final = (
        start_provider_edge(
            LiveEdge(observe_request=capture.observe),
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
        )
        if reachable_from_proxy
        else start_provider_edge(LiveEdge(observe_request=capture.observe))
    )
    resources.defer(edge.shutdown)
    return f"{edge.edge.api_base('openai')}/v1"


def _retention_backends() -> tuple[object, ...]:
    """gpt-5.5 straight to /v1/chat/completions, and the openai/responses/ bridge
    (chat completions in, Responses out) to /v1/responses."""
    return (
        pytest.param(
            OPENAI_BACKEND,
            "/chat/completions",
            marks=(
                pytest.mark.covers("llm.chat_completions.openai.prompt_cache_24h.nonstream.forwards_retention"),
                meta(
                    Subject(
                        domain=Domain.LLM_TRANSLATION,
                        route=Route.CHAT_COMPLETIONS,
                        providers=(Provider.OPENAI,),
                        models=(OPENAI_BACKEND,),
                        mode=Mode.NONSTREAM,
                    )
                ),
            ),
            id="chat_completions",
        ),
        pytest.param(
            OPENAI_RESPONSES_BRIDGE_BACKEND,
            "/responses",
            marks=(
                pytest.mark.covers("llm.chat_completions.openai.prompt_cache_24h.nonstream.bridge_forwards_retention"),
                meta(
                    Subject(
                        domain=Domain.LLM_TRANSLATION,
                        route=Route.CHAT_COMPLETIONS,
                        providers=(Provider.OPENAI,),
                        models=(OPENAI_RESPONSES_BRIDGE_BACKEND,),
                        mode=Mode.NONSTREAM,
                    )
                ),
            ),
            id="responses_bridge",
        ),
    )


class TestPromptCacheRetention:
    @pytest.mark.provider_edge_host
    @pytest.mark.parametrize(("backend", "openai_path"), _retention_backends())
    def test_proxy_chat_completions_forwards_prompt_cache_retention_to_openai(
        self, proxy: ProxyClient, resources: ResourceManager, backend: str, openai_path: str
    ) -> None:
        capture: Final = _OpenAIBodyCapture(openai_path)
        api_base: Final = _openai_edge_api_base(resources, capture, reachable_from_proxy=True)
        model: Final = f"e2e-cache-retention-{unique_marker()}"
        model_id: Final = proxy.create_model(
            model, LiteLLMParamsBody(model=backend, api_key="os.environ/OPENAI_API_KEY", api_base=api_base)
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        unwrap(
            proxy.chat(
                resources.key(),
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="reply with one word")],
                    prompt_cache_retention=PROMPT_CACHE_RETENTION,
                ),
            )
        )

        assert capture.retentions() == (PROMPT_CACHE_RETENTION,), (
            f"one {openai_path} call to OpenAI must carry prompt_cache_retention={PROMPT_CACHE_RETENTION!r}"
        )

    @pytest.mark.parametrize(("backend", "openai_path"), _retention_backends())
    def test_sdk_completion_forwards_prompt_cache_retention_to_openai(
        self, resources: ResourceManager, backend: str, openai_path: str
    ) -> None:
        import litellm

        capture: Final = _OpenAIBodyCapture(openai_path)
        api_base: Final = _openai_edge_api_base(resources, capture, reachable_from_proxy=False)

        litellm.completion(
            model=backend,
            messages=[{"role": "user", "content": "reply with one word"}],
            api_base=api_base,
            prompt_cache_retention=PROMPT_CACHE_RETENTION,
        )

        assert capture.retentions() == (PROMPT_CACHE_RETENTION,), (
            f"one {openai_path} call to OpenAI must carry prompt_cache_retention={PROMPT_CACHE_RETENTION!r}"
        )

    @pytest.mark.parametrize(("backend", "openai_path"), _retention_backends())
    def test_router_completion_forwards_prompt_cache_retention_to_openai(
        self, resources: ResourceManager, backend: str, openai_path: str
    ) -> None:
        import litellm

        capture: Final = _OpenAIBodyCapture(openai_path)
        api_base: Final = _openai_edge_api_base(resources, capture, reachable_from_proxy=False)
        router: Final = litellm.Router(
            model_list=[{"model_name": "cache-retention", "litellm_params": {"model": backend, "api_base": api_base}}]
        )

        router.completion(  # pyright: ignore[reportUnknownMemberType]  # Router.completion forwards an untyped **kwargs to litellm.completion
            model="cache-retention",
            messages=[{"role": "user", "content": "reply with one word"}],
            prompt_cache_retention=PROMPT_CACHE_RETENTION,
        )

        assert capture.retentions() == (PROMPT_CACHE_RETENTION,), (
            f"one {openai_path} call to OpenAI must carry prompt_cache_retention={PROMPT_CACHE_RETENTION!r}"
        )
