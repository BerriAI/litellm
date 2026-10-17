"""Live provider checks for embeddings and chat fallback from an explicitly restricted model."""

from __future__ import annotations

import dataclasses
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, ModelInfoBody, ModelMode, ModelNewBody
from openai import OpenAI
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

EMBEDDING_BACKEND: Final = "openai/text-embedding-3-small"
CHAT_FALLBACK_BACKEND: Final = "openai/gpt-5.4-nano"
EMBEDDINGS_ONLY: Final = ("/v1/embeddings",)
PROMPT: Final = "Say this is a test!"


def _subject(route: Route) -> Subject:
    return Subject(
        domain=Domain.LLM_TRANSLATION,
        route=route,
        providers=(Provider.OPENAI,),
        models=(EMBEDDING_BACKEND,),
        mode=Mode.NONSTREAM,
    )


def _register(
    proxy: ProxyClient,
    resources: ResourceManager,
    prefix: str,
    supported_endpoints: tuple[str, ...] | None,
    backend: str = EMBEDDING_BACKEND,
    mode: ModelMode = "embedding",
) -> str:
    model: Final = f"{prefix}-{unique_marker()}"
    model_id: Final = proxy.register_model(
        ModelNewBody(
            model_name=model,
            litellm_params=LiteLLMParamsBody(model=backend, api_key="os.environ/OPENAI_API_KEY"),
            model_info=ModelInfoBody(
                enforce_supported_endpoints=supported_endpoints is not None,
                mode=mode,
                supported_endpoints=None if supported_endpoints is None else list(supported_endpoints),
            ),
        )
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _openai_at(sdk: SdkClients, api_prefix: str, key: str) -> OpenAI:
    return dataclasses.replace(sdk, base_url=f"{sdk.base_url}{api_prefix}").openai(key)


class TestEmbeddingOnlySupportedEndpoints:
    @pytest.mark.covers("llm.embeddings.openai.supported_endpoints_gate.nonstream.works")
    @meta(_subject(Route.EMBEDDINGS))
    @pytest.mark.parametrize(
        ("prefix", "supported_endpoints"),
        (
            pytest.param("e2e-embed-only", EMBEDDINGS_ONLY, id="embed-only"),
            pytest.param("e2e-embed-control", None, id="control"),
        ),
    )
    def test_embeddings_return_a_vector(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        prefix: str,
        supported_endpoints: tuple[str, ...] | None,
    ) -> None:
        model: Final = _register(proxy, resources, prefix, supported_endpoints)
        embeddings: Final = sdk.openai(resources.key()).embeddings.create(
            model=model, input=PROMPT, extra_body=NO_PROXY_CACHE
        )
        assert embeddings.data, f"/embeddings returned no data for {model}: {embeddings!r}"
        assert any(component != 0.0 for component in embeddings.data[0].embedding), (
            f"/embeddings returned an empty or all-zero vector for {model}"
        )

    @pytest.mark.covers("llm.embeddings.openai.supported_endpoints_gate.nonstream.works")
    @meta(_subject(Route.CHAT_COMPLETIONS))
    def test_chat_completions_with_a_fallback_are_answered_by_the_fallback(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, "e2e-embed-only", EMBEDDINGS_ONLY)
        fallback: Final = _register(proxy, resources, "e2e-chat-fallback", None, CHAT_FALLBACK_BACKEND, "chat")
        completion: Final = _openai_at(sdk, "/v1", resources.key()).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            extra_body={**NO_PROXY_CACHE, "fallbacks": [fallback]},
        )
        assert completion.choices and completion.choices[0].message.content, (
            f"the fallback {fallback} should answer a chat call to {model}, got {completion!r}"
        )
