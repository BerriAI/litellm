"""Live e2e: an embedding model whose deployments declare supported_endpoints refuses chat-shaped calls.

Each test registers an OpenAI embedding deployment whose model_info.supported_endpoints lists only
/v1/embeddings, or a control deployment on the same backend that declares nothing. /embeddings serves
real vectors on both. Every chat-shaped route to the declared model answers 400 with the proxy's own
message, while the same chat call to the control goes to OpenAI and comes back with whatever OpenAI
answers, so getting the proxy's message is the proof that no provider call was made.
"""

from __future__ import annotations

import dataclasses
from typing import Final

import anthropic
import openai
import pytest
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import AnthropicErrorBody, AnthropicErrorEvent, LiteLLMParamsBody, ModelInfoBody, ModelNewBody
from openai import OpenAI
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

EMBEDDING_BACKEND: Final = "openai/text-embedding-3-small"
EMBEDDINGS_ONLY: Final = ("/v1/embeddings",)
PROMPT: Final = "Say this is a test!"
UNSUPPORTED_MARKER: Final = "is not supported by model="


class _OpenAiErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    type: str
    param: str | None = None


def _subject(route: Route) -> Subject:
    return Subject(
        domain=Domain.LLM_TRANSLATION,
        route=route,
        providers=(Provider.OPENAI,),
        models=(EMBEDDING_BACKEND,),
        mode=Mode.NONSTREAM,
    )


def _register(
    proxy: ProxyClient, resources: ResourceManager, prefix: str, supported_endpoints: tuple[str, ...] | None
) -> str:
    model: Final = f"{prefix}-{unique_marker()}"
    model_id: Final = proxy.register_model(
        ModelNewBody(
            model_name=model,
            litellm_params=LiteLLMParamsBody(model=EMBEDDING_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
            model_info=ModelInfoBody(
                mode="embedding",
                supported_endpoints=None if supported_endpoints is None else list(supported_endpoints),
            ),
        )
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _openai_at(sdk: SdkClients, api_prefix: str, key: str) -> OpenAI:
    return dataclasses.replace(sdk, base_url=f"{sdk.base_url}{api_prefix}").openai(key)


def _unsupported_message(endpoint: str, model: str) -> str:
    return f"{endpoint} {UNSUPPORTED_MARKER}{model}. Supported endpoints: {', '.join(sorted(EMBEDDINGS_ONLY))}"


def _chat_error_message(client: OpenAI, model: str) -> str:
    try:
        _ = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}], extra_body=NO_PROXY_CACHE
        )
    except openai.APIStatusError as error:
        return f"{error.status_code} {error.message}"
    return ""


def _assert_openai_rejection(error: openai.APIStatusError, endpoint: str, model: str) -> None:
    assert error.status_code == 400, f"expected the proxy's 400, got {error.status_code}: {error.message}"
    assert _OpenAiErrorDetail.model_validate(error.body) == _OpenAiErrorDetail(
        message=_unsupported_message(endpoint, model), type="invalid_request_error", param="model"
    ), f"unexpected error body: {error.body!r}"


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
    @pytest.mark.parametrize("api_prefix", ("/v1", ""), ids=("v1-chat-completions", "chat-completions"))
    def test_chat_completions_are_rejected_before_any_provider_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, api_prefix: str
    ) -> None:
        model: Final = _register(proxy, resources, "e2e-embed-only", EMBEDDINGS_ONLY)
        with pytest.raises(openai.APIStatusError) as raised:
            _ = _openai_at(sdk, api_prefix, resources.key()).chat.completions.create(
                model=model, messages=[{"role": "user", "content": PROMPT}], extra_body=NO_PROXY_CACHE
            )
        _assert_openai_rejection(raised.value, "/v1/chat/completions", model)

    @pytest.mark.covers("llm.embeddings.openai.supported_endpoints_gate.nonstream.works")
    @meta(_subject(Route.RESPONSES))
    @pytest.mark.parametrize("api_prefix", ("/v1", ""), ids=("v1-responses", "responses"))
    def test_responses_are_rejected_before_any_provider_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, api_prefix: str
    ) -> None:
        model: Final = _register(proxy, resources, "e2e-embed-only", EMBEDDINGS_ONLY)
        with pytest.raises(openai.APIStatusError) as raised:
            _ = _openai_at(sdk, api_prefix, resources.key()).responses.create(
                model=model, input=PROMPT, extra_body=NO_PROXY_CACHE
            )
        _assert_openai_rejection(raised.value, "/v1/responses", model)

    @pytest.mark.covers("llm.embeddings.openai.supported_endpoints_gate.nonstream.works")
    @meta(_subject(Route.MESSAGES))
    def test_messages_are_rejected_before_any_provider_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, "e2e-embed-only", EMBEDDINGS_ONLY)
        with pytest.raises(anthropic.APIStatusError) as raised:
            _ = sdk.anthropic(resources.key()).messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": PROMPT}],
                extra_body=NO_PROXY_CACHE,
            )
        assert raised.value.status_code == 400, (
            f"expected the proxy's 400, got {raised.value.status_code}: {raised.value.message}"
        )
        assert AnthropicErrorEvent.model_validate(raised.value.body) == AnthropicErrorEvent(
            type="error",
            error=AnthropicErrorBody(type="invalid_request_error", message=_unsupported_message("/v1/messages", model)),
        ), f"unexpected Anthropic error envelope: {raised.value.body!r}"

    @pytest.mark.covers("llm.embeddings.openai.supported_endpoints_gate.nonstream.works")
    @meta(_subject(Route.CHAT_COMPLETIONS))
    def test_chat_completions_without_supported_endpoints_reach_the_provider(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, "e2e-embed-control", None)
        error_message: Final = _chat_error_message(_openai_at(sdk, "/v1", resources.key()), model)
        assert UNSUPPORTED_MARKER not in error_message, (
            f"a model that declares no supported_endpoints must reach the provider, but the proxy "
            f"rejected it: {error_message}"
        )
