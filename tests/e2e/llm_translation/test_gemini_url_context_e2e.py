from __future__ import annotations

import os
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    GeminiUrlContextTool,
    LiteLLMParamsBody,
)
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

GEMINI_BACKEND: Final = "gemini/gemini-3.8-flash"
URL_CONTEXT_TARGET: Final = "https://www.iana.org/domains/reserved"


class TestGeminiUrlContext:
    @pytest.mark.covers("llm.translation.gemini.url_context")
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY is required for Gemini URL-context provider access",
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.GEMINI,),
            models=(GEMINI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_url_context_metadata_is_returned(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model_name: Final = f"e2e-gemini-url-context-{unique_marker()}"
        model_id: Final = proxy.create_model(
            model_name,
            LiteLLMParamsBody(model=GEMINI_BACKEND, api_key="os.environ/GEMINI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key: Final = resources.key()
        marker: Final = unique_marker()
        request: Final = ChatBody(
            model=model_name,
            messages=[
                ChatMessage(
                    role="user",
                    content=f"Use URL context to identify the page title at {URL_CONTEXT_TARGET}. {marker}",
                )
            ],
            tools=[GeminiUrlContextTool()],
        )
        raw_response: Final = sdk.openai(key).chat.completions.create(
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
            extra_body={
                **NO_PROXY_CACHE,
                "tools": [tool.model_dump(by_alias=True) for tool in request.tools],
            },
        )
        response: Final[ChatResponse] = ChatResponse.model_validate(raw_response.model_dump())

        assert response.vertex_ai_url_context_metadata
        assert response.vertex_ai_url_context_metadata[0].urlMetadata
        assert response.vertex_ai_url_context_metadata[0].urlMetadata[0].retrievedUrl == URL_CONTEXT_TARGET
        assert response.vertex_ai_url_context_metadata[0].urlMetadata[0].urlRetrievalStatus == (
            "URL_RETRIEVAL_STATUS_SUCCESS"
        )
