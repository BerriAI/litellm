"""Chat Authorization header matrix on LLM routes (LIT-4778).

Virtual-key chat must reject missing and malformed Authorization headers before
any provider call. These cases sit next to the existing valid/invalid key check
and pin the bearer-token failure matrix.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import AuthHeaders, NoBody, StreamingResponse
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

OPENAI_BACKEND = "openai/gpt-4o-mini"
CHAT_PATH = "/chat/completions"


class RawAuthorizationHeaders(BaseModel):
    Authorization: str


def _register_model(proxy: ProxyClient, resources: ResourceManager) -> str:
    model = f"e2e-auth-headers-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _chat_with_headers(
    proxy: ProxyClient, headers: BaseModel, model: str
) -> StreamingResponse:
    return proxy.transport.send(
        CHAT_PATH,
        headers=headers,
        json=ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content="should not run")],
            max_tokens=8,
        ),
    )


def _assert_auth_denied(result: StreamingResponse, context: str) -> None:
    assert result.status_code in (401, 403), (
        f"{context}: expected 401/403, got {result.status_code}: {result.body[:300]}"
    )


class TestChatAuthHeaders:
    @pytest.mark.covers("other.auth.llm_chat.missing_header_denied")
    def test_missing_authorization_header_is_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = _register_model(proxy, resources)
        result = _chat_with_headers(proxy, NoBody(), model)
        _assert_auth_denied(result, "missing Authorization")

    @pytest.mark.covers("other.auth.llm_chat.invalid_bearer_denied")
    def test_bearer_invalid_token_is_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = _register_model(proxy, resources)
        result = _chat_with_headers(
            proxy, AuthHeaders(authorization="Bearer invalid_token"), model
        )
        _assert_auth_denied(result, "Bearer invalid_token")

    @pytest.mark.covers("other.auth.llm_chat.no_bearer_prefix_denied")
    def test_token_without_bearer_prefix_is_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = _register_model(proxy, resources)
        result = _chat_with_headers(
            proxy, RawAuthorizationHeaders(Authorization="invalid_token"), model
        )
        _assert_auth_denied(result, "token without Bearer prefix")

    @pytest.mark.covers("other.auth.llm_chat.empty_bearer_denied")
    def test_empty_bearer_token_is_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = _register_model(proxy, resources)
        result = _chat_with_headers(
            proxy, AuthHeaders(authorization="Bearer "), model
        )
        _assert_auth_denied(result, "empty Bearer token")

    @pytest.mark.covers("other.auth.llm_chat.not_bearer_scheme_denied")
    def test_not_bearer_scheme_is_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = _register_model(proxy, resources)
        result = _chat_with_headers(
            proxy, RawAuthorizationHeaders(Authorization="NotBearer validtoken123"), model
        )
        _assert_auth_denied(result, "NotBearer scheme")
