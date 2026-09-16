"""Chat Authorization header matrix on LLM routes (LIT-4778).

Virtual-key chat must reject missing and malformed Authorization headers before
any provider call. These cases sit next to the existing valid/invalid key check
and pin the bearer-token failure matrix.
"""

from __future__ import annotations

import pytest
from e2e_http import AuthHeaders, NoBody, StreamingResponse, assert_auth_denied
from models import ChatBody, ChatMessage
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

CHAT_PATH = "/chat/completions"
UNREACHABLE_MODEL = "auth-must-fail-before-model-resolution"


def _chat_with_headers(proxy: ProxyClient, headers: AuthHeaders | NoBody) -> StreamingResponse:
    return proxy.transport.send(
        CHAT_PATH,
        headers=headers,
        json=ChatBody(
            model=UNREACHABLE_MODEL,
            messages=[ChatMessage(role="user", content="should not run")],
            max_tokens=8,
        ),
    )


class TestChatAuthHeaders:
    @pytest.mark.covers("other.auth.llm_chat.missing_header_denied")
    def test_missing_authorization_header_is_denied(self, proxy: ProxyClient) -> None:
        result = _chat_with_headers(proxy, NoBody())
        assert_auth_denied(result, "missing Authorization")

    @pytest.mark.covers("other.auth.llm_chat.invalid_bearer_denied")
    def test_bearer_invalid_token_is_denied(self, proxy: ProxyClient) -> None:
        result = _chat_with_headers(proxy, AuthHeaders(authorization="Bearer invalid_token"))
        assert_auth_denied(result, "Bearer invalid_token")

    @pytest.mark.covers("other.auth.llm_chat.no_bearer_prefix_denied")
    def test_token_without_bearer_prefix_is_denied(self, proxy: ProxyClient) -> None:
        result = _chat_with_headers(proxy, AuthHeaders(authorization="invalid_token"))
        assert_auth_denied(result, "token without Bearer prefix")

    @pytest.mark.covers("other.auth.llm_chat.empty_bearer_denied")
    def test_empty_bearer_token_is_denied(self, proxy: ProxyClient) -> None:
        result = _chat_with_headers(proxy, AuthHeaders(authorization="Bearer "))
        assert_auth_denied(result, "empty Bearer token")

    @pytest.mark.covers("other.auth.llm_chat.not_bearer_scheme_denied")
    def test_not_bearer_scheme_is_denied(self, proxy: ProxyClient) -> None:
        result = _chat_with_headers(proxy, AuthHeaders(authorization="NotBearer validtoken123"))
        assert_auth_denied(result, "NotBearer scheme")
