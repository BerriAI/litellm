"""Live e2e: `/utils/token_counter?call_endpoint=true` counts Gemini `contents` upstream.

Google's countTokens API is the only tokenizer that knows Gemini's real token
boundaries, so the proxy must forward `contents` to it for both the AI Studio and
Vertex deployments and hand back the provider's `promptTokensDetails`. Claude on
Vertex is covered by `/v1/messages/count_tokens`; this is the Gemini `contents`
route the claude_code rows never reach
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import require_successful_call
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

GEMINI_DEPLOYMENTS = ("gemini-2.5-flash", "gemini-2.5-flash-vertex")


class _Part(BaseModel):
    text: str


class _Content(BaseModel):
    parts: tuple[_Part, ...]


class _TokenCountBody(BaseModel):
    model: str
    contents: tuple[_Content, ...]


class _CallEndpoint(BaseModel):
    call_endpoint: bool = True


class _ModalityTokens(BaseModel):
    modality: str
    tokenCount: int


class _CountTokensUpstream(BaseModel):
    totalTokens: int
    promptTokensDetails: tuple[_ModalityTokens, ...]


class _TokenCountResponse(BaseModel):
    total_tokens: int
    request_model: str
    model_used: str
    tokenizer_type: str
    original_response: _CountTokensUpstream


class TestGeminiContentsTokenCounting:
    @pytest.mark.parametrize("model", GEMINI_DEPLOYMENTS)
    def test_contents_are_counted_by_the_provider_endpoint(
        self, proxy: ProxyClient, scoped_key: str, model: str
    ) -> None:
        text = f"Hello world, how are you doing today? {unique_marker()}"
        body = _TokenCountBody(model=model, contents=(_Content(parts=(_Part(text=text),)),))

        result = proxy.transport.send(
            "/utils/token_counter",
            headers=proxy.transport.bearer(scoped_key),
            json=body,
            params=_CallEndpoint(),
        )

        require_successful_call(result)
        counted = _TokenCountResponse.model_validate_json(result.body)
        assert counted.request_model == model, counted
        assert counted.original_response.totalTokens == counted.total_tokens > 0, counted
        assert counted.original_response.promptTokensDetails, counted
        assert all(detail.tokenCount > 0 for detail in counted.original_response.promptTokensDetails), counted
