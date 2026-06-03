"""Regression tests for LIT-3550: Anthropic passthrough returns "Invalid bearer token".

The bug: ``/anthropic/v1/messages`` forwarded the client virtual key as the
upstream credential because (a) ``passthrough_endpoint_router`` did not know
about API keys configured on deployments added via the UI (which does not set
``use_in_pass_through=true``), and (b) ``forward_headers_from_request`` did not
strip inbound credential headers when the operator set an upstream credential
of a different header type.
"""
import asyncio
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)


# ---------------------------------------------------------------------------
# 1. forward_headers_from_request drops inbound credential headers
# ---------------------------------------------------------------------------
class TestForwardHeadersFromRequestDropsCredentials:
    """LIT-3550: when ``custom_headers`` sets any credential header, every
    inbound credential header must be dropped before merging, so the LiteLLM
    virtual key cannot leak to the upstream provider."""

    @staticmethod
    def _call(request_headers: dict, custom_headers: dict) -> dict:
        from litellm.passthrough.utils import BasePassthroughUtils
        return BasePassthroughUtils.forward_headers_from_request(
            request_headers=request_headers.copy(),
            headers=custom_headers,
            forward_headers=True,
        )

    def test_x_api_key_upstream_strips_inbound_authorization(self):
        """The Anthropic case: operator sets x-api-key upstream; the inbound
        Authorization carrying the virtual key MUST be dropped."""
        forwarded = self._call(
            request_headers={
                "authorization": "Bearer sk-virtualkey-from-team",
                "x-api-key": "sk-virtualkey-from-team",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            custom_headers={"x-api-key": "sk-ant-upstream"},
        )
        assert forwarded["x-api-key"] == "sk-ant-upstream"
        assert "authorization" not in forwarded
        assert "Authorization" not in forwarded
        assert forwarded["anthropic-version"] == "2023-06-01"
        assert forwarded["content-type"] == "application/json"

    def test_authorization_upstream_strips_inbound_x_api_key(self):
        """Mirror: operator sets Authorization upstream; inbound x-api-key
        carrying the virtual key MUST be dropped."""
        forwarded = self._call(
            request_headers={
                "authorization": "Bearer sk-virtualkey-from-team",
                "x-api-key": "sk-virtualkey-from-team",
                "anthropic-version": "2023-06-01",
            },
            custom_headers={"authorization": "Bearer sk-upstream-oauth"},
        )
        assert forwarded["authorization"] == "Bearer sk-upstream-oauth"
        assert "x-api-key" not in forwarded

    def test_case_insensitive_match_drops_inbound_dupes(self):
        """X-Api-Key (mixed case) on custom_headers still drops inbound
        x-api-key (lower case) — RFC 9110 header names are case-insensitive."""
        forwarded = self._call(
            request_headers={
                "authorization": "Bearer sk-virtualkey",
                "x-api-key": "sk-virtualkey",
            },
            custom_headers={"X-Api-Key": "sk-ant-upstream"},
        )
        assert "x-api-key" not in forwarded
        assert forwarded["X-Api-Key"] == "sk-ant-upstream"
        assert "authorization" not in forwarded

    def test_no_credential_in_custom_preserves_inbound(self):
        """When custom_headers has no credential header, inbound headers are
        forwarded as-is (existing behavior, no regression)."""
        forwarded = self._call(
            request_headers={
                "authorization": "Bearer sk-virtualkey",
                "x-api-key": "sk-virtualkey",
                "anthropic-version": "2023-06-01",
            },
            custom_headers={"x-litellm-trace-id": "trace-1"},
        )
        assert forwarded.get("authorization") == "Bearer sk-virtualkey"
        assert forwarded.get("x-api-key") == "sk-virtualkey"
        assert forwarded["x-litellm-trace-id"] == "trace-1"
        assert forwarded["anthropic-version"] == "2023-06-01"

    def test_drops_content_length_and_host(self):
        """Pre-existing behavior: content-length and host are stripped."""
        forwarded = self._call(
            request_headers={
                "host": "litellm.example.com",
                "content-length": "100",
                "content-type": "application/json",
            },
            custom_headers={"x-api-key": "sk-ant-upstream"},
        )
        assert "host" not in forwarded
        assert "content-length" not in forwarded
        assert forwarded["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# 2. _resolve_anthropic_api_key_from_router fallback
# ---------------------------------------------------------------------------
class TestResolveAnthropicApiKeyFromRouter:
    """LIT-3550 fallback: when no explicit pass-through credential is set,
    anthropic_proxy_route discovers the api_key from any deployed Anthropic
    model in llm_router (so the UI add-model flow works)."""

    def _set_router(self, model_list):
        import litellm
        import litellm.proxy.proxy_server as proxy_server
        proxy_server.llm_router = (
            litellm.Router(model_list=model_list) if model_list else None
        )

    def test_finds_anthropic_deployment_api_key(self):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_api_key_from_router,
        )
        self._set_router([
            {
                "model_name": "claude-sonnet-4-5",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "api_key": "sk-ant-deploy-1",
                },
            }
        ])
        assert _resolve_anthropic_api_key_from_router() == "sk-ant-deploy-1"

    def test_returns_none_when_router_none(self):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_api_key_from_router,
        )
        self._set_router(None)
        assert _resolve_anthropic_api_key_from_router() is None

    def test_returns_none_when_no_anthropic_deployment(self):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_api_key_from_router,
        )
        self._set_router([
            {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-openai"}},
            {"model_name": "mistral", "litellm_params": {"model": "mistral/mistral-large-latest", "api_key": "sk-mistral"}},
        ])
        assert _resolve_anthropic_api_key_from_router() is None

    def test_returns_none_when_anthropic_deployment_has_no_key(self):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_api_key_from_router,
        )
        self._set_router([
            {"model_name": "claude", "litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
        ])
        assert _resolve_anthropic_api_key_from_router() is None

    def test_router_pre_resolves_os_environ_placeholder(self, monkeypatch):
        """Router.set_model_list resolves os.environ/... placeholders before
        the helper sees them (router.py ~L8227). The helper sees the
        already-resolved concrete secret."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_api_key_from_router,
        )
        monkeypatch.setenv("MY_ANTHROPIC_KEY", "sk-ant-from-env")
        self._set_router([
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "api_key": "os.environ/MY_ANTHROPIC_KEY",
                },
            }
        ])
        assert _resolve_anthropic_api_key_from_router() == "sk-ant-from-env"

    def test_skips_deployment_with_unresolved_env_placeholder_literal(self, monkeypatch):
        """Router stores the raw "os.environ/..." literal back into
        litellm_params when get_secret returns None (router.py ~L8230). The
        helper must skip such deployments and never return the literal as a
        credential — that would forward nonsense upstream."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_api_key_from_router,
        )
        # Manually craft a model_list entry that already contains the literal
        # os.environ/... string (i.e. Router could not resolve it). The helper
        # must skip this and find the next usable key.
        import litellm.proxy.proxy_server as proxy_server
        proxy_server.llm_router = type("FakeRouter", (), {"model_list": [
            {
                "model_name": "claude-a",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "api_key": "os.environ/MISSING_ANTHROPIC_KEY",  # literal — unresolved
                },
            },
            {
                "model_name": "claude-b",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-5",
                    "api_key": "sk-ant-fallback",
                },
            },
        ]})()
        assert _resolve_anthropic_api_key_from_router() == "sk-ant-fallback"

    def test_matches_bare_claude_model_name(self):
        """Deployment model strings like ``claude-sonnet-4-5`` (no
        ``anthropic/`` prefix) are still recognised as Anthropic deployments."""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            _resolve_anthropic_api_key_from_router,
        )
        self._set_router([
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "claude-sonnet-4-5",
                    "api_key": "sk-ant-bare",
                },
            }
        ])
        assert _resolve_anthropic_api_key_from_router() == "sk-ant-bare"


# ---------------------------------------------------------------------------
# 3. anthropic_proxy_route raises clean 401 instead of leaking virtual key
# ---------------------------------------------------------------------------
class TestAnthropicProxyRouteRaisesWhenNoCredential:
    """LIT-3550: when no Anthropic credential is configured anywhere, the
    route must raise a clean ProxyException(401) instead of silently
    forwarding the client virtual key as the upstream credential."""

    def test_raises_401_when_no_credential_configured(self, monkeypatch):
        from fastapi import Response

        from litellm.proxy._types import ProxyException, UserAPIKeyAuth
        import litellm.proxy.proxy_server as proxy_server
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            anthropic_proxy_route, passthrough_endpoint_router,
        )

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        passthrough_endpoint_router.credentials = {}
        proxy_server.llm_router = None

        req = MagicMock()
        req.method = "POST"
        req.headers = {"authorization": "Bearer sk-virtualkey", "anthropic-version": "2023-06-01"}
        req.query_params = {}
        url = MagicMock(); url.path = "/anthropic/v1/messages"
        req.url = url
        req.state = None

        async def _no_stream(r):
            return False

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            _no_stream,
            raising=False,
        )

        async def _run():
            return await anthropic_proxy_route(
                endpoint="v1/messages",
                request=req,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-vk"),
            )

        with pytest.raises(ProxyException) as exc:
            asyncio.run(_run())
        assert exc.value.code == "401"
        assert "ANTHROPIC_API_KEY" in exc.value.message
        # Critical: error message must NOT echo the inbound virtual key.
        assert "sk-virtualkey" not in str(exc.value.message)
