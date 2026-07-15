"""
Tests for the multi-backend Anthropic routing handler.

Covers HealthTracker, AnthropicRouter, AnthropicProxy, config loading,
and integration with the passthrough endpoint.
"""

import json
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler import (
    AnthropicProxy,
    AnthropicRouter,
    AnthropicRouterConfig,
    AuthConfig,
    Backend,
    HealthTracker,
    Route,
    RouterSettings,
    Status,
    extract_model_from_body,
    get_anthropic_router,
    init_anthropic_router_from_config,
)


# ---------------------------------------------------------------------------
# HealthTracker
# ---------------------------------------------------------------------------

class TestHealthTracker:
    """Tests for the HealthTracker class."""

    def setup_method(self):
        self.health = HealthTracker(cooldown_seconds=30, max_failures=3)

    def test_initial_status_is_healthy(self):
        assert self.health.status("backend-1") == Status.HEALTHY

    def test_is_healthy_returns_true_initially(self):
        assert self.health.is_healthy("backend-1") is True

    def test_single_failure_keeps_healthy(self):
        self.health.record_failure("backend-1")
        assert self.health.status("backend-1") == Status.HEALTHY

    def test_two_failures_becomes_degraded(self):
        self.health.record_failure("backend-1")
        self.health.record_failure("backend-1")
        assert self.health.status("backend-1") == Status.DEGRADED

    def test_three_failures_becomes_dead(self):
        for _ in range(3):
            self.health.record_failure("backend-1")
        assert self.health.status("backend-1") == Status.DEAD

    def test_dead_backend_not_healthy(self):
        for _ in range(3):
            self.health.record_failure("backend-1")
        assert self.health.is_healthy("backend-1") is False

    def test_record_success_resets_to_healthy(self):
        for _ in range(3):
            self.health.record_failure("backend-1")
        assert self.health.status("backend-1") == Status.DEAD

        self.health.record_success("backend-1")
        assert self.health.status("backend-1") == Status.HEALTHY
        assert self.health.is_healthy("backend-1") is True

    def test_dead_cooldown_expires_to_degraded(self):
        health = HealthTracker(cooldown_seconds=0, max_failures=3)
        for _ in range(3):
            health.record_failure("backend-1")
        assert health.status("backend-1") == Status.DEAD

        # With cooldown_seconds=0, is_healthy should flip to degraded
        assert health.is_healthy("backend-1") is True
        assert health.status("backend-1") == Status.DEGRADED

    def test_dead_cooldown_not_reset_on_retry(self):
        """record_failure on an already-DEAD backend must NOT extend cooldown.

        Under continuous traffic, dead backends are tried as last resort.
        If each retry resets the cooldown timer, the backend can never
        recover (livelock).  The cooldown is set once on the initial
        transition to DEAD and must not be extended by subsequent failures.
        """
        health = HealthTracker(cooldown_seconds=30, max_failures=3)

        # Transition to DEAD — cooldown is set once.
        for _ in range(3):
            health.record_failure("backend-1")
        assert health.status("backend-1") == Status.DEAD

        # Capture the cooldown timestamp after the initial transition.
        first_cooldown = health._entries["backend-1"].cooldown_until
        assert first_cooldown > 0

        # Advance time a bit, then record another failure while DEAD.
        import time as _time
        _time.sleep(0.01)

        health.record_failure("backend-1")
        assert health.status("backend-1") == Status.DEAD

        # Cooldown must NOT have changed — the backend recovers at the
        # originally-scheduled time, not after the last retry.
        second_cooldown = health._entries["backend-1"].cooldown_until
        assert second_cooldown == first_cooldown

    def test_multiple_backends_tracked_independently(self):
        self.health.record_failure("backend-a")
        self.health.record_failure("backend-a")
        self.health.record_failure("backend-a")

        self.health.record_success("backend-b")

        assert self.health.status("backend-a") == Status.DEAD
        assert self.health.status("backend-b") == Status.HEALTHY


# ---------------------------------------------------------------------------
# AnthropicRouter
# ---------------------------------------------------------------------------

class TestAnthropicRouter:
    """Tests for the AnthropicRouter class."""

    def setup_method(self):
        config = AnthropicRouterConfig(
            routes=[
                Route(
                    model="claude-sonnet-*",
                    backends=[
                        Backend(
                            name="primary",
                            url="https://api.deepseek.com/anthropic",
                            auth=AuthConfig(type="api-key", key_env="DEEPSEEK_API_KEY"),
                        ),
                        Backend(
                            name="fallback",
                            url="https://api.anthropic.com",
                            auth=AuthConfig(type="api-key", key_env="ANTHROPIC_API_KEY"),
                        ),
                    ],
                ),
                Route(
                    model="claude-opus-*",
                    backends=[
                        Backend(
                            name="opus-primary",
                            url="https://api.anthropic.com",
                            auth=AuthConfig(type="bearer", key_env="ANTHROPIC_API_KEY"),
                        ),
                    ],
                ),
            ],
            settings=RouterSettings(cooldown_seconds=30, max_failures=3),
        )
        self.router = AnthropicRouter(config)

    def test_resolve_exact_model_match(self):
        backends = self.router.resolve("claude-sonnet-5")
        assert len(backends) == 2
        assert backends[0].name == "primary"
        assert backends[1].name == "fallback"

    def test_resolve_glob_match(self):
        backends = self.router.resolve("claude-sonnet-3.5")
        assert len(backends) == 2
        assert backends[0].name == "primary"

    def test_resolve_opus_model(self):
        backends = self.router.resolve("claude-opus-4")
        assert len(backends) == 1
        assert backends[0].name == "opus-primary"

    def test_resolve_no_match_returns_empty(self):
        backends = self.router.resolve("gpt-4")
        assert backends == []

    def test_healthy_backends_ordered_first(self):
        # Mark primary as dead
        for _ in range(3):
            self.router.health.record_failure("primary")

        backends = self.router.resolve("claude-sonnet-5")
        assert len(backends) == 2
        # Healthy (fallback) first, dead (primary) last
        assert backends[0].name == "fallback"
        assert backends[1].name == "primary"

    def test_routes_property(self):
        assert len(self.router.routes) == 2
        assert self.router.routes[0].model == "claude-sonnet-*"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Tests for config parsing and lazy initialisation."""

    def test_init_with_valid_config(self):
        config_dict = {
            "anthropic_router": {
                "routes": [
                    {
                        "model": "claude-haiku-*",
                        "backends": [
                            {
                                "name": "h1",
                                "url": "https://api.anthropic.com",
                                "auth": {"type": "api-key", "key_env": "ANTHROPIC_API_KEY"},
                            },
                        ],
                    },
                ],
                "settings": {"cooldown_seconds": 60, "max_failures": 5},
            }
        }

        router = init_anthropic_router_from_config(config_dict)
        assert router is not None
        assert len(router.routes) == 1
        assert router.routes[0].model == "claude-haiku-*"
        assert router._settings.cooldown_seconds == 60
        assert router._settings.max_failures == 5

    def test_init_without_config_returns_none(self):
        router = init_anthropic_router_from_config({})
        assert router is None

    def test_init_with_missing_key_returns_none(self):
        config_dict = {"general_settings": {}, "litellm_settings": {}}
        router = init_anthropic_router_from_config(config_dict)
        assert router is None

    def test_init_with_invalid_config_raises(self):
        """When the key is present but malformed, raise so the caller can
        distinguish 'key absent' (intentional) from 'parse error' (retry)."""
        from pydantic import ValidationError

        config_dict = {
            "anthropic_router": {
                "routes": "not_a_list",  # invalid type
            }
        }
        with pytest.raises(ValidationError):
            init_anthropic_router_from_config(config_dict)

    def test_init_with_default_settings(self):
        config_dict = {
            "anthropic_router": {
                "routes": [
                    {
                        "model": "*",
                        "backends": [
                            {
                                "name": "default",
                                "url": "https://api.anthropic.com",
                                "auth": {"type": "bearer", "key_env": "KEY"},
                            },
                        ],
                    },
                ],
            }
        }
        router = init_anthropic_router_from_config(config_dict)
        assert router is not None
        # Default settings should apply
        assert router._settings.cooldown_seconds == 30
        assert router._settings.max_failures == 3

    def test_get_anthropic_router_lazy_init(self):
        """When proxy_config is not available, get_anthropic_router should
        return None gracefully without permanently disabling the router."""
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler._state.key_present",
            None,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler._state.instance",
            None,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler._state.next_retry",
            0.0,
        ), patch(
            "litellm.proxy.proxy_server.proxy_config",
            None,
        ):
            router = get_anthropic_router()
            # Lazy init should not crash when proxy_config is unavailable
            assert router is None

    def test_get_anthropic_router_retries_after_failure(self):
        """When config is available but parsing fails, the router should
        schedule a retry instead of permanently disabling itself."""
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler._state.key_present",
            None,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler._state.instance",
            None,
        ):
            # Simulate config that has the key but with invalid data
            bad_config = {"anthropic_router": {"routes": "not_a_list"}}
            with patch(
                "litellm.proxy.proxy_server.proxy_config.get_config_state",
                return_value=bad_config,
            ):
                router = get_anthropic_router()
                # Should return None (init failed) but NOT permanently disable
                assert router is None
                # _state.key_present should be True (key was present, retry later)
                from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler import (
                    _state,
                )
                assert _state.key_present is True
                assert _state.next_retry > 0

    def test_get_anthropic_router_no_key_stops_retrying(self):
        """When the config has no anthropic_router key, the router should
        permanently disable itself (it's an intentional configuration)."""
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler._state.key_present",
            None,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler._state.instance",
            None,
        ):
            empty_config = {"general_settings": {}, "litellm_settings": {}}
            with patch(
                "litellm.proxy.proxy_server.proxy_config.get_config_state",
                return_value=empty_config,
            ):
                router = get_anthropic_router()
                assert router is None
                # _state.key_present should be False (intentional)
                from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler import (
                    _state,
                )
                assert _state.key_present is False


# ---------------------------------------------------------------------------
# AnthropicProxy — credential resolution
# ---------------------------------------------------------------------------

class TestAnthropicProxy:
    """Tests for the AnthropicProxy credential resolution."""

    def test_resolve_credential_from_env(self):
        with patch.dict(os.environ, {"TEST_KEY": "test-value"}):
            result = AnthropicProxy._resolve_credential("TEST_KEY")
            assert result == "test-value"

    def test_resolve_credential_missing_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            result = AnthropicProxy._resolve_credential("NONEXISTENT_KEY")
            assert result == ""

    def test_resolve_credential_via_secret_str(self):
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_routing_handler.get_secret_str",
            return_value="secret-value",
        ):
            result = AnthropicProxy._resolve_credential("os.environ/SECRET")
            assert result == "secret-value"


# ---------------------------------------------------------------------------
# extract_model_from_body
# ---------------------------------------------------------------------------

class TestExtractModelFromBody:
    def test_extracts_model_from_valid_json(self):
        body = json.dumps({"model": "claude-sonnet-5", "max_tokens": 100}).encode()
        assert extract_model_from_body(body) == "claude-sonnet-5"

    def test_extract_model_missing_field_returns_unknown(self):
        body = json.dumps({"max_tokens": 100}).encode()
        assert extract_model_from_body(body) == "unknown"

    def test_extract_model_from_invalid_json_returns_unknown(self):
        body = b"not-json"
        assert extract_model_from_body(body) == "unknown"

    def test_extract_model_from_empty_body_returns_unknown(self):
        assert extract_model_from_body(b"") == "unknown"


# ---------------------------------------------------------------------------
# Integration — multi-backend route
# ---------------------------------------------------------------------------

class TestMultiBackendRoute:
    """Integration tests for the multi-backend passthrough route."""

    @pytest.mark.asyncio
    async def test_multi_backend_first_succeeds(self):
        """When the first backend responds successfully, return its response."""
        from fastapi import Response as FastAPIResponse

        # Build a mock request
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url = "http://proxy/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json", "host": "proxy"}
        mock_request.query_params = {}
        body = json.dumps(
            {"model": "claude-sonnet-5", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
        ).encode()

        # Make body() return bytes and cache it for re-reads
        async def mock_body():
            return body
        mock_request.body = mock_body
        # Also mock json() since the passthrough pipeline calls it
        mock_request.json = AsyncMock(return_value=json.loads(body))

        mock_fastapi_response = MagicMock(spec=FastAPIResponse)
        mock_user_api_key = MagicMock()
        mock_user_api_key.api_key = "test-key"

        # Build router with one backend
        config = AnthropicRouterConfig(
            routes=[
                Route(
                    model="claude-sonnet-*",
                    backends=[
                        Backend(
                            name="test-backend",
                            url="https://api.anthropic.com",
                            auth=AuthConfig(type="api-key", key_env="ANTHROPIC_API_KEY"),
                        ),
                    ],
                ),
            ],
        )
        router = AnthropicRouter(config)

        # Mock the passthrough endpoint function returned by create_pass_through_route
        mock_passthrough_response = MagicMock()
        mock_passthrough_response.status_code = 200

        mock_endpoint_func = AsyncMock(return_value=mock_passthrough_response)

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            return_value=mock_endpoint_func,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            return_value=False,
        ), patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "test-api-key"}
        ):
            from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
                _route_anthropic_with_multi_backend,
            )

            result = await _route_anthropic_with_multi_backend(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_fastapi_response,
                user_api_key_dict=mock_user_api_key,
                router=router,
            )

            assert result == mock_passthrough_response
            # Backend should be healthy after success
            assert router.health.status("test-backend") == Status.HEALTHY

    @pytest.mark.asyncio
    async def test_multi_backend_failover_to_fallback(self):
        """When the primary fails, the fallback should be tried."""
        from fastapi import Response as FastAPIResponse

        from litellm.proxy._types import ProxyException

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url = "http://proxy/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json", "host": "proxy"}
        mock_request.query_params = {}
        body = json.dumps({"model": "claude-sonnet-5", "max_tokens": 100}).encode()

        async def mock_body():
            return body
        mock_request.body = mock_body
        mock_request.json = AsyncMock(return_value=json.loads(body))

        mock_fastapi_response = MagicMock(spec=FastAPIResponse)
        mock_user_api_key = MagicMock()
        mock_user_api_key.api_key = "test-key"

        config = AnthropicRouterConfig(
            routes=[
                Route(
                    model="claude-sonnet-*",
                    backends=[
                        Backend(
                            name="primary",
                            url="https://api.deepseek.com/anthropic",
                            auth=AuthConfig(type="api-key", key_env="DEEPSEEK_API_KEY"),
                        ),
                        Backend(
                            name="fallback",
                            url="https://api.anthropic.com",
                            auth=AuthConfig(type="api-key", key_env="ANTHROPIC_API_KEY"),
                        ),
                    ],
                ),
            ],
        )
        router = AnthropicRouter(config)

        mock_fallback_response = MagicMock()
        mock_fallback_response.status_code = 200

        # First call raises ProxyException, second succeeds.
        # Use real async functions instead of AsyncMock(side_effect=...)
        # so the exception propagates correctly through await.
        async def raise_proxy_error(*args: Any, **kwargs: Any) -> Any:
            raise ProxyException(message="Primary failed", type="connection_error", param="", code=502)

        async def succeed(*args: Any, **kwargs: Any) -> Any:
            return mock_fallback_response

        create_route_calls = [raise_proxy_error, succeed]

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=create_route_calls,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            return_value=False,
        ), patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "ds-key", "ANTHROPIC_API_KEY": "anthro-key"}
        ):
            from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
                _route_anthropic_with_multi_backend,
            )

            result = await _route_anthropic_with_multi_backend(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_fastapi_response,
                user_api_key_dict=mock_user_api_key,
                router=router,
            )

            assert result == mock_fallback_response
            # Primary should have its failure recorded (1 failure, still HEALTHY
            # with default max_failures=3 — only 2+ changes to DEGRADED)
            assert router.health.status("primary") == Status.HEALTHY
            # Secondary effect: failure count incremented
            assert router.health._entries["primary"].failures == 1
            # Fallback should be healthy
            assert router.health.status("fallback") == Status.HEALTHY

    @pytest.mark.asyncio
    async def test_multi_backend_all_fail_returns_502(self):
        """When all backends fail, return 502."""
        from fastapi import HTTPException, Response as FastAPIResponse

        from litellm.proxy._types import ProxyException

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url = "http://proxy/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json", "host": "proxy"}
        mock_request.query_params = {}
        body = json.dumps({"model": "claude-sonnet-5", "max_tokens": 100}).encode()

        async def mock_body():
            return body
        mock_request.body = mock_body
        mock_request.json = AsyncMock(return_value=json.loads(body))

        mock_fastapi_response = MagicMock(spec=FastAPIResponse)
        mock_user_api_key = MagicMock()
        mock_user_api_key.api_key = "test-key"

        config = AnthropicRouterConfig(
            routes=[
                Route(
                    model="claude-sonnet-*",
                    backends=[
                        Backend(
                            name="b1",
                            url="https://api.backend1.com",
                            auth=AuthConfig(type="api-key", key_env="KEY1"),
                        ),
                        Backend(
                            name="b2",
                            url="https://api.backend2.com",
                            auth=AuthConfig(type="api-key", key_env="KEY2"),
                        ),
                    ],
                ),
            ],
        )
        router = AnthropicRouter(config)

        # Both calls raise ProxyException
        async def raise_backend_error(*args: Any, **kwargs: Any) -> Any:
            raise ProxyException(message="Backend unavailable", type="connection_error", param="", code=502)

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            return_value=raise_backend_error,
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            return_value=False,
        ), patch.dict(
            os.environ, {"KEY1": "k1", "KEY2": "k2"}
        ):
            from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
                _route_anthropic_with_multi_backend,
            )

            with pytest.raises(HTTPException) as exc_info:
                await _route_anthropic_with_multi_backend(
                    endpoint="v1/messages",
                    request=mock_request,
                    fastapi_response=mock_fastapi_response,
                    user_api_key_dict=mock_user_api_key,
                    router=router,
                )

            assert exc_info.value.status_code == 502
            detail = exc_info.value.detail
            assert "error" in detail
            assert detail["error"]["type"] == "router_error"

    @pytest.mark.asyncio
    async def test_multi_backend_no_route_match_returns_502(self):
        """When the model doesn't match any route, return 502."""
        from fastapi import HTTPException, Response as FastAPIResponse

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url = "http://proxy/anthropic/v1/messages"
        mock_request.headers = {"content-type": "application/json", "host": "proxy"}
        mock_request.query_params = {}
        body = json.dumps({"model": "unknown-model-xyz", "max_tokens": 100}).encode()

        async def mock_body():
            return body
        mock_request.body = mock_body

        mock_fastapi_response = MagicMock(spec=FastAPIResponse)
        mock_user_api_key = MagicMock()

        config = AnthropicRouterConfig(
            routes=[
                Route(
                    model="claude-sonnet-*",
                    backends=[
                        Backend(
                            name="b1",
                            url="https://api.anthropic.com",
                            auth=AuthConfig(type="api-key", key_env="KEY"),
                        ),
                    ],
                ),
            ],
        )
        router = AnthropicRouter(config)

        with patch.dict(os.environ, {"KEY": "k1"}):
            from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
                _route_anthropic_with_multi_backend,
            )

            with pytest.raises(HTTPException) as exc_info:
                await _route_anthropic_with_multi_backend(
                    endpoint="v1/messages",
                    request=mock_request,
                    fastapi_response=mock_fastapi_response,
                    user_api_key_dict=mock_user_api_key,
                    router=router,
                )

            assert exc_info.value.status_code == 502
            assert "unknown-model-xyz" in str(exc_info.value.detail)
