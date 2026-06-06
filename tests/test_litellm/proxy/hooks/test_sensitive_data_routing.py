"""
Tests for Sensitive Data Routing feature.

This feature allows guardrails to route requests to a different model
(typically on-premise) when sensitive data is detected, instead of blocking.
All subsequent requests in the same session are routed to the same model.
"""

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.exceptions import SensitiveDataRouteException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.sensitive_data_routing import (
    _PROXY_SensitiveDataRoutingHandler,
    SENSITIVE_ROUTING_CACHE_PREFIX,
    DEFAULT_SENSITIVE_ROUTING_TTL,
)


class MockInternalUsageCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._ttls: Dict[str, int] = {}
        self.dual_cache = MagicMock()
        self.dual_cache.redis_cache = None

    async def async_get_cache(self, key: str, **kwargs) -> Optional[Any]:
        return self._cache.get(key)

    async def async_set_cache(self, key: str, value: Any, ttl: int = 3600, **kwargs):
        self._cache[key] = value
        self._ttls[key] = ttl


class TestSensitiveDataRoutingHandler:
    @pytest.fixture
    def handler(self):
        cache = MockInternalUsageCache()
        return _PROXY_SensitiveDataRoutingHandler(internal_usage_cache=cache)

    @pytest.fixture
    def user_api_key_dict(self):
        return UserAPIKeyAuth(api_key="test-key")

    @pytest.mark.asyncio
    async def test_set_session_routing(self, handler):
        key = UserAPIKeyAuth(api_key="hashed-key")
        await handler.set_session_routing(
            session_id="test-session-123",
            model="on-premise-model",
            user_api_key_dict=key,
            guardrail_name="test-guardrail",
        )

        routed_model = await handler._get_routed_model("test-session-123", key)
        assert routed_model == "on-premise-model"

    def test_get_session_id_from_metadata(self):
        data = {"metadata": {"session_id": "session-from-metadata"}}
        session_id = get_session_id_from_request_data(data)
        assert session_id == "session-from-metadata"

    def test_get_session_id_from_litellm_metadata(self):
        data = {"litellm_metadata": {"session_id": "session-from-litellm-metadata"}}
        session_id = get_session_id_from_request_data(data)
        assert session_id == "session-from-litellm-metadata"

    def test_get_session_id_from_litellm_session_id(self):
        data = {"litellm_session_id": "session-direct"}
        session_id = get_session_id_from_request_data(data)
        assert session_id == "session-direct"

    @pytest.mark.asyncio
    async def test_pre_call_hook_no_session(self, handler, user_api_key_dict):
        data = {"model": "gpt-4"}
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
        assert result is None
        assert data["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_pre_call_hook_with_routing_override(
        self, handler, user_api_key_dict
    ):
        await handler.set_session_routing(
            session_id="routed-session",
            model="on-premise-model",
            user_api_key_dict=user_api_key_dict,
        )

        data = {
            "model": "gpt-4",
            "metadata": {"session_id": "routed-session"},
        }
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

        assert result is not None
        assert result["model"] == "on-premise-model"
        assert result["metadata"]["sensitive_data_routing_applied"] is True
        assert result["metadata"]["sensitive_data_routing_original_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_pre_call_hook_no_override_needed(self, handler, user_api_key_dict):
        data = {
            "model": "gpt-4",
            "metadata": {"session_id": "no-override-session"},
        }
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
        assert result is None
        assert data["model"] == "gpt-4"


class TestSensitiveDataRouteException:
    def test_exception_creation(self):
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="test-session",
            guardrail_name="test-guardrail",
            detection_info={"detected_entities": ["SSN", "CREDIT_CARD"]},
        )

        assert exc.route_to_model == "on-premise-model"
        assert exc.session_id == "test-session"
        assert exc.guardrail_name == "test-guardrail"
        assert "SSN" in exc.detection_info["detected_entities"]


class TestCustomGuardrailSensitiveDataRouting:
    def test_should_route_on_sensitive_data_false_by_default(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        assert guardrail.should_route_on_sensitive_data() is False

    def test_should_route_on_sensitive_data_true(self):
        guardrail = CustomGuardrail(
            guardrail_name="test",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
        )
        assert guardrail.should_route_on_sensitive_data() is True

    def test_should_route_on_sensitive_data_missing_model(self):
        guardrail = CustomGuardrail(
            guardrail_name="test",
            on_sensitive_data="route",
        )
        assert guardrail.should_route_on_sensitive_data() is False

    def test_raise_sensitive_data_route_exception(self):
        guardrail = CustomGuardrail(
            guardrail_name="test",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
        )

        request_data = {"model": "gpt-4", "metadata": {"session_id": "test-session"}}

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            guardrail.raise_sensitive_data_route_exception(
                route_to_model="on-premise-model",
                request_data=request_data,
                detection_info={"type": "PII"},
            )

        assert exc_info.value.route_to_model == "on-premise-model"
        assert exc_info.value.session_id == "test-session"

    def test_raise_exception_carries_sticky_flag_false(self):
        guardrail = CustomGuardrail(
            guardrail_name="test",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
            sticky_session_routing=False,
        )

        request_data = {"metadata": {"session_id": "test-session"}}

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            guardrail.raise_sensitive_data_route_exception(
                route_to_model="on-premise-model",
                request_data=request_data,
            )

        assert exc_info.value.sticky_session_routing is False

    def test_raise_exception_carries_sticky_flag_default_true(self):
        guardrail = CustomGuardrail(
            guardrail_name="test",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
        )

        request_data = {"metadata": {"session_id": "test-session"}}

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            guardrail.raise_sensitive_data_route_exception(
                route_to_model="on-premise-model",
                request_data=request_data,
            )

        assert exc_info.value.sticky_session_routing is True

    def test_raise_sensitive_data_route_exception_missing_session(self):
        guardrail = CustomGuardrail(guardrail_name="test")

        request_data = {"model": "gpt-4"}

        with pytest.raises(ValueError) as exc_info:
            guardrail.raise_sensitive_data_route_exception(
                route_to_model="on-premise-model",
                request_data=request_data,
            )

        assert "session_id" in str(exc_info.value)

    def test_handle_sensitive_data_detection_route(self):
        guardrail = CustomGuardrail(
            guardrail_name="test",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
        )

        request_data = {"model": "gpt-4", "metadata": {"session_id": "test-session"}}

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            guardrail.handle_sensitive_data_detection(
                request_data=request_data,
                detection_info={"type": "PII"},
            )

        assert exc_info.value.route_to_model == "on-premise-model"

    def test_handle_sensitive_data_detection_block(self):
        from litellm.exceptions import GuardrailRaisedException

        guardrail = CustomGuardrail(guardrail_name="test")

        request_data = {"model": "gpt-4", "metadata": {"session_id": "test-session"}}

        with pytest.raises(GuardrailRaisedException):
            guardrail.handle_sensitive_data_detection(
                request_data=request_data,
            )

    def test_handle_sensitive_data_detection_route_no_session_falls_back_to_block(self):
        from litellm.exceptions import GuardrailRaisedException

        guardrail = CustomGuardrail(
            guardrail_name="test",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
        )

        request_data = {"model": "gpt-4"}

        with pytest.raises(GuardrailRaisedException) as exc_info:
            guardrail.handle_sensitive_data_detection(
                request_data=request_data,
                detection_info={"type": "PII"},
            )

        assert "session_id" in str(exc_info.value)


class TestStickySessionRouting:
    @pytest.fixture
    def handler(self):
        cache = MockInternalUsageCache()
        return _PROXY_SensitiveDataRoutingHandler(internal_usage_cache=cache)

    @pytest.fixture
    def user_api_key_dict(self):
        return UserAPIKeyAuth(api_key="test-key")

    @pytest.mark.asyncio
    async def test_sticky_routing_persists(self, handler, user_api_key_dict):
        session_id = "sticky-session"
        await handler.set_session_routing(
            session_id=session_id,
            model="on-premise-model",
            user_api_key_dict=user_api_key_dict,
        )

        for i in range(5):
            data = {
                "model": f"gpt-{i}",
                "metadata": {"session_id": session_id},
            }
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data=data,
                call_type="completion",
            )

            assert result is not None
            assert result["model"] == "on-premise-model"
            assert (
                result["metadata"]["sensitive_data_routing_original_model"]
                == f"gpt-{i}"
            )

    @pytest.mark.asyncio
    async def test_different_sessions_independent(self, handler, user_api_key_dict):
        await handler.set_session_routing(
            session_id="session-a",
            model="on-premise-model-a",
            user_api_key_dict=user_api_key_dict,
        )
        await handler.set_session_routing(
            session_id="session-b",
            model="on-premise-model-b",
            user_api_key_dict=user_api_key_dict,
        )

        data_a = {"model": "gpt-4", "metadata": {"session_id": "session-a"}}
        data_b = {"model": "gpt-4", "metadata": {"session_id": "session-b"}}
        data_c = {"model": "gpt-4", "metadata": {"session_id": "session-c"}}

        result_a = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data_a,
            call_type="completion",
        )
        result_b = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data_b,
            call_type="completion",
        )
        result_c = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data_c,
            call_type="completion",
        )

        assert result_a["model"] == "on-premise-model-a"
        assert result_b["model"] == "on-premise-model-b"
        assert result_c is None

    @pytest.mark.asyncio
    async def test_routing_is_isolated_per_api_key(self, handler):
        shared_session = "shared-session-id"
        await handler.set_session_routing(
            session_id=shared_session,
            model="on-premise-model",
            user_api_key_dict=UserAPIKeyAuth(api_key="tenant-a"),
        )

        data_for_tenant_b = {
            "model": "gpt-4",
            "metadata": {"session_id": shared_session},
        }
        result = await handler.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="tenant-b"),
            cache=DualCache(),
            data=data_for_tenant_b,
            call_type="completion",
        )
        assert result is None
        assert data_for_tenant_b["model"] == "gpt-4"

        data_for_tenant_a = {
            "model": "gpt-4",
            "metadata": {"session_id": shared_session},
        }
        result = await handler.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="tenant-a"),
            cache=DualCache(),
            data=data_for_tenant_a,
            call_type="completion",
        )
        assert result is not None
        assert result["model"] == "on-premise-model"


class TestCacheKeyAndTTL:
    def test_cache_prefix_constant(self):
        assert SENSITIVE_ROUTING_CACHE_PREFIX == "sensitive_route"

    def test_default_ttl_constant(self):
        assert DEFAULT_SENSITIVE_ROUTING_TTL == 3600

    def test_make_cache_key_format(self):
        cache = MockInternalUsageCache()
        handler = _PROXY_SensitiveDataRoutingHandler(internal_usage_cache=cache)
        key = handler._make_cache_key("test-session-123", "hashed-key")
        assert key == "{sensitive_route:hashed-key:test-session-123}:model"

    def test_make_cache_key_is_tenant_scoped(self):
        cache = MockInternalUsageCache()
        handler = _PROXY_SensitiveDataRoutingHandler(internal_usage_cache=cache)
        key_a = handler._make_cache_key("shared-session", "key-a")
        key_b = handler._make_cache_key("shared-session", "key-b")
        assert key_a != key_b

    def test_resolve_tenant_prefers_api_key(self):
        tenant = _PROXY_SensitiveDataRoutingHandler._resolve_tenant(
            UserAPIKeyAuth(api_key="hashed-key", user_id="alice")
        )
        assert tenant == "hashed-key"

    def test_resolve_tenant_falls_back_to_jwt_principal(self):
        tenant = _PROXY_SensitiveDataRoutingHandler._resolve_tenant(
            UserAPIKeyAuth(api_key=None, user_id="alice", team_id="t1", org_id="o1")
        )
        assert tenant == "user:alice|team:t1|org:o1"

    def test_resolve_tenant_distinguishes_keyless_principals(self):
        tenant_a = _PROXY_SensitiveDataRoutingHandler._resolve_tenant(
            UserAPIKeyAuth(api_key=None, user_id="alice")
        )
        tenant_b = _PROXY_SensitiveDataRoutingHandler._resolve_tenant(
            UserAPIKeyAuth(api_key=None, user_id="bob")
        )
        assert tenant_a != tenant_b

    def test_resolve_tenant_defaults_when_anonymous(self):
        assert _PROXY_SensitiveDataRoutingHandler._resolve_tenant(None) == "default"
        assert (
            _PROXY_SensitiveDataRoutingHandler._resolve_tenant(
                UserAPIKeyAuth(api_key=None)
            )
            == "default"
        )


class TestCustomGuardrailSessionIdExtraction:
    def test_get_session_id_from_litellm_session_id(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        request_data = {"litellm_session_id": "session-direct-123"}
        session_id = guardrail._get_session_id_from_request_data(request_data)
        assert session_id == "session-direct-123"

    def test_get_session_id_from_metadata(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        request_data = {"metadata": {"session_id": "session-metadata-456"}}
        session_id = guardrail._get_session_id_from_request_data(request_data)
        assert session_id == "session-metadata-456"

    def test_get_session_id_from_litellm_metadata(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        request_data = {"litellm_metadata": {"session_id": "session-litellm-meta-789"}}
        session_id = guardrail._get_session_id_from_request_data(request_data)
        assert session_id == "session-litellm-meta-789"

    def test_get_session_id_returns_none_when_missing(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        request_data = {"model": "gpt-4"}
        session_id = guardrail._get_session_id_from_request_data(request_data)
        assert session_id is None

    def test_get_session_id_priority_litellm_session_id_first(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        request_data = {
            "litellm_session_id": "priority-session",
            "metadata": {"session_id": "should-not-use"},
            "litellm_metadata": {"session_id": "also-not-this"},
        }
        session_id = guardrail._get_session_id_from_request_data(request_data)
        assert session_id == "priority-session"

    def test_get_session_id_converts_to_string(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        request_data = {"litellm_session_id": 12345}
        session_id = guardrail._get_session_id_from_request_data(request_data)
        assert session_id == "12345"
        assert isinstance(session_id, str)


class TestCustomGuardrailInit:
    def test_init_with_routing_config(self):
        guardrail = CustomGuardrail(
            guardrail_name="test-guardrail",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
            sticky_session_routing=True,
        )
        assert guardrail.on_sensitive_data == "route"
        assert guardrail.sensitive_data_route_to_model == "on-premise-model"
        assert guardrail.sticky_session_routing is True

    def test_init_default_values(self):
        guardrail = CustomGuardrail(guardrail_name="test")
        assert guardrail.on_sensitive_data is None
        assert guardrail.sensitive_data_route_to_model is None
        assert guardrail.sticky_session_routing is True


class TestSensitiveDataRouteExceptionStr:
    def test_exception_str_representation(self):
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="test-session",
            guardrail_name="pii-detector",
        )
        assert (
            str(exc)
            == "Sensitive data detected by pii-detector. Routing to model: on-premise-model"
        )

    def test_exception_custom_message(self):
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="test-session",
            guardrail_name="pii-detector",
            message="Custom error message",
        )
        assert str(exc) == "Custom error message"
        assert exc.message == "Custom error message"


class TestRedisCache:
    @pytest.fixture
    def handler_with_redis(self):
        cache = MockInternalUsageCache()
        mock_redis = AsyncMock()
        cache.dual_cache.redis_cache = mock_redis
        return _PROXY_SensitiveDataRoutingHandler(internal_usage_cache=cache)

    @pytest.mark.asyncio
    async def test_get_routed_model_from_redis(self, handler_with_redis):
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_cache = AsyncMock(
            return_value="redis-model"
        )
        result = await handler_with_redis._get_routed_model(
            "session-123", UserAPIKeyAuth(api_key="hashed-key")
        )
        assert result == "redis-model"

    @pytest.mark.asyncio
    async def test_get_routed_model_backfills_in_memory_after_redis_hit(
        self, handler_with_redis
    ):
        cache_key = "{sensitive_route:hashed-key:session-123}:model"
        key = UserAPIKeyAuth(api_key="hashed-key")
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_cache = AsyncMock(
            return_value="on-premise-model"
        )
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_ttl = (
            AsyncMock(return_value=120)
        )

        first = await handler_with_redis._get_routed_model("session-123", key)
        assert first == "on-premise-model"
        assert handler_with_redis.internal_usage_cache._cache[cache_key] == (
            "on-premise-model"
        )

        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_cache = AsyncMock(
            side_effect=Exception("Redis went down")
        )
        second = await handler_with_redis._get_routed_model("session-123", key)
        assert second == "on-premise-model"

    @pytest.mark.asyncio
    async def test_backfill_uses_remaining_redis_ttl(self, handler_with_redis):
        cache_key = "{sensitive_route:hashed-key:session-123}:model"
        key = UserAPIKeyAuth(api_key="hashed-key")
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_cache = AsyncMock(
            return_value="on-premise-model"
        )
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_ttl = (
            AsyncMock(return_value=42)
        )

        await handler_with_redis._get_routed_model("session-123", key)

        assert handler_with_redis.internal_usage_cache._ttls[cache_key] == 42

    @pytest.mark.asyncio
    async def test_backfill_falls_back_to_full_ttl_when_redis_ttl_missing(
        self, handler_with_redis
    ):
        cache_key = "{sensitive_route:hashed-key:session-123}:model"
        key = UserAPIKeyAuth(api_key="hashed-key")
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_cache = AsyncMock(
            return_value="on-premise-model"
        )
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_ttl = (
            AsyncMock(return_value=None)
        )

        await handler_with_redis._get_routed_model("session-123", key)

        assert (
            handler_with_redis.internal_usage_cache._ttls[cache_key]
            == handler_with_redis.ttl
        )

    @pytest.mark.asyncio
    async def test_get_routed_model_redis_fallback_on_error(self, handler_with_redis):
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_get_cache = AsyncMock(
            side_effect=Exception("Redis connection error")
        )
        handler_with_redis.internal_usage_cache._cache[
            "{sensitive_route:hashed-key:session-123}:model"
        ] = "fallback-model"
        result = await handler_with_redis._get_routed_model(
            "session-123", UserAPIKeyAuth(api_key="hashed-key")
        )
        assert result == "fallback-model"

    @pytest.mark.asyncio
    async def test_set_session_routing_with_redis(self, handler_with_redis):
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_set_cache = (
            AsyncMock()
        )
        await handler_with_redis.set_session_routing(
            session_id="session-456",
            model="on-premise-model",
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
            guardrail_name="test-guardrail",
        )
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_set_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_session_routing_redis_fallback_on_error(
        self, handler_with_redis
    ):
        handler_with_redis.internal_usage_cache.dual_cache.redis_cache.async_set_cache = AsyncMock(
            side_effect=Exception("Redis connection error")
        )
        await handler_with_redis.set_session_routing(
            session_id="session-789",
            model="on-premise-model",
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
        )
        cache_key = "{sensitive_route:hashed-key:session-789}:model"
        assert (
            handler_with_redis.internal_usage_cache._cache[cache_key]
            == "on-premise-model"
        )


class TestPreCallHookEdgeCases:
    @pytest.fixture
    def handler(self):
        cache = MockInternalUsageCache()
        return _PROXY_SensitiveDataRoutingHandler(internal_usage_cache=cache)

    @pytest.fixture
    def user_api_key_dict(self):
        return UserAPIKeyAuth(api_key="test-key")

    @pytest.mark.asyncio
    async def test_pre_call_hook_same_model_no_change(self, handler, user_api_key_dict):
        await handler.set_session_routing(
            session_id="same-model-session",
            model="gpt-4",
            user_api_key_dict=user_api_key_dict,
        )
        data = {
            "model": "gpt-4",
            "metadata": {"session_id": "same-model-session"},
        }
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
        assert result is None


class TestHandleSensitiveDataDetectionWithRouting:
    def test_handle_sensitive_data_detection_full_flow(self):
        guardrail = CustomGuardrail(
            guardrail_name="pii-guardrail",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-premise-model",
        )

        request_data = {
            "model": "gpt-4",
            "metadata": {"session_id": "flow-test-session"},
            "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}],
        }

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            guardrail.handle_sensitive_data_detection(
                request_data=request_data,
                detection_info={"detected_entities": ["SSN"]},
            )

        exc = exc_info.value
        assert exc.route_to_model == "on-premise-model"
        assert exc.session_id == "flow-test-session"
        assert exc.guardrail_name == "pii-guardrail"
        assert exc.detection_info == {"detected_entities": ["SSN"]}


class TestProxyHandleSensitiveDataRouteException:
    @pytest.fixture
    def proxy_logging(self):
        from litellm.proxy.utils import ProxyLogging

        return ProxyLogging(user_api_key_cache=DualCache())

    @pytest.fixture
    def routing_hook(self):
        cache = MockInternalUsageCache()
        return _PROXY_SensitiveDataRoutingHandler(internal_usage_cache=cache)

    @pytest.mark.asyncio
    async def test_sticky_routing_persists_override(self, proxy_logging, routing_hook):
        proxy_logging.proxy_hook_mapping["sensitive_data_routing"] = routing_hook
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="sess-sticky",
            guardrail_name="pii",
            sticky_session_routing=True,
        )
        data = {"model": "gpt-4", "metadata": {"session_id": "sess-sticky"}}

        result = await proxy_logging._handle_sensitive_data_route_exception(
            exc, data, UserAPIKeyAuth(api_key="tenant-a")
        )

        assert result["model"] == "on-premise-model"
        assert (
            await routing_hook._get_routed_model(
                "sess-sticky", UserAPIKeyAuth(api_key="tenant-a")
            )
            == "on-premise-model"
        )

    @pytest.mark.asyncio
    async def test_non_sticky_routing_does_not_persist_override(
        self, proxy_logging, routing_hook
    ):
        proxy_logging.proxy_hook_mapping["sensitive_data_routing"] = routing_hook
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="sess-non-sticky",
            guardrail_name="pii",
            sticky_session_routing=False,
        )
        data = {"model": "gpt-4", "metadata": {"session_id": "sess-non-sticky"}}

        result = await proxy_logging._handle_sensitive_data_route_exception(
            exc, data, UserAPIKeyAuth(api_key="tenant-a")
        )

        assert result["model"] == "on-premise-model"
        assert (
            await routing_hook._get_routed_model(
                "sess-non-sticky", UserAPIKeyAuth(api_key="tenant-a")
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_sticky_routing_handles_none_user_api_key_dict(
        self, proxy_logging, routing_hook
    ):
        proxy_logging.proxy_hook_mapping["sensitive_data_routing"] = routing_hook
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="sess-no-key",
            guardrail_name="pii",
            sticky_session_routing=True,
        )
        data = {"model": "gpt-4", "metadata": {"session_id": "sess-no-key"}}

        result = await proxy_logging._handle_sensitive_data_route_exception(
            exc, data, None
        )

        assert result["model"] == "on-premise-model"
        assert (
            await routing_hook._get_routed_model("sess-no-key", None)
            == "on-premise-model"
        )

    @pytest.mark.asyncio
    async def test_sticky_routing_scopes_jwt_users_by_principal(
        self, proxy_logging, routing_hook
    ):
        proxy_logging.proxy_hook_mapping["sensitive_data_routing"] = routing_hook
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="shared-jwt-session",
            guardrail_name="pii",
            sticky_session_routing=True,
        )
        attacker = UserAPIKeyAuth(api_key=None, user_id="attacker", team_id="team-x")
        await proxy_logging._handle_sensitive_data_route_exception(
            exc,
            {"model": "gpt-4", "metadata": {"session_id": "shared-jwt-session"}},
            attacker,
        )

        victim = UserAPIKeyAuth(api_key=None, user_id="victim", team_id="team-y")
        victim_data = {
            "model": "gpt-4",
            "metadata": {"session_id": "shared-jwt-session"},
        }
        result = await routing_hook.async_pre_call_hook(
            user_api_key_dict=victim,
            cache=DualCache(),
            data=victim_data,
            call_type="completion",
        )
        assert result is None
        assert victim_data["model"] == "gpt-4"

        attacker_data = {
            "model": "gpt-4",
            "metadata": {"session_id": "shared-jwt-session"},
        }
        result = await routing_hook.async_pre_call_hook(
            user_api_key_dict=attacker,
            cache=DualCache(),
            data=attacker_data,
            call_type="completion",
        )
        assert result is not None
        assert result["model"] == "on-premise-model"

    @pytest.mark.asyncio
    async def test_sticky_routing_warns_when_hook_not_registered(self, proxy_logging):
        exc = SensitiveDataRouteException(
            route_to_model="on-premise-model",
            session_id="sess-no-hook",
            sticky_session_routing=True,
        )
        data = {"model": "gpt-4", "metadata": {"session_id": "sess-no-hook"}}

        with patch("litellm.proxy.utils.verbose_proxy_logger.warning") as mock_warning:
            result = await proxy_logging._handle_sensitive_data_route_exception(
                exc, data, UserAPIKeyAuth(api_key="tenant-a")
            )

        assert result["model"] == "on-premise-model"
        mock_warning.assert_called_once()


class _RoutingGuardrail(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.handle_sensitive_data_detection(request_data=data)


class _RecordingGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ran = False

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.ran = True
        return None


class _BlockingGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ran = False

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        from litellm.exceptions import GuardrailRaisedException

        self.ran = True
        raise GuardrailRaisedException(
            message="blocked", guardrail_name=self.guardrail_name
        )


class TestPreCallHookDeferredRouting:
    """Guardrails after the one that triggers routing must still run."""

    @pytest.fixture
    def proxy_logging(self):
        from litellm.proxy.utils import ProxyLogging

        return ProxyLogging(user_api_key_cache=DualCache())

    @pytest.fixture(autouse=True)
    def restore_callbacks(self):
        import litellm

        original = litellm.callbacks
        litellm.callbacks = []
        yield
        litellm.callbacks = original

    @pytest.mark.asyncio
    async def test_later_guardrail_runs_and_routing_applied(self, proxy_logging):
        import litellm

        router = _RoutingGuardrail(
            guardrail_name="router",
            default_on=True,
            event_hook="pre_call",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-prem-model",
            sticky_session_routing=False,
        )
        recorder = _RecordingGuardrail(
            guardrail_name="recorder",
            default_on=True,
            event_hook="pre_call",
        )
        litellm.callbacks = [router, recorder]

        data = {"model": "gpt-4", "metadata": {"session_id": "sess-defer"}}
        result = await proxy_logging.pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="tenant-a"),
            data=data,
            call_type="completion",
        )

        assert recorder.ran is True
        assert result["model"] == "on-prem-model"
        assert result["metadata"]["sensitive_data_routing_applied"] is True

    @pytest.mark.asyncio
    async def test_later_blocking_guardrail_overrides_routing(self, proxy_logging):
        import litellm
        from litellm.exceptions import GuardrailRaisedException

        router = _RoutingGuardrail(
            guardrail_name="router",
            default_on=True,
            event_hook="pre_call",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-prem-model",
            sticky_session_routing=False,
        )
        blocker = _BlockingGuardrail(
            guardrail_name="blocker",
            default_on=True,
            event_hook="pre_call",
        )
        litellm.callbacks = [router, blocker]

        data = {"model": "gpt-4", "metadata": {"session_id": "sess-block"}}
        with pytest.raises(GuardrailRaisedException):
            await proxy_logging.pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="tenant-a"),
                data=data,
                call_type="completion",
            )

        assert blocker.ran is True

    @pytest.mark.asyncio
    async def test_routing_guardrail_records_service_span(self, proxy_logging):
        import litellm
        from litellm.types.services import ServiceTypes

        class _SlowRoutingGuardrail(CustomGuardrail):
            async def async_pre_call_hook(
                self, user_api_key_dict, cache, data, call_type
            ):
                await asyncio.sleep(0.02)
                self.handle_sensitive_data_detection(request_data=data)

        router = _SlowRoutingGuardrail(
            guardrail_name="router",
            default_on=True,
            event_hook="pre_call",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-prem-model",
            sticky_session_routing=False,
        )
        litellm.callbacks = [router]

        recorded = AsyncMock()
        proxy_logging.service_logging_obj.async_service_success_hook = recorded

        data = {"model": "gpt-4", "metadata": {"session_id": "sess-span"}}
        result = await proxy_logging.pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="tenant-a"),
            data=data,
            call_type="completion",
        )

        assert result["model"] == "on-prem-model"
        recorded.assert_called_once()
        assert recorded.call_args.kwargs["call_type"] == "_SlowRoutingGuardrail"
        assert recorded.call_args.kwargs["service"] == ServiceTypes.PROXY_PRE_CALL

    @pytest.mark.asyncio
    async def test_routing_recorded_as_intervention_not_prometheus_error(
        self, proxy_logging
    ):
        import litellm
        from litellm.integrations.prometheus import PrometheusLogger

        router = _RoutingGuardrail(
            guardrail_name="router",
            default_on=True,
            event_hook="pre_call",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-prem-model",
            sticky_session_routing=False,
        )
        prom = MagicMock(spec=PrometheusLogger)
        litellm.callbacks = [router, prom]

        data = {"model": "gpt-4", "metadata": {"session_id": "sess-prom"}}
        result = await proxy_logging.pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="tenant-a"),
            data=data,
            call_type="completion",
        )

        assert result["model"] == "on-prem-model"
        prom._record_guardrail_metrics.assert_called_once()
        metrics_kwargs = prom._record_guardrail_metrics.call_args.kwargs
        assert metrics_kwargs["status"] == "intervened"
        assert metrics_kwargs["error_type"] is None
