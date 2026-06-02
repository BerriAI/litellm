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
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.sensitive_data_routing import (
    _PROXY_SensitiveDataRoutingHandler,
)


class MockInternalUsageCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self.dual_cache = MagicMock()
        self.dual_cache.redis_cache = None

    async def async_get_cache(self, key: str, **kwargs) -> Optional[Any]:
        return self._cache.get(key)

    async def async_set_cache(self, key: str, value: Any, ttl: int = 3600, **kwargs):
        self._cache[key] = value


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
        await handler.set_session_routing(
            session_id="test-session-123",
            model="on-premise-model",
            guardrail_name="test-guardrail",
        )

        routed_model = await handler._get_routed_model("test-session-123")
        assert routed_model == "on-premise-model"

    @pytest.mark.asyncio
    async def test_get_session_id_from_metadata(self, handler):
        data = {"metadata": {"session_id": "session-from-metadata"}}
        session_id = handler._get_session_id(data)
        assert session_id == "session-from-metadata"

    @pytest.mark.asyncio
    async def test_get_session_id_from_litellm_metadata(self, handler):
        data = {"litellm_metadata": {"session_id": "session-from-litellm-metadata"}}
        session_id = handler._get_session_id(data)
        assert session_id == "session-from-litellm-metadata"

    @pytest.mark.asyncio
    async def test_get_session_id_from_litellm_session_id(self, handler):
        data = {"litellm_session_id": "session-direct"}
        session_id = handler._get_session_id(data)
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
            assert result["metadata"]["sensitive_data_routing_original_model"] == f"gpt-{i}"

    @pytest.mark.asyncio
    async def test_different_sessions_independent(self, handler, user_api_key_dict):
        await handler.set_session_routing(
            session_id="session-a",
            model="on-premise-model-a",
        )
        await handler.set_session_routing(
            session_id="session-b",
            model="on-premise-model-b",
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
