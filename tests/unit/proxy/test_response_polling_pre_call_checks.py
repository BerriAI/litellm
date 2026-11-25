"""
Unit tests for pre-call checks running before polling ID creation.

Tests that rate limits, guardrails, and budget checks are enforced
BEFORE a polling ID is created, so rate-limited requests get a
synchronous error instead of a polling ID that immediately fails.
"""

from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, Response


import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


class TestSkipPreCallLogic:
    """Test that skip_pre_call_logic parameter works correctly"""

    @pytest.mark.asyncio
    async def test_skip_pre_call_logic_skips_common_processing(self):
        """When skip_pre_call_logic=True, common_processing_pre_call_logic should not be called"""
        mock_logging_obj = MagicMock()
        data = {
            "model": "gpt-4",
            "stream": True,
            "litellm_logging_obj": mock_logging_obj,
        }
        processor = ProxyBaseLLMRequestProcessing(data=data)

        mock_proxy_logging = AsyncMock()
        mock_proxy_logging.during_call_hook = AsyncMock()

        with (
            patch.object(processor, "common_processing_pre_call_logic", new_callable=AsyncMock) as mock_pre_call,
            patch(
                "litellm.proxy.common_request_processing.route_request",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            try:
                await processor.base_process_llm_request(
                    request=MagicMock(spec=Request),
                    fastapi_response=MagicMock(spec=Response),
                    user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                    route_type="aresponses",
                    proxy_logging_obj=mock_proxy_logging,
                    llm_router=MagicMock(),
                    general_settings={},
                    proxy_config=MagicMock(),
                    skip_pre_call_logic=True,
                )
            except Exception:
                pass  # We only care that common_processing_pre_call_logic was not called

            mock_pre_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_without_skip_runs_common_processing(self):
        """When skip_pre_call_logic=False (default), common_processing_pre_call_logic should be called"""
        data = {"model": "gpt-4"}
        processor = ProxyBaseLLMRequestProcessing(data=data)

        mock_logging_obj = MagicMock()
        mock_proxy_logging = AsyncMock()
        mock_proxy_logging.during_call_hook = AsyncMock()

        with (
            patch.object(
                processor,
                "common_processing_pre_call_logic",
                new_callable=AsyncMock,
                return_value=(data, mock_logging_obj),
            ) as mock_pre_call,
            patch(
                "litellm.proxy.common_request_processing.route_request",
                new_callable=AsyncMock,
            ),
        ):
            try:
                await processor.base_process_llm_request(
                    request=MagicMock(spec=Request, headers={}),
                    fastapi_response=MagicMock(spec=Response),
                    user_api_key_dict=UserAPIKeyAuth(),
                    route_type="aresponses",
                    proxy_logging_obj=mock_proxy_logging,
                    llm_router=MagicMock(),
                    general_settings={},
                    proxy_config=MagicMock(get_hierarchical_router_settings=AsyncMock(return_value=None)),
                )
            except Exception:
                pass

            mock_pre_call.assert_called_once()


class TestPollingEndpointPreCallGuard:
    """Test that the polling endpoint enforces pre-call checks before polling ID creation"""

    @pytest.mark.asyncio
    async def test_rate_limit_error_prevents_polling_id_creation(self):
        """responses_api() must raise 429 and never call generate_polling_id when rate-limited"""
        from litellm.proxy.response_api_endpoints.endpoints import responses_api
        from litellm.proxy.response_polling.polling_handler import (
            ResponsePollingHandler,
        )

        rate_limit_exc = litellm.RateLimitError(
            message="TPM limit exceeded",
            llm_provider="",
            model="gpt-4",
        )
        generate_polling_id_mock = MagicMock(return_value="litellm_poll_test")

        proxy_server_patches = {
            "litellm.proxy.proxy_server.read_request_body": AsyncMock(
                return_value={"model": "gpt-4", "background": True}
            ),
            "litellm.proxy.proxy_server.general_settings": {},
            "litellm.proxy.proxy_server.llm_router": MagicMock(),
            "litellm.proxy.proxy_server.native_background_mode": None,
            "litellm.proxy.proxy_server.polling_cache_ttl": 3600,
            "litellm.proxy.proxy_server.polling_via_cache_enabled": True,
            "litellm.proxy.proxy_server.proxy_config": MagicMock(),
            "litellm.proxy.proxy_server.proxy_logging_obj": AsyncMock(),
            "litellm.proxy.proxy_server.redis_usage_cache": AsyncMock(),
            "litellm.proxy.proxy_server.user_api_base": None,
            "litellm.proxy.proxy_server.user_max_tokens": None,
            "litellm.proxy.proxy_server.user_model": None,
            "litellm.proxy.proxy_server.user_request_timeout": None,
            "litellm.proxy.proxy_server.user_temperature": None,
            "litellm.proxy.proxy_server.version": "1.0.0",
        }

        with (
            patch.multiple(
                "litellm.proxy.proxy_server",
                **{k.split(".")[-1]: v for k, v in proxy_server_patches.items()},
            ),
            patch(
                "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
                return_value=True,
            ),
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "common_processing_pre_call_logic",
                new_callable=AsyncMock,
                side_effect=rate_limit_exc,
            ),
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "handle_llm_api_exception",
                new_callable=AsyncMock,
                return_value=HTTPException(status_code=429, detail="Rate limit exceeded"),
            ),
            patch.object(ResponsePollingHandler, "generate_polling_id", generate_polling_id_mock),
            # Prevent background task from running (avoids noise from incomplete mocks)
            patch("asyncio.create_task"),
            patch.object(
                ResponsePollingHandler,
                "create_initial_state",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await responses_api(
                    request=MagicMock(spec=Request),
                    fastapi_response=MagicMock(spec=Response),
                    user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                )

        assert exc_info.value.status_code == 429
        generate_polling_id_mock.assert_not_called()


async def _polled_background_response(
    body: Mapping[str, object], headers: Mapping[str, str]
) -> tuple[MagicMock, Mapping[str, object]]:
    from litellm.proxy.response_api_endpoints.endpoints import responses_api
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    generate_polling_id: Final = MagicMock(return_value="litellm_poll_test")
    pre_call_saw: Final[list[Mapping[str, object]]] = []

    async def pre_call(processor: ProxyBaseLLMRequestProcessing, **_: object) -> tuple[Mapping[str, object], MagicMock]:
        pre_call_saw.append(dict(processor.data))
        return processor.data, MagicMock()

    request: Final = MagicMock(spec=Request)
    request.headers = headers
    proxy_server_patches: Final[Mapping[str, object]] = {
        "read_request_body": AsyncMock(return_value=dict(body)),
        "general_settings": {},
        "llm_router": MagicMock(),
        "native_background_mode": None,
        "polling_cache_ttl": 3600,
        "polling_via_cache_enabled": True,
        "proxy_config": MagicMock(),
        "proxy_logging_obj": AsyncMock(),
        "redis_usage_cache": AsyncMock(),
        "user_api_base": None,
        "user_max_tokens": None,
        "user_model": None,
        "user_request_timeout": None,
        "user_temperature": None,
        "version": "1.0.0",
    }
    with (
        patch.multiple("litellm.proxy.proxy_server", **proxy_server_patches),
        patch("litellm.proxy.response_polling.polling_handler.should_use_polling_for_request", return_value=True),
        patch.object(ProxyBaseLLMRequestProcessing, "common_processing_pre_call_logic", pre_call),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "handle_llm_api_exception",
            new_callable=AsyncMock,
            side_effect=lambda e, **_: e,
        ),
        patch("litellm.proxy.response_api_endpoints.endpoints.raise_if_required_body_param_missing"),
        patch.object(ResponsePollingHandler, "generate_polling_id", generate_polling_id),
        patch("asyncio.create_task"),
        patch.object(ResponsePollingHandler, "create_initial_state", new_callable=AsyncMock, return_value=MagicMock()),
    ):
        await responses_api(
            request=request, fastapi_response=MagicMock(spec=Response), user_api_key_dict=MagicMock(spec=UserAPIKeyAuth)
        )
    return generate_polling_id, pre_call_saw[0] if pre_call_saw else {}


@pytest.mark.asyncio
async def test_a_body_total_timeout_on_a_polled_background_response_is_refused_before_a_polling_id_is_issued() -> None:
    from litellm.proxy._types import ProxyException

    with pytest.raises(ProxyException) as refused:
        _ = await _polled_background_response({"model": "gpt-4", "background": True, "total_timeout": 10}, {})

    assert refused.value.code == "400"
    assert refused.value.param == "total_timeout"


@pytest.mark.asyncio
async def test_a_header_total_timeout_on_a_polled_background_response_is_dropped_and_polling_proceeds() -> None:
    generate_polling_id, pre_call_saw = await _polled_background_response(
        {"model": "gpt-4", "background": True, "total_timeout": "ignored"}, {"x-litellm-total-timeout": "10"}
    )

    generate_polling_id.assert_called_once()
    assert "total_timeout" not in pre_call_saw
