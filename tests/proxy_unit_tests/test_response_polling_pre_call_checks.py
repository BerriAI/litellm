"""
Unit tests for pre-call checks running before polling ID creation.

Tests that rate limits, guardrails, and budget checks are enforced
BEFORE a polling ID is created, so rate-limited requests get a
synchronous error instead of a polling ID that immediately fails.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, Response

sys.path.insert(0, os.path.abspath("../.."))

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
            patch.object(
                processor, "common_processing_pre_call_logic", new_callable=AsyncMock
            ) as mock_pre_call,
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
                    request=MagicMock(spec=Request),
                    fastapi_response=MagicMock(spec=Response),
                    user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                    route_type="aresponses",
                    proxy_logging_obj=mock_proxy_logging,
                    llm_router=MagicMock(),
                    general_settings={},
                    proxy_config=MagicMock(),
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
            "litellm.proxy.proxy_server._read_request_body": AsyncMock(
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
            "litellm.proxy.proxy_server.select_data_generator": None,
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
                "_handle_llm_api_exception",
                new_callable=AsyncMock,
                return_value=HTTPException(
                    status_code=429, detail="Rate limit exceeded"
                ),
            ),
            patch.object(
                ResponsePollingHandler, "generate_polling_id", generate_polling_id_mock
            ),
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
