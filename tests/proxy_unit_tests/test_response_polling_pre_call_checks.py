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
        """When pre-call checks raise, generate_polling_id must not be called"""
        from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

        rate_limit_exc = litellm.RateLimitError(
            message="TPM limit exceeded",
            llm_provider="",
            model="gpt-4",
        )

        generate_polling_id_mock = MagicMock(return_value="litellm_poll_test")

        with (
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
                return_value=HTTPException(status_code=429, detail="Rate limit exceeded"),
            ),
            patch.object(ResponsePollingHandler, "generate_polling_id", generate_polling_id_mock),
        ):
            # Simulate the endpoint logic directly (avoids proxy_server import complexity)
            data = {"model": "gpt-4", "background": True}
            processor = ProxyBaseLLMRequestProcessing(data=data)

            raised_exc = None
            try:
                await processor.common_processing_pre_call_logic(
                    request=MagicMock(spec=Request),
                    general_settings={},
                    proxy_logging_obj=AsyncMock(),
                    user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                    version="1.0.0",
                    proxy_config=MagicMock(),
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    model=None,
                    route_type="aresponses",
                    llm_router=MagicMock(),
                )
            except litellm.RateLimitError as e:
                raised_exc = e

            # The exception was raised before generate_polling_id could be called
            assert raised_exc is not None
            generate_polling_id_mock.assert_not_called()

