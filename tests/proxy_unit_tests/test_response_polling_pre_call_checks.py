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
from fastapi import Request, Response

sys.path.insert(0, os.path.abspath("../.."))

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


