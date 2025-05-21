import copy
import uuid
import pytest
import litellm
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request

from litellm.integrations.opentelemetry import UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    ProxyConfig,
)
from litellm.proxy.utils import ProxyLogging


class TestProxyBaseLLMRequestProcessing:

    @pytest.mark.asyncio
    async def test_common_processing_pre_call_logic_pre_call_hook_receives_litellm_call_id(
        self, monkeypatch
    ):

        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return {}

        async def mock_common_processing_pre_call_logic(
            user_api_key_dict, data, call_type
        ):
            data_copy = copy.deepcopy(data)
            return data_copy

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(
            side_effect=mock_common_processing_pre_call_logic
        )
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )
        mock_general_settings = {}
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_proxy_config = MagicMock(spec=ProxyConfig)
        route_type = "acompletion"

        # Call the actual method.
        returned_data, logging_obj = (
            await processing_obj.common_processing_pre_call_logic(
                request=mock_request,
                general_settings=mock_general_settings,
                user_api_key_dict=mock_user_api_key_dict,
                proxy_logging_obj=mock_proxy_logging_obj,
                proxy_config=mock_proxy_config,
                route_type=route_type,
            )
        )

        mock_proxy_logging_obj.pre_call_hook.assert_called_once()

        _, call_kwargs = mock_proxy_logging_obj.pre_call_hook.call_args
        data_passed = call_kwargs.get("data", {})

        assert "litellm_call_id" in data_passed
        try:
            uuid.UUID(data_passed["litellm_call_id"])
        except ValueError:
            pytest.fail("litellm_call_id is not a valid UUID")
        assert data_passed["litellm_call_id"] == returned_data["litellm_call_id"]
