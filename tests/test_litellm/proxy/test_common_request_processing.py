import copy
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse, StreamingResponse

import litellm
from litellm._uuid import uuid
from litellm.integrations.opentelemetry import UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    ProxyConfig,
    _extract_error_from_sse_chunk,
    _get_cost_breakdown_from_logging_obj,
    _override_openai_response_model,
    _parse_event_data_for_error,
    create_response,
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
        (
            returned_data,
            logging_obj,
        ) = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings=mock_general_settings,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=mock_proxy_config,
            route_type=route_type,
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

    @pytest.mark.asyncio
    async def test_should_apply_hierarchical_router_settings_as_override(
        self, monkeypatch
    ):
        """
        Test that hierarchical router settings are stored as router_settings_override
        instead of creating a full user_config with model_list.
        
        This approach avoids expensive per-request Router instantiation by passing
        settings as kwargs overrides to the main router.
        """
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
        
        mock_router_settings = {
            "routing_strategy": "least-busy",
            "timeout": 30.0,
            "num_retries": 3,
        }
        mock_proxy_config._get_hierarchical_router_settings = AsyncMock(
            return_value=mock_router_settings
        )

        mock_llm_router = MagicMock()

        mock_prisma_client = MagicMock()
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma_client,
        )

        route_type = "acompletion"

        returned_data, logging_obj = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings=mock_general_settings,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=mock_proxy_config,
            route_type=route_type,
            llm_router=mock_llm_router,
        )

        mock_proxy_config._get_hierarchical_router_settings.assert_called_once_with(
            user_api_key_dict=mock_user_api_key_dict,
            prisma_client=mock_prisma_client,
            proxy_logging_obj=mock_proxy_logging_obj,
        )
        # get_model_list should NOT be called - we no longer copy model list for per-request routers
        mock_llm_router.get_model_list.assert_not_called()

        # Settings should be stored as router_settings_override (not user_config)
        # This allows passing them as kwargs to the main router instead of creating a new one
        assert "router_settings_override" in returned_data
        assert "user_config" not in returned_data
        
        router_settings_override = returned_data["router_settings_override"]
        assert router_settings_override["routing_strategy"] == "least-busy"
        assert router_settings_override["timeout"] == 30.0
        assert router_settings_override["num_retries"] == 3
        # model_list should NOT be in the override settings
        assert "model_list" not in router_settings_override

    @pytest.mark.asyncio
    async def test_stream_timeout_header_processing(self):
        """
        Test that x-litellm-stream-timeout header gets processed and added to request data as stream_timeout.
        """
        from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

        # Test with stream timeout header
        headers_with_timeout = {"x-litellm-stream-timeout": "30.5"}
        result = LiteLLMProxyRequestSetup._get_stream_timeout_from_request(headers_with_timeout)
        assert result == 30.5
        
        # Test without stream timeout header
        headers_without_timeout = {}
        result = LiteLLMProxyRequestSetup._get_stream_timeout_from_request(headers_without_timeout)
        assert result is None
        
        # Test with invalid header value (should raise ValueError when converting to float)
        headers_with_invalid = {"x-litellm-stream-timeout": "invalid"}
        with pytest.raises(ValueError):
            LiteLLMProxyRequestSetup._get_stream_timeout_from_request(headers_with_invalid)

    @pytest.mark.asyncio
    async def test_add_litellm_data_to_request_with_stream_timeout_header(self):
        """
        Test that x-litellm-stream-timeout header gets processed and added to request data 
        when calling add_litellm_data_to_request.
        """
        from litellm.integrations.opentelemetry import UserAPIKeyAuth
        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

        # Create test data with a basic completion request
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        # Mock request with stream timeout header
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"x-litellm-stream-timeout": "45.0"}
        mock_request.url.path = "/v1/chat/completions"
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.client = None
        
        # Create a minimal mock with just the required attributes
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test_api_key_hash"
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0
        mock_user_api_key_dict.allowed_model_region = None
        mock_user_api_key_dict.key_alias = None
        mock_user_api_key_dict.user_id = None
        mock_user_api_key_dict.team_id = None
        mock_user_api_key_dict.metadata = {}  # Prevent enterprise feature check
        mock_user_api_key_dict.team_metadata = None
        mock_user_api_key_dict.org_id = None
        mock_user_api_key_dict.team_alias = None
        mock_user_api_key_dict.end_user_id = None
        mock_user_api_key_dict.user_email = None
        mock_user_api_key_dict.request_route = None
        mock_user_api_key_dict.team_max_budget = None
        mock_user_api_key_dict.team_spend = None
        mock_user_api_key_dict.model_max_budget = None
        mock_user_api_key_dict.parent_otel_span = None
        mock_user_api_key_dict.team_model_aliases = None
        
        general_settings = {}
        mock_proxy_config = MagicMock()
        
        # Call the actual function that processes headers and adds data
        result_data = await add_litellm_data_to_request(
            data=test_data,
            request=mock_request,
            general_settings=general_settings,
            user_api_key_dict=mock_user_api_key_dict,
            version=None,
            proxy_config=mock_proxy_config,
        )
        
        # Verify that stream_timeout was extracted from header and added to request data
        assert "stream_timeout" in result_data
        assert result_data["stream_timeout"] == 45.0
        
        # Verify that the original test data is preserved
        assert result_data["model"] == "gpt-3.5-turbo"
        assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    def test_get_custom_headers_with_discount_info(self):
        """
        Test that discount information is correctly extracted from logging object
        and included in response headers.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0
        
        # Create logging object with cost breakdown including discount
        logging_obj = LiteLLMLoggingObj(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        
        # Set cost breakdown with discount information
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.000095,  # After 5% discount
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            discount_percent=0.05,
            discount_amount=0.000005,
        )
        
        # Call get_custom_headers with discount info
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            response_cost=0.000095,
            litellm_logging_obj=logging_obj,
        )
        
        # Verify discount headers are present
        assert "x-litellm-response-cost" in headers
        assert float(headers["x-litellm-response-cost"]) == 0.000095
        
        assert "x-litellm-response-cost-original" in headers
        assert float(headers["x-litellm-response-cost-original"]) == 0.0001
        
        assert "x-litellm-response-cost-discount-amount" in headers
        assert float(headers["x-litellm-response-cost-discount-amount"]) == 0.000005

    def test_get_custom_headers_without_discount_info(self):
        """
        Test that when no discount is applied, discount headers are not included.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0
        
        # Create logging object without discount
        logging_obj = LiteLLMLoggingObj(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        
        # Set cost breakdown without discount information
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.0001,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
        )
        
        # Call get_custom_headers
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            response_cost=0.0001,
            litellm_logging_obj=logging_obj,
        )
        
        # Verify discount headers are NOT present
        assert "x-litellm-response-cost" in headers
        assert float(headers["x-litellm-response-cost"]) == 0.0001
        
        # Discount headers should not be in the final dict
        assert "x-litellm-response-cost-original" not in headers
        assert "x-litellm-response-cost-discount-amount" not in headers

    def test_get_custom_headers_with_margin_info(self):
        """
        Test that margin headers are included when margin is applied.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0
        
        # Create logging object with margin
        logging_obj = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-margin",
            function_id="test-function",
        )
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.00011,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            margin_percent=0.10,
            margin_total_amount=0.00001,
        )
        
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.00011,
            litellm_logging_obj=logging_obj,
        )
        
        # Verify margin headers are present
        assert "x-litellm-response-cost" in headers
        assert float(headers["x-litellm-response-cost"]) == 0.00011
        
        assert "x-litellm-response-cost-margin-amount" in headers
        assert float(headers["x-litellm-response-cost-margin-amount"]) == 0.00001
        
        assert "x-litellm-response-cost-margin-percent" in headers
        assert float(headers["x-litellm-response-cost-margin-percent"]) == 0.10

    def test_get_custom_headers_without_margin_info(self):
        """
        Test that when no margin is applied, margin headers are not included.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Create mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0
        
        # Create logging object without margin
        logging_obj = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-no-margin",
            function_id="test-function",
        )
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.0001,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
        )
        
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.0001,
            litellm_logging_obj=logging_obj,
        )
        
        # Verify margin headers are not present
        assert "x-litellm-response-cost-margin-amount" not in headers
        assert "x-litellm-response-cost-margin-percent" not in headers

    def test_get_cost_breakdown_from_logging_obj_helper(self):
        """
        Test the helper function that extracts cost breakdown information.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        # Test with discount info
        logging_obj = LiteLLMLoggingObj(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        logging_obj.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.000095,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            discount_percent=0.05,
            discount_amount=0.000005,
        )
        
        original_cost, discount_amount, margin_total_amount, margin_percent = _get_cost_breakdown_from_logging_obj(logging_obj)
        assert original_cost == 0.0001
        assert discount_amount == 0.000005
        assert margin_total_amount is None
        assert margin_percent is None
        
        # Test with margin info
        logging_obj_with_margin = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-margin",
            function_id="test-function-id-margin",
        )
        logging_obj_with_margin.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.00011,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            margin_percent=0.10,
            margin_total_amount=0.00001,
        )
        
        original_cost, discount_amount, margin_total_amount, margin_percent = _get_cost_breakdown_from_logging_obj(logging_obj_with_margin)
        assert original_cost == 0.0001
        assert discount_amount is None
        assert margin_total_amount == 0.00001
        assert margin_percent == 0.10
        
        # Test with no discount or margin info
        logging_obj_no_discount = LiteLLMLoggingObj(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-2",
            function_id="test-function-id-2",
        )
        logging_obj_no_discount.set_cost_breakdown(
            input_cost=0.00005,
            output_cost=0.00005,
            total_cost=0.0001,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
        )
        
        original_cost, discount_amount, margin_total_amount, margin_percent = _get_cost_breakdown_from_logging_obj(logging_obj_no_discount)
        assert original_cost is None
        assert discount_amount is None
        assert margin_total_amount is None
        assert margin_percent is None
        
        # Test with None logging object
        original_cost, discount_amount, margin_total_amount, margin_percent = _get_cost_breakdown_from_logging_obj(None)
        assert original_cost is None
        assert discount_amount is None
        assert margin_total_amount is None
        assert margin_percent is None

    def test_get_custom_headers_key_spend_includes_response_cost(self):
        """
        Test that x-litellm-key-spend header includes the current request's response_cost.
        
        This ensures that the spend header reflects the updated spend including the current
        request, even though spend tracking updates happen asynchronously after the response.
        """
        # Create mock user API key dict with initial spend
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.001  # Initial spend: $0.001

        # Test case 1: response_cost is provided as float
        response_cost_1 = 0.0005  # Current request cost: $0.0005
        headers_1 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-1",
            response_cost=response_cost_1,
        )
        
        assert "x-litellm-key-spend" in headers_1
        expected_spend_1 = 0.001 + 0.0005  # Initial spend + current request cost
        assert float(headers_1["x-litellm-key-spend"]) == pytest.approx(expected_spend_1, abs=1e-10)
        assert float(headers_1["x-litellm-response-cost"]) == response_cost_1

        # Test case 2: response_cost is provided as string
        response_cost_2 = "0.0003"  # Current request cost as string
        headers_2 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-2",
            response_cost=response_cost_2,
        )
        
        assert "x-litellm-key-spend" in headers_2
        expected_spend_2 = 0.001 + 0.0003  # Initial spend + current request cost
        assert float(headers_2["x-litellm-key-spend"]) == pytest.approx(expected_spend_2, abs=1e-10)

        # Test case 3: response_cost is None (should use original spend)
        headers_3 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-3",
            response_cost=None,
        )
        
        assert "x-litellm-key-spend" in headers_3
        assert float(headers_3["x-litellm-key-spend"]) == 0.001  # Should use original spend

        # Test case 4: response_cost is 0 (should not change spend)
        headers_4 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-4",
            response_cost=0.0,
        )
        
        assert "x-litellm-key-spend" in headers_4
        assert float(headers_4["x-litellm-key-spend"]) == 0.001  # Should remain unchanged for 0 cost

        # Test case 5: user_api_key_dict.spend is None (should default to 0.0)
        mock_user_api_key_dict.spend = None
        headers_5 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-5",
            response_cost=0.0002,
        )
        
        assert "x-litellm-key-spend" in headers_5
        assert float(headers_5["x-litellm-key-spend"]) == 0.0002  # 0.0 + 0.0002

        # Test case 6: response_cost is negative (should not be added, use original spend)
        mock_user_api_key_dict.spend = 0.001
        headers_6 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-6",
            response_cost=-0.0001,  # Negative cost (should not be added)
        )
        
        assert "x-litellm-key-spend" in headers_6
        assert float(headers_6["x-litellm-key-spend"]) == 0.001  # Should use original spend

        # Test case 7: response_cost is invalid string (should fallback to original spend)
        headers_7 = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-7",
            response_cost="invalid",  # Invalid string
        )
        
        assert "x-litellm-key-spend" in headers_7
        assert float(headers_7["x-litellm-key-spend"]) == 0.001  # Should use original spend on error


@pytest.mark.asyncio
class TestCommonRequestProcessingHelpers:
    async def consume_stream(self, streaming_response: StreamingResponse) -> list:
        content = []
        async for chunk_bytes in streaming_response.body_iterator:
            content.append(chunk_bytes)
        return content

    @pytest.mark.parametrize(
        "event_line, expected_code",
        [
            (
                'data: {"error": {"code": 400, "message": "bad request"}}',
                400,
            ),  # Valid integer code
            (
                'data: {"error": {"code": "401", "message": "unauthorized"}}',
                401,
            ),  # Valid string-integer code
            (
                'data: {"error": {"code": "invalid_code", "message": "error"}}',
                None,
            ),  # Invalid string code
            (
                'data: {"error": {"code": 99, "message": "too low"}}',
                None,
            ),  # Integer code too low
            (
                'data: {"error": {"code": 600, "message": "too high"}}',
                None,
            ),  # Integer code too high
            (
                'data: {"id": "123", "content": "hello"}',
                None,
            ),  # Non-error SSE event
            ("data: [DONE]", None),  # SSE [DONE] event
            ("data: ", None),  # SSE empty data event
            (
                'data: {"error": {"code": 400',
                None,
            ),  # Malformed JSON
            ("id: 123", None),  # Non-SSE event line
            (
                'data: {"error": {"message": "some error"}}',
                None,
            ),  # Error event without 'code' field
            (
                'data: {"error": {"code": null, "message": "code is null"}}',
                None,
            ),  # Error with null code
        ],
    )
    async def test_parse_event_data_for_error(self, event_line, expected_code):
        assert await _parse_event_data_for_error(event_line) == expected_code

    async def test_create_streaming_response_first_chunk_is_error(self):
        """
        Test that when the first chunk is an error, a JSON error response is returned
        instead of an SSE streaming response
        """
        async def mock_generator():
            yield 'data: {"error": {"code": 403, "message": "forbidden"}}\n\n'
            yield 'data: {"content": "more data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(
            mock_generator(), "text/event-stream", {}
        )
        # Should return JSONResponse instead of StreamingResponse
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_403_FORBIDDEN
        # Verify the response is in standard JSON error format
        import json
        body = json.loads(response.body.decode())
        assert "error" in body
        assert body["error"]["code"] == 403
        assert body["error"]["message"] == "forbidden"

    async def test_create_streaming_response_first_chunk_not_error(self):
        async def mock_generator():
            yield 'data: {"content": "first part"}\n\n'
            yield 'data: {"content": "second part"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(
            mock_generator(), "text/event-stream", {}
        )
        assert response.status_code == status.HTTP_200_OK
        content = await self.consume_stream(response)
        assert content == [
            'data: {"content": "first part"}\n\n',
            'data: {"content": "second part"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_empty_generator(self):
        async def mock_generator():
            if False:  # Never yields
                yield
            # Implicitly raises StopAsyncIteration

        response = await create_response(
            mock_generator(), "text/event-stream", {}
        )
        assert response.status_code == status.HTTP_200_OK
        content = await self.consume_stream(response)
        assert content == []

    async def test_create_streaming_response_generator_raises_stop_async_iteration_immediately(
        self,
    ):
        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = StopAsyncIteration

        response = await create_response(mock_gen, "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK
        content = await self.consume_stream(response)
        assert content == []

    async def test_create_streaming_response_generator_raises_unexpected_exception(
        self,
    ):
        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = ValueError("Test error from generator")

        response = await create_response(mock_gen, "text/event-stream", {})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        content = await self.consume_stream(response)
        expected_error_data = {
            "error": {
                "message": "Error processing stream start",
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        }
        assert len(content) == 2
        # Use json.dumps to match the formatting in create_streaming_response's exception handler
        import json

        assert content[0] == f"data: {json.dumps(expected_error_data)}\n\n"
        assert content[1] == "data: [DONE]\n\n"

    async def test_create_streaming_response_first_chunk_error_string_code(self):
        """
        Test that when the first chunk contains a string error code, a JSON error response is returned
        """
        async def mock_generator():
            yield 'data: {"error": {"code": "429", "message": "too many requests"}}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(
            mock_generator(), "text/event-stream", {}
        )
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        # Verify the response is in standard JSON error format
        import json
        body = json.loads(response.body.decode())
        assert "error" in body
        assert body["error"]["code"] == "429"
        assert body["error"]["message"] == "too many requests"

    async def test_create_streaming_response_custom_headers(self):
        async def mock_generator():
            yield 'data: {"content": "data"}\n\n'
            yield "data: [DONE]\n\n"

        custom_headers = {"X-Custom-Header": "TestValue"}
        response = await create_response(
            mock_generator(), "text/event-stream", custom_headers
        )
        assert response.headers["x-custom-header"] == "TestValue"

    async def test_create_streaming_response_non_default_status_code(self):
        async def mock_generator():
            yield 'data: {"content": "data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(
            mock_generator(),
            "text/event-stream",
            {},
            default_status_code=status.HTTP_201_CREATED,
        )
        assert response.status_code == status.HTTP_201_CREATED
        content = await self.consume_stream(response)
        assert content == [
            'data: {"content": "data"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_first_chunk_is_done(self):
        async def mock_generator():
            yield "data: [DONE]\n\n"

        response = await create_response(
            mock_generator(), "text/event-stream", {}
        )
        assert response.status_code == status.HTTP_200_OK  # Default status
        content = await self.consume_stream(response)
        assert content == ["data: [DONE]\n\n"]

    async def test_create_streaming_response_first_chunk_is_empty_data(self):
        async def mock_generator():
            yield "data: \n\n"
            yield 'data: {"content": "actual data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(
            mock_generator(), "text/event-stream", {}
        )
        assert response.status_code == status.HTTP_200_OK  # Default status
        content = await self.consume_stream(response)
        assert content == [
            "data: \n\n",
            'data: {"content": "actual data"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_all_chunks_have_dd_trace(self):
        """Test that all stream chunks are wrapped with dd trace at the streaming generator level"""
        import json
        from unittest.mock import patch

        # Create a mock tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.trace.return_value.__enter__.return_value = mock_span
        mock_tracer.trace.return_value.__exit__.return_value = None

        # Mock generator with multiple chunks
        async def mock_generator():
            yield 'data: {"content": "chunk 1"}\n\n'
            yield 'data: {"content": "chunk 2"}\n\n'
            yield 'data: {"content": "chunk 3"}\n\n'
            yield "data: [DONE]\n\n"

        # Patch the tracer in the common_request_processing module
        with patch("litellm.proxy.common_request_processing.tracer", mock_tracer):
            response = await create_response(
                mock_generator(), "text/event-stream", {}
            )

            assert response.status_code == 200

            # Consume the stream to trigger the tracer calls
            content = await self.consume_stream(response)

            # Verify all chunks are present
            assert len(content) == 4
            assert content[0] == 'data: {"content": "chunk 1"}\n\n'
            assert content[1] == 'data: {"content": "chunk 2"}\n\n'
            assert content[2] == 'data: {"content": "chunk 3"}\n\n'
            assert content[3] == "data: [DONE]\n\n"

            # Verify that tracer.trace was called for each chunk (4 chunks total)
            assert mock_tracer.trace.call_count == 4

            # Verify that each call was made with the correct operation name
            expected_calls = [
                (("streaming.chunk.yield",), {}),
                (("streaming.chunk.yield",), {}),
                (("streaming.chunk.yield",), {}),
                (("streaming.chunk.yield",), {}),
            ]

            actual_calls = mock_tracer.trace.call_args_list
            assert len(actual_calls) == 4

            for i, call in enumerate(actual_calls):
                args, kwargs = call
                assert (
                    args[0] == "streaming.chunk.yield"
                ), f"Call {i} should have operation name 'streaming.chunk.yield', got {args[0]}"

    async def test_create_streaming_response_dd_trace_with_error_chunk(self):
        """
        Test that when the first chunk contains an error, JSONResponse is returned
        and tracing is not triggered (since it's not a streaming response)
        """
        from unittest.mock import patch

        # Create a mock tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.trace.return_value.__enter__.return_value = mock_span
        mock_tracer.trace.return_value.__exit__.return_value = None

        # Mock generator with error in first chunk
        async def mock_generator():
            yield 'data: {"error": {"code": 400, "message": "bad request"}}\n\n'
            yield 'data: {"content": "chunk after error"}\n\n'
            yield "data: [DONE]\n\n"

        # Patch the tracer in the common_request_processing module
        with patch("litellm.proxy.common_request_processing.tracer", mock_tracer):
            response = await create_response(
                mock_generator(), "text/event-stream", {}
            )

            # Should return JSONResponse instead of StreamingResponse
            assert isinstance(response, JSONResponse)
            assert response.status_code == 400

            # Verify the response is in standard JSON error format
            import json
            body = json.loads(response.body.decode())
            assert "error" in body
            assert body["error"]["code"] == 400
            assert body["error"]["message"] == "bad request"

            # Since JSONResponse is returned instead of StreamingResponse, streaming tracing should not be triggered
            # tracer.trace should not be called
            assert mock_tracer.trace.call_count == 0


class TestExtractErrorFromSSEChunk:
    """Tests for _extract_error_from_sse_chunk function"""

    def test_extract_error_from_sse_chunk_with_valid_error(self):
        """Test extracting error information from a standard SSE chunk"""
        chunk = 'data: {"error": {"code": 403, "message": "forbidden", "type": "auth_error", "param": "api_key"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["code"] == 403
        assert error["message"] == "forbidden"
        assert error["type"] == "auth_error"
        assert error["param"] == "api_key"

    def test_extract_error_from_sse_chunk_with_string_code(self):
        """Test error code as string type"""
        chunk = 'data: {"error": {"code": "429", "message": "too many requests"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["code"] == "429"
        assert error["message"] == "too many requests"

    def test_extract_error_from_sse_chunk_with_bytes(self):
        """Test input as bytes type"""
        chunk = b'data: {"error": {"code": 500, "message": "internal error"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["code"] == 500
        assert error["message"] == "internal error"

    def test_extract_error_from_sse_chunk_with_done(self):
        """Test [DONE] marker should return default error"""
        chunk = "data: [DONE]\n\n"
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"
        assert error["param"] is None

    def test_extract_error_from_sse_chunk_without_error_field(self):
        """Test missing error field should return default error"""
        chunk = 'data: {"content": "some content"}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_with_invalid_json(self):
        """Test invalid JSON should return default error"""
        chunk = 'data: {invalid json}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_without_data_prefix(self):
        """Test missing 'data:' prefix should return default error"""
        chunk = '{"error": {"code": 400, "message": "bad request"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_with_empty_string(self):
        """Test empty string should return default error"""
        chunk = ""
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "Unknown error"
        assert error["type"] == "internal_server_error"
        assert error["code"] == "500"

    def test_extract_error_from_sse_chunk_with_minimal_error(self):
        """Test minimal error object"""
        chunk = 'data: {"error": {"message": "error occurred"}}\n\n'
        error = _extract_error_from_sse_chunk(chunk)

        assert error["message"] == "error occurred"
        # Other fields should be obtained from the original error object (if exists)


class TestOverrideOpenAIResponseModel:
    """Tests for _override_openai_response_model function"""

    def test_override_model_preserves_fallback_model_when_fallback_occurred_object(self):
        """
        Test that when a fallback occurred (x-litellm-attempted-fallbacks > 0),
        the actual model used (fallback model) is preserved instead of being
        overridden with the requested model.
        
        This is the regression test to ensure the model being called is properly
        displayed when a fallback happens.
        """
        requested_model = "gpt-4"
        fallback_model = "gpt-3.5-turbo"
        
        # Create a mock object response with fallback model
        # _hidden_params is an attribute (not a dict key) accessed via getattr
        response_obj = MagicMock()
        response_obj.model = fallback_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 1
            }
        }
        
        # Call the function - should preserve fallback model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        
        # Verify the model was NOT overridden - should still be the fallback model
        assert response_obj.model == fallback_model
        assert response_obj.model != requested_model

    def test_override_model_preserves_fallback_model_multiple_fallbacks(self):
        """
        Test that when multiple fallbacks occurred, the actual model used
        (fallback model) is preserved.
        """
        requested_model = "gpt-4"
        fallback_model = "claude-haiku-4-5-20251001"
        
        # Create a mock object response with fallback model
        response_obj = MagicMock()
        response_obj.model = fallback_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 2  # Multiple fallbacks
            }
        }
        
        # Call the function - should preserve fallback model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        
        # Verify the model was NOT overridden - should still be the fallback model
        assert response_obj.model == fallback_model
        assert response_obj.model != requested_model

    def test_override_model_overrides_when_no_fallback_dict(self):
        """
        Test that when no fallback occurred, the model is overridden
        to match the requested model (dict response).
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"
        
        # Create a dict response without fallback
        # For dict responses, _hidden_params won't be found via getattr,
        # so the fallback check won't trigger and model will be overridden
        response_obj = {"model": downstream_model}
        
        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        
        # Verify the model WAS overridden to requested model
        assert response_obj["model"] == requested_model

    def test_override_model_overrides_when_no_fallback_object(self):
        """
        Test that when no fallback occurred (object response), the model is overridden
        to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"
        
        # Create a mock object response without fallback
        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "additional_headers": {}  # No attempted_fallbacks header
        }
        
        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        
        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_overrides_when_attempted_fallbacks_is_zero(self):
        """
        Test that when attempted_fallbacks is 0 (no fallback occurred),
        the model is overridden to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"
        
        # Create a mock object response
        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 0  # Zero means no fallback occurred
            }
        }
        
        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        
        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_overrides_when_attempted_fallbacks_is_none(self):
        """
        Test that when attempted_fallbacks is None (not set),
        the model is overridden to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"
        
        # Create a mock object response
        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": None
            }
        }
        
        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        
        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_no_hidden_params(self):
        """
        Test that when _hidden_params is not present, the model is overridden
        to match the requested model.
        """
        requested_model = "gpt-4"
        downstream_model = "gpt-3.5-turbo"
        
        # Create a mock object response without _hidden_params
        response_obj = MagicMock()
        response_obj.model = downstream_model
        # Don't set _hidden_params - getattr will return {}
        
        # Call the function - should override to requested model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        
        # Verify the model WAS overridden to requested model
        assert response_obj.model == requested_model

    def test_override_model_no_requested_model(self):
        """
        Test that when requested_model is None or empty, the function returns early
        without modifying the response.
        """
        fallback_model = "gpt-3.5-turbo"
        
        # Create a mock object response
        response_obj = MagicMock()
        response_obj.model = fallback_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 1
            }
        }
        
        # Call the function with None requested_model
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=None,
            log_context="test_context",
        )
        
        # Verify the model was not changed
        assert response_obj.model == fallback_model
        
        # Call with empty string
        _override_openai_response_model(
            response_obj=response_obj,
            requested_model="",
            log_context="test_context",
        )
        
        # Verify the model was not changed
        assert response_obj.model == fallback_model


