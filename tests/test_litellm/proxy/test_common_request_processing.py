import copy
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, status
from fastapi.responses import StreamingResponse

import litellm
from litellm._uuid import uuid
from litellm.integrations.opentelemetry import UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    ProxyConfig,
    _get_cost_breakdown_from_logging_obj,
    _parse_event_data_for_error,
    create_streaming_response,
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
        
        original_cost, discount_amount = _get_cost_breakdown_from_logging_obj(logging_obj)
        assert original_cost == 0.0001
        assert discount_amount == 0.000005
        
        # Test with no discount info
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
        
        original_cost, discount_amount = _get_cost_breakdown_from_logging_obj(logging_obj_no_discount)
        assert original_cost is None
        assert discount_amount is None
        
        # Test with None logging object
        original_cost, discount_amount = _get_cost_breakdown_from_logging_obj(None)
        assert original_cost is None
        assert discount_amount is None


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
        async def mock_generator():
            yield 'data: {"error": {"code": 403, "message": "forbidden"}}\n\n'
            yield 'data: {"content": "more data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_streaming_response(
            mock_generator(), "text/event-stream", {}
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN
        content = await self.consume_stream(response)
        assert content == [
            'data: {"error": {"code": 403, "message": "forbidden"}}\n\n',
            'data: {"content": "more data"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_first_chunk_not_error(self):
        async def mock_generator():
            yield 'data: {"content": "first part"}\n\n'
            yield 'data: {"content": "second part"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_streaming_response(
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

        response = await create_streaming_response(
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

        response = await create_streaming_response(mock_gen, "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK
        content = await self.consume_stream(response)
        assert content == []

    async def test_create_streaming_response_generator_raises_unexpected_exception(
        self,
    ):
        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = ValueError("Test error from generator")

        response = await create_streaming_response(mock_gen, "text/event-stream", {})
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
        async def mock_generator():
            yield 'data: {"error": {"code": "429", "message": "too many requests"}}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_streaming_response(
            mock_generator(), "text/event-stream", {}
        )
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        content = await self.consume_stream(response)
        assert content == [
            'data: {"error": {"code": "429", "message": "too many requests"}}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_custom_headers(self):
        async def mock_generator():
            yield 'data: {"content": "data"}\n\n'
            yield "data: [DONE]\n\n"

        custom_headers = {"X-Custom-Header": "TestValue"}
        response = await create_streaming_response(
            mock_generator(), "text/event-stream", custom_headers
        )
        assert response.headers["x-custom-header"] == "TestValue"

    async def test_create_streaming_response_non_default_status_code(self):
        async def mock_generator():
            yield 'data: {"content": "data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_streaming_response(
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

        response = await create_streaming_response(
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

        response = await create_streaming_response(
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
            response = await create_streaming_response(
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
        """Test that dd trace is applied even when the first chunk contains an error"""
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
            response = await create_streaming_response(
                mock_generator(), "text/event-stream", {}
            )

            # Even with error, status should be set to error code but tracing should still work
            assert response.status_code == 400

            # Consume the stream to trigger the tracer calls
            content = await self.consume_stream(response)

            # Verify all chunks are present
            assert len(content) == 3

            # Verify that tracer.trace was called for each chunk
            assert mock_tracer.trace.call_count == 3

            # Verify that each call was made with the correct operation name
            actual_calls = mock_tracer.trace.call_args_list
            assert len(actual_calls) == 3

            for i, call in enumerate(actual_calls):
                args, kwargs = call
                assert (
                    args[0] == "streaming.chunk.yield"
                ), f"Call {i} should have operation name 'streaming.chunk.yield', got {args[0]}"
