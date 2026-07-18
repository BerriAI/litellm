import asyncio
import copy
import datetime
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

import litellm
from litellm._uuid import uuid
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry import UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    ProxyConfig,
    _await_llm_call_cancelling_on_disconnect,
    _bill_partial_streamed_spend_on_disconnect,
    _buffer_first_chunk_honoring_disconnect,
    _cancel_llm_call_on_client_disconnect,
    _ClientDisconnectedBeforeFirstChunk,
    _extract_error_from_sse_chunk,
    _get_cost_breakdown_from_logging_obj,
    _has_attribute_error_in_chain,
    _is_azure_model_router_request,
    _override_openai_response_model,
    _parse_event_data_for_error,
    _UpstreamClosingStreamingResponse,
    create_response,
)
from litellm.proxy.dd_span_tagger import DDSpanTagger
from litellm.proxy.utils import ProxyLogging


class TestProxyBaseLLMRequestProcessing:
    @pytest.mark.asyncio
    async def test_base_passthrough_process_llm_request_preserves_litellm_headers_for_non_streaming_response(
        self, monkeypatch
    ):
        processing_obj = ProxyBaseLLMRequestProcessing(data={})

        async def fake_base_process_llm_request(**kwargs):
            passthrough_response = kwargs["fastapi_response"]
            passthrough_response.headers["x-litellm-call-id"] = "test-call-id"
            passthrough_response.headers["x-litellm-version"] = "test-version"
            return httpx.Response(
                status_code=200,
                content=b'{"ok":true}',
                headers={
                    "content-type": "application/json",
                    "x-amzn-requestid": "bedrock-request-id",
                },
            )

        monkeypatch.setattr(
            processing_obj,
            "base_process_llm_request",
            fake_base_process_llm_request,
        )

        result = await processing_obj.base_passthrough_process_llm_request(
            request=MagicMock(spec=Request),
            fastapi_response=Response(),
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            proxy_logging_obj=MagicMock(spec=ProxyLogging),
            general_settings={},
            proxy_config=MagicMock(spec=ProxyConfig),
            select_data_generator=MagicMock(),
            model="bedrock-test-model",
        )

        assert result.status_code == 200
        assert result.body == b'{"ok":true}'
        assert result.headers["x-amzn-requestid"] == "bedrock-request-id"
        assert result.headers["x-litellm-call-id"] == "test-call-id"
        assert result.headers["x-litellm-version"] == "test-version"

    @pytest.mark.asyncio
    async def test_base_passthrough_process_llm_request_returns_fastapi_response_from_guardrails(self, monkeypatch):
        """Post-call guardrails return a FastAPI Response; must not call httpx aread()."""
        import json

        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        guardrailed_body = {
            "output": {"message": {"content": [{"text": "masked"}]}},
            "stopReason": "end_turn",
        }

        async def fake_base_process_llm_request(**kwargs):
            return Response(
                content=json.dumps(guardrailed_body).encode(),
                status_code=200,
                media_type="application/json",
            )

        monkeypatch.setattr(
            processing_obj,
            "base_process_llm_request",
            fake_base_process_llm_request,
        )

        result = await processing_obj.base_passthrough_process_llm_request(
            request=MagicMock(spec=Request),
            fastapi_response=Response(),
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            proxy_logging_obj=MagicMock(spec=ProxyLogging),
            general_settings={},
            proxy_config=MagicMock(spec=ProxyConfig),
            select_data_generator=MagicMock(),
            model="bedrock-test-model",
        )

        assert isinstance(result, Response)
        assert json.loads(result.body) == guardrailed_body

    @pytest.mark.asyncio
    async def test_handle_non_streaming_allm_passthrough_route_forwards_upstream_headers(
        self, monkeypatch
    ):
        """The guardrail JSON path must forward upstream response headers (e.g.
        x-amzn-requestid) alongside the x-litellm-* headers, matching the
        non-guardrail passthrough path, while dropping length headers that no
        longer match the rewritten body."""
        processing_obj = ProxyBaseLLMRequestProcessing(
            data={"custom_llm_provider": "bedrock"}
        )
        monkeypatch.setattr(
            processing_obj,
            "_has_post_call_guardrails_for_passthrough",
            lambda: True,
        )

        upstream = httpx.Response(
            status_code=200,
            content=b'{"output": {"message": {"content": [{"text": "hi"}]}}}',
            headers={
                "content-type": "application/json",
                "x-amzn-requestid": "bedrock-request-id",
                "content-length": "999",
            },
        )

        proxy_logging_obj = MagicMock(spec=ProxyLogging)

        async def fake_post_call_success_hook(**kwargs):
            return kwargs["response"]

        proxy_logging_obj.post_call_success_hook = fake_post_call_success_hook
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})

        result = await processing_obj._handle_non_streaming_allm_passthrough_route(
            response=upstream,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            custom_headers={"x-litellm-call-id": "test-call-id"},
            request_headers={},
        )

        assert isinstance(result, Response)
        assert result.status_code == 200
        assert result.headers["x-amzn-requestid"] == "bedrock-request-id"
        assert result.headers["x-litellm-call-id"] == "test-call-id"
        assert result.headers["content-length"] == str(len(result.body))

    @pytest.mark.asyncio
    async def test_handle_event_stream_allm_passthrough_route_forwards_upstream_headers(
        self, monkeypatch
    ):
        """The guardrail event-stream branch must also forward upstream response
        headers alongside the x-litellm-* headers."""
        processing_obj = ProxyBaseLLMRequestProcessing(
            data={"custom_llm_provider": "bedrock"}
        )
        monkeypatch.setattr(
            processing_obj,
            "_has_post_call_guardrails_for_passthrough",
            lambda: True,
        )

        async def fake_event_stream(**kwargs):
            return b"rewritten-frames"

        monkeypatch.setattr(
            processing_obj,
            "_handle_event_stream_allm_passthrough_route",
            fake_event_stream,
        )

        upstream = httpx.Response(
            status_code=200,
            content=b"original-frames",
            headers={
                "content-type": "application/vnd.amazon.eventstream",
                "x-amzn-requestid": "bedrock-request-id",
            },
        )

        proxy_logging_obj = MagicMock(spec=ProxyLogging)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})

        result = await processing_obj._handle_non_streaming_allm_passthrough_route(
            response=upstream,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            custom_headers={"x-litellm-call-id": "test-call-id"},
            request_headers={},
        )

        assert isinstance(result, Response)
        assert result.body == b"rewritten-frames"
        assert result.headers["x-amzn-requestid"] == "bedrock-request-id"
        assert result.headers["x-litellm-call-id"] == "test-call-id"

    @pytest.mark.asyncio
    async def test_handle_non_streaming_allm_passthrough_route_applies_response_headers_hook(
        self, monkeypatch
    ):
        """Guardrailed non-streaming passthrough responses must include headers
        injected by post_call_response_headers_hook, matching the headers a
        non-guardrailed passthrough response would carry."""
        processing_obj = ProxyBaseLLMRequestProcessing(
            data={"custom_llm_provider": "bedrock"}
        )
        monkeypatch.setattr(
            processing_obj,
            "_has_post_call_guardrails_for_passthrough",
            lambda: True,
        )

        upstream = httpx.Response(
            status_code=200,
            content=b'{"output": {"message": {"content": [{"text": "hi"}]}}}',
            headers={"content-type": "application/json"},
        )

        proxy_logging_obj = MagicMock(spec=ProxyLogging)

        async def fake_post_call_success_hook(**kwargs):
            return kwargs["response"]

        proxy_logging_obj.post_call_success_hook = fake_post_call_success_hook
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(
            return_value={"x-litellm-custom": "from-hook"}
        )

        result = await processing_obj._handle_non_streaming_allm_passthrough_route(
            response=upstream,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            custom_headers={"x-litellm-call-id": "test-call-id"},
            request_headers={"authorization": "Bearer sk-test"},
        )

        assert isinstance(result, Response)
        assert result.headers["x-litellm-custom"] == "from-hook"
        assert result.headers["x-litellm-call-id"] == "test-call-id"
        proxy_logging_obj.post_call_response_headers_hook.assert_awaited_once()
        _, kwargs = proxy_logging_obj.post_call_response_headers_hook.call_args
        assert kwargs["request_headers"] == {"authorization": "Bearer sk-test"}

    @pytest.mark.asyncio
    async def test_common_processing_pre_call_logic_pre_call_hook_receives_litellm_call_id(self, monkeypatch):
        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return {}

        async def mock_common_processing_pre_call_logic(user_api_key_dict, data, call_type):
            data_copy = copy.deepcopy(data)
            return data_copy

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=mock_common_processing_pre_call_logic)
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

    def test_add_dd_apm_tags_for_litellm_call_id_uses_dd_tracing_helper(self, monkeypatch):
        mock_set_active_span_tag = MagicMock(return_value=True)
        import litellm.proxy.dd_span_tagger

        monkeypatch.setattr(
            litellm.proxy.dd_span_tagger,
            "set_active_span_tag",
            mock_set_active_span_tag,
        )

        DDSpanTagger.tag_call_id("test-call-id")

        mock_set_active_span_tag.assert_called_once_with("litellm.call_id", "test-call-id")

    @pytest.mark.asyncio
    async def test_should_apply_hierarchical_router_settings_as_override(self, monkeypatch):
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

        async def mock_common_processing_pre_call_logic(user_api_key_dict, data, call_type):
            data_copy = copy.deepcopy(data)
            return data_copy

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=mock_common_processing_pre_call_logic)
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
        mock_proxy_config._get_hierarchical_router_settings = AsyncMock(return_value=mock_router_settings)

        mock_llm_router = MagicMock()

        mock_prisma_client = MagicMock()
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma_client,
        )

        route_type = "acompletion"

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
    async def test_build_litellm_proxy_success_headers_from_llm_response(self):
        """
        Google native :generateContent uses this helper instead of base_process_llm_request;
        ensure x-litellm-* headers and callback hooks merge like the main proxy path.
        """
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        class _FakeGenaiResponse:
            _hidden_params = {
                "model_id": "deployment-model-id",
                "cache_key": "ck-test",
                "api_base": "https://generativelanguage.googleapis.com/v1beta",
                "response_cost": 0.001,
                "additional_headers": {"llm_provider-ratelimit-requests": "1000"},
            }

        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "call-id-test"

        mock_user = MagicMock()
        mock_user.tpm_limit = None
        mock_user.rpm_limit = None
        mock_user.max_budget = None
        mock_user.spend = 0.0
        mock_user.allowed_model_region = None

        proxy_logging_obj = MagicMock(spec=ProxyLogging)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(
            return_value={"x-ratelimit-remaining-requests": "999"}
        )

        headers = await ProxyBaseLLMRequestProcessing.build_litellm_proxy_success_headers_from_llm_response(
            response=_FakeGenaiResponse(),
            request_data={"model": "gemini/gemini-1.5-flash"},
            request=mock_request,
            user_api_key_dict=mock_user,
            logging_obj=logging_obj,
            version="9.9.9",
            proxy_logging_obj=proxy_logging_obj,
        )

        assert headers["x-litellm-call-id"] == "call-id-test"
        assert headers["x-litellm-model-id"] == "deployment-model-id"
        assert headers["x-litellm-version"] == "9.9.9"
        assert headers["llm_provider-ratelimit-requests"] == "1000"
        assert headers["x-ratelimit-remaining-requests"] == "999"
        proxy_logging_obj.post_call_response_headers_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_build_litellm_proxy_success_headers_streaming_style_iterator(self):
        """AsyncGoogleGenAIGenerateContentStreamingIterator sets _hidden_params at init; headers must propagate."""

        class _FakeStreamLike:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            _hidden_params = {
                "model_id": "stream-model-id",
                "api_base": "https://generativelanguage.googleapis.com/v1beta",
                "cache_key": "",
                "response_cost": "",
                "additional_headers": {"llm_provider-x": "y"},
            }

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "cid-stream"
        mock_user = MagicMock()
        mock_user.tpm_limit = None
        mock_user.rpm_limit = None
        mock_user.max_budget = None
        mock_user.spend = 0.0
        mock_user.allowed_model_region = None
        proxy_logging_obj = MagicMock(spec=ProxyLogging)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})

        headers = await ProxyBaseLLMRequestProcessing.build_litellm_proxy_success_headers_from_llm_response(
            response=_FakeStreamLike(),
            request_data={"model": "gemini/gemini-2.0-flash"},
            request=mock_request,
            user_api_key_dict=mock_user,
            logging_obj=logging_obj,
            version="1.0.0",
            proxy_logging_obj=proxy_logging_obj,
        )

        assert headers["x-litellm-model-id"] == "stream-model-id"
        assert headers["x-litellm-model-api-base"] == ("https://generativelanguage.googleapis.com/v1beta")
        assert headers["llm_provider-x"] == "y"

    @pytest.mark.asyncio
    async def test_build_litellm_proxy_success_headers_no_hidden_params_metadata_fallback(
        self,
    ):
        """When response has no _hidden_params, model_id can still come from litellm_metadata."""

        class _BareResponse:
            pass

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "cid-meta"
        mock_user = MagicMock()
        mock_user.tpm_limit = None
        mock_user.rpm_limit = None
        mock_user.max_budget = None
        mock_user.spend = 0.0
        mock_user.allowed_model_region = None
        proxy_logging_obj = MagicMock(spec=ProxyLogging)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})

        headers = await ProxyBaseLLMRequestProcessing.build_litellm_proxy_success_headers_from_llm_response(
            response=_BareResponse(),
            request_data={
                "model": "gemini/gemini-1.5-flash",
                "litellm_metadata": {"model_info": {"id": "meta-model-id"}},
            },
            request=mock_request,
            user_api_key_dict=mock_user,
            logging_obj=logging_obj,
            version="1.0.0",
            proxy_logging_obj=proxy_logging_obj,
        )

        assert headers["x-litellm-model-id"] == "meta-model-id"

    @pytest.mark.asyncio
    async def test_add_litellm_data_to_request_with_stream_timeout_header(self):
        """
        Test that x-litellm-stream-timeout header gets processed and added to request data
        when calling add_litellm_data_to_request.
        """
        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

        # Create test data with a basic completion request
        test_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
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

        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(logging_obj)
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

        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(logging_obj_with_margin)
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

        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(logging_obj_no_discount)
        assert original_cost is None
        assert discount_amount is None
        assert margin_total_amount is None
        assert margin_percent is None

        # Test with None logging object
        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(None)
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
    async def test_queue_time_seconds_is_set_in_metadata(self, monkeypatch):
        """
        Test that queue_time_seconds is correctly calculated and stored in metadata
        after add_litellm_data_to_request populates arrival_time.

        This verifies the fix for the bug where queue_time_seconds was always None
        because arrival_time was read BEFORE add_litellm_data_to_request set it.
        """
        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/chat/completions"

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            data = kwargs.get("data", args[0] if args else {})
            # Simulate what add_litellm_data_to_request does: set arrival_time
            import time

            data["proxy_server_request"] = {
                "url": "/v1/chat/completions",
                "method": "POST",
                "headers": {},
                "body": {},
                "arrival_time": time.time() - 0.5,  # Simulate request arrived 0.5s ago
            }
            data["metadata"] = data.get("metadata", {})
            return data

        async def mock_pre_call_hook(user_api_key_dict, data, call_type):
            return copy.deepcopy(data)

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )
        mock_general_settings = {}
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_proxy_config = MagicMock(spec=ProxyConfig)
        route_type = "acompletion"

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

        # Verify queue_time_seconds is set and non-negative
        metadata = returned_data.get("metadata", {})
        assert "queue_time_seconds" in metadata, "queue_time_seconds should be set in metadata"
        assert metadata["queue_time_seconds"] >= 0.5, (
            f"queue_time_seconds should be at least 0.5, got {metadata['queue_time_seconds']}"
        )


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

        response = await create_response(mock_generator(), "text/event-stream", {})
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

        response = await create_response(mock_generator(), "text/event-stream", {})
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

        response = await create_response(mock_generator(), "text/event-stream", {})
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
        # Streaming SSE error frame now mirrors ProxyException.to_dict() shape
        # so streaming and non-streaming surfaces emit byte-identical errors.
        expected_error_data = {
            "error": {
                "message": "Error processing stream start",
                "type": "None",
                "param": "None",
                "code": str(status.HTTP_500_INTERNAL_SERVER_ERROR),
            }
        }
        assert len(content) == 2
        import json

        assert content[0] == f"data: {json.dumps(expected_error_data)}\n\n"
        assert content[1] == "data: [DONE]\n\n"

    async def test_create_streaming_response_generator_raises_http_exception(
        self,
    ):
        """
        Test that when a generator raises HTTPException, the response preserves
        the original status code instead of hardcoding 500.
        """
        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = HTTPException(status_code=400, detail="Content blocked by guardrail")

        response = await create_response(mock_gen, "text/event-stream", {})
        assert response.status_code == 400
        content = await self.consume_stream(response)
        import json

        expected_error_data = {
            "error": {
                "message": "Content blocked by guardrail",
                "type": "None",
                "param": "None",
                "code": "400",
            }
        }
        assert len(content) == 2
        assert content[0] == f"data: {json.dumps(expected_error_data)}\n\n"
        assert content[1] == "data: [DONE]\n\n"

    async def test_create_streaming_response_http_exception_dict_detail_bedrock_shape(
        self,
    ):
        """
        Bedrock-style dict detail (with the post-L3 shape) must be preserved as
        structured `provider_specific_fields` in the SSE error frame, not stringified
        into a Python-repr blob inside `error.message`. Regression for case
        2026-04-10-internal-bedrock-guardrail-streaming-error.
        """
        import json

        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy",
                "bedrock_guardrail_response": "Sorry, the model cannot answer this question. Prompt is blocked",
                "guardrailIdentifier": "amgllac6xf3r",
                "guardrailVersion": "1",
                "assessments": [
                    {
                        "policy": "sensitiveInformationPolicy",
                        "matches": [
                            {
                                "category": "piiEntities",
                                "type": "NAME",
                                "action": "BLOCKED",
                                "match": "Jack",
                            }
                        ],
                    }
                ],
                "guardrail_name": "bedrock-pii-guard",
                "guardrail_mode": "post_call",
            },
        )

        response = await create_response(mock_gen, "text/event-stream", {})
        assert response.status_code == 400
        content = await self.consume_stream(response)
        assert len(content) == 2
        assert content[1] == "data: [DONE]\n\n"

        payload = json.loads(content[0][len("data: ") :].strip())
        assert payload["error"]["message"] == "Violated guardrail policy"
        assert payload["error"]["code"] == "400"
        psf = payload["error"]["provider_specific_fields"]
        assert psf["guardrail_name"] == "bedrock-pii-guard"
        assert psf["guardrail_mode"] == "post_call"
        assert psf["guardrailIdentifier"] == "amgllac6xf3r"
        assert psf["assessments"][0]["policy"] == "sensitiveInformationPolicy"
        assert psf["assessments"][0]["matches"][0]["type"] == "NAME"

    async def test_create_streaming_response_http_exception_dict_detail_nested_error_shape(
        self,
    ):
        """PANW Prisma AIRS-style nested `{"error": {"message": ...}}` detail must
        extract `error.message` as the human-readable summary while preserving the
        full payload."""
        import json

        mock_gen = AsyncMock()
        mock_gen.__anext__.side_effect = HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "MCP request blocked: no rewritable argument field present",
                    "type": "guardrail_violation",
                    "code": "panw_prisma_airs_blocked",
                }
            },
        )
        response = await create_response(mock_gen, "text/event-stream", {})
        content = await self.consume_stream(response)
        payload = json.loads(content[0][len("data: ") :].strip())
        assert payload["error"]["message"] == "MCP request blocked: no rewritable argument field present"
        assert payload["error"]["provider_specific_fields"]["error"]["code"] == "panw_prisma_airs_blocked"

    async def test_serialize_http_exception_detail_helper(self):
        """Direct unit coverage for the L1 helper across all branches."""
        from litellm.proxy.common_request_processing import (
            _serialize_http_exception_detail,
        )
        import json as _json

        assert _serialize_http_exception_detail("plain") == ("plain", None)

        msg, fields = _serialize_http_exception_detail({"error": "Violated", "extra": "x"})
        assert msg == "Violated"
        assert fields == {"error": "Violated", "extra": "x"}

        msg, fields = _serialize_http_exception_detail({"error": {"message": "blocked", "code": "x"}})
        assert msg == "blocked"
        assert fields == {"error": {"message": "blocked", "code": "x"}}

        msg, fields = _serialize_http_exception_detail({"message": "top-level"})
        assert msg == "top-level"
        assert fields == {"message": "top-level"}

        msg, fields = _serialize_http_exception_detail({"weird": ["a", "b"]})
        assert msg == _json.dumps({"weird": ["a", "b"]})
        assert fields == {"weird": ["a", "b"]}

        assert _serialize_http_exception_detail(42) == ("42", None)

    async def test_create_streaming_response_first_chunk_error_string_code(self):
        """
        Test that when the first chunk contains a string error code, a JSON error response is returned
        """

        async def mock_generator():
            yield 'data: {"error": {"code": "429", "message": "too many requests"}}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(mock_generator(), "text/event-stream", {})
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
        response = await create_response(mock_generator(), "text/event-stream", custom_headers)
        assert response.headers["x-custom-header"] == "TestValue"

    async def test_create_streaming_response_disables_proxy_buffering(self):
        """Regression for #28384: every StreamingResponse create_response returns
        must carry the headers that stop nginx/ingress/Envoy from buffering the
        SSE stream into one batch, while preserving caller-supplied headers."""

        async def normal_stream():
            yield 'data: {"content": "part"}\n\n'
            yield "data: [DONE]\n\n"

        async def empty_stream():
            if False:  # never yields -> StopAsyncIteration
                yield

        error_stream = AsyncMock()
        error_stream.__anext__.side_effect = ValueError("boom")

        for generator in (normal_stream(), empty_stream(), error_stream):
            response = await create_response(generator, "text/event-stream", {"X-Custom-Header": "keep"})
            assert isinstance(response, StreamingResponse)
            assert response.headers["x-accel-buffering"] == "no"
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["x-custom-header"] == "keep"

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

        response = await create_response(mock_generator(), "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK  # Default status
        content = await self.consume_stream(response)
        assert content == ["data: [DONE]\n\n"]

    async def test_create_streaming_response_first_chunk_is_empty_data(self):
        async def mock_generator():
            yield "data: \n\n"
            yield 'data: {"content": "actual data"}\n\n'
            yield "data: [DONE]\n\n"

        response = await create_response(mock_generator(), "text/event-stream", {})
        assert response.status_code == status.HTTP_200_OK  # Default status
        content = await self.consume_stream(response)
        assert content == [
            "data: \n\n",
            'data: {"content": "actual data"}\n\n',
            "data: [DONE]\n\n",
        ]

    async def test_create_streaming_response_all_chunks_have_dd_trace(self):
        """Test that all stream chunks are wrapped with dd trace at the streaming generator level"""
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

        # Patch the tracer in the common_request_processing module. The
        # per-chunk span is gated on _DD_STREAMING_TRACE_ENABLED (resolved at
        # import from the real tracer, a NullTracer by default), so enable it
        # explicitly to exercise the tracing path.
        with (
            patch("litellm.proxy.common_request_processing.tracer", mock_tracer),
            patch(
                "litellm.proxy.common_request_processing._DD_STREAMING_TRACE_ENABLED",
                True,
            ),
        ):
            response = await create_response(mock_generator(), "text/event-stream", {})

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
            actual_calls = mock_tracer.trace.call_args_list
            assert len(actual_calls) == 4

            for i, call in enumerate(actual_calls):
                args, kwargs = call
                assert args[0] == "streaming.chunk.yield", (
                    f"Call {i} should have operation name 'streaming.chunk.yield', got {args[0]}"
                )

    async def test_create_streaming_response_skips_dd_trace_when_disabled(self):
        """When DD tracing is disabled (the default), the per-chunk span
        context manager is skipped entirely but all chunks still stream."""
        from unittest.mock import patch

        mock_tracer = MagicMock()

        async def mock_generator():
            yield 'data: {"content": "chunk 1"}\n\n'
            yield 'data: {"content": "chunk 2"}\n\n'
            yield "data: [DONE]\n\n"

        with (
            patch("litellm.proxy.common_request_processing.tracer", mock_tracer),
            patch(
                "litellm.proxy.common_request_processing._DD_STREAMING_TRACE_ENABLED",
                False,
            ),
        ):
            response = await create_response(mock_generator(), "text/event-stream", {})

            assert response.status_code == 200

            content = await self.consume_stream(response)

            # All chunks stream through unchanged ...
            assert content == [
                'data: {"content": "chunk 1"}\n\n',
                'data: {"content": "chunk 2"}\n\n',
                "data: [DONE]\n\n",
            ]
            # ... but no per-chunk span was created.
            assert mock_tracer.trace.call_count == 0

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
            response = await create_response(mock_generator(), "text/event-stream", {})

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
        chunk = "data: {invalid json}\n\n"
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

    def test_override_model_preserves_fallback_model_when_fallback_occurred_object(
        self,
    ):
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
        response_obj._hidden_params = {"additional_headers": {"x-litellm-attempted-fallbacks": 1}}

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
        response_obj._hidden_params = {"additional_headers": {"x-litellm-attempted-fallbacks": None}}

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
        response_obj._hidden_params = {"additional_headers": {"x-litellm-attempted-fallbacks": 1}}

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

    def test_override_model_preserves_azure_model_router_actual_model(self):
        """
        Test that when the requested model is an Azure Model Router, the actual
        model used (returned in the response) is preserved instead of being
        overridden.
        """
        requested_model = "azure_ai/model_router"
        actual_model_used = "azure_ai/gpt-5-nano-2025-08-07"

        response_obj = MagicMock()
        response_obj.model = actual_model_used
        response_obj._hidden_params = {"additional_headers": {}}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == actual_model_used
        assert response_obj.model != requested_model

    def test_override_model_preserves_azure_model_router_with_deployment_name(self):
        """
        Test that Azure Model Router with deployment name pattern also preserves
        the actual model used.
        """
        requested_model = "azure_ai/model_router/my-deployment"
        actual_model_used = "azure_ai/gpt-4.1-nano-2025-04-14"

        response_obj = MagicMock()
        response_obj.model = actual_model_used
        response_obj._hidden_params = {"additional_headers": {}}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == actual_model_used
        assert response_obj.model != requested_model

    def test_override_model_preserves_azure_model_router_with_hyphen(self):
        """
        Test that Azure Model Router with hyphen pattern (model-router) also preserves
        the actual model used.
        """
        requested_model = "azure_ai/model-router"
        actual_model_used = "azure_ai/gpt-5-nano-2025-08-07"

        response_obj = MagicMock()
        response_obj.model = actual_model_used
        response_obj._hidden_params = {"additional_headers": {}}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == actual_model_used
        assert response_obj.model != requested_model

    def test_override_model_uses_winning_model_for_fastest_response(self):
        """
        Test that when fastest_response batch completion is used with a
        comma-separated model list, the response model is set to the winning
        model's group name (not the comma-separated list).
        """
        requested_model = "openai/gpt-4o,gemini/gemini-2.5-flash"
        winning_model_group = "gemini/gemini-2.5-flash"
        downstream_model = "gemini-2.5-flash"

        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "fastest_response_batch_completion": True,
            "additional_headers": {
                "x-litellm-model-group": winning_model_group,
            },
        }

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        assert response_obj.model == winning_model_group
        assert response_obj.model != requested_model

    def test_override_model_preserves_response_when_fastest_response_no_model_group(
        self,
    ):
        """
        Test that when fastest_response is set but no model group header is
        available, the actual downstream model is preserved.
        """
        requested_model = "openai/gpt-4o,gemini/gemini-2.5-flash"
        downstream_model = "gpt-4o-2024-08-06"

        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "fastest_response_batch_completion": True,
            "additional_headers": {},
        }

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        assert response_obj.model == downstream_model

    def test_override_model_normal_when_fastest_response_not_set(self):
        """
        Test that when fastest_response_batch_completion is not set, the
        normal override behavior applies (model is set to requested_model).
        """
        requested_model = "openai/gpt-4o"
        downstream_model = "gpt-4o-2024-08-06"

        response_obj = MagicMock()
        response_obj.model = downstream_model
        response_obj._hidden_params = {
            "additional_headers": {
                "x-litellm-model-group": "openai/gpt-4o",
            },
        }

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )

        assert response_obj.model == requested_model

    def test_skips_model_override_when_response_has_no_model_attribute(self):
        from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult

        response_obj = SearchResponse(
            results=[SearchResult(title="t", url="http://x.com", snippet="s")],
            object="search",
        )

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model="my-search-tool",
            log_context="test_context",
        )

        assert not hasattr(response_obj, "model")

    def test_skips_model_override_for_dict_without_model_key(self):
        response_obj = {
            "object": "search",
            "results": [{"title": "t", "url": "http://x.com", "snippet": "s"}],
        }

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model="my-search-tool",
            log_context="test_context",
        )

        assert "model" not in response_obj

    def test_override_model_swallows_setattr_failure(self):
        class ReadOnlyModelResponse:
            @property
            def model(self) -> str:
                return "downstream-model"

        response_obj = ReadOnlyModelResponse()

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model="my-model",
            log_context="test_context",
        )

        assert response_obj.model == "downstream-model"


class TestIsAzureModelRouterRequest:
    """Tests for _is_azure_model_router_request helper"""

    def test_detects_model_router_with_underscore(self):
        assert _is_azure_model_router_request("azure_ai/model_router") is True
        assert _is_azure_model_router_request("azure_ai/model_router/my-deployment") is True

    def test_detects_model_router_with_hyphen(self):
        assert _is_azure_model_router_request("azure_ai/model-router") is True
        assert _is_azure_model_router_request("model-router") is True

    def test_rejects_regular_models(self):
        assert _is_azure_model_router_request("azure_ai/gpt-4") is False
        assert _is_azure_model_router_request("gpt-4") is False
        assert _is_azure_model_router_request("openai/gpt-3.5-turbo") is False


class TestStreamingOverheadHeader:
    """
    Tests that x-litellm-overhead-duration-ms is emitted in streaming responses.

    Regression tests for: streaming requests not including overhead header.
    """

    def test_get_custom_headers_includes_overhead_when_set(self):
        """
        get_custom_headers() returns x-litellm-overhead-duration-ms
        when litellm_overhead_time_ms is in hidden_params.
        """
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.0
        mock_user_api_key_dict.allowed_model_region = None

        hidden_params = {
            "litellm_overhead_time_ms": 42.5,
            "_response_ms": 500.0,
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com",
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            model_id="test-model-id",
            cache_key="",
            api_base="https://api.openai.com",
            version="1.0.0",
            response_cost=0.001,
            model_region="",
            hidden_params=hidden_params,
        )

        assert "x-litellm-overhead-duration-ms" in headers
        assert headers["x-litellm-overhead-duration-ms"] == "42.5"

    def test_get_custom_headers_omits_overhead_when_none(self):
        """
        get_custom_headers() omits x-litellm-overhead-duration-ms
        when litellm_overhead_time_ms is not in hidden_params.
        """
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.0
        mock_user_api_key_dict.allowed_model_region = None

        hidden_params = {
            "_response_ms": 500.0,
            "model_id": "test-model-id",
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            model_id="test-model-id",
            cache_key="",
            api_base="https://api.openai.com",
            version="1.0.0",
            response_cost=0.001,
            model_region="",
            hidden_params=hidden_params,
        )

        # Should be absent (None gets filtered by exclude_values)
        assert "x-litellm-overhead-duration-ms" not in headers

    def test_update_response_metadata_sets_overhead_on_stream_wrapper(self):
        """
        update_response_metadata() sets litellm_overhead_time_ms on
        a streaming response's _hidden_params when llm_api_duration_ms is available.
        """
        from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
            update_response_metadata,
        )

        # Mock the logging object with llm_api_duration_ms set
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "llm_api_duration_ms": 200.0,
            "litellm_params": {},
        }
        mock_logging_obj.caching_details = None
        mock_logging_obj.callback_duration_ms = None
        mock_logging_obj.litellm_call_id = "test-call-id"
        mock_logging_obj._response_cost_calculator = MagicMock(return_value=0.001)

        # Simulate a streaming result object with _hidden_params (like CustomStreamWrapper)
        stream_result = MagicMock()
        stream_result._hidden_params = {
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com",
            "additional_headers": {},
        }

        start_time = datetime.datetime.now() - datetime.timedelta(milliseconds=300)
        end_time = datetime.datetime.now()

        update_response_metadata(
            result=stream_result,
            logging_obj=mock_logging_obj,
            model="gpt-4o",
            kwargs={},
            start_time=start_time,
            end_time=end_time,
        )

        assert "litellm_overhead_time_ms" in stream_result._hidden_params
        overhead = stream_result._hidden_params["litellm_overhead_time_ms"]
        assert overhead is not None
        assert isinstance(overhead, float)
        # overhead = total_response_ms (~300ms) - llm_api_duration_ms (200ms) = ~100ms
        assert overhead > 0

    @pytest.mark.asyncio
    async def test_streaming_response_includes_overhead_header(self):
        """
        StreamingResponse returned by create_response() includes
        x-litellm-overhead-duration-ms in its headers.
        """

        async def mock_generator() -> AsyncGenerator[str, None]:
            yield 'data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        headers = {
            "x-litellm-overhead-duration-ms": "42.5",
            "x-litellm-call-id": "test-call-id",
            "x-litellm-model-id": "test-model-id",
        }

        response = await create_response(
            generator=mock_generator(),
            media_type="text/event-stream",
            headers=headers,
        )

        assert isinstance(response, StreamingResponse)
        assert response.headers.get("x-litellm-overhead-duration-ms") == "42.5"

    def test_streaming_overhead_header_in_custom_headers_from_stream_hidden_params(
        self,
    ):
        """
        Verifies that when get_custom_headers() is called with a streaming
        response's hidden_params (containing litellm_overhead_time_ms),
        the x-litellm-overhead-duration-ms header is correctly populated.

        This tests the critical path: update_response_metadata sets the value
        → get_custom_headers reads it → StreamingResponse header is set.
        """
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0.0
        mock_user_api_key_dict.allowed_model_region = None

        # This is what CustomStreamWrapper._hidden_params looks like after
        # update_response_metadata() has been called on it
        hidden_params = {
            "model_id": "openai-gpt4o-deployment",
            "api_base": "https://api.openai.com",
            "additional_headers": {},
            "litellm_overhead_time_ms": 55.3,  # set by update_response_metadata
            "_response_ms": 280.0,
            "litellm_call_id": "test-call-id",
            "response_cost": 0.002,
            "cache_key": None,
            "fastest_response_batch_completion": None,
            "callback_duration_ms": None,
        }

        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id",
            model_id=hidden_params.get("model_id"),
            cache_key=hidden_params.get("cache_key") or "",
            api_base=hidden_params.get("api_base") or "",
            version="1.0.0",
            response_cost=hidden_params.get("response_cost"),
            model_region="",
            hidden_params=hidden_params,
        )

        # The overhead header must be present and correct
        assert "x-litellm-overhead-duration-ms" in custom_headers, (
            "x-litellm-overhead-duration-ms header must be emitted during streaming. "
            "It was missing — this is the streaming overhead header regression."
        )
        assert custom_headers["x-litellm-overhead-duration-ms"] == "55.3"


class TestDDSpanTaggerTagRequest:
    """Tests for DDSpanTagger.tag_request - key/model DD span tagging."""

    def _make_user_api_key_dict(self, key_alias=None, token=None):
        from litellm.proxy._types import UserAPIKeyAuth

        d = UserAPIKeyAuth()
        d.key_alias = key_alias
        d.token = token
        return d

    def test_tags_key_alias_and_model(self):
        """key_alias and requested_model are set on the span when present."""
        user_key = self._make_user_api_key_dict(key_alias="my-prod-key", token="hashed123")

        with patch("litellm.proxy.dd_span_tagger.set_active_span_tag") as mock_set_tag:
            DDSpanTagger.tag_request(
                user_api_key_dict=user_key,
                requested_model="gpt-4o",
            )

        mock_set_tag.assert_any_call("litellm.key_alias", "my-prod-key")
        mock_set_tag.assert_any_call("litellm.key_hash", "hashed123")
        mock_set_tag.assert_any_call("litellm.requested_model", "gpt-4o")

    def test_no_tags_when_key_absent(self):
        """No key tags are set when key_alias and token are None (e.g. 401 path)."""
        user_key = self._make_user_api_key_dict(key_alias=None, token=None)

        with patch("litellm.proxy.dd_span_tagger.set_active_span_tag") as mock_set_tag:
            DDSpanTagger.tag_request(
                user_api_key_dict=user_key,
                requested_model=None,
            )

        mock_set_tag.assert_not_called()

    def test_only_model_tagged_when_no_key_info(self):
        """requested_model is tagged even when there's no key info."""
        user_key = self._make_user_api_key_dict(key_alias=None, token=None)

        with patch("litellm.proxy.dd_span_tagger.set_active_span_tag") as mock_set_tag:
            DDSpanTagger.tag_request(
                user_api_key_dict=user_key,
                requested_model="claude-3-5-sonnet",
            )

        mock_set_tag.assert_called_once_with("litellm.requested_model", "claude-3-5-sonnet")


class TestHasAttributeErrorInChain:
    """Tests for _has_attribute_error_in_chain helper."""

    def test_direct_attribute_error(self):
        exc = AttributeError("'str' object has no attribute 'get'")
        assert _has_attribute_error_in_chain(exc) is True

    def test_no_attribute_error(self):
        exc = ValueError("some other error")
        assert _has_attribute_error_in_chain(exc) is False

    def test_attribute_error_in_cause(self):
        inner = AttributeError("bad attribute")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert _has_attribute_error_in_chain(outer) is True

    def test_attribute_error_in_context(self):
        inner = AttributeError("bad attribute")
        outer = RuntimeError("wrapper")
        outer.__context__ = inner
        assert _has_attribute_error_in_chain(outer) is True

    def test_attribute_error_in_original_exception(self):
        inner = AttributeError("bad attribute")
        outer = RuntimeError("wrapper")
        outer.original_exception = inner  # type: ignore
        assert _has_attribute_error_in_chain(outer) is True

    def test_attribute_error_nested_two_levels(self):
        """Simulates the real failure: AttributeError -> OpenAIException -> APIConnectionError."""
        attr_err = AttributeError("'str' object has no attribute 'get'")
        mid = Exception("OpenAIException wrapper")
        mid.__context__ = attr_err
        outer = Exception("APIConnectionError wrapper")
        outer.__context__ = mid
        assert _has_attribute_error_in_chain(outer) is True

    def test_depth_limit_prevents_infinite_loop(self):
        """Ensure circular references don't cause infinite recursion."""
        exc_a = RuntimeError("a")
        exc_b = RuntimeError("b")
        exc_a.__context__ = exc_b
        exc_b.__context__ = exc_a  # circular
        assert _has_attribute_error_in_chain(exc_a) is False


@pytest.mark.asyncio
class TestHandleLLMApiExceptionDictDetail:
    """
    Coverage for `_handle_llm_api_exception` HTTPException branch (Site 2).
    Regression for case 2026-04-10-internal-bedrock-guardrail-streaming-error:
    dict-detail HTTPExceptions raised by guardrails must round-trip cleanly
    through ProxyException instead of being str()-mangled into a Python repr.
    """

    async def _invoke(self, exc: Exception):
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth

        processor = ProxyBaseLLMRequestProcessing(data={})
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})

        try:
            await processor._handle_llm_api_exception(
                e=exc,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=proxy_logging_obj,
            )
        except ProxyException as raised:
            return raised
        raise AssertionError("ProxyException was not raised")

    async def test_dict_detail_bedrock_shape_preserved(self):
        exc = HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy",
                "bedrock_guardrail_response": "...",
                "guardrail_name": "bedrock-pii-guard",
            },
        )
        proxy_exc = await self._invoke(exc)
        assert proxy_exc.message == "Violated guardrail policy"
        assert proxy_exc.provider_specific_fields["guardrail_name"] == "bedrock-pii-guard"
        # No Python repr leakage of the dict into the message field.
        assert "{'error':" not in proxy_exc.message

    async def test_string_detail_unchanged(self):
        exc = HTTPException(status_code=400, detail="Content blocked by guardrail")
        proxy_exc = await self._invoke(exc)
        assert proxy_exc.message == "Content blocked by guardrail"
        assert proxy_exc.provider_specific_fields is None

    async def test_not_found_error_preserves_404(self):
        """NotFoundError with status_code=404 should map to ProxyException code=404."""
        from litellm.exceptions import NotFoundError

        exc = NotFoundError(
            message="Model gemini-3.1-flash-lite-preview not found",
            model="gemini-3.1-flash-lite-preview",
            llm_provider="gemini",
        )
        proxy_exc = await self._invoke(exc)
        assert proxy_exc.code == "404"
        assert "NotFoundError" in proxy_exc.message

    async def test_exception_with_status_code_propagates(self):
        """Exception with a statically-set status_code should propagate it."""
        from litellm.llms.vertex_ai.common_utils import VertexAIError

        exc = VertexAIError(
            status_code=429,
            message="Rate limit exceeded",
        )
        proxy_exc = await self._invoke(exc)
        assert proxy_exc.code == "429"

    async def test_exception_without_status_code_defaults_to_500(self):
        """Exception with no status_code attribute defaults to 500."""
        exc = ValueError("Something broke")
        proxy_exc = await self._invoke(exc)
        assert proxy_exc.code == "500"

    async def test_already_normalized_proxy_exception_is_honored(self):
        """A ProxyException raised mid-request (e.g. a guardrail block) is already
        the OpenAI wire format. The funnel must re-raise it untouched instead of
        re-deriving the status from a (nonexistent) status_code attribute and
        defaulting to 500. Regression for LIT-3751."""
        from litellm.proxy._types import ProxyException

        exc = ProxyException(
            message='"Leroy Jenkins" detected as name',
            type="invalid_request_error",
            param=None,
            code=400,
            openai_code="content_policy_violation",
        )
        proxy_exc = await self._invoke(exc)
        assert proxy_exc is exc
        assert proxy_exc.code == "400"
        assert proxy_exc.type == "invalid_request_error"
        assert proxy_exc.param is None
        assert proxy_exc.openai_code == "content_policy_violation"
        assert proxy_exc.message == '"Leroy Jenkins" detected as name'

        # The body the OpenAI-SDK client actually receives. The HTTP status line
        # comes from int(exc.code) == 400; the wire ``code`` stays the status
        # string. ``openai_code`` ("content_policy_violation") is intentionally
        # NOT serialized here - to_dict() emits only ``code`` - so this asserts
        # the real contract rather than the write-only attribute.
        assert int(proxy_exc.code) == 400
        assert proxy_exc.to_dict() == {
            "message": '"Leroy Jenkins" detected as name',
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }


class TestStreamCloseOnDisconnect:
    """
    Coverage for closing the upstream LLM stream when the client disconnects
    mid-stream. Starlette abandons the response body iterator without calling
    aclose(), so without these hooks the proxy->backend connection stays open
    and the backend (e.g. vLLM) keeps generating into a dead pipe.
    """

    async def test_response_closes_body_iterator_when_task_cancelled(self):
        """Cancellation landing in send() leaves the generator suspended at a
        yield; only the response-level finally can close it."""
        closed = asyncio.Event()

        async def body():
            try:
                while True:
                    yield "data: x\n\n"
            finally:
                closed.set()

        response = _UpstreamClosingStreamingResponse(
            body(), media_type="text/event-stream"
        )

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                await asyncio.Event().wait()

        task = asyncio.create_task(response({"type": "http"}, receive, send))
        await asyncio.sleep(0.05)
        assert not closed.is_set()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert closed.is_set()

    async def test_response_closes_body_iterator_on_http_disconnect(self):
        closed = asyncio.Event()
        disconnected = asyncio.Event()
        body_sends = 0

        async def body():
            try:
                for i in range(1000):
                    yield f"data: {i}\n\n"
            finally:
                closed.set()

        response = _UpstreamClosingStreamingResponse(
            body(), media_type="text/event-stream"
        )

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal body_sends
            if message["type"] == "http.response.body":
                body_sends += 1
                if body_sends == 3:
                    disconnected.set()
                await asyncio.sleep(0.05)

        await response({"type": "http"}, receive, send)

        assert closed.is_set()
        assert body_sends < 1000

    async def test_upstream_closed_even_if_body_iterator_aclose_raises(self):
        """A BaseException from body_iterator.aclose() (e.g. CancelledError)
        must not prevent the upstream generator from being closed."""
        upstream_closed = asyncio.Event()

        class ExplodingIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                raise asyncio.CancelledError()

        async def upstream():
            try:
                yield "data: a\n\n"
            finally:
                upstream_closed.set()

        upstream_gen = upstream()
        await upstream_gen.__anext__()
        response = _UpstreamClosingStreamingResponse(
            ExplodingIterator(),
            media_type="text/event-stream",
            upstream_generator=upstream_gen,
        )

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            pass

        await response({"type": "http"}, receive, send)

        assert upstream_closed.is_set()

    async def test_create_response_closes_wrapped_generator_on_cancellation(self):
        """End to end through create_response: the upstream-facing generator
        must be closed even when the body iterator was never started (client
        gone before the first chunk could be sent)."""
        inner_closed = asyncio.Event()

        async def wrapped():
            try:
                while True:
                    yield "data: a\n\n"
            finally:
                inner_closed.set()

        response = await create_response(
            generator=wrapped(), media_type="text/event-stream", headers={}
        )

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            await asyncio.Event().wait()

        task = asyncio.create_task(response({"type": "http"}, receive, send))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert inner_closed.is_set()

    async def test_async_streaming_data_generator_closes_upstream_on_early_close(
        self,
    ):
        class FakeUpstream:
            def __init__(self):
                self.aclosed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                return {"type": "chunk"}

            async def aclose(self):
                self.aclosed = True

        ProxyLogging._callback_capabilities_cache.clear()
        upstream = FakeUpstream()
        gen = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=upstream,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            request_data={"model": "mock-model"},
            proxy_logging_obj=ProxyLogging(user_api_key_cache=MagicMock()),
            serialize_chunk=lambda c: "data: x\n\n",
            serialize_error=lambda e: "data: error\n\n",
        )

        await gen.__anext__()
        await gen.__anext__()
        assert not upstream.aclosed

        await gen.aclose()

        assert upstream.aclosed

    @staticmethod
    def _request_that_disconnects() -> Request:
        async def receive():
            return {"type": "http.disconnect"}

        return Request({"type": "http", "method": "POST", "headers": []}, receive)

    @staticmethod
    def _request_that_stays_connected() -> Request:
        async def receive():
            await asyncio.Event().wait()

        return Request({"type": "http", "method": "POST", "headers": []}, receive)

    async def test_create_response_returns_499_on_disconnect_before_first_chunk(self):
        """LIT-3568: client disconnects during the time-to-first-token wait.

        create_response buffers the first chunk before Starlette starts serving
        the StreamingResponse, so this window has no disconnect listener. The
        request must be cancelled (upstream generator closed) and a 499 returned
        instead of blocking until the request timeout.
        """
        upstream_closed = asyncio.Event()

        async def never_yields_first_chunk():
            try:
                await asyncio.Event().wait()
                yield "data: never\n\n"
            finally:
                upstream_closed.set()

        response = await asyncio.wait_for(
            create_response(
                generator=never_yields_first_chunk(),
                media_type="text/event-stream",
                headers={},
                request=self._request_that_disconnects(),
            ),
            timeout=5,
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 499
        assert upstream_closed.is_set()

    async def test_create_response_streams_normally_when_connected(self):
        """The disconnect race must not steal a first chunk that does arrive:
        a connected client still gets a StreamingResponse, not a 499."""

        async def yields_immediately():
            yield "data: hello\n\n"
            yield "data: world\n\n"

        response = await asyncio.wait_for(
            create_response(
                generator=yields_immediately(),
                media_type="text/event-stream",
                headers={},
                request=self._request_that_stays_connected(),
            ),
            timeout=5,
        )

        assert isinstance(response, StreamingResponse)
        assert response.status_code == status.HTTP_200_OK

    async def test_buffer_first_chunk_without_request_is_passthrough(self):
        """No request -> preserve the original eager __anext__ behavior."""

        async def gen():
            yield "data: first\n\n"

        first = await _buffer_first_chunk_honoring_disconnect(gen(), request=None)
        assert first == "data: first\n\n"

    async def test_create_response_prioritizes_disconnect_in_same_scheduler_turn(self):
        """Same-turn race: the first chunk and the disconnect both resolve before
        the branch runs. Because the disconnect watcher has already consumed
        http.disconnect, returning the chunk would leave Starlette's later
        listener blind to it and the upstream running. The observed disconnect
        must win -> 499 and the generator closed."""
        closed = asyncio.Event()

        async def yields_immediately():
            try:
                yield "data: hello\n\n"
            finally:
                closed.set()

        response = await asyncio.wait_for(
            create_response(
                generator=yields_immediately(),
                media_type="text/event-stream",
                headers={},
                request=self._request_that_disconnects(),
            ),
            timeout=5,
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 499
        assert closed.is_set()

    async def test_receive_error_does_not_trigger_false_disconnect(self):
        """A request.receive() that raises must not masquerade as a disconnect;
        a first chunk that arrives is still served as a normal stream."""

        async def receive():
            raise RuntimeError("receive boom")

        request = Request({"type": "http", "method": "POST", "headers": []}, receive)

        async def yields_immediately():
            yield "data: hello\n\n"

        response = await asyncio.wait_for(
            create_response(
                generator=yields_immediately(),
                media_type="text/event-stream",
                headers={},
                request=request,
            ),
            timeout=5,
        )

        assert isinstance(response, StreamingResponse)
        assert response.status_code == status.HTTP_200_OK

    async def test_disconnect_cancellation_survives_generator_aclose_error(self):
        """A failing upstream aclose() during disconnect cleanup must not swallow
        the disconnect signal: the sentinel is still raised."""

        class AcloseRaises:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()
                raise StopAsyncIteration

            async def aclose(self):
                raise RuntimeError("aclose boom")

        with pytest.raises(_ClientDisconnectedBeforeFirstChunk):
            await asyncio.wait_for(
                _buffer_first_chunk_honoring_disconnect(
                    AcloseRaises(), request=self._request_that_disconnects()
                ),
                timeout=5,
            )

    async def test_buffer_first_chunk_raises_sentinel_and_closes_on_disconnect(self):
        closed = asyncio.Event()

        async def blocking_gen():
            try:
                await asyncio.Event().wait()
                yield "data: never\n\n"
            finally:
                closed.set()

        with pytest.raises(_ClientDisconnectedBeforeFirstChunk):
            await asyncio.wait_for(
                _buffer_first_chunk_honoring_disconnect(
                    blocking_gen(), request=self._request_that_disconnects()
                ),
                timeout=5,
            )
        assert closed.is_set()


class TestHandleLLMApiExceptionRetryAfter:
    """RouterRateLimitError cooldown_time must surface as a retry-after header."""

    async def _invoke(self, exc: Exception, callback_headers: Optional[dict] = None):
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth

        processor = ProxyBaseLLMRequestProcessing(data={})
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(
            return_value=callback_headers or {}
        )

        try:
            await processor._handle_llm_api_exception(
                e=exc,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=proxy_logging_obj,
            )
        except ProxyException as raised:
            return raised
        raise AssertionError("ProxyException was not raised")

    async def test_handle_llm_api_exception_sets_retry_after_from_cooldown_time(self):
        from litellm.types.router import RouterRateLimitError

        exc = RouterRateLimitError(
            model="gpt-4",
            cooldown_time=42.3,
            enable_pre_call_checks=False,
            cooldown_list=[],
        )
        proxy_exc = await self._invoke(exc)
        assert proxy_exc.headers["retry-after"] == "43"
        assert proxy_exc.code == "429"

    async def test_handle_llm_api_exception_skips_retry_after_when_cooldown_is_zero(
        self,
    ):
        from litellm.types.router import RouterRateLimitError

        exc = RouterRateLimitError(
            model="gpt-4",
            cooldown_time=0,
            enable_pre_call_checks=False,
            cooldown_list=[],
        )
        proxy_exc = await self._invoke(exc)
        assert "retry-after" not in proxy_exc.headers

    async def test_handle_llm_api_exception_no_retry_after_for_plain_exception(self):
        proxy_exc = await self._invoke(ValueError("some other failure"))
        assert "retry-after" not in proxy_exc.headers

    async def test_handle_llm_api_exception_retry_after_survives_callback_headers(self):
        from litellm.types.router import RouterRateLimitError

        exc = RouterRateLimitError(
            model="gpt-4",
            cooldown_time=42.3,
            enable_pre_call_checks=False,
            cooldown_list=[],
        )
        proxy_exc = await self._invoke(
            exc, callback_headers={"retry-after": "", "x-custom": "1"}
        )
        assert proxy_exc.headers["retry-after"] == "43"
        assert proxy_exc.headers["x-custom"] == "1"


class TestAsyncStreamingDataGeneratorFastPath:
    """Fast/slow path branching in async_streaming_data_generator."""

    @staticmethod
    async def _aiter(items):
        for item in items:
            yield item

    @pytest.mark.asyncio
    async def test_fast_path_skips_per_chunk_hook(self, monkeypatch):
        """With no callbacks/guardrails/cost-injection, chunks pass through
        unchanged and the per-chunk hook is NOT awaited."""
        monkeypatch.setattr(litellm, "callbacks", [])
        ProxyLogging._callback_capabilities_cache.clear()

        proxy_logging_obj = ProxyLogging(user_api_key_cache=MagicMock())
        hook_spy = AsyncMock(side_effect=lambda **kw: kw["response"])
        monkeypatch.setattr(proxy_logging_obj, "async_post_call_streaming_hook", hook_spy)

        chunks = [b"event: a\ndata: {}\n\n", b"event: b\ndata: {}\n\n"]
        out = [
            c
            async for c in ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
                response=self._aiter(chunks),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                request_data={"model": "claude-x"},
                proxy_logging_obj=proxy_logging_obj,
                serialize_chunk=ProxyBaseLLMRequestProcessing.return_sse_chunk,
                serialize_error=lambda e: "data: error\n\n",
            )
        ]

        assert out == chunks  # bytes pass through return_sse_chunk untouched
        hook_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slow_path_runs_per_chunk_hook(self, monkeypatch):
        """A callback that overrides async_post_call_streaming_hook forces the
        slow path and the per-chunk hook is invoked."""

        class _StreamingCb(CustomLogger):
            async def async_post_call_streaming_hook(self, user_api_key_dict, response):
                return response

        cb = _StreamingCb()
        monkeypatch.setattr(litellm, "callbacks", [cb])
        ProxyLogging._callback_capabilities_cache.clear()

        proxy_logging_obj = ProxyLogging(user_api_key_cache=MagicMock())
        hook_spy = AsyncMock(side_effect=lambda **kw: kw["response"])
        monkeypatch.setattr(proxy_logging_obj, "async_post_call_streaming_hook", hook_spy)

        out = [
            c
            async for c in ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
                response=self._aiter([{"type": "message_stop"}]),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                request_data={"model": "claude-x"},
                proxy_logging_obj=proxy_logging_obj,
                serialize_chunk=ProxyBaseLLMRequestProcessing.return_sse_chunk,
                serialize_error=lambda e: "data: error\n\n",
            )
        ]

        assert len(out) == 1
        hook_spy.assert_awaited_once()

        ProxyLogging._callback_capabilities_cache.clear()


class TestDisconnectGatherCleanup:
    def _disconnect_request(self) -> Request:
        messages = [
            {"type": "http.request", "body": b"", "more_body": False},
            {"type": "http.disconnect"},
        ]

        async def receive():
            if messages:
                return messages.pop(0)
            await asyncio.Event().wait()

        return Request(scope={"type": "http", "headers": []}, receive=receive)

    @pytest.mark.asyncio
    async def test_base_process_llm_request_raises_499_on_client_disconnect(
        self, monkeypatch
    ):
        """With cancel_on_disconnect enabled, base_process_llm_request returns 499."""
        import asyncio

        import litellm.proxy.common_request_processing as cpr
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        async def slow_llm():
            await asyncio.sleep(9999)

        async def fake_route_request(**_kwargs):
            return slow_llm()

        mock_logging_obj = MagicMock()
        mock_logging_obj.litellm_call_id = "test-call-id"
        mock_logging_obj._defer_async_logging = False

        mock_proxy_logging = MagicMock(spec=ProxyLogging)
        mock_proxy_logging.during_call_hook = AsyncMock(return_value=None)
        mock_proxy_logging._callback_capabilities_cache = {}

        monkeypatch.setattr(cpr, "route_request", fake_route_request)

        processing_obj = ProxyBaseLLMRequestProcessing(data={"model": "gemini-2.0-flash"})
        monkeypatch.setattr(
            processing_obj,
            "common_processing_pre_call_logic",
            AsyncMock(return_value=({"model": "gemini-2.0-flash"}, mock_logging_obj)),
        )
        monkeypatch.setattr(
            processing_obj, "_has_post_call_guardrails", MagicMock(return_value=False)
        )

        with pytest.raises(HTTPException) as exc_info:
            await processing_obj.base_process_llm_request(
                request=self._disconnect_request(),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                proxy_logging_obj=mock_proxy_logging,
                general_settings={"cancel_on_disconnect": True},
                proxy_config=MagicMock(spec=ProxyConfig),
                route_type="acompletion",
                version=None,
            )

        assert exc_info.value.status_code == 499
        assert "disconnected" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_base_process_llm_request_reraises_cancelled_error_without_client_disconnect(
        self, monkeypatch
    ):
        import asyncio

        import litellm.proxy.common_request_processing as cpr
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        async def fake_gather(*_tasks, **_kwargs):
            raise asyncio.CancelledError()

        mock_logging_obj = MagicMock()
        mock_logging_obj.litellm_call_id = "test-call-id"
        mock_logging_obj._defer_async_logging = False

        mock_proxy_logging = MagicMock(spec=ProxyLogging)
        mock_proxy_logging.during_call_hook = AsyncMock(return_value=None)
        mock_proxy_logging._callback_capabilities_cache = {}

        monkeypatch.setattr(cpr.asyncio, "gather", fake_gather)

        processing_obj = ProxyBaseLLMRequestProcessing(data={"model": "gemini-2.0-flash"})
        monkeypatch.setattr(
            processing_obj,
            "common_processing_pre_call_logic",
            AsyncMock(return_value=({"model": "gemini-2.0-flash"}, mock_logging_obj)),
        )
        monkeypatch.setattr(
            processing_obj, "_has_post_call_guardrails", MagicMock(return_value=False)
        )
        monkeypatch.setattr(
            cpr,
            "route_request",
            AsyncMock(return_value=asyncio.sleep(9999)),
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        with pytest.raises(asyncio.CancelledError):
            await processing_obj.base_process_llm_request(
                request=mock_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                proxy_logging_obj=mock_proxy_logging,
                general_settings={},
                proxy_config=MagicMock(spec=ProxyConfig),
                route_type="acompletion",
                version=None,
            )

    @pytest.mark.asyncio
    async def test_disconnect_cancels_during_call_hook_task(self, monkeypatch):
        import asyncio

        import litellm.proxy.common_request_processing as cpr
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        hook_cancelled = False

        async def slow_during_call_hook(**_kwargs):
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                nonlocal hook_cancelled
                hook_cancelled = True
                raise

        async def slow_llm():
            await asyncio.sleep(9999)

        async def fake_route_request(**_kwargs):
            return slow_llm()

        mock_logging_obj = MagicMock()
        mock_logging_obj.litellm_call_id = "test-call-id"
        mock_logging_obj._defer_async_logging = False

        mock_proxy_logging = MagicMock(spec=ProxyLogging)
        mock_proxy_logging.during_call_hook = slow_during_call_hook
        mock_proxy_logging._callback_capabilities_cache = {}

        monkeypatch.setattr(cpr, "route_request", fake_route_request)

        processing_obj = ProxyBaseLLMRequestProcessing(data={"model": "gemini-2.0-flash"})
        monkeypatch.setattr(
            processing_obj,
            "common_processing_pre_call_logic",
            AsyncMock(return_value=({"model": "gemini-2.0-flash"}, mock_logging_obj)),
        )
        monkeypatch.setattr(
            processing_obj, "_has_post_call_guardrails", MagicMock(return_value=False)
        )

        with pytest.raises(HTTPException):
            await processing_obj.base_process_llm_request(
                request=self._disconnect_request(),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                proxy_logging_obj=mock_proxy_logging,
                general_settings={"cancel_on_disconnect": True},
                proxy_config=MagicMock(spec=ProxyConfig),
                route_type="acompletion",
                version=None,
            )

        assert hook_cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_pending_gather_tasks_skips_already_done_tasks(self):
        import asyncio

        from litellm.proxy.common_request_processing import _cancel_pending_gather_tasks

        async def failing_task():
            raise ValueError("llm api error")

        task = asyncio.create_task(failing_task())
        with pytest.raises(ValueError, match="llm api error"):
            await task

        await _cancel_pending_gather_tasks([task])

    @pytest.mark.asyncio
    async def test_cancel_pending_gather_tasks_swallows_guardrail_converted_cancel(
        self,
    ):
        import asyncio

        from litellm.proxy.common_request_processing import _cancel_pending_gather_tasks

        async def hook_converts_cancel_to_runtime_error():
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                raise RuntimeError("guardrail converted cancel")

        task = asyncio.create_task(hook_converts_cancel_to_runtime_error())
        await asyncio.sleep(0)
        await _cancel_pending_gather_tasks([task])
        assert task.done()

    @pytest.mark.asyncio
    async def test_base_process_llm_request_preserves_llm_error_after_gather(
        self, monkeypatch
    ):
        import asyncio

        import litellm.proxy.common_request_processing as cpr
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        async def failing_llm():
            raise ValueError("llm api error")

        async def successful_hook(**_kwargs):
            return None

        async def fake_route_request(**_kwargs):
            return failing_llm()

        mock_logging_obj = MagicMock()
        mock_logging_obj.litellm_call_id = "test-call-id"
        mock_logging_obj._defer_async_logging = False

        mock_proxy_logging = MagicMock(spec=ProxyLogging)
        mock_proxy_logging.during_call_hook = successful_hook
        mock_proxy_logging._callback_capabilities_cache = {}

        monkeypatch.setattr(cpr, "route_request", fake_route_request)

        processing_obj = ProxyBaseLLMRequestProcessing(data={"model": "gemini-2.0-flash"})
        monkeypatch.setattr(
            processing_obj,
            "common_processing_pre_call_logic",
            AsyncMock(return_value=({"model": "gemini-2.0-flash"}, mock_logging_obj)),
        )
        monkeypatch.setattr(
            processing_obj, "_has_post_call_guardrails", MagicMock(return_value=False)
        )

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=False)
        mock_request.headers = {}

        with pytest.raises(ValueError, match="llm api error"):
            await processing_obj.base_process_llm_request(
                request=mock_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                proxy_logging_obj=mock_proxy_logging,
                general_settings={},
                proxy_config=MagicMock(spec=ProxyConfig),
                route_type="acompletion",
                version=None,
            )


class TestStreamingClientDisconnectLogging:
    @pytest.mark.asyncio
    async def test_record_streaming_client_disconnect_sets_error_information(self):
        from litellm.proxy.common_request_processing import (
            _record_streaming_client_disconnect_if_needed,
        )

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"litellm_params": {}, "metadata": {}}
        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=True)
        request_data = {
            "litellm_call_id": "test-call-id",
            "litellm_logging_obj": mock_logging_obj,
            "metadata": {},
            "litellm_params": {"metadata": {}},
        }

        recorded = await _record_streaming_client_disconnect_if_needed(
            mock_request, request_data
        )

        assert recorded is True
        assert request_data["metadata"]["client_disconnected"] is True
        assert (
            request_data["metadata"]["error_information"]["error_code"] == "499"
        )
        assert (
            mock_logging_obj.model_call_details["litellm_params"]["metadata"][
                "error_information"
            ]["error_code"]
            == "499"
        )

    @pytest.mark.asyncio
    async def test_record_streaming_client_disconnect_no_op_when_connected(self):
        from litellm.proxy.common_request_processing import (
            _record_streaming_client_disconnect_if_needed,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=False)
        request_data = {"metadata": {}}

        recorded = await _record_streaming_client_disconnect_if_needed(
            mock_request, request_data
        )

        assert recorded is False
        assert "client_disconnected" not in request_data["metadata"]

    @pytest.mark.asyncio
    async def test_record_streaming_client_disconnect_handles_none_metadata(self):
        from litellm.proxy.common_request_processing import (
            _record_streaming_client_disconnect_if_needed,
        )

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "litellm_params": {"metadata": None},
            "metadata": None,
        }
        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=True)
        request_data = {
            "litellm_call_id": "test-call-id",
            "litellm_logging_obj": mock_logging_obj,
            "metadata": {},
            "litellm_params": {"metadata": {}},
        }

        recorded = await _record_streaming_client_disconnect_if_needed(
            mock_request, request_data
        )

        assert recorded is True
        assert request_data["metadata"]["client_disconnected"] is True
        assert (
            mock_logging_obj.model_call_details["litellm_params"]["metadata"][
                "client_disconnected"
            ]
            is True
        )
        assert (
            mock_logging_obj.model_call_details["metadata"]["client_disconnected"]
            is True
        )

    @pytest.mark.asyncio
    async def test_record_streaming_client_disconnect_handles_none_request_data_metadata(self):
        from litellm.proxy.common_request_processing import (
            _record_streaming_client_disconnect_if_needed,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=True)
        request_data = {
            "litellm_call_id": "test-call-id",
            "metadata": None,
            "litellm_params": {"metadata": None},
        }

        recorded = await _record_streaming_client_disconnect_if_needed(
            mock_request, request_data
        )

        assert recorded is True
        assert request_data["metadata"]["client_disconnected"] is True
        assert (
            request_data["litellm_params"]["metadata"]["client_disconnected"] is True
        )

    @pytest.mark.asyncio
    async def test_apply_client_disconnect_metadata_none_returns_early(self):
        from litellm.proxy.common_request_processing import (
            _apply_client_disconnect_metadata,
        )

        _apply_client_disconnect_metadata(None)

    @pytest.mark.asyncio
    async def test_finalize_streaming_generator_cleanup_fires_deferred_logging(
        self, monkeypatch
    ):
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        fire_spy = MagicMock()
        monkeypatch.setattr(
            "litellm.proxy.utils.ProxyLogging._fire_deferred_stream_logging",
            fire_spy,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=True)
        mock_response = MagicMock()
        mock_response.aclose = AsyncMock()
        request_data = {
            "metadata": {},
            "litellm_params": {"metadata": {}},
            "litellm_logging_obj": MagicMock(model_call_details={"metadata": {}, "litellm_params": {}}),
        }

        await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
            request=mock_request,
            request_data=request_data,
            response=mock_response,
        )

        fire_spy.assert_called_once_with(request_data)
        mock_response.aclose.assert_awaited_once()
        assert request_data["metadata"]["error_information"]["error_code"] == "499"

    @pytest.mark.asyncio
    async def test_finalize_streaming_generator_cleanup_skips_disconnect_after_completion(
        self, monkeypatch
    ):
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        fire_spy = MagicMock()
        monkeypatch.setattr(
            "litellm.proxy.utils.ProxyLogging._fire_deferred_stream_logging",
            fire_spy,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=True)
        mock_response = MagicMock()
        mock_response.aclose = AsyncMock()
        request_data = {"metadata": {}, "litellm_params": {"metadata": {}}}

        await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
            request=mock_request,
            request_data=request_data,
            response=mock_response,
            stream_completed=True,
        )

        fire_spy.assert_not_called()
        mock_request.is_disconnected.assert_not_awaited()
        mock_response.aclose.assert_awaited_once()
        assert "client_disconnected" not in request_data["metadata"]

    @pytest.mark.asyncio
    async def test_async_streaming_data_generator_records_499_on_early_aclose(
        self, monkeypatch
    ):
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        monkeypatch.setattr(
            "litellm.proxy.utils.ProxyLogging._fire_deferred_stream_logging",
            MagicMock(),
        )

        async def mock_streaming_iterator(*_args, **_kwargs):
            yield {"choices": [{"delta": {"content": "hi"}}]}
            yield {"choices": [{"delta": {"content": " there"}}]}

        mock_proxy_logging = MagicMock(spec=ProxyLogging)
        mock_proxy_logging.async_post_call_streaming_iterator_hook = (
            mock_streaming_iterator
        )
        ProxyLogging._callback_capabilities_cache.clear()

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=True)
        mock_response = MagicMock()
        mock_response.aclose = AsyncMock()
        request_data = {
            "model": "gemini-2.0-flash",
            "metadata": {},
            "litellm_params": {"metadata": {}},
            "litellm_logging_obj": MagicMock(
                model_call_details={"metadata": {}, "litellm_params": {}}
            ),
        }

        gen = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=mock_response,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            request_data=request_data,
            proxy_logging_obj=mock_proxy_logging,
            serialize_chunk=lambda chunk: f"data: {chunk}\n\n",
            serialize_error=lambda proxy_exc: f"data: {proxy_exc.to_dict()}\n\n",
            request=mock_request,
        )
        await gen.__anext__()
        await gen.aclose()

        assert request_data["metadata"]["client_disconnected"] is True
        assert request_data["metadata"]["error_information"]["error_code"] == "499"

        ProxyLogging._callback_capabilities_cache.clear()
class TestCancelOnDisconnect:
    """
    Coverage for the opt-in `general_settings.cancel_on_disconnect` flag:
    cancelling the in-flight upstream LLM call when the HTTP client disconnects
    (issue #13774), without changing the default code path and without skipping
    failure accounting (post_call_failure_hook) on the resulting 499.
    """

    def _request(self, messages: list) -> Request:
        async def receive():
            if messages:
                return messages.pop(0)
            await asyncio.Event().wait()

        return Request(scope={"type": "http", "headers": []}, receive=receive)

    async def test_monitor_cancels_llm_call_and_sets_event_on_disconnect(self):
        request = self._request(
            [
                {"type": "http.request", "body": b"", "more_body": False},
                {"type": "http.disconnect"},
            ]
        )
        llm_call = asyncio.get_running_loop().create_future()
        disconnect_event = asyncio.Event()

        await _cancel_llm_call_on_client_disconnect(
            request, llm_call, disconnect_event
        )

        assert llm_call.cancelled()
        assert disconnect_event.is_set()

    async def test_monitor_is_noop_while_client_stays_connected(self):
        request = self._request(
            [{"type": "http.request", "body": b"", "more_body": False}]
        )
        llm_call = asyncio.get_running_loop().create_future()
        disconnect_event = asyncio.Event()

        monitor = asyncio.create_task(
            _cancel_llm_call_on_client_disconnect(request, llm_call, disconnect_event)
        )
        await asyncio.sleep(0.01)

        assert not monitor.done()
        assert not llm_call.cancelled()
        assert not disconnect_event.is_set()
        monitor.cancel()

    async def test_monitor_survives_receive_failure_without_cancelling(self):
        """If request.receive() fails (e.g. transport reset) the watcher must
        degrade to a no-op instead of crashing or cancelling the LLM call."""

        async def receive():
            raise RuntimeError("transport reset")

        request = Request(scope={"type": "http", "headers": []}, receive=receive)
        llm_call = asyncio.get_running_loop().create_future()
        disconnect_event = asyncio.Event()

        await _cancel_llm_call_on_client_disconnect(
            request, llm_call, disconnect_event
        )

        assert not llm_call.cancelled()
        assert not disconnect_event.is_set()

    async def test_cancellation_without_disconnect_reraises_cancelled_error(self):
        """A CancelledError that is NOT client-initiated (e.g. server shutdown)
        must propagate as-is instead of being masked as a 499."""
        request = self._request([])
        llm_call = asyncio.get_running_loop().create_future()
        llm_call.cancel()

        with pytest.raises(asyncio.CancelledError):
            await _await_llm_call_cancelling_on_disconnect(request, llm_call)

    async def _drive_base_process_llm_request(
        self, monkeypatch, general_settings: dict, llm_call, request: Request
    ):
        from litellm.proxy._types import UserAPIKeyAuth

        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "test-cancel-on-disconnect"
        logging_obj._defer_async_logging = False
        logging_obj._on_deferred_stream_complete = None
        logging_obj.cost_breakdown = None

        processor = ProxyBaseLLMRequestProcessing(
            data={"model": "fake-model", "litellm_logging_obj": logging_obj}
        )

        proxy_logging_obj = MagicMock(spec=ProxyLogging)
        proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
        proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_success_hook = AsyncMock(
            side_effect=lambda data, user_api_key_dict, response: response
        )
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(
            return_value=None
        )

        async def fake_route_request(**kwargs):
            return llm_call()

        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "route_request",
            fake_route_request,
        )

        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
            route_type="acompletion",
            proxy_logging_obj=proxy_logging_obj,
            general_settings=general_settings,
            proxy_config=MagicMock(spec=ProxyConfig),
            skip_pre_call_logic=True,
        )

    async def test_disconnect_ignored_when_flag_disabled(self, monkeypatch):
        upstream_cancelled = asyncio.Event()
        model_response = litellm.ModelResponse()

        async def llm_call():
            try:
                await asyncio.sleep(0.05)
                return model_response
            except asyncio.CancelledError:
                upstream_cancelled.set()
                raise

        result = await self._drive_base_process_llm_request(
            monkeypatch,
            general_settings={},
            llm_call=llm_call,
            request=self._request([{"type": "http.disconnect"}]),
        )

        assert result is model_response
        assert not upstream_cancelled.is_set()

    async def test_disconnect_cancels_upstream_when_flag_enabled(self, monkeypatch):
        upstream_cancelled = asyncio.Event()

        async def llm_call():
            try:
                await asyncio.sleep(5)
                return litellm.ModelResponse()
            except asyncio.CancelledError:
                upstream_cancelled.set()
                raise

        with pytest.raises(HTTPException) as exc_info:
            await self._drive_base_process_llm_request(
                monkeypatch,
                general_settings={"cancel_on_disconnect": True},
                llm_call=llm_call,
                request=self._request([{"type": "http.disconnect"}]),
            )

        assert exc_info.value.status_code == 499
        assert upstream_cancelled.is_set()

    async def test_499_still_fires_post_call_failure_hook(self):
        """Regression guard: the 499 path must NOT bypass post_call_failure_hook,
        which releases max_parallel_requests slots and fires spend/alerting
        callbacks (cf. #14457; P1 review finding on #25776/#27146)."""
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth

        processor = ProxyBaseLLMRequestProcessing(data={})
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})

        with pytest.raises(ProxyException) as exc_info:
            await processor._handle_llm_api_exception(
                e=HTTPException(
                    status_code=499, detail="Client disconnected the request"
                ),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
                proxy_logging_obj=proxy_logging_obj,
            )

        assert exc_info.value.code == "499"
        proxy_logging_obj.post_call_failure_hook.assert_awaited_once()


class TestAllmPassthroughRoutePostCallGuardrails:
    """
    Regression: non-streaming allm_passthrough_route responses are httpx.Response objects.
    The generic post_call_success_hook path passes them as-is, but our Bedrock guardrail
    handler short-circuits on non-dict inputs.  The fix buffers JSON responses before the
    hook so guardrails receive a dict (and output_parse_pii de-anonymisation works).
    """

    def _make_guardrail_cb(self, name: str = "presidio-pre-guard") -> MagicMock:
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        cb = MagicMock(spec=CustomGuardrail)
        cb.guardrail_name = name
        cb.event_hook = [GuardrailEventHooks.pre_call.value, GuardrailEventHooks.post_call.value]
        cb._event_hook_is_event_type = lambda et: et.value in cb.event_hook
        cb.should_run_guardrail = MagicMock(return_value=True)
        return cb

    @pytest.mark.asyncio
    async def test_post_call_hook_receives_parsed_dict_not_httpx_response(self, monkeypatch):
        """
        post_call_success_hook must be called with the parsed JSON dict when the
        non-streaming allm_passthrough_route response is application/json.
        """
        import json

        bedrock_response_body = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello, <PERSON_1>!"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 8},
        }

        httpx_response = httpx.Response(
            status_code=200,
            content=json.dumps(bedrock_response_body).encode(),
            headers={"content-type": "application/json"},
        )

        received_responses = []

        async def capture_hook(data, user_api_key_dict, response):
            received_responses.append(response)
            return response

        cb = self._make_guardrail_cb()
        monkeypatch.setattr(litellm, "callbacks", [cb])
        ProxyLogging._callback_capabilities_cache.clear()

        proxy_logging_obj = ProxyLogging(user_api_key_cache=MagicMock())
        monkeypatch.setattr(proxy_logging_obj, "post_call_success_hook", capture_hook)

        with patch.object(ProxyBaseLLMRequestProcessing, "_has_post_call_guardrails_for_passthrough", return_value=True):
            processing_obj = ProxyBaseLLMRequestProcessing(data={})
            result = await processing_obj._handle_non_streaming_allm_passthrough_route(
                response=httpx_response,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                custom_headers={},
                request_headers={},
            )

        assert len(received_responses) == 1
        assert isinstance(received_responses[0], dict), (
            "post_call_success_hook must receive parsed dict, not httpx.Response"
        )
        assert received_responses[0]["stopReason"] == "end_turn"
        assert isinstance(result, Response)
        body = json.loads(result.body)
        assert body["stopReason"] == "end_turn"

        ProxyLogging._callback_capabilities_cache.clear()

    @pytest.mark.asyncio
    async def test_non_dict_hook_return_falls_back_to_original_body(self, monkeypatch):
        """
        When post_call_success_hook returns a non-dict (e.g. a non-serializable
        object), the JSON branch must return the original body bytes unchanged
        rather than raising a TypeError from json.dumps.
        """
        import json

        original = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
        }
        httpx_response = httpx.Response(
            status_code=200,
            content=json.dumps(original).encode(),
            headers={"content-type": "application/json"},
        )

        async def non_dict_hook(data, user_api_key_dict, response):
            return object()

        cb = self._make_guardrail_cb()
        monkeypatch.setattr(litellm, "callbacks", [cb])
        ProxyLogging._callback_capabilities_cache.clear()

        proxy_logging_obj = ProxyLogging(user_api_key_cache=MagicMock())
        monkeypatch.setattr(proxy_logging_obj, "post_call_success_hook", non_dict_hook)

        with patch.object(ProxyBaseLLMRequestProcessing, "_has_post_call_guardrails_for_passthrough", return_value=True):
            processing_obj = ProxyBaseLLMRequestProcessing(data={})
            result = await processing_obj._handle_non_streaming_allm_passthrough_route(
                response=httpx_response,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                custom_headers={},
                request_headers={},
            )

        assert isinstance(result, Response)
        assert json.loads(result.body) == original

        ProxyLogging._callback_capabilities_cache.clear()

    @pytest.mark.asyncio
    async def test_malformed_json_body_passes_through_without_500(self, monkeypatch):
        """
        A 2xx response advertising application/json but carrying a non-JSON body
        must pass the original bytes through unchanged instead of raising
        JSONDecodeError (which would surface as a 500). The post-call hook is
        never invoked since there is no dict to guardrail.
        """
        malformed_body = b"not-json-at-all"
        httpx_response = httpx.Response(
            status_code=200,
            content=malformed_body,
            headers={"content-type": "application/json"},
        )

        cb = self._make_guardrail_cb()
        monkeypatch.setattr(litellm, "callbacks", [cb])
        ProxyLogging._callback_capabilities_cache.clear()

        proxy_logging_obj = ProxyLogging(user_api_key_cache=MagicMock())
        hook_spy = AsyncMock()
        monkeypatch.setattr(proxy_logging_obj, "post_call_success_hook", hook_spy)

        with patch.object(ProxyBaseLLMRequestProcessing, "_has_post_call_guardrails_for_passthrough", return_value=True):
            processing_obj = ProxyBaseLLMRequestProcessing(data={})
            result = await processing_obj._handle_non_streaming_allm_passthrough_route(
                response=httpx_response,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                custom_headers={},
                request_headers={},
            )

        hook_spy.assert_not_awaited()
        assert isinstance(result, Response)
        assert result.status_code == 200
        assert result.body == malformed_body

        ProxyLogging._callback_capabilities_cache.clear()

    @pytest.mark.asyncio
    async def test_no_aread_when_no_post_call_guardrails(self, monkeypatch):
        """
        When _has_post_call_guardrails_for_passthrough() is False the httpx
        response must not be read — the caller handles streaming or error paths
        normally.
        """
        import json

        httpx_response = httpx.Response(
            status_code=200,
            content=json.dumps({"output": "x"}).encode(),
            headers={"content-type": "application/json"},
        )
        spy_read = AsyncMock(wraps=httpx_response.aread)
        httpx_response.aread = spy_read

        monkeypatch.setattr(litellm, "callbacks", [])
        ProxyLogging._callback_capabilities_cache.clear()

        proxy_logging_obj = ProxyLogging(user_api_key_cache=MagicMock())
        hook_spy = AsyncMock()
        monkeypatch.setattr(proxy_logging_obj, "post_call_success_hook", hook_spy)

        with patch.object(ProxyBaseLLMRequestProcessing, "_has_post_call_guardrails_for_passthrough", return_value=False):
            processing_obj = ProxyBaseLLMRequestProcessing(data={})
            result = await processing_obj._handle_non_streaming_allm_passthrough_route(
                response=httpx_response,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                custom_headers={},
                request_headers={},
            )

        spy_read.assert_not_called()
        hook_spy.assert_not_called()
        assert result is None

        ProxyLogging._callback_capabilities_cache.clear()


def _build_event_stream_frame(event_type: str, payload: dict) -> bytes:
    import json
    import struct
    from botocore.eventstream import crc32 as esm_crc32

    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()

    def _encode_str_header(name: str, value: str) -> bytes:
        name_b = name.encode()
        value_b = value.encode()
        return (
            struct.pack("!B", len(name_b))
            + name_b
            + struct.pack("!B", 7)  # type 7 = string
            + struct.pack("!H", len(value_b))
            + value_b
        )

    headers_bytes = (
        _encode_str_header(":event-type", event_type)
        + _encode_str_header(":content-type", "application/json")
        + _encode_str_header(":message-type", "event")
    )

    headers_length = len(headers_bytes)
    total_length = 12 + headers_length + len(payload_bytes) + 4
    prelude = struct.pack("!II", total_length, headers_length)
    prelude_crc_val = esm_crc32(prelude) & 0xFFFFFFFF
    prelude_crc_b = struct.pack("!I", prelude_crc_val)
    part_for_msg = prelude_crc_b + headers_bytes + payload_bytes
    msg_crc_val = esm_crc32(part_for_msg, prelude_crc_val) & 0xFFFFFFFF
    msg_crc_b = struct.pack("!I", msg_crc_val)
    return prelude + prelude_crc_b + headers_bytes + payload_bytes + msg_crc_b


class TestEventStreamAllmPassthroughRoute:
    @pytest.mark.asyncio
    async def test_bedrock_provider_dispatches_to_handler(self):
        stream_bytes = _build_event_stream_frame("messageStart", {"role": "assistant"})
        expected_bytes = _build_event_stream_frame("messageStart", {"role": "assistant"}) + b"extra"

        proxy_logging_obj = MagicMock()
        user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        with patch(
            "litellm.llms.bedrock.passthrough.guardrail_translation.handler.BedrockPassthroughGuardrailHandler.de_anonymize_event_stream",
            new=AsyncMock(return_value=expected_bytes),
        ) as mock_handler:
            processing_obj = ProxyBaseLLMRequestProcessing(data={"custom_llm_provider": "bedrock"})
            result = await processing_obj._handle_event_stream_allm_passthrough_route(
                body_bytes=stream_bytes,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
            )

        mock_handler.assert_awaited_once()
        assert result == expected_bytes

    @pytest.mark.asyncio
    async def test_non_bedrock_provider_returns_original_bytes(self):
        stream_bytes = _build_event_stream_frame("messageStart", {"role": "assistant"})
        proxy_logging_obj = MagicMock()

        processing_obj = ProxyBaseLLMRequestProcessing(data={"custom_llm_provider": "anthropic"})
        result = await processing_obj._handle_event_stream_allm_passthrough_route(
            body_bytes=stream_bytes,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
        )

        assert result is stream_bytes

    @pytest.mark.asyncio
    async def test_non_streaming_response_includes_custom_headers(self):
        import json

        body = {"output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}}}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json", "content-length": "99"}
        mock_response.aread = AsyncMock(return_value=json.dumps(body).encode())

        async def mock_hook(data, user_api_key_dict, response):
            return response

        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_success_hook = mock_hook
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})

        custom_headers = {
            "x-litellm-call-id": "test-call-123",
            "x-litellm-model-id": "bedrock/claude",
            "content-length": "99",
        }

        with patch.object(ProxyBaseLLMRequestProcessing, "_has_post_call_guardrails_for_passthrough", return_value=True):
            processing_obj = ProxyBaseLLMRequestProcessing(data={})
            result = await processing_obj._handle_non_streaming_allm_passthrough_route(
                response=mock_response,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                custom_headers=custom_headers,
                request_headers={},
            )

        assert result is not None
        assert result.headers.get("x-litellm-call-id") == "test-call-123"
        assert result.headers.get("x-litellm-model-id") == "bedrock/claude"
        # content-length from custom_headers is filtered; Starlette sets the correct value from body
        assert result.headers.get("content-length") != "99"


class TestAllmPassthroughStreamingProviderGate:
    """
    Regression: the streaming-buffer gate for allm_passthrough_route must only
    fire for provider+endpoint pairs that have an event-stream guardrail handler
    able to rewrite frames (Bedrock converse-stream).

    A non-Bedrock streaming passthrough response must keep streaming even when a
    post-call guardrail is registered globally, instead of being silently
    buffered into a non-streaming Response. A Bedrock endpoint the Converse
    handler cannot rewrite (e.g. invoke-with-response-stream) must also keep
    streaming. Only converse-stream is buffered so its frames can be
    de-anonymized.
    """

    def _build_processing_obj(
        self, custom_llm_provider: str, endpoint: str = ""
    ) -> ProxyBaseLLMRequestProcessing:
        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "call-123"
        logging_obj.cost_breakdown = None
        data = {
            "custom_llm_provider": custom_llm_provider,
            "endpoint": endpoint,
            "litellm_logging_obj": logging_obj,
        }
        return ProxyBaseLLMRequestProcessing(data=data)

    async def _run(self, processing_obj, monkeypatch, chunks):
        import litellm.proxy.common_request_processing as crp
        from litellm.proxy._types import UserAPIKeyAuth as RealUserAPIKeyAuth

        async def streaming_response():
            for chunk in chunks:
                yield chunk

        async def fake_route_request(**kwargs):
            async def _llm_call():
                return streaming_response()

            return _llm_call()

        monkeypatch.setattr(crp, "route_request", fake_route_request)

        proxy_logging_obj = MagicMock(spec=ProxyLogging)
        proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
        proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_success_hook = AsyncMock()

        return await processing_obj.base_process_llm_request(
            request=MagicMock(spec=Request, headers={}),
            fastapi_response=Response(),
            user_api_key_dict=RealUserAPIKeyAuth(api_key="sk-test"),
            route_type="allm_passthrough_route",
            proxy_logging_obj=proxy_logging_obj,
            general_settings={},
            proxy_config=MagicMock(spec=ProxyConfig),
            select_data_generator=None,
            llm_router=None,
            skip_pre_call_logic=True,
        )

    @pytest.mark.asyncio
    async def test_non_bedrock_stream_is_not_buffered(self, monkeypatch):
        processing_obj = self._build_processing_obj("anthropic")
        chunks = [b"chunk-1", b"chunk-2"]

        with patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails",
            return_value=False,
        ), patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails_for_passthrough",
            return_value=True,
        ):
            result = await self._run(processing_obj, monkeypatch, chunks)

        assert isinstance(result, StreamingResponse)
        streamed = [chunk async for chunk in result.body_iterator]
        assert streamed == chunks

    @pytest.mark.asyncio
    async def test_bedrock_converse_stream_is_buffered_through_handler(
        self, monkeypatch
    ):
        processing_obj = self._build_processing_obj(
            "bedrock", "model/us.amazon.nova-lite-v1:0/converse-stream"
        )
        chunks = [b"raw-1", b"raw-2"]

        with patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails",
            return_value=False,
        ), patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails_for_passthrough",
            return_value=True,
        ), patch(
            "litellm.llms.bedrock.passthrough.guardrail_translation.handler."
            "BedrockPassthroughGuardrailHandler.de_anonymize_event_stream",
            new=AsyncMock(return_value=b"modified-body"),
        ) as mock_handler:
            result = await self._run(processing_obj, monkeypatch, chunks)

        assert isinstance(result, Response)
        assert not isinstance(result, StreamingResponse)
        assert result.body == b"modified-body"
        assert result.headers["content-type"] == "application/vnd.amazon.eventstream"
        mock_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bedrock_invoke_stream_is_not_buffered(self, monkeypatch):
        processing_obj = self._build_processing_obj(
            "bedrock", "model/us.amazon.nova-lite-v1:0/invoke-with-response-stream"
        )
        chunks = [b"raw-1", b"raw-2"]

        with patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails",
            return_value=False,
        ), patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails_for_passthrough",
            return_value=True,
        ), patch(
            "litellm.llms.bedrock.passthrough.guardrail_translation.handler."
            "BedrockPassthroughGuardrailHandler.de_anonymize_event_stream",
            new=AsyncMock(return_value=b"modified-body"),
        ) as mock_handler:
            result = await self._run(processing_obj, monkeypatch, chunks)

        assert isinstance(result, StreamingResponse)
        streamed = [chunk async for chunk in result.body_iterator]
        assert streamed == chunks
        mock_handler.assert_not_awaited()


class TestResponseCostHeaderForTypedDictResponses:
    """
    Regression for LIT-4076. x-litellm-response-cost went missing on Anthropic
    /v1/messages and Google :generateContent even though it appeared on
    /chat/completions and /responses. /v1/messages returns a TypedDict that cannot
    hold _hidden_params at all, and :generateContent carries _hidden_params but no
    synchronously-populated response_cost. In both cases the raw response_cost is
    empty at header-build time. The non-streaming header build now recovers the cost
    from the logging object whenever the response itself never recorded one, while
    leaving object responses (ModelResponse etc.) untouched.
    """

    def _build_logging_obj(self, *, model_call_details, response_cost_calculator):
        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "call-lit4076"
        logging_obj.cost_breakdown = None
        logging_obj.model_call_details = model_call_details
        logging_obj._response_cost_calculator = response_cost_calculator
        logging_obj._enqueue_deferred_logging = None
        logging_obj._on_deferred_stream_complete = None
        return logging_obj

    async def _drive_non_streaming(self, *, monkeypatch, response, logging_obj, route_type, return_result=False):
        import litellm.proxy.common_request_processing as crp
        from litellm.proxy._types import UserAPIKeyAuth as RealUserAPIKeyAuth

        async def fake_route_request(**kwargs):
            async def _llm_call():
                return response

            return _llm_call()

        monkeypatch.setattr(crp, "route_request", fake_route_request)

        async def fake_post_call_success_hook(data, user_api_key_dict, response):
            return response

        proxy_logging_obj = MagicMock(spec=ProxyLogging)
        proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
        proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value={})
        proxy_logging_obj.post_call_success_hook = fake_post_call_success_hook

        fastapi_response = Response()
        processing_obj = ProxyBaseLLMRequestProcessing(data={"litellm_logging_obj": logging_obj})

        with patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails",
            return_value=False,
        ):
            result = await processing_obj.base_process_llm_request(
                request=MagicMock(spec=Request, headers={}),
                fastapi_response=fastapi_response,
                user_api_key_dict=RealUserAPIKeyAuth(api_key="sk-test"),
                route_type=route_type,
                proxy_logging_obj=proxy_logging_obj,
                general_settings={},
                proxy_config=MagicMock(spec=ProxyConfig),
                select_data_generator=None,
                llm_router=None,
                skip_pre_call_logic=True,
            )
        if return_result:
            return fastapi_response, result
        return fastapi_response

    @pytest.mark.asyncio
    async def test_messages_typeddict_emits_cost_header_from_stored_cost(self, monkeypatch):
        from litellm.types.utils import AnthropicMessagesResponse

        response = AnthropicMessagesResponse(
            id="msg_1",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "hi"}],
            model="claude-haiku-4-5",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        recompute = MagicMock(return_value=999.0)
        logging_obj = self._build_logging_obj(
            model_call_details={"response_cost": 0.00123},
            response_cost_calculator=recompute,
        )

        fastapi_response = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="anthropic_messages",
        )

        assert fastapi_response.headers["x-litellm-response-cost"] == "0.00123"
        recompute.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_content_typeddict_emits_cost_header_via_recompute(self, monkeypatch):
        from litellm.types.llms.vertex_ai import GenerateContentResponseBody

        response = GenerateContentResponseBody(
            candidates=[{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
            usageMetadata={
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        )
        recompute = MagicMock(return_value=0.00456)
        logging_obj = self._build_logging_obj(
            model_call_details={},
            response_cost_calculator=recompute,
        )

        fastapi_response = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="agenerate_content",
        )

        assert fastapi_response.headers["x-litellm-response-cost"] == "0.00456"
        recompute.assert_called_once()
        assert recompute.call_args.kwargs["result"] is response

    @pytest.mark.asyncio
    async def test_generate_content_emits_real_nonzero_cost_header_from_usage_metadata(self, monkeypatch):
        """
        End-to-end regression for LIT-4076 using the real cost calculator (not a
        mock). A native :generateContent body reports tokens under usageMetadata,
        which the cost calculator did not read, so the synchronously-recovered
        cost was 0.0 and the header was dropped even though the async logging path
        billed a real non-zero amount. The header must now carry the true cost.
        """
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
        from litellm.types.llms.vertex_ai import GenerateContentResponseBody
        from litellm.types.utils import ModelResponse, Usage

        response = GenerateContentResponseBody(
            candidates=[{"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}],
            usageMetadata={
                "promptTokenCount": 1000,
                "candidatesTokenCount": 500,
                "totalTokenCount": 1500,
            },
        )

        real_logging = LiteLLMLoggingObj(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="agenerate_content",
            start_time=None,
            litellm_call_id="call-lit4076-real",
            function_id="fn",
        )
        real_logging.model_call_details["custom_llm_provider"] = "gemini"
        real_logging.optional_params = {}

        logging_obj = self._build_logging_obj(
            model_call_details={},
            response_cost_calculator=real_logging._response_cost_calculator,
        )

        fastapi_response = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="agenerate_content",
        )

        expected_cost = litellm.completion_cost(
            completion_response=ModelResponse(
                model="gemini-2.5-flash",
                usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            ),
            model="gemini-2.5-flash",
            custom_llm_provider="gemini",
        )
        assert expected_cost > 0
        assert float(fastapi_response.headers["x-litellm-response-cost"]) == pytest.approx(expected_cost)

    @pytest.mark.asyncio
    async def test_generate_content_with_hidden_params_emits_cost_header(self, monkeypatch):
        """
        Models the real :generateContent response: it DOES carry a _hidden_params
        attribute (which is why x-litellm-model-group / x-litellm-model-api-base
        appear), but no response_cost is populated synchronously at header-build
        time. The cost is only available on the logging object. The previous
        ``not hasattr(response, "_hidden_params")`` guard skipped recovery here, so
        x-litellm-response-cost went missing even though the cost was computed.
        """
        from types import SimpleNamespace

        response = SimpleNamespace(
            _hidden_params={
                "additional_headers": {"x-litellm-model-group": "gemini-2.5-flash"},
            }
        )
        recompute = MagicMock(return_value=999.0)
        logging_obj = self._build_logging_obj(
            model_call_details={"response_cost": 0.0004521},
            response_cost_calculator=recompute,
        )

        fastapi_response = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="agenerate_content",
        )

        assert fastapi_response.headers["x-litellm-response-cost"] == "0.0004521"
        assert fastapi_response.headers["x-litellm-model-group"] == "gemini-2.5-flash"
        recompute.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_content_with_hidden_params_zero_cost_drops_header(self, monkeypatch):
        """
        A recovered cost of 0 must normalize to a dropped header, exactly like
        /chat/completions, so :generateContent does not start emitting
        x-litellm-response-cost: 0.0 where nothing was emitted before.
        """
        from types import SimpleNamespace

        response = SimpleNamespace(
            _hidden_params={
                "additional_headers": {"x-litellm-model-group": "gemini-2.5-flash"},
            }
        )
        recompute = MagicMock(return_value=999.0)
        logging_obj = self._build_logging_obj(
            model_call_details={"response_cost": 0.0},
            response_cost_calculator=recompute,
        )

        fastapi_response = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="agenerate_content",
        )

        assert "x-litellm-response-cost" not in fastapi_response.headers
        recompute.assert_not_called()

    @pytest.mark.asyncio
    async def test_object_response_with_hidden_params_is_unaffected(self, monkeypatch):
        from types import SimpleNamespace

        response = SimpleNamespace(_hidden_params={"response_cost": 0.009})
        recompute = MagicMock(side_effect=AssertionError("must not recompute for object responses"))
        logging_obj = self._build_logging_obj(
            model_call_details={"response_cost": 123.0},
            response_cost_calculator=recompute,
        )

        fastapi_response = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="acompletion",
        )

        assert fastapi_response.headers["x-litellm-response-cost"] == "0.009"
        recompute.assert_not_called()

    @pytest.mark.asyncio
    async def test_object_response_zero_cost_drops_header_like_chat_completions(self, monkeypatch):
        from types import SimpleNamespace

        response = SimpleNamespace(_hidden_params={"response_cost": 0.0})
        recompute = MagicMock(side_effect=AssertionError("must not recompute for object responses"))
        logging_obj = self._build_logging_obj(
            model_call_details={"response_cost": 0.00789},
            response_cost_calculator=recompute,
        )

        fastapi_response = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="acompletion",
        )

        assert "x-litellm-response-cost" not in fastapi_response.headers
        recompute.assert_not_called()

    @pytest.mark.asyncio
    async def test_messages_typeddict_does_not_leak_hidden_params_into_response_body(self, monkeypatch):
        """
        Router.set_response_headers now writes rate-limit headers onto dict-shaped
        responses (e.g. Anthropic /v1/messages, whose AnthropicMessagesResponse is a
        TypedDict) via response["_hidden_params"] = ... . Unlike a pydantic model's
        private attribute, that key is indistinguishable from any other dict key and
        would otherwise serialize verbatim into the client-facing JSON body, leaking
        response_cost/model_id/api_base/fallback errors. base_process_llm_request
        must strip it before returning the response to the endpoint layer.
        """
        from litellm.types.utils import AnthropicMessagesResponse

        response = AnthropicMessagesResponse(
            id="msg_1",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "hi"}],
            model="claude-haiku-4-5",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        response["_hidden_params"] = {
            "additional_headers": {"x-ratelimit-limit-input-tokens": "25"},
            "response_cost": 0.00123,
            "model_id": "internal-deployment-id",
        }
        logging_obj = self._build_logging_obj(
            model_call_details={"response_cost": 0.00123},
            response_cost_calculator=MagicMock(return_value=999.0),
        )

        fastapi_response, result = await self._drive_non_streaming(
            monkeypatch=monkeypatch,
            response=response,
            logging_obj=logging_obj,
            route_type="anthropic_messages",
            return_result=True,
        )

        assert "_hidden_params" not in result
        assert fastapi_response.headers["x-ratelimit-limit-input-tokens"] == "25"
        assert fastapi_response.headers["x-litellm-response-cost"] == "0.00123"


class TestPreCallWithFallbacksOnLocalRateLimit:

    @pytest.mark.asyncio
    async def test_fallback_triggered_on_local_rate_limit(self):
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        primary_model = "gpt-4"
        fallback_model = "gpt-3.5-turbo"

        processor = ProxyBaseLLMRequestProcessing(data={"model": primary_model})

        call_count = 0

        async def mock_pre_call_logic(**kwargs):
            nonlocal call_count
            call_count += 1
            model_in_data = processor.data.get("model")
            if model_in_data == primary_model:
                raise ProxyRateLimitError(
                    detail="TPM limit exceeded for gpt-4",
                    headers={"retry-after": "30"},
                )
            logging_obj = MagicMock()
            return processor.data, logging_obj

        mock_router = MagicMock()
        mock_router.fallbacks = [{"gpt-4": ["gpt-3.5-turbo"]}]

        with patch.object(
            processor,
            "common_processing_pre_call_logic",
            side_effect=mock_pre_call_logic,
        ):
            data, logging_obj = await processor._pre_call_with_fallbacks(
                request=MagicMock(),
                general_settings={},
                proxy_logging_obj=MagicMock(),
                user_api_key_dict=MagicMock(router_settings=None),
                version=None,
                proxy_config=MagicMock(),
                user_model=None,
                user_temperature=None,
                user_request_timeout=None,
                user_max_tokens=None,
                user_api_base=None,
                model=primary_model,
                route_type="acompletion",
                llm_router=mock_router,
            )

        assert processor.data["model"] == fallback_model
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_no_fallbacks_configured(self):
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-4"})

        async def mock_pre_call_logic(**kwargs):
            raise ProxyRateLimitError(
                detail="TPM limit exceeded",
                headers={"retry-after": "30"},
            )

        mock_router = MagicMock()
        mock_router.fallbacks = None

        with patch.object(
            processor,
            "common_processing_pre_call_logic",
            side_effect=mock_pre_call_logic,
        ):
            with pytest.raises(ProxyRateLimitError):
                await processor._pre_call_with_fallbacks(
                    request=MagicMock(),
                    general_settings={},
                    proxy_logging_obj=MagicMock(),
                    user_api_key_dict=MagicMock(router_settings=None),
                    version=None,
                    proxy_config=MagicMock(),
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    model="gpt-4",
                    route_type="acompletion",
                    llm_router=mock_router,
                )

    @pytest.mark.asyncio
    async def test_raises_when_all_fallbacks_also_rate_limited(self):
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-4"})

        async def mock_pre_call_logic(**kwargs):
            raise ProxyRateLimitError(
                detail=f"TPM limit exceeded for {processor.data.get('model')}",
                headers={"retry-after": "30"},
            )

        mock_router = MagicMock()
        mock_router.fallbacks = [{"gpt-4": ["gpt-3.5-turbo", "claude-3-haiku"]}]

        with patch.object(
            processor,
            "common_processing_pre_call_logic",
            side_effect=mock_pre_call_logic,
        ):
            with pytest.raises(ProxyRateLimitError, match="gpt-4"):
                await processor._pre_call_with_fallbacks(
                    request=MagicMock(),
                    general_settings={},
                    proxy_logging_obj=MagicMock(),
                    user_api_key_dict=MagicMock(router_settings=None),
                    version=None,
                    proxy_config=MagicMock(),
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    model="gpt-4",
                    route_type="acompletion",
                    llm_router=mock_router,
                )

        assert processor.data["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_fallback_uses_key_level_router_settings(self):
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-4"})

        async def mock_pre_call_logic(**kwargs):
            if processor.data.get("model") == "gpt-4":
                raise ProxyRateLimitError(
                    detail="TPM limit exceeded",
                    headers={"retry-after": "30"},
                )
            return processor.data, MagicMock()

        mock_router = MagicMock()
        mock_router.fallbacks = [{"gpt-4": ["gpt-3.5-turbo"]}]

        user_api_key_dict = MagicMock()
        user_api_key_dict.router_settings = {
            "fallbacks": [{"gpt-4": ["claude-3-haiku"]}]
        }

        with patch.object(
            processor,
            "common_processing_pre_call_logic",
            side_effect=mock_pre_call_logic,
        ):
            data, _ = await processor._pre_call_with_fallbacks(
                request=MagicMock(),
                general_settings={},
                proxy_logging_obj=MagicMock(),
                user_api_key_dict=user_api_key_dict,
                version=None,
                proxy_config=MagicMock(),
                user_model=None,
                user_temperature=None,
                user_request_timeout=None,
                user_max_tokens=None,
                user_api_base=None,
                model="gpt-4",
                route_type="acompletion",
                llm_router=mock_router,
            )

        assert processor.data["model"] == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_disable_fallbacks_flag_respected(self):
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        processor = ProxyBaseLLMRequestProcessing(
            data={"model": "gpt-4", "disable_fallbacks": True}
        )

        async def mock_pre_call_logic(**kwargs):
            raise ProxyRateLimitError(
                detail="TPM limit exceeded",
                headers={"retry-after": "30"},
            )

        mock_router = MagicMock()
        mock_router.fallbacks = [{"gpt-4": ["gpt-3.5-turbo"]}]

        with patch.object(
            processor,
            "common_processing_pre_call_logic",
            side_effect=mock_pre_call_logic,
        ):
            with pytest.raises(ProxyRateLimitError):
                await processor._pre_call_with_fallbacks(
                    request=MagicMock(),
                    general_settings={},
                    proxy_logging_obj=MagicMock(),
                    user_api_key_dict=MagicMock(router_settings=None),
                    version=None,
                    proxy_config=MagicMock(),
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    model="gpt-4",
                    route_type="acompletion",
                    llm_router=mock_router,
                )

    @pytest.mark.asyncio
    async def test_model_restored_on_non_rate_limit_exception(self):
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        primary_model = "gpt-4"

        processor = ProxyBaseLLMRequestProcessing(data={"model": primary_model})

        async def mock_pre_call_logic(**kwargs):
            model_in_data = processor.data.get("model")
            if model_in_data == primary_model:
                raise ProxyRateLimitError(
                    detail="TPM limit exceeded for gpt-4",
                    headers={"retry-after": "30"},
                )
            raise ValueError("unexpected auth failure on fallback")

        mock_router = MagicMock()
        mock_router.fallbacks = [{"gpt-4": ["gpt-3.5-turbo"]}]

        with patch.object(
            processor,
            "common_processing_pre_call_logic",
            side_effect=mock_pre_call_logic,
        ):
            with pytest.raises(ValueError, match="unexpected auth failure"):
                await processor._pre_call_with_fallbacks(
                    request=MagicMock(),
                    general_settings={},
                    proxy_logging_obj=MagicMock(),
                    user_api_key_dict=MagicMock(router_settings=None),
                    version=None,
                    proxy_config=MagicMock(),
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    model="gpt-4",
                    route_type="acompletion",
                    llm_router=mock_router,
                )

        assert processor.data["model"] == primary_model

    @pytest.mark.asyncio
    async def test_real_parallel_request_limiter_model_tpm_limit_triggers_fallback(self):
        """
        Customer-reported scenario from LIT-3890 / GH #8822.

        The prior tests in this class hand-build a ``ProxyRateLimitError``. The
        customer's production setup is different: they set a *per-key per-model*
        TPM cap on the key itself::

            Model TPM Limits: {"gpt-4.1-20250414-test": 100}

        and configure a proxy-side fallback (gpt-4.1-...-test -> gpt-4.1-...).
        When the per-model TPM cap trips, the real
        ``parallel_request_limiter`` raises ``ProxyRateLimitError`` from inside
        ``proxy_logging_obj.pre_call_hook`` — the seam ``_pre_call_with_fallbacks``
        wraps. This test drives that *real* limiter (not a mock error) end-to-end
        to prove the customer's exact knob triggers the gateway fallback instead
        of returning a 429 to the client.
        """
        from litellm.caching.caching import DualCache
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )
        from litellm.proxy.common_utils.proxy_rate_limit_error import (
            ProxyRateLimitError,
        )
        from litellm.proxy.hooks.parallel_request_limiter import (
            _PROXY_MaxParallelRequestsHandler,
        )
        from litellm.proxy.utils import InternalUsageCache

        primary_model = "gpt-4"
        fallback_model = "gpt-3.5-turbo"

        # Freeze the limiter's clock so the per-minute counter key is stable and
        # the pre-seeded counter is guaranteed to be the one it reads.
        class _FrozenClock(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 1, 1, 12, 30, 0)

        precise_minute = "2026-01-01-12-30"

        # Real per-key per-model TPM limiter + a key carrying the customer's
        # `model_tpm_limit` metadata (only the primary is capped).
        limiter = _PROXY_MaxParallelRequestsHandler(
            internal_usage_cache=InternalUsageCache(DualCache())
        )
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-lit3890",
            metadata={"model_tpm_limit": {primary_model: 100}},
        )

        # Pre-seed the primary's per-model token counter at the cap so the very
        # next request trips it. The counter key uses the *hashed* api_key.
        counter_key = (
            f"{user_api_key_dict.api_key}::{primary_model}"
            f"::{precise_minute}::request_count"
        )
        await limiter.internal_usage_cache.async_set_cache(
            key=counter_key,
            value={"current_requests": 0, "current_tpm": 100, "current_rpm": 0},
            litellm_parent_otel_span=None,
            local_only=True,
        )

        processor = ProxyBaseLLMRequestProcessing(data={"model": primary_model})

        # Stand in for common_processing_pre_call_logic's pre_call_hook step by
        # invoking the real limiter for whatever model is currently selected.
        limiter_calls = []

        async def real_limiter_pre_call(**kwargs):
            current_model = processor.data["model"]
            limiter_calls.append(current_model)
            await limiter.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data={
                    "model": current_model,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                call_type="acompletion",
            )
            return processor.data, MagicMock()

        mock_router = MagicMock()
        mock_router.fallbacks = [{primary_model: [fallback_model]}]

        with patch(
            "litellm.proxy.hooks.parallel_request_limiter.datetime", _FrozenClock
        ):
            with patch.object(
                processor,
                "common_processing_pre_call_logic",
                side_effect=real_limiter_pre_call,
            ):
                data, logging_obj = await processor._pre_call_with_fallbacks(
                    request=MagicMock(),
                    general_settings={},
                    proxy_logging_obj=MagicMock(),
                    user_api_key_dict=user_api_key_dict,
                    version=None,
                    proxy_config=MagicMock(),
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    model=primary_model,
                    route_type="acompletion",
                    llm_router=mock_router,
                )

        # The capped primary tripped the real limiter, and the fallback (which
        # has no per-model cap) served the request — no 429 to the client.
        assert processor.data["model"] == fallback_model
        assert limiter_calls == [primary_model, fallback_model]

        # Sanity-check the premise: the limiter genuinely raises a
        # ProxyRateLimitError for the capped primary under the frozen clock.
        with patch(
            "litellm.proxy.hooks.parallel_request_limiter.datetime", _FrozenClock
        ):
            with pytest.raises(ProxyRateLimitError):
                await limiter.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=DualCache(),
                    data={
                        "model": primary_model,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    call_type="acompletion",
                )


class _RecordingSuccessLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.success_events = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.success_events.append({"kwargs": kwargs, "response_obj": response_obj})


class TestStreamingClientDisconnectBilling:
    """
    A client disconnect throws GeneratorExit into the proxy streaming
    generator; neither the success nor failure logging callback fires from the
    stream wrapper, so without disconnect-time finalization the chunks already
    streamed (and any sub-call cost folded into the logging object) never
    reach spend tracking.
    """

    async def _start_partial_stream(self):
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "tell me a story"}],
            mock_response="The codename is AZURE-FALCON-42 and the story is long.",
            stream=True,
            api_key="test-key",
        )
        stream_iter = response.__aiter__()
        await stream_iter.__anext__()
        await stream_iter.__anext__()
        return response

    @pytest.mark.asyncio
    async def test_disconnect_bills_partial_streamed_spend(self):
        recorder = _RecordingSuccessLogger()
        original_callbacks = litellm.callbacks
        litellm.callbacks = [recorder]
        try:
            response = await self._start_partial_stream()
            logging_obj = response.logging_obj
            logging_obj.model_call_details["additional_response_cost"] = 0.002

            await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
                request=None,
                request_data={"litellm_logging_obj": logging_obj},
                response=response,
                stream_completed=False,
                client_disconnected=True,
            )

            for _ in range(50):
                if recorder.success_events:
                    break
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.5)
        finally:
            litellm.callbacks = original_callbacks

        assert len(recorder.success_events) == 1
        standard_logging_object = recorder.success_events[0]["kwargs"]["standard_logging_object"]
        assert standard_logging_object["total_tokens"] > 0
        assert standard_logging_object["response_cost"] >= 0.002

    @pytest.mark.asyncio
    async def test_completed_stream_does_not_double_bill_on_late_disconnect(self):
        recorder = _RecordingSuccessLogger()
        original_callbacks = litellm.callbacks
        litellm.callbacks = [recorder]
        try:
            response = await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="hello there",
                stream=True,
                api_key="test-key",
            )
            async for _ in response:
                pass

            await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
                request=None,
                request_data={"litellm_logging_obj": response.logging_obj},
                response=response,
                stream_completed=False,
                client_disconnected=True,
            )

            for _ in range(50):
                if recorder.success_events:
                    break
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.5)
        finally:
            litellm.callbacks = original_callbacks

        assert len(recorder.success_events) == 1

    @pytest.mark.asyncio
    async def test_disconnect_bills_partial_spend_for_router_stream(self):
        """
        The router wraps streamed responses in FallbackStreamWrapper, whose
        __anext__ bypasses the base class, so its own chunk list stays empty
        unless it aliases the inner stream's chunks; without the alias the
        disconnect path sees no chunks and bills nothing for router requests,
        which is every proxy request.
        """
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "test-key"},
                }
            ]
        )
        recorder = _RecordingSuccessLogger()
        original_callbacks = litellm.callbacks
        litellm.callbacks = [recorder]
        try:
            response = await router.acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "tell me a story"}],
                mock_response="The codename is AZURE-FALCON-42 and the story is long.",
                stream=True,
            )
            stream_iter = response.__aiter__()
            await stream_iter.__anext__()
            await stream_iter.__anext__()

            await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
                request=None,
                request_data={"litellm_logging_obj": response.logging_obj},
                response=response,
                stream_completed=False,
                client_disconnected=True,
            )

            for _ in range(50):
                if recorder.success_events:
                    break
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.5)
        finally:
            litellm.callbacks = original_callbacks

        assert len(recorder.success_events) == 1
        standard_logging_object = recorder.success_events[0]["kwargs"]["standard_logging_object"]
        assert standard_logging_object["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_disconnect_billing_does_not_double_release_slot(self):
        """
        The disconnect billing fires a success event whose limiter callback
        already releases the max_parallel_requests slot. The shielded cleanup
        must therefore NOT also release the slot explicitly; two releases of
        the same acquisition race and double-decrement under the limiter's
        in-memory fallback.
        """
        import types

        original_callbacks = litellm.callbacks
        litellm.callbacks = [_RecordingSuccessLogger()]
        try:
            response = await self._start_partial_stream()
            proxy_logging_obj = types.SimpleNamespace(
                _arelease_max_parallel_requests_on_disconnect=AsyncMock(),
            )

            billed = await _bill_partial_streamed_spend_on_disconnect(
                {"litellm_logging_obj": response.logging_obj}, response
            )
            assert billed is True

            await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
                request=None,
                request_data={"litellm_logging_obj": response.logging_obj},
                response=response,
                stream_completed=False,
                client_disconnected=True,
                user_api_key_dict=MagicMock(),
                proxy_logging_obj=proxy_logging_obj,
            )
        finally:
            litellm.callbacks = original_callbacks

        proxy_logging_obj._arelease_max_parallel_requests_on_disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_without_billable_chunks_releases_slot(self):
        """
        When there is nothing to bill (no chunks streamed), no success event
        fires, so the slot would leak unless the cleanup releases it
        explicitly. The explicit release must run exactly once in that case.
        """
        import types

        response = await self._start_partial_stream()
        # No chunks to assemble -> billing dispatches no success event.
        empty_response = types.SimpleNamespace(chunks=[], messages=None)
        proxy_logging_obj = types.SimpleNamespace(
            _arelease_max_parallel_requests_on_disconnect=AsyncMock(),
        )

        await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
            request=None,
            request_data={"litellm_logging_obj": response.logging_obj},
            response=empty_response,
            stream_completed=False,
            client_disconnected=True,
            user_api_key_dict=MagicMock(),
            proxy_logging_obj=proxy_logging_obj,
        )

        proxy_logging_obj._arelease_max_parallel_requests_on_disconnect.assert_awaited_once()
