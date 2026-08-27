import asyncio
import copy
import datetime
import json
from types import SimpleNamespace
from typing import AsyncGenerator, Callable, Final, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

import litellm
from litellm._uuid import uuid
from litellm.constants import RETURN_RAW_MODEL_NAME_METADATA_KEY
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
    CostBreakdownHeaderValues,
    _has_attribute_error_in_chain,
    _is_azure_model_router_request,
    open_sse_before_first_byte,
    ttft_keepalive_interval,
    _override_openai_response_model,
    _parse_event_data_for_error,
    _resolve_per_request_model_group_alias,
    _should_return_raw_model_name,
    _UpstreamClosingStreamingResponse,
    create_response,
)
from litellm.proxy.dd_span_tagger import DDSpanTagger
from litellm.proxy._types import ProxyException
from litellm.proxy._types import UserAPIKeyAuth as ProxyUserAPIKeyAuth
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

    @pytest.mark.asyncio
    async def test_common_processing_pre_call_logic_refreshes_proxy_server_request_body_after_guardrails(
        self, monkeypatch
    ):
        """
        A guardrail (e.g. Presidio PII masking) mutates data["messages"] in place inside
        pre_call_hook. The proxy_server_request.body snapshot is taken before that hook
        runs, so it must be refreshed afterward or SpendLogs (when store_prompts_in_spend_logs
        is enabled) persists the raw pre-guardrail body, bypassing the masking entirely.
        """
        processing_obj = ProxyBaseLLMRequestProcessing(data={})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        raw_messages = [{"role": "user", "content": "my ssn is 123-45-6789"}]

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return {
                "messages": raw_messages,
                "proxy_server_request": {
                    "url": "http://testserver/chat/completions",
                    "method": "POST",
                    "body": {"messages": raw_messages},
                },
            }

        async def mock_pre_call_hook(user_api_key_dict, data, call_type):
            data["messages"] = [{"role": "user", "content": "my ssn is <MASKED>"}]
            return data

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )

        returned_data, _ = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings={},
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=MagicMock(spec=ProxyConfig),
            route_type="acompletion",
        )

        persisted_body = returned_data["proxy_server_request"]["body"]
        assert persisted_body["messages"] == returned_data["messages"]
        assert "123-45-6789" not in json.dumps(persisted_body["messages"])
        # litellm_logging_obj is stamped onto `data` by function_setup between the
        # initial snapshot and pre_call_hook; it must never leak into the persisted
        # audit body, which needs to stay plain-JSON-serializable end to end.
        assert "litellm_logging_obj" not in persisted_body
        json.dumps(persisted_body)

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
        with pytest.raises(ValueError, match="could not convert string to float: 'invalid"):
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

    def test_get_custom_headers_per_component_cost_breakdown(self):
        """Test per-component cost headers against the stored production breakdown.

        cost_calculator stores full prompt cost (cache pricing included) as input_cost
        and full completion cost (reasoning included) as output_cost. The input header
        subtracts the cache components so the emitted contract is additive:
        input + cache_read + cache_creation + output + tool_usage == total, with
        reasoning remaining a subset of output.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        logging_obj = LiteLLMLoggingObj(
            model="gpt-5.4-nano",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-components",
            function_id="test-function",
        )

        input_cost: Final = 0.00002
        output_cost: Final = 0.00004
        cache_read_cost: Final = 0.000005
        cache_creation_cost: Final = 0.00001
        reasoning_cost: Final = 0.000015
        tool_usage_cost: Final = 0.00003
        total_cost: Final = input_cost + output_cost + tool_usage_cost
        uncached_input_cost: Final = input_cost - cache_read_cost - cache_creation_cost

        logging_obj.set_cost_breakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            cost_for_built_in_tools_cost_usd_dollar=tool_usage_cost,
            cache_read_cost=cache_read_cost,
            cache_creation_cost=cache_creation_cost,
            reasoning_cost=reasoning_cost,
        )

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            call_id="test-call-id-components",
            response_cost=total_cost,
            litellm_logging_obj=logging_obj,
        )

        assert "x-litellm-response-cost" in headers
        assert float(headers["x-litellm-response-cost"]) == pytest.approx(total_cost)

        assert "x-litellm-response-cost-input" in headers
        assert float(headers["x-litellm-response-cost-input"]) == pytest.approx(uncached_input_cost)

        assert "x-litellm-response-cost-output" in headers
        assert float(headers["x-litellm-response-cost-output"]) == pytest.approx(output_cost)

        assert "x-litellm-response-cost-cache-read" in headers
        assert float(headers["x-litellm-response-cost-cache-read"]) == pytest.approx(cache_read_cost)

        assert "x-litellm-response-cost-cache-creation" in headers
        assert float(headers["x-litellm-response-cost-cache-creation"]) == pytest.approx(cache_creation_cost)

        assert "x-litellm-response-cost-reasoning" in headers
        assert float(headers["x-litellm-response-cost-reasoning"]) == pytest.approx(reasoning_cost)

        assert "x-litellm-response-cost-tool-usage" in headers
        assert float(headers["x-litellm-response-cost-tool-usage"]) == pytest.approx(tool_usage_cost)

        component_sum: Final = (
            float(headers["x-litellm-response-cost-input"])
            + float(headers["x-litellm-response-cost-cache-read"])
            + float(headers["x-litellm-response-cost-cache-creation"])
            + float(headers["x-litellm-response-cost-output"])
            + float(headers["x-litellm-response-cost-tool-usage"])
        )
        assert component_sum == pytest.approx(float(headers["x-litellm-response-cost"]))
        assert float(headers["x-litellm-response-cost-reasoning"]) <= float(headers["x-litellm-response-cost-output"])

    def test_get_custom_headers_without_cost_breakdown_omits_component_headers(self):
        """Test that when litellm_logging_obj has no cost_breakdown, component headers are omitted."""
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        logging_obj = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-no-breakdown",
            function_id="test-function",
        )

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.0001,
            litellm_logging_obj=logging_obj,
        )

        assert "x-litellm-response-cost" in headers
        assert "x-litellm-response-cost-input" not in headers
        assert "x-litellm-response-cost-output" not in headers
        assert "x-litellm-response-cost-cache-read" not in headers
        assert "x-litellm-response-cost-cache-creation" not in headers
        assert "x-litellm-response-cost-reasoning" not in headers
        assert "x-litellm-response-cost-tool-usage" not in headers

    def test_get_custom_headers_per_component_with_discount_and_margin(self):
        """Test that component headers co-exist accurately with discount and margin headers."""
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        logging_obj = LiteLLMLoggingObj(
            model="gpt-4",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-combined",
            function_id="test-function",
        )

        logging_obj.set_cost_breakdown(
            input_cost=0.00006,
            output_cost=0.00004,
            total_cost=0.000105,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            original_cost=0.0001,
            discount_percent=0.05,
            discount_amount=0.000005,
            margin_percent=0.10,
            margin_total_amount=0.00001,
        )

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.000105,
            litellm_logging_obj=logging_obj,
        )

        assert float(headers["x-litellm-response-cost"]) == pytest.approx(0.000105)
        assert float(headers["x-litellm-response-cost-original"]) == pytest.approx(0.0001)
        assert float(headers["x-litellm-response-cost-discount-amount"]) == pytest.approx(0.000005)
        assert float(headers["x-litellm-response-cost-margin-amount"]) == pytest.approx(0.00001)
        assert float(headers["x-litellm-response-cost-margin-percent"]) == pytest.approx(0.10)
        assert float(headers["x-litellm-response-cost-input"]) == pytest.approx(0.00006)
        assert float(headers["x-litellm-response-cost-output"]) == pytest.approx(0.00004)
        assert "x-litellm-response-cost-cache-read" not in headers
        assert "x-litellm-response-cost-cache-creation" not in headers
        assert "x-litellm-response-cost-reasoning" not in headers
        assert float(headers["x-litellm-response-cost-tool-usage"]) == pytest.approx(0.0)

    @pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
    def test_get_custom_headers_classifier_cost_from_routing_decision(self, metadata_key):
        """The auto-router's LLM classifier cost must surface as its own header.

        x-litellm-response-cost stays the final routed call's cost (it feeds the
        margin/discount family and chargeback); the classifier's cost is read from the
        routing_decision the pre-routing hook recorded in the request metadata. The
        bucket is metadata on chat-style routes and litellm_metadata on messages-style
        routes, so both must work.
        """
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.00023,
            request_data={
                metadata_key: {
                    "routing_decision": {"cause": "llm_classifier", "classifier_cost": 8.1e-05},
                }
            },
        )

        assert headers["x-litellm-classifier-cost"] == "8.1e-05"
        assert float(headers["x-litellm-response-cost"]) == 0.00023

    @pytest.mark.parametrize(
        "request_data",
        [
            None,
            {},
            {"metadata": {}},
            {"metadata": {"routing_decision": {"cause": "heuristic_scorer"}}},
            {"metadata": {"routing_decision": {"cause": "llm_classifier", "classifier_cost": "bogus"}}},
            {"metadata": {"routing_decision": {"cause": "llm_classifier", "classifier_cost": True}}},
        ],
    )
    def test_get_custom_headers_omits_classifier_cost_without_a_priced_decision(self, request_data):
        """No routing decision, a decision without a classifier call, or a malformed cost
        must all omit the header entirely rather than emit 0 or a junk value."""
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.spend = 0

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=mock_user_api_key_dict,
            response_cost=0.00023,
            request_data=request_data,
        )

        assert "x-litellm-classifier-cost" not in headers

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

        breakdown = _get_cost_breakdown_from_logging_obj(logging_obj)
        assert breakdown.original_cost == 0.0001
        assert breakdown.discount_amount == 0.000005
        assert breakdown.margin_total_amount is None
        assert breakdown.margin_percent is None
        assert breakdown.input_cost == 0.00005
        assert breakdown.output_cost == 0.00005
        assert breakdown.tool_usage_cost == 0.0

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

        breakdown_with_margin = _get_cost_breakdown_from_logging_obj(logging_obj_with_margin)
        assert breakdown_with_margin.original_cost == 0.0001
        assert breakdown_with_margin.discount_amount is None
        assert breakdown_with_margin.margin_total_amount == 0.00001
        assert breakdown_with_margin.margin_percent == 0.10

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

        breakdown_no_discount = _get_cost_breakdown_from_logging_obj(logging_obj_no_discount)
        assert breakdown_no_discount.original_cost is None
        assert breakdown_no_discount.discount_amount is None
        assert breakdown_no_discount.margin_total_amount is None
        assert breakdown_no_discount.margin_percent is None
        assert breakdown_no_discount.input_cost == 0.00005
        assert breakdown_no_discount.output_cost == 0.00005

        # Test that cache components stored nested inside input_cost are subtracted out
        logging_obj_with_cache = LiteLLMLoggingObj(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=None,
            litellm_call_id="test-call-id-cache",
            function_id="test-function-id-cache",
        )
        logging_obj_with_cache.set_cost_breakdown(
            input_cost=0.00008,
            output_cost=0.00002,
            total_cost=0.0001,
            cost_for_built_in_tools_cost_usd_dollar=0.0,
            cache_read_cost=0.00003,
            cache_creation_cost=0.00004,
        )

        breakdown_with_cache = _get_cost_breakdown_from_logging_obj(logging_obj_with_cache)
        assert breakdown_with_cache.input_cost == pytest.approx(0.00001)
        assert breakdown_with_cache.cache_read_cost == 0.00003
        assert breakdown_with_cache.cache_creation_cost == 0.00004
        assert breakdown_with_cache.output_cost == 0.00002

        # Test with None logging object
        breakdown_none = _get_cost_breakdown_from_logging_obj(None)
        assert all(value is None for value in breakdown_none)

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

        # Verify queue_time_seconds is set and non-negative. Ends at start_time
        # (captured before this mock runs, so it can precede the mock's own
        # time.time() by a handful of microseconds) rather than a freshly
        # captured time.time(), so a tiny tolerance below 0.5 is expected and
        # correct -- see LIT-6012.
        metadata = returned_data.get("metadata", {})
        assert "queue_time_seconds" in metadata, "queue_time_seconds should be set in metadata"
        assert metadata["queue_time_seconds"] >= 0.49, (
            f"queue_time_seconds should be at least ~0.5, got {metadata['queue_time_seconds']}"
        )

        # queue_time_seconds must end exactly where logging_obj.start_time begins
        # (the same start_time litellm_request_total_latency_metric's window
        # starts from) so the two windows share a boundary, not an overlap.
        # A mutant that reintroduces a separately-captured processing_start_time
        # would make this assertion fail.
        arrival_time = returned_data["proxy_server_request"]["arrival_time"]
        assert arrival_time + metadata["queue_time_seconds"] == pytest.approx(
            logging_obj.start_time.timestamp(), abs=1e-6
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

    @pytest.mark.parametrize("return_raw_model_name", [False, True])
    def test_raw_model_name_toggle(self, return_raw_model_name):
        response_obj = {"model": "gpt-4o-mini"}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model="auto_router/complexity_router",
            log_context="test_context",
            return_raw_model_name=return_raw_model_name,
        )

        expected_model = "gpt-4o-mini" if return_raw_model_name else "auto_router/complexity_router"
        assert response_obj["model"] == expected_model

    @pytest.mark.parametrize(
        "request_data, expected",
        [
            ({"metadata": {}}, False),
            ({"metadata": {RETURN_RAW_MODEL_NAME_METADATA_KEY: True}}, True),
            ({"litellm_metadata": {RETURN_RAW_MODEL_NAME_METADATA_KEY: True}}, True),
        ],
    )
    def test_raw_model_name_toggle_metadata(self, request_data, expected):
        assert _should_return_raw_model_name(request_data) is expected

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

    def test_override_model_preserves_model_router_model_for_alias_without_router_in_name(
        self,
    ):
        """
        The client sends a model group alias, which carries no model_router/ prefix, so the
        name check alone only fires when the operator happened to put "model-router" in the
        alias. With the stamp on the response the actual model survives whatever it is named.
        """
        from litellm.llms.azure_ai.common_utils import (
            AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY,
        )

        requested_model = "smart-pick"
        actual_model_used = "azure_ai/grok-4-1-fast-reasoning"

        response_obj = MagicMock()
        response_obj.model = actual_model_used
        response_obj._hidden_params = {
            "additional_headers": {},
            AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY: actual_model_used,
        }

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == actual_model_used

    def test_override_model_still_restamps_non_router_alias_without_stamp(self):
        """
        Control for the test above: absent the stamp, an ordinary deployment keeps being
        restamped to the requested model, so the stamp is doing the work rather than the
        preserve branch having gone unconditional.
        """
        requested_model = "smart-pick"

        response_obj = MagicMock()
        response_obj.model = "azure_ai/grok-4-1-fast-reasoning"
        response_obj._hidden_params = {"additional_headers": {}}

        _override_openai_response_model(
            response_obj=response_obj,
            requested_model=requested_model,
            log_context="test_context",
        )
        assert response_obj.model == requested_model

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

    async def _invoke(self, exc: Exception, callback_headers: Optional[dict] = None):
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth

        processor = ProxyBaseLLMRequestProcessing(data={})
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value=callback_headers or {})

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


class TestHandleLLMApiExceptionFramingHeaders:
    """HTTP-framing headers on the provider exception must be stripped before the
    proxy builds its own response, or they conflict with the framing the proxy
    itself sets. Non-framing headers must survive unchanged."""

    async def _invoke(self, exc: Exception, callback_headers: Optional[dict] = None):
        from litellm.proxy._types import ProxyException, UserAPIKeyAuth

        processor = ProxyBaseLLMRequestProcessing(data={})
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
        proxy_logging_obj.post_call_response_headers_hook = AsyncMock(return_value=callback_headers or {})

        try:
            await processor._handle_llm_api_exception(
                e=exc,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=proxy_logging_obj,
            )
        except ProxyException as raised:
            return raised
        raise AssertionError("ProxyException was not raised")

    async def test_strips_framing_headers_preserves_others(self):
        exc = litellm.RateLimitError(
            message="Resource exhausted",
            llm_provider="vertex_ai",
            model="gemini-2.0-flash",
        )
        exc.headers = {
            "content-length": "42",
            "transfer-encoding": "chunked",
            "content-encoding": "gzip",
            "content-type": "application/json",
            "x-request-id": "abc-123",
        }
        proxy_exc = await self._invoke(exc)
        assert "content-length" not in proxy_exc.headers
        assert "transfer-encoding" not in proxy_exc.headers
        assert "content-encoding" not in proxy_exc.headers
        assert "content-type" not in proxy_exc.headers
        assert proxy_exc.headers["x-request-id"] == "abc-123"

    async def test_strips_framing_headers_on_existing_proxy_exception(self):
        from litellm.proxy._types import ProxyException

        exc = ProxyException(
            message="Resource exhausted",
            type="rate_limit_error",
            param=None,
            code=429,
            headers={
                "content-length": "42",
                "transfer-encoding": "chunked",
                "x-request-id": "abc-123",
            },
        )
        proxy_exc = await self._invoke(exc)
        assert "content-length" not in proxy_exc.headers
        assert "transfer-encoding" not in proxy_exc.headers
        assert proxy_exc.headers["x-request-id"] == "abc-123"

    async def test_strips_browser_security_headers(self):
        exc = litellm.RateLimitError(
            message="Resource exhausted",
            llm_provider="vertex_ai",
            model="gemini-2.0-flash",
        )
        exc.headers = {
            "access-control-allow-origin": "https://evil.example.com",
            "content-security-policy": "default-src https://evil.example.com",
            "clear-site-data": '"cache", "cookies", "storage"',
            "strict-transport-security": "max-age=0",
            "x-frame-options": "ALLOWALL",
            "x-request-id": "abc-123",
        }
        proxy_exc = await self._invoke(exc)
        assert "access-control-allow-origin" not in proxy_exc.headers
        assert "content-security-policy" not in proxy_exc.headers
        assert "clear-site-data" not in proxy_exc.headers
        assert "strict-transport-security" not in proxy_exc.headers
        assert "x-frame-options" not in proxy_exc.headers
        assert proxy_exc.headers["x-request-id"] == "abc-123"

    async def test_strips_unsafe_headers_added_by_response_headers_hook(self):
        exc = litellm.RateLimitError(
            message="Resource exhausted",
            llm_provider="vertex_ai",
            model="gemini-2.0-flash",
        )
        exc.headers = {"x-request-id": "abc-123"}
        proxy_exc = await self._invoke(
            exc,
            callback_headers={
                "x-frame-options": "ALLOWALL",
                "content-length": "42",
                "x-custom-safe": "1",
            },
        )
        assert "x-frame-options" not in proxy_exc.headers
        assert "content-length" not in proxy_exc.headers
        assert proxy_exc.headers["x-custom-safe"] == "1"
        assert proxy_exc.headers["x-request-id"] == "abc-123"


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

    @pytest.mark.asyncio
    async def test_bedrock_invoke_stream_sets_event_stream_content_type(self, monkeypatch):
        """
        Regression for LIT-4561. The unbuffered Bedrock event-stream relay
        (invoke-with-response-stream, no post-call guardrail rewriting) must set
        content-type: application/vnd.amazon.eventstream instead of emitting no
        content-type header at all, which trips Claude Code's content-type guard
        added in 2.1.208
        """
        processing_obj = self._build_processing_obj(
            "bedrock", "model/us.anthropic.claude-sonnet-4-20250514-v1:0/invoke-with-response-stream"
        )
        chunks = [b"raw-1", b"raw-2"]

        with patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails",
            return_value=False,
        ), patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails_for_passthrough",
            return_value=False,
        ):
            result = await self._run(processing_obj, monkeypatch, chunks)

        assert isinstance(result, StreamingResponse)
        assert result.media_type == "application/vnd.amazon.eventstream"
        assert result.headers["content-type"] == "application/vnd.amazon.eventstream"
        streamed = [chunk async for chunk in result.body_iterator]
        assert streamed == chunks

    @pytest.mark.asyncio
    async def test_non_bedrock_stream_keeps_default_content_type(self, monkeypatch):
        """
        A provider with no registered event-stream media type must not have one
        forced onto its unbuffered stream, so the response default is unchanged
        """
        processing_obj = self._build_processing_obj("anthropic")
        chunks = [b"chunk-1", b"chunk-2"]

        with patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails",
            return_value=False,
        ), patch.object(
            ProxyBaseLLMRequestProcessing,
            "_has_post_call_guardrails_for_passthrough",
            return_value=False,
        ):
            result = await self._run(processing_obj, monkeypatch, chunks)

        assert isinstance(result, StreamingResponse)
        assert result.media_type is None
        assert "content-type" not in result.headers


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


class TestCostHeadersForCallsPricedAtZero:
    """
    Regression for LIT-5602. Pricing responses reads and vector-store management routes at
    zero dropped the entire x-litellm-response-cost family off those replies: the header
    build reads a falsy zero as "this response never recorded a cost" and filters it out,
    and a call that returns before pricing stores no cost breakdown for the component
    headers to read. A client parsing the cost off a read got a KeyError where it had
    previously been handed a number. Those calls now advertise the whole family at zero.
    """

    @staticmethod
    def _responses_read(*, background=False):
        from litellm.types.llms.openai import ResponsesAPIResponse

        return ResponsesAPIResponse(
            id="resp_lit5602",
            created_at=0,
            model="gpt-4.1-mini",
            object="response",
            output=[],
            status="completed",
            background=background,
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    @staticmethod
    def _logging_obj(*, call_type, recovered_cost=0.0):
        logging_obj = MagicMock()
        logging_obj.litellm_call_id = "call-lit5602"
        logging_obj.call_type = call_type
        logging_obj.litellm_params = {}
        logging_obj.cost_breakdown = None
        logging_obj.model_call_details = {"response_cost": recovered_cost}
        logging_obj._response_cost_calculator = MagicMock(return_value=recovered_cost)
        logging_obj._enqueue_deferred_logging = None
        logging_obj._on_deferred_stream_complete = None
        return logging_obj

    async def _drive(self, *, monkeypatch, response, logging_obj, route_type):
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
            ProxyBaseLLMRequestProcessing, "_has_post_call_guardrails", return_value=False
        ):
            await processing_obj.base_process_llm_request(
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
        return fastapi_response

    @pytest.mark.asyncio
    async def test_responses_read_emits_the_cost_header_family_at_zero(self, monkeypatch):
        fastapi_response = await self._drive(
            monkeypatch=monkeypatch,
            response=self._responses_read(),
            logging_obj=self._logging_obj(call_type="aget_responses"),
            route_type="aget_responses",
        )

        assert fastapi_response.headers["x-litellm-response-cost"] == "0.0"
        for component in (
            "original",
            "discount-amount",
            "margin-amount",
            "margin-percent",
            "input",
            "output",
            "tool-usage",
        ):
            assert fastapi_response.headers[f"x-litellm-response-cost-{component}"] == "0.0"

    @pytest.mark.asyncio
    async def test_reading_a_background_response_keeps_its_real_cost(self, monkeypatch):
        fastapi_response = await self._drive(
            monkeypatch=monkeypatch,
            response=self._responses_read(background=True),
            logging_obj=self._logging_obj(call_type="aget_responses", recovered_cost=0.00042),
            route_type="aget_responses",
        )

        assert float(fastapi_response.headers["x-litellm-response-cost"]) == pytest.approx(0.00042)

    @pytest.mark.asyncio
    async def test_an_inference_call_without_a_recorded_cost_still_omits_the_header(self, monkeypatch):
        """A chat completion has no zero-priced route, so a falsy cost there means the cost was
        never recorded and the header stays absent rather than advertising a made-up zero."""
        fastapi_response = await self._drive(
            monkeypatch=monkeypatch,
            response=SimpleNamespace(_hidden_params={}),
            logging_obj=self._logging_obj(call_type="acompletion"),
            route_type="acompletion",
        )

        assert "x-litellm-response-cost" not in fastapi_response.headers

    def test_cost_breakdown_reports_zero_components_for_a_call_priced_at_zero(self):
        breakdown = _get_cost_breakdown_from_logging_obj(
            litellm_logging_obj=self._logging_obj(call_type="aget_responses")
        )

        assert breakdown.original_cost == 0.0
        assert breakdown.input_cost == 0.0
        assert breakdown.output_cost == 0.0
        assert breakdown.tool_usage_cost == 0.0

    def test_cost_breakdown_stays_empty_for_an_inference_call(self):
        breakdown = _get_cost_breakdown_from_logging_obj(
            litellm_logging_obj=self._logging_obj(call_type="acompletion")
        )

        assert breakdown == CostBreakdownHeaderValues()

    def test_cost_breakdown_never_zeroes_the_split_under_a_real_total(self):
        """Reading a background response prices normally, so a breakdown that has not landed by the
        time headers are built is reported as absent rather than as a zero split contradicting the
        real total alongside it."""
        breakdown = _get_cost_breakdown_from_logging_obj(
            litellm_logging_obj=self._logging_obj(call_type="aget_responses"),
            response_cost=1.96e-05,
        )

        assert breakdown == CostBreakdownHeaderValues()

    def test_cost_breakdown_reports_zero_components_under_a_zero_total(self):
        breakdown = _get_cost_breakdown_from_logging_obj(
            litellm_logging_obj=self._logging_obj(call_type="aget_responses"),
            response_cost=0.0,
        )

        assert breakdown.original_cost == 0.0
        assert breakdown.input_cost == 0.0
        assert breakdown.output_cost == 0.0


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

    async def _bill_and_collect_success_event(self, prepare=None, request_data=None):
        recorder = _RecordingSuccessLogger()
        original_callbacks = litellm.callbacks
        litellm.callbacks = [recorder]
        try:
            response = await self._start_partial_stream()
            if prepare is not None:
                prepare(response)
            billed = await _bill_partial_streamed_spend_on_disconnect(
                {"litellm_logging_obj": response.logging_obj, **(request_data or {})}, response
            )
            assert billed is True
            for _ in range(50):
                if recorder.success_events:
                    break
                await asyncio.sleep(0.1)
        finally:
            litellm.callbacks = original_callbacks
        assert len(recorder.success_events) == 1
        return recorder.success_events[0]

    @pytest.mark.asyncio
    async def test_disconnect_billing_prices_alias_restamped_chunks_at_real_model(self):
        assert "openai/my-public-alias" not in litellm.model_cost

        def restamp_chunks_to_alias(response):
            for chunk in response.chunks:
                chunk.model = "my-public-alias"

        event = await self._bill_and_collect_success_event(restamp_chunks_to_alias)

        assert event["response_obj"].model == "gpt-4o-mini"
        standard_logging_object = event["kwargs"]["standard_logging_object"]
        assert standard_logging_object["response_cost"] > 0.0

    @pytest.mark.asyncio
    async def test_disconnect_billing_prices_a_partly_restamped_chunk_list_at_real_model(self):
        """
        A chunk that carries usage is stored as a copy before the proxy restamps the
        one it forwards, so an aliased stream can reach billing with its first chunk
        still on the deployment model and the rest on the client's name.
        """
        assert "openai/my-public-alias" not in litellm.model_cost

        def restamp_only_the_chunks_the_proxy_forwarded(response):
            for chunk in response.chunks[1:]:
                chunk.model = "my-public-alias"

        event = await self._bill_and_collect_success_event(
            restamp_only_the_chunks_the_proxy_forwarded,
            request_data={"model": "my-public-alias"},
        )

        assert event["response_obj"].model == "gpt-4o-mini"
        standard_logging_object = event["kwargs"]["standard_logging_object"]
        assert standard_logging_object["response_cost"] > 0.0

    @pytest.mark.asyncio
    async def test_disconnect_billing_keeps_the_model_azure_model_router_picked(self):
        def restamp_like_azure_model_router(response):
            response.chunks[0].model = "azure-model-router"
            for chunk in response.chunks[1:]:
                chunk.model = "gpt-4.1-nano-2025-04-14"

        event = await self._bill_and_collect_success_event(
            restamp_like_azure_model_router,
            request_data={"model": "azure-model-router"},
        )

        assert event["response_obj"].model == "gpt-4.1-nano-2025-04-14"
        standard_logging_object = event["kwargs"]["standard_logging_object"]
        assert standard_logging_object["response_cost"] > 0.0

    @pytest.mark.asyncio
    async def test_disconnect_billing_keeps_the_routed_model_when_request_data_model_was_rewritten(self):
        """
        Pre-call processing rewrites request_data["model"] for aliasing and routing, so the
        routed model on the later chunks can end up matching it. Only the name the client
        sent says whether the proxy restamped this stream.
        """

        def restamp_like_azure_model_router(response):
            response.chunks[0].model = "azure-model-router"
            for chunk in response.chunks[1:]:
                chunk.model = "gpt-4.1-nano-2025-04-14"

        event = await self._bill_and_collect_success_event(
            restamp_like_azure_model_router,
            request_data={
                "model": "gpt-4.1-nano-2025-04-14",
                "_litellm_client_requested_model": "azure-model-router",
            },
        )

        assert event["response_obj"].model == "gpt-4.1-nano-2025-04-14"
        standard_logging_object = event["kwargs"]["standard_logging_object"]
        assert standard_logging_object["response_cost"] > 0.0

    @pytest.mark.asyncio
    async def test_disconnect_billing_backfills_missing_cache_fields(self):
        event = await self._bill_and_collect_success_event()

        usage = event["response_obj"].usage
        assert getattr(usage, "cache_creation_input_tokens", None) == 0
        assert getattr(usage, "cache_read_input_tokens", None) == 0
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 0

    @pytest.mark.asyncio
    async def test_disconnect_billing_carries_up_openai_style_cached_tokens(self):
        from litellm.types.utils import (
            Delta,
            ModelResponseStream,
            PromptTokensDetailsWrapper,
            StreamingChoices,
            Usage,
        )

        def append_openai_style_cached_usage_chunk(response):
            response.chunks.append(
                ModelResponseStream(
                    id=response.chunks[0].id,
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                    choices=[
                        StreamingChoices(
                            finish_reason=None,
                            index=0,
                            delta=Delta(content=" and more", role="assistant"),
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=1000,
                        completion_tokens=10,
                        total_tokens=1010,
                        prompt_tokens_details=PromptTokensDetailsWrapper(
                            cached_tokens=500
                        ),
                    ),
                )
            )

        event = await self._bill_and_collect_success_event(
            append_openai_style_cached_usage_chunk
        )

        usage = event["response_obj"].usage
        assert getattr(usage, "cache_read_input_tokens", None) == 500
        assert getattr(usage, "cache_creation_input_tokens", None) == 0

    @pytest.mark.asyncio
    async def test_disconnect_billing_keeps_cache_values_recovered_from_chunks(self):
        from litellm.types.utils import (
            Delta,
            ModelResponseStream,
            StreamingChoices,
            Usage,
        )

        def append_usage_chunk(response):
            response.chunks.append(
                ModelResponseStream(
                    id=response.chunks[0].id,
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                    choices=[
                        StreamingChoices(
                            finish_reason=None,
                            index=0,
                            delta=Delta(content=" and more", role="assistant"),
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=40,
                        completion_tokens=5,
                        total_tokens=45,
                        cache_read_input_tokens=7,
                        cache_creation_input_tokens=3,
                    ),
                )
            )

        event = await self._bill_and_collect_success_event(append_usage_chunk)

        usage = event["response_obj"].usage
        assert getattr(usage, "cache_read_input_tokens", None) == 7
        assert getattr(usage, "cache_creation_input_tokens", None) == 3
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 7


def _apply_stream_usage_tracking(
    data: dict,
    general_settings: dict,
    route_type: str,
    supports_stream_options: Callable[[], bool] = lambda: True,
) -> None:
    from litellm.proxy.common_request_processing import _stream_usage_tracking_updates

    data.update(
        _stream_usage_tracking_updates(
            data=data,
            general_settings=general_settings,
            route_type=route_type,
            supports_stream_options=supports_stream_options,
        )
    )


class TestApplyStreamUsageTracking:
    def test_default_injects_usage_and_marks_strip_for_chat_completions(self):
        data = {"stream": True, "model": "gpt-5.4-nano"}

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="acompletion")

        assert data["stream_options"] == {"include_usage": True}
        assert data["_litellm_strip_stream_usage"] is True

    def test_default_preserves_other_client_stream_options_keys(self):
        data = {"stream": True, "stream_options": {"include_obfuscation": True}}

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="acompletion")

        assert data["stream_options"] == {"include_obfuscation": True, "include_usage": True}
        assert data["_litellm_strip_stream_usage"] is True

    def test_client_requested_usage_is_left_untouched_and_not_stripped(self):
        data = {"stream": True, "stream_options": {"include_usage": True}}

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="acompletion")

        assert data["stream_options"] == {"include_usage": True}
        assert "_litellm_strip_stream_usage" not in data

    def test_client_include_usage_false_is_overridden_and_stripped(self):
        data = {"stream": True, "stream_options": {"include_usage": False}}

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="acompletion")

        assert data["stream_options"]["include_usage"] is True
        assert data["_litellm_strip_stream_usage"] is True

    def test_explicit_false_flag_disables_injection_entirely(self):
        data = {"stream": True}

        _apply_stream_usage_tracking(
            data=data,
            general_settings={"always_include_stream_usage": False},
            route_type="acompletion",
        )

        assert "stream_options" not in data
        assert "_litellm_strip_stream_usage" not in data

    def test_flag_true_injects_without_strip_marker(self):
        data = {"stream": True}

        _apply_stream_usage_tracking(
            data=data,
            general_settings={"always_include_stream_usage": True},
            route_type="acompletion",
        )

        assert data["stream_options"] == {"include_usage": True}
        assert "_litellm_strip_stream_usage" not in data

    def test_flag_true_respects_client_explicit_include_usage_false(self):
        data = {"stream": True, "stream_options": {"include_usage": False}}

        _apply_stream_usage_tracking(
            data=data,
            general_settings={"always_include_stream_usage": True},
            route_type="acompletion",
        )

        assert data["stream_options"] == {"include_usage": False}
        assert "_litellm_strip_stream_usage" not in data

    def test_default_does_not_touch_non_chat_completion_routes(self):
        data = {"stream": True}

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="anthropic_messages")

        assert "stream_options" not in data
        assert "_litellm_strip_stream_usage" not in data

    def test_non_streaming_request_is_untouched(self):
        data = {"model": "gpt-5.4-nano"}

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="acompletion")

        assert "stream_options" not in data
        assert "_litellm_strip_stream_usage" not in data

    def test_default_skips_injection_when_provider_lacks_stream_options_support(self):
        data = {"stream": True, "model": "bytez-model"}

        _apply_stream_usage_tracking(
            data=data,
            general_settings={},
            route_type="acompletion",
            supports_stream_options=lambda: False,
        )

        assert "stream_options" not in data
        assert "_litellm_strip_stream_usage" not in data

    def test_client_supplied_strip_marker_is_neutralized(self):
        data = {
            "stream": True,
            "stream_options": {"include_usage": True},
            "_litellm_strip_stream_usage": True,
        }

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="acompletion")

        assert data["_litellm_strip_stream_usage"] is False
        assert data["stream_options"] == {"include_usage": True}

    def test_client_supplied_strip_marker_is_neutralized_with_flag_true(self):
        data = {
            "stream": True,
            "stream_options": {"include_usage": True},
            "_litellm_strip_stream_usage": True,
        }

        _apply_stream_usage_tracking(
            data=data,
            general_settings={"always_include_stream_usage": True},
            route_type="acompletion",
        )

        assert data["_litellm_strip_stream_usage"] is False

    def test_client_supplied_strip_marker_is_neutralized_on_non_streaming_request(self):
        data = {"_litellm_strip_stream_usage": True}

        _apply_stream_usage_tracking(data=data, general_settings={}, route_type="acompletion")

        assert data["_litellm_strip_stream_usage"] is False


class TestModelDeploymentsSupportStreamOptions:
    def _support(self, model, llm_router=None, team_id=None) -> bool:
        from litellm.proxy.common_request_processing import (
            _model_deployments_support_stream_options,
        )

        return _model_deployments_support_stream_options(model=model, llm_router=llm_router, team_id=team_id)

    def test_openai_compatible_deployment_supports_stream_options(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "azure-nano",
                    "litellm_params": {
                        "model": "azure/gpt-5.4-nano",
                        "api_key": "fake",
                        "api_base": "https://example.openai.azure.com",
                    },
                }
            ]
        )

        assert self._support("azure-nano", router) is True

    def test_deployment_on_provider_rejecting_stream_options_is_not_injected(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "tiny",
                    "litellm_params": {"model": "bytez/openai-community/gpt2", "api_key": "fake"},
                }
            ]
        )

        assert self._support("tiny", router) is False

    def test_mixed_provider_model_group_is_not_injected(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "mixed",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                },
                {
                    "model_name": "mixed",
                    "litellm_params": {"model": "oci/cohere.command-r-plus", "api_key": "fake"},
                },
            ]
        )

        assert self._support("mixed", router) is False

    def test_wildcard_route_resolves_provider_support(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "openai/*",
                    "litellm_params": {"model": "openai/*", "api_key": "fake"},
                }
            ]
        )

        assert self._support("openai/gpt-4o", router) is True

    def test_provider_prefixed_model_without_router_is_resolved_directly(self):
        assert self._support("openai/gpt-4o", None) is True
        assert self._support("bytez/openai-community/gpt2", None) is False

    def test_unmapped_model_name_is_not_injected(self):
        assert self._support("some-unmapped-public-alias", None) is False

    def test_team_alias_model_resolves_with_team_id(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "model_name_team-1_8b6a0b3f",
                    "litellm_params": {"model": "azure/gpt-5.4-nano", "api_key": "fake"},
                    "model_info": {
                        "team_id": "team-1",
                        "team_public_model_name": "team-gpt",
                    },
                }
            ]
        )

        assert self._support("team-gpt", router, team_id="team-1") is True
        assert self._support("team-gpt", router, team_id=None) is False

    def test_non_string_model_is_not_injected(self):
        assert self._support(None, None) is False


class TestPerRequestModelGroupAlias:
    """``router_settings.model_group_alias`` on a key or team has to be resolved
    by the proxy: the Router resolves aliases from its own shared instance
    attribute, which only ever holds the global config map."""

    @staticmethod
    def _router() -> litellm.Router:
        return litellm.Router(
            model_list=[
                {
                    "model_name": "group-a",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                },
                {
                    "model_name": "group-b",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"},
                },
            ]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "alias_map, expected",
        [
            ({"group-a": "group-b"}, "group-b"),
            ({"group-a": {"model": "group-b", "hidden": True}}, "group-b"),
            ({"group-b": "group-a"}, None),
            ({"group-a": "group-a"}, None),
            ({"group-a": {"hidden": True}}, None),
            ({}, None),
            (None, None),
        ],
    )
    async def test_resolves_alias_for_the_requested_model_group(self, alias_map, expected):
        resolved = await _resolve_per_request_model_group_alias(
            requested_model="group-a",
            router_settings={"model_group_alias": alias_map},
            user_api_key_dict=ProxyUserAPIKeyAuth(api_key="hash", models=[]),
            llm_router=self._router(),
        )

        assert resolved == expected

    @pytest.mark.asyncio
    async def test_alias_target_outside_the_key_allowlist_is_rejected(self):
        """Access was authorized against the requested group, so a rewrite that
        the key could not have requested directly must not be served."""
        with pytest.raises(ProxyException) as exc_info:
            await _resolve_per_request_model_group_alias(
                requested_model="group-a",
                router_settings={"model_group_alias": {"group-a": "group-b"}},
                user_api_key_dict=ProxyUserAPIKeyAuth(api_key="hash", models=["group-a"]),
                llm_router=self._router(),
            )

        assert exc_info.value.code == "403"
        assert "group-b" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_alias_target_inside_the_key_allowlist_resolves(self):
        resolved = await _resolve_per_request_model_group_alias(
            requested_model="group-a",
            router_settings={"model_group_alias": {"group-a": "group-b"}},
            user_api_key_dict=ProxyUserAPIKeyAuth(api_key="hash", models=["group-a", "group-b"]),
            llm_router=self._router(),
        )

        assert resolved == "group-b"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("requested_model", [None, ["group-a", "group-b"]])
    async def test_non_string_requested_model_is_left_alone(self, requested_model):
        """The routed model is not always a string (a batch request carries a
        list), and an unhashable one must not blow up the alias lookup."""
        resolved = await _resolve_per_request_model_group_alias(
            requested_model=requested_model,
            router_settings={"model_group_alias": {"group-a": "group-b"}},
            user_api_key_dict=ProxyUserAPIKeyAuth(api_key="hash", models=[]),
            llm_router=self._router(),
        )

        assert resolved is None

    @pytest.mark.asyncio
    async def test_pre_call_logic_rewrites_the_requested_model(self, monkeypatch):
        """End to end through the request path: a key carrying the alias must
        leave pre-call processing pointing at the alias target, not at the
        group the caller asked for."""
        processing_obj = ProxyBaseLLMRequestProcessing(data={"model": "group-a"})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return kwargs.get("data", {})

        async def passthrough_pre_call_hook(user_api_key_dict, data, call_type):
            return copy.deepcopy(data)

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=passthrough_pre_call_hook)
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())

        mock_proxy_config = MagicMock(spec=ProxyConfig)
        mock_proxy_config._get_hierarchical_router_settings = AsyncMock(
            return_value={"model_group_alias": {"group-a": "group-b"}}
        )

        returned_data, _ = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings={},
            user_api_key_dict=ProxyUserAPIKeyAuth(api_key="hash", models=[]),
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=mock_proxy_config,
            route_type="acompletion",
            llm_router=self._router(),
        )

        assert returned_data["model"] == "group-b"
        assert returned_data["router_settings_override"] == {"model_group_alias": {"group-a": "group-b"}}
        # The rewrite has to land before the pre-call hooks: they are where
        # per-model budgets and rate limits are enforced, so resolving later
        # applies the requested group's limits to a call the target serves.
        assert mock_proxy_logging_obj.pre_call_hook.call_args.kwargs["data"]["model"] == "group-b"

    @pytest.mark.asyncio
    async def test_team_level_alias_rewrites_the_requested_model(self, monkeypatch):
        """The team path is separate resolution, not a variant of the key path:
        settings are looked up on the team only when the key carries none. Runs
        the real hierarchical lookup rather than mocking it, so this covers the
        team half of the fix end to end."""
        from litellm.proxy.proxy_server import ProxyConfig as RealProxyConfig

        processing_obj = ProxyBaseLLMRequestProcessing(data={"model": "group-a"})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return kwargs.get("data", {})

        async def passthrough_pre_call_hook(user_api_key_dict, data, call_type):
            return copy.deepcopy(data)

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=passthrough_pre_call_hook)
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.get_team_object",
            AsyncMock(return_value=SimpleNamespace(router_settings={"model_group_alias": {"group-a": "group-b"}})),
        )

        returned_data, _ = await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings={},
            user_api_key_dict=ProxyUserAPIKeyAuth(api_key="hash", models=[], team_id="team-1"),
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=RealProxyConfig(),
            route_type="acompletion",
            llm_router=self._router(),
        )

        assert returned_data["model"] == "group-b"

    @pytest.mark.asyncio
    async def test_model_level_guardrails_resolve_against_the_alias_target(self, monkeypatch):
        """Model-level guardrails are merged by model group name, so the merge
        must see the target rather than the group the caller named."""
        processing_obj = ProxyBaseLLMRequestProcessing(data={"model": "group-a"})
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        async def mock_add_litellm_data_to_request(*args, **kwargs):
            return kwargs.get("data", {})

        async def passthrough_pre_call_hook(user_api_key_dict, data, call_type):
            return copy.deepcopy(data)

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
        mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=passthrough_pre_call_hook)
        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "add_litellm_data_to_request",
            mock_add_litellm_data_to_request,
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())

        merged_for: list = []

        def recording_merge(data, llm_router, trust_client_model_info=True):
            merged_for.append(data.get("model"))
            return data

        monkeypatch.setattr(
            litellm.proxy.common_request_processing,
            "_check_and_merge_model_level_guardrails",
            recording_merge,
        )

        mock_proxy_config = MagicMock(spec=ProxyConfig)
        mock_proxy_config._get_hierarchical_router_settings = AsyncMock(
            return_value={"model_group_alias": {"group-a": "group-b"}}
        )

        await processing_obj.common_processing_pre_call_logic(
            request=mock_request,
            general_settings={},
            user_api_key_dict=ProxyUserAPIKeyAuth(api_key="hash", models=[]),
            proxy_logging_obj=mock_proxy_logging_obj,
            proxy_config=mock_proxy_config,
            route_type="acompletion",
            llm_router=self._router(),
        )

        assert merged_for == ["group-b"]


class TestInjectCostIntoUsageDict:
    @staticmethod
    def _expected_cost(model, prompt_tokens, completion_tokens):
        pricing = litellm.model_cost[model]
        return prompt_tokens * pricing["input_cost_per_token"] + completion_tokens * pricing["output_cost_per_token"]

    def test_openai_chat_completion_chunk_usage_gets_cost(self):
        event = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, "gpt-4o-mini")

        assert result is not None
        assert result["usage"]["cost"] == pytest.approx(self._expected_cost("gpt-4o-mini", 11, 4))
        assert result["usage"]["cost"] > 0
        assert result["usage"]["prompt_tokens"] == 11
        assert result["id"] == "chatcmpl-1"
        assert "cost" not in event["usage"]

    def test_anthropic_message_delta_usage_still_gets_cost(self):
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, "claude-haiku-4-5")

        assert result is not None
        assert result["usage"]["cost"] == pytest.approx(self._expected_cost("claude-haiku-4-5", 11, 4))
        assert result["usage"]["cost"] > 0
        assert result["usage"]["output_tokens"] == 4

    def test_openai_chunk_with_flex_service_tier_uses_flex_pricing(self):
        event = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "service_tier": "flex",
            "choices": [],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, "gpt-5-mini")

        assert result is not None
        pricing = litellm.model_cost["gpt-5-mini"]
        expected_flex_cost = 1000 * pricing["input_cost_per_token_flex"] + 100 * pricing["output_cost_per_token_flex"]
        assert result["usage"]["cost"] == pytest.approx(expected_flex_cost)
        assert result["usage"]["cost"] < self._expected_cost("gpt-5-mini", 1000, 100)

    def test_openai_chunk_with_null_usage_is_not_modified(self):
        event = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
            "usage": None,
        }

        assert ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, "gpt-4o-mini") is None

    def test_unrecognized_event_shape_with_usage_is_not_modified(self):
        event = {"kind": "custom", "usage": {"prompt_tokens": 11, "completion_tokens": 4}}

        assert ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, "gpt-4o-mini") is None

    def test_sse_frame_with_coalesced_done_line_injects_into_usage_frame(self):
        frame = (
            'data: {"object":"chat.completion.chunk","choices":[],'
            '"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}\n\n'
            "data: [DONE]\n\n"
        )

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_sse_frame_str(frame, "gpt-4o-mini")

        assert result is not None
        assert "data: [DONE]" in result
        injected = json.loads(result.split("\n")[0].split("data:", 1)[1].strip())
        assert injected["usage"]["cost"] == pytest.approx(self._expected_cost("gpt-4o-mini", 11, 4))

    def test_message_delta_cost_charges_the_non_cached_input_tokens(self):
        """Anthropic reports ``input_tokens`` excluding cache tokens, so reading it as the whole
        prompt total drops the non-cached input from the bill on every cache hit."""
        model = "claude-haiku-4-5"
        pricing = litellm.model_cost[model]
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 14,
                "output_tokens": 8,
                "cache_read_input_tokens": 3202,
                "cache_creation_input_tokens": 0,
            },
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, model)

        assert result is not None
        expected = (
            14 * pricing["input_cost_per_token"]
            + 3202 * pricing["cache_read_input_token_cost"]
            + 8 * pricing["output_cost_per_token"]
        )
        dropped_input = expected - 14 * pricing["input_cost_per_token"]
        assert result["usage"]["cost"] == pytest.approx(expected)
        assert result["usage"]["cost"] > dropped_input

    def test_message_delta_prices_1h_cache_creation_above_the_5m_rate(self):
        """The ``cache_creation`` 5m/1h split has to survive into ``prompt_tokens_details``,
        otherwise a 1h write is billed at the cheaper 5m rate."""
        model = "claude-haiku-4-5"
        pricing = litellm.model_cost[model]
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 14,
                "output_tokens": 8,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 2000,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 2000},
            },
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, model)

        assert result is not None
        base = 14 * pricing["input_cost_per_token"] + 8 * pricing["output_cost_per_token"]
        expected_1h = base + 2000 * pricing["cache_creation_input_token_cost_above_1hr"]
        flat_5m = base + 2000 * pricing["cache_creation_input_token_cost"]
        assert expected_1h != pytest.approx(flat_5m)
        assert result["usage"]["cost"] == pytest.approx(expected_1h)

    def test_message_delta_prices_through_the_logging_obj_so_custom_pricing_applies(self):
        """Costing by model name alone yields sticker price, so a deployment with a negotiated
        discount streamed a ``usage.cost`` that disagreed with the callback's ``response_cost``."""

        class _StubLoggingObj:
            def __init__(self, cost):
                self._cost = cost
                self.captured_result = None

            def _response_cost_calculator(self, result):
                self.captured_result = result
                return self._cost

        model = "claude-haiku-4-5"
        discounted_cost = 0.00099
        stub = _StubLoggingObj(discounted_cost)
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 14,
                "output_tokens": 8,
                "cache_read_input_tokens": 3202,
                "cache_creation_input_tokens": 500,
                "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 400},
            },
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, model, stub)

        assert result is not None
        assert result["usage"]["cost"] == discounted_cost
        assert result["usage"]["cost"] != pytest.approx(self._expected_cost(model, 14 + 500 + 3202, 8))
        usage = stub.captured_result.usage
        assert usage.prompt_tokens == 14 + 500 + 3202
        details = usage.prompt_tokens_details.cache_creation_token_details
        assert details.ephemeral_5m_input_tokens == 100
        assert details.ephemeral_1h_input_tokens == 400

    def test_message_delta_falls_back_to_model_pricing_when_the_logging_obj_returns_no_cost(self):
        class _StubLoggingObj:
            def _response_cost_calculator(self, result):
                return None

        model = "claude-haiku-4-5"
        pricing = litellm.model_cost[model]
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 14, "output_tokens": 8, "cache_read_input_tokens": 3202},
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, model, _StubLoggingObj())

        assert result is not None
        assert result["usage"]["cost"] == pytest.approx(
            14 * pricing["input_cost_per_token"]
            + 3202 * pricing["cache_read_input_token_cost"]
            + 8 * pricing["output_cost_per_token"]
        )

    def test_message_delta_falls_back_to_model_pricing_when_the_logging_obj_raises(self):
        """A pricing failure mid-stream must not break the frame, so the raise falls back to
        model-name pricing rather than propagating into the response body."""

        class _StubLoggingObj:
            def _response_cost_calculator(self, result):
                raise ValueError("no pricing for this deployment")

        model = "claude-haiku-4-5"
        pricing = litellm.model_cost[model]
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 14, "output_tokens": 8, "cache_read_input_tokens": 3202},
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, model, _StubLoggingObj())

        assert result is not None
        assert result["usage"]["cost"] == pytest.approx(
            14 * pricing["input_cost_per_token"]
            + 3202 * pricing["cache_read_input_token_cost"]
            + 8 * pricing["output_cost_per_token"]
        )

    def test_pricing_a_frame_leaves_the_real_logging_obj_unchanged(self):
        """Pricing runs against the live logging object, and the pass-through handlers never
        recompute cost_breakdown, so a frame-derived breakdown would reach the spend log."""
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.types.utils import ModelResponse, Usage

        logging_obj = LiteLLMLoggingObj(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            call_type="completion",
            start_time=None,
            litellm_call_id="lit4902-breakdown-test",
            function_id="lit4902-breakdown-test",
        )
        logging_obj.update_environment_variables(litellm_params={}, optional_params={})
        logging_obj.model_call_details["custom_llm_provider"] = "anthropic"
        assert logging_obj.cost_breakdown is None

        model_response = ModelResponse(
            usage=Usage(prompt_tokens=3216, completion_tokens=8, total_tokens=3224)
        )
        cost = ProxyBaseLLMRequestProcessing._logging_obj_cost_or_none(model_response, logging_obj)

        assert cost is not None and cost > 0
        assert logging_obj.cost_breakdown is None
        assert "response_cost_failure_debug_information" not in logging_obj.model_call_details

    def test_pricing_a_frame_restores_a_breakdown_the_request_already_had(self):
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.types.utils import ModelResponse, Usage

        logging_obj = LiteLLMLoggingObj(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            call_type="completion",
            start_time=None,
            litellm_call_id="lit4902-breakdown-restore",
            function_id="lit4902-breakdown-restore",
        )
        logging_obj.update_environment_variables(litellm_params={}, optional_params={})
        logging_obj.model_call_details["custom_llm_provider"] = "anthropic"
        logging_obj.set_cost_breakdown(
            input_cost=0.5, output_cost=0.25, total_cost=0.75, cost_for_built_in_tools_cost_usd_dollar=0.0
        )
        existing = logging_obj.cost_breakdown

        model_response = ModelResponse(
            usage=Usage(prompt_tokens=3216, completion_tokens=8, total_tokens=3224)
        )
        ProxyBaseLLMRequestProcessing._logging_obj_cost_or_none(model_response, logging_obj)

        assert logging_obj.cost_breakdown is existing
        assert logging_obj.cost_breakdown["total_cost"] == 0.75

    def test_openai_chunk_prices_through_the_logging_obj_so_custom_pricing_applies(self):
        """The chat.completion.chunk path rides the same pricer, so a discounted deployment
        streaming /v1/chat/completions gets its negotiated price instead of sticker."""

        class _StubLoggingObj:
            def __init__(self, cost):
                self._cost = cost
                self.captured_result = None

            def _response_cost_calculator(self, result):
                self.captured_result = result
                return self._cost

        discounted_cost = 0.00031
        stub = _StubLoggingObj(discounted_cost)
        event = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, "gpt-4o-mini", stub)

        assert result is not None
        assert result["usage"]["cost"] == discounted_cost
        assert result["usage"]["cost"] != pytest.approx(self._expected_cost("gpt-4o-mini", 1000, 100))
        usage = stub.captured_result.usage
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 100

    def test_openai_chunk_falls_back_to_model_pricing_when_the_logging_obj_returns_no_cost(self):
        class _StubLoggingObj:
            def _response_cost_calculator(self, result):
                return None

        event = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }

        result = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(event, "gpt-4o-mini", _StubLoggingObj())

        assert result is not None
        assert result["usage"]["cost"] == pytest.approx(self._expected_cost("gpt-4o-mini", 11, 4))


class TestProcessChunkWithCostInjection:
    def test_complete_usage_frame_chunk_is_injected(self, monkeypatch):
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
        chunk = (
            b'data: {"object":"chat.completion.chunk","choices":[],'
            b'"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}\n\n'
        )

        result = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(chunk, "gpt-4o-mini")

        assert result != chunk
        assert result.endswith(b"\n\n")
        payload = json.loads(result.decode("utf-8").split("data:", 1)[1].strip())
        assert payload["usage"]["cost"] > 0

    def test_chunk_ending_in_partial_frame_passes_through_byte_identical(self, monkeypatch):
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
        chunk = (
            b'data: {"object":"chat.completion.chunk","choices":[],'
            b'"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}\n\ndata: [DO'
        )

        assert ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(chunk, "gpt-4o-mini") == chunk

    def test_chunk_with_invalid_utf8_passes_through_byte_identical(self, monkeypatch):
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
        chunk = (
            b'\xa8data: {"object":"chat.completion.chunk","choices":[],'
            b'"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}\n\n'
        )

        assert ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(chunk, "gpt-4o-mini") == chunk

    def test_message_delta_frame_is_priced_with_the_logging_obj(self, monkeypatch):
        """Pins that the logging object reaches the pricer through the byte-frame entry point,
        which is how the proxy actually calls this on a streamed Messages API request."""
        monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)

        class _StubLoggingObj:
            def _response_cost_calculator(self, result):
                return 0.00042

        chunk = (
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"input_tokens":14,"output_tokens":8,"cache_read_input_tokens":3202}}\n\n'
        )

        result = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
            chunk, "claude-haiku-4-5", _StubLoggingObj()
        )

        assert result != chunk
        data_line = next(ln for ln in result.decode("utf-8").splitlines() if ln.startswith("data:"))
        payload = json.loads(data_line.split("data:", 1)[1].strip())
        assert payload["usage"]["cost"] == 0.00042
        assert payload["usage"]["cache_read_input_tokens"] == 3202


# ---------------------------------------------------------------------------
# SSE keepalive during the time-to-first-token (issue #34819)
# ---------------------------------------------------------------------------

TTFT_PING = b": ping\n\n"


async def _drain(response):
    return [chunk async for chunk in response.body_iterator]


def _sse_response(chunks, upstream_generator=None):
    async def gen():
        for chunk in chunks:
            yield chunk

    if upstream_generator is None:
        return StreamingResponse(gen(), media_type="text/event-stream")
    return _UpstreamClosingStreamingResponse(
        gen(),
        media_type="text/event-stream",
        upstream_generator=upstream_generator,
    )


@pytest.mark.asyncio
async def test_ttft_keepalive_fills_the_wire_while_the_upstream_is_still_silent():
    """Regression for #34819. The upstream withholds its headers until the first
    token, so the whole wait happens before a byte can be written and an
    idle-timeout hop drops a healthy connection."""

    async def slow_upstream():
        await asyncio.sleep(0.35)
        return _sse_response(['data: {"first": true}\n\n'])

    response = await open_sse_before_first_byte(slow_upstream(), ping_interval_seconds=0.05)

    assert isinstance(response, StreamingResponse)
    assert response.headers["x-accel-buffering"] == "no"
    collected = await _drain(response)
    assert collected[0] == TTFT_PING
    assert collected.count(TTFT_PING) >= 3
    assert collected[-1] == b'data: {"first": true}\n\n'


@pytest.mark.asyncio
async def test_ttft_keepalive_is_a_no_op_when_the_upstream_answers_in_time():
    produced = _sse_response(['data: {"fast": true}\n\n'])

    async def fast_upstream():
        return produced

    response = await open_sse_before_first_byte(fast_upstream(), ping_interval_seconds=5.0)

    assert response is produced
    assert await _drain(response) == ['data: {"fast": true}\n\n']


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [None, 0, "", "abc", float("inf"), float("nan"), -1])
async def test_ttft_keepalive_unconfigured_leaves_the_call_completely_untouched(interval):
    produced = _sse_response(['data: {"x": 1}\n\n'])
    started_at = asyncio.get_running_loop().time()

    async def slow_upstream():
        await asyncio.sleep(0.15)
        return produced

    response = await open_sse_before_first_byte(slow_upstream(), ping_interval_seconds=interval)

    assert response is produced
    assert asyncio.get_running_loop().time() - started_at >= 0.15


@pytest.mark.asyncio
async def test_ttft_keepalive_reraises_a_fast_failure_so_it_keeps_its_http_status():
    async def fast_failure():
        raise HTTPException(status_code=429, detail="rate limited")

    with pytest.raises(HTTPException) as excinfo:
        await open_sse_before_first_byte(fast_failure(), ping_interval_seconds=5.0)

    assert excinfo.value.status_code == 429


@pytest.mark.asyncio
async def test_ttft_keepalive_delivers_a_late_failure_as_an_sse_frame():
    """Once a ping is on the wire the status line is committed, so a failure
    discovered afterwards can only reach the client as a frame."""

    async def slow_failure():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=429, detail="rate limited")

    response = await open_sse_before_first_byte(slow_failure(), ping_interval_seconds=0.05)
    collected = await _drain(response)

    assert collected[0] == TTFT_PING
    assert collected[-1] == b"data: [DONE]\n\n"
    error_frame = json.loads(collected[-2].decode().removeprefix("data: ").strip())
    assert error_frame["error"]["code"] == "429"
    assert error_frame["error"]["message"] == "rate limited"


@pytest.mark.asyncio
async def test_ttft_keepalive_relays_a_late_non_streaming_body_as_an_sse_frame():
    async def slow_json():
        await asyncio.sleep(0.2)
        return JSONResponse(status_code=400, content={"error": {"message": "bad request"}})

    response = await open_sse_before_first_byte(slow_json(), ping_interval_seconds=0.05)
    collected = await _drain(response)

    assert collected[0] == TTFT_PING
    assert json.loads(collected[-2].decode().removeprefix("data: ").strip()) == {"error": {"message": "bad request"}}
    assert collected[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_ttft_keepalive_closes_the_upstream_stream_it_relayed():
    """Starlette never calls the produced response, so its own cleanup never runs
    and the upstream LLM connection would leak."""
    upstream_closed = asyncio.Event()

    async def upstream():
        try:
            yield 'data: {"a": 1}\n\n'
        finally:
            upstream_closed.set()

    upstream_gen = upstream()
    # Started, as create_response leaves it: aclose() on a never-started generator
    # skips its body, so an unstarted fixture cannot tell cleanup from no cleanup.
    await upstream_gen.__anext__()

    async def slow_upstream():
        await asyncio.sleep(0.2)
        return _sse_response(['data: {"a": 1}\n\n'], upstream_generator=upstream_gen)

    response = await open_sse_before_first_byte(slow_upstream(), ping_interval_seconds=0.05)
    await _drain(response)

    assert upstream_closed.is_set()


@pytest.mark.asyncio
async def test_ttft_keepalive_cancels_the_in_flight_call_when_the_client_gives_up():
    upstream_cancelled = asyncio.Event()

    async def never_answers():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            upstream_cancelled.set()
            raise

    response = await open_sse_before_first_byte(never_answers(), ping_interval_seconds=0.05)
    assert await response.body_iterator.__anext__() == TTFT_PING
    await response.body_iterator.aclose()
    await asyncio.sleep(0)

    assert upstream_cancelled.is_set()


@pytest.mark.parametrize(
    "request_data, global_interval, expected",
    [
        ({"stream": True}, 30.0, 30.0),
        ({"stream": True}, None, None),
        ({"stream": False}, 30.0, None),
        ({}, 30.0, None),
        ({"stream": "true"}, 30.0, None),
    ],
)
def test_ttft_keepalive_interval_only_arms_for_a_streaming_request(request_data, global_interval, expected):
    with patch.object(litellm, "sse_keepalive_ping_interval_seconds", global_interval):
        assert ttft_keepalive_interval(request_data) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_requested, expect_ping", [(True, True), (False, False)])
async def test_base_process_llm_request_pings_while_the_upstream_call_is_still_running(
    stream_requested, expect_ping
):
    """The wiring, not the helper: every route funnels through this method, and the
    whole time-to-first-token is spent inside the call it wraps."""

    async def slow_inner(self, **kwargs):
        await asyncio.sleep(0.25)
        return _sse_response(['data: {"late": true}\n\n'])

    processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-4o", "stream": stream_requested})

    with patch.object(litellm, "sse_keepalive_ping_interval_seconds", 0.05):
        with patch.object(ProxyBaseLLMRequestProcessing, "_process_llm_request", slow_inner):
            response = await processor.base_process_llm_request(
                request=MagicMock(spec=Request),
                fastapi_response=Response(),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                route_type="acompletion",
                proxy_logging_obj=MagicMock(spec=ProxyLogging),
                general_settings={},
                proxy_config=MagicMock(spec=ProxyConfig),
            )

    collected = await _drain(response)
    assert (collected[0] == TTFT_PING) is expect_ping
    assert collected[-1] == (b'data: {"late": true}\n\n' if expect_ping else 'data: {"late": true}\n\n')


def _request_disconnecting_after(delay_seconds):
    """A Request whose ASGI channel delivers one http.disconnect, then goes quiet."""
    request = MagicMock(spec=Request)
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            await asyncio.Event().wait()
        await asyncio.sleep(delay_seconds)
        delivered["done"] = True
        return {"type": "http.disconnect"}

    request.receive = receive
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disconnect_after, expect_full_delivery",
    [(0.25, False), (999.0, True)],
)
async def test_opening_the_response_early_still_closes_the_upstream_on_disconnect(
    disconnect_after, expect_full_delivery
):
    """Once the response is opened early, create_response's own disconnect
    monitoring runs while Starlette is already serving, so both read the same ASGI
    channel. Whichever observes the disconnect, the upstream LLM stream must close.
    """
    upstream_closed = asyncio.Event()
    delivered = []

    async def upstream():
        try:
            await asyncio.sleep(0.4)
            for chunk in ('data: {"a": 1}\n\n', "data: [DONE]\n\n"):
                delivered.append(chunk)
                yield chunk
        finally:
            upstream_closed.set()

    request = _request_disconnecting_after(disconnect_after)

    async def produce():
        await asyncio.sleep(0.15)
        return await create_response(
            generator=upstream(),
            media_type="text/event-stream",
            headers={},
            request=request,
        )

    response = await open_sse_before_first_byte(produce(), ping_interval_seconds=0.05)
    collected = await _drain(response)
    await asyncio.sleep(0.05)

    assert collected[0] == TTFT_PING
    assert upstream_closed.is_set()
    # The control has to actually deliver, or "the upstream closed" proves nothing.
    assert (delivered == ['data: {"a": 1}\n\n', "data: [DONE]\n\n"]) is expect_full_delivery


@pytest.mark.asyncio
async def test_a_disconnect_after_the_upstream_answered_still_closes_the_response():
    """The upstream can answer while nobody is draining the relay, e.g. the client
    vanished first. Nothing else holds that response, so only this teardown closes
    it; cancelling the produce task is not enough because it already finished."""
    upstream_closed = asyncio.Event()
    body_closed = asyncio.Event()

    async def upstream():
        try:
            yield 'data: {"a": 1}\n\n'
            await asyncio.Event().wait()
        finally:
            upstream_closed.set()

    async def body():
        try:
            yield 'data: {"a": 1}\n\n'
            await asyncio.Event().wait()
        finally:
            body_closed.set()

    # Both started, as create_response leaves them: aclose() on a never-started
    # generator skips its body, so an unstarted fixture cannot tell cleanup apart
    # from no cleanup at all.
    upstream_gen, body_gen = upstream(), body()
    await upstream_gen.__anext__()
    await body_gen.__anext__()

    async def produce():
        await asyncio.sleep(0.15)
        return _UpstreamClosingStreamingResponse(
            body_gen, media_type="text/event-stream", upstream_generator=upstream_gen
        )

    response = await open_sse_before_first_byte(produce(), ping_interval_seconds=0.05)
    assert await response.body_iterator.__anext__() == TTFT_PING
    await asyncio.sleep(0.25)  # the produce task finishes while nothing is pulling
    await response.body_iterator.aclose()
    await asyncio.sleep(0.05)

    assert body_closed.is_set()
    assert upstream_closed.is_set()


@pytest.mark.asyncio
async def test_a_late_failure_is_reported_to_the_failure_hook():
    """Once a keepalive is on the wire this can no longer raise, so the caller's
    own `except` never runs and the failure would otherwise go unaudited."""
    audited = []

    async def slow_failure():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=500, detail="upstream exploded")

    async def record(exc):
        audited.append(exc)

    response = await open_sse_before_first_byte(
        slow_failure(), ping_interval_seconds=0.05, on_late_failure=record
    )
    collected = await _drain(response)

    assert [type(exc).__name__ for exc in audited] == ["HTTPException"]
    assert getattr(audited[0], "detail", None) == "upstream exploded"
    assert collected[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_a_failing_audit_hook_never_costs_the_client_its_error_frame():
    async def slow_failure():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=500, detail="upstream exploded")

    async def broken_hook(exc):
        raise RuntimeError("the audit backend is down")

    response = await open_sse_before_first_byte(
        slow_failure(), ping_interval_seconds=0.05, on_late_failure=broken_hook
    )
    collected = await _drain(response)

    error_frame = json.loads(collected[-2].decode().removeprefix("data: ").strip())
    assert error_frame["error"]["message"] == "upstream exploded"
    assert collected[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_base_process_llm_request_audits_a_failure_that_lands_after_its_keepalive():
    """The helper honouring on_late_failure is not enough: this pins that the shared
    funnel actually passes one, which is where the route's own except would have
    fired before the response was opened early."""

    async def slow_failure(self, **kwargs):
        await asyncio.sleep(0.25)
        raise HTTPException(status_code=503, detail="upstream exploded")

    proxy_logging_obj = MagicMock(spec=ProxyLogging)
    # None is what a hook that only audits returns; a bare AsyncMock would hand
    # back a MagicMock, which the code correctly reads as a sanitized replacement.
    proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-4o", "stream": True})

    with patch.object(litellm, "sse_keepalive_ping_interval_seconds", 0.05):
        with patch.object(ProxyBaseLLMRequestProcessing, "_process_llm_request", slow_failure):
            response = await processor.base_process_llm_request(
                request=MagicMock(spec=Request),
                fastapi_response=Response(),
                user_api_key_dict=user_api_key_dict,
                route_type="acompletion",
                proxy_logging_obj=proxy_logging_obj,
                general_settings={},
                proxy_config=MagicMock(spec=ProxyConfig),
            )
        collected = await _drain(response)

    proxy_logging_obj.post_call_failure_hook.assert_awaited_once()
    call = proxy_logging_obj.post_call_failure_hook.await_args.kwargs
    assert call["user_api_key_dict"] is user_api_key_dict
    assert call["request_data"] is processor.data
    assert getattr(call["original_exception"], "detail", None) == "upstream exploded"

    assert collected[0] == TTFT_PING
    assert collected[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment_keepalive, expect_ping",
    [(0, False), (None, True)],
    ids=["operator-hard-disabled-this-deployment", "deployment-says-nothing"],
)
async def test_base_process_llm_request_honours_a_deployment_hard_disable(
    deployment_keepalive, expect_ping
):
    """`keepalive_seconds: 0` is documented as a disable a request cannot lift. The
    funnel has to hand its router to the gate for that to hold before the upstream
    has answered, since no deployment has served the request yet."""
    params = {"model": "openai/gpt-4o"}
    if deployment_keepalive is not None:
        params["keepalive_seconds"] = deployment_keepalive

    llm_router = MagicMock()
    llm_router.get_model_list = MagicMock(return_value=[{"model_name": "m", "litellm_params": params}])

    async def slow_inner(self, **kwargs):
        await asyncio.sleep(0.25)
        return _sse_response(['data: {"late": true}\n\n'])

    processor = ProxyBaseLLMRequestProcessing(data={"model": "m", "stream": True})

    with patch.object(litellm, "sse_keepalive_ping_interval_seconds", 0.05):
        with patch.object(ProxyBaseLLMRequestProcessing, "_process_llm_request", slow_inner):
            response = await processor.base_process_llm_request(
                request=MagicMock(spec=Request),
                fastapi_response=Response(),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
                route_type="acompletion",
                proxy_logging_obj=MagicMock(spec=ProxyLogging),
                general_settings={},
                proxy_config=MagicMock(spec=ProxyConfig),
                llm_router=llm_router,
            )

    collected = await _drain(response)
    assert (collected[0] == TTFT_PING) is expect_ping


@pytest.mark.asyncio
async def test_a_hook_returning_a_replacement_decides_what_the_client_sees():
    """post_call_failure_hook exists partly to sanitize client-facing errors.
    Serializing the original would leak provider detail a deployment configured away."""

    async def slow_failure():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=500, detail="upstream said host=10.0.0.7 key=sk-internal")

    async def sanitize(exc):
        return HTTPException(status_code=502, detail="upstream unavailable")

    response = await open_sse_before_first_byte(
        slow_failure(), ping_interval_seconds=0.05, on_late_failure=sanitize
    )
    collected = await _drain(response)

    error_frame = json.loads(collected[-2].decode().removeprefix("data: ").strip())
    assert error_frame["error"]["message"] == "upstream unavailable"
    assert "sk-internal" not in collected[-2].decode()


@pytest.mark.asyncio
async def test_a_hook_raising_a_replacement_also_decides_what_the_client_sees():
    """The hook's contract is return *or* raise, and raising is the path a
    suppress(Exception) around the call would silently discard."""

    async def slow_failure():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=500, detail="upstream said host=10.0.0.7 key=sk-internal")

    async def sanitize_by_raising(exc):
        raise HTTPException(status_code=403, detail="blocked by policy")

    response = await open_sse_before_first_byte(
        slow_failure(), ping_interval_seconds=0.05, on_late_failure=sanitize_by_raising
    )
    collected = await _drain(response)

    error_frame = json.loads(collected[-2].decode().removeprefix("data: ").strip())
    assert error_frame["error"]["message"] == "blocked by policy"
    assert "sk-internal" not in collected[-2].decode()


@pytest.mark.asyncio
async def test_a_hook_that_returns_nothing_leaves_the_real_error_intact():
    async def slow_failure():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=429, detail="rate limited")

    async def audit_only(exc):
        return None

    response = await open_sse_before_first_byte(
        slow_failure(), ping_interval_seconds=0.05, on_late_failure=audit_only
    )
    collected = await _drain(response)

    error_frame = json.loads(collected[-2].decode().removeprefix("data: ").strip())
    assert error_frame["error"]["message"] == "rate limited"
    assert error_frame["error"]["code"] == "429"


@pytest.mark.asyncio
async def test_a_broken_hook_does_not_replace_the_real_error_with_its_own_bug():
    async def slow_failure():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code=429, detail="rate limited")

    async def broken_hook(exc):
        raise RuntimeError("the audit backend is down")

    response = await open_sse_before_first_byte(
        slow_failure(), ping_interval_seconds=0.05, on_late_failure=broken_hook
    )
    collected = await _drain(response)

    error_frame = json.loads(collected[-2].decode().removeprefix("data: ").strip())
    assert error_frame["error"]["message"] == "rate limited"
    assert "audit backend" not in collected[-2].decode()


@pytest.mark.parametrize(
    "exc,expect_traceback",
    [
        pytest.param(HTTPException(status_code=400, detail="Invalid model name passed in"), False, id="expected_400"),
        pytest.param(ValueError("unexpected internal error"), True, id="unexpected_error"),
    ],
)
def test_log_llm_api_exception_traceback_only_for_unexpected_errors(exc, expect_traceback, caplog):
    """Regression for LIT-6043: expected 4xx errors log without formatting a
    traceback; unexpected errors keep logger.exception behavior."""
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.common_request_processing import _log_llm_api_exception

    verbose_proxy_logger.propagate = True
    try:
        with caplog.at_level("ERROR", logger="LiteLLM Proxy"):
            try:
                raise exc
            except Exception as raised:
                _log_llm_api_exception(raised)
    finally:
        verbose_proxy_logger.propagate = False

    records = [r for r in caplog.records if "_handle_llm_api_exception(): Exception occured" in r.getMessage()]
    assert len(records) == 1
    assert (records[0].exc_info is not None) is expect_traceback
