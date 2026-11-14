import asyncio
import base64
import json
from collections.abc import Iterator
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.openai_files_endpoints.common_utils import get_model_id_from_unified_batch_id
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import HttpPassThroughEndpointHelpers
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


class TestGeminiPassthroughLoggingHandler:
    """Test the Gemini passthrough logging handler for cost tracking."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.handler = GeminiPassthroughLoggingHandler()

        # Mock Gemini generateContent response
        self.mock_gemini_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! How can I help you today?"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18,
            },
        }

    def _create_mock_httpx_response(self) -> httpx.Response:
        """Create a mock httpx.Response for testing"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(self.mock_gemini_response)
        mock_response.json.return_value = self.mock_gemini_response
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object for testing"""
        mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        return mock_logging_obj

    def _create_passthrough_logging_payload(self) -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload for testing"""
        return PassthroughStandardLoggingPayload(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            request_method="POST",
        )

    def test_is_gemini_route(self):
        """Test that Gemini routes are correctly identified"""
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging,
        )

        handler = PassThroughEndpointLogging()

        # Test generateContent endpoint
        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                custom_llm_provider="gemini",
            )
            is True
        )

        # Test streamGenerateContent endpoint
        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent",
                custom_llm_provider="gemini",
            )
            is True
        )

        # Test non-Gemini endpoint
        assert (
            handler.is_gemini_route(
                "https://api.openai.com/v1/chat/completions",
                custom_llm_provider="openai",
            )
            is False
        )

    def test_extract_model_from_url(self):
        """Test that model is correctly extracted from Gemini URLs"""
        # Test generateContent endpoint
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        )
        assert model == "gemini-1.5-flash"

        # Test streamGenerateContent endpoint
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent"
        )
        assert model == "gemini-1.5-pro"

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
    def test_gemini_passthrough_handler_success(self, mock_get_standard_logging, mock_completion_cost):
        """Test successful cost tracking for Gemini generateContent endpoint"""
        # Arrange
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000045
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()

        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["model"] == "gemini-1.5-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

    @patch("litellm.completion_cost")
    def test_gemini_passthrough_handler_streaming(self, mock_completion_cost):
        """Test cost tracking for Gemini streaming endpoint"""
        # Arrange
        mock_completion_cost.return_value = 0.000030

        # Mock streaming response chunks
        mock_chunks = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " there!"}]}}]},
        ]

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        # Act - Use generateContent URL since that's what the handler processes
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_chunks,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000030
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()

    def test_gemini_passthrough_handler_non_gemini_route(self):
        """Test that non-Gemini routes return None"""
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",  # Non-Gemini route (no generateContent)
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            **kwargs,
        )

        # Assert - the handler should return a dict with None result for non-Gemini routes
        assert result is not None
        assert result["result"] is None
        assert "kwargs" in result

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler.litellm.completion_cost",
        return_value=0.000050,
    )
    async def test_pass_through_success_handler_gemini_routing(self, mock_completion_cost):
        """Test that the success handler correctly routes Gemini requests to the Gemini handler"""
        handler = PassThroughEndpointLogging()

        # Mock the logging object
        mock_logging_obj = self._create_mock_logging_obj()

        # Mock the _handle_logging method to capture the call
        handler.handle_logging = AsyncMock()

        # Mock httpx response
        mock_response = self._create_mock_httpx_response()

        # Create passthrough logging payload
        passthrough_logging_payload = self._create_passthrough_logging_payload()

        # Call the success handler with Gemini route and provider
        result = await handler.pass_through_async_success_handler(
            httpx_response=mock_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            passthrough_logging_payload=passthrough_logging_payload,
            custom_llm_provider="gemini",
        )

        # Assert - The success handler returns None on success (following the pattern from other tests)
        assert result is None

        assert mock_logging_obj.model_call_details["model"] == "gemini-2.0-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

        # Verify that _handle_logging was called with the correct kwargs
        handler.handle_logging.assert_called_once()
        call_kwargs = handler.handle_logging.call_args[1]
        assert call_kwargs["response_cost"] == 0.000050
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["custom_llm_provider"] == "gemini"

    @patch("litellm.completion_cost")
    def test_veo3_passthrough_cost_tracking(self, mock_completion_cost):
        """Test Veo3 video generation cost tracking for passthrough requests"""
        # Mock the completion_cost to return the expected video generation cost
        # For veo-2.0-generate-001 with 8 seconds: 0.35 * 8 = 2.8
        expected_cost = 0.35 * 8.0  # $2.80
        mock_completion_cost.return_value = expected_cost

        # Mock Veo3 predictLongRunning response
        mock_veo_response = {"name": "operations/1234567890123456789"}

        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.status_code = 200
        mock_httpx_response.json.return_value = mock_veo_response
        mock_httpx_response.headers = {"content-type": "application/json"}

        mock_logging_obj = self._create_mock_logging_obj()

        # Request body with durationSeconds
        request_body = {
            "instances": [{"prompt": "A close up of two people staring at a cryptic drawing on a wall,"}],
            "parameters": {"durationSeconds": 8},
        }

        kwargs = {
            "passthrough_logging_payload": PassthroughStandardLoggingPayload(
                url="https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predictLongRunning",
                request_body=request_body,
                request_method="POST",
            ),
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_veo_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predictLongRunning",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result

        # Verify the cost is calculated correctly
        assert result["kwargs"]["response_cost"] == expected_cost
        assert result["kwargs"]["model"] == "veo-2.0-generate-001"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify completion_cost was called with create_video call_type
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args.kwargs.get("call_type") == "create_video"
        assert call_args.kwargs.get("custom_llm_provider") == "gemini"
        assert call_args.kwargs.get("model") == "veo-2.0-generate-001"

        # Verify the response object has _hidden_params with response_cost
        video_response = result["result"]
        assert hasattr(video_response, "_hidden_params")
        assert video_response._hidden_params.get("response_cost") == expected_cost

        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == expected_cost
        assert mock_logging_obj.model_call_details["model"] == "veo-2.0-generate-001"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

    def test_interactions_create_response_is_priced_as_gemini(self):
        """Regression for LIT-6896: Gemini API Interactions passthrough must not log zero usage."""
        usage = {
            "total_tokens": 1030,
            "total_input_tokens": 10,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 10}],
            "total_output_tokens": 1020,
            "output_tokens_by_modality": [
                {"modality": "text", "tokens": 20},
                {"modality": "video", "tokens": 1000},
            ],
            "total_tool_use_tokens": 0,
            "total_thought_tokens": 0,
        }
        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.json.return_value = {
            "id": "interactions/abc",
            "model": "gemini-omni-flash-preview",
            "status": "completed",
            "usage": usage,
        }
        mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.litellm_call_id = "call-6896"

        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_httpx_response.json.return_value,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/interactions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gemini-omni-flash-preview", "input": "make a clip"},
        )

        model_info = litellm.get_model_info(model="gemini-omni-flash-preview", custom_llm_provider="gemini")
        expected_cost = (
            10 * model_info["input_cost_per_token"]
            + 20 * model_info["output_cost_per_token"]
            + 1000 * model_info["output_cost_per_video_token"]
        )
        assert result["result"].id == "call-6896"
        assert result["result"].usage.completion_tokens_details.video_tokens == 1000
        assert result["kwargs"]["response_cost"] == pytest.approx(expected_cost)
        assert result["kwargs"]["custom_llm_provider"] == "gemini"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"


def test_build_complete_streaming_response_assembles_gemini_sse_chunks():
    logging_obj: Final = LiteLLMLoggingObj(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="gemini-stream-call-id",
        function_id="gemini-stream-function-id",
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={}, model="gemini-2.5-flash")
    first_chunk: Final = {"candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}, "index": 0}]}
    last_chunk: Final = {
        "candidates": [
            {"content": {"parts": [{"text": " there!"}], "role": "model"}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8, "totalTokenCount": 18},
    }

    response: Final = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=[f"data: {json.dumps(first_chunk)}", f"data: {json.dumps(last_chunk)}"],
        litellm_logging_obj=logging_obj,
        model="gemini-2.5-flash",
        url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent",
    )

    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "Hello there!"
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (10, 8, 18)


_PROBE_MODEL: Final = "gemini-pricing-probe"
_LIST_INPUT_RATE: Final = 1e-06
_LIST_OUTPUT_RATE: Final = 4e-06
_CUSTOM_RATES: Final = {"input_cost_per_token": 2e-09, "output_cost_per_token": 3e-09}
_CUSTOM_PRICED_DEPLOYMENT: Final = "gemini-custom-priced-deployment"
_PROBE_REPLY: Final = {
    "candidates": [{"content": {"parts": [{"text": "pong"}], "role": "model"}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1, "totalTokenCount": 8},
    "modelVersion": _PROBE_MODEL,
}


@pytest.fixture
def probe_rate_card(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    rates: Final = {
        "input_cost_per_token": _LIST_INPUT_RATE,
        "output_cost_per_token": _LIST_OUTPUT_RATE,
        "litellm_provider": "gemini",
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, f"gemini/{_PROBE_MODEL}", rates)
    monkeypatch.setitem(
        litellm.model_cost, _CUSTOM_PRICED_DEPLOYMENT, {**_CUSTOM_RATES, "litellm_provider": "gemini", "mode": "chat"}
    )
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _cost_the_logger_records_for_gemini(*, stream: bool, model_info: dict[str, object] | None) -> float:
    method: Final = "streamGenerateContent?alt=sse" if stream else "generateContent"
    url_route: Final = f"https://generativelanguage.googleapis.com/v1beta/models/{_PROBE_MODEL}:{method}"
    logging_obj: Final = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "ping"}],
        stream=stream,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": {"model_info": model_info}} if model_info is not None else {}, optional_params={}
    )
    if stream:
        asyncio.run(
            PassThroughStreamingHandler.route_streaming_logging_to_handler(
                litellm_logging_obj=logging_obj,
                passthrough_success_handler_obj=PassThroughEndpointLogging(),
                url_route=url_route,
                request_body={},
                endpoint_type=HttpPassThroughEndpointHelpers.get_endpoint_type(url_route),
                start_time=datetime(2026, 1, 1),
                raw_bytes=[f"data: {json.dumps(_PROBE_REPLY)}\n\n".encode()],
                end_time=datetime(2026, 1, 1),
            )
        )
        return logging_obj.model_call_details["response_cost"]
    response: Final = httpx.Response(200, json=_PROBE_REPLY)
    result: Final = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
        httpx_response=response,
        response_body=_PROBE_REPLY,
        logging_obj=logging_obj,
        url_route=url_route,
        result=response.text,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        cache_hit=False,
        request_body={},
    )
    logging_obj._process_hidden_params_and_response_cost(
        result["result"], datetime(2026, 1, 1), datetime(2026, 1, 1), build_logging_payload=False
    )
    return logging_obj.model_call_details["response_cost"]


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
@pytest.mark.parametrize(
    "model_info, input_rate, output_rate",
    [
        (None, _LIST_INPUT_RATE, _LIST_OUTPUT_RATE),
        (
            {"id": _CUSTOM_PRICED_DEPLOYMENT, **_CUSTOM_RATES},
            _CUSTOM_RATES["input_cost_per_token"],
            _CUSTOM_RATES["output_cost_per_token"],
        ),
    ],
    ids=["list-price", "custom-price"],
)
def test_gemini_passthrough_spend_matches_whether_or_not_the_call_streams(
    probe_rate_card: None,
    stream: bool,
    model_info: dict[str, object] | None,
    input_rate: float,
    output_rate: float,
) -> None:
    recorded: Final = _cost_the_logger_records_for_gemini(stream=stream, model_info=model_info)

    assert recorded == pytest.approx(7 * input_rate + 1 * output_rate), stream


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
@pytest.mark.parametrize("base_model", [123, [f"gemini/{_PROBE_MODEL}"], {"model": f"gemini/{_PROBE_MODEL}"}])
def test_gemini_passthrough_bills_the_list_price_when_the_model_info_base_model_is_not_a_string(
    probe_rate_card: None, stream: bool, base_model: object
) -> None:
    recorded: Final = _cost_the_logger_records_for_gemini(stream=stream, model_info={"base_model": base_model})

    assert recorded == pytest.approx(7 * _LIST_INPUT_RATE + 1 * _LIST_OUTPUT_RATE), stream


_BATCH_OPERATION: Final = {
    "name": "batches/wire7h3x9batch",
    "metadata": {
        "@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch",
        "model": "models/gemini-3.8-flash",
        "inputConfig": {"fileName": "files/wireinput123"},
        "createTime": "2026-11-11T23:15:29.000000Z",
        "updateTime": "2026-11-11T23:15:29.000000Z",
        "state": "BATCH_STATE_PENDING",
    },
}
_BATCH_ROUTE: Final = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:batchGenerateContent"


def _decoded(unified_object_id: str) -> str:
    return base64.urlsafe_b64decode(unified_object_id + "=" * (-len(unified_object_id) % 4)).decode()


def _gemini_router(model: str) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {"model_name": model, "litellm_params": {"model": f"vertex_ai/{model}", "vertex_project": "p"}},
            {
                "model_name": model,
                "litellm_params": {"model": f"gemini/{model}", "api_key": "scripted"},
                "model_info": {"id": "gemini-deployment-1"},
            },
        ]
    )


def _batch_create_through_passthrough(
    status_code: int, body: dict[str, object], router: litellm.Router | None
) -> tuple[PassThroughEndpointLoggingTypedDict, MagicMock]:
    hook = MagicMock()
    hook.store_unified_object_id = AsyncMock()
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.model_call_details = {}
    kwargs = {"litellm_params": {"metadata": {"user_api_key": "hashed-key", "user_api_key_team_id": "team-3"}}}
    response = httpx.Response(status_code, json=body, request=httpx.Request("POST", _BATCH_ROUTE))

    async def run():
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as proxy_logging,
            patch("litellm.proxy.proxy_server.llm_router", router),
        ):
            proxy_logging.get_proxy_hook.return_value = hook
            outcome = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
                httpx_response=response,
                response_body=body,
                logging_obj=logging_obj,
                url_route=_BATCH_ROUTE,
                result="",
                start_time=datetime(2026, 11, 11, 23, 15, 29),
                end_time=datetime(2026, 11, 11, 23, 15, 30),
                cache_hit=False,
                request_body={"batch": {"inputConfig": {"fileName": "files/wireinput123"}}},
                **kwargs,
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return outcome

    return asyncio.run(run()), hook


def test_batch_generate_content_is_a_tracked_gemini_route():
    assert PassThroughEndpointLogging().is_gemini_route(_BATCH_ROUTE, "gemini") is True


def test_a_created_gemini_batch_is_registered_against_the_gemini_deployment_at_zero_cost():
    outcome, hook = _batch_create_through_passthrough(200, _BATCH_OPERATION, _gemini_router("gemini-3.8-flash"))

    stored = hook.store_unified_object_id.await_args.kwargs
    assert stored["model_object_id"] == "wire7h3x9batch"
    assert get_model_id_from_unified_batch_id(_decoded(stored["unified_object_id"])) == "gemini-deployment-1"
    assert stored["file_purpose"] == "batch"
    assert stored["file_object"].status == "validating"
    assert stored["user_api_key_dict"].api_key == "hashed-key"
    assert stored["user_api_key_dict"].team_id == "team-3"
    assert stored["unified_object_id"] == outcome["kwargs"]["unified_object_id"]
    assert outcome["kwargs"]["model"] == "gemini-3.8-flash"
    assert outcome["kwargs"]["custom_llm_provider"] == "gemini"
    assert outcome["kwargs"]["batch_id"] == "wire7h3x9batch"
    assert outcome["kwargs"]["response_cost"] == 0.0
    assert outcome["result"].object == "batch"


def test_a_created_gemini_batch_without_a_gemini_deployment_registers_nothing_but_names_the_model():
    outcome, hook = _batch_create_through_passthrough(200, _BATCH_OPERATION, _gemini_router("gemini-other"))

    hook.store_unified_object_id.assert_not_awaited()
    assert outcome["result"] is None
    assert outcome["kwargs"]["model"] == "gemini-3.8-flash"
    assert outcome["kwargs"]["batch_id"] == "wire7h3x9batch"


def test_a_rejected_gemini_batch_create_registers_nothing():
    outcome, hook = _batch_create_through_passthrough(
        400, {"error": {"code": 400, "message": "bad file"}}, _gemini_router("gemini-3.8-flash")
    )

    hook.store_unified_object_id.assert_not_awaited()
    assert outcome["result"] is None
    assert outcome["kwargs"]["model"] == "gemini-3.8-flash"
    assert "batch_id" not in outcome["kwargs"]
