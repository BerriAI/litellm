import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)


class TestAnthropicLoggingHandlerModelFallback:
    """Test the model fallback logic in the anthropic passthrough logging handler."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.mock_chunks = [
            '{"type": "message_start", "message": {"id": "msg_123", "model": "claude-3-haiku-20240307"}}',
            '{"type": "content_block_delta", "delta": {"text": "Hello"}}',
            '{"type": "content_block_delta", "delta": {"text": " world"}}',
            '{"type": "message_stop"}',
        ]

    def _create_mock_logging_obj(
        self, model_in_details: str = None
    ) -> LiteLLMLoggingObj:
        """Create a mock logging object with optional model in model_call_details"""
        mock_logging_obj = MagicMock()

        if model_in_details:
            # Create a dict-like mock that returns the model for the 'model' key
            mock_model_call_details = {"model": model_in_details}
            mock_logging_obj.model_call_details = mock_model_call_details
        else:
            # Create empty dict or None
            mock_logging_obj.model_call_details = {}

        return mock_logging_obj

    def _create_mock_passthrough_handler(self):
        """Create a mock passthrough success handler"""
        mock_handler = MagicMock()
        return mock_handler

    @patch.object(
        AnthropicPassthroughLoggingHandler, "_build_complete_streaming_response"
    )
    @patch.object(
        AnthropicPassthroughLoggingHandler, "_create_anthropic_response_logging_payload"
    )
    def test_model_from_request_body_used_when_present(
        self, mock_create_payload, mock_build_response
    ):
        """Test that model from request_body is used when present"""
        # Arrange
        request_body = {"model": "claude-3-sonnet-20240229"}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-haiku-20240307"
        )
        passthrough_handler = self._create_mock_passthrough_handler()

        # Mock successful response building
        mock_build_response.return_value = MagicMock()
        mock_create_payload.return_value = {"test": "payload"}

        # Act
        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=passthrough_handler,
            url_route="/anthropic/v1/messages",
            request_body=request_body,
            endpoint_type="messages",
            start_time=self.start_time,
            all_chunks=self.mock_chunks,
            end_time=self.end_time,
        )

        # Assert
        assert result is not None
        # Verify that _build_complete_streaming_response was called with the request_body model
        mock_build_response.assert_called_once()
        call_args = mock_build_response.call_args
        assert (
            call_args[1]["model"] == "claude-3-sonnet-20240229"
        )  # Should use request_body model

    def test_model_fallback_logic_isolated(self):
        """Test just the model fallback logic in isolation"""
        # Test case 1: Model from request body
        request_body = {"model": "claude-3-sonnet-20240229"}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-haiku-20240307"
        )

        # Extract the logic directly from the function
        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == "claude-3-sonnet-20240229"  # Should use request_body model

        # Test case 2: Fallback to logging obj
        request_body = {}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-haiku-20240307"
        )

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == "claude-3-haiku-20240307"  # Should use fallback model

        # Test case 3: Empty string in request body, fallback to logging obj
        request_body = {"model": ""}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-opus-20240229"
        )

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == "claude-3-opus-20240229"  # Should use fallback model

        # Test case 4: Both empty
        request_body = {}
        logging_obj = self._create_mock_logging_obj()

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == ""  # Should be empty

    def test_edge_case_missing_model_call_details_attribute(self):
        """Test fallback behavior when logging_obj doesn't have model_call_details attribute"""
        # Case where logging_obj doesn't have the attribute at all
        request_body = {"model": ""}  # Empty model in request body
        logging_obj = MagicMock()
        # Remove the attribute to simulate it not existing
        if hasattr(logging_obj, "model_call_details"):
            delattr(logging_obj, "model_call_details")

        # Extract the logic directly from the function
        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == ""  # Should remain empty since no fallback available

        # Case where model_call_details exists but get returns None
        request_body = {"model": ""}
        logging_obj = self._create_mock_logging_obj()  # Empty dict

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == ""  # Should remain empty


class TestAzureAnthropicCostCalculation:
    """Test the custom_llm_provider cost calculation logic for Azure AI Anthropic."""

    def _create_mock_logging_obj(
        self, model: str = None, custom_llm_provider: str = None
    ) -> LiteLLMLoggingObj:
        """Create a mock logging object with optional model and custom_llm_provider"""
        mock_logging_obj = MagicMock()
        mock_model_call_details = {}
        if model:
            mock_model_call_details["model"] = model
        if custom_llm_provider:
            mock_model_call_details["custom_llm_provider"] = custom_llm_provider
        mock_logging_obj.model_call_details = mock_model_call_details
        mock_logging_obj.litellm_call_id = "test-call-id"
        return mock_logging_obj

    @patch("litellm.completion_cost")
    def test_cost_calculation_with_azure_ai_custom_llm_provider(
        self, mock_completion_cost
    ):
        """Test that custom_llm_provider is passed to completion_cost for Azure AI Anthropic"""
        from litellm.types.utils import ModelResponse
        from datetime import datetime

        mock_completion_cost.return_value = 0.001

        logging_obj = self._create_mock_logging_obj(
            model="claude-sonnet-4-5_gb_20250929", custom_llm_provider="azure_ai"
        )

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.id = "test-id"
        mock_response.model = "claude-sonnet-4-5_gb_20250929"

        kwargs = {}
        start_time = datetime.now()
        end_time = datetime.now()

        AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=mock_response,
            model="claude-sonnet-4-5_gb_20250929",
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        # Verify completion_cost was called with the correct parameters
        mock_completion_cost.assert_called_once()
        call_kwargs = mock_completion_cost.call_args[1]
        assert call_kwargs["model"] == "azure_ai/claude-sonnet-4-5_gb_20250929"
        assert call_kwargs["custom_llm_provider"] == "azure_ai"

    @patch("litellm.completion_cost")
    def test_cost_calculation_without_custom_llm_provider(self, mock_completion_cost):
        """Test that cost calculation works without custom_llm_provider (standard Anthropic)"""
        from litellm.types.utils import ModelResponse
        from datetime import datetime

        mock_completion_cost.return_value = 0.001

        # No custom_llm_provider in model_call_details
        logging_obj = self._create_mock_logging_obj(model="claude-3-sonnet-20240229")

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.id = "test-id"
        mock_response.model = "claude-3-sonnet-20240229"

        kwargs = {}
        start_time = datetime.now()
        end_time = datetime.now()

        AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=mock_response,
            model="claude-3-sonnet-20240229",
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        # Verify completion_cost was called without provider prefix
        mock_completion_cost.assert_called_once()
        call_kwargs = mock_completion_cost.call_args[1]
        assert call_kwargs["model"] == "claude-3-sonnet-20240229"
        assert call_kwargs["custom_llm_provider"] is None

    @patch("litellm.completion_cost")
    def test_cost_calculation_does_not_duplicate_provider_prefix(
        self, mock_completion_cost
    ):
        """Test that provider prefix is not duplicated if already present in model name"""
        from litellm.types.utils import ModelResponse
        from datetime import datetime

        mock_completion_cost.return_value = 0.001

        logging_obj = self._create_mock_logging_obj(
            model="azure_ai/claude-sonnet-4-5_gb_20250929",
            custom_llm_provider="azure_ai",
        )

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.id = "test-id"
        mock_response.model = "azure_ai/claude-sonnet-4-5_gb_20250929"

        kwargs = {}
        start_time = datetime.now()
        end_time = datetime.now()

        # Model already has the provider prefix
        AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=mock_response,
            model="azure_ai/claude-sonnet-4-5_gb_20250929",
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        # Verify provider prefix was not duplicated
        mock_completion_cost.assert_called_once()
        call_kwargs = mock_completion_cost.call_args[1]
        assert call_kwargs["model"] == "azure_ai/claude-sonnet-4-5_gb_20250929"
        assert call_kwargs["custom_llm_provider"] == "azure_ai"

    @patch("litellm.completion_cost")
    def test_cost_calculation_resolves_unknown_model_from_litellm_params(
        self, mock_completion_cost
    ):
        """When the body model is the "unknown" sentinel, the deployment model
        from litellm_params must be used for costing, not "unknown" (which makes
        completion_cost raise and the cost silently fall back to $0)."""
        from datetime import datetime

        from litellm.types.utils import ModelResponse

        mock_completion_cost.return_value = 0.001

        logging_obj = self._create_mock_logging_obj(model="unknown")
        logging_obj.model_call_details["litellm_params"] = {
            "model": "anthropic/claude-3-5-haiku-20241022",
            "metadata": {
                "model_group": "passthrough/anthropic/claude-3-5-haiku-20241022"
            },
        }
        logging_obj.litellm_params = logging_obj.model_call_details["litellm_params"]

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.id = "test-id"
        mock_response.model = "unknown"

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=mock_response,
            model="unknown",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        mock_completion_cost.assert_called_once()
        assert (
            mock_completion_cost.call_args[1]["model"]
            == "anthropic/claude-3-5-haiku-20241022"
        )
        assert kwargs["response_cost"] == 0.001
        assert kwargs["model"] == "anthropic/claude-3-5-haiku-20241022"

    @patch("litellm.completion_cost")
    def test_cost_calculation_resolves_unknown_model_from_model_group(
        self, mock_completion_cost
    ):
        """With only model_group available (no deployment litellm_params.model),
        the leading passthrough/ prefix must be stripped so the cost map can
        resolve the model."""
        from datetime import datetime

        from litellm.types.utils import ModelResponse

        mock_completion_cost.return_value = 0.002

        logging_obj = self._create_mock_logging_obj(model="unknown")
        logging_obj.model_call_details["litellm_params"] = {
            "metadata": {
                "model_group": "passthrough/anthropic/claude-3-5-haiku-20241022"
            }
        }
        logging_obj.litellm_params = logging_obj.model_call_details["litellm_params"]

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.id = "test-id"
        mock_response.model = "unknown"

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=mock_response,
            model="unknown",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        mock_completion_cost.assert_called_once()
        assert (
            mock_completion_cost.call_args[1]["model"]
            == "anthropic/claude-3-5-haiku-20241022"
        )
        assert kwargs["response_cost"] == 0.002

    @patch("litellm.completion_cost")
    def test_cost_calculation_skips_unknown_litellm_params_model_for_model_group(
        self, mock_completion_cost
    ):
        """When litellm_params.model is itself the "unknown" sentinel, the
        deployment-model branch must not short-circuit; resolution falls through
        to model_group so costing still prices the real model instead of "unknown"."""
        from datetime import datetime

        from litellm.types.utils import ModelResponse

        mock_completion_cost.return_value = 0.003

        logging_obj = self._create_mock_logging_obj(model="unknown")
        logging_obj.model_call_details["litellm_params"] = {
            "model": "unknown",
            "metadata": {
                "model_group": "passthrough/anthropic/claude-3-5-haiku-20241022"
            },
        }
        logging_obj.litellm_params = logging_obj.model_call_details["litellm_params"]

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.id = "test-id"
        mock_response.model = "unknown"

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=mock_response,
            model="unknown",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        mock_completion_cost.assert_called_once()
        assert (
            mock_completion_cost.call_args[1]["model"]
            == "anthropic/claude-3-5-haiku-20241022"
        )
        assert kwargs["response_cost"] == 0.003
        assert kwargs["model"] == "anthropic/claude-3-5-haiku-20241022"

    @patch("litellm.completion_cost")
    def test_streaming_cost_calculation_resolves_model_from_message_start_chunk(
        self, mock_completion_cost
    ):
        """On the bare /anthropic passthrough path litellm_params carries no model
        or model_group and the body model is the "unknown" sentinel; the model
        must be recovered from the message_start SSE event so completion_cost
        prices the real model instead of failing on "unknown" and logging $0."""
        from datetime import datetime

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as RealLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        mock_completion_cost.return_value = 0.001

        def _sse(event, data):
            return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

        frames = [
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-5-haiku-20241022",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"},
                },
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            ),
            _sse("message_stop", {"type": "message_stop"}),
        ]
        all_chunks = list(
            PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(frames)
        )

        logging_obj = RealLoggingObj(
            model="unknown",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="pass_through_endpoint",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="1",
        )
        logging_obj.model_call_details["model"] = "unknown"
        logging_obj.model_call_details["stream"] = True
        logging_obj.model_call_details["litellm_params"] = {}
        logging_obj.litellm_params = {}

        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/anthropic/v1/messages",
            request_body={"stream": True},
            endpoint_type="messages",
            start_time=datetime.now(),
            all_chunks=all_chunks,
            end_time=datetime.now(),
        )

        assert result["result"] is not None
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args[1]["model"] == "claude-3-5-haiku-20241022"
        assert result["kwargs"]["response_cost"] == 0.001
        assert result["kwargs"]["model"] == "claude-3-5-haiku-20241022"

    def test_extract_model_skips_non_dict_data_payload(self):
        """A scalar data: payload (e.g. `data: null`) must be skipped, not crash
        the streaming log handler with AttributeError, which would propagate out
        and break spend logging for the whole request."""
        chunks = [
            "event: ping\ndata: null\n\n",
            'event: message_start\ndata: {"type": "message_start", "message": '
            '{"model": "claude-3-5-haiku-20241022"}}\n\n',
        ]

        assert (
            AnthropicPassthroughLoggingHandler._extract_model_from_anthropic_chunks(
                chunks
            )
            == "claude-3-5-haiku-20241022"
        )

    def test_extract_model_parses_per_line_not_first_data_substring(self):
        """A raw multi-line SSE event whose non-data line contains the substring
        "data:" must not derail parsing: matching only lines that start with
        "data:" recovers the message_start model, whereas a first-substring slice
        would consume the wrong offset, fail to parse JSON, and return None."""
        raw_event = (
            "event: ping data: not-json\n"
            'data: {"type": "message_start", "message": '
            '{"model": "claude-3-5-haiku-20241022"}}\n\n'
        )

        assert (
            AnthropicPassthroughLoggingHandler._extract_model_from_anthropic_chunks(
                [raw_event]
            )
            == "claude-3-5-haiku-20241022"
        )

    def test_passthrough_logging_sets_response_cost_with_server_tool_use_dict(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        logging_obj = self._create_mock_logging_obj(model="claude-3-7-sonnet-20250219")
        logging_obj.get_router_model_id.return_value = None
        logging_obj.litellm_params = {}

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="test", role="assistant"),
                )
            ],
            created=1234567890,
            model="claude-3-7-sonnet-20250219",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "server_tool_use": {"web_search_requests": 1},
            },
        )

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=response,
            model="claude-3-7-sonnet-20250219",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        assert "response_cost" in kwargs
        assert kwargs["response_cost"] > 0


class TestAnthropicBatchPassthroughCostTracking:
    """Test cases for Anthropic batch passthrough cost tracking functionality"""

    @pytest.fixture
    def mock_httpx_response(self):
        """Mock httpx response for batch job creation"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msgbatch_01Wj7gkQk7gn4MpAKR8ZEDU2",
            "archived_at": None,
            "cancel_initiated_at": None,
            "created_at": "2024-08-20T18:37:24.100435Z",
            "ended_at": None,
            "expires_at": "2024-08-21T18:37:24.100435Z",
            "processing_status": "in_progress",
            "request_counts": {
                "canceled": 0,
                "errored": 0,
                "expired": 0,
                "processing": 1,
                "succeeded": 0,
            },
            "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_01Wj7gkQk7gn4MpAKR8ZEDU2/results",
            "type": "message_batch",
        }
        return mock_response

    @pytest.fixture
    def mock_logging_obj(self):
        """Mock logging object"""
        mock = MagicMock()
        mock.litellm_call_id = "test-call-id-123"
        mock.model_call_details = {}
        mock.model = None
        return mock

    @pytest.fixture
    def mock_request_body(self):
        """Mock request body for batch creation"""
        return {
            "requests": [
                {
                    "custom_id": "my-custom-id-1",
                    "params": {
                        "max_tokens": 1024,
                        "messages": [{"content": "Hello, world", "role": "user"}],
                        "model": "claude-sonnet-4-5-20250929",
                    },
                }
            ]
        }

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler._store_batch_managed_object"
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router"
    )
    @patch("litellm.llms.anthropic.batches.transformation.AnthropicBatchesConfig")
    def test_batch_creation_handler_success(
        self,
        mock_batches_config,
        mock_get_model_id,
        mock_store_batch,
        mock_httpx_response,
        mock_logging_obj,
        mock_request_body,
    ):
        """Test successful batch creation and managed object storage"""
        from litellm.types.utils import LiteLLMBatch

        # Setup mocks
        mock_get_model_id.return_value = "claude-sonnet-4-5-20250929"

        mock_batch_response = LiteLLMBatch(
            id="msgbatch_01Wj7gkQk7gn4MpAKR8ZEDU2",
            object="batch",
            endpoint="/v1/messages",
            errors=None,
            input_file_id="None",
            completion_window="24h",
            status="validating",
            output_file_id="msgbatch_01Wj7gkQk7gn4MpAKR8ZEDU2",
            error_file_id=None,
            created_at=1704067200,
            in_progress_at=1704067200,
            expires_at=1704153600,
            finalizing_at=None,
            completed_at=None,
            failed_at=None,
            expired_at=None,
            cancelling_at=None,
            cancelled_at=None,
            request_counts={"total": 1, "completed": 0, "failed": 0},
            metadata={},
        )

        mock_batches_config_instance = MagicMock()
        mock_batches_config_instance.transform_retrieve_batch_response.return_value = (
            mock_batch_response
        )
        mock_batches_config.return_value = mock_batches_config_instance

        # Test the handler
        result = AnthropicPassthroughLoggingHandler.batch_creation_handler(
            httpx_response=mock_httpx_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.anthropic.com/v1/messages/batches",
            result="success",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body=mock_request_body,
        )

        # Verify the result
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        # Model should be extracted from request body
        assert result["kwargs"]["model"] == "claude-sonnet-4-5-20250929"
        assert result["kwargs"]["batch_id"] == "msgbatch_01Wj7gkQk7gn4MpAKR8ZEDU2"
        assert result["kwargs"]["batch_job_state"] == "in_progress"
        assert "unified_object_id" in result["kwargs"]

        # Verify batch was stored
        mock_store_batch.assert_called_once()
        call_kwargs = mock_store_batch.call_args[1]
        assert call_kwargs["model_object_id"] == "msgbatch_01Wj7gkQk7gn4MpAKR8ZEDU2"
        assert call_kwargs["batch_object"].status == "validating"

        # Verify the response object
        assert result["result"].model == "claude-sonnet-4-5-20250929"
        assert result["result"].object == "batch"

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler._store_batch_managed_object"
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router"
    )
    def test_batch_creation_handler_model_extraction_from_nested_request(
        self, mock_get_model_id, mock_store_batch, mock_httpx_response, mock_logging_obj
    ):
        """Test that model is correctly extracted from nested request structure"""
        from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig
        from litellm.types.utils import LiteLLMBatch

        # Setup mocks
        mock_get_model_id.return_value = "claude-sonnet-4-5-20250929"

        mock_batch_response = LiteLLMBatch(
            id="msgbatch_123",
            object="batch",
            endpoint="/v1/messages",
            input_file_id="None",
            completion_window="24h",
            status="validating",
            created_at=1704067200,
            request_counts={"total": 1, "completed": 0, "failed": 0},
        )

        with patch.object(
            AnthropicBatchesConfig,
            "transform_retrieve_batch_response",
            return_value=mock_batch_response,
        ):
            # Request body with nested model in requests[0].params.model
            request_body = {
                "requests": [
                    {
                        "custom_id": "test-1",
                        "params": {
                            "model": "claude-sonnet-4-5-20250929",
                            "messages": [{"role": "user", "content": "test"}],
                        },
                    }
                ]
            }

            result = AnthropicPassthroughLoggingHandler.batch_creation_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route="https://api.anthropic.com/v1/messages/batches",
                result="success",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
                request_body=request_body,
            )

            # Verify model was extracted correctly
            assert result["kwargs"]["model"] == "claude-sonnet-4-5-20250929"

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router"
    )
    def test_batch_creation_handler_model_prefix_when_not_in_router(
        self,
        mock_get_model_id,
        mock_httpx_response,
        mock_logging_obj,
        mock_request_body,
    ):
        """Test that model gets 'anthropic/' prefix when not found in router"""
        from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig
        from litellm.types.utils import LiteLLMBatch
        import base64

        # Model not in router - returns same model name
        mock_get_model_id.return_value = "claude-sonnet-4-5-20250929"

        mock_batch_response = LiteLLMBatch(
            id="msgbatch_123",
            object="batch",
            endpoint="/v1/messages",
            input_file_id="None",
            completion_window="24h",
            status="validating",
            created_at=1704067200,
            request_counts={"total": 1, "completed": 0, "failed": 0},
        )

        with patch.object(
            AnthropicBatchesConfig,
            "transform_retrieve_batch_response",
            return_value=mock_batch_response,
        ):
            with patch.object(
                AnthropicPassthroughLoggingHandler, "_store_batch_managed_object"
            ):
                result = AnthropicPassthroughLoggingHandler.batch_creation_handler(
                    httpx_response=mock_httpx_response,
                    logging_obj=mock_logging_obj,
                    url_route="https://api.anthropic.com/v1/messages/batches",
                    result="success",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    cache_hit=False,
                    request_body=mock_request_body,
                )

                # Verify unified_object_id contains anthropic/ prefix
                unified_object_id = result["kwargs"]["unified_object_id"]
                decoded = base64.urlsafe_b64decode(unified_object_id + "==").decode()
                assert (
                    "anthropic/claude-sonnet-4-5-20250929" in decoded
                    or "claude-sonnet-4-5-20250929" in decoded
                )

    @pytest.mark.parametrize(
        "kwargs,expected_user_id,expected_team_id",
        [
            (
                {
                    "litellm_params": {
                        "metadata": {
                            "user_api_key_user_id": "real-user-123",
                            "user_api_key_team_id": "team-456",
                        }
                    }
                },
                "real-user-123",
                "team-456",
            ),
            ({}, "default-user", None),
        ],
    )
    def test_store_batch_managed_object_propagates_user_identity_from_metadata(
        self,
        mock_logging_obj,
        kwargs,
        expected_user_id,
        expected_team_id,
    ):
        """The fabricated UserAPIKeyAuth must inherit user_id/team_id from the
        request's litellm_params.metadata, not the (always-empty) top-level
        kwargs lookup. Falls back to "default-user" only when metadata is
        absent."""
        mock_managed_files_hook = MagicMock()
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_pl,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.verbose_proxy_logger"
            ),
        ):
            mock_pl.get_proxy_hook.return_value = mock_managed_files_hook

            AnthropicPassthroughLoggingHandler._store_batch_managed_object(
                unified_object_id="uoi",
                batch_object={"id": "b1", "object": "batch", "status": "validating"},
                model_object_id="b1",
                logging_obj=mock_logging_obj,
                **kwargs,
            )

            mock_managed_files_hook.store_unified_object_id.assert_called_once()
            call_kwargs = mock_managed_files_hook.store_unified_object_id.call_args[1]
            assert call_kwargs["user_api_key_dict"].user_id == expected_user_id
            assert call_kwargs["user_api_key_dict"].team_id == expected_team_id

    def test_batch_creation_handler_failure_status_code(
        self, mock_logging_obj, mock_request_body
    ):
        """Test batch creation handler with non-200 status code"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}

        result = AnthropicPassthroughLoggingHandler.batch_creation_handler(
            httpx_response=mock_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.anthropic.com/v1/messages/batches",
            result="error",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body=mock_request_body,
        )

        # Verify error response
        assert result is not None
        assert result["kwargs"]["batch_job_state"] == "failed"
        assert result["kwargs"]["response_cost"] == 0.0

    @patch("litellm.proxy.proxy_server.proxy_logging_obj")
    def test_store_batch_managed_object_success(
        self, mock_proxy_logging_obj, mock_logging_obj
    ):
        """Test storing batch managed object"""
        from litellm.types.utils import LiteLLMBatch

        # Setup mocks
        mock_managed_files_hook = MagicMock()
        mock_managed_files_hook.store_unified_object_id = AsyncMock()
        mock_proxy_logging_obj.get_proxy_hook.return_value = mock_managed_files_hook

        batch_object = LiteLLMBatch(
            id="msgbatch_123",
            object="batch",
            endpoint="/v1/messages",
            input_file_id="None",
            completion_window="24h",
            status="validating",
            created_at=1704067200,
            request_counts={"total": 1, "completed": 0, "failed": 0},
        )

        with patch("asyncio.create_task"):
            AnthropicPassthroughLoggingHandler._store_batch_managed_object(
                unified_object_id="test-unified-id",
                batch_object=batch_object,
                model_object_id="msgbatch_123",
                logging_obj=mock_logging_obj,
                user_id="test-user",
            )

            # Verify managed files hook was called
            mock_proxy_logging_obj.get_proxy_hook.assert_called_once_with(
                "managed_files"
            )


class TestBuildCompleteStreamingResponseRobustness:
    """_build_complete_streaming_response must tolerate non-standard SSE frames."""

    def _build(self, chunks: List[str]):
        return AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=chunks,
            litellm_logging_obj=MagicMock(),
            model="claude-3-sonnet-20240229",
        )

    def test_done_frame_is_skipped(self):
        """A bare 'data: [DONE]' control frame must not break reconstruction."""
        chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-sonnet-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
            "data: [DONE]",
        ]
        result = self._build(chunks)
        assert result is not None
        assert result.choices[0].message.content == "Hi"

    def test_non_json_sse_line_is_skipped(self):
        """Non-JSON SSE lines (comments, keep-alive pings) must be skipped."""
        chunks = [
            ": ping",
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-sonnet-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
            "this is not json at all",
        ]
        # Must not raise; a malformed stream simply yields no usable response.
        result = self._build(chunks)
        assert result is None or hasattr(result, "choices")

    def test_mixed_valid_and_invalid_frames(self):
        """Valid events are still collected when interleaved with invalid ones."""
        chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-sonnet-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
            "data: [DONE]",
            ": keep-alive",
            "not-json",
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]
        result = self._build(chunks)
        assert result is not None
        assert result.choices[0].message.content == "Hello"

    def test_done_in_text_payload_is_not_dropped(self):
        """A valid event whose text content contains '[DONE]' must NOT be skipped."""
        chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-sonnet-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The stream ends with [DONE]"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":8}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]
        result = self._build(chunks)
        assert result is not None
        assert result.choices[0].message.content == "The stream ends with [DONE]"


class TestPureTextFastPathParity:
    """
    The pure-text fast path in _build_complete_streaming_response must produce
    a response (and downstream logging/cost payload) byte-identical to the
    legacy stream_chunk_builder path. Anything non-text must fall back.
    """

    @staticmethod
    def _sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    @staticmethod
    def _to_all_chunks(raw_frames):
        # Mirror production: raw bytes -> _convert_raw_bytes_to_str_lines.
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        return PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_frames)

    @staticmethod
    def _norm(resp):
        if resp is None:
            return None
        d = resp.model_dump()
        # id / created are non-deterministic even between two legacy runs.
        d.pop("id", None)
        d.pop("created", None)
        return d

    def _text_stream(
        self,
        texts,
        *,
        input_tokens=12,
        cache_creation=0,
        cache_read=0,
        stop_reason="end_turn",
        with_ping=True,
        blocks=1,
    ):
        frames = [
            self._sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_abc",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-5-sonnet-20241022",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 0,
                            "cache_creation_input_tokens": cache_creation,
                            "cache_read_input_tokens": cache_read,
                        },
                    },
                },
            )
        ]
        per_block = max(1, len(texts) // blocks)
        ti = 0
        for b in range(blocks):
            frames.append(
                self._sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": b,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            if with_ping:
                frames.append(self._sse("ping", {"type": "ping"}))
            chunk_texts = texts[ti : ti + per_block] if b < blocks - 1 else texts[ti:]
            ti += per_block
            for t in chunk_texts:
                frames.append(
                    self._sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": b,
                            "delta": {"type": "text_delta", "text": t},
                        },
                    )
                )
            frames.append(
                self._sse(
                    "content_block_stop", {"type": "content_block_stop", "index": b}
                )
            )
        frames.append(
            self._sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": len(texts)},
                },
            )
        )
        frames.append(self._sse("message_stop", {"type": "message_stop"}))
        return frames

    def _assert_parity(self, raw_frames):
        all_chunks = self._to_all_chunks(raw_frames)
        lo1 = MagicMock()
        lo1.model_call_details = {}
        lo2 = MagicMock()
        lo2.model_call_details = {}

        legacy = AnthropicPassthroughLoggingHandler._build_complete_streaming_response_legacy(
            all_chunks=list(all_chunks),
            litellm_logging_obj=lo1,
            model="claude-3-5-sonnet-20241022",
        )
        fast = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=list(all_chunks),
            litellm_logging_obj=lo2,
            model="claude-3-5-sonnet-20241022",
        )
        assert self._norm(fast) == self._norm(legacy)

        # Downstream logged/billed payload must also match.
        start = datetime.now()
        end = datetime.now()
        k_legacy = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=legacy,
            model="claude-3-5-sonnet-20241022",
            kwargs={},
            start_time=start,
            end_time=end,
            logging_obj=lo1,
        )
        k_fast = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=fast,
            model="claude-3-5-sonnet-20241022",
            kwargs={},
            start_time=start,
            end_time=end,
            logging_obj=lo2,
        )
        # Usage drives cost; it must be byte-identical between paths.
        assert getattr(fast, "usage", None) == getattr(legacy, "usage", None)

        # And the full logged payload (sans non-deterministic response id).
        def _scrub(p):
            d = dict(p)
            r = d.get("complete_streaming_response_in_db") or d.get(
                "complete_streaming_response"
            )
            return d, getattr(r, "usage", None)

        assert _scrub(k_fast)[1] == _scrub(k_legacy)[1]

    def test_parity_simple_text(self):
        self._assert_parity(self._text_stream(["Hello", " ", "world", "!"]))

    def test_parity_single_delta(self):
        self._assert_parity(self._text_stream(["Just one piece of text."]))

    def test_parity_cache_tokens(self):
        self._assert_parity(
            self._text_stream(
                ["a", "b", "c"], input_tokens=20, cache_creation=5, cache_read=7
            )
        )

    def test_parity_max_tokens_stop(self):
        self._assert_parity(self._text_stream(["tok"] * 8, stop_reason="max_tokens"))

    def test_parity_no_ping(self):
        self._assert_parity(self._text_stream(["x", "y"], with_ping=False))

    def test_parity_empty_text_deltas(self):
        self._assert_parity(self._text_stream(["", "hi", "", "there"]))

    def test_parity_multi_text_block(self):
        self._assert_parity(self._text_stream(["p1", "p2", "p3", "p4"], blocks=2))

    def test_parity_multibyte_batched_frames(self):
        # Several SSE events delivered in one network chunk.
        frames = self._text_stream(["alpha", "beta", "gamma"])
        merged = b"".join(frames)
        self._assert_parity([merged])

    def test_collapse_returns_none_for_tool_use(self):
        frames = [
            self._sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "m",
                        "model": "x",
                        "role": "assistant",
                        "type": "message",
                        "content": [],
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            self._sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "get_weather",
                        "input": {},
                    },
                },
            ),
            self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
            ),
            self._sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            self._sse("message_stop", {"type": "message_stop"}),
        ]
        all_chunks = self._to_all_chunks(frames)
        assert (
            AnthropicPassthroughLoggingHandler._collapse_pure_text_chunks(
                list(all_chunks)
            )
            is None
        )

    def test_collapse_returns_none_for_thinking(self):
        frames = [
            self._sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "m",
                        "model": "x",
                        "role": "assistant",
                        "type": "message",
                        "content": [],
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            self._sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                },
            ),
            self._sse("message_stop", {"type": "message_stop"}),
        ]
        all_chunks = self._to_all_chunks(frames)
        assert (
            AnthropicPassthroughLoggingHandler._collapse_pure_text_chunks(
                list(all_chunks)
            )
            is None
        )

    def test_collapse_actually_shrinks_chunk_count(self):
        frames = self._text_stream(["a"] * 50)
        all_chunks = list(self._to_all_chunks(frames))
        collapsed = AnthropicPassthroughLoggingHandler._collapse_pure_text_chunks(
            all_chunks
        )
        assert collapsed is not None
        # 50 text deltas + 50 event markers + 1 ping collapse to far fewer.
        assert len(collapsed) < len(all_chunks) / 2

    def test_collapse_returns_none_for_interleaved_block_indexes(self):
        """
        Anthropic sends content blocks strictly sequentially (start/deltas/stop
        for one, then the next). If a stream ever interleaves deltas across
        block indexes, the fast path must bail to legacy rather than merge text
        from different blocks under a single index.
        """
        frames = [
            self._sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_abc",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-5-sonnet-20241022",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            self._sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            self._sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            # Interleave: delta for block 0, then delta for block 1, with no
            # content_block_stop between them.
            self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello "},
                },
            ),
            self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "world"},
                },
            ),
            self._sse("message_stop", {"type": "message_stop"}),
        ]
        all_chunks = list(self._to_all_chunks(frames))
        assert (
            AnthropicPassthroughLoggingHandler._collapse_pure_text_chunks(all_chunks)
            is None
        )


class TestInterruptedStreamOutputTokenRecovery:
    """
    When an Anthropic pass-through stream is interrupted (client disconnect)
    before the terminal ``message_delta``, the only usage signal is the
    ``message_start`` ``output_tokens`` placeholder (typically 1-3), so
    completion tokens and spend are undercounted ~20x. The handler must
    re-tokenize the buffered ``content_block_delta`` text to recover a
    realistic ``output_tokens``; completed streams must stay untouched.
    """

    @staticmethod
    def _sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    _MODEL = "claude-3-5-haiku-20241022"
    _OUTPUT_TEXT = (
        "The history of computing spans centuries, beginning with mechanical "
        "calculators and the abacus, advancing through Charles Babbage's "
        "analytical engine, Ada Lovelace's first algorithm, Alan Turing's "
        "theoretical machine, and the electronic computers of the twentieth "
        "century that gave rise to the modern information age."
    )

    def _interrupted_chunks(self, *, placeholder_output_tokens: int = 2):
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        words = self._OUTPUT_TEXT.split(" ")
        frames = [
            self._sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_interrupted",
                        "type": "message",
                        "role": "assistant",
                        "model": self._MODEL,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 29,
                            "output_tokens": placeholder_output_tokens,
                        },
                    },
                },
            ),
            self._sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            frames.append(
                self._sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )
        # Client disconnects here: no content_block_stop / message_delta /
        # message_stop are ever received.
        return list(PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(frames))

    def _completed_chunks(self, *, final_output_tokens: int = 80):
        chunks = self._interrupted_chunks()
        chunks.append(
            "data: "
            + json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": final_output_tokens},
                }
            )
        )
        chunks.append('data: {"type": "message_stop"}')
        return chunks

    def _run(self, all_chunks):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"model": self._MODEL, "stream": True}
        logging_obj.litellm_call_id = "test-call-id"
        logging_obj.litellm_params = {}
        logging_obj.get_router_model_id.return_value = None

        return AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/anthropic/v1/messages",
            request_body={"model": self._MODEL, "stream": True},
            endpoint_type="messages",
            start_time=datetime.now(),
            all_chunks=all_chunks,
            end_time=datetime.now(),
        )

    def test_interrupted_stream_retokenizes_buffered_output(self):
        import litellm

        placeholder = 2
        result = self._run(
            self._interrupted_chunks(placeholder_output_tokens=placeholder)
        )
        usage = result["result"].usage

        expected = litellm.token_counter(
            model=self._MODEL,
            text=self._OUTPUT_TEXT,
            count_response_tokens=True,
        )

        assert expected > placeholder * 5
        assert usage.completion_tokens == expected
        assert usage.completion_tokens > placeholder
        assert usage.total_tokens == usage.prompt_tokens + expected
        # Anthropic spend is priced off completion_tokens_details.text_tokens; if the
        # placeholder leaks through here, cost stays undercounted even though
        # completion_tokens looks right.
        assert usage.completion_tokens_details.text_tokens == expected

    def test_completed_stream_keeps_message_delta_tokens(self):
        final = 80
        result = self._run(self._completed_chunks(final_output_tokens=final))
        usage = result["result"].usage

        # Terminal message_delta present: recovery must not fire; the authoritative
        # provider count is preserved verbatim.
        assert usage.completion_tokens == final


class TestStreamFalseDeduplication:
    """
    Regression tests for the duplicate-callback bug where a streaming pass-through
    request had stream=False hardcoded on its Logging object.

    Before the fix:
    - logging_obj.stream was always False for pass-through requests
    - _is_assembled_stream_success() checked `self.stream is not True` and returned
      False immediately, so has_dispatched_final_stream_success was never set
    - Any second dispatch_success_handlers call went through unchecked

    After the fix:
    - pass_through_endpoints.py sets logging_obj.stream = True after detecting stream
    - _create_anthropic_response_logging_payload sets complete_streaming_response on
      model_call_details so callbacks see the correct assembled response state
    - _is_assembled_stream_success returns True, dedup guard fires on first dispatch
    """

    @staticmethod
    def _sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    @staticmethod
    def _make_logging_obj(stream: bool = False) -> LiteLLMLoggingObj:
        logging_obj = LiteLLMLoggingObj(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "hello"}],
            stream=stream,
            call_type="pass_through_endpoint",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="1245",
        )
        return logging_obj

    @staticmethod
    def _build_chunks():
        frames = [
            TestStreamFalseDeduplication._sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_abc",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-5-sonnet-20241022",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            TestStreamFalseDeduplication._sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            TestStreamFalseDeduplication._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            ),
            TestStreamFalseDeduplication._sse(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            ),
            TestStreamFalseDeduplication._sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            ),
            TestStreamFalseDeduplication._sse("message_stop", {"type": "message_stop"}),
        ]
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        return PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(frames)

    def test_complete_streaming_response_set_on_model_call_details(self):
        """
        After the fix, _create_anthropic_response_logging_payload must set
        complete_streaming_response on logging_obj.model_call_details so that
        callbacks like _PROXY_track_cost_callback see the assembled response
        instead of None.

        Before the fix: model_call_details had no complete_streaming_response key.
        The log showed: "kwargs stream: True + complete streaming response: None"
        """
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )

        # pass_through_request sets the stream flag before the streaming handler
        # reconstructs the response; mirror that here.
        logging_obj = self._make_logging_obj(stream=True)
        logging_obj.model_call_details["stream"] = True
        all_chunks = list(self._build_chunks())

        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/anthropic/v1/messages",
            request_body={"model": "claude-3-5-sonnet-20241022", "stream": True},
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            all_chunks=all_chunks,
            end_time=datetime.now(),
        )

        # The assembled response must be stored on model_call_details so callbacks
        # can identify this as a completed streaming call, not an in-progress one.
        assert (
            logging_obj.model_call_details.get("complete_streaming_response")
            is not None
        ), "complete_streaming_response must be set on model_call_details after assembly"

        # The returned result must match what was stored
        assert result["result"] is logging_obj.model_call_details.get(
            "complete_streaming_response"
        )

    def test_dedup_guard_fires_when_stream_true_on_logging_obj(self):
        """
        When logging_obj.stream is True (set by pass_through_endpoints.py after
        detecting a streaming request), dispatch_success_handlers must set
        has_dispatched_final_stream_success=True on the first call so that any
        second call is a no-op.

        This is the _is_assembled_stream_success gate: with stream=False it
        always returned False and the guard was permanently disabled.
        """
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )
        from litellm.types.utils import ModelResponse

        # Simulate what pass_through_endpoints.py now does after stream detection
        logging_obj = self._make_logging_obj(stream=False)
        logging_obj.stream = True  # fix applied
        logging_obj.model_call_details["stream"] = True

        # Simulate what _create_anthropic_response_logging_payload now does
        mock_response = ModelResponse(model="claude-3-5-sonnet-20241022")
        logging_obj.model_call_details["complete_streaming_response"] = mock_response

        assert logging_obj._is_assembled_stream_success(result=mock_response) is True

        # First dispatch sets the flag
        assert not logging_obj.model_call_details.get(
            "has_dispatched_final_stream_success"
        )
        logging_obj.model_call_details["has_dispatched_final_stream_success"] = True

        # Second dispatch would be blocked — simulate the guard check
        would_skip = bool(
            logging_obj._is_assembled_stream_success(result=mock_response)
            and logging_obj.model_call_details.get(
                "has_dispatched_final_stream_success"
            )
        )
        assert would_skip is True, (
            "Dedup guard must block a second dispatch_success_handlers call for the "
            "same assembled streaming response"
        )

    def test_sse_fallback_path_sets_stream_true_for_dedup(self):
        """
        When a nominally non-streaming request receives an SSE response
        (_is_streaming_response returns True), the fallback branch in
        pass_through_endpoints.py must set logging_obj.stream = True so the
        dedup guard activates.

        Before the fix the fallback path never set stream=True, so
        _is_assembled_stream_success always returned False and duplicate
        callback dispatches were never blocked.
        """
        from litellm.types.utils import ModelResponse

        # logging_obj starts with stream=False, as created before the request
        logging_obj = self._make_logging_obj(stream=False)
        assert logging_obj._is_assembled_stream_success(result=MagicMock()) is False

        # Simulate what the SSE fallback branch in pass_through_endpoints.py now does
        logging_obj.stream = True
        logging_obj.model_call_details["stream"] = True

        mock_response = ModelResponse(model="claude-3-5-sonnet-20241022")
        logging_obj.model_call_details["complete_streaming_response"] = mock_response

        # With stream=True the dedup guard must be active
        assert logging_obj._is_assembled_stream_success(result=mock_response) is True

        logging_obj.model_call_details["has_dispatched_final_stream_success"] = True

        would_skip = bool(
            logging_obj._is_assembled_stream_success(result=mock_response)
            and logging_obj.model_call_details.get(
                "has_dispatched_final_stream_success"
            )
        )
        assert would_skip is True

    def test_stream_false_logging_obj_bypasses_dedup_guard(self):
        """
        Demonstrates the pre-fix state: with stream=False on the logging object,
        _is_assembled_stream_success always returns False regardless of whether
        complete_streaming_response is set. This means the dedup guard can never
        fire, so duplicate dispatches go through unchecked.

        This test documents the old broken behavior so the fix is clearly justified.
        """
        from litellm.types.utils import ModelResponse

        logging_obj = self._make_logging_obj(stream=False)
        mock_response = ModelResponse(model="claude-3-5-sonnet-20241022")
        logging_obj.model_call_details["complete_streaming_response"] = mock_response

        # With stream=False, _is_assembled_stream_success returns False even though
        # complete_streaming_response is present — the guard is permanently disabled.
        assert logging_obj._is_assembled_stream_success(result=mock_response) is False


class TestNonStreamingResponseRedaction:
    """
    Regression tests ensuring _create_anthropic_response_logging_payload only sets
    complete_streaming_response for streaming responses. perform_redaction scrubs
    that field exclusively when model_call_details["stream"] is True, so storing it
    on a non-streaming response would deliver the unredacted response to logging
    callbacks when message logging is disabled.
    """

    @staticmethod
    def _make_logging_obj(stream: bool) -> LiteLLMLoggingObj:
        logging_obj = LiteLLMLoggingObj(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "hello"}],
            stream=stream,
            call_type="pass_through_endpoint",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="1245",
        )
        # pass_through_request mirrors the stream flag onto model_call_details,
        # which is the key perform_redaction inspects.
        logging_obj.model_call_details["stream"] = stream
        return logging_obj

    def test_non_streaming_does_not_set_complete_streaming_response(self):
        from litellm.types.utils import ModelResponse

        logging_obj = self._make_logging_obj(stream=False)
        response = ModelResponse(model="claude-3-5-sonnet-20241022")

        AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=response,
            model="claude-3-5-sonnet-20241022",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        assert (
            "complete_streaming_response" not in logging_obj.model_call_details
        ), "non-streaming responses must not populate complete_streaming_response"

    def test_streaming_sets_complete_streaming_response(self):
        from litellm.types.utils import ModelResponse

        logging_obj = self._make_logging_obj(stream=True)
        response = ModelResponse(model="claude-3-5-sonnet-20241022")

        AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=response,
            model="claude-3-5-sonnet-20241022",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        assert (
            logging_obj.model_call_details.get("complete_streaming_response")
            is response
        )

    def test_non_streaming_response_is_redacted_when_message_logging_off(self):
        from litellm.litellm_core_utils.redact_messages import (
            redact_message_input_output_from_logging,
        )
        from litellm.types.utils import Choices, Message, ModelResponse

        logging_obj = self._make_logging_obj(stream=False)
        response = ModelResponse(
            model="claude-3-5-sonnet-20241022",
            choices=[Choices(message=Message(role="assistant", content="secret"))],
        )

        AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=response,
            model="claude-3-5-sonnet-20241022",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        logging_obj.model_call_details["litellm_params"] = {
            "metadata": {"headers": {"x-litellm-enable-message-redaction": True}}
        }

        redacted = redact_message_input_output_from_logging(
            model_call_details=logging_obj.model_call_details,
            result=response,
        )

        leaked = logging_obj.model_call_details.get("complete_streaming_response")
        assert leaked is None
        assert redacted.choices[0].message.content == "redacted-by-litellm"


def _sse_bytes(data: dict) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


class TestAnthropicUsageOnlyFallback:
    """When stream_chunk_builder cannot reassemble a large/agentic stream (returns
    None or raises), Anthropic still emits token usage in the message_start /
    message_delta SSE events. The handler must recover usage-only so the request is
    priced instead of being dropped from SpendLogs while Anthropic billed the tokens."""

    _CHUNKS = [
        _sse_bytes(
            {
                "type": "message_start",
                "message": {
                    "model": "claude-3-5-haiku-20241022",
                    "usage": {
                        "input_tokens": 100,
                        "cache_read_input_tokens": 40,
                        "cache_creation_input_tokens": 20,
                        "output_tokens": 1,
                    },
                },
            }
        ),
        _sse_bytes(
            {
                "type": "message_delta",
                "usage": {
                    "output_tokens": 55,
                    "server_tool_use": {"web_search_requests": 2},
                },
            }
        ),
    ]

    def test_build_usage_only_recovers_cache_inclusive_usage(self):
        response = (
            AnthropicPassthroughLoggingHandler._build_usage_only_response_from_chunks(
                all_chunks=self._CHUNKS, model="claude-3-5-haiku-20241022"
            )
        )
        assert response is not None
        usage = response.usage
        # prompt_tokens must be cache-inclusive (input + cache_read + cache_creation)
        assert usage.prompt_tokens == 160
        assert usage.completion_tokens == 55
        assert usage._cache_read_input_tokens == 40
        assert usage._cache_creation_input_tokens == 20
        assert usage.prompt_tokens_details.cached_tokens == 40
        assert usage.server_tool_use.web_search_requests == 2

    def test_build_usage_only_returns_none_without_usage_events(self):
        chunks = [_sse_bytes({"type": "content_block_delta", "delta": {"text": "hi"}})]
        assert (
            AnthropicPassthroughLoggingHandler._build_usage_only_response_from_chunks(
                all_chunks=chunks, model="claude-3-5-haiku-20241022"
            )
            is None
        )

    def test_build_usage_only_recovers_cache_split_server_tools_and_model(self):
        # the model is "unknown" up-front and only the 5m/1h cache split is sent
        # (no flat cache_creation_input_tokens); web/tool-search and geo arrive in
        # message_delta. All must be recovered and priced, not left at $0.
        chunks = [
            "event: ping\ndata: [DONE]\n\n",  # ignored sentinel between real events
            _sse_bytes(
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-opus-4-6",
                        "usage": {
                            "input_tokens": 80,
                            "output_tokens": 1,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": 12,
                                "ephemeral_1h_input_tokens": 8,
                            },
                            "inference_geo": "us",
                        },
                    },
                }
            ),
            _sse_bytes(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {
                        "output_tokens": 40,
                        "cache_read_input_tokens": 5,
                        "inference_geo": "us",
                        "server_tool_use": {
                            "web_search_requests": 1,
                            "tool_search_requests": 3,
                        },
                    },
                }
            ),
        ]
        response = (
            AnthropicPassthroughLoggingHandler._build_usage_only_response_from_chunks(
                all_chunks=chunks, model="unknown"
            )
        )
        assert response is not None
        assert response.model == "claude-opus-4-6"
        # the real stop_reason is surfaced, not a hardcoded "stop"
        assert response.choices[0].finish_reason == "tool_calls"
        usage = response.usage
        # 80 input + 20 cache_creation (derived from 12+8) + 5 cache_read
        assert usage.prompt_tokens == 105
        assert usage.completion_tokens == 40
        assert usage._cache_creation_input_tokens == 20
        assert usage._cache_read_input_tokens == 5
        assert usage.server_tool_use.web_search_requests == 1
        assert usage.server_tool_use.tool_search_requests == 3

    @pytest.mark.parametrize(
        "event_str,expected",
        [
            ("data: [DONE]", None),
            ("data: ", None),
            ("data: {not-json", None),
            ("event: ping", None),
            ('data: {"a": 1}', {"a": 1}),
        ],
    )
    def test_extract_sse_data_handles_malformed_and_sentinel_lines(
        self, event_str, expected
    ):
        assert (
            AnthropicPassthroughLoggingHandler._extract_sse_data(event_str) == expected
        )

    def _real_logging_obj(self):
        from litellm.litellm_core_utils.litellm_logging import Logging as RealLoggingObj

        logging_obj = RealLoggingObj(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="pass_through_endpoint",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="1",
        )
        logging_obj.model_call_details["litellm_params"] = {}
        logging_obj.litellm_params = {}
        return logging_obj

    @patch("litellm.completion_cost")
    @patch.object(
        AnthropicPassthroughLoggingHandler, "_build_complete_streaming_response"
    )
    def test_handler_falls_back_when_assembly_returns_none(
        self, mock_assemble, mock_cost
    ):
        mock_assemble.return_value = None
        mock_cost.return_value = 0.0021
        logging_obj = self._real_logging_obj()

        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/anthropic/v1/messages",
            request_body={"model": "claude-3-5-haiku-20241022", "stream": True},
            endpoint_type="messages",
            start_time=datetime.now(),
            all_chunks=list(self._CHUNKS),
            end_time=datetime.now(),
        )

        assert result["result"] is not None
        assert result["result"].usage.completion_tokens == 55
        assert result["kwargs"]["response_cost"] == 0.0021

    @patch("litellm.completion_cost")
    @patch.object(
        AnthropicPassthroughLoggingHandler, "_build_complete_streaming_response"
    )
    def test_handler_falls_back_when_assembly_raises(self, mock_assemble, mock_cost):
        import litellm

        mock_assemble.side_effect = litellm.APIError(
            status_code=500,
            message="boom",
            llm_provider="anthropic",
            model="claude-3-5-haiku-20241022",
        )
        mock_cost.return_value = 0.0021
        logging_obj = self._real_logging_obj()

        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/anthropic/v1/messages",
            request_body={"model": "claude-3-5-haiku-20241022", "stream": True},
            endpoint_type="messages",
            start_time=datetime.now(),
            all_chunks=list(self._CHUNKS),
            end_time=datetime.now(),
        )

        # a raise from stream_chunk_builder must be treated like a None result,
        # not propagate out and drop the request from SpendLogs
        assert result["result"] is not None
        assert result["result"].usage.completion_tokens == 55
        assert result["kwargs"]["response_cost"] == 0.0021

    @patch.object(
        AnthropicPassthroughLoggingHandler, "_build_complete_streaming_response"
    )
    def test_handler_returns_none_when_no_usage_recoverable(self, mock_assemble):
        # assembly fails AND the chunks carry no usage event, so there is nothing
        # to price; the handler must return None rather than fabricate a response
        mock_assemble.return_value = None
        logging_obj = self._real_logging_obj()
        chunks = [_sse_bytes({"type": "content_block_delta", "delta": {"text": "hi"}})]

        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/anthropic/v1/messages",
            request_body={"model": "claude-3-5-haiku-20241022", "stream": True},
            endpoint_type="messages",
            start_time=datetime.now(),
            all_chunks=chunks,
            end_time=datetime.now(),
        )

        assert result["result"] is None
        assert result["kwargs"] == {}

    @patch.object(
        AnthropicPassthroughLoggingHandler, "_build_usage_only_response_from_chunks"
    )
    @patch.object(
        AnthropicPassthroughLoggingHandler, "_build_complete_streaming_response"
    )
    def test_handler_does_not_crash_when_usage_only_fallback_raises(
        self, mock_assemble, mock_fallback
    ):
        # if the usage-only fallback itself raises, it must be treated as None and
        # drop gracefully, not propagate out and crash the success handler
        mock_assemble.return_value = None
        mock_fallback.side_effect = Exception("fallback boom")
        logging_obj = self._real_logging_obj()

        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="/anthropic/v1/messages",
            request_body={"model": "claude-3-5-haiku-20241022", "stream": True},
            endpoint_type="messages",
            start_time=datetime.now(),
            all_chunks=list(self._CHUNKS),
            end_time=datetime.now(),
        )

        assert result["result"] is None
        assert result["kwargs"] == {}


class TestAnthropicResponseCostRecordedOnModelCallDetails:
    """The pass-through success path reads spend from
    model_call_details["response_cost"], not from kwargs, so the streaming payload
    builder must record it there or streaming pass-through logs $0."""

    def test_create_payload_records_response_cost_on_model_call_details(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        logging_obj.get_router_model_id.return_value = None
        logging_obj.litellm_params = {}
        logging_obj.litellm_call_id = "test-call-id"

        response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="hello", role="assistant"),
                )
            ],
            created=1234567890,
            model="claude-3-7-sonnet-20250219",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=response,
            model="claude-3-7-sonnet-20250219",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        assert (
            logging_obj.model_call_details["response_cost"] == kwargs["response_cost"]
        )
        assert logging_obj.model_call_details["response_cost"] > 0
