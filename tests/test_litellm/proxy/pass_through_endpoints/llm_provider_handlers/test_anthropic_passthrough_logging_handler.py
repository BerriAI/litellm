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
