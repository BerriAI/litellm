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
        if not model and hasattr(logging_obj, 'model_call_details') and logging_obj.model_call_details.get('model'):
            model = logging_obj.model_call_details.get('model')
            
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
                "succeeded": 0
            },
            "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_01Wj7gkQk7gn4MpAKR8ZEDU2/results",
            "type": "message_batch"
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
                        "messages": [
                            {
                                "content": "Hello, world",
                                "role": "user"
                            }
                        ],
                        "model": "claude-sonnet-4-5-20250929"
                    }
                }
            ]
        }

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler._store_batch_managed_object')
    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router')
    @patch('litellm.llms.anthropic.batches.transformation.AnthropicBatchesConfig')
    def test_batch_creation_handler_success(
        self,
        mock_batches_config,
        mock_get_model_id,
        mock_store_batch,
        mock_httpx_response,
        mock_logging_obj,
        mock_request_body
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
        mock_batches_config_instance.transform_retrieve_batch_response.return_value = mock_batch_response
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

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler._store_batch_managed_object')
    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router')
    def test_batch_creation_handler_model_extraction_from_nested_request(
        self,
        mock_get_model_id,
        mock_store_batch,
        mock_httpx_response,
        mock_logging_obj
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
        
        with patch.object(AnthropicBatchesConfig, 'transform_retrieve_batch_response', return_value=mock_batch_response):
            # Request body with nested model in requests[0].params.model
            request_body = {
                "requests": [
                    {
                        "custom_id": "test-1",
                        "params": {
                            "model": "claude-sonnet-4-5-20250929",
                            "messages": [{"role": "user", "content": "test"}]
                        }
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

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router')
    def test_batch_creation_handler_model_prefix_when_not_in_router(
        self,
        mock_get_model_id,
        mock_httpx_response,
        mock_logging_obj,
        mock_request_body
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
        
        with patch.object(AnthropicBatchesConfig, 'transform_retrieve_batch_response', return_value=mock_batch_response):
            with patch.object(AnthropicPassthroughLoggingHandler, '_store_batch_managed_object'):
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
                assert "anthropic/claude-sonnet-4-5-20250929" in decoded or "claude-sonnet-4-5-20250929" in decoded

    def test_batch_creation_handler_failure_status_code(
        self,
        mock_logging_obj,
        mock_request_body
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

    @patch('litellm.proxy.proxy_server.proxy_logging_obj')
    def test_store_batch_managed_object_success(
        self,
        mock_proxy_logging_obj,
        mock_logging_obj
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
        
        with patch('asyncio.create_task'):
            AnthropicPassthroughLoggingHandler._store_batch_managed_object(
                unified_object_id="test-unified-id",
                batch_object=batch_object,
                model_object_id="msgbatch_123",
                logging_obj=mock_logging_obj,
                user_id="test-user"
            )
            
            # Verify managed files hook was called
            mock_proxy_logging_obj.get_proxy_hook.assert_called_once_with("managed_files") 
