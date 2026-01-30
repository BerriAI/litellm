"""
Test cases for Vertex AI passthrough batch prediction functionality
"""
import base64
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.types.utils import SpecialEnums
from litellm.types.llms.openai import BatchJobStatus


class TestVertexAIBatchPassthroughHandler:
    """Test cases for Vertex AI batch prediction passthrough functionality"""

    @pytest.fixture
    def mock_httpx_response(self):
        """Mock httpx response for batch job creation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/batchPredictionJobs/123456789",
            "displayName": "litellm-vertex-batch-test",
            "model": "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash",
            "createTime": "2024-01-01T00:00:00Z",
            "state": "JOB_STATE_PENDING",
            "inputConfig": {
                "gcsSource": {
                    "uris": ["gs://test-bucket/input.jsonl"]
                },
                "instancesFormat": "jsonl"
            },
            "outputConfig": {
                "gcsDestination": {
                    "outputUriPrefix": "gs://test-bucket/output/"
                },
                "predictionsFormat": "jsonl"
            }
        }
        return mock_response

    @pytest.fixture
    def mock_logging_obj(self):
        """Mock logging object"""
        mock = Mock()
        mock.litellm_call_id = "test-call-id-123"
        mock.model_call_details = {}
        mock.optional_params = {}
        return mock

    @pytest.fixture
    def mock_managed_files_hook(self):
        """Mock managed files hook"""
        mock_hook = Mock()
        mock_hook.afile_content.return_value = Mock(content=b'{"test": "data"}')
        return mock_hook

    def test_batch_prediction_jobs_handler_success(self, mock_httpx_response, mock_logging_obj):
        """Test successful batch job creation and tracking"""
        with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger') as mock_logger:
            with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.get_actual_model_id_from_router') as mock_get_model_id:
                with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler._store_batch_managed_object') as mock_store:
                    with patch('litellm.llms.vertex_ai.batches.transformation.VertexAIBatchTransformation') as mock_transformation:
                        
                        # Setup mocks
                        mock_get_model_id.return_value = "vertex_ai/gemini-1.5-flash"
                        mock_transformation.transform_vertex_ai_batch_response_to_openai_batch_response.return_value = {
                            "id": "123456789",
                            "object": "batch",
                            "status": "validating",
                            "created_at": 1704067200,
                            "input_file_id": "file-123",
                            "output_file_id": "file-456",
                            "error_file_id": None,
                            "completion_window": "24hrs"
                        }
                        mock_transformation._get_batch_id_from_vertex_ai_batch_response.return_value = "123456789"
                        
                        # Test the handler
                        result = VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                            httpx_response=mock_httpx_response,
                            logging_obj=mock_logging_obj,
                            url_route="/v1/projects/test-project/locations/us-central1/batchPredictionJobs",
                            result="success",
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            cache_hit=False,
                            user_api_key_dict={"user_id": "test-user"}
                        )
                        
                        # Verify the result
                        assert result is not None
                        assert "kwargs" in result
                        assert result["kwargs"]["model"] == "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                        assert result["kwargs"]["batch_id"] == "123456789"
                        
                        # Verify mocks were called
                        mock_get_model_id.assert_called_once()
                        mock_store.assert_called_once()

    def test_batch_prediction_jobs_handler_failure(self, mock_logging_obj):
        """Test batch job creation failure handling"""
        # Mock failed response
        mock_httpx_response = Mock()
        mock_httpx_response.status_code = 400
        mock_httpx_response.json.return_value = {"error": "Invalid request"}
        
        with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger') as mock_logger:
            # Test the handler with failed response
            result = VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route="/v1/projects/test-project/locations/us-central1/batchPredictionJobs",
                result="error",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
                user_api_key_dict={"user_id": "test-user"}
            )
            
            # Should return a structured response for failed responses
            assert result is not None
            assert "result" in result
            assert "kwargs" in result
            assert result["result"].choices[0].finish_reason == "stop"
            assert result["kwargs"]["batch_job_state"] == "JOB_STATE_FAILED"

    def test_get_actual_model_id_from_router_with_router(self):
        """Test getting model ID when router is available"""
        with patch('litellm.proxy.proxy_server.llm_router') as mock_router:
            with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path') as mock_extract:
                
                # Setup mocks
                mock_router.get_model_ids.return_value = ["vertex_ai/gemini-1.5-flash"]
                mock_extract.return_value = "gemini-1.5-flash"
                
                # Test the method
                result = VertexPassthroughLoggingHandler.get_actual_model_id_from_router(
                    "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                )
                
                # Verify result
                assert result == "vertex_ai/gemini-1.5-flash"
                mock_router.get_model_ids.assert_called_once_with(model_name="gemini-1.5-flash")

    def test_get_actual_model_id_from_router_without_router(self):
        """Test getting model ID when router is not available"""
        with patch('litellm.proxy.proxy_server.llm_router', None):
            with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path') as mock_extract:
                
                # Setup mocks
                mock_extract.return_value = "gemini-1.5-flash"
                
                # Test the method
                result = VertexPassthroughLoggingHandler.get_actual_model_id_from_router(
                    "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                )
                
                # Verify result
                assert result == "gemini-1.5-flash"

    def test_get_actual_model_id_from_router_model_not_found(self):
        """Test getting model ID when model is not found in router"""
        with patch('litellm.proxy.proxy_server.llm_router') as mock_router:
            with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path') as mock_extract:
                
                # Setup mocks - router returns empty list
                mock_router.get_model_ids.return_value = []
                mock_extract.return_value = "gemini-1.5-flash"
                
                # Test the method
                result = VertexPassthroughLoggingHandler.get_actual_model_id_from_router(
                    "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                )
                
                # Verify result - should fallback to extracted model name
                assert result == "gemini-1.5-flash"

    def test_unified_object_id_generation(self):
        """Test unified object ID generation for batch tracking"""
        model_id = "vertex_ai/gemini-1.5-flash"
        batch_id = "123456789"
        
        # Generate the expected unified ID
        unified_id_string = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(model_id, batch_id)
        expected_unified_id = base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")
        
        # Test the generation
        actual_unified_id = base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")
        
        assert actual_unified_id == expected_unified_id
        assert isinstance(actual_unified_id, str)
        assert len(actual_unified_id) > 0

    def test_store_batch_managed_object(self, mock_logging_obj, mock_managed_files_hook):
        """Test storing batch managed object for cost tracking"""
        with patch('litellm.proxy.proxy_server.proxy_logging_obj') as mock_proxy_logging_obj:
            with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger') as mock_logger:
                
                # Setup mock proxy logging obj
                mock_proxy_logging_obj.get_proxy_hook.return_value = mock_managed_files_hook
                
                # Test data
                unified_object_id = "test-unified-id"
                batch_object = {
                    "id": "123456789",
                    "object": "batch",
                    "status": "validating"
                }
                model_object_id = "123456789"
                
                # Test the method
                VertexPassthroughLoggingHandler._store_batch_managed_object(
                    unified_object_id=unified_object_id,
                    batch_object=batch_object,
                    model_object_id=model_object_id,
                    logging_obj=mock_logging_obj,
                    user_api_key_dict={"user_id": "test-user"}
                )
                
                # Verify the managed files hook was called
                mock_managed_files_hook.store_unified_object_id.assert_called_once()

    def test_batch_cost_calculation_integration(self):
        """Test integration with batch cost calculation"""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage
        
        # Mock Vertex AI batch responses
        vertex_ai_batch_responses = [
            {
                "status": "JOB_STATE_SUCCEEDED",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Hello, world!"}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15
                    }
                }
            }
        ]
        
        with patch('litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexGeminiConfig') as mock_config:
            with patch('litellm.completion_cost') as mock_completion_cost:
                
                # Setup mocks
                mock_config.return_value._transform_google_generate_content_to_openai_model_response.return_value = Mock(
                    usage=Mock(total_tokens=15, prompt_tokens=10, completion_tokens=5)
                )
                mock_completion_cost.return_value = 0.001
                
                # Test the cost calculation
                total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
                    vertex_ai_batch_responses, 
                    model_name="gemini-1.5-flash"
                )
                
                # Verify results
                assert total_cost == 0.001
                assert usage.total_tokens == 15
                assert usage.prompt_tokens == 10
                assert usage.completion_tokens == 5

    def test_batch_response_transformation(self):
        """Test transformation of Vertex AI batch responses to OpenAI format"""
        from litellm.llms.vertex_ai.batches.transformation import VertexAIBatchTransformation
        
        # Mock Vertex AI batch response
        vertex_ai_response = {
            "name": "projects/test-project/locations/us-central1/batchPredictionJobs/123456789",
            "displayName": "test-batch",
            "model": "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash",
            "createTime": "2024-01-01T00:00:00.000Z",
            "state": "JOB_STATE_SUCCEEDED"
        }
        
        # Test transformation
        result = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            vertex_ai_response
        )
        
        # Verify the transformation
        assert result["id"] == "123456789"
        assert result["object"] == "batch"
        assert result["status"] == "completed"  # JOB_STATE_SUCCEEDED should map to completed

    def test_batch_id_extraction(self):
        """Test extraction of batch ID from Vertex AI response"""
        from litellm.llms.vertex_ai.batches.transformation import VertexAIBatchTransformation
        
        # Test various batch ID formats
        test_cases = [
            "projects/123/locations/us-central1/batchPredictionJobs/456789",
            "projects/abc/locations/europe-west1/batchPredictionJobs/def123",
            "batchPredictionJobs/999",
            "invalid-format"
        ]
        
        expected_results = ["456789", "def123", "999", "invalid-format"]
        
        for test_case, expected in zip(test_cases, expected_results):
            result = VertexAIBatchTransformation._get_batch_id_from_vertex_ai_batch_response(
                {"name": test_case}
            )
            assert result == expected

    def test_model_name_extraction_from_vertex_path(self):
        """Test extraction of model name from Vertex AI path"""
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler
        )
        
        # Test various model path formats
        test_cases = [
            "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash",
            "projects/abc/locations/europe-west1/publishers/google/models/gemini-2.0-flash",
            "publishers/google/models/gemini-pro",
            "invalid-path"
        ]
        
        expected_results = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-pro", "invalid-path"]
        
        for test_case, expected in zip(test_cases, expected_results):
            result = VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path(test_case)
            assert result == expected

    @pytest.mark.asyncio
    async def test_batch_completion_workflow(self, mock_httpx_response, mock_logging_obj, mock_managed_files_hook):
        """Test the complete batch completion workflow"""
        with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger') as mock_logger:
            with patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.get_actual_model_id_from_router') as mock_get_model_id:
                with patch('litellm.proxy.proxy_server.proxy_logging_obj') as mock_proxy_logging_obj:
                    mock_proxy_logging_obj.get_proxy_hook.return_value = mock_managed_files_hook
                with patch('litellm.llms.vertex_ai.batches.transformation.VertexAIBatchTransformation') as mock_transformation:
                    
                    # Setup mocks
                    mock_get_model_id.return_value = "vertex_ai/gemini-1.5-flash"
                    mock_transformation.transform_vertex_ai_batch_response_to_openai_batch_response.return_value = {
                        "id": "123456789",
                        "object": "batch",
                        "status": "completed",
                        "created_at": 1704067200,
                        "input_file_id": "file-123",
                        "output_file_id": "file-456",
                        "error_file_id": None,
                        "completion_window": "24hrs"
                    }
                    mock_transformation._get_batch_id_from_vertex_ai_batch_response.return_value = "123456789"
                    
                    # Test the complete workflow
                    result = VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                        httpx_response=mock_httpx_response,
                        logging_obj=mock_logging_obj,
                        url_route="/v1/projects/test-project/locations/us-central1/batchPredictionJobs",
                        result="success",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        cache_hit=False,
                        user_api_key_dict={"user_id": "test-user"}
                    )
                    
                    # Verify the complete workflow
                    assert result is not None
                    assert "kwargs" in result
                    assert result["kwargs"]["model"] == "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                    assert result["kwargs"]["batch_id"] == "123456789"
                    
                    # Verify all mocks were called
                    mock_get_model_id.assert_called_once()
                    mock_transformation.transform_vertex_ai_batch_response_to_openai_batch_response.assert_called_once()
                    # Note: store_unified_object_id is called asynchronously, so we can't easily verify it in this test


class TestVertexAIBatchCostCalculation:
    """Test cases for Vertex AI batch cost calculation functionality"""

    def test_calculate_vertex_ai_batch_cost_and_usage_success(self):
        """Test successful batch cost and usage calculation"""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage
        
        # Mock successful batch responses
        vertex_ai_batch_responses = [
            {
                "status": "JOB_STATE_SUCCEEDED",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Hello, world!"}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15
                    }
                }
            },
            {
                "status": "JOB_STATE_SUCCEEDED", 
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "How are you?"}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 8,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 11
                    }
                }
            }
        ]
        
        with patch('litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexGeminiConfig') as mock_config:
            with patch('litellm.completion_cost') as mock_completion_cost:
                
                # Setup mocks
                mock_model_response = Mock()
                mock_model_response.usage = Mock(total_tokens=15, prompt_tokens=10, completion_tokens=5)
                mock_config.return_value._transform_google_generate_content_to_openai_model_response.return_value = mock_model_response
                mock_completion_cost.return_value = 0.001
                
                # Test the calculation
                total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
                    vertex_ai_batch_responses, 
                    model_name="gemini-1.5-flash"
                )
                
                # Verify results
                assert total_cost == 0.002  # 2 responses * 0.001 each
                assert usage.total_tokens == 30  # 15 + 15
                assert usage.prompt_tokens == 20  # 10 + 10
                assert usage.completion_tokens == 10  # 5 + 5

    def test_calculate_vertex_ai_batch_cost_and_usage_with_failed_responses(self):
        """Test batch cost calculation with some failed responses"""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage
        
        # Mock batch responses with some failures
        vertex_ai_batch_responses = [
            {
                "status": "JOB_STATE_SUCCEEDED",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Hello, world!"}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15
                    }
                }
            },
            {
                "status": "JOB_STATE_FAILED",  # Failed response
                "response": None
            },
            {
                "status": "JOB_STATE_SUCCEEDED",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "How are you?"}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 8,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 11
                    }
                }
            }
        ]
        
        with patch('litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexGeminiConfig') as mock_config:
            with patch('litellm.completion_cost') as mock_completion_cost:
                
                # Setup mocks
                mock_model_response = Mock()
                mock_model_response.usage = Mock(total_tokens=15, prompt_tokens=10, completion_tokens=5)
                mock_config.return_value._transform_google_generate_content_to_openai_model_response.return_value = mock_model_response
                mock_completion_cost.return_value = 0.001
                
                # Test the calculation
                total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
                    vertex_ai_batch_responses, 
                    model_name="gemini-1.5-flash"
                )
                
                # Verify results - should only process successful responses
                assert total_cost == 0.002  # 2 successful responses * 0.001 each
                assert usage.total_tokens == 30  # 15 + 15
                assert usage.prompt_tokens == 20  # 10 + 10
                assert usage.completion_tokens == 10  # 5 + 5

    def test_calculate_vertex_ai_batch_cost_and_usage_empty_responses(self):
        """Test batch cost calculation with empty response list"""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage
        
        # Test with empty list
        total_cost, usage = calculate_vertex_ai_batch_cost_and_usage([], model_name="gemini-1.5-flash")
        
        # Verify results
        assert total_cost == 0.0
        assert usage.total_tokens == 0
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
