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
                "gcsSource": {"uris": ["gs://test-bucket/input.jsonl"]},
                "instancesFormat": "jsonl",
            },
            "outputConfig": {
                "gcsDestination": {"outputUriPrefix": "gs://test-bucket/output/"},
                "predictionsFormat": "jsonl",
            },
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

    def test_batch_prediction_jobs_handler_success(
        self, mock_httpx_response, mock_logging_obj
    ):
        """Test successful batch job creation and tracking"""
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger"
        ) as mock_logger:
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.get_actual_model_id_from_router"
            ) as mock_get_model_id:
                with patch(
                    "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler._store_batch_managed_object"
                ) as mock_store:
                    with patch(
                        "litellm.llms.vertex_ai.batches.transformation.VertexAIBatchTransformation"
                    ) as mock_transformation:

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
                            "completion_window": "24h",
                        }
                        mock_transformation._get_batch_id_from_vertex_ai_batch_response.return_value = (
                            "123456789"
                        )

                        # Test the handler
                        result = VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                            httpx_response=mock_httpx_response,
                            logging_obj=mock_logging_obj,
                            url_route="/v1/projects/test-project/locations/us-central1/batchPredictionJobs",
                            result="success",
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            cache_hit=False,
                            user_api_key_dict={"user_id": "test-user"},
                        )

                        # Verify the result
                        assert result is not None
                        assert "kwargs" in result
                        assert (
                            result["kwargs"]["model"]
                            == "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                        )
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

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger"
        ) as mock_logger:
            # Test the handler with failed response
            result = VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route="/v1/projects/test-project/locations/us-central1/batchPredictionJobs",
                result="error",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
                user_api_key_dict={"user_id": "test-user"},
            )

            # Should return a structured response for failed responses
            assert result is not None
            assert "result" in result
            assert "kwargs" in result
            assert result["result"].choices[0].finish_reason == "stop"
            assert result["kwargs"]["batch_job_state"] == "JOB_STATE_FAILED"

    def test_get_actual_model_id_from_router_with_router(self):
        """Test getting model ID when router is available"""
        with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path"
            ) as mock_extract:

                # Setup mocks
                mock_router.get_model_ids.return_value = ["vertex_ai/gemini-1.5-flash"]
                mock_extract.return_value = "gemini-1.5-flash"

                # Test the method
                result = VertexPassthroughLoggingHandler.get_actual_model_id_from_router(
                    "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                )

                # Verify result
                assert result == "vertex_ai/gemini-1.5-flash"
                mock_router.get_model_ids.assert_called_once_with(
                    model_name="gemini-1.5-flash"
                )

    def test_get_actual_model_id_from_router_without_router(self):
        """Test getting model ID when router is not available"""
        with patch("litellm.proxy.proxy_server.llm_router", None):
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path"
            ) as mock_extract:

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
        with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path"
            ) as mock_extract:

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
        unified_id_string = (
            SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(
                model_id, batch_id
            )
        )
        expected_unified_id = (
            base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")
        )

        # Test the generation
        actual_unified_id = (
            base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")
        )

        assert actual_unified_id == expected_unified_id
        assert isinstance(actual_unified_id, str)
        assert len(actual_unified_id) > 0

    def test_store_batch_managed_object(
        self, mock_logging_obj, mock_managed_files_hook
    ):
        """Test storing batch managed object for cost tracking"""
        with patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_proxy_logging_obj:
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger"
            ) as mock_logger:

                # Setup mock proxy logging obj
                mock_proxy_logging_obj.get_proxy_hook.return_value = (
                    mock_managed_files_hook
                )

                # Test data
                unified_object_id = "test-unified-id"
                batch_object = {
                    "id": "123456789",
                    "object": "batch",
                    "status": "validating",
                }
                model_object_id = "123456789"

                # Test the method
                VertexPassthroughLoggingHandler._store_batch_managed_object(
                    unified_object_id=unified_object_id,
                    batch_object=batch_object,
                    model_object_id=model_object_id,
                    logging_obj=mock_logging_obj,
                    is_batch_create=True,
                    user_api_key_dict={"user_id": "test-user"},
                )

                # Verify the managed files hook was called
                mock_managed_files_hook.store_unified_object_id.assert_called_once()

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
        mock_managed_files_hook,
        kwargs,
        expected_user_id,
        expected_team_id,
    ):
        """The fabricated UserAPIKeyAuth must inherit user_id/team_id from the
        request's litellm_params.metadata, not the (always-empty) top-level
        kwargs lookup. Falls back to "default-user" only when metadata is
        absent."""
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_pl,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger"
            ),
        ):
            mock_pl.get_proxy_hook.return_value = mock_managed_files_hook

            VertexPassthroughLoggingHandler._store_batch_managed_object(
                unified_object_id="uoi",
                batch_object={"id": "b1", "object": "batch", "status": "validating"},
                model_object_id="b1",
                logging_obj=mock_logging_obj,
                is_batch_create=True,
                **kwargs,
            )

            mock_managed_files_hook.store_unified_object_id.assert_called_once()
            call_kwargs = mock_managed_files_hook.store_unified_object_id.call_args[1]
            assert call_kwargs["user_api_key_dict"].user_id == expected_user_id
            assert call_kwargs["user_api_key_dict"].team_id == expected_team_id

    def _store_with_metadata(self, mock_logging_obj, mock_managed_files_hook, metadata):
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_pl,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger"
            ),
        ):
            mock_pl.get_proxy_hook.return_value = mock_managed_files_hook
            VertexPassthroughLoggingHandler._store_batch_managed_object(
                unified_object_id="uoi",
                batch_object={"id": "b1", "object": "batch", "status": "validating"},
                model_object_id="b1",
                logging_obj=mock_logging_obj,
                is_batch_create=True,
                litellm_params={"metadata": metadata},
            )
        mock_managed_files_hook.store_unified_object_id.assert_called_once()
        return mock_managed_files_hook.store_unified_object_id.call_args[1]

    def test_persisted_tags_are_db_safe(self, mock_logging_obj, mock_managed_files_hook):
        """Regression for PostgreSQL 22P05, asserted for Vertex too so moving the
        sanitation somewhere that only covers Anthropic fails loudly."""
        call_kwargs = self._store_with_metadata(
            mock_logging_obj,
            mock_managed_files_hook,
            {"user_api_key": "hashed-key-a", "tags": ["clean", "bad\x00tag"]},
        )

        assert call_kwargs["request_tags"] == ("clean", "badtag")

    def test_create_persists_key_hash_and_tags(
        self, mock_logging_obj, mock_managed_files_hook
    ):
        """Regression (spend loss): the batch create must persist the creating key's hashed
        token and its tags so CheckBatchCost can attribute the batch-cost spend row. Before
        this fix the stored api_key was always "" and the row was dropped as unattributed."""
        call_kwargs = self._store_with_metadata(
            mock_logging_obj,
            mock_managed_files_hook,
            {
                "user_api_key": "hashed-key-a",
                "user_api_key_user_id": "alice",
                "user_api_key_team_id": "team-alpha",
                "user_api_key_auth_metadata": {"tags": ["env:prod", 7, "team:ml"]},
            },
        )

        assert call_kwargs["user_api_key_dict"].api_key == "hashed-key-a"
        # non-string tags are dropped so downstream tag budgets cannot be bypassed
        assert call_kwargs["request_tags"] == ("env:prod", "team:ml")
        assert call_kwargs["persist_attribution"] is True

    @pytest.mark.parametrize(
        "metadata, expected",
        [
            # a request that sent its own tags (x-litellm-tags header or body metadata)
            ({"tags": ["req:a", "req:b"]}, ("req:a", "req:b")),
            # request tags win over the key's own tags
            (
                {"tags": ["req:a"], "user_api_key_auth_metadata": {"tags": ["key:b"]}},
                ("req:a",),
            ),
            # no request tags: fall back to the tags the key itself carries
            ({"user_api_key_auth_metadata": {"tags": ["key:b"]}}, ("key:b",)),
            # neither: no tags on the spend row
            ({}, None),
        ],
    )
    def test_request_tags_precedence(
        self, mock_logging_obj, mock_managed_files_hook, metadata, expected
    ):
        """Request tags take precedence over the key's tags, and the key's tags are the
        fallback because a tagged key does not put its tags in the top-level metadata."""
        call_kwargs = self._store_with_metadata(
            mock_logging_obj,
            mock_managed_files_hook,
            {"user_api_key": "hashed-key-a", **metadata},
        )

        assert call_kwargs["request_tags"] == expected

    @pytest.mark.parametrize(
        "url_route, expected",
        [
            ("/v1/projects/p/locations/us-central1/batchPredictionJobs", True),
            ("/v1/projects/p/locations/us-central1/batchPredictionJobs/", True),
            ("/v1/projects/p/locations/us-central1/batchPredictionJobs?alt=json", True),
            ("/v1/projects/p/locations/us-central1/batchPredictionJobs/123456", False),
            ("/v1/projects/p/locations/us-central1/batchPredictionJobs/123456?alt=json", False),
        ],
    )
    def test_batch_is_registered_from_the_create_route_only(
        self, mock_logging_obj, url_route, expected
    ):
        """Only a POST to the collection route is the create, and only the create claims
        attribution. Every id-scoped route is a poll or retrieve, which still reports the
        batch so its status and file object stay in sync, but carries is_batch_create=False
        so it neither claims the batch nor creates a row it would then own."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "name": "projects/p/locations/us-central1/batchPredictionJobs/123456",
            "model": "publishers/google/models/gemini-2.5-flash",
        }

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger"
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler._store_batch_managed_object"
            ) as mock_store,
            patch(
                "litellm.llms.vertex_ai.batches.transformation.VertexAIBatchTransformation"
            ) as mock_transformation,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.get_actual_model_id_from_router",
                return_value="gemini-2.5-flash",
            ),
        ):
            mock_transformation.transform_vertex_ai_batch_response_to_openai_batch_response.return_value = {
                "id": "123456",
                "object": "batch",
                "status": "validating",
                "created_at": 1704067200,
                "input_file_id": "gs://bucket/in.jsonl",
                "completion_window": "24h",
            }
            mock_transformation._get_batch_id_from_vertex_ai_batch_response.return_value = "123456"

            VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                httpx_response=response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
            )

        # every route reports the batch; only the create claims it
        mock_store.assert_called_once()
        assert mock_store.call_args[1]["unified_object_id"]
        assert mock_store.call_args[1]["is_batch_create"] is expected

    def test_batch_cost_calculation_integration(self):
        """Single Vertex AI response → non-zero cost with correct token counts."""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage

        vertex_ai_batch_responses = [
            {
                "response": {
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                    }
                }
            }
        ]

        total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
            vertex_ai_batch_responses, model_name="gemini-2.0-flash-001"
        )

        assert usage.total_tokens == 15
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert total_cost > 0, "batch_cost_calculator should return a non-zero cost"

    def test_batch_response_transformation(self):
        """Test transformation of Vertex AI batch responses to OpenAI format"""
        from litellm.llms.vertex_ai.batches.transformation import (
            VertexAIBatchTransformation,
        )

        # Mock Vertex AI batch response
        vertex_ai_response = {
            "name": "projects/test-project/locations/us-central1/batchPredictionJobs/123456789",
            "displayName": "test-batch",
            "model": "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash",
            "createTime": "2024-01-01T00:00:00.000Z",
            "state": "JOB_STATE_SUCCEEDED",
        }

        # Test transformation
        result = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            vertex_ai_response
        )

        # Verify the transformation
        assert result["id"] == "123456789"
        assert result["object"] == "batch"
        assert (
            result["status"] == "completed"
        )  # JOB_STATE_SUCCEEDED should map to completed

    def test_batch_id_extraction(self):
        """Test extraction of batch ID from Vertex AI response"""
        from litellm.llms.vertex_ai.batches.transformation import (
            VertexAIBatchTransformation,
        )

        # Test various batch ID formats
        test_cases = [
            "projects/123/locations/us-central1/batchPredictionJobs/456789",
            "projects/abc/locations/europe-west1/batchPredictionJobs/def123",
            "batchPredictionJobs/999",
            "invalid-format",
        ]

        expected_results = ["456789", "def123", "999", "invalid-format"]

        for test_case, expected in zip(test_cases, expected_results):
            result = (
                VertexAIBatchTransformation._get_batch_id_from_vertex_ai_batch_response(
                    {"name": test_case}
                )
            )
            assert result == expected

    def test_model_name_extraction_from_vertex_path(self):
        """Test extraction of model name from Vertex AI path"""
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        # Test various model path formats
        test_cases = [
            "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash",
            "projects/abc/locations/europe-west1/publishers/google/models/gemini-2.0-flash",
            "publishers/google/models/gemini-pro",
            "invalid-path",
        ]

        expected_results = [
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-pro",
            "invalid-path",
        ]

        for test_case, expected in zip(test_cases, expected_results):
            result = (
                VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path(
                    test_case
                )
            )
            assert result == expected

    @pytest.mark.asyncio
    async def test_batch_completion_workflow(
        self, mock_httpx_response, mock_logging_obj, mock_managed_files_hook
    ):
        """Test the complete batch completion workflow"""
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.verbose_proxy_logger"
        ) as mock_logger:
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler.VertexPassthroughLoggingHandler.get_actual_model_id_from_router"
            ) as mock_get_model_id:
                with patch(
                    "litellm.proxy.proxy_server.proxy_logging_obj"
                ) as mock_proxy_logging_obj:
                    mock_proxy_logging_obj.get_proxy_hook.return_value = (
                        mock_managed_files_hook
                    )
                with patch(
                    "litellm.llms.vertex_ai.batches.transformation.VertexAIBatchTransformation"
                ) as mock_transformation:

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
                        "completion_window": "24h",
                    }
                    mock_transformation._get_batch_id_from_vertex_ai_batch_response.return_value = (
                        "123456789"
                    )

                    # Test the complete workflow
                    result = VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                        httpx_response=mock_httpx_response,
                        logging_obj=mock_logging_obj,
                        url_route="/v1/projects/test-project/locations/us-central1/batchPredictionJobs",
                        result="success",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        cache_hit=False,
                        user_api_key_dict={"user_id": "test-user"},
                    )

                    # Verify the complete workflow
                    assert result is not None
                    assert "kwargs" in result
                    assert (
                        result["kwargs"]["model"]
                        == "projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-flash"
                    )
                    assert result["kwargs"]["batch_id"] == "123456789"

                    # Verify all mocks were called
                    mock_get_model_id.assert_called_once()
                    mock_transformation.transform_vertex_ai_batch_response_to_openai_batch_response.assert_called_once()
                    # Note: store_unified_object_id is called asynchronously, so we can't easily verify it in this test


class TestVertexAIBatchCostCalculation:
    """Test cases for Vertex AI batch cost calculation functionality.

    The function under test (calculate_vertex_ai_batch_cost_and_usage) extracts
    usageMetadata directly from Vertex AI response dicts and calls
    batch_cost_calculator — no VertexGeminiConfig transformation involved.
    """

    def test_should_aggregate_cost_and_usage_across_responses(self):
        """Two successful responses → costs and token counts are summed."""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage

        responses = [
            {
                "response": {
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                    }
                }
            },
            {
                "response": {
                    "usageMetadata": {
                        "promptTokenCount": 8,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 11,
                    }
                }
            },
        ]

        total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
            responses, model_name="gemini-2.0-flash-001"
        )

        assert usage.prompt_tokens == 18
        assert usage.completion_tokens == 8
        assert usage.total_tokens == 26
        assert total_cost > 0, "batch_cost_calculator should return a non-zero cost"

    def test_should_skip_responses_with_null_response_body(self):
        """Failed lines (response: None) are skipped without error."""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage

        responses = [
            {
                "response": {
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                    }
                }
            },
            {"status": "JOB_STATE_FAILED", "response": None},
            {
                "response": {
                    "usageMetadata": {
                        "promptTokenCount": 8,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 11,
                    }
                }
            },
        ]

        total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
            responses, model_name="gemini-2.0-flash-001"
        )

        assert usage.prompt_tokens == 18
        assert usage.completion_tokens == 8
        assert usage.total_tokens == 26
        assert total_cost > 0

    def test_should_return_zeros_for_empty_response_list(self):
        """Empty input → zero cost and zero usage."""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage

        total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
            [], model_name="gemini-2.0-flash-001"
        )

        assert total_cost == 0.0
        assert usage.total_tokens == 0
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_should_handle_missing_usage_metadata_gracefully(self):
        """Response without usageMetadata → 0 tokens, 0 cost for that line."""
        from litellm.batches.batch_utils import calculate_vertex_ai_batch_cost_and_usage

        responses = [
            {"response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}},
        ]

        total_cost, usage = calculate_vertex_ai_batch_cost_and_usage(
            responses, model_name="gemini-2.0-flash-001"
        )

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_openai_shaped_output_records_nonzero_cost_and_usage(self):
        """
        Regression test for the bug where Vertex batch cost/usage was always 0.

        After PR #25627 (transform_file_content_response), the GCS predictions.jsonl
        is rewritten into OpenAI batch shape before the cost-tracking path sees it.
        With disable_vertex_batch_output_transformation=False (default), the cost
        dispatch must fall through to the generic aggregation path rather than
        calling calculate_vertex_ai_batch_cost_and_usage (which only reads raw
        usageMetadata fields).
        """
        import litellm
        from litellm.batches.batch_utils import calculate_batch_cost_and_usage

        openai_shaped_responses = [
            {
                "id": "batch_req_abc123",
                "custom_id": "request-1",
                "response": {
                    "status_code": 200,
                    "request_id": "chatcmpl-xyz",
                    "body": {
                        "id": "chatcmpl-xyz",
                        "object": "chat.completion",
                        "model": "gemini-2.0-flash-001",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Hello!"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                },
                "error": None,
            },
            {
                "id": "batch_req_def456",
                "custom_id": "request-2",
                "response": {
                    "status_code": 200,
                    "request_id": "chatcmpl-uvw",
                    "body": {
                        "id": "chatcmpl-uvw",
                        "object": "chat.completion",
                        "model": "gemini-2.0-flash-001",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "World!"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 3,
                            "total_tokens": 11,
                        },
                    },
                },
                "error": None,
            },
        ]

        original_flag = getattr(
            litellm, "disable_vertex_batch_output_transformation", False
        )
        try:
            litellm.disable_vertex_batch_output_transformation = False

            cost, usage, _ = await calculate_batch_cost_and_usage(
                file_content_dictionary=openai_shaped_responses,
                custom_llm_provider="vertex_ai",
                model_name="gemini-2.0-flash-001",
            )
        finally:
            litellm.disable_vertex_batch_output_transformation = original_flag

        assert (
            usage.prompt_tokens == 18
        ), f"expected 18 prompt tokens, got {usage.prompt_tokens}"
        assert (
            usage.completion_tokens == 8
        ), f"expected 8 completion tokens, got {usage.completion_tokens}"
        assert (
            usage.total_tokens == 26
        ), f"expected 26 total tokens, got {usage.total_tokens}"
        assert (
            cost > 0
        ), f"expected non-zero cost for completed Vertex batch, got {cost}"

    @pytest.mark.asyncio
    async def test_raw_vertex_output_still_works_when_transformation_disabled(self):
        """
        When disable_vertex_batch_output_transformation=True the GCS file is returned
        as raw Vertex predictions.jsonl; the specialized reader must be used.
        """
        import litellm
        from litellm.batches.batch_utils import calculate_batch_cost_and_usage

        raw_vertex_responses = [
            {
                "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                "status": "",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "Hello!"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                    },
                },
                "processed_time": "2026-01-01T00:00:00Z",
            },
        ]

        original_flag = getattr(
            litellm, "disable_vertex_batch_output_transformation", False
        )
        try:
            litellm.disable_vertex_batch_output_transformation = True

            cost, usage, _ = await calculate_batch_cost_and_usage(
                file_content_dictionary=raw_vertex_responses,
                custom_llm_provider="vertex_ai",
                model_name="gemini-2.0-flash-001",
            )
        finally:
            litellm.disable_vertex_batch_output_transformation = original_flag

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15
        assert cost > 0, "raw Vertex shape should also produce non-zero cost"
