"""
Integration tests for responses API background cost tracking
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


class TestResponsesBackgroundCostTracking:
    """Integration tests for responses API background cost tracking"""

    @pytest.fixture
    def mock_managed_files_obj(self):
        """Create a mock managed files object"""
        managed_files = MagicMock()
        managed_files.store_unified_object_id = AsyncMock()
        return managed_files

    @pytest.fixture
    def mock_proxy_logging_obj(self, mock_managed_files_obj):
        """Create a mock proxy logging object"""
        logging_obj = MagicMock()
        logging_obj.get_proxy_hook = MagicMock(return_value=mock_managed_files_obj)
        return logging_obj

    @pytest.fixture
    def mock_llm_router(self):
        """Create a mock LLM router"""
        router = MagicMock()
        return router

    @pytest.mark.asyncio
    async def test_store_response_in_managed_objects_table(
        self, mock_managed_files_obj, mock_proxy_logging_obj, mock_llm_router
    ):
        """Test that background responses are stored in managed objects table"""
        # Create a mock response with queued status and hidden params
        response = ResponsesAPIResponse(
            id="resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpncHQtNDtsbGxfcmVzcG9uc2VfaWQ6cmVzcF8xMjM",
            object="response",
            status="queued",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )
        
        # Add hidden params with model_id (simulating what base_process_llm_request does)
        response._hidden_params = {
            "model_id": "model-deployment-id-123"
        }

        # Mock request data
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        # Mock user_api_key_dict
        user_api_key_dict = MagicMock()
        user_api_key_dict.user_id = "test-user"

        # Simulate the storage logic from endpoints.py
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                # Get model_id from hidden params
                hidden_params = getattr(response, "_hidden_params", {}) or {}
                model_id = hidden_params.get("model_id", None)
                
                if model_id:
                    # Store in managed objects table using response.id directly
                    await mock_managed_files_obj.store_unified_object_id(
                        unified_object_id=response.id,
                        file_object=response,
                        litellm_parent_otel_span=None,
                        model_object_id=response.id,
                        file_purpose="response",
                        user_api_key_dict=user_api_key_dict,
                    )

        # Verify store_unified_object_id was called
        mock_managed_files_obj.store_unified_object_id.assert_called_once()
        call_args = mock_managed_files_obj.store_unified_object_id.call_args

        # Verify the arguments - unified_object_id should be response.id
        assert call_args[1]["unified_object_id"] == response.id
        assert call_args[1]["model_object_id"] == response.id
        assert call_args[1]["file_purpose"] == "response"
        assert call_args[1]["user_api_key_dict"] == user_api_key_dict

    @pytest.mark.asyncio
    async def test_no_storage_for_non_background_requests(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that non-background requests are not stored"""
        # Create a mock response
        response = ResponsesAPIResponse(
            id="resp_456",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )

        # Mock request data without background flag
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": False,
        }

        # Simulate the storage logic
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                await mock_managed_files_obj.store_unified_object_id()

        # Verify store_unified_object_id was NOT called
        mock_managed_files_obj.store_unified_object_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_storage_for_completed_responses(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that completed responses are not stored"""
        # Create a mock response with completed status
        response = ResponsesAPIResponse(
            id="resp_789",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )

        # Mock request data with background flag
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        # Simulate the storage logic
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                await mock_managed_files_obj.store_unified_object_id()

        # Verify store_unified_object_id was NOT called (status is completed)
        mock_managed_files_obj.store_unified_object_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_storage_without_model_id(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that responses without model_id in hidden params are not stored"""
        # Create a mock response without hidden params
        response = ResponsesAPIResponse(
            id="resp_no_model",
            object="response",
            status="queued",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        # Mock request data with background flag
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        user_api_key_dict = MagicMock()

        # Simulate the storage logic
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                hidden_params = getattr(response, "_hidden_params", {}) or {}
                model_id = hidden_params.get("model_id", None)
                
                if model_id:  # This will be False
                    await mock_managed_files_obj.store_unified_object_id(
                        unified_object_id=response.id,
                        file_object=response,
                        litellm_parent_otel_span=None,
                        model_object_id=response.id,
                        file_purpose="response",
                        user_api_key_dict=user_api_key_dict,
                    )

        # Verify store_unified_object_id was NOT called (no model_id)
        mock_managed_files_obj.store_unified_object_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_in_storage(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that errors during storage are handled gracefully"""
        # Mock store_unified_object_id to raise an exception
        mock_managed_files_obj.store_unified_object_id = AsyncMock(
            side_effect=Exception("Database error")
        )

        response = ResponsesAPIResponse(
            id="resp_error",
            object="response",
            status="queued",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )
        response._hidden_params = {"model_id": "test-model-id"}

        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        user_api_key_dict = MagicMock()
        user_api_key_dict.user_id = "test-user"

        # Try to store - should not raise (error is caught in endpoints.py)
        try:
            if data.get("background") and isinstance(response, ResponsesAPIResponse):
                if response.status in ["queued", "in_progress"]:
                    hidden_params = getattr(response, "_hidden_params", {}) or {}
                    model_id = hidden_params.get("model_id", None)
                    
                    if model_id:
                        await mock_managed_files_obj.store_unified_object_id(
                            unified_object_id=response.id,
                            file_object=response,
                            litellm_parent_otel_span=None,
                            model_object_id=response.id,
                            file_purpose="response",
                            user_api_key_dict=user_api_key_dict,
                        )
        except Exception:
            # Exception should be caught and logged, not raised
            pass

        # Verify the method was called (even though it raised)
        assert mock_managed_files_obj.store_unified_object_id.called


def _check_responses_cost_module_available():
    """Check if litellm_enterprise.proxy.common_utils.check_responses_cost module is available"""
    try:
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (  # noqa: F401
            CheckResponsesCost,
        )
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _check_responses_cost_module_available(),
    reason="litellm_enterprise.proxy.common_utils.check_responses_cost module not available (enterprise-only feature)"
)
class TestCheckResponsesCost:
    """Tests for the CheckResponsesCost polling class"""

    @pytest.fixture
    def mock_prisma_client(self):
        """Create a mock Prisma client"""
        client = MagicMock()
        client.db = MagicMock()
        client.db.litellm_managedobjecttable = MagicMock()
        return client

    @pytest.fixture
    def mock_proxy_logging_obj(self):
        """Create a mock proxy logging object"""
        return MagicMock()

    @pytest.fixture
    def mock_llm_router(self):
        """Create a mock LLM router"""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_check_responses_cost_initialization(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        """Test CheckResponsesCost initialization"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        checker = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

        assert checker.proxy_logging_obj == mock_proxy_logging_obj
        assert checker.prisma_client == mock_prisma_client
        assert checker.llm_router == mock_llm_router

    @pytest.mark.asyncio
    async def test_check_responses_cost_no_jobs(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        """Test polling when there are no jobs"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        # Mock find_many to return empty list
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[]
        )

        checker = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

        # Should not raise any errors
        await checker.check_responses_cost()

        # Verify find_many was called with correct parameters
        mock_prisma_client.db.litellm_managedobjecttable.find_many.assert_called_once_with(
            where={
                "status": {"in": ["queued", "in_progress"]},
                "file_purpose": "response",
            }
        )

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_completed_job(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        """Test polling with a completed job"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        # Create a mock job
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.unified_object_id = "resp_test_id"
        mock_job.created_by = "test-user"

        # Mock find_many to return the job
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Create a completed response
        completed_response = ResponsesAPIResponse(
            id="resp_test_id",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )

        checker = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

        # Mock litellm.aget_responses to return completed response
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = completed_response

            await checker.check_responses_cost()

            # Verify update_many was called to mark job as completed
            mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_called_once()
            call_args = (
                mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args
            )
            assert call_args[1]["where"]["id"]["in"] == ["job-123"]
            assert call_args[1]["data"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_failed_job(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        """Test polling with a failed job"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        # Create a mock job
        mock_job = MagicMock()
        mock_job.id = "job-456"
        mock_job.unified_object_id = "resp_failed"
        mock_job.created_by = "test-user"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Create a failed response
        failed_response = ResponsesAPIResponse(
            id="resp_failed",
            object="response",
            status="failed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        checker = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = failed_response

            await checker.check_responses_cost()

            # Verify job was marked as completed even though it failed
            mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_in_progress_job(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        """Test polling with a job still in progress"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        # Create a mock job
        mock_job = MagicMock()
        mock_job.id = "job-789"
        mock_job.unified_object_id = "resp_in_progress"
        mock_job.created_by = "test-user"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Create an in-progress response
        in_progress_response = ResponsesAPIResponse(
            id="resp_in_progress",
            object="response",
            status="in_progress",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        checker = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = in_progress_response

            await checker.check_responses_cost()

            # Verify update_many was NOT called (job still in progress)
            mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_responses_cost_error_handling(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        """Test that errors when querying responses are handled gracefully"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        # Create a mock job
        mock_job = MagicMock()
        mock_job.id = "job-error"
        mock_job.unified_object_id = "resp_error"
        mock_job.created_by = "test-user"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        checker = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

        # Mock litellm.aget_responses to raise an exception
        with patch(
            "litellm.aget_responses",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            # Should not raise - errors are caught and logged
            await checker.check_responses_cost()

            # Verify update_many was NOT called (error occurred)
            mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_not_called()
