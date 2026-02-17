"""
Unit tests for CheckResponsesCost class
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


class TestCheckResponsesCost:
    """Test suite for CheckResponsesCost class"""

    @pytest.fixture
    def mock_prisma_client(self):
        """Create a mock Prisma client"""
        client = MagicMock()
        client.db = MagicMock()
        client.db.litellm_managedobjecttable = MagicMock()
        return client

    @pytest.fixture
    def mock_proxy_logging_obj(self):
        """Create a mock ProxyLogging object"""
        logging_obj = MagicMock()
        logging_obj.get_proxy_hook = MagicMock(return_value=None)
        return logging_obj

    @pytest.fixture
    def mock_llm_router(self):
        """Create a mock LLM Router"""
        router = MagicMock()
        router.aget_responses = AsyncMock()
        router.get_deployment = MagicMock()
        return router

    @pytest.fixture
    def check_responses_cost_instance(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router
    ):
        """Create a CheckResponsesCost instance with mocked dependencies"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        return CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )

    def test_initialization(self, check_responses_cost_instance):
        """Test that CheckResponsesCost initializes correctly"""
        assert check_responses_cost_instance.proxy_logging_obj is not None
        assert check_responses_cost_instance.prisma_client is not None
        assert check_responses_cost_instance.llm_router is not None

    @pytest.mark.asyncio
    async def test_check_responses_cost_no_jobs(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost when there are no jobs to process"""
        # Mock empty job list
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[]
        )

        # Should not raise any errors
        await check_responses_cost_instance.check_responses_cost()

        # Verify find_many was called with correct parameters
        mock_prisma_client.db.litellm_managedobjecttable.find_many.assert_called_once_with(
            where={
                "status": {"in": ["queued", "in_progress"]},
                "file_purpose": "response",
            }
        )

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_completed_response(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Test check_responses_cost with a completed response"""
        # Mock job with response ID
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_123"
        mock_job.created_by = "test-user"
        mock_job.id = "job-123"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Mock completed response
        mock_response = ResponsesAPIResponse(
            id="resp_123",
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

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Run the check with mocked litellm.aget_responses
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # Verify the job was marked as completed
        mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_called_once()
        call_args = mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args
        assert call_args[1]["data"]["status"] == "completed"
        assert call_args[1]["where"]["id"]["in"] == ["job-123"]

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_failed_response(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Test check_responses_cost with a failed response"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_456"
        mock_job.created_by = "test-user"
        mock_job.id = "job-456"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Mock failed response
        mock_response = ResponsesAPIResponse(
            id="resp_456",
            object="response",
            status="failed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # Verify the job was marked as completed (even though response failed)
        mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_called_once()
        call_args = mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args
        assert call_args[1]["data"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_cancelled_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with a cancelled response"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_789"
        mock_job.created_by = "test-user"
        mock_job.id = "job-789"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Mock cancelled response
        mock_response = ResponsesAPIResponse(
            id="resp_789",
            object="response",
            status="cancelled",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # Verify the job was marked as completed
        mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_in_progress_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with a response still in progress"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_in_progress"
        mock_job.created_by = "test-user"
        mock_job.id = "job-in-progress"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Mock in-progress response
        mock_response = ResponsesAPIResponse(
            id="resp_in_progress",
            object="response",
            status="in_progress",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # Verify no updates were made (response still in progress)
        mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_queued_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with a queued response"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_queued"
        mock_job.created_by = "test-user"
        mock_job.id = "job-queued"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Mock queued response
        mock_response = ResponsesAPIResponse(
            id="resp_queued",
            object="response",
            status="queued",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # Verify no updates were made (response still queued)
        mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_exception(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost handles exceptions gracefully"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_error"
        mock_job.created_by = "test-user"
        mock_job.id = "job-error"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Run the check with mocked exception
        with patch(
            "litellm.aget_responses",
            new_callable=AsyncMock,
            side_effect=Exception("Provider error"),
        ):
            # Should not raise, just skip the job
            await check_responses_cost_instance.check_responses_cost()

        # Verify no updates were made (job was skipped due to error)
        mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_responses_cost_multiple_jobs(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with multiple jobs"""
        # Mock multiple jobs
        mock_job1 = MagicMock()
        mock_job1.unified_object_id = "resp_test_1"
        mock_job1.created_by = "user1"
        mock_job1.id = "job-1"

        mock_job2 = MagicMock()
        mock_job2.unified_object_id = "resp_test_2"
        mock_job2.created_by = "user2"
        mock_job2.id = "job-2"

        mock_job3 = MagicMock()
        mock_job3.unified_object_id = "resp_test_3"
        mock_job3.created_by = "user3"
        mock_job3.id = "job-3"

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job1, mock_job2, mock_job3]
        )

        # Mock responses - 2 completed, 1 in progress
        mock_response1 = ResponsesAPIResponse(
            id="resp_1",
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

        mock_response2 = ResponsesAPIResponse(
            id="resp_2",
            object="response",
            status="in_progress",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        mock_response3 = ResponsesAPIResponse(
            id="resp_3",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
            ),
        )

        # Mock update_many
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock()

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.side_effect = [mock_response1, mock_response2, mock_response3]

            await check_responses_cost_instance.check_responses_cost()

        # Verify only the 2 completed jobs were marked as complete
        mock_prisma_client.db.litellm_managedobjecttable.update_many.assert_called_once()
        call_args = mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args
        assert len(call_args[1]["where"]["id"]["in"]) == 2
        assert "job-1" in call_args[1]["where"]["id"]["in"]
        assert "job-3" in call_args[1]["where"]["id"]["in"]
        assert "job-2" not in call_args[1]["where"]["id"]["in"]
