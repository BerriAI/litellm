"""
Unit tests for CheckResponsesCost class
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


def _update_many_calls_writing(mock_prisma_client, matches_data):
    return [
        call
        for call in mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        if matches_data(call.kwargs["data"])
    ]


def _completion_calls(mock_prisma_client):
    return _update_many_calls_writing(
        mock_prisma_client, lambda data: data.get("status") == "completed"
    )


def _completed_job_ids(mock_prisma_client):
    return [call.kwargs["where"]["id"] for call in _completion_calls(mock_prisma_client)]


def _claim_calls(mock_prisma_client):
    return _update_many_calls_writing(
        mock_prisma_client, lambda data: data == {"batch_processed": True}
    )


def _release_calls(mock_prisma_client):
    return _update_many_calls_writing(
        mock_prisma_client, lambda data: data == {"batch_processed": False}
    )


def _routed_response_id(provider_response_id):
    """A LiteLLM-encoded id names a deployment, which is what sends the poll's read through the router."""
    from litellm.responses.utils import ResponsesAPIRequestUtils

    return ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai",
        model_id="deployment-xyz",
        response_id=provider_response_id,
    )


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

        instance = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
        )
        # Mock _expire_stale_rows (raw SQL) so _cleanup_stale_managed_objects
        # succeeds without a real DB.  Individual tests can override this.
        instance._expire_stale_rows = AsyncMock(return_value=0)
        return instance

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
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[]
        )

        await check_responses_cost_instance.check_responses_cost()

        # Verify find_many was called with pagination params
        find_many_call = (
            mock_prisma_client.db.litellm_managedobjecttable.find_many.call_args
        )
        assert find_many_call[1]["where"] == {
            "status": {"in": ["queued", "in_progress"]},
            "file_purpose": "response",
        }
        assert find_many_call[1]["take"] == MAX_OBJECTS_PER_POLL_CYCLE
        assert find_many_call[1]["order"] == {"created_at": "asc"}

    @pytest.mark.asyncio
    async def test_cleanup_stale_managed_objects(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Stale rows are expired via _expire_stale_rows before polling."""
        from litellm.constants import STALE_OBJECT_CLEANUP_BATCH_SIZE

        check_responses_cost_instance._expire_stale_rows = AsyncMock(return_value=5)
        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[]
        )

        await check_responses_cost_instance.check_responses_cost()

        # _expire_stale_rows should have been called with a cutoff datetime and batch size
        check_responses_cost_instance._expire_stale_rows.assert_called_once()
        call_args = check_responses_cost_instance._expire_stale_rows.call_args
        assert call_args[0][1] == STALE_OBJECT_CLEANUP_BATCH_SIZE

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_completed_response(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Test check_responses_cost with a completed response"""
        # Mock job with response ID
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_123"
        mock_job.model_object_id = "resp_test_123"
        mock_job.created_by = "test-user"
        mock_job.id = "job-123"
        mock_job.file_object = {"model": "gpt-4o", "id": "resp_test_123"}

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

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        # Run the check with mocked litellm.aget_responses
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        assert _completed_job_ids(mock_prisma_client) == ["job-123"]
        assert _release_calls(mock_prisma_client) == []

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_failed_response(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Test check_responses_cost with a failed response"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_456"
        mock_job.model_object_id = "resp_test_456"
        mock_job.created_by = "test-user"
        mock_job.id = "job-456"
        mock_job.file_object = {"model": "gpt-4o", "id": "resp_test_456"}

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

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        assert _completed_job_ids(mock_prisma_client) == ["job-456"]
        assert _release_calls(mock_prisma_client) == []

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_cancelled_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with a cancelled response"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_789"
        mock_job.model_object_id = "resp_test_789"
        mock_job.created_by = "test-user"
        mock_job.id = "job-789"
        mock_job.file_object = {"model": "gpt-4o", "id": "resp_test_789"}

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

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        assert _completed_job_ids(mock_prisma_client) == ["job-789"]
        assert _release_calls(mock_prisma_client) == []

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_in_progress_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with a response still in progress"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_in_progress"
        mock_job.model_object_id = "resp_test_in_progress"
        mock_job.created_by = "test-user"
        mock_job.id = "job-in-progress"
        mock_job.file_object = {"model": "gpt-4o", "id": "resp_test_in_progress"}

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

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # No job completion update_many — response is still in progress
        assert _completion_calls(mock_prisma_client) == []
        # Stale cleanup still ran via _expire_stale_rows
        check_responses_cost_instance._expire_stale_rows.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_queued_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with a queued response"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_queued"
        mock_job.model_object_id = "resp_test_queued"
        mock_job.created_by = "test-user"
        mock_job.id = "job-queued"
        mock_job.file_object = {"model": "gpt-4o", "id": "resp_test_queued"}

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

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # No job completion update_many — response is still queued
        assert _completion_calls(mock_prisma_client) == []
        # Stale cleanup still ran via _expire_stale_rows
        check_responses_cost_instance._expire_stale_rows.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_exception(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost handles exceptions gracefully"""
        # Mock job
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_error"
        mock_job.model_object_id = "resp_test_error"
        mock_job.created_by = "test-user"
        mock_job.id = "job-error"
        mock_job.file_object = {"model": "gpt-4o", "id": "resp_test_error"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        # Run the check with mocked exception
        with patch(
            "litellm.aget_responses",
            new_callable=AsyncMock,
            side_effect=Exception("Provider error"),
        ):
            # Should not raise, just skip the job
            await check_responses_cost_instance.check_responses_cost()

        # No job completion update_many — exception skipped the job
        assert _completion_calls(mock_prisma_client) == []
        # Stale cleanup still ran via _expire_stale_rows
        check_responses_cost_instance._expire_stale_rows.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_responses_cost_multiple_jobs(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """Test check_responses_cost with multiple jobs"""
        # Mock multiple jobs
        mock_job1 = MagicMock()
        mock_job1.unified_object_id = "resp_test_1"
        mock_job1.model_object_id = "resp_test_1"
        mock_job1.created_by = "user1"
        mock_job1.id = "job-1"
        mock_job1.file_object = {"model": "gpt-4o", "id": "resp_test_1"}

        mock_job2 = MagicMock()
        mock_job2.unified_object_id = "resp_test_2"
        mock_job2.model_object_id = "resp_test_2"
        mock_job2.created_by = "user2"
        mock_job2.id = "job-2"
        mock_job2.file_object = {"model": "gpt-4o", "id": "resp_test_2"}

        mock_job3 = MagicMock()
        mock_job3.unified_object_id = "resp_test_3"
        mock_job3.model_object_id = "resp_test_3"
        mock_job3.created_by = "user3"
        mock_job3.id = "job-3"
        mock_job3.file_object = {"model": "gpt-4o", "id": "resp_test_3"}

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

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.side_effect = [mock_response1, mock_response2, mock_response3]

            await check_responses_cost_instance.check_responses_cost()

        assert _completed_job_ids(mock_prisma_client) == ["job-1", "job-3"]

    @pytest.mark.asyncio
    async def test_encoded_response_id_is_fetched_through_router(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """
        Regression test for https://github.com/BerriAI/litellm/issues/35131

        A background response created against a deployment whose credentials only
        exist in the config (e.g. Azure api_base/api_key) must be fetched through
        the router so the deployment credentials are applied. Calling
        litellm.aget_responses directly only sees provider env vars, fails, and
        leaves the row in "queued" forever.
        """
        from litellm.responses.utils import ResponsesAPIRequestUtils

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="azure",
            model_id="deployment-abc",
            response_id="resp_upstream_123",
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = encoded_response_id
        mock_job.model_object_id = encoded_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-router"
        mock_job.file_object = {"model": "azure-gpt-5", "id": encoded_response_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_llm_router.aget_responses = AsyncMock(
            return_value=ResponsesAPIResponse(
                id=encoded_response_id,
                object="response",
                status="completed",
                created_at=int(datetime.now().timestamp()),
                output=[],
                usage=ResponseAPIUsage(
                    input_tokens=100, output_tokens=50, total_tokens=150
                ),
            )
        )

        with patch(
            "litellm.aget_responses",
            new_callable=AsyncMock,
            side_effect=AssertionError(
                "must not bypass the router for a deployment-scoped response id"
            ),
        ) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_not_called()
        assert (
            mock_llm_router.aget_responses.call_args[1]["response_id"]
            == encoded_response_id
        )

        assert _completed_job_ids(mock_prisma_client) == ["job-router"]

    @pytest.mark.asyncio
    async def test_encrypted_response_id_is_fetched_through_router(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router, monkeypatch
    ):
        """
        Rows store the *encrypted* response id when responses id security is on.
        After decryption the id still carries the deployment model_id, so the
        fetch must go through the router (issue #35131).
        """
        from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
        from litellm.responses.utils import ResponsesAPIRequestUtils
        from litellm.types.utils import SpecialEnums

        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key-for-response-ids")

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai",
            model_id="deployment-xyz",
            response_id="resp_upstream_456",
        )
        encrypted_response_id = "resp_" + str(
            encrypt_value_helper(
                value=SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                    encoded_response_id, "test-user", "test-team"
                )
            )
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = encrypted_response_id
        mock_job.model_object_id = encrypted_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-encrypted"
        mock_job.file_object = {"model": "gpt-5", "id": encrypted_response_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_llm_router.aget_responses = AsyncMock(
            return_value=ResponsesAPIResponse(
                id=encoded_response_id,
                object="response",
                status="completed",
                created_at=int(datetime.now().timestamp()),
                output=[],
                usage=None,
            )
        )

        with patch(
            "litellm.aget_responses",
            new_callable=AsyncMock,
            side_effect=AssertionError(
                "must not bypass the router for a deployment-scoped response id"
            ),
        ) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_not_called()
        assert (
            mock_llm_router.aget_responses.call_args[1]["response_id"]
            == encoded_response_id
        )
        assert _completed_job_ids(mock_prisma_client) == ["job-encrypted"]

    @pytest.mark.asyncio
    async def test_response_id_without_model_id_uses_sdk(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Ids that carry no deployment info can't be routed, so fall back to the SDK."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_plain_upstream_id"
        mock_job.model_object_id = "resp_plain_upstream_id"
        mock_job.created_by = "test-user"
        mock_job.id = "job-plain"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_plain_upstream_id"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_llm_router.aget_responses = AsyncMock(
            side_effect=AssertionError("router cannot route an id without a model_id")
        )

        mock_response = ResponsesAPIResponse(
            id="resp_plain_upstream_id",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_sdk_aget:
            mock_sdk_aget.return_value = mock_response
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_called_once()
        mock_llm_router.aget_responses.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_deployment_falls_back_to_sdk(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """
        An encoded id whose deployment was removed from the router must fall back
        to the SDK so provider env credentials can still retrieve it, instead of
        failing every poll cycle until stale expiration.
        """
        from litellm.responses.utils import ResponsesAPIRequestUtils

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai",
            model_id="deployment-deleted",
            response_id="resp_upstream_789",
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = encoded_response_id
        mock_job.model_object_id = encoded_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-missing-deployment"
        mock_job.file_object = {"model": "gpt-5", "id": encoded_response_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )
        mock_llm_router.get_deployment = MagicMock(return_value=None)
        mock_llm_router.aget_responses = AsyncMock(
            side_effect=AssertionError("router has no deployment for this model_id")
        )

        mock_response = ResponsesAPIResponse(
            id=encoded_response_id,
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_sdk_aget:
            mock_sdk_aget.return_value = mock_response
            await check_responses_cost_instance.check_responses_cost()

        mock_llm_router.get_deployment.assert_called_once_with(model_id="deployment-deleted")
        mock_llm_router.aget_responses.assert_not_called()
        mock_sdk_aget.assert_called_once()
        assert mock_sdk_aget.call_args[1]["response_id"] == encoded_response_id

        assert _completed_job_ids(mock_prisma_client) == ["job-missing-deployment"]

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_incomplete_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """'incomplete' is terminal in the Responses API, so the row must not stay queued."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_incomplete"
        mock_job.model_object_id = "resp_test_incomplete"
        mock_job.created_by = "test-user"
        mock_job.id = "job-incomplete"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_incomplete"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_response = ResponsesAPIResponse(
            id="resp_incomplete",
            object="response",
            status="incomplete",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response
            await check_responses_cost_instance.check_responses_cost()

        assert _completed_job_ids(mock_prisma_client) == ["job-incomplete"]

    @pytest.mark.asyncio
    async def test_check_responses_cost_no_model_in_file_object(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """When file_object has no 'model' key, model_name is None and metadata skips model fields."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_no_model"
        mock_job.model_object_id = "resp_test_no_model"
        mock_job.created_by = "test-user"
        mock_job.team_id = None
        mock_job.api_key = None
        mock_job.id = "job-no-model"
        mock_job.file_object = {}  # no "model" key → model_name=None branch

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_response = MagicMock()
        mock_response.status = "completed"

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response
            await check_responses_cost_instance.check_responses_cost()

        # aget_responses should be called without model metadata
        call_kwargs = mock_aget.call_args[1]
        assert "model" not in call_kwargs.get("litellm_metadata", {})
        assert "model_group" not in call_kwargs.get("litellm_metadata", {})
        assert "user_api_key_team_id" not in call_kwargs["litellm_metadata"]
        assert "user_api_key" not in call_kwargs["litellm_metadata"]
        assert "user_api_key_hash" not in call_kwargs["litellm_metadata"]

    @pytest.mark.asyncio
    async def test_poll_stamps_internal_call_origin_so_the_read_is_billed(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """A background create returns queued with no usage, so this poll's retrieval is the only
        place the job's spend is ever seen. Without the origin stamp it is priced at zero like a
        user-facing read (LIT-5602) and the job is never billed."""
        from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
        from litellm.litellm_core_utils.internal_call_metadata import (
            is_unbilled_non_inference_call,
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_billed"
        mock_job.model_object_id = "resp_test_billed"
        mock_job.created_by = "test-user"
        mock_job.team_id = "team-billed"
        mock_job.api_key = "sk-billed"
        mock_job.id = "job-billed"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_billed"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_response = MagicMock()
        mock_response.status = "completed"

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response
            await check_responses_cost_instance.check_responses_cost()

        metadata = mock_aget.call_args[1]["litellm_metadata"]
        assert metadata[INTERNAL_CALL_ORIGIN_METADATA_KEY] == "background_response_cost_poll"
        assert metadata["user_api_key_team_id"] == "team-billed"
        assert metadata["user_api_key"] == "sk-billed"
        assert metadata["user_api_key_hash"] == "sk-billed"
        assert is_unbilled_non_inference_call("aget_responses", metadata) is False
        assert is_unbilled_non_inference_call("aget_responses", None) is True

    @pytest.mark.asyncio
    async def test_job_claimed_by_another_pod_is_never_read_or_completed(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Every pod and uvicorn worker polls the same table, and the read is what writes the
        spend log, so losing the claim has to skip the read entirely or the job is billed twice."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_claimed_elsewhere"
        mock_job.model_object_id = _routed_response_id("resp_test_claimed_elsewhere")
        mock_job.created_by = "test-user"
        mock_job.id = "job-claimed-elsewhere"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_claimed_elsewhere"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
        )

        await check_responses_cost_instance.check_responses_cost()

        mock_llm_router.aget_responses.assert_not_awaited()
        assert _completion_calls(mock_prisma_client) == []
        assert _release_calls(mock_prisma_client) == []

        claim_calls = _claim_calls(mock_prisma_client)
        assert len(claim_calls) == 1
        claim_where = claim_calls[0].kwargs["where"]
        assert claim_where["id"] == "job-claimed-elsewhere"
        assert {"batch_processed": False} in claim_where["OR"]

    @pytest.mark.asyncio
    async def test_claim_is_taken_back_from_a_pod_that_died_holding_it(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A pod that dies holding a claim strands the row forever, so the lease has to outlast a
        live cycle and still fire well before stale expiry gives up on the row unbilled."""
        from litellm.constants import PROXY_BATCH_POLLING_INTERVAL
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CLAIM_ABANDONED_AFTER_POLL_CYCLES,
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_abandoned"
        mock_job.model_object_id = _routed_response_id("resp_test_abandoned")
        mock_job.created_by = "test-user"
        mock_job.id = "job-abandoned"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_abandoned"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_response = ResponsesAPIResponse(
            id="resp_abandoned",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )

        mock_llm_router.aget_responses = AsyncMock(return_value=mock_response)

        await check_responses_cost_instance.check_responses_cost()

        claim_where = _claim_calls(mock_prisma_client)[0].kwargs["where"]
        abandoned_arm = next(arm for arm in claim_where["OR"] if "updated_at" in arm)
        lease = timedelta(
            seconds=CLAIM_ABANDONED_AFTER_POLL_CYCLES * PROXY_BATCH_POLLING_INTERVAL
        )
        untouched_for = datetime.now(timezone.utc) - abandoned_arm["updated_at"]["lt"]
        assert lease <= untouched_for < lease + timedelta(seconds=30)
        assert lease > timedelta(seconds=PROXY_BATCH_POLLING_INTERVAL)

    @pytest.mark.asyncio
    async def test_claim_is_taken_before_the_billing_read_and_kept_on_a_terminal_status(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """The read prices the job, so the claim has to be taken before it, and keeping the claim
        afterwards is what stops a second pod reading and billing the same row again."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_ordering"
        mock_job.model_object_id = _routed_response_id("resp_test_ordering")
        mock_job.created_by = "test-user"
        mock_job.id = "job-ordering"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_ordering"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        writes_and_reads = []

        async def record_update_many(**kwargs):
            writes_and_reads.append(kwargs["data"])
            return 1

        async def record_read(**kwargs):
            writes_and_reads.append("provider_read")
            return ResponsesAPIResponse(
                id="resp_ordering",
                object="response",
                status="completed",
                created_at=int(datetime.now().timestamp()),
                output=[],
                usage=ResponseAPIUsage(
                    input_tokens=100, output_tokens=50, total_tokens=150
                ),
            )

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            side_effect=record_update_many
        )

        mock_llm_router.aget_responses = AsyncMock(side_effect=record_read)

        await check_responses_cost_instance.check_responses_cost()

        assert len(writes_and_reads) == 3
        assert writes_and_reads[0] == {"batch_processed": True}
        assert writes_and_reads[1] == "provider_read"
        assert writes_and_reads[2]["status"] == "completed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_status", ["queued", "in_progress"])
    async def test_non_terminal_status_releases_the_claim(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router, provider_status
    ):
        """A response the provider has not finished yet has no spend to record, so its row must go
        back to batch_processed=False; holding the claim retires it before it is ever billed."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_still_running"
        mock_job.model_object_id = _routed_response_id("resp_test_still_running")
        mock_job.created_by = "test-user"
        mock_job.id = "job-still-running"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_still_running"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_response = ResponsesAPIResponse(
            id="resp_still_running",
            object="response",
            status=provider_status,
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        mock_llm_router.aget_responses = AsyncMock(return_value=mock_response)

        await check_responses_cost_instance.check_responses_cost()

        assert _completion_calls(mock_prisma_client) == []
        release_calls = _release_calls(mock_prisma_client)
        assert len(release_calls) == 1
        assert release_calls[0].kwargs["where"] == {
            "id": "job-still-running",
            "batch_processed": True,
        }

    @pytest.mark.asyncio
    async def test_failed_provider_read_releases_the_claim(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A read that raised billed nothing, so the claim has to be handed back or the row is
        retired unbilled and no later poll cycle ever retries it."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_read_error"
        mock_job.model_object_id = _routed_response_id("resp_test_read_error")
        mock_job.created_by = "test-user"
        mock_job.id = "job-read-error"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_read_error"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=1
        )

        mock_llm_router.aget_responses = AsyncMock(side_effect=Exception("Provider error"))

        await check_responses_cost_instance.check_responses_cost()

        assert _completion_calls(mock_prisma_client) == []
        release_calls = _release_calls(mock_prisma_client)
        assert len(release_calls) == 1
        assert release_calls[0].kwargs["where"] == {
            "id": "job-read-error",
            "batch_processed": True,
        }

    @pytest.mark.asyncio
    async def test_a_job_claimed_elsewhere_does_not_block_the_next_job(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Losing one row to another pod must skip only that row: the rest of the poll page still
        has to be read and billed in the same cycle."""
        mock_job1 = MagicMock()
        mock_job1.unified_object_id = "resp_test_first"
        mock_job1.model_object_id = _routed_response_id("resp_test_first")
        mock_job1.created_by = "user1"
        mock_job1.id = "job-first"
        mock_job1.file_object = {"model": "gpt-5", "id": "resp_test_first"}

        mock_job2 = MagicMock()
        mock_job2.unified_object_id = "resp_test_second"
        mock_job2.model_object_id = _routed_response_id("resp_test_second")
        mock_job2.created_by = "user2"
        mock_job2.id = "job-second"
        mock_job2.file_object = {"model": "gpt-5", "id": "resp_test_second"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job1, mock_job2]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            side_effect=[0, 1, 1]
        )

        mock_response = ResponsesAPIResponse(
            id="resp_second",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )

        mock_llm_router.aget_responses = AsyncMock(return_value=mock_response)

        await check_responses_cost_instance.check_responses_cost()

        mock_llm_router.aget_responses.assert_awaited_once()
        assert mock_llm_router.aget_responses.await_args.kwargs["response_id"] == _routed_response_id(
            "resp_test_second"
        )

        assert _completed_job_ids(mock_prisma_client) == ["job-second"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "db_error_message",
        [
            "column LiteLLM_ManagedObjectTable.batch_processed does not exist",
            "Unknown column in where clause",
            "The column P2022 does not exist in the current database",
        ],
    )
    async def test_claim_fails_open_on_a_schema_without_the_claim_column(
        self, check_responses_cost_instance, mock_prisma_client, db_error_message
    ):
        """A deployment that never ran the batch_processed migration cannot claim anything, so it
        keeps the pre-claim behavior of billing rather than silently billing nothing."""
        mock_job = MagicMock()
        mock_job.id = "job-old-schema"

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            side_effect=Exception(db_error_message)
        )

        assert (
            await check_responses_cost_instance._claim_job_for_costing(mock_job) is True
        )

    @pytest.mark.asyncio
    async def test_claim_is_lost_when_the_database_fails_for_any_other_reason(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """A dropped connection is no proof the row is free, so the read that would bill it is
        not allowed to run."""
        mock_job = MagicMock()
        mock_job.id = "job-db-down"

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            side_effect=Exception("connection to server was lost")
        )

        assert (
            await check_responses_cost_instance._claim_job_for_costing(mock_job) is False
        )

    @pytest.mark.asyncio
    async def test_old_schema_without_the_claim_column_still_bills_and_completes(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """End to end on a pre-migration schema: the claim write fails, the response is still read
        (which is what bills it) and the row is still marked completed."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_old_schema"
        mock_job.model_object_id = _routed_response_id("resp_test_old_schema")
        mock_job.created_by = "test-user"
        mock_job.id = "job-old-schema"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_old_schema"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        async def reject_batch_processed_writes(**kwargs):
            if "batch_processed" in kwargs["data"]:
                raise Exception(
                    'column "batch_processed" of relation '
                    '"LiteLLM_ManagedObjectTable" does not exist'
                )
            return 1

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            side_effect=reject_batch_processed_writes
        )

        mock_response = ResponsesAPIResponse(
            id="resp_old_schema",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )

        mock_llm_router.aget_responses = AsyncMock(return_value=mock_response)

        await check_responses_cost_instance.check_responses_cost()

        mock_llm_router.aget_responses.assert_awaited_once()
        assert _completed_job_ids(mock_prisma_client) == ["job-old-schema"]

    @pytest.mark.asyncio
    async def test_a_failed_persist_does_not_abort_the_rest_of_the_poll_cycle(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """One row's write failing must not take the whole cycle down with it: the jobs behind it
        are already read and billed, so losing their write loses their usage for good."""
        mock_job1 = MagicMock()
        mock_job1.unified_object_id = "resp_test_persist_fails"
        mock_job1.model_object_id = _routed_response_id("resp_test_persist_fails")
        mock_job1.created_by = "user1"
        mock_job1.id = "job-persist-fails"
        mock_job1.file_object = {"model": "gpt-5", "id": "resp_test_persist_fails"}

        mock_job2 = MagicMock()
        mock_job2.unified_object_id = "resp_test_persist_works"
        mock_job2.model_object_id = _routed_response_id("resp_test_persist_works")
        mock_job2.created_by = "user2"
        mock_job2.id = "job-persist-works"
        mock_job2.file_object = {"model": "gpt-5", "id": "resp_test_persist_works"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job1, mock_job2]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            side_effect=[1, 1, Exception("deadlock detected"), 1]
        )

        mock_response = ResponsesAPIResponse(
            id="resp_persisted",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )

        mock_llm_router.aget_responses = AsyncMock(return_value=mock_response)

        await check_responses_cost_instance.check_responses_cost()

        assert mock_llm_router.aget_responses.await_count == 2
        assert _completed_job_ids(mock_prisma_client) == [
            "job-persist-fails",
            "job-persist-works",
        ]

    @pytest.mark.asyncio
    async def test_poller_fetches_the_provider_id_from_model_object_id(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router, monkeypatch
    ):
        """The row's provider id drives the fetch, not the nonce-encrypted advertised id.

        A background create advertises a freshly encrypted id per call, so unified_object_id
        is no handle on the generation.
        """
        from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
        from litellm.types.utils import SpecialEnums

        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key-for-response-ids")

        provider_response_id = _routed_response_id("resp_upstream_stable")
        stale_advertised_id = "resp_" + str(
            encrypt_value_helper(
                value=SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                    "resp_a_previous_encoding", "test-user", "test-team"
                )
            )
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = stale_advertised_id
        mock_job.model_object_id = provider_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-provider-id"
        mock_job.file_object = {"model": "gpt-5", "id": stale_advertised_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=[mock_job])
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        mock_llm_router.aget_responses = AsyncMock(
            return_value=ResponsesAPIResponse(
                id=provider_response_id,
                object="response",
                status="completed",
                created_at=int(datetime.now().timestamp()),
                output=[],
                usage=ResponseAPIUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )

        await check_responses_cost_instance.check_responses_cost()

        assert mock_llm_router.aget_responses.call_args[1]["response_id"] == provider_response_id
        assert _completed_job_ids(mock_prisma_client) == ["job-provider-id"]

    @pytest.mark.asyncio
    async def test_poller_still_reads_rows_written_before_the_provider_id_was_stored(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router, monkeypatch
    ):
        """Rows created earlier carry the encrypted advertised id in both columns."""
        from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
        from litellm.types.utils import SpecialEnums

        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key-for-response-ids")

        provider_response_id = _routed_response_id("resp_legacy_upstream")
        legacy_id = "resp_" + str(
            encrypt_value_helper(
                value=SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                    provider_response_id, "test-user", "test-team"
                )
            )
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = legacy_id
        mock_job.model_object_id = legacy_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-legacy"
        mock_job.file_object = {"model": "gpt-5", "id": legacy_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=[mock_job])
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        mock_llm_router.aget_responses = AsyncMock(
            return_value=ResponsesAPIResponse(
                id=provider_response_id,
                object="response",
                status="completed",
                created_at=int(datetime.now().timestamp()),
                output=[],
                usage=ResponseAPIUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )

        await check_responses_cost_instance.check_responses_cost()

        assert mock_llm_router.aget_responses.call_args[1]["response_id"] == provider_response_id
        assert _completed_job_ids(mock_prisma_client) == ["job-legacy"]
