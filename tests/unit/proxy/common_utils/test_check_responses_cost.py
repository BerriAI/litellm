"""
Unit tests for CheckResponsesCost class
"""

import asyncio
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


class _RecordingManagedObjectTable:
    def __init__(self, rows: tuple[object, ...]) -> None:
        self.rows: Final = rows
        self.updates: tuple[tuple[Mapping[str, object], Mapping[str, object]], ...] = ()

    async def find_many(self, **query: object) -> tuple[object, ...]:
        return self.rows

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int:
        self.updates = (*self.updates, (where, data))
        return len(self.rows)


@dataclass(frozen=True)
class _StoredResponse:
    id: str
    unified_object_id: str
    status: str
    file_purpose: str = "response"
    file_object: Mapping[str, object] = dataclasses.field(default_factory=lambda: {"model": "gpt-5"})
    created_by: str | None = None
    api_key: str | None = None
    team_id: str | None = None
    org_id: str | None = None
    request_tags: object = None


class _InMemoryManagedObjectTable:
    """Applies where filters and status writes the way Postgres does, so two pollers sharing it
    race over the same rows."""

    def __init__(self, rows: tuple[_StoredResponse, ...]) -> None:
        self._rows: Final = rows
        self.statuses: Final = {row.id: row.status for row in rows}

    def _matches(self, row: _StoredResponse, where: Mapping[str, object]) -> bool:
        def value_of(field: str) -> object:
            return self.statuses[row.id] if field == "status" else getattr(row, field)

        return all(
            value_of(field) in condition["in"] if isinstance(condition, dict) else value_of(field) == condition
            for field, condition in where.items()
        )

    def _current(self, where: Mapping[str, object]) -> list[_StoredResponse]:
        return [
            dataclasses.replace(row, status=self.statuses[row.id]) for row in self._rows if self._matches(row, where)
        ]

    async def find_many(self, where: Mapping[str, object], **_: object) -> list[_StoredResponse]:
        return self._current(where)

    async def find_first(self, where: Mapping[str, object]) -> _StoredResponse | None:
        return next(iter(self._current(where)), None)

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, str]) -> int:
        matched: Final = self._current(where)
        self.statuses.update({row.id: data["status"] for row in matched})
        return len(matched)


def _poller_over(table: _InMemoryManagedObjectTable):
    from litellm_enterprise.proxy.common_utils.check_responses_cost import CheckResponsesCost

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_managedobjecttable = table
    poller: Final = CheckResponsesCost(
        proxy_logging_obj=MagicMock(), prisma_client=prisma_client, llm_router=MagicMock()
    )
    poller._expire_stale_rows = AsyncMock(return_value=0)
    return poller


def _terminal_response(status: str = "completed") -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_upstream",
        object="response",
        status=status,
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    )


def _claim(job_id: str) -> tuple[Mapping[str, object], Mapping[str, object]]:
    return ({"id": job_id, "status": {"in": ["queued", "in_progress"]}}, {"status": "completed"})


def _is_billed(litellm_metadata: Mapping[str, object]) -> bool:
    from litellm.litellm_core_utils.internal_call_metadata import is_unbilled_non_inference_call

    return not is_unbilled_non_inference_call("aget_responses", dict(litellm_metadata))


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

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert [(call[1]["where"], call[1]["data"]) for call in calls] == [_claim("job-123")]
        assert [_is_billed(call[1]["litellm_metadata"]) for call in mock_aget.call_args_list] == [False, True]

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
            return_value=0
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # update_many should only contain the job completion call
        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert len(calls) == 1
        assert calls[0][1]["data"]["status"] == "completed"

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
            return_value=0
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # update_many should only contain the job completion call
        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert len(calls) == 1
        assert calls[0][1]["data"]["status"] == "completed"

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
            return_value=0
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # No job completion update_many — response is still in progress
        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert len(calls) == 0
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
            return_value=0
        )

        # Run the check
        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_response

            await check_responses_cost_instance.check_responses_cost()

        # No job completion update_many — response is still queued
        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert len(calls) == 0
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
        mock_job.created_by = "test-user"
        mock_job.id = "job-error"
        mock_job.file_object = {"model": "gpt-4o", "id": "resp_test_error"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )

        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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
        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert len(calls) == 0
        # Stale cleanup still ran via _expire_stale_rows
        check_responses_cost_instance._expire_stale_rows.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_responses_cost_marks_404_response_stale_expired(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A provider 404 on a response polled through its deployment marks the row stale_expired."""
        import litellm
        from litellm.responses.utils import ResponsesAPIRequestUtils

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai",
            model_id="deployment-404",
            response_id="resp_upstream_404",
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = encoded_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-404"
        mock_job.file_object = {"model": "gpt-4o", "id": encoded_response_id}

        mock_llm_router.get_deployment.return_value = {"model_id": "deployment-404"}
        mock_llm_router.aget_responses = AsyncMock(
            side_effect=litellm.NotFoundError(
                message="Response with id 'resp_upstream_404' not found.", model="gpt-5", llm_provider="openai"
            )
        )

        table = _RecordingManagedObjectTable(rows=(mock_job,))
        mock_prisma_client.db.litellm_managedobjecttable = table

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_not_called()
        mock_llm_router.get_deployment.assert_called_once_with(model_id="deployment-404")
        assert mock_llm_router.aget_responses.call_args.kwargs["response_id"] == encoded_response_id
        assert table.updates == (({"id": {"in": ["job-404"]}}, {"status": "stale_expired"}),)

    @pytest.mark.asyncio
    async def test_check_responses_cost_marks_mapped_provider_404_stale_expired(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """The provider's own 404 body, mapped the way the GET path maps it, still expires the row."""
        import json

        import openai

        import litellm
        from litellm.llms.openai.common_utils import OpenAIError
        from litellm.responses.utils import ResponsesAPIRequestUtils

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai",
            model_id="deployment-404-mapped",
            response_id="resp_upstream_404_mapped",
        )
        provider_body = json.dumps(
            {
                "error": {
                    "message": "Response with id 'resp_upstream_404_mapped' not found.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            }
        )
        with pytest.raises(openai.APIStatusError) as mapped:
            raise litellm.exception_type(
                model="gpt-5.5",
                custom_llm_provider="openai",
                original_exception=OpenAIError(message=provider_body, status_code=404),
                completion_kwargs={},
                extra_kwargs={},
            )
        assert mapped.value.status_code == 404

        mock_job = MagicMock()
        mock_job.unified_object_id = encoded_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-404-mapped"
        mock_job.file_object = {"model": "gpt-5.5", "id": encoded_response_id}

        mock_llm_router.get_deployment.return_value = {"model_id": "deployment-404-mapped"}
        mock_llm_router.aget_responses = AsyncMock(side_effect=mapped.value)

        table = _RecordingManagedObjectTable(rows=(mock_job,))
        mock_prisma_client.db.litellm_managedobjecttable = table

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_not_called()
        assert table.updates == (({"id": {"in": ["job-404-mapped"]}}, {"status": "stale_expired"}),)

    @pytest.mark.asyncio
    async def test_check_responses_cost_404_without_router_deployment_keeps_row_for_retry(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A 404 while polling without a resolved deployment skips the row for retry."""
        import litellm
        from litellm.responses.utils import ResponsesAPIRequestUtils

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai",
            model_id="deployment-gone",
            response_id="resp_upstream_gone",
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = encoded_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-404-fallback"
        mock_job.file_object = {"model": "gpt-4o", "id": encoded_response_id}

        mock_llm_router.get_deployment.return_value = None

        table = _RecordingManagedObjectTable(rows=(mock_job,))
        mock_prisma_client.db.litellm_managedobjecttable = table

        with patch(
            "litellm.aget_responses",
            new_callable=AsyncMock,
            side_effect=litellm.NotFoundError(
                message="Response not found", model="gpt-5", llm_provider="openai"
            ),
        ) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_awaited_once()
        mock_llm_router.aget_responses.assert_not_called()
        assert table.updates == ()

    @pytest.mark.asyncio
    async def test_check_responses_cost_404_not_naming_the_response_keeps_row_for_retry(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A 404 that does not name the response (a gateway or a reconfigured deployment) is retried, not expired."""
        import litellm
        from litellm.responses.utils import ResponsesAPIRequestUtils

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="azure",
            model_id="deployment-renamed",
            response_id="resp_upstream_still_there",
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = encoded_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-404-other"
        mock_job.file_object = {"model": "azure-gpt-5", "id": encoded_response_id}

        mock_llm_router.get_deployment.return_value = {"model_id": "deployment-renamed"}
        mock_llm_router.aget_responses = AsyncMock(
            side_effect=litellm.NotFoundError(
                message="Resource not found", model="gpt-5", llm_provider="azure"
            )
        )

        table = _RecordingManagedObjectTable(rows=(mock_job,))
        mock_prisma_client.db.litellm_managedobjecttable = table

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_not_called()
        assert mock_llm_router.aget_responses.call_args.kwargs["response_id"] == encoded_response_id
        assert table.updates == ()

    @pytest.mark.asyncio
    async def test_check_responses_cost_deployment_lookup_error_skips_only_that_job(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A deployment lookup that raises skips its own job and the cycle still records the next job."""
        from litellm.responses.utils import ResponsesAPIRequestUtils

        broken_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai", model_id="deployment-broken", response_id="resp_upstream_broken"
        )
        healthy_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai", model_id="deployment-healthy", response_id="resp_upstream_healthy"
        )

        broken_job = MagicMock()
        broken_job.unified_object_id = broken_response_id
        broken_job.created_by = "test-user"
        broken_job.id = "job-broken"
        broken_job.file_object = {"model": "gpt-5.5", "id": broken_response_id}

        healthy_job = MagicMock()
        healthy_job.unified_object_id = healthy_response_id
        healthy_job.created_by = "test-user"
        healthy_job.id = "job-healthy"
        healthy_job.file_object = {"model": "gpt-5.5", "id": healthy_response_id}

        mock_llm_router.get_deployment.side_effect = [
            Exception("Model invalid format - <class 'str'>"),
            {"model_id": "deployment-healthy"},
            {"model_id": "deployment-healthy"},
        ]
        mock_llm_router.aget_responses = AsyncMock(
            return_value=ResponsesAPIResponse(
                id=healthy_response_id,
                object="response",
                status="completed",
                created_at=int(datetime.now().timestamp()),
                output=[],
                usage=ResponseAPIUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
        )

        table = _RecordingManagedObjectTable(rows=(broken_job, healthy_job))
        mock_prisma_client.db.litellm_managedobjecttable = table

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_not_called()
        assert [call.kwargs["response_id"] for call in mock_llm_router.aget_responses.call_args_list] == [
            healthy_response_id,
            healthy_response_id,
        ]
        assert table.updates == (_claim("job-healthy"),)

    @pytest.mark.asyncio
    async def test_check_responses_cost_non_404_error_keeps_row_for_retry(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """A non-404 provider error through a resolved deployment skips the job so it is retried next cycle."""
        import litellm
        from litellm.responses.utils import ResponsesAPIRequestUtils

        encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider="openai",
            model_id="deployment-500",
            response_id="resp_upstream_500",
        )

        mock_job = MagicMock()
        mock_job.unified_object_id = encoded_response_id
        mock_job.created_by = "test-user"
        mock_job.id = "job-500"
        mock_job.file_object = {"model": "gpt-4o", "id": encoded_response_id}

        mock_llm_router.get_deployment.return_value = {"model_id": "deployment-500"}
        mock_llm_router.aget_responses = AsyncMock(
            side_effect=litellm.InternalServerError(
                message="boom", model="gpt-5", llm_provider="openai"
            )
        )

        table = _RecordingManagedObjectTable(rows=(mock_job,))
        mock_prisma_client.db.litellm_managedobjecttable = table

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_sdk_aget:
            await check_responses_cost_instance.check_responses_cost()

        mock_sdk_aget.assert_not_called()
        mock_llm_router.aget_responses.assert_awaited_once()
        assert table.updates == ()

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
        mock_job1.file_object = {"model": "gpt-4o", "id": "resp_test_1"}

        mock_job2 = MagicMock()
        mock_job2.unified_object_id = "resp_test_2"
        mock_job2.created_by = "user2"
        mock_job2.id = "job-2"
        mock_job2.file_object = {"model": "gpt-4o", "id": "resp_test_2"}

        mock_job3 = MagicMock()
        mock_job3.unified_object_id = "resp_test_3"
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

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.side_effect = [mock_response1, mock_response1, mock_response2, mock_response3, mock_response3]

            await check_responses_cost_instance.check_responses_cost()

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert [(call[1]["where"], call[1]["data"]) for call in calls] == [_claim("job-1"), _claim("job-3")]
        assert [_is_billed(call[1]["litellm_metadata"]) for call in mock_aget.call_args_list] == [
            False,
            True,
            False,
            False,
            True,
        ]

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
        mock_job.created_by = "test-user"
        mock_job.id = "job-router"
        mock_job.file_object = {"model": "azure-gpt-5", "id": encoded_response_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert [(call[1]["where"], call[1]["data"]) for call in calls] == [_claim("job-router")]

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
        mock_job.created_by = "test-user"
        mock_job.id = "job-encrypted"
        mock_job.file_object = {"model": "gpt-5", "id": encrypted_response_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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
        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert [(call[1]["where"], call[1]["data"]) for call in calls] == [_claim("job-encrypted")]

    @pytest.mark.asyncio
    async def test_response_id_without_model_id_uses_sdk(
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router
    ):
        """Ids that carry no deployment info can't be routed, so fall back to the SDK."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_plain_upstream_id"
        mock_job.created_by = "test-user"
        mock_job.id = "job-plain"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_plain_upstream_id"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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
        mock_job.created_by = "test-user"
        mock_job.id = "job-missing-deployment"
        mock_job.file_object = {"model": "gpt-5", "id": encoded_response_id}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert [(call[1]["where"], call[1]["data"]) for call in calls] == [_claim("job-missing-deployment")]

    @pytest.mark.asyncio
    async def test_check_responses_cost_with_incomplete_response(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """'incomplete' is terminal in the Responses API, so the row must not stay queued."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_incomplete"
        mock_job.created_by = "test-user"
        mock_job.id = "job-incomplete"
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_incomplete"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(
            return_value=[mock_job]
        )
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(
            return_value=0
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

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert [(call[1]["where"], call[1]["data"]) for call in calls] == [_claim("job-incomplete")]

    @pytest.mark.asyncio
    async def test_check_responses_cost_no_model_in_file_object(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """When file_object has no 'model' key, model_name is None and metadata skips model fields."""
        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_no_model"
        mock_job.created_by = "test-user"
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
        mock_job.created_by = "test-user"
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
        assert is_unbilled_non_inference_call("aget_responses", metadata) is False
        assert is_unbilled_non_inference_call("aget_responses", None) is True

    @pytest.mark.asyncio
    async def test_poll_bills_the_read_to_the_creating_key_team_org_and_tags(
        self, check_responses_cost_instance, mock_prisma_client
    ):
        """The poll's read is the one billed read of a background job, so its spend row must name
        the key, team, org, user, and tags of the create it settles, read off the managed object row,
        never the poller itself."""
        from types import SimpleNamespace

        mock_job = MagicMock()
        mock_job.unified_object_id = "resp_test_attributed"
        mock_job.id = "job-attributed"
        mock_job.created_by = "alice"
        mock_job.api_key = "hash-alice"
        mock_job.team_id = "team-alpha"
        mock_job.org_id = "org-1"
        mock_job.request_tags = ["env:prod"]
        mock_job.file_object = {"model": "gpt-5", "id": "resp_test_attributed"}

        mock_prisma_client.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=[mock_job])
        mock_prisma_client.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=1)
        mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
            return_value=SimpleNamespace(key_alias="prod-key", organization_id=None)
        )
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=SimpleNamespace(team_alias="Team Alpha", organization_id=None)
        )
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=SimpleNamespace(user_email="alice@example.com", user_alias=None)
        )

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = MagicMock(status="completed")
            await check_responses_cost_instance.check_responses_cost()

        metadata = mock_aget.call_args[1]["litellm_metadata"]
        assert metadata["user_api_key"] == "hash-alice"
        assert metadata["user_api_key_hash"] == "hash-alice"
        assert metadata["user_api_key_user_id"] == "alice"
        assert metadata["user_api_key_user_email"] == "alice@example.com"
        assert metadata["user_api_key_team_id"] == "team-alpha"
        assert metadata["user_api_key_org_id"] == "org-1"
        assert metadata["user_api_key_alias"] == "prod-key"
        assert metadata["user_api_key_team_alias"] == "Team Alpha"
        assert metadata["tags"] == ["env:prod"]
        assert metadata["model_group"] == "gpt-5"


@pytest.mark.asyncio
async def test_two_pollers_racing_over_one_completed_response_bill_it_once():
    """Every worker and pod runs its own poll, so two of them can pick up the same completed
    response in one window; only the one that moves the row out of a pending status bills it."""
    table: Final = _InMemoryManagedObjectTable(
        rows=(_StoredResponse(id="job-race", unified_object_id="resp_race", status="in_progress"),)
    )
    reads: Final[list[Mapping[str, object]]] = []

    async def provider_read(response_id: str, litellm_metadata: Mapping[str, object]) -> ResponsesAPIResponse:
        reads.append(litellm_metadata)
        await asyncio.sleep(0)
        return _terminal_response()

    with patch("litellm.aget_responses", side_effect=provider_read):
        await asyncio.gather(_poller_over(table).check_responses_cost(), _poller_over(table).check_responses_cost())

    assert [_is_billed(metadata) for metadata in reads].count(True) == 1
    assert len(reads) == 3
    assert table.statuses == {"job-race": "completed"}


@pytest.mark.asyncio
async def test_a_failed_billing_read_puts_the_response_back_for_the_next_poll():
    table: Final = _InMemoryManagedObjectTable(
        rows=(_StoredResponse(id="job-retry", unified_object_id="resp_retry", status="queued"),)
    )

    async def billing_read_fails_once(response_id: str, litellm_metadata: Mapping[str, object]) -> ResponsesAPIResponse:
        if _is_billed(litellm_metadata) and billed_attempts.pop():
            raise RuntimeError("provider unavailable")
        return _terminal_response()

    billed_attempts: Final = [False, True]
    poller: Final = _poller_over(table)
    with patch("litellm.aget_responses", side_effect=billing_read_fails_once):
        await poller.check_responses_cost()
        assert table.statuses == {"job-retry": "queued"}
        await poller.check_responses_cost()

    assert billed_attempts == []
    assert table.statuses == {"job-retry": "completed"}


@pytest.mark.asyncio
async def test_only_the_first_read_that_sees_a_response_finished_bills_it():
    """A client polling its background response bills it on the first read that sees it finished,
    under the creating key, so it is billed even if the provider forgets it before the poll runs;
    reads before and after that one stay free."""
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-read",
                unified_object_id="resp_read",
                status="queued",
                created_by="alice",
                api_key="hash-alice",
                team_id="team-alpha",
            ),
        )
    )
    poller: Final = _poller_over(table)
    poller.prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    poller.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    poller.prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    with patch("litellm.aget_responses", new_callable=AsyncMock, return_value=_terminal_response()) as provider_read:
        await poller.bill_if_finished(unified_object_id="resp_read", status="in_progress")
        await poller.bill_if_finished(unified_object_id="resp_read", status="completed")
        await poller.bill_if_finished(unified_object_id="resp_read", status="completed")
        await poller.check_responses_cost()

    billed_reads: Final = [call.kwargs["litellm_metadata"] for call in provider_read.call_args_list]
    assert [_is_billed(metadata) for metadata in billed_reads] == [True]
    assert billed_reads[0]["user_api_key_hash"] == "hash-alice"
    assert billed_reads[0]["user_api_key_team_id"] == "team-alpha"
    assert table.statuses == {"job-read": "completed"}
