"""
Unit tests for CheckResponsesCost class
"""

import asyncio
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from litellm_enterprise.proxy.common_utils.check_responses_cost import BilledBackgroundResponse

from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo


class _RecordingBilledLogger:
    """Stands in for the spend logger so a test reads the billing events a poller wrote, and can
    make the next writes raise."""

    def __init__(self, failures: tuple[BaseException, ...] = (), cost: float = 0.25) -> None:
        self.billed: tuple[BilledBackgroundResponse, ...] = ()
        self._failures = failures
        self.cost: Final = cost

    async def __call__(self, billed: BilledBackgroundResponse) -> float:
        await asyncio.sleep(0)
        if self._failures:
            failure, *remaining = self._failures
            self._failures = tuple(remaining)
            raise failure
        self.billed = (*self.billed, billed)
        return self.cost


def _deployment(model_id: str) -> Deployment:
    return Deployment(
        model_name="gpt-5.6",
        litellm_params=LiteLLM_Params(model="openai/gpt-5.6"),
        model_info=ModelInfo(id=model_id),
    )


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
    model_object_id: str = ""
    file_purpose: str = "response"
    file_object: Mapping[str, object] = dataclasses.field(default_factory=lambda: {"model": "gpt-5"})
    created_by: str | None = None
    api_key: str | None = None
    team_id: str | None = None
    org_id: str | None = None
    request_tags: object = None
    end_user_id: str | None = None


class _InMemoryManagedObjectTable:
    """Applies where filters and status writes the way Postgres does, so two pollers sharing it
    race over the same rows."""

    def __init__(self, rows: tuple[_StoredResponse, ...]) -> None:
        self._rows: Final = rows
        self.statuses: Final = {row.id: row.status for row in rows}

    def _matches(self, row: _StoredResponse, where: Mapping[str, object]) -> bool:
        def value_of(field: str) -> object:
            return self.statuses[row.id] if field == "status" else getattr(row, field)

        def holds(field: str, condition: object) -> bool:
            if field == "OR" and isinstance(condition, list):
                return any(self._matches(row, alternative) for alternative in condition)
            if isinstance(condition, dict):
                return value_of(field) in condition["in"]
            return value_of(field) == condition

        return all(holds(field, condition) for field, condition in where.items())

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


def _poller_over(table: _InMemoryManagedObjectTable, billed_logger: _RecordingBilledLogger):
    from litellm_enterprise.proxy.common_utils.check_responses_cost import CheckResponsesCost

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_managedobjecttable = table
    poller: Final = CheckResponsesCost(
        proxy_logging_obj=MagicMock(),
        prisma_client=prisma_client,
        llm_router=MagicMock(),
        log_billed_response=billed_logger,
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


_READ_OF_A_BACKGROUND_RESPONSE: Final = MappingProxyType({"background": True})


def _is_billed(litellm_metadata: Mapping[str, object]) -> bool:
    from litellm.litellm_core_utils.internal_call_metadata import is_unbilled_non_inference_call

    return not is_unbilled_non_inference_call(
        "aget_responses", dict(litellm_metadata), _READ_OF_A_BACKGROUND_RESPONSE
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
        router.get_deployment = MagicMock(side_effect=lambda model_id: _deployment(model_id))
        return router

    @pytest.fixture
    def billed_logger(self) -> _RecordingBilledLogger:
        return _RecordingBilledLogger()

    @pytest.fixture
    def check_responses_cost_instance(
        self, mock_proxy_logging_obj, mock_prisma_client, mock_llm_router, billed_logger
    ):
        """Create a CheckResponsesCost instance with mocked dependencies"""
        from litellm_enterprise.proxy.common_utils.check_responses_cost import (
            CheckResponsesCost,
        )

        instance = CheckResponsesCost(
            proxy_logging_obj=mock_proxy_logging_obj,
            prisma_client=mock_prisma_client,
            llm_router=mock_llm_router,
            log_billed_response=billed_logger,
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
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router, billed_logger
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
        assert [_is_billed(call[1]["litellm_metadata"]) for call in mock_aget.call_args_list] == [False]
        assert [(billed.response.id, billed.response.usage) for billed in billed_logger.billed] == [
            ("resp_test_123", mock_response.usage)
        ]

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

        mock_llm_router.get_deployment.side_effect = None
        mock_llm_router.get_deployment.return_value = _deployment("deployment-404")
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

        mock_llm_router.get_deployment.side_effect = None
        mock_llm_router.get_deployment.return_value = _deployment("deployment-404-mapped")
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

        mock_llm_router.get_deployment.side_effect = None
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

        mock_llm_router.get_deployment.side_effect = None
        mock_llm_router.get_deployment.return_value = _deployment("deployment-renamed")
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
        self, check_responses_cost_instance, mock_prisma_client, mock_llm_router, billed_logger
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
            _deployment("deployment-healthy"),
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
        ]
        assert table.updates == (_claim("job-healthy"),)
        assert [billed.response.id for billed in billed_logger.billed] == [healthy_response_id]

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

        mock_llm_router.get_deployment.side_effect = None
        mock_llm_router.get_deployment.return_value = _deployment("deployment-500")
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
        self, check_responses_cost_instance, mock_prisma_client, billed_logger
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
            mock_aget.side_effect = [mock_response1, mock_response2, mock_response3]

            await check_responses_cost_instance.check_responses_cost()

        calls = (
            mock_prisma_client.db.litellm_managedobjecttable.update_many.call_args_list
        )
        assert [(call[1]["where"], call[1]["data"]) for call in calls] == [_claim("job-1"), _claim("job-3")]
        assert [_is_billed(call[1]["litellm_metadata"]) for call in mock_aget.call_args_list] == [False, False, False]
        assert [(billed.response.id, billed.response.usage) for billed in billed_logger.billed] == [
            ("resp_test_1", mock_response1.usage),
            ("resp_test_3", mock_response3.usage),
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
    async def test_poll_stamps_the_billing_event_so_it_is_priced_and_its_status_read_is_free(
        self, check_responses_cost_instance, mock_prisma_client, billed_logger
    ):
        """A background create returns queued with no usage, so the event the poll logs from the
        finished response is the only place the job's spend is ever seen. Without the origin stamp
        it is priced at zero like a user-facing read (LIT-5602) and the job is never billed."""
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

        assert [_is_billed(call[1]["litellm_metadata"]) for call in mock_aget.call_args_list] == [False]
        (billed,) = billed_logger.billed
        assert billed.metadata[INTERNAL_CALL_ORIGIN_METADATA_KEY] == "background_response_cost_poll"
        assert (
            is_unbilled_non_inference_call("aget_responses", dict(billed.metadata), _READ_OF_A_BACKGROUND_RESPONSE)
            is False
        )

    @pytest.mark.asyncio
    async def test_poll_bills_the_response_to_the_creating_key_team_org_and_tags(
        self, check_responses_cost_instance, mock_prisma_client, billed_logger
    ):
        """The poll's billing event is the one billed event of a background job, so its spend row must name
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
            return_value=SimpleNamespace(key_alias="prod-key", organization_id=None, spend=4.0, max_budget=5.0)
        )
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=SimpleNamespace(team_alias="Team Alpha", organization_id=None, spend=40.0, max_budget=50.0)
        )
        mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
            return_value=SimpleNamespace(user_email="alice@example.com", user_alias=None, spend=1.0, max_budget=2.0)
        )

        with patch("litellm.aget_responses", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = MagicMock(status="completed")
            await check_responses_cost_instance.check_responses_cost()

        (billed,) = billed_logger.billed
        metadata = billed.metadata
        assert metadata["user_api_key"] == "hash-alice"
        assert metadata["user_api_key_hash"] == "hash-alice"
        assert metadata["user_api_key_user_id"] == "alice"
        assert metadata["user_api_key_user_email"] == "alice@example.com"
        assert metadata["user_api_key_team_id"] == "team-alpha"
        assert metadata["user_api_key_org_id"] == "org-1"
        assert metadata["user_api_key_alias"] == "prod-key"
        assert metadata["user_api_key_team_alias"] == "Team Alpha"
        assert (metadata["user_api_key_spend"], metadata["user_api_key_max_budget"]) == (4.0, 5.0)
        assert (metadata["user_api_key_team_spend"], metadata["user_api_key_team_max_budget"]) == (40.0, 50.0)
        assert (metadata["user_api_key_user_spend"], metadata["user_api_key_user_max_budget"]) == (1.0, 2.0)
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
    billed_logger: Final = _RecordingBilledLogger()

    async def provider_read(response_id: str, litellm_metadata: Mapping[str, object]) -> ResponsesAPIResponse:
        reads.append(litellm_metadata)
        await asyncio.sleep(0)
        return _terminal_response()

    with patch("litellm.aget_responses", side_effect=provider_read):
        await asyncio.gather(
            _poller_over(table, billed_logger).check_responses_cost(),
            _poller_over(table, billed_logger).check_responses_cost(),
        )

    assert [_is_billed(metadata) for metadata in reads] == [False, False]
    assert len(billed_logger.billed) == 1
    assert table.statuses == {"job-race": "completed"}


@pytest.mark.asyncio
async def test_a_failed_billing_event_puts_the_response_back_for_the_next_poll():
    table: Final = _InMemoryManagedObjectTable(
        rows=(_StoredResponse(id="job-retry", unified_object_id="resp_retry", status="queued"),)
    )
    billed_logger: Final = _RecordingBilledLogger(failures=(RuntimeError("spend log unavailable"),))
    poller: Final = _poller_over(table, billed_logger)
    with patch("litellm.aget_responses", new_callable=AsyncMock, return_value=_terminal_response()):
        await poller.check_responses_cost()
        assert table.statuses == {"job-retry": "queued"}
        assert billed_logger.billed == ()
        await poller.check_responses_cost()

    assert len(billed_logger.billed) == 1
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
    billed_logger: Final = _RecordingBilledLogger()
    poller: Final = _poller_over(table, billed_logger)
    poller.prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    poller.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    poller.prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    finished: Final = _terminal_response("completed")

    with patch("litellm.aget_responses", side_effect=AssertionError("billing must not read the provider again")):
        costs: Final = (
            await poller.bill_if_finished(response_id="resp_read", response=_terminal_response("in_progress")),
            await poller.bill_if_finished(response_id="resp_read", response=finished),
            await poller.bill_if_finished(response_id="resp_read", response=finished),
        )
        await poller.check_responses_cost()

    assert costs == (None, billed_logger.cost, None)
    (billed,) = billed_logger.billed
    assert billed.response.usage == finished.usage
    assert billed.response.id == "resp_read"
    assert billed.metadata["user_api_key_hash"] == "hash-alice"
    assert billed.metadata["user_api_key_team_id"] == "team-alpha"
    assert table.statuses == {"job-read": "completed"}


_ROTATION_SALT_KEY: Final = "check-responses-cost-unit-test-salt"


@pytest.mark.asyncio
async def test_a_read_through_a_reissued_response_id_bills_the_stored_response(monkeypatch: pytest.MonkeyPatch):
    """The proxy re-encrypts a response id on every read, so a client polling with the id its last
    read returned still bills the row stored under the id the create returned."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity

    monkeypatch.setenv("LITELLM_SALT_KEY", _ROTATION_SALT_KEY)
    issuer: Final = ResponsesIDSecurity(general_settings_reader=lambda: {}, signing_key_reader=lambda: _ROTATION_SALT_KEY)

    def issued_id() -> str:
        return issuer._encrypt_response_id(
            ResponsesAPIResponse(id="resp_provider_job", created_at=1, output=[], status="queued"),
            UserAPIKeyAuth(user_id="alice"),
        ).id

    created_id: Final = issued_id()
    reissued_id: Final = issued_id()
    assert created_id != reissued_id
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-rotated",
                unified_object_id=created_id,
                model_object_id="resp_provider_job",
                status="queued",
            ),
        )
    )

    billed_logger: Final = _RecordingBilledLogger()
    with patch("litellm.aget_responses", side_effect=AssertionError("billing must not read the provider again")):
        await _poller_over(table, billed_logger).bill_if_finished(response_id=reissued_id, response=_terminal_response())

    assert len(billed_logger.billed) == 1
    assert table.statuses == {"job-rotated": "completed"}


@pytest.mark.asyncio
async def test_a_cancelled_billing_event_puts_the_response_back_for_the_next_poll():
    """A client that disconnects mid-read cancels the billing event, so the claim is released and
    the next poll still bills the response."""
    table: Final = _InMemoryManagedObjectTable(
        rows=(_StoredResponse(id="job-cancelled", unified_object_id="resp_cancelled", status="in_progress"),)
    )
    billed_logger: Final = _RecordingBilledLogger(failures=(asyncio.CancelledError(),))

    with pytest.raises(asyncio.CancelledError):
        await _poller_over(table, billed_logger).bill_if_finished(
            response_id="resp_cancelled", response=_terminal_response()
        )

    assert billed_logger.billed == ()
    assert table.statuses == {"job-cancelled": "in_progress"}


@pytest.mark.asyncio
async def test_the_billing_event_carries_the_creators_end_user():
    """The charge belongs to the end user the create named, whoever reads the response later."""
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-end-user", unified_object_id="resp_end_user", status="queued", end_user_id="customer-42"
            ),
        )
    )
    billed_logger: Final = _RecordingBilledLogger()

    await _poller_over(table, billed_logger).bill_if_finished(
        response_id="resp_end_user", response=_terminal_response()
    )

    assert table.statuses == {"job-end-user": "completed"}
    (billed,) = billed_logger.billed
    assert billed.metadata["user_api_key_end_user_id"] == "customer-42"


@pytest.mark.asyncio
async def test_settling_from_the_deployment_bills_a_finished_response_once_from_a_free_fetch():
    """A read the caller retargets, or a delete, bills from a fetch through the configured
    deployment, since the client controls the body it would otherwise hand in."""
    table: Final = _InMemoryManagedObjectTable(
        rows=(_StoredResponse(id="job-settle", unified_object_id="resp_settle", status="in_progress"),)
    )
    billed_logger: Final = _RecordingBilledLogger()
    poller: Final = _poller_over(table, billed_logger)
    fetched: Final = _terminal_response()

    with patch("litellm.aget_responses", new_callable=AsyncMock, return_value=fetched) as fetch:
        cost: Final = await poller.bill_from_deployment("resp_settle")
    with patch("litellm.aget_responses", side_effect=AssertionError("a billed response is never fetched again")):
        assert await poller.bill_from_deployment("resp_settle") is None

    assert cost == billed_logger.cost
    assert [_is_billed(call.kwargs["litellm_metadata"]) for call in fetch.call_args_list] == [False]
    (billed,) = billed_logger.billed
    assert billed.response.usage == fetched.usage
    assert table.statuses == {"job-settle": "completed"}


@pytest.mark.asyncio
async def test_settling_from_the_deployment_leaves_a_running_response_for_the_poll():
    table: Final = _InMemoryManagedObjectTable(
        rows=(_StoredResponse(id="job-running", unified_object_id="resp_running", status="queued"),)
    )
    billed_logger: Final = _RecordingBilledLogger()

    with patch("litellm.aget_responses", new_callable=AsyncMock, return_value=_terminal_response("in_progress")):
        assert await _poller_over(table, billed_logger).bill_from_deployment("resp_running") is None

    assert billed_logger.billed == ()
    assert table.statuses == {"job-running": "queued"}


class _SpendCapture:
    def __init__(self) -> None:
        self.payloads: tuple[Mapping[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        payload = kwargs["standard_logging_object"]
        assert isinstance(payload, Mapping)
        self.payloads = (*self.payloads, payload)


def _spend_capture_logger():
    from litellm.integrations.custom_logger import CustomLogger

    class _CapturingLogger(_SpendCapture, CustomLogger):
        pass

    return _CapturingLogger()


_PRICED_COST: Final = 100 * 0.001 + 50 * 0.002


@dataclass(frozen=True)
class _PricedBilling:
    table: _InMemoryManagedObjectTable
    response_id: str
    cost: float | None


async def _bill_through_a_priced_deployment(read: ResponsesAPIResponse) -> _PricedBilling:
    import litellm
    from litellm_enterprise.proxy.common_utils.check_responses_cost import CheckResponsesCost

    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.responses.utils import ResponsesAPIRequestUtils

    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id="deployment-custom", response_id="resp_upstream"
    )
    table: Final = _InMemoryManagedObjectTable(
        rows=(_StoredResponse(id="job-priced", unified_object_id=response_id, status="queued", api_key="hash-alice"),)
    )
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_managedobjecttable = table
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "priced-gpt",
                "litellm_params": {
                    "model": "openai/gpt-5.6",
                    "api_key": "sk-unused",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
                "model_info": {"id": "deployment-custom"},
            }
        ]
    )
    cost: Final = await CheckResponsesCost(
        proxy_logging_obj=MagicMock(), prisma_client=prisma_client, llm_router=router
    ).bill_if_finished(response_id=response_id, response=read)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    return _PricedBilling(table=table, response_id=response_id, cost=cost)


@pytest.mark.asyncio
async def test_the_billing_event_prices_the_fetched_usage_at_the_deployment_rates(monkeypatch: pytest.MonkeyPatch):
    """Billing logs the body the read already fetched instead of reading it again, so the event must
    carry the deployment's own pricing (100 input tokens at 0.001 plus 50 output tokens at 0.002 is
    0.2), and neither take the read's zero price nor change the response the reader gets back."""
    import litellm

    capture: Final = _spend_capture_logger()
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    read: Final = _terminal_response()
    read.hidden_params = {"response_cost": 0.0}

    billing: Final = await _bill_through_a_priced_deployment(read)

    (payload,) = capture.payloads
    assert payload["response_cost"] == pytest.approx(_PRICED_COST)
    assert billing.cost == pytest.approx(_PRICED_COST)
    assert payload["model_id"] == "deployment-custom"
    assert payload["model_group"] == "priced-gpt"
    assert payload["metadata"]["user_api_key_hash"] == "hash-alice"
    assert read.hidden_params == {"response_cost": 0.0}
    assert billing.table.statuses == {"job-priced": "completed"}


@pytest.mark.asyncio
async def test_the_billing_event_reaches_sync_only_success_callbacks(monkeypatch: pytest.MonkeyPatch):
    """Sync-only success callbacks (legacy string integrations, plain functions) get every async
    call's event from a worker thread, so they must get the billed background response too."""
    import queue

    import litellm

    priced_costs: Final[queue.Queue[object]] = queue.Queue()

    def record_cost(kwargs: Mapping[str, object], *_: object) -> None:
        priced_costs.put(kwargs.get("response_cost"))

    monkeypatch.setattr(litellm, "success_callback", [record_cost])
    monkeypatch.setattr(litellm, "_async_success_callback", [])

    await _bill_through_a_priced_deployment(_terminal_response())

    assert priced_costs.get(timeout=10) == pytest.approx(_PRICED_COST)


@pytest.mark.asyncio
async def test_the_billing_event_keys_its_spend_row_apart_from_the_read_it_bills(monkeypatch: pytest.MonkeyPatch):
    """The billing event logs the body the read returned, under the same response id, and spend logs
    keep one row per request id. Its row must take its own request id, or the read's free row takes
    the slot and the charge reaches the key with no spend row for an admin to see."""
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    class _RequestIdCapture(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.request_ids: tuple[str, ...] = ()

        async def async_log_success_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            self.request_ids = (
                *self.request_ids,
                get_logging_payload(kwargs, response_obj, start_time, end_time)["request_id"],
            )

    capture: Final = _RequestIdCapture()
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    read: Final = _terminal_response()

    billing: Final = await _bill_through_a_priced_deployment(read)

    (billed_request_id,) = capture.request_ids
    assert billed_request_id != billing.response_id
    assert billed_request_id.startswith(billing.response_id)

