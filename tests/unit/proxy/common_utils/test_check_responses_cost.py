"""
Unit tests for CheckResponsesCost class
"""

import asyncio
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from prisma.errors import UniqueViolationError
from litellm_enterprise.proxy.common_utils.check_responses_cost import BilledBackgroundResponse, BilledCharge

from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo


class _RecordingBilledLogger:
    """Stands in for the spend logger so a test reads the billing events a poller wrote, and can
    make the next writes raise."""

    def __init__(self, failures: tuple[BaseException, ...] = (), cost: float = 0.25) -> None:
        self.billed: tuple[BilledBackgroundResponse, ...] = ()
        self._failures = failures
        self.charge: Final = BilledCharge(cost=cost, logging_obj=MagicMock())

    async def __call__(self, billed: BilledBackgroundResponse) -> BilledCharge:
        await asyncio.sleep(0)
        if self._failures:
            failure, *remaining = self._failures
            self._failures = tuple(remaining)
            raise failure
        self.billed = (*self.billed, billed)
        return self.charge


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
        self.model_object_ids: Final = {row.id: row.model_object_id for row in rows}

    def _matches(self, row: _StoredResponse, where: Mapping[str, object]) -> bool:
        def value_of(field: str) -> object:
            if field == "status":
                return self.statuses[row.id]
            if field == "model_object_id":
                return self.model_object_ids[row.id]
            return getattr(row, field)

        def holds(field: str, condition: object) -> bool:
            if field == "OR" and isinstance(condition, list):
                return any(self._matches(row, alternative) for alternative in condition)
            if isinstance(condition, dict):
                return value_of(field) in condition["in"]
            return value_of(field) == condition

        return all(holds(field, condition) for field, condition in where.items())

    def _current(self, where: Mapping[str, object]) -> list[_StoredResponse]:
        return [
            dataclasses.replace(row, status=self.statuses[row.id], model_object_id=self.model_object_ids[row.id])
            for row in self._rows
            if self._matches(row, where)
        ]

    async def find_many(self, where: Mapping[str, object], **_: object) -> list[_StoredResponse]:
        return self._current(where)

    async def find_first(self, where: Mapping[str, object]) -> _StoredResponse | None:
        return next(iter(self._current(where)), None)

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, str]) -> int:
        matched: Final = self._current(where)
        if "model_object_id" in data and any(
            other.id not in {row.id for row in matched} and value == data["model_object_id"]
            for other, value in ((other, self.model_object_ids[other.id]) for other in self._rows)
        ):
            raise UniqueViolationError({"user_facing_error": {"error_code": "P2002"}})
        self.statuses.update({row.id: data["status"] for row in matched if "status" in data})
        self.model_object_ids.update({row.id: data["model_object_id"] for row in matched if "model_object_id" in data})
        return len(matched)


def _prisma_with_no_creator_rows() -> MagicMock:
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_deprecatedverificationtoken.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_deletedverificationtoken.find_first = AsyncMock(return_value=None)
    prisma_client.db.litellm_deletedteamtable.find_first = AsyncMock(return_value=None)
    return prisma_client


def _poller_over(table: _InMemoryManagedObjectTable, billed_logger: _RecordingBilledLogger):
    from litellm_enterprise.proxy.common_utils.check_responses_cost import CheckResponsesCost

    prisma_client: Final = _prisma_with_no_creator_rows()
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
        client = _prisma_with_no_creator_rows()
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

    assert costs == (None, billed_logger.charge, None)
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


def _issued_ids(monkeypatch: pytest.MonkeyPatch, count: int) -> tuple[str, ...]:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity

    monkeypatch.setenv("LITELLM_SALT_KEY", _ROTATION_SALT_KEY)
    issuer: Final = ResponsesIDSecurity(general_settings_reader=lambda: {}, signing_key_reader=lambda: _ROTATION_SALT_KEY)
    return tuple(
        issuer._encrypt_response_id(
            ResponsesAPIResponse(id="resp_provider_job", created_at=1, output=[], status="queued"),
            UserAPIKeyAuth(user_id="alice"),
        ).id
        for _ in range(count)
    )


def _stored_before_the_upgrade(created_id: str) -> _StoredResponse:
    return _StoredResponse(
        id="job-legacy", unified_object_id=created_id, model_object_id=created_id, status="queued"
    )


@pytest.mark.asyncio
async def test_a_response_stored_before_the_upgrade_bills_once_whichever_issued_id_reads_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """A row the old proxy stored keys the response by the encrypted id the create returned, so the
    read that sends that id rekeys it by the provider id, and a read through any later id finds it
    already billed."""
    created_id, reissued_id = _issued_ids(monkeypatch, 2)
    table: Final = _InMemoryManagedObjectTable(rows=(_stored_before_the_upgrade(created_id),))
    billed_logger: Final = _RecordingBilledLogger()
    poller: Final = _poller_over(table, billed_logger)
    track_untracked: Final = AsyncMock(return_value=False)

    with patch("litellm.aget_responses", side_effect=AssertionError("billing must not read the provider again")):
        await poller.bill_if_finished(response_id=created_id, response=_terminal_response())
        await poller.bill_if_finished(
            response_id=reissued_id, response=_terminal_response(), track_untracked=track_untracked
        )

    assert len(billed_logger.billed) == 1
    assert table.model_object_ids == {"job-legacy": "resp_provider_job"}
    track_untracked.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_rows, billed_by_the_poll",
    [((), 1), ((_StoredResponse(id="job-read", unified_object_id="read-id", model_object_id="resp_provider_job", status="completed"),), 0)],
    ids=["no_read_yet", "a_read_already_billed_it"],
)
async def test_the_poll_bills_a_response_stored_before_the_upgrade_only_when_no_read_billed_it_first(
    monkeypatch: pytest.MonkeyPatch, read_rows: tuple[_StoredResponse, ...], billed_by_the_poll: int
):
    """A read through a later id misses the old row and tracks the response under its provider id,
    so the poll retires the old row unbilled when that row exists and bills it once otherwise."""
    (created_id,) = _issued_ids(monkeypatch, 1)
    table: Final = _InMemoryManagedObjectTable(rows=(_stored_before_the_upgrade(created_id), *read_rows))
    billed_logger: Final = _RecordingBilledLogger()

    with patch("litellm.aget_responses", new_callable=AsyncMock, return_value=_terminal_response()):
        await _poller_over(table, billed_logger).check_responses_cost()
        await _poller_over(table, billed_logger).check_responses_cost()

    assert len(billed_logger.billed) == billed_by_the_poll
    assert table.statuses["job-legacy"] == "completed"


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
async def test_the_billing_event_carries_the_creating_keys_project_and_every_creator_model_budget():
    key_budget: Final = {"gpt-5": {"max_budget": 1.0, "budget_duration": "1d"}}
    team_budget: Final = {"gpt-5": {"max_budget": 2.0, "budget_duration": "7d"}}
    user_budget: Final = {"gpt-5": {"max_budget": 3.0, "budget_duration": "30d"}}
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-budgets",
                unified_object_id="resp_budgets",
                status="queued",
                created_by="alice",
                api_key="hash-alice",
                team_id="team-alpha",
            ),
        )
    )
    billed_logger: Final = _RecordingBilledLogger()
    poller: Final = _poller_over(table, billed_logger)
    poller.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            key_alias=None,
            organization_id=None,
            spend=0.0,
            max_budget=None,
            project_id="project-1",
            litellm_project_table=SimpleNamespace(project_alias="project-one"),
            model_max_budget=key_budget,
        )
    )
    poller.prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            team_alias=None, organization_id=None, spend=0.0, max_budget=None, model_max_budget=team_budget
        )
    )
    poller.prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            user_email=None, user_alias=None, spend=0.0, max_budget=None, model_max_budget=user_budget
        )
    )

    await poller.bill_if_finished(response_id="resp_budgets", response=_terminal_response())

    (billed,) = billed_logger.billed
    assert {
        field: billed.metadata.get(field)
        for field in (
            "user_api_key_project_id",
            "user_api_key_project_alias",
            "user_api_key_model_max_budget",
            "user_api_key_team_model_max_budget",
            "user_api_key_user_model_max_budget",
        )
    } == {
        "user_api_key_project_id": "project-1",
        "user_api_key_project_alias": "project-one",
        "user_api_key_model_max_budget": key_budget,
        "user_api_key_team_model_max_budget": team_budget,
        "user_api_key_user_model_max_budget": user_budget,
    }


_TIER_MODEL_BUDGET: Final = {"gpt-5": {"max_budget": 4.0, "budget_duration": "1d"}}
_KEY_MODEL_BUDGET: Final = {"gpt-5": {"max_budget": 1.0, "budget_duration": "1d"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_max_budget", "key_model_max_budget", "billed_max_budget", "billed_model_max_budget"),
    [
        (None, {}, 5.0, _TIER_MODEL_BUDGET),
        (None, None, 5.0, _TIER_MODEL_BUDGET),
        (2.0, _KEY_MODEL_BUDGET, 2.0, _KEY_MODEL_BUDGET),
    ],
    ids=["key_with_an_empty_budget_map", "key_with_no_budget_map", "key_with_its_own_budgets"],
)
async def test_the_billing_event_takes_the_keys_budget_tier_only_where_the_key_sets_no_budget_of_its_own(
    key_max_budget: float | None,
    key_model_max_budget: dict[str, object] | None,
    billed_max_budget: float,
    billed_model_max_budget: dict[str, object],
):
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-tier",
                unified_object_id="resp_tier",
                status="queued",
                created_by="key:hash-tier",
                api_key="hash-tier",
            ),
        )
    )
    billed_logger: Final = _RecordingBilledLogger()
    poller: Final = _poller_over(table, billed_logger)
    poller.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            key_alias=None,
            organization_id=None,
            spend=0.0,
            max_budget=key_max_budget,
            project_id=None,
            model_max_budget=key_model_max_budget,
            litellm_budget_table=SimpleNamespace(max_budget=5.0, model_max_budget=_TIER_MODEL_BUDGET),
        )
    )

    await poller.bill_if_finished(response_id="resp_tier", response=_terminal_response())

    (billed,) = billed_logger.billed
    assert (billed.metadata.get("user_api_key_max_budget"), billed.metadata.get("user_api_key_model_max_budget")) == (
        billed_max_budget,
        billed_model_max_budget,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("created_by", "billed_user_id"),
    [("alice", "alice"), ("key:hashed-token", None)],
    ids=["key_with_a_user", "key_with_no_user_or_team"],
)
async def test_the_billing_event_bills_the_creating_user_and_never_a_keys_owner_scope(
    created_by: str, billed_user_id: str | None
):
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-owner",
                unified_object_id="resp_owner",
                status="queued",
                created_by=created_by,
                api_key="hashed-token",
            ),
        )
    )
    billed_logger: Final = _RecordingBilledLogger()

    await _poller_over(table, billed_logger).bill_if_finished(response_id="resp_owner", response=_terminal_response())

    (billed,) = billed_logger.billed
    assert billed.metadata["user_api_key_user_id"] == billed_user_id


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
        charge: Final = await poller.bill_from_deployment("resp_settle")
    with patch("litellm.aget_responses", side_effect=AssertionError("a billed response is never fetched again")):
        assert await poller.bill_from_deployment("resp_settle") is None

    assert charge == billed_logger.charge
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
    logging_obj: object


@dataclass(frozen=True)
class _PricedPoller:
    poller: object
    prisma_client: MagicMock
    table: _InMemoryManagedObjectTable
    response_id: str

    async def bill(self, read: ResponsesAPIResponse) -> object:
        from litellm_enterprise.proxy.common_utils.check_responses_cost import CheckResponsesCost

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        assert isinstance(self.poller, CheckResponsesCost)
        charge: Final = await self.poller.bill_if_finished(response_id=self.response_id, response=read)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
        return charge


_PRICED_LITELLM_PARAMS: Final[Mapping[str, object]] = MappingProxyType(
    {"model": "openai/gpt-5.6", "api_key": "sk-unused", "input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
)
_CREATING_KEY_HASH: Final = hashlib.sha256(b"sk-alice").hexdigest()


def _team_alpha_when_its_rows_exist(
    team_row: object, deleted_team_row: object, default_team_settings: tuple[Mapping[str, object], ...]
) -> str | None:
    return "team-alpha" if team_row is not None or deleted_team_row is not None or default_team_settings else None


def _priced_poller(
    litellm_params: Mapping[str, object] = _PRICED_LITELLM_PARAMS,
    model_info: Mapping[str, object] = MappingProxyType({}),
    key_row: object = None,
    team_row: object = None,
    key_lookup_error: Exception | None = None,
    default_team_settings: tuple[Mapping[str, object], ...] = (),
    api_key: str = _CREATING_KEY_HASH,
    rotated_to: Mapping[str, object] = MappingProxyType({}),
    deleted_key_row: object = None,
    deleted_team_row: object = None,
    model_name: str = "priced-gpt",
    requested_model: str = "priced-gpt",
    model_group_alias: Mapping[str, str] = MappingProxyType({}),
    team_id: str | None = None,
) -> _PricedPoller:
    from types import SimpleNamespace

    import litellm
    from litellm_enterprise.proxy.common_utils.check_responses_cost import CheckResponsesCost

    from litellm.responses.utils import ResponsesAPIRequestUtils

    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id="deployment-custom", response_id="resp_upstream"
    )
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-priced",
                unified_object_id=response_id,
                status="queued",
                api_key=api_key,
                file_object={"model": requested_model},
                team_id=team_id or _team_alpha_when_its_rows_exist(team_row, deleted_team_row, default_team_settings),
            ),
        )
    )
    key_rows: Final = {api_key: key_row, **{token: row for token, row in rotated_to.items()}}
    prisma_client: Final = _prisma_with_no_creator_rows()
    prisma_client.db.litellm_managedobjecttable = table
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        side_effect=key_lookup_error or (lambda where, **_: key_rows.get(where["token"]))
    )
    rotation_chain: Final = (api_key, *rotated_to)
    rotated_from: Final = dict(zip(rotation_chain, rotation_chain[1:]))
    prisma_client.db.litellm_deprecatedverificationtoken.find_unique = AsyncMock(
        side_effect=lambda where: (
            SimpleNamespace(token=where["token"], active_token_id=rotated_from[where["token"]])
            if where["token"] in rotated_from
            else None
        )
    )
    prisma_client.db.litellm_deletedverificationtoken.find_first = AsyncMock(return_value=deleted_key_row)
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    prisma_client.db.litellm_deletedteamtable.find_first = AsyncMock(return_value=deleted_team_row)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": dict(litellm_params),
                "model_info": {"id": "deployment-custom", **model_info},
            }
        ],
        model_group_alias=dict(model_group_alias),
    )
    return _PricedPoller(
        poller=CheckResponsesCost(
            proxy_logging_obj=MagicMock(),
            prisma_client=prisma_client,
            llm_router=router,
            proxy_config=_proxy_config_with(default_team_settings),
        ),
        prisma_client=prisma_client,
        table=table,
        response_id=response_id,
    )


async def _bill_through_a_priced_deployment(
    read: ResponsesAPIResponse,
    litellm_params: Mapping[str, object] = _PRICED_LITELLM_PARAMS,
    model_info: Mapping[str, object] = MappingProxyType({}),
    key_row: object = None,
    team_row: object = None,
    key_lookup_error: Exception | None = None,
    default_team_settings: tuple[Mapping[str, object], ...] = (),
    api_key: str = _CREATING_KEY_HASH,
    rotated_to: Mapping[str, object] = MappingProxyType({}),
    deleted_key_row: object = None,
    deleted_team_row: object = None,
    model_name: str = "priced-gpt",
    requested_model: str = "priced-gpt",
    model_group_alias: Mapping[str, str] = MappingProxyType({}),
    team_id: str | None = None,
) -> _PricedBilling:
    priced: Final = _priced_poller(
        litellm_params=litellm_params,
        model_info=model_info,
        key_row=key_row,
        team_row=team_row,
        key_lookup_error=key_lookup_error,
        default_team_settings=default_team_settings,
        api_key=api_key,
        rotated_to=rotated_to,
        deleted_key_row=deleted_key_row,
        deleted_team_row=deleted_team_row,
        model_name=model_name,
        requested_model=requested_model,
        model_group_alias=model_group_alias,
        team_id=team_id,
    )
    charge: Final = await priced.bill(read)
    return _PricedBilling(
        table=priced.table,
        response_id=priced.response_id,
        cost=getattr(charge, "cost", None),
        logging_obj=getattr(charge, "logging_obj", None),
    )


def _proxy_config_with(default_team_settings: tuple[Mapping[str, object], ...]):
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config: Final = ProxyConfig()
    proxy_config.config = (
        {"litellm_settings": {"default_team_settings": [dict(settings) for settings in default_team_settings]}}
        if default_team_settings
        else {}
    )
    return proxy_config


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
    assert payload["metadata"]["user_api_key_hash"] == _CREATING_KEY_HASH
    assert read.hidden_params == {"response_cost": 0.0}
    assert billing.table.statuses == {"job-priced": "completed"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "deployment_model", "requested_model", "model_group_alias", "model_info", "model_group"),
    [
        ("openai/*", "openai/*", "openai/gpt-5.6", {}, {}, "openai/gpt-5.6"),
        ("priced-gpt", "openai/gpt-5.6", "priced-alias", {"priced-alias": "priced-gpt"}, {}, "priced-alias"),
        ("priced-gpt", "openai/gpt-5.6", "gpt-5.6-2026-01-01", {}, {}, "priced-gpt"),
        (
            "model_name_team-1_abc",
            "openai/gpt-5.6",
            "team-gpt",
            {},
            {"team_id": "team-1", "team_public_model_name": "team-gpt"},
            "team-gpt",
        ),
        (
            "model_name_team-1_wild",
            "openai/*",
            "openai/gpt-5.6",
            {},
            {"team_id": "team-1", "team_public_model_name": "openai/*"},
            "openai/gpt-5.6",
        ),
        (
            "model_name_team-2_abc",
            "openai/gpt-5.6",
            "team-gpt",
            {},
            {"team_id": "team-2", "team_public_model_name": "team-gpt"},
            "model_name_team-2_abc",
        ),
    ],
    ids=[
        "wildcard_route",
        "model_group_alias",
        "fallback_kept_the_provider_model_name",
        "team_model",
        "team_wildcard",
        "another_teams_model",
    ],
)
async def test_the_billing_event_names_the_requested_model_as_its_model_group_only_while_it_routes_to_the_deployment(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    deployment_model: str,
    requested_model: str,
    model_group_alias: Mapping[str, str],
    model_info: Mapping[str, object],
    model_group: str,
):
    import litellm

    capture: Final = _spend_capture_logger()
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])

    await _bill_through_a_priced_deployment(
        _terminal_response(),
        litellm_params={**_PRICED_LITELLM_PARAMS, "model": deployment_model},
        model_info=model_info,
        model_name=model_name,
        requested_model=requested_model,
        model_group_alias=model_group_alias,
        team_id="team-1",
    )

    (payload,) = capture.payloads
    assert payload["model_group"] == model_group


@pytest.mark.asyncio
@pytest.mark.parametrize("base_model_in", ["litellm_params", "model_info"])
async def test_the_billing_event_prices_the_usage_at_the_deployments_base_model(
    monkeypatch: pytest.MonkeyPatch, base_model_in: str
):
    """A deployment that names a base_model for cost tracking bills its background responses at that
    model's price, whether the base_model sits in litellm_params or model_info."""
    import litellm

    capture: Final = _spend_capture_logger()
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    monkeypatch.setitem(
        litellm.model_cost,
        "qa-priced-base-model",
        {
            "litellm_provider": "openai",
            "mode": "responses",
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
        },
    )
    base_model: Final = {"base_model": "qa-priced-base-model"}

    billing: Final = await _bill_through_a_priced_deployment(
        _terminal_response(),
        litellm_params={
            "model": "openai/gpt-5.6",
            "api_key": "sk-unused",
            **(base_model if base_model_in == "litellm_params" else {}),
        },
        model_info=base_model if base_model_in == "model_info" else {},
    )

    (payload,) = capture.payloads
    assert payload["response_cost"] == pytest.approx(_PRICED_COST)
    assert billing.cost == pytest.approx(_PRICED_COST)


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


_CREATOR_LANGFUSE_VARS: Final = MappingProxyType(
    {
        "langfuse_public_key": "pk-lf-creator",
        "langfuse_secret_key": "sk-lf-creator",
        "langfuse_host": "http://127.0.0.1:9",
    }
)


def _key_row(metadata: Mapping[str, object]) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(key_alias=None, organization_id=None, spend=0.0, max_budget=None, metadata=dict(metadata))


def _team_row(metadata: Mapping[str, object]) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(team_alias=None, organization_id=None, spend=0.0, max_budget=None, metadata=dict(metadata))


def _langfuse_callback_settings(callback_vars: Mapping[str, object]) -> Mapping[str, object]:
    return {"callback_settings": {"success_callback": ["langfuse"], "callback_vars": dict(callback_vars)}}


@dataclass(frozen=True)
class _CreatorLogging:
    key_row: object
    team_row: object
    default_team_settings: tuple[Mapping[str, object], ...]


def _creator_logging(logging_on: str) -> _CreatorLogging:
    logging: Final = {
        "logging": [
            {"callback_name": "langfuse", "callback_type": "success", "callback_vars": dict(_CREATOR_LANGFUSE_VARS)}
        ]
    }
    team_metadata: Final = {
        "team logging": logging,
        "team callback_settings": _langfuse_callback_settings(_CREATOR_LANGFUSE_VARS),
    }
    return _CreatorLogging(
        key_row=_key_row(logging if logging_on == "key logging" else {}),
        team_row=_team_row(team_metadata.get(logging_on, {})),
        default_team_settings=(
            ({"team_id": "team-alpha", "success_callback": ["langfuse"], **_CREATOR_LANGFUSE_VARS},)
            if logging_on == "config team settings"
            else ()
        ),
    )


def _recording_langfuse():
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.integrations.langfuse.langfuse_prompt_management import LangfusePromptManagement

    class _RecordingLangfuse(LangfusePromptManagement):
        def __init__(self) -> None:
            CustomLogger.__init__(self)
            self.events: tuple[tuple[object, object], ...] = ()

        async def async_log_success_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            params: Final = kwargs["standard_callback_dynamic_params"]
            assert isinstance(params, Mapping)
            self.events = (*self.events, (kwargs.get("response_cost"), params.get("langfuse_public_key")))

    return _RecordingLangfuse()


async def _log_a_request_from(creator: _CreatorLogging) -> None:
    import litellm
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import key_and_team_logging_fields

    await litellm.acompletion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
        **key_and_team_logging_fields(
            user_api_key_dict=UserAPIKeyAuth(
                metadata=getattr(creator.key_row, "metadata"),
                team_id="team-alpha",
                team_metadata=getattr(creator.team_row, "metadata"),
            ),
            proxy_config=_proxy_config_with(creator.default_team_settings),
        ),
    )
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_logs_to_langfuse", [False, True])
@pytest.mark.parametrize(
    "logging_on", ["key logging", "team logging", "team callback_settings", "config team settings"]
)
async def test_the_billing_event_reaches_the_langfuse_projects_the_creators_own_requests_log_to(
    monkeypatch: pytest.MonkeyPatch, logging_on: str, proxy_logs_to_langfuse: bool
):
    """A key or team that logs to its own Langfuse project must see the billed charge in the same projects
    its own requests log to, wherever that logging is configured and whether or not the proxy logs to
    Langfuse too, or the charge only shows in the proxy's spend logs."""
    import litellm
    from litellm.litellm_core_utils import litellm_logging

    langfuse: Final = _recording_langfuse()
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [langfuse])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [langfuse] if proxy_logs_to_langfuse else [])
    creator: Final = _creator_logging(logging_on)

    await _log_a_request_from(creator)
    request_projects: Final = tuple(public_key for _, public_key in langfuse.events)
    await _bill_through_a_priced_deployment(
        _terminal_response(),
        key_row=creator.key_row,
        team_row=creator.team_row,
        default_team_settings=creator.default_team_settings,
    )
    billed_events: Final = langfuse.events[len(request_projects) :]

    assert _CREATOR_LANGFUSE_VARS["langfuse_public_key"] in request_projects
    assert tuple(public_key for _, public_key in billed_events) == request_projects
    assert all(cost == pytest.approx(_PRICED_COST) for cost, _ in billed_events)


def _rotations_for(retired: str, key_logging: object) -> Mapping[str, object]:
    rotated_once: Final = hashlib.sha256(b"sk-alice-rotated").hexdigest()
    if retired == "rotated key":
        return {rotated_once: key_logging}
    if retired == "key rotated twice":
        return {rotated_once: None, hashlib.sha256(b"sk-alice-rotated-again").hexdigest(): key_logging}
    return {}


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", ["rotated key", "key rotated twice", "deleted key", "deleted team"])
async def test_a_creator_rotated_or_deleted_before_the_bill_still_gets_the_charge_in_its_own_langfuse_project(
    monkeypatch: pytest.MonkeyPatch, retired: str
):
    """A key rotated or deleted, or a team deleted, while its background response ran logged that response's
    create to its own project, so the charge follows the settings kept on the key it was rotated to or on
    its deleted row instead of leaving that project for the proxy's callbacks alone."""
    import litellm
    from litellm.litellm_core_utils import litellm_logging

    langfuse: Final = _recording_langfuse()
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [langfuse])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    key_logging: Final = _creator_logging("key logging").key_row

    await _bill_through_a_priced_deployment(
        _terminal_response(),
        rotated_to=_rotations_for(retired, key_logging),
        deleted_key_row=key_logging if retired == "deleted key" else SimpleNamespace(metadata={}),
        deleted_team_row=_creator_logging("team callback_settings").team_row if retired == "deleted team" else None,
    )

    assert langfuse.events == ((pytest.approx(_PRICED_COST), _CREATOR_LANGFUSE_VARS["langfuse_public_key"]),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_lookup, live_key_logging, team_logging",
    [
        ("litellm_verificationtoken.find_unique", "key logging", "none"),
        ("litellm_teamtable.find_unique", "none", "team callback_settings"),
        ("litellm_deprecatedverificationtoken.find_unique", "none", "none"),
        ("litellm_deletedverificationtoken.find_first", "none", "none"),
    ],
    ids=["live key", "live team", "rotated key", "deleted key"],
)
async def test_a_creator_lookup_that_keeps_failing_still_bills_the_response_once_without_the_creators_settings(
    monkeypatch: pytest.MonkeyPatch, failing_lookup: str, live_key_logging: str, team_logging: str
):
    """A creator lookup that fails on every read must not leave the response unbilled. It bills once under the
    key and team the managed row stores, through the proxy's callbacks only, with the text left out since the
    creator's redaction setting could not be read."""
    import litellm
    from litellm.litellm_core_utils import litellm_logging

    langfuse: Final = _recording_langfuse()
    capture: Final = _spend_capture_logger()
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [langfuse])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)
    priced: Final = _priced_poller(
        key_row=_creator_logging(live_key_logging).key_row if live_key_logging != "none" else None,
        team_row=_creator_logging(team_logging).team_row if team_logging != "none" else None,
    )
    table_name, method = failing_lookup.split(".")
    setattr(
        getattr(priced.prisma_client.db, table_name),
        method,
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )

    charge: Final = await priced.bill(_read_with_text("the stored answer"))
    await priced.bill(_read_with_text("the stored answer"))

    assert getattr(charge, "cost", None) == pytest.approx(_PRICED_COST)
    assert priced.table.statuses == {"job-priced": "completed"}
    assert langfuse.events == ()
    (payload,) = capture.payloads
    assert payload["response_cost"] == pytest.approx(_PRICED_COST)
    assert "the stored answer" not in json.dumps(payload, default=str)
    metadata: Final = payload["metadata"]
    assert isinstance(metadata, Mapping)
    assert metadata["user_api_key_hash"] == _CREATING_KEY_HASH
    assert metadata["user_api_key_team_id"] == ("team-alpha" if team_logging != "none" else None)


_TEAM_MODEL_BUDGET: Final = {"gpt-5": {"max_budget": 2.0, "budget_duration": "7d"}}
_LOADED_KEY_ATTRIBUTION: Final = {"user_api_key_alias": "prod-key", "user_api_key_max_budget": 5.0}
_LOADED_TEAM_ATTRIBUTION: Final = {
    "user_api_key_team_alias": "Team Alpha",
    "user_api_key_team_max_budget": 50.0,
    "user_api_key_team_model_max_budget": _TEAM_MODEL_BUDGET,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_lookup, key_is_live, loaded_attribution",
    [
        ("litellm_verificationtoken.find_unique", True, _LOADED_TEAM_ATTRIBUTION),
        ("litellm_teamtable.find_unique", True, _LOADED_KEY_ATTRIBUTION),
        ("litellm_deprecatedverificationtoken.find_unique", False, _LOADED_TEAM_ATTRIBUTION),
        ("litellm_deletedverificationtoken.find_first", False, _LOADED_TEAM_ATTRIBUTION),
    ],
    ids=["live key", "live team", "rotated key", "deleted key"],
)
async def test_a_creator_lookup_that_fails_keeps_the_aliases_and_budgets_of_the_rows_that_did_load(
    failing_lookup: str, key_is_live: bool, loaded_attribution: Mapping[str, object]
):
    """One failed creator lookup must not strip what the other rows give the charge: the team's per-model budget
    only counts a billed charge whose event carries it, and the aliases name who spent it."""
    table: Final = _InMemoryManagedObjectTable(
        rows=(
            _StoredResponse(
                id="job-attributed",
                unified_object_id="resp_attributed",
                status="queued",
                api_key=_CREATING_KEY_HASH,
                team_id="team-alpha",
            ),
        )
    )
    billed_logger: Final = _RecordingBilledLogger()
    poller: Final = _poller_over(table, billed_logger)
    poller.prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=(
            SimpleNamespace(key_alias="prod-key", organization_id=None, spend=4.0, max_budget=5.0, metadata={})
            if key_is_live
            else None
        )
    )
    poller.prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            team_alias="Team Alpha",
            organization_id=None,
            spend=40.0,
            max_budget=50.0,
            model_max_budget=_TEAM_MODEL_BUDGET,
            metadata={},
        )
    )
    table_name, method = failing_lookup.split(".")
    setattr(
        getattr(poller.prisma_client.db, table_name),
        method,
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )

    await poller.bill_if_finished(response_id="resp_attributed", response=_terminal_response())

    assert table.statuses == {"job-attributed": "completed"}
    (billed,) = billed_logger.billed
    assert {field: billed.metadata.get(field) for field in loaded_attribution} == loaded_attribution


@pytest.mark.asyncio
async def test_a_team_logging_setting_the_logger_rejects_still_bills_the_response_once(monkeypatch: pytest.MonkeyPatch):
    """A team whose stored callback settings the logger rejects, such as an os.environ reference it no
    longer resolves, must not stop the bill: the charge reaches the proxy's callbacks and the response is
    not put back for every later poll to fail on again."""
    import litellm

    capture: Final = _spend_capture_logger()
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])

    billing: Final = await _bill_through_a_priced_deployment(
        _terminal_response(),
        key_row=_key_row({}),
        team_row=_team_row(_langfuse_callback_settings({"langfuse_public_key": "os.environ/QA_UNSET_LANGFUSE_KEY"})),
    )

    (payload,) = capture.payloads
    assert payload["response_cost"] == pytest.approx(_PRICED_COST)
    assert billing.cost == pytest.approx(_PRICED_COST)
    assert billing.table.statuses == {"job-priced": "completed"}


def _read_with_text(text: str) -> ResponsesAPIResponse:
    return _terminal_response().model_copy(
        update={
            "output": [
                {
                    "type": "message",
                    "id": "msg_stored",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ]
        }
    )


def _key_logging_to_langfuse(callback_vars: Mapping[str, object]) -> object:
    return _key_row(
        {"logging": [{"callback_name": "langfuse", "callback_type": "success", "callback_vars": dict(callback_vars)}]}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_row", "deleted_key_row", "text_logged"),
    [
        (_key_logging_to_langfuse(_CREATOR_LANGFUSE_VARS), None, True),
        (_key_logging_to_langfuse({**_CREATOR_LANGFUSE_VARS, "turn_off_message_logging": "true"}), None, False),
        (None, _key_logging_to_langfuse({**_CREATOR_LANGFUSE_VARS, "turn_off_message_logging": "true"}), False),
        (None, None, True),
    ],
    ids=[
        "creator keeps message logging",
        "creator turns message logging off",
        "deleted creator turned message logging off",
        "creator has no key row anywhere",
    ],
)
async def test_the_billing_event_logs_the_response_text_only_as_the_creators_logging_settings_allow(
    monkeypatch: pytest.MonkeyPatch, key_row: object, deleted_key_row: object, text_logged: bool
):
    """The billed event follows the creating key's or team's message logging setting, a deleted key's included,
    which falls back to the proxy's the way the batch poll decides it, so a key with no stored settings
    anywhere logs the text exactly as the proxy is set."""
    import litellm
    from litellm.litellm_core_utils import litellm_logging

    capture: Final = _spend_capture_logger()
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [_recording_langfuse()])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)

    await _bill_through_a_priced_deployment(
        _read_with_text("the stored answer"), key_row=key_row, deleted_key_row=deleted_key_row
    )

    (payload,) = capture.payloads
    assert payload["response_cost"] == pytest.approx(_PRICED_COST)
    assert ("the stored answer" in json.dumps(payload, default=str)) is text_logged


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_token", ["oauth2-raw-access-token", "sk-custom-auth-1234"])
async def test_a_response_created_without_a_virtual_key_bills_into_its_teams_langfuse_project(
    monkeypatch: pytest.MonkeyPatch, caller_token: str
):
    """Under OAuth2 the row keeps the caller's raw token, and under custom auth the hash of an sk- token that
    has no key row, so the charge follows the team's logging settings like that caller's own requests."""
    from litellm.proxy._types import UserAPIKeyAuth

    import litellm
    from litellm.litellm_core_utils import litellm_logging

    langfuse: Final = _recording_langfuse()
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [langfuse])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])

    await _bill_through_a_priced_deployment(
        _terminal_response(),
        api_key=UserAPIKeyAuth(api_key=caller_token).api_key or caller_token,
        team_row=_creator_logging("team callback_settings").team_row,
    )

    assert langfuse.events == ((pytest.approx(_PRICED_COST), _CREATOR_LANGFUSE_VARS["langfuse_public_key"]),)


@pytest.mark.asyncio
async def test_a_key_that_disabled_a_callback_has_its_charge_logged_wherever_its_own_requests_are(
    monkeypatch: pytest.MonkeyPatch,
):
    """A key's stored list of disabled callbacks must treat the billed charge like that key's own requests, or
    a project that shows the key's requests at $0 never shows the charge they add up to."""
    import litellm
    from litellm.litellm_core_utils import litellm_logging
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True)
    langfuse: Final = _recording_langfuse()
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [langfuse])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [langfuse])
    creator: Final = _CreatorLogging(
        key_row=_key_row({"litellm_disabled_callbacks": ["langfuse"]}), team_row=_team_row({}), default_team_settings=()
    )

    await _log_a_request_from(creator)
    request_events: Final = langfuse.events
    await _bill_through_a_priced_deployment(_terminal_response(), key_row=creator.key_row)
    billed_events: Final = langfuse.events[len(request_events) :]

    assert len(billed_events) == len(request_events)
    assert all(cost == pytest.approx(_PRICED_COST) for cost, _ in billed_events)
