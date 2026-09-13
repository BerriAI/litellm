from builtins import ExceptionGroup
from collections.abc import Callable
from typing import Final
from unittest.mock import Mock, call

import pytest
from batch_cleanup import BATCH_CANCEL_TIMEOUT_SECONDS, CLEANUP_DELAYS, cleanup_batch, cleanup_file, cleanup_result
from batch_client import AZURE_FILE_EXPIRY_SECONDS, BatchObject, FileDeleteResponse, batch_upload_form
from capabilities import CAPABILITIES, Capability
from e2e_http import NetworkError, RateLimitedError, Result, Success, UnknownApiError
from lifecycle import ResourceManager
from models import KeyGenerateBody

MANAGED_FILE_ID: Final = "bGl0ZWxsbV9wcm94eTtmaWxlLTE="
MANAGED_BATCH_ID: Final = "bGl0ZWxsbV9wcm94eTtiYXRjaC0x"


class ExpectedCalls[T]:
    def __init__(self, values: tuple[T, ...]) -> None:
        self.values: Final = values
        self.recorder: Final = Mock()

    def __call__(self, value: T) -> None:
        self.recorder(value)

    def assert_done(self) -> None:
        assert tuple(self.recorder.call_args_list) == tuple(call(value) for value in self.values)


class CleanupClient:
    def __init__(
        self,
        *,
        calls: ExpectedCalls[str],
        files: tuple[Result[FileDeleteResponse], ...] = (),
        batches: tuple[Result[BatchObject], ...] = (),
        cancellations: tuple[Result[BatchObject], ...] = (),
    ) -> None:
        self.calls: Final = calls
        self.file_response: Final[Callable[[], Result[FileDeleteResponse]]] = Mock(side_effect=files)
        self.batch_response: Final[Callable[[], Result[BatchObject]]] = Mock(side_effect=batches)
        self.cancel_response: Final[Callable[[], Result[BatchObject]]] = Mock(side_effect=cancellations)

    def delete_file(self, file_id: str, *, key: str, provider: str | None = None) -> Result[FileDeleteResponse]:
        self.calls(f"delete {provider} {file_id}")
        return self.file_response()

    def retrieve_batch(self, batch_id: str, *, key: str, provider: str | None = None) -> Result[BatchObject]:
        self.calls(f"retrieve {provider} {batch_id}")
        return self.batch_response()

    def cancel_batch(self, batch_id: str, *, key: str, provider: str | None = None) -> Result[BatchObject]:
        self.calls(f"cancel {provider} {batch_id}")
        return self.cancel_response()

    def generate_key(self, body: KeyGenerateBody) -> str:
        return "test-key"

    def delete_key(self, key: str) -> None:
        self.calls(f"delete key {key}")

    def delete_customers(self, user_ids: list[str]) -> None:
        self.calls(f"delete customers {user_ids}")


def batch(status: str) -> Success[BatchObject]:
    return Success(status_code=200, data=BatchObject(id="batch-1", status=status))


def deleted_file(*, deleted: bool = True) -> Success[FileDeleteResponse]:
    return Success(status_code=200, data=FileDeleteResponse(id="file-1", deleted=deleted))


class TestFileCleanup:
    def test_managed_delete_accepts_the_deleted_file_object(self) -> None:
        response: Final = Success(
            status_code=200, data=FileDeleteResponse.model_validate({"id": MANAGED_FILE_ID, "object": "file"})
        )
        client: Final = CleanupClient(calls=ExpectedCalls((f"delete None {MANAGED_FILE_ID}",)), files=(response,))
        cleanup_file(client, MANAGED_FILE_ID, key="test-key")
        client.calls.assert_done()

    @pytest.mark.parametrize("file_id", ["file-1", MANAGED_FILE_ID])
    def test_a_success_status_without_a_deletion_confirmation_is_rejected(self, file_id: str) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls((f"delete None {file_id}",)),
            files=(Success(status_code=200, data=FileDeleteResponse(id=file_id)),),
        )
        with pytest.raises(AssertionError, match="did not confirm deletion"):
            cleanup_file(client, file_id, key="test-key")
        client.calls.assert_done()

    @pytest.mark.parametrize("cap", CAPABILITIES, ids=[cap.id for cap in CAPABILITIES])
    def test_deletes_raw_files_through_the_upload_provider(self, cap: Capability) -> None:
        expected_provider: Final = cap.provider if cap.scenario in {"model_param", "provider_fallback"} else None
        client: Final = CleanupClient(
            calls=ExpectedCalls((f"delete {expected_provider} file-1",)), files=(deleted_file(),)
        )
        cleanup_file(client, "file-1", key="test-key", provider=cap.file_provider)
        client.calls.assert_done()

    def test_failed_delete_is_reported_after_remaining_resources_are_cleaned(self) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(("delete azure file-1", "delete key test-key")),
            files=(UnknownApiError(status_code=403, body="secret response"),),
        )
        manager: Final = ResourceManager(client=client, strict_cleanup=True)
        key: Final = manager.key()
        manager.defer(lambda: cleanup_file(client, "file-1", key=key, provider="azure"))
        with pytest.raises(ExceptionGroup) as caught:
            manager.teardown()
        client.calls.assert_done()
        assert len(caught.value.exceptions) == 1
        assert str(caught.value.exceptions[0]) == "Delete file file-1 failed: HTTP 403"

    def test_success_response_must_confirm_deletion(self) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(("delete None file-1",)), files=(deleted_file(deleted=False),)
        )
        with pytest.raises(AssertionError, match="did not confirm deletion"):
            cleanup_file(client, "file-1", key="test-key")
        client.calls.assert_done()

    def test_cleanup_is_idempotent_when_file_is_already_deleted(self) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(("delete azure file-1",)),
            files=(UnknownApiError(status_code=404, body="missing"),),
        )
        cleanup_file(client, "file-1", key="test-key", provider="azure")
        client.calls.assert_done()

    def test_default_resource_cleanup_keeps_existing_best_effort_behavior(self) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(("delete None file-1", "delete key test-key")),
            files=(UnknownApiError(status_code=403, body="forbidden"),),
        )
        manager: Final = ResourceManager(client=client)
        key: Final = manager.key()
        manager.defer(lambda: cleanup_file(client, "file-1", key=key))
        manager.teardown()
        client.calls.assert_done()


class TestCleanupRetries:
    @pytest.mark.parametrize(
        "failure",
        [NetworkError(message="offline"), RateLimitedError(), UnknownApiError(status_code=503, body="unavailable")],
    )
    def test_transient_error_retries_and_returns_success(self, failure: Result[FileDeleteResponse]) -> None:
        responses: Final = (failure, deleted_file())
        outcomes: Final = Mock(side_effect=responses)
        delays: Final = ExpectedCalls((1.0,))
        result: Final[Result[FileDeleteResponse]] = cleanup_result(outcomes, wait=delays)
        assert isinstance(result, Success) and result.data.deleted
        delays.assert_done()

    def test_persistent_error_has_bounded_retries(self) -> None:
        failure: Final = UnknownApiError(status_code=503, body="unavailable")
        outcomes: Final = Mock(return_value=failure)
        delays: Final = ExpectedCalls(CLEANUP_DELAYS)
        result: Final[Result[FileDeleteResponse]] = cleanup_result(outcomes, wait=delays)
        assert result is failure
        delays.assert_done()
        assert outcomes.call_count == len(CLEANUP_DELAYS) + 1

    def test_permanent_error_is_not_retried(self) -> None:
        failure: Final = UnknownApiError(status_code=403, body="forbidden")
        responses: Final = (failure, deleted_file())
        outcomes: Final = Mock(side_effect=responses)
        delays: Final = ExpectedCalls[float](())
        assert cleanup_result(outcomes, wait=delays) is failure
        delays.assert_done()
        assert outcomes.call_count == 1


class TestBatchCancellation:
    def test_cancelling_batch_is_polled_until_terminal_without_cancelling_again(self) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls((f"retrieve None {MANAGED_BATCH_ID}",) * 3),
            batches=(batch("cancelling"), batch("cancelling"), batch("cancelled")),
        )
        delays: Final = ExpectedCalls((10.0,))
        cleanup_batch(client, MANAGED_BATCH_ID, key="test-key", wait=delays)
        client.calls.assert_done()
        delays.assert_done()

    def test_cancellation_timeout_is_reported_but_file_and_key_cleanup_still_run(self) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(
                (
                    f"retrieve None {MANAGED_BATCH_ID}",
                    f"retrieve None {MANAGED_BATCH_ID}",
                    "delete None file-1",
                    "delete key test-key",
                )
            ),
            batches=(batch("cancelling"), batch("cancelling")),
            files=(deleted_file(),),
        )
        times: Final = (0.0, BATCH_CANCEL_TIMEOUT_SECONDS)
        ticks: Final[Callable[[], float]] = Mock(side_effect=times)
        manager: Final = ResourceManager(client=client, strict_cleanup=True)
        key: Final = manager.key()
        manager.defer(lambda: cleanup_file(client, "file-1", key=key))
        manager.defer(lambda: cleanup_batch(client, MANAGED_BATCH_ID, key=key, clock=ticks))
        with pytest.raises(ExceptionGroup) as caught:
            manager.teardown()
        assert "cancellation did not finish" in str(caught.value.exceptions[0])
        client.calls.assert_done()

    @pytest.mark.parametrize("status", ["completed", "failed", "expired", "cancelled"])
    def test_inactive_batch_needs_no_cancellation(self, status: str) -> None:
        client: Final = CleanupClient(calls=ExpectedCalls(("retrieve None batch-1",)), batches=(batch(status),))
        cleanup_batch(client, "batch-1", key="test-key")
        client.calls.assert_done()

    def test_active_batch_is_cancelled_through_its_provider(self) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(("retrieve azure batch-1", "cancel azure batch-1")),
            batches=(batch("in_progress"), batch("cancelled")),
            cancellations=(batch("cancelling"),),
        )
        cleanup_batch(client, "batch-1", key="test-key", provider="azure")
        client.calls.assert_done()

    @pytest.mark.parametrize("batch_id", ["batch-1", MANAGED_BATCH_ID])
    @pytest.mark.parametrize("pending_status", ["validating", "in_progress"])
    def test_accepted_cancellation_waits_through_stale_provider_status(
        self, batch_id: str, pending_status: str
    ) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(
                (
                    f"retrieve vertex_ai {batch_id}",
                    f"cancel vertex_ai {batch_id}",
                    f"retrieve vertex_ai {batch_id}",
                    f"retrieve vertex_ai {batch_id}",
                    f"retrieve vertex_ai {batch_id}",
                    "delete vertex_ai file-1",
                    "delete key test-key",
                )
            ),
            batches=(batch("validating"), batch(pending_status), batch(pending_status), batch("cancelled")),
            cancellations=(batch(pending_status),),
            files=(deleted_file(),),
        )
        delays: Final = ExpectedCalls((10.0, 10.0))
        manager: Final = ResourceManager(client=client, strict_cleanup=True)
        key: Final = manager.key()
        manager.defer(lambda: cleanup_file(client, "file-1", key=key, provider="vertex_ai"))
        manager.defer(lambda: cleanup_batch(client, batch_id, key=key, provider="vertex_ai", wait=delays))
        manager.teardown()
        client.calls.assert_done()
        delays.assert_done()

    @pytest.mark.parametrize("output_delete_fails", [False, True])
    def test_batch_that_completed_before_cleanup_deletes_output_and_error_files(
        self, output_delete_fails: bool
    ) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(("retrieve openai batch-1", "delete openai file-output", "delete openai file-error")),
            batches=(
                Success(
                    status_code=200,
                    data=BatchObject(
                        id="batch-1",
                        status="completed",
                        input_file_id="file-input",
                        output_file_id="file-output",
                        error_file_id="file-error",
                    ),
                ),
            ),
            files=(
                UnknownApiError(status_code=403, body="forbidden") if output_delete_fails else deleted_file(),
                deleted_file(),
            ),
        )
        if output_delete_fails:
            with pytest.raises(ExceptionGroup, match="output cleanup failed"):
                cleanup_batch(client, "batch-1", key="test-key", provider="openai", delete_output_files=True)
        else:
            cleanup_batch(client, "batch-1", key="test-key", provider="openai", delete_output_files=True)
        client.calls.assert_done()

    @pytest.mark.parametrize("status", ["completed", "in_progress"])
    def test_cancellation_conflict_is_accepted_only_when_batch_became_inactive(self, status: str) -> None:
        client: Final = CleanupClient(
            calls=ExpectedCalls(("retrieve None batch-1", "cancel None batch-1", "retrieve None batch-1")),
            batches=(batch("in_progress"), batch(status)),
            cancellations=(UnknownApiError(status_code=409, body="conflict"),),
        )
        if status == "completed":
            cleanup_batch(client, "batch-1", key="test-key")
        else:
            with pytest.raises(AssertionError, match="Cancel batch batch-1 left status in_progress"):
                cleanup_batch(client, "batch-1", key="test-key")
        client.calls.assert_done()


class TestAzureFileExpiry:
    def test_azure_form_serializes_native_expiry_for_the_proxy(self) -> None:
        form: Final = batch_upload_form("azure", target_model_names="azure-test")
        assert form.model_dump(by_alias=True, exclude_none=True) == {
            "purpose": "batch",
            "target_model_names": "azure-test",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": AZURE_FILE_EXPIRY_SECONDS,
        }

    @pytest.mark.parametrize("provider", ["openai", "vertex_ai", "bedrock"])
    def test_other_providers_keep_their_existing_upload_fields(self, provider: str) -> None:
        assert batch_upload_form(provider).model_dump(by_alias=True, exclude_none=True) == {"purpose": "batch"}
