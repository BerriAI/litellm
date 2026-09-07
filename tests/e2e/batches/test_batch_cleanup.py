from builtins import ExceptionGroup
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

import pytest

from batch_cleanup import BATCH_CANCEL_TIMEOUT_SECONDS, CLEANUP_DELAYS, cleanup_batch, cleanup_file, cleanup_result
from batch_client import AZURE_FILE_EXPIRY_SECONDS, BatchObject, FileDeleteResponse, batch_upload_form
from capabilities import CAPABILITIES, Capability
from e2e_http import NetworkError, RateLimitedError, Result, Success, UnknownApiError
from lifecycle import ResourceManager
from models import KeyGenerateBody


@dataclass
class CleanupClient:
    files: Iterator[Result[FileDeleteResponse]] = field(default_factory=lambda: iter(()))
    batches: Iterator[Result[BatchObject]] = field(default_factory=lambda: iter(()))
    cancellations: Iterator[Result[BatchObject]] = field(default_factory=lambda: iter(()))
    calls: list[str] = field(default_factory=list)

    def delete_file(self, file_id: str, *, key: str, provider: str | None = None) -> Result[FileDeleteResponse]:
        self.calls.append(f"delete {provider} {file_id}")
        return next(self.files)

    def retrieve_batch(self, batch_id: str, *, key: str, provider: str | None = None) -> Result[BatchObject]:
        self.calls.append(f"retrieve {provider} {batch_id}")
        return next(self.batches)

    def cancel_batch(self, batch_id: str, *, key: str, provider: str | None = None) -> Result[BatchObject]:
        self.calls.append(f"cancel {provider} {batch_id}")
        return next(self.cancellations)

    def generate_key(self, body: KeyGenerateBody) -> str:
        return "test-key"

    def delete_key(self, key: str) -> None:
        self.calls.append(f"delete key {key}")

    def delete_customers(self, user_ids: list[str]) -> None:
        self.calls.append(f"delete customers {user_ids}")


def batch(status: str) -> Success[BatchObject]:
    return Success(status_code=200, data=BatchObject(id="batch-1", status=status))


def deleted_file(*, deleted: bool = True) -> Success[FileDeleteResponse]:
    return Success(status_code=200, data=FileDeleteResponse(id="file-1", deleted=deleted))


class TestFileCleanup:
    @pytest.mark.parametrize("cap", CAPABILITIES, ids=[cap.id for cap in CAPABILITIES])
    def test_deletes_raw_files_through_the_upload_provider(self, cap: Capability) -> None:
        client: Final = CleanupClient(files=iter((deleted_file(),)))
        cleanup_file(client, "file-1", key="test-key", provider=cap.file_provider)
        expected_provider: Final = cap.provider if cap.scenario in {"model_param", "provider_fallback"} else None
        assert client.calls == [f"delete {expected_provider} file-1"]

    def test_failed_delete_is_reported_after_remaining_resources_are_cleaned(self) -> None:
        client: Final = CleanupClient(files=iter((UnknownApiError(status_code=403, body="secret response"),)))
        manager: Final = ResourceManager(client=client, strict_cleanup=True)
        key: Final = manager.key()
        manager.defer(lambda: cleanup_file(client, "file-1", key=key, provider="azure"))
        with pytest.raises(ExceptionGroup) as caught:
            manager.teardown()
        assert client.calls == ["delete azure file-1", "delete key test-key"]
        assert len(caught.value.exceptions) == 1
        assert str(caught.value.exceptions[0]) == "Delete file file-1 failed: HTTP 403"

    def test_success_response_must_confirm_deletion(self) -> None:
        client: Final = CleanupClient(files=iter((deleted_file(deleted=False),)))
        with pytest.raises(AssertionError, match="did not confirm deletion"):
            cleanup_file(client, "file-1", key="test-key")

    def test_cleanup_is_idempotent_when_file_is_already_deleted(self) -> None:
        client: Final = CleanupClient(files=iter((UnknownApiError(status_code=404, body="missing"),)))
        cleanup_file(client, "file-1", key="test-key", provider="azure")
        assert client.calls == ["delete azure file-1"]

    def test_default_resource_cleanup_keeps_existing_best_effort_behavior(self) -> None:
        client: Final = CleanupClient(files=iter((UnknownApiError(status_code=403, body="forbidden"),)))
        manager: Final = ResourceManager(client=client)
        key: Final = manager.key()
        manager.defer(lambda: cleanup_file(client, "file-1", key=key))
        manager.teardown()
        assert client.calls == ["delete None file-1", "delete key test-key"]


class TestCleanupRetries:
    @pytest.mark.parametrize(
        "failure",
        [NetworkError(message="offline"), RateLimitedError(), UnknownApiError(status_code=503, body="unavailable")],
    )
    def test_transient_error_retries_and_returns_success(self, failure: Result[FileDeleteResponse]) -> None:
        outcomes: Final = iter((failure, deleted_file()))
        delays: Final[list[float]] = []
        result: Final[Result[FileDeleteResponse]] = cleanup_result(lambda: next(outcomes), wait=delays.append)
        assert isinstance(result, Success) and result.data.deleted
        assert delays == [1.0]

    def test_persistent_error_has_bounded_retries(self) -> None:
        failure: Final = UnknownApiError(status_code=503, body="unavailable")
        outcomes: Final[Iterator[Result[FileDeleteResponse]]] = iter((failure,) * (len(CLEANUP_DELAYS) + 1))
        delays: Final[list[float]] = []
        result: Final[Result[FileDeleteResponse]] = cleanup_result(lambda: next(outcomes), wait=delays.append)
        assert result is failure
        assert tuple(delays) == CLEANUP_DELAYS
        assert next(outcomes, None) is None

    def test_permanent_error_is_not_retried(self) -> None:
        failure: Final = UnknownApiError(status_code=403, body="forbidden")
        outcomes: Final = iter((failure, deleted_file()))
        delays: Final[list[float]] = []
        assert cleanup_result(lambda: next(outcomes), wait=delays.append) is failure
        assert delays == []
        assert isinstance(next(outcomes), Success)


class TestBatchCancellation:
    def test_cancelling_batch_is_polled_until_terminal_without_cancelling_again(self) -> None:
        client: Final = CleanupClient(batches=iter((batch("cancelling"), batch("cancelling"), batch("cancelled"))))
        delays: Final[list[float]] = []
        cleanup_batch(client, "batch-1", key="test-key", wait=delays.append)
        assert client.calls == ["retrieve None batch-1"] * 3
        assert delays == [10.0]

    def test_cancellation_timeout_is_reported_but_file_and_key_cleanup_still_run(self) -> None:
        client: Final = CleanupClient(
            batches=iter((batch("cancelling"), batch("cancelling"))), files=iter((deleted_file(),))
        )
        ticks: Final = iter((0.0, BATCH_CANCEL_TIMEOUT_SECONDS))
        manager: Final = ResourceManager(client=client, strict_cleanup=True)
        key: Final = manager.key()
        manager.defer(lambda: cleanup_file(client, "file-1", key=key))
        manager.defer(lambda: cleanup_batch(client, "batch-1", key=key, clock=lambda: next(ticks)))
        with pytest.raises(ExceptionGroup) as caught:
            manager.teardown()
        assert "cancellation did not finish" in str(caught.value.exceptions[0])
        assert client.calls == [
            "retrieve None batch-1",
            "retrieve None batch-1",
            "delete None file-1",
            "delete key test-key",
        ]

    @pytest.mark.parametrize("status", ["completed", "failed", "expired", "cancelled"])
    def test_inactive_batch_needs_no_cancellation(self, status: str) -> None:
        client: Final = CleanupClient(batches=iter((batch(status),)))
        cleanup_batch(client, "batch-1", key="test-key")
        assert client.calls == ["retrieve None batch-1"]

    def test_active_batch_is_cancelled_through_its_provider(self) -> None:
        client: Final = CleanupClient(
            batches=iter((batch("in_progress"), batch("cancelled"))), cancellations=iter((batch("cancelling"),))
        )
        cleanup_batch(client, "batch-1", key="test-key", provider="azure")
        assert client.calls == ["retrieve azure batch-1", "cancel azure batch-1", "retrieve azure batch-1"]

    @pytest.mark.parametrize("status", ["completed", "in_progress"])
    def test_cancellation_conflict_is_accepted_only_when_batch_became_inactive(self, status: str) -> None:
        client: Final = CleanupClient(
            batches=iter((batch("in_progress"), batch(status))),
            cancellations=iter((UnknownApiError(status_code=409, body="conflict"),)),
        )
        if status == "completed":
            cleanup_batch(client, "batch-1", key="test-key")
        else:
            with pytest.raises(AssertionError, match="Cancel batch batch-1 left status in_progress"):
                cleanup_batch(client, "batch-1", key="test-key")
        assert client.calls == ["retrieve None batch-1", "cancel None batch-1", "retrieve None batch-1"]


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
