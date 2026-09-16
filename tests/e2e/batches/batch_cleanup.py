from builtins import ExceptionGroup
from collections.abc import Callable
from itertools import count
from time import monotonic, sleep
from typing import Final, Protocol

from batch_client import BatchObject, FileDeleteResponse
from capabilities import is_managed_id
from e2e_http import NetworkError, RateLimitedError, Result, Success, UnknownApiError
from pydantic import BaseModel

CLEANUP_DELAYS: Final = (1.0, 2.0, 4.0)
BATCH_TERMINAL_STATUSES: Final = frozenset({"completed", "failed", "expired", "cancelled"})
BATCH_PENDING_STATUSES: Final = frozenset({"validating", "in_progress", "finalizing", "cancelling"})
BATCH_CANCEL_TIMEOUT_SECONDS: Final = 660.0
BATCH_CANCEL_POLL_SECONDS: Final = 10.0


class BatchCleanupClient(Protocol):
    def delete_file(self, file_id: str, *, key: str, provider: str | None = None) -> Result[FileDeleteResponse]: ...

    def retrieve_batch(self, batch_id: str, *, key: str, provider: str | None = None) -> Result[BatchObject]: ...

    def cancel_batch(self, batch_id: str, *, key: str, provider: str | None = None) -> Result[BatchObject]: ...


def cleanup_result[R: BaseModel](
    action: Callable[[], Result[R]], *, wait: Callable[[float], None] = sleep
) -> Result[R]:
    for delay, result in ((delay, action()) for delay in CLEANUP_DELAYS):
        match result:
            case NetworkError() | RateLimitedError():
                wait(delay)
            case UnknownApiError(status_code=code) if code in {408, 429, 500, 502, 503, 504}:
                wait(delay)
            case _:
                return result
    return action()


def _require_cleanup_success[R: BaseModel](result: Result[R], operation: str) -> R:
    match result:
        case Success(data=data):
            return data
        case UnknownApiError(status_code=code):
            raise AssertionError(f"{operation} failed: HTTP {code}")
        case _:
            raise AssertionError(f"{operation} failed: {result.kind}")


def cleanup_file(client: BatchCleanupClient, file_id: str, *, key: str, provider: str | None = None) -> None:
    result: Final = cleanup_result(lambda: client.delete_file(file_id, key=key, provider=provider))
    if isinstance(result, UnknownApiError) and result.status_code == 404:
        return
    deleted: Final = _require_cleanup_success(result, f"Delete file {file_id}")
    assert deleted.deleted is True or (
        deleted.deleted is None and is_managed_id(file_id) and deleted.id == file_id and deleted.object == "file"
    ), f"Delete file {file_id} did not confirm deletion"


def cleanup_batch(
    client: BatchCleanupClient,
    batch_id: str,
    *,
    key: str,
    provider: str | None = None,
    delete_output_files: bool = False,
    wait: Callable[[float], None] = sleep,
    clock: Callable[[], float] = monotonic,
) -> None:
    needs_terminal_state: Final = is_managed_id(batch_id)
    fetched: Final = _require_cleanup_success(
        cleanup_result(lambda: client.retrieve_batch(batch_id, key=key, provider=provider)),
        f"Retrieve batch {batch_id} for cleanup",
    )
    if fetched.status in BATCH_TERMINAL_STATUSES:
        if delete_output_files:
            _cleanup_batch_outputs(client, fetched, key=key, provider=provider)
        return
    if fetched.status == "cancelling" and not needs_terminal_state:
        return
    result: Final = (
        Success(status_code=200, data=fetched)
        if fetched.status == "cancelling"
        else cleanup_result(lambda: client.cancel_batch(batch_id, key=key, provider=provider))
    )
    conflicted: Final = isinstance(result, UnknownApiError) and result.status_code in {400, 409}
    if not conflicted:
        cancelled: Final = _require_cleanup_success(result, f"Cancel batch {batch_id}")
        assert cancelled.status in BATCH_TERMINAL_STATUSES | BATCH_PENDING_STATUSES, (
            f"Cancel batch {batch_id} left status {cancelled.status}"
        )
        if cancelled.status in BATCH_TERMINAL_STATUSES:
            if delete_output_files:
                _cleanup_batch_outputs(client, cancelled, key=key, provider=provider)
            return
        if cancelled.status == "cancelling" and not needs_terminal_state:
            return
    deadline: Final = clock() + BATCH_CANCEL_TIMEOUT_SECONDS
    for current in (
        _require_cleanup_success(
            cleanup_result(lambda: client.retrieve_batch(batch_id, key=key, provider=provider)),
            f"Retrieve batch {batch_id} after cancellation",
        )
        for _ in count()
    ):
        if current.status in BATCH_TERMINAL_STATUSES:
            if delete_output_files:
                _cleanup_batch_outputs(client, current, key=key, provider=provider)
            return
        assert current.status in ({"cancelling"} if conflicted else BATCH_PENDING_STATUSES), (
            f"Cancel batch {batch_id} left status {current.status}"
        )
        if current.status == "cancelling" and not needs_terminal_state:
            return
        assert clock() < deadline, (
            f"Batch {batch_id} cancellation did not finish within {BATCH_CANCEL_TIMEOUT_SECONDS}s"
        )
        wait(BATCH_CANCEL_POLL_SECONDS)


def _cleanup_batch_outputs(client: BatchCleanupClient, batch: BatchObject, *, key: str, provider: str | None) -> None:
    errors: Final = tuple(
        error
        for file_id in dict.fromkeys((batch.output_file_id, batch.error_file_id))
        if file_id is not None and file_id != batch.input_file_id
        if (error := _output_cleanup_error(client, file_id, key=key, provider=provider)) is not None
    )
    if errors:
        raise ExceptionGroup(f"Batch {batch.id} output cleanup failed", errors)


def _output_cleanup_error(
    client: BatchCleanupClient, file_id: str, *, key: str, provider: str | None
) -> Exception | None:
    try:
        cleanup_file(client, file_id, key=key, provider=provider)
    except Exception as error:
        return error
    return None
