from collections.abc import Callable
from time import monotonic, sleep
from typing import Final, Protocol

from pydantic import BaseModel

from batch_client import BatchObject, FileDeleteResponse
from capabilities import is_managed_id
from e2e_http import NetworkError, RateLimitedError, Result, Success, UnknownApiError

CLEANUP_DELAYS: Final = (1.0, 2.0, 4.0)
BATCH_TERMINAL_STATUSES: Final = frozenset({"completed", "failed", "expired", "cancelled"})
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
    wait: Callable[[float], None] = sleep,
    clock: Callable[[], float] = monotonic,
) -> None:
    needs_terminal_state: Final = is_managed_id(batch_id)
    fetched: Final = _require_cleanup_success(
        cleanup_result(lambda: client.retrieve_batch(batch_id, key=key, provider=provider)),
        f"Retrieve batch {batch_id} for cleanup",
    )
    if fetched.status in BATCH_TERMINAL_STATUSES:
        return
    if fetched.status == "cancelling" and not needs_terminal_state:
        return
    if fetched.status != "cancelling":
        result: Final = cleanup_result(lambda: client.cancel_batch(batch_id, key=key, provider=provider))
        if not (isinstance(result, UnknownApiError) and result.status_code in {400, 409}):
            cancelled: Final = _require_cleanup_success(result, f"Cancel batch {batch_id}")
            assert cancelled.status in BATCH_TERMINAL_STATUSES | {"cancelling"}, (
                f"Cancel batch {batch_id} left status {cancelled.status}"
            )
            if not needs_terminal_state:
                return
    deadline: Final = clock() + BATCH_CANCEL_TIMEOUT_SECONDS
    while True:
        current = _require_cleanup_success(
            cleanup_result(lambda: client.retrieve_batch(batch_id, key=key, provider=provider)),
            f"Retrieve batch {batch_id} after cancellation",
        )
        if current.status in BATCH_TERMINAL_STATUSES:
            return
        assert current.status == "cancelling", f"Cancel batch {batch_id} left status {current.status}"
        if not needs_terminal_state:
            return
        assert clock() < deadline, (
            f"Batch {batch_id} cancellation did not finish within {BATCH_CANCEL_TIMEOUT_SECONDS}s"
        )
        wait(BATCH_CANCEL_POLL_SECONDS)
