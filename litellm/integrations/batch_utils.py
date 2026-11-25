import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Final, Generic, TypeVar

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError

_BatchItem = TypeVar("_BatchItem")

_RETRYABLE_CLIENT_STATUS_CODES: Final = frozenset({408, 429})


def is_retryable_status(status_code: int) -> bool:
    return not 400 <= status_code < 500 or status_code in _RETRYABLE_CLIENT_STATUS_CODES


def undelivered_after_http_error(
    batch: Sequence[_BatchItem],
    status_code: int,
    integration_name: str,
    detail: str,
) -> tuple[_BatchItem, ...]:
    """The records to requeue after a non-2xx: all of them on a status a retry can clear, none on
    a 4xx that would only repeat, since retaining those retries a misconfiguration forever."""
    if is_retryable_status(status_code):
        verbose_logger.error(
            "%s API error: status_code=%s, will retry %s records - %s",
            integration_name,
            status_code,
            len(batch),
            detail,
        )
        return tuple(batch)
    verbose_logger.error(
        "%s API error: status_code=%s is not retryable, dropped %s records - %s",
        integration_name,
        status_code,
        len(batch),
        detail,
    )
    return ()


def requeue_after_http_error(
    batch: Sequence[_BatchItem],
    status_code: int,
    integration_name: str,
    detail: str,
) -> tuple[_BatchItem, ...]:
    verbose_logger.error(
        "%s API error: status_code=%s, will retry %s records - %s",
        integration_name,
        status_code,
        len(batch),
        detail,
    )
    return tuple(batch)


class BatchSendCancelled(asyncio.CancelledError, Generic[_BatchItem]):
    """Cancellation of a batch send, carrying only the records the destination never accepted.

    A batch split under the size cap is delivered in pieces, so requeueing all of it after a
    cancellation partway through would send the accepted pieces a second time.
    """

    def __init__(self, undelivered: tuple[_BatchItem, ...]) -> None:
        super().__init__()
        self.undelivered: Final = undelivered


async def _keep_the_remainder_on_cancel(
    send: Awaitable[tuple[_BatchItem, ...]],
    remainder: Sequence[_BatchItem],
) -> tuple[_BatchItem, ...]:
    try:
        return await send
    except BatchSendCancelled as cancelled:
        raise BatchSendCancelled((*cancelled.undelivered, *remainder)) from cancelled


async def send_batch_with_413_split(
    batch: Sequence[_BatchItem],
    send_batch: Callable[[Sequence[_BatchItem]], Awaitable[httpx.Response]],
    exceeds_limits: Callable[[Sequence[_BatchItem]], bool],
    success_status_codes: frozenset[int],
    integration_name: str,
    drop_error_message: str,
    non_success_handler: Callable[
        [Sequence[_BatchItem], int, str, str], tuple[_BatchItem, ...]
    ] = requeue_after_http_error,
) -> tuple[_BatchItem, ...]:
    async def _halve() -> tuple[_BatchItem, ...]:
        midpoint: Final = len(batch) // 2
        left_batch: Final = batch[:midpoint]
        right_batch: Final = batch[midpoint:]
        left_undelivered: Final = await _keep_the_remainder_on_cancel(
            send_batch_with_413_split(
                batch=left_batch,
                send_batch=send_batch,
                exceeds_limits=exceeds_limits,
                success_status_codes=success_status_codes,
                integration_name=integration_name,
                drop_error_message=drop_error_message,
                non_success_handler=non_success_handler,
            ),
            right_batch,
        )
        if left_undelivered:
            return (*left_undelivered, *right_batch)
        return await send_batch_with_413_split(
            batch=right_batch,
            send_batch=send_batch,
            exceeds_limits=exceeds_limits,
            success_status_codes=success_status_codes,
            integration_name=integration_name,
            drop_error_message=drop_error_message,
            non_success_handler=non_success_handler,
        )

    async def _handle_413() -> tuple[_BatchItem, ...]:
        if len(batch) == 1:
            verbose_logger.error(drop_error_message)
            return ()
        return await _halve()

    if not batch:
        return ()

    try:
        oversized: Final = exceeds_limits(batch)
    except Exception as e:  # noqa: BLE001  # any record that cannot be serialized is isolated and dropped alone
        if len(batch) > 1:
            return await _halve()
        verbose_logger.exception("%s dropped a record that cannot be serialized - %s", integration_name, e)
        return ()
    if oversized and len(batch) > 1:
        return await _halve()

    try:
        response: Final = await send_batch(batch)
    except MaskedHTTPStatusError as e:
        if e.status_code == 413:
            return await _handle_413()
        return non_success_handler(batch, e.status_code, integration_name, str(e))
    except asyncio.CancelledError as cancelled:
        raise BatchSendCancelled(tuple(batch)) from cancelled
    except Exception as e:
        verbose_logger.exception("%s Error sending batch API - %s", integration_name, e)
        return tuple(batch)

    if response.status_code == 413:
        return await _handle_413()
    if response.status_code not in success_status_codes:
        return non_success_handler(batch, response.status_code, integration_name, response.text)

    verbose_logger.debug("%s delivered %s records, status_code=%s", integration_name, len(batch), response.status_code)
    return ()
