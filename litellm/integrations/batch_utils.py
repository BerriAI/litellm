from collections.abc import Awaitable, Callable, Sequence
from typing import Final, TypeVar

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError

_BatchItem = TypeVar("_BatchItem")


async def send_batch_with_413_split(
    batch: Sequence[_BatchItem],
    send_batch: Callable[[Sequence[_BatchItem]], Awaitable[httpx.Response]],
    exceeds_limits: Callable[[Sequence[_BatchItem]], bool],
    success_status_codes: frozenset[int],
    integration_name: str,
    drop_error_message: str,
) -> tuple[_BatchItem, ...]:
    async def _halve() -> tuple[_BatchItem, ...]:
        midpoint: Final = len(batch) // 2
        left_undelivered: Final = await send_batch_with_413_split(
            batch=batch[:midpoint],
            send_batch=send_batch,
            exceeds_limits=exceeds_limits,
            success_status_codes=success_status_codes,
            integration_name=integration_name,
            drop_error_message=drop_error_message,
        )
        if left_undelivered:
            return left_undelivered + tuple(batch[midpoint:])
        return await send_batch_with_413_split(
            batch=batch[midpoint:],
            send_batch=send_batch,
            exceeds_limits=exceeds_limits,
            success_status_codes=success_status_codes,
            integration_name=integration_name,
            drop_error_message=drop_error_message,
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
    except (TypeError, ValueError) as e:
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
        verbose_logger.exception("%s Error sending batch API - %s", integration_name, e)
        return tuple(batch)
    except Exception as e:
        verbose_logger.exception("%s Error sending batch API - %s", integration_name, e)
        return tuple(batch)

    if response.status_code == 413:
        return await _handle_413()
    if response.status_code not in success_status_codes:
        verbose_logger.error(
            "%s API error: status_code=%s, response=%s",
            integration_name,
            response.status_code,
            response.text,
        )
        return tuple(batch)

    verbose_logger.debug("%s delivered %s records, status_code=%s", integration_name, len(batch), response.status_code)
    return ()
