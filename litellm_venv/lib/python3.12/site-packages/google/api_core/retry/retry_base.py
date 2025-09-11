# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared classes and functions for retrying requests.

:class:`_BaseRetry` is the base class for :class:`Retry`,
:class:`AsyncRetry`, :class:`StreamingRetry`, and :class:`AsyncStreamingRetry`.
"""

from __future__ import annotations

import logging
import random
import time

from enum import Enum
from typing import Any, Callable, Optional, Iterator, TYPE_CHECKING

import requests.exceptions

from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

_DEFAULT_INITIAL_DELAY = 1.0  # seconds
_DEFAULT_MAXIMUM_DELAY = 60.0  # seconds
_DEFAULT_DELAY_MULTIPLIER = 2.0
_DEFAULT_DEADLINE = 60.0 * 2.0  # seconds

_LOGGER = logging.getLogger("google.api_core.retry")


def if_exception_type(
    *exception_types: type[Exception],
) -> Callable[[Exception], bool]:
    """Creates a predicate to check if the exception is of a given type.

    Args:
        exception_types (Sequence[:func:`type`]): The exception types to check
            for.

    Returns:
        Callable[Exception]: A predicate that returns True if the provided
            exception is of the given type(s).
    """

    def if_exception_type_predicate(exception: Exception) -> bool:
        """Bound predicate for checking an exception type."""
        return isinstance(exception, exception_types)

    return if_exception_type_predicate


# pylint: disable=invalid-name
# Pylint sees this as a constant, but it is also an alias that should be
# considered a function.
if_transient_error = if_exception_type(
    exceptions.InternalServerError,
    exceptions.TooManyRequests,
    exceptions.ServiceUnavailable,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    auth_exceptions.TransportError,
)
"""A predicate that checks if an exception is a transient API error.

The following server errors are considered transient:

- :class:`google.api_core.exceptions.InternalServerError` - HTTP 500, gRPC
    ``INTERNAL(13)`` and its subclasses.
- :class:`google.api_core.exceptions.TooManyRequests` - HTTP 429
- :class:`google.api_core.exceptions.ServiceUnavailable` - HTTP 503
- :class:`requests.exceptions.ConnectionError`
- :class:`requests.exceptions.ChunkedEncodingError` - The server declared
    chunked encoding but sent an invalid chunk.
- :class:`google.auth.exceptions.TransportError` - Used to indicate an
    error occurred during an HTTP request.
"""
# pylint: enable=invalid-name


def exponential_sleep_generator(
    initial: float, maximum: float, multiplier: float = _DEFAULT_DELAY_MULTIPLIER
):
    """Generates sleep intervals based on the exponential back-off algorithm.

    This implements the `Truncated Exponential Back-off`_ algorithm.

    .. _Truncated Exponential Back-off:
        https://cloud.google.com/storage/docs/exponential-backoff

    Args:
        initial (float): The minimum amount of time to delay. This must
            be greater than 0.
        maximum (float): The maximum amount of time to delay.
        multiplier (float): The multiplier applied to the delay.

    Yields:
        float: successive sleep intervals.
    """
    max_delay = min(initial, maximum)
    while True:
        yield random.uniform(0.0, max_delay)
        max_delay = min(max_delay * multiplier, maximum)


class RetryFailureReason(Enum):
    """
    The cause of a failed retry, used when building exceptions
    """

    TIMEOUT = 0
    NON_RETRYABLE_ERROR = 1


def build_retry_error(
    exc_list: list[Exception],
    reason: RetryFailureReason,
    timeout_val: float | None,
    **kwargs: Any,
) -> tuple[Exception, Exception | None]:
    """
    Default exception_factory implementation.

    Returns a RetryError if the failure is due to a timeout, otherwise
    returns the last exception encountered.

    Args:
      - exc_list: list of exceptions that occurred during the retry
      - reason: reason for the retry failure.
            Can be TIMEOUT or NON_RETRYABLE_ERROR
      - timeout_val: the original timeout value for the retry (in seconds), for use in the exception message

    Returns:
      - tuple: a tuple of the exception to be raised, and the cause exception if any
    """
    if reason == RetryFailureReason.TIMEOUT:
        # return RetryError with the most recent exception as the cause
        src_exc = exc_list[-1] if exc_list else None
        timeout_val_str = f"of {timeout_val:0.1f}s " if timeout_val is not None else ""
        return (
            exceptions.RetryError(
                f"Timeout {timeout_val_str}exceeded",
                src_exc,
            ),
            src_exc,
        )
    elif exc_list:
        # return most recent exception encountered
        return exc_list[-1], None
    else:
        # no exceptions were given in exc_list. Raise generic RetryError
        return exceptions.RetryError("Unknown error", None), None


def _retry_error_helper(
    exc: Exception,
    deadline: float | None,
    sleep_iterator: Iterator[float],
    error_list: list[Exception],
    predicate_fn: Callable[[Exception], bool],
    on_error_fn: Callable[[Exception], None] | None,
    exc_factory_fn: Callable[
        [list[Exception], RetryFailureReason, float | None],
        tuple[Exception, Exception | None],
    ],
    original_timeout: float | None,
) -> float:
    """
    Shared logic for handling an error for all retry implementations

    - Raises an error on timeout or non-retryable error
    - Calls on_error_fn if provided
    - Logs the error

    Args:
       - exc: the exception that was raised
       - deadline: the deadline for the retry, calculated as a diff from time.monotonic()
       - sleep_iterator: iterator to draw the next backoff value from
       - error_list: the list of exceptions that have been raised so far
       - predicate_fn: takes `exc` and returns true if the operation should be retried
       - on_error_fn: callback to execute when a retryable error occurs
       - exc_factory_fn: callback used to build the exception to be raised on terminal failure
       - original_timeout_val: the original timeout value for the retry (in seconds),
           to be passed to the exception factory for building an error message
    Returns:
        - the sleep value chosen before the next attempt
    """
    error_list.append(exc)
    if not predicate_fn(exc):
        final_exc, source_exc = exc_factory_fn(
            error_list,
            RetryFailureReason.NON_RETRYABLE_ERROR,
            original_timeout,
        )
        raise final_exc from source_exc
    if on_error_fn is not None:
        on_error_fn(exc)
    # next_sleep is fetched after the on_error callback, to allow clients
    # to update sleep_iterator values dynamically in response to errors
    try:
        next_sleep = next(sleep_iterator)
    except StopIteration:
        raise ValueError("Sleep generator stopped yielding sleep values.") from exc
    if deadline is not None and time.monotonic() + next_sleep > deadline:
        final_exc, source_exc = exc_factory_fn(
            error_list,
            RetryFailureReason.TIMEOUT,
            original_timeout,
        )
        raise final_exc from source_exc
    _LOGGER.debug(
        "Retrying due to {}, sleeping {:.1f}s ...".format(error_list[-1], next_sleep)
    )
    return next_sleep


class _BaseRetry(object):
    """
    Base class for retry configuration objects. This class is intended to capture retry
    and backoff configuration that is common to both synchronous and asynchronous retries,
    for both unary and streaming RPCs. It is not intended to be instantiated directly,
    but rather to be subclassed by the various retry configuration classes.
    """

    def __init__(
        self,
        predicate: Callable[[Exception], bool] = if_transient_error,
        initial: float = _DEFAULT_INITIAL_DELAY,
        maximum: float = _DEFAULT_MAXIMUM_DELAY,
        multiplier: float = _DEFAULT_DELAY_MULTIPLIER,
        timeout: Optional[float] = _DEFAULT_DEADLINE,
        on_error: Optional[Callable[[Exception], Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._predicate = predicate
        self._initial = initial
        self._multiplier = multiplier
        self._maximum = maximum
        self._timeout = kwargs.get("deadline", timeout)
        self._deadline = self._timeout
        self._on_error = on_error

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Not implemented in base class")

    @property
    def deadline(self) -> float | None:
        """
        DEPRECATED: use ``timeout`` instead.  Refer to the ``Retry`` class
        documentation for details.
        """
        return self._timeout

    @property
    def timeout(self) -> float | None:
        return self._timeout

    def with_deadline(self, deadline: float | None) -> Self:
        """Return a copy of this retry with the given timeout.

        DEPRECATED: use :meth:`with_timeout` instead. Refer to the ``Retry`` class
        documentation for details.

        Args:
            deadline (float|None): How long to keep retrying, in seconds. If None,
                no timeout is enforced.

        Returns:
            Retry: A new retry instance with the given timeout.
        """
        return self.with_timeout(deadline)

    def with_timeout(self, timeout: float | None) -> Self:
        """Return a copy of this retry with the given timeout.

        Args:
            timeout (float): How long to keep retrying, in seconds. If None,
                no timeout will be enforced.

        Returns:
            Retry: A new retry instance with the given timeout.
        """
        return type(self)(
            predicate=self._predicate,
            initial=self._initial,
            maximum=self._maximum,
            multiplier=self._multiplier,
            timeout=timeout,
            on_error=self._on_error,
        )

    def with_predicate(self, predicate: Callable[[Exception], bool]) -> Self:
        """Return a copy of this retry with the given predicate.

        Args:
            predicate (Callable[Exception]): A callable that should return
                ``True`` if the given exception is retryable.

        Returns:
            Retry: A new retry instance with the given predicate.
        """
        return type(self)(
            predicate=predicate,
            initial=self._initial,
            maximum=self._maximum,
            multiplier=self._multiplier,
            timeout=self._timeout,
            on_error=self._on_error,
        )

    def with_delay(
        self,
        initial: Optional[float] = None,
        maximum: Optional[float] = None,
        multiplier: Optional[float] = None,
    ) -> Self:
        """Return a copy of this retry with the given delay options.

        Args:
            initial (float): The minimum amount of time to delay (in seconds). This must
                be greater than 0. If None, the current value is used.
            maximum (float): The maximum amount of time to delay (in seconds). If None, the
                current value is used.
            multiplier (float): The multiplier applied to the delay. If None, the current
                value is used.

        Returns:
            Retry: A new retry instance with the given delay options.
        """
        return type(self)(
            predicate=self._predicate,
            initial=initial if initial is not None else self._initial,
            maximum=maximum if maximum is not None else self._maximum,
            multiplier=multiplier if multiplier is not None else self._multiplier,
            timeout=self._timeout,
            on_error=self._on_error,
        )

    def __str__(self) -> str:
        return (
            "<{} predicate={}, initial={:.1f}, maximum={:.1f}, "
            "multiplier={:.1f}, timeout={}, on_error={}>".format(
                type(self).__name__,
                self._predicate,
                self._initial,
                self._maximum,
                self._multiplier,
                self._timeout,  # timeout can be None, thus no {:.1f}
                self._on_error,
            )
        )
