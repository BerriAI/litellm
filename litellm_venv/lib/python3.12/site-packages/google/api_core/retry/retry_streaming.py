# Copyright 2023 Google LLC
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

"""
Generator wrapper for retryable streaming RPCs.
"""
from __future__ import annotations

from typing import (
    Callable,
    Optional,
    List,
    Tuple,
    Iterable,
    Generator,
    TypeVar,
    Any,
    TYPE_CHECKING,
)

import sys
import time
import functools

from google.api_core.retry.retry_base import _BaseRetry
from google.api_core.retry.retry_base import _retry_error_helper
from google.api_core.retry import exponential_sleep_generator
from google.api_core.retry import build_retry_error
from google.api_core.retry import RetryFailureReason

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    _P = ParamSpec("_P")  # target function call parameters
    _Y = TypeVar("_Y")  # yielded values


def retry_target_stream(
    target: Callable[_P, Iterable[_Y]],
    predicate: Callable[[Exception], bool],
    sleep_generator: Iterable[float],
    timeout: Optional[float] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    exception_factory: Callable[
        [List[Exception], RetryFailureReason, Optional[float]],
        Tuple[Exception, Optional[Exception]],
    ] = build_retry_error,
    init_args: tuple = (),
    init_kwargs: dict = {},
    **kwargs,
) -> Generator[_Y, Any, None]:
    """Create a generator wrapper that retries the wrapped stream if it fails.

    This is the lowest-level retry helper. Generally, you'll use the
    higher-level retry helper :class:`Retry`.

    Args:
        target: The generator function to call and retry.
        predicate: A callable used to determine if an
            exception raised by the target should be considered retryable.
            It should return True to retry or False otherwise.
        sleep_generator: An infinite iterator that determines
            how long to sleep between retries.
        timeout: How long to keep retrying the target.
            Note: timeout is only checked before initiating a retry, so the target may
            run past the timeout value as long as it is healthy.
        on_error: If given, the on_error callback will be called with each
            retryable exception raised by the target. Any error raised by this
            function will *not* be caught.
        exception_factory: A function that is called when the retryable reaches
            a terminal failure state, used to construct an exception to be raised.
            It takes a list of all exceptions encountered, a retry.RetryFailureReason
            enum indicating the failure cause, and the original timeout value
            as arguments. It should return a tuple of the exception to be raised,
            along with the cause exception if any. The default implementation will raise
            a RetryError on timeout, or the last exception encountered otherwise.
        init_args: Positional arguments to pass to the target function.
        init_kwargs: Keyword arguments to pass to the target function.

    Returns:
        Generator: A retryable generator that wraps the target generator function.

    Raises:
        ValueError: If the sleep generator stops yielding values.
        Exception: a custom exception specified by the exception_factory if provided.
            If no exception_factory is provided:
                google.api_core.RetryError: If the timeout is exceeded while retrying.
                Exception: If the target raises an error that isn't retryable.
    """

    timeout = kwargs.get("deadline", timeout)
    deadline: Optional[float] = (
        time.monotonic() + timeout if timeout is not None else None
    )
    error_list: list[Exception] = []
    sleep_iter = iter(sleep_generator)

    # continue trying until an attempt completes, or a terminal exception is raised in _retry_error_helper
    # TODO: support max_attempts argument: https://github.com/googleapis/python-api-core/issues/535
    while True:
        # Start a new retry loop
        try:
            # Note: in the future, we can add a ResumptionStrategy object
            # to generate new args between calls. For now, use the same args
            # for each attempt.
            subgenerator = target(*init_args, **init_kwargs)
            return (yield from subgenerator)
        # handle exceptions raised by the subgenerator
        # pylint: disable=broad-except
        # This function explicitly must deal with broad exceptions.
        except Exception as exc:
            # defer to shared logic for handling errors
            next_sleep = _retry_error_helper(
                exc,
                deadline,
                sleep_iter,
                error_list,
                predicate,
                on_error,
                exception_factory,
                timeout,
            )
            # if exception not raised, sleep before next attempt
            time.sleep(next_sleep)


class StreamingRetry(_BaseRetry):
    """Exponential retry decorator for streaming synchronous RPCs.

    This class returns a Generator when called, which wraps the target
    stream in retry logic. If any exception is raised by the target, the
    entire stream will be retried within the wrapper.

    Although the default behavior is to retry transient API errors, a
    different predicate can be provided to retry other exceptions.

    Important Note: when a stream encounters a retryable error, it will
    silently construct a fresh iterator instance in the background
    and continue yielding (likely duplicate) values as if no error occurred.
    This is the most general way to retry a stream, but it often is not the
    desired behavior. Example: iter([1, 2, 1/0]) -> [1, 2, 1, 2, ...]

    There are two ways to build more advanced retry logic for streams:

    1. Wrap the target
        Use a ``target`` that maintains state between retries, and creates a
        different generator on each retry call. For example, you can wrap a
        network call in a function that modifies the request based on what has
        already been returned:

        .. code-block:: python

            def attempt_with_modified_request(target, request, seen_items=[]):
                # remove seen items from request on each attempt
                new_request = modify_request(request, seen_items)
                new_generator = target(new_request)
                for item in new_generator:
                    yield item
                    seen_items.append(item)

            retry_wrapped_fn = StreamingRetry()(attempt_with_modified_request)
            retryable_generator = retry_wrapped_fn(target, request)

    2. Wrap the retry generator
        Alternatively, you can wrap the retryable generator itself before
        passing it to the end-user to add a filter on the stream. For
        example, you can keep track of the items that were successfully yielded
        in previous retry attempts, and only yield new items when the
        new attempt surpasses the previous ones:

        .. code-block:: python

            def retryable_with_filter(target):
                stream_idx = 0
                # reset stream_idx when the stream is retried
                def on_error(e):
                    nonlocal stream_idx
                    stream_idx = 0
                # build retryable
                retryable_gen = StreamingRetry(...)(target)
                # keep track of what has been yielded out of filter
                seen_items = []
                for item in retryable_gen():
                    if stream_idx >= len(seen_items):
                        seen_items.append(item)
                        yield item
                    elif item != seen_items[stream_idx]:
                        raise ValueError("Stream differs from last attempt")
                    stream_idx += 1

            filter_retry_wrapped = retryable_with_filter(target)

    Args:
        predicate (Callable[Exception]): A callable that should return ``True``
            if the given exception is retryable.
        initial (float): The minimum amount of time to delay in seconds. This
            must be greater than 0.
        maximum (float): The maximum amount of time to delay in seconds.
        multiplier (float): The multiplier applied to the delay.
        timeout (float): How long to keep retrying, in seconds.
            Note: timeout is only checked before initiating a retry, so the target may
            run past the timeout value as long as it is healthy.
        on_error (Callable[Exception]): A function to call while processing
            a retryable exception. Any error raised by this function will
            *not* be caught.
        deadline (float): DEPRECATED: use `timeout` instead. For backward
            compatibility, if specified it will override the ``timeout`` parameter.
    """

    def __call__(
        self,
        func: Callable[_P, Iterable[_Y]],
        on_error: Callable[[Exception], Any] | None = None,
    ) -> Callable[_P, Generator[_Y, Any, None]]:
        """Wrap a callable with retry behavior.

        Args:
            func (Callable): The callable to add retry behavior to.
            on_error (Optional[Callable[Exception]]): If given, the
                on_error callback will be called with each retryable exception
                raised by the wrapped function. Any error raised by this
                function will *not* be caught. If on_error was specified in the
                constructor, this value will be ignored.

        Returns:
            Callable: A callable that will invoke ``func`` with retry
                behavior.
        """
        if self._on_error is not None:
            on_error = self._on_error

        @functools.wraps(func)
        def retry_wrapped_func(
            *args: _P.args, **kwargs: _P.kwargs
        ) -> Generator[_Y, Any, None]:
            """A wrapper that calls target function with retry."""
            sleep_generator = exponential_sleep_generator(
                self._initial, self._maximum, multiplier=self._multiplier
            )
            return retry_target_stream(
                func,
                predicate=self._predicate,
                sleep_generator=sleep_generator,
                timeout=self._timeout,
                on_error=on_error,
                init_args=args,
                init_kwargs=kwargs,
            )

        return retry_wrapped_func
