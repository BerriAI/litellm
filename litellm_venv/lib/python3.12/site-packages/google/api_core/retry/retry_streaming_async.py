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
Generator wrapper for retryable async streaming RPCs.
"""
from __future__ import annotations

from typing import (
    cast,
    Any,
    Callable,
    Iterable,
    AsyncIterator,
    AsyncIterable,
    Awaitable,
    TypeVar,
    AsyncGenerator,
    TYPE_CHECKING,
)

import asyncio
import time
import sys
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


async def retry_target_stream(
    target: Callable[_P, AsyncIterable[_Y] | Awaitable[AsyncIterable[_Y]]],
    predicate: Callable[[Exception], bool],
    sleep_generator: Iterable[float],
    timeout: float | None = None,
    on_error: Callable[[Exception], None] | None = None,
    exception_factory: Callable[
        [list[Exception], RetryFailureReason, float | None],
        tuple[Exception, Exception | None],
    ] = build_retry_error,
    init_args: tuple = (),
    init_kwargs: dict = {},
    **kwargs,
) -> AsyncGenerator[_Y, None]:
    """Create a generator wrapper that retries the wrapped stream if it fails.

    This is the lowest-level retry helper. Generally, you'll use the
    higher-level retry helper :class:`AsyncRetry`.

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
        AsyncGenerator: A retryable generator that wraps the target generator function.

    Raises:
        ValueError: If the sleep generator stops yielding values.
        Exception: a custom exception specified by the exception_factory if provided.
            If no exception_factory is provided:
                google.api_core.RetryError: If the timeout is exceeded while retrying.
                Exception: If the target raises an error that isn't retryable.
    """
    target_iterator: AsyncIterator[_Y] | None = None
    timeout = kwargs.get("deadline", timeout)
    deadline = time.monotonic() + timeout if timeout else None
    # keep track of retryable exceptions we encounter to pass in to exception_factory
    error_list: list[Exception] = []
    sleep_iter = iter(sleep_generator)
    target_is_generator: bool | None = None

    # continue trying until an attempt completes, or a terminal exception is raised in _retry_error_helper
    # TODO: support max_attempts argument: https://github.com/googleapis/python-api-core/issues/535
    while True:
        # Start a new retry loop
        try:
            # Note: in the future, we can add a ResumptionStrategy object
            # to generate new args between calls. For now, use the same args
            # for each attempt.
            target_output: AsyncIterable[_Y] | Awaitable[AsyncIterable[_Y]] = target(
                *init_args, **init_kwargs
            )
            try:
                # gapic functions return the generator behind an awaitable
                # unwrap the awaitable so we can work with the generator directly
                target_output = await target_output  # type: ignore
            except TypeError:
                # was not awaitable, continue
                pass
            target_iterator = cast(AsyncIterable["_Y"], target_output).__aiter__()

            if target_is_generator is None:
                # Check if target supports generator features (asend, athrow, aclose)
                target_is_generator = bool(getattr(target_iterator, "asend", None))

            sent_in = None
            while True:
                ## Read from target_iterator
                # If the target is a generator, we will advance it with `asend`
                # otherwise, we will use `anext`
                if target_is_generator:
                    next_value = await target_iterator.asend(sent_in)  # type: ignore
                else:
                    next_value = await target_iterator.__anext__()
                ## Yield from Wrapper to caller
                try:
                    # yield latest value from target
                    # exceptions from `athrow` and `aclose` are injected here
                    sent_in = yield next_value
                except GeneratorExit:
                    # if wrapper received `aclose` while waiting on yield,
                    # it will raise GeneratorExit here
                    if target_is_generator:
                        # pass to inner target_iterator for handling
                        await cast(AsyncGenerator["_Y", None], target_iterator).aclose()
                    else:
                        raise
                    return
                except:  # noqa: E722
                    # bare except catches any exception passed to `athrow`
                    if target_is_generator:
                        # delegate error handling to target_iterator
                        await cast(AsyncGenerator["_Y", None], target_iterator).athrow(
                            cast(BaseException, sys.exc_info()[1])
                        )
                    else:
                        raise
            return
        except StopAsyncIteration:
            # if iterator exhausted, return
            return
        # handle exceptions raised by the target_iterator
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
            await asyncio.sleep(next_sleep)

        finally:
            if target_is_generator and target_iterator is not None:
                await cast(AsyncGenerator["_Y", None], target_iterator).aclose()


class AsyncStreamingRetry(_BaseRetry):
    """Exponential retry decorator for async streaming rpcs.

    This class returns an AsyncGenerator when called, which wraps the target
    stream in retry logic. If any exception is raised by the target, the
    entire stream will be retried within the wrapper.

    Although the default behavior is to retry transient API errors, a
    different predicate can be provided to retry other exceptions.

    Important Note: when a stream is encounters a retryable error, it will
    silently construct a fresh iterator instance in the background
    and continue yielding (likely duplicate) values as if no error occurred.
    This is the most general way to retry a stream, but it often is not the
    desired behavior. Example: iter([1, 2, 1/0]) -> [1, 2, 1, 2, ...]

    There are two ways to build more advanced retry logic for streams:

    1. Wrap the target
        Use a ``target`` that maintains state between retries, and creates a
        different generator on each retry call. For example, you can wrap a
        grpc call in a function that modifies the request based on what has
        already been returned:

        .. code-block:: python

            async def attempt_with_modified_request(target, request, seen_items=[]):
                # remove seen items from request on each attempt
                new_request = modify_request(request, seen_items)
                new_generator = await target(new_request)
                async for item in new_generator:
                    yield item
                    seen_items.append(item)

            retry_wrapped = AsyncRetry(is_stream=True,...)(attempt_with_modified_request, target, request, [])

        2. Wrap the retry generator
            Alternatively, you can wrap the retryable generator itself before
            passing it to the end-user to add a filter on the stream. For
            example, you can keep track of the items that were successfully yielded
            in previous retry attempts, and only yield new items when the
            new attempt surpasses the previous ones:

            .. code-block:: python

                async def retryable_with_filter(target):
                    stream_idx = 0
                    # reset stream_idx when the stream is retried
                    def on_error(e):
                        nonlocal stream_idx
                        stream_idx = 0
                    # build retryable
                    retryable_gen = AsyncRetry(is_stream=True, ...)(target)
                    # keep track of what has been yielded out of filter
                    seen_items = []
                    async for item in retryable_gen:
                        if stream_idx >= len(seen_items):
                            yield item
                            seen_items.append(item)
                        elif item != previous_stream[stream_idx]:
                            raise ValueError("Stream differs from last attempt")"
                        stream_idx += 1

                filter_retry_wrapped = retryable_with_filter(target)

    Args:
        predicate (Callable[Exception]): A callable that should return ``True``
            if the given exception is retryable.
        initial (float): The minimum amount of time to delay in seconds. This
            must be greater than 0.
        maximum (float): The maximum amount of time to delay in seconds.
        multiplier (float): The multiplier applied to the delay.
        timeout (Optional[float]): How long to keep retrying in seconds.
            Note: timeout is only checked before initiating a retry, so the target may
            run past the timeout value as long as it is healthy.
        on_error (Optional[Callable[Exception]]): A function to call while processing
            a retryable exception. Any error raised by this function will
            *not* be caught.
        is_stream (bool): Indicates whether the input function
            should be treated as a stream function (i.e. an AsyncGenerator,
            or function or coroutine that returns an AsyncIterable).
            If True, the iterable will be wrapped with retry logic, and any
            failed outputs will restart the stream. If False, only the input
            function call itself will be retried. Defaults to False.
            To avoid duplicate values, retryable streams should typically be
            wrapped in additional filter logic before use.
        deadline (float): DEPRECATED use ``timeout`` instead. If set it will
        override ``timeout`` parameter.
    """

    def __call__(
        self,
        func: Callable[..., AsyncIterable[_Y] | Awaitable[AsyncIterable[_Y]]],
        on_error: Callable[[Exception], Any] | None = None,
    ) -> Callable[_P, Awaitable[AsyncGenerator[_Y, None]]]:
        """Wrap a callable with retry behavior.

        Args:
            func (Callable): The callable or stream to add retry behavior to.
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
        async def retry_wrapped_func(
            *args: _P.args, **kwargs: _P.kwargs
        ) -> AsyncGenerator[_Y, None]:
            """A wrapper that calls target function with retry."""
            sleep_generator = exponential_sleep_generator(
                self._initial, self._maximum, multiplier=self._multiplier
            )
            return retry_target_stream(
                func,
                self._predicate,
                sleep_generator,
                self._timeout,
                on_error,
                init_args=args,
                init_kwargs=kwargs,
            )

        return retry_wrapped_func
