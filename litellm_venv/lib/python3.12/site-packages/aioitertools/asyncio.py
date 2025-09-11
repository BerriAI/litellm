# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

"""
Friendlier version of asyncio standard library.

Provisional library.  Must be imported as `aioitertools.asyncio`.
"""

import asyncio
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from .builtins import iter as aiter, maybe_await
from .types import AnyIterable, AsyncIterator, MaybeAwaitable, T


async def as_completed(
    aws: Iterable[Awaitable[T]],
    *,
    timeout: Optional[float] = None,
) -> AsyncIterator[T]:
    """
    Run awaitables in `aws` concurrently, and yield results as they complete.

    Unlike `asyncio.as_completed`, this yields actual results, and does not require
    awaiting each item in the iterable.

    Cancels all remaining awaitables if a timeout is given and the timeout threshold
    is reached.

    Example::

        async for value in as_completed(futures):
            ...  # use value immediately

    """
    done: Set[Awaitable[T]] = set()
    pending: Set[Awaitable[T]] = {asyncio.ensure_future(a) for a in aws}
    remaining: Optional[float] = None

    if timeout and timeout > 0:
        threshold = time.time() + timeout
    else:
        timeout = None

    while pending:
        if timeout:
            remaining = threshold - time.time()
            if remaining <= 0:
                for fut in pending:
                    if isinstance(fut, asyncio.Future):
                        fut.cancel()
                    else:  # pragma: no cover
                        pass
                raise asyncio.TimeoutError()

        # asyncio.Future inherits from typing.Awaitable
        # asyncio.wait takes Iterable[Union[Future, Generator, Awaitable]], but
        # returns Tuple[Set[Future], Set[Future]. Because mypy doesn't like assigning
        # these values to existing Set[Awaitable] or even Set[Union[Awaitable, Future]],
        # we need to first cast the results to something that we can actually use
        # asyncio.Future: https://github.com/python/typeshed/blob/72ff7b94e534c610ddf8939bacbc55343e9465d2/stdlib/3/asyncio/futures.pyi#L30  # noqa: E501
        # asyncio.wait(): https://github.com/python/typeshed/blob/72ff7b94e534c610ddf8939bacbc55343e9465d2/stdlib/3/asyncio/tasks.pyi#L89  # noqa: E501
        done, pending = cast(
            Tuple[Set[Awaitable[T]], Set[Awaitable[T]]],
            await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            ),
        )

        for item in done:
            yield await item


async def as_generated(
    iterables: Iterable[AsyncIterable[T]],
    *,
    return_exceptions: bool = False,
) -> AsyncIterable[T]:
    """
    Yield results from one or more async iterables, in the order they are produced.

    Like :func:`as_completed`, but for async iterators or generators instead of futures.
    Creates a separate task to drain each iterable, and a single queue for results.

    If ``return_exceptions`` is ``False``, then any exception will be raised, and
    pending iterables and tasks will be cancelled, and async generators will be closed.
    If ``return_exceptions`` is ``True``, any exceptions will be yielded as results,
    and execution will continue until all iterables have been fully consumed.

    Example::

        async def generator(x):
            for i in range(x):
                yield i

        gen1 = generator(10)
        gen2 = generator(12)

        async for value in as_generated([gen1, gen2]):
            ...  # intermixed values yielded from gen1 and gen2
    """

    exc_queue: asyncio.Queue[Exception] = asyncio.Queue()
    queue: asyncio.Queue[T] = asyncio.Queue()

    async def tailer(iter: AsyncIterable[T]) -> None:
        try:
            async for item in iter:
                await queue.put(item)
        except asyncio.CancelledError:
            if isinstance(iter, AsyncGenerator):  # pragma:nocover
                await iter.aclose()
            raise
        except Exception as e:
            await exc_queue.put(e)

    tasks = [asyncio.ensure_future(tailer(iter)) for iter in iterables]
    pending = set(tasks)

    try:
        while pending:
            try:
                exc = exc_queue.get_nowait()
                if return_exceptions:
                    yield exc  # type: ignore
                else:
                    raise exc
            except asyncio.QueueEmpty:
                pass

            try:
                value = queue.get_nowait()
                yield value
            except asyncio.QueueEmpty:
                for task in list(pending):
                    if task.done():
                        pending.remove(task)
                await asyncio.sleep(0.001)

    except (asyncio.CancelledError, GeneratorExit):
        pass

    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass


async def gather(
    *args: Awaitable[T],
    return_exceptions: bool = False,
    limit: int = -1,
) -> List[Any]:
    """
    Like asyncio.gather but with a limit on concurrency.

    Note that all results are buffered.

    If gather is cancelled all tasks that were internally created and still pending
    will be cancelled as well.

    Example::

        futures = [some_coro(i) for i in range(10)]

        results = await gather(*futures, limit=2)
    """

    # For detecting input duplicates and reconciling them at the end
    input_map: Dict[Awaitable[T], List[int]] = {}
    # This is keyed on what we'll get back from asyncio.wait
    pos: Dict[asyncio.Future[T], int] = {}
    ret: List[Any] = [None] * len(args)

    pending: Set[asyncio.Future[T]] = set()
    done: Set[asyncio.Future[T]] = set()

    next_arg = 0

    while True:
        while next_arg < len(args) and (limit == -1 or len(pending) < limit):
            # We have to defer the creation of the Task as long as possible
            # because once we do, it starts executing, regardless of what we
            # have in the pending set.
            if args[next_arg] in input_map:
                input_map[args[next_arg]].append(next_arg)
            else:
                # We call ensure_future directly to ensure that we have a Task
                # because the return value of asyncio.wait will be an implicit
                # task otherwise, and we won't be able to know which input it
                # corresponds to.
                task: asyncio.Future[T] = asyncio.ensure_future(args[next_arg])
                pending.add(task)
                pos[task] = next_arg
                input_map[args[next_arg]] = [next_arg]
            next_arg += 1

        # pending might be empty if the last items of args were dupes;
        # asyncio.wait([]) will raise an exception.
        if pending:
            try:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for x in done:
                    if return_exceptions and x.exception():
                        ret[pos[x]] = x.exception()
                    else:
                        ret[pos[x]] = x.result()
            except asyncio.CancelledError:
                # Since we created these tasks we should cancel them
                for x in pending:
                    x.cancel()
                # we insure that all tasks are cancelled before we raise
                await asyncio.gather(*pending, return_exceptions=True)
                raise

        if not pending and next_arg == len(args):
            break

    for lst in input_map.values():
        for i in range(1, len(lst)):
            ret[lst[i]] = ret[lst[0]]

    return ret


async def gather_iter(
    itr: AnyIterable[MaybeAwaitable[T]],
    return_exceptions: bool = False,
    limit: int = -1,
) -> List[T]:
    """
    Wrapper around gather to handle gathering an iterable instead of ``*args``.

    Note that the iterable values don't have to be awaitable.
    """
    return await gather(
        *[maybe_await(i) async for i in aiter(itr)],
        return_exceptions=return_exceptions,
        limit=limit,
    )
