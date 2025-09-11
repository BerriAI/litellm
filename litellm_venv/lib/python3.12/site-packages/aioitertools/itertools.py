# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

"""
Async-compatible version of itertools standard library functions.

These functions build on top of the async builtins components,
enabling use of both standard iterables and async iterables, without
needing to use if/else clauses or awkward logic.  Standard iterables
get wrapped in async generators, and all functions are designed for
use with `await`, `async for`, etc.

See https://docs.python.org/3/library/itertools.html for reference.
"""

import asyncio
import builtins
import itertools
import operator
from typing import Any, AsyncIterator, List, Optional, overload, Tuple

from .builtins import enumerate, iter, list, next, tuple, zip
from .helpers import maybe_await
from .types import (
    Accumulator,
    AnyFunction,
    AnyIterable,
    AnyIterableIterable,
    AnyStop,
    KeyFunction,
    N,
    Predicate,
    R,
    T,
)


async def accumulate(
    itr: AnyIterable[T], func: Accumulator[T] = operator.add
) -> AsyncIterator[T]:
    """
    Yield the running accumulation of an iterable and operator.

    Accepts both a standard function or a coroutine for accumulation.

    Example::

        data = [1, 2, 3, 4]

        async def mul(a, b):
            return a * b

        async for total in accumulate(data, func=mul):
            ...  # 1, 2, 6, 24

    """
    itr = iter(itr)
    try:
        total: T = await next(itr)
    except AnyStop:
        return

    yield total
    async for item in itr:
        total = await maybe_await(func(total, item))
        yield total


async def batched(
    iterable: AnyIterable[T],
    n: int,
    *,
    strict: bool = False,
) -> AsyncIterator[Tuple[T, ...]]:
    """
    Yield batches of values from the given iterable. The final batch may be shorter.

    Example::

        async for batch in batched(range(15), 5):
            ...  # (0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14)

    """
    if n < 1:
        raise ValueError("n must be at least one")
    aiterator = iter(iterable)
    while batch := await tuple(islice(aiterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched: incomplete batch")
        yield batch


class Chain:
    def __call__(self, *itrs: AnyIterable[T]) -> AsyncIterator[T]:
        """
        Yield values from one or more iterables in series.

        Consumes the first iterable lazily, in entirety, then the second, and so on.

        Example::

            async for value in chain([1, 2, 3], [7, 8, 9]):
                ...  # 1, 2, 3, 7, 8, 9

        """
        return self.from_iterable(itrs)

    async def from_iterable(self, itrs: AnyIterableIterable[T]) -> AsyncIterator[T]:
        """
        Like chain, but takes an iterable of iterables.

        Alias for chain(*itrs)
        """
        async for itr in iter(itrs):
            async for item in iter(itr):
                yield item


chain = Chain()


async def combinations(itr: AnyIterable[T], r: int) -> AsyncIterator[Tuple[T, ...]]:
    """
    Yield r length subsequences from the given iterable.

    Simple wrapper around itertools.combinations for asyncio.
    This will consume the entire iterable before yielding values.

    Example::

        async for value in combinations(range(4), 3):
            ...  # (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)

    """
    pool: List[T] = await list(itr)
    for value in itertools.combinations(pool, r):
        yield value


async def combinations_with_replacement(
    itr: AnyIterable[T], r: int
) -> AsyncIterator[Tuple[T, ...]]:
    """
    Yield r length subsequences from the given iterable with replacement.

    Simple wrapper around itertools.combinations_with_replacement.
    This will consume the entire iterable before yielding values.

    Example::

        async for value in combinations_with_replacement("ABC", 2):
            ...  # ("A", "A"), ("A", "B"), ("A", "C"), ("B", "B"), ...

    """
    pool: List[T] = await list(itr)
    for value in itertools.combinations_with_replacement(pool, r):
        yield value


async def compress(
    itr: AnyIterable[T], selectors: AnyIterable[Any]
) -> AsyncIterator[T]:
    """
    Yield elements only when the corresponding selector evaluates to True.

    Stops when either the iterable or the selectors have been exhausted.

    Example::

        async for value in compress(range(5), [1, 0, 0, 1, 1]):
            ...  # 0, 3, 4
    """
    async for value, selector in zip(itr, selectors):
        if selector:
            yield value


async def count(start: N = 0, step: N = 1) -> AsyncIterator[N]:
    """
    Yield an infinite series, starting at the given value and increasing by step.

    Example::

        async for value in counter(10, -1):
            ...  # 10, 9, 8, 7, ...

    """

    value = start
    while True:
        yield value
        value += step


async def cycle(itr: AnyIterable[T]) -> AsyncIterator[T]:
    """
    Yield a repeating series from the given iterable.

    Lazily consumes the iterable when the next value is needed, and caching
    the values in memory for future iterations of the series.

    Example::

        async for value in cycle([1, 2]):
            ...  # 1, 2, 1, 2, 1, 2, ...

    """
    items = []
    async for item in iter(itr):
        yield item
        items.append(item)

    while True:
        for item in items:
            yield item


async def dropwhile(
    predicate: Predicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    """
    Drops all items until the predicate evaluates False; yields all items afterwards.

    Accepts both standard functions and coroutines for the predicate.

    Example::

        def pred(x):
            return x < 4

        async for item in dropwhile(pred, range(6)):
            ...  # 4, 5, 6

    """
    itr = iter(iterable)
    async for item in itr:
        if not await maybe_await(predicate(item)):
            yield item
            break
    async for item in itr:
        yield item


async def filterfalse(
    predicate: Predicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    """
    Yield items from the iterable only when the predicate evaluates to False.

    Accepts both standard functions and coroutines for the predicate.

    Example::

        def pred(x):
            return x < 4

        async for item in filterfalse(pred, range(6)):
            ...  # 4, 5

    """
    async for item in iter(iterable):
        if not await maybe_await(predicate(item)):
            yield item


@overload
def groupby(itr: AnyIterable[T]) -> AsyncIterator[Tuple[T, List[T]]]:  # pragma: nocover
    pass


@overload
def groupby(
    itr: AnyIterable[T], key: KeyFunction[T, R]
) -> AsyncIterator[Tuple[R, List[T]]]:  # pragma: nocover
    pass


async def groupby(
    itr: AnyIterable[T], key: Optional[KeyFunction[T, R]] = None
) -> AsyncIterator[Tuple[Any, List[T]]]:
    """
    Yield consecutive keys and groupings from the given iterable.

    Items will be grouped based on the key function, which defaults to
    the identity of each item.  Accepts both standard functions and
    coroutines for the key function.  Suggest sorting by the key
    function before using groupby.

    Example::

        data = ["A", "a", "b", "c", "C", "c"]

        async for key, group in groupby(data, key=str.lower):
            key  # "a", "b", "c"
            group  # ["A", "a"], ["b"], ["c", "C", "c"]

    """
    if key is None:
        key = lambda x: x  # noqa: E731

    grouping: List[T] = []

    it = iter(itr)
    try:
        item = await next(it)
    except StopAsyncIteration:
        return
    grouping = [item]

    j = await maybe_await(key(item))
    async for item in it:
        k = await maybe_await(key(item))
        if k != j:
            yield j, grouping
            grouping = [item]
        else:
            grouping.append(item)
        j = k

    yield j, grouping


@overload
def islice(
    itr: AnyIterable[T], __stop: Optional[int]
) -> AsyncIterator[T]:  # pragma: nocover
    pass


@overload
def islice(
    itr: AnyIterable[T], __start: int, __stop: Optional[int], __step: int = 1
) -> AsyncIterator[T]:  # pragma: nocover
    pass


async def islice(itr: AnyIterable[T], *args: Optional[int]) -> AsyncIterator[T]:
    """
    Yield selected items from the given iterable.

    islice(iterable, stop)
    islice(iterable, start, stop[, step])

    Starting from the start index (or zero), stopping at the stop
    index (or until exhausted), skipping items if step > 0.

    Example::

        data = range(10)

        async for item in islice(data, 5):
            ...  # 0, 1, 2, 3, 4

        async for item in islice(data, 2, 5):
            ...  # 2, 3, 4

        async for item in islice(data, 1, 7, 2):
            ...  # 1, 3, 5

    """
    start = 0
    step = 1
    if not args:
        raise ValueError("must pass stop index")
    if len(args) == 1:
        (stop,) = args
    elif len(args) == 2:
        start, stop = args  # type: ignore
    elif len(args) == 3:
        start, stop, step = args  # type: ignore
    else:
        raise ValueError("too many arguments given")
    assert start >= 0 and (stop is None or stop >= 0) and step >= 0
    step = max(1, step)

    if stop == 0:
        return

    async for index, item in enumerate(itr):
        if index >= start and (index - start) % step == 0:
            yield item
        if stop is not None and index + 1 >= stop:
            break


async def permutations(
    itr: AnyIterable[T], r: Optional[int] = None
) -> AsyncIterator[Tuple[T, ...]]:
    """
    Yield r length permutations of elements in the iterable.

    Simple wrapper around itertools.combinations for asyncio.
    This will consume the entire iterable before yielding values.

    Example::

        async for value in permutations(range(3)):
            ...  # (0, 1, 2), (0, 2, 1), (1, 0, 2), ...

    """
    pool: List[T] = await list(itr)
    for value in itertools.permutations(pool, r):
        yield value


async def product(
    *itrs: AnyIterable[T], repeat: int = 1
) -> AsyncIterator[Tuple[T, ...]]:
    """
    Yield cartesian products of all iterables.

    Simple wrapper around itertools.combinations for asyncio.
    This will consume all iterables before yielding any values.

    Example::

        async for value in product("abc", "xy"):
            ...  # ("a", "x"), ("a", "y"), ("b", "x"), ...

        async for value in product(range(3), repeat=3):
            ...  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ...

    """
    pools = await asyncio.gather(*[list(itr) for itr in itrs])
    for value in itertools.product(*pools, repeat=repeat):
        yield value


async def repeat(elem: T, n: int = -1) -> AsyncIterator[T]:
    """
    Yield the given value repeatedly, forever or up to n times.

    Example::

        async for value in repeat(7):
            ...  # 7, 7, 7, 7, 7, 7, ...

    """
    while True:
        if n == 0:
            break
        yield elem
        n -= 1


async def starmap(
    fn: AnyFunction[R], iterable: AnyIterableIterable[Any]
) -> AsyncIterator[R]:
    """
    Yield values from a function using an iterable of iterables for arguments.

    Each iterable contained within will be unpacked and consumed before
    executing the function or coroutine.

    Example::

        data = [(1, 1), (1, 1, 1), (2, 2)]

        async for value in starmap(operator.add, data):
            ...  # 2, 3, 4

    """
    async for itr in iter(iterable):
        args = await list(itr)
        yield await maybe_await(fn(*args))


async def takewhile(
    predicate: Predicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    """
    Yield values from the iterable until the predicate evaluates False.

    Accepts both standard functions and coroutines for the predicate.

    Example::

        def pred(x):
            return x < 4

        async for value in takewhile(pred, range(8)):
            ...  # 0, 1, 2, 3

    """
    async for item in iter(iterable):
        if await maybe_await(predicate(item)):
            yield item
        else:
            break


def tee(itr: AnyIterable[T], n: int = 2) -> Tuple[AsyncIterator[T], ...]:
    """
    Return n iterators that each yield items from the given iterable.

    The first iterator lazily fetches from the original iterable, and then
    queues the values for the other iterators to yield when needed.

    Caveat: all iterators are dependent on the first iterator – if it is
    consumed more slowly than the rest, the other consumers will be blocked
    until the first iterator continues forward.  Similarly, if the first
    iterator is consumed more quickly than the rest, more memory will be
    used in keeping values in the queues until the other iterators finish
    consuming them.

    Example::

        it1, it2 = tee(range(5), n=2)

        async for value in it1:
            ...  # 0, 1, 2, 3, 4

        async for value in it2:
            ...  # 0, 1, 2, 3, 4

    """
    assert n > 0
    sentinel = object()
    queues: List[asyncio.Queue] = [asyncio.Queue() for k in range(n)]

    async def gen(k: int, q: asyncio.Queue) -> AsyncIterator[T]:
        if k == 0:
            try:
                async for value in iter(itr):
                    await asyncio.gather(
                        *[queue.put((None, value)) for queue in queues[1:]]
                    )
                    yield value
            except Exception as e:
                await asyncio.gather(*[queue.put((e, None)) for queue in queues[1:]])
                raise

            await asyncio.gather(*[queue.put((None, sentinel)) for queue in queues[1:]])

        else:
            while True:
                error, value = await q.get()
                if error is not None:
                    raise error
                if value is sentinel:
                    break
                yield value

    return builtins.tuple(gen(k, q) for k, q in builtins.enumerate(queues))


async def zip_longest(
    *itrs: AnyIterable[Any], fillvalue: Any = None
) -> AsyncIterator[Tuple[Any, ...]]:
    """
    Yield a tuple of items from mixed iterables until all are consumed.

    If shorter iterables are exhausted, the default value will be used
    until all iterables are exhausted.

    Example::

        a = range(3)
        b = range(5)

        async for a, b in zip_longest(a, b, fillvalue=-1):
            a  # 0, 1, 2, -1, -1
            b  # 0, 1, 2,  3,  4

    """
    its: List[AsyncIterator[Any]] = [iter(itr) for itr in itrs]
    itr_count = len(its)
    finished = 0

    while True:
        values = await asyncio.gather(
            *[it.__anext__() for it in its], return_exceptions=True
        )
        for idx, value in builtins.enumerate(values):
            if isinstance(value, AnyStop):
                finished += 1
                values[idx] = fillvalue
                its[idx] = repeat(fillvalue)
            elif isinstance(value, BaseException):
                raise value
        if finished >= itr_count:
            break
        yield builtins.tuple(values)
