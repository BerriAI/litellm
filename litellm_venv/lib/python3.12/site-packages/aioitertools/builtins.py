# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

"""
Async-compatible versions of builtin functions for iterables.

These functions intentionally shadow their builtins counterparts,
enabling use with both standard iterables and async iterables, without
needing to use if/else clauses or awkward logic.  Standard iterables
get wrapped in async generators, and all functions are designed for
use with `await`, `async for`, etc.
"""

import asyncio
import builtins
from enum import Enum
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    cast,
    Iterable,
    List,
    Optional,
    overload,
    Set,
    Tuple,
    Union,
)

from . import asyncio as ait_asyncio
from .helpers import maybe_await, Orderable
from .types import (
    AnyIterable,
    AnyIterator,
    AnyStop,
    MaybeAwaitable,
    R,
    T,
    T1,
    T2,
    T3,
    T4,
    T5,
)


class Sentinel(Enum):
    """
    :meta private:
    """

    MISSING = object()


async def all(itr: AnyIterable[MaybeAwaitable[Any]]) -> bool:
    """
    Return True if all values are truthy in a mixed iterable, else False.
    The iterable will be fully consumed and any awaitables will
    automatically be awaited.

    Example::

        if await all(it):
            ...

    """
    return builtins.all(await ait_asyncio.gather_iter(itr))


async def any(itr: AnyIterable[MaybeAwaitable[Any]]) -> bool:
    """
    Return True if any value is truthy in a mixed iterable, else False.
    The iterable will be fully consumed and any awaitables will
    automatically be awaited.

    Example::

        if await any(it):
            ...

    """
    return builtins.any(await ait_asyncio.gather_iter(itr))


def iter(itr: AnyIterable[T]) -> AsyncIterator[T]:
    """
    Get an async iterator from any mixed iterable.

    Async iterators will be returned directly.
    Async iterables will return an async iterator.
    Standard iterables will be wrapped in an async generator yielding
    each item in the iterable in the same order.

    Examples::

        async for value in iter(range(10)):
            ...

    """
    if isinstance(itr, AsyncIterator):
        return itr

    if isinstance(itr, AsyncIterable):
        return itr.__aiter__()

    async def gen() -> AsyncIterator[T]:
        for item in cast(Iterable[T], itr):
            yield item

    return gen()


@overload
async def next(itr: AnyIterator[T]) -> T:  # pragma: no cover
    ...


@overload
async def next(itr: AnyIterator[T1], default: T2) -> Union[T1, T2]:  # pragma: no cover
    ...


async def next(
    itr: AnyIterator[T1], default: Union[T2, Sentinel] = Sentinel.MISSING
) -> Union[T1, T2]:
    """
    Return the next item of any mixed iterator.

    Calls builtins.next() on standard iterators, and awaits itr.__anext__()
    on async iterators.

    Example::

        value = await next(it)

    """
    try:
        if isinstance(itr, AsyncIterator):
            return await itr.__anext__()

        try:
            return builtins.next(itr)
        except StopIteration:
            raise StopAsyncIteration
    except StopAsyncIteration:
        if default is Sentinel.MISSING:
            raise
        return default


async def list(itr: AnyIterable[T]) -> List[T]:
    """
    Consume a mixed iterable and return a list of items in order.

    Example::

        await list(range(5))
        -> [0, 1, 2, 3, 4]

    """
    return [item async for item in iter(itr)]


async def tuple(itr: AnyIterable[T]) -> Tuple[T, ...]:
    """
    Consume a mixed iterable and return a tuple of items in order.

    Example::

        await tuple(range(5))
        -> (0, 1, 2, 3, 4)

    """
    # Suboptimal, but tuple can't be created from AsyncIterable directly.
    return builtins.tuple(await list(itr))


async def set(itr: AnyIterable[T]) -> Set[T]:
    """
    Consume a mixed iterable and return a set of items.

    Example::

        await set([0, 1, 2, 3, 0, 1, 2, 3])
        -> {0, 1, 2, 3}

    """
    return {item async for item in iter(itr)}


async def enumerate(
    itr: AnyIterable[T], start: int = 0
) -> AsyncIterator[Tuple[int, T]]:
    """
    Consume a mixed iterable and yield the current index and item.

    Example::

        async for index, value in enumerate(...):
            ...

    """
    index = start
    async for item in iter(itr):
        yield index, item
        index += 1


async def map(fn: Callable[[T], R], itr: AnyIterable[T]) -> AsyncIterator[R]:
    """
    Modify item of a mixed iterable using the given function or coroutine.

    Example::

        async for response in map(func, data):
            ...

    """
    # todo: queue items eagerly
    async for item in iter(itr):
        yield await maybe_await(fn(item))


@overload
async def max(
    itr: AnyIterable[Orderable], *, key: Optional[Callable] = None
) -> Orderable:  # pragma: no cover
    pass


@overload
async def max(
    itr: AnyIterable[Orderable], *, default: T, key: Optional[Callable] = None
) -> Union[Orderable, T]:  # pragma: no cover
    pass


async def max(itr: AnyIterable[Orderable], **kwargs: Any) -> Any:
    """
    Return the largest item in an iterable or the largest of two or more arguments.

    Example::

        await max(range(5))
        -> 4

    """
    for k in kwargs:
        if k not in ("key", "default"):
            raise ValueError(f"kwarg {k} not supported")

    value: Orderable
    vkey: Any

    keyfunc = kwargs.get("key", None)
    it = iter(itr)

    try:
        value = await next(it)
        if keyfunc:
            vkey = keyfunc(value)

    except StopAsyncIteration:
        if "default" in kwargs:
            return kwargs["default"]
        raise ValueError("iterable is empty and no default value given")

    if keyfunc:
        async for item in it:
            ikey = keyfunc(item)
            if ikey > vkey:
                value = item
                vkey = ikey

    else:
        async for item in it:
            if item > value:
                value = item

    return value


@overload
async def min(
    itr: AnyIterable[Orderable], *, key: Optional[Callable] = None
) -> Orderable:  # pragma: no cover
    pass


@overload
async def min(
    itr: AnyIterable[Orderable], *, default: T, key: Optional[Callable] = None
) -> Union[Orderable, T]:  # pragma: no cover
    pass


async def min(itr: AnyIterable[Orderable], **kwargs: Any) -> Any:
    """
    Return the smallest item in an iterable or the smallest of two or more arguments.

    Example::

        await min(range(5))
        -> 0

    """
    for k in kwargs:
        if k not in ("key", "default"):
            raise ValueError(f"kwarg {k} not supported")

    value: Orderable
    vkey: Any

    keyfunc = kwargs.get("key", None)
    it = iter(itr)

    try:
        value = await next(it)
        if keyfunc:
            vkey = keyfunc(value)

    except StopAsyncIteration:
        if "default" in kwargs:
            return kwargs["default"]
        raise ValueError("iterable is empty and no default value given")

    if keyfunc:
        async for item in it:
            ikey = keyfunc(item)
            if ikey < vkey:
                value = item
                vkey = ikey

    else:
        async for item in it:
            if item < value:
                value = item

    return value


async def sum(itr: AnyIterable[T], start: Optional[T] = None) -> T:
    """
    Compute the sum of a mixed iterable, adding each value with the start value.

    Example::

        await sum(generator())
        -> 1024

    """
    value: T
    if start is None:
        value = cast(T, 0)  # emulate stdlib but still type nicely for non-ints
    else:
        value = start

    async for item in iter(itr):
        value += item  # type: ignore  # mypy doesn't know T + T

    return value


@overload
def zip(__iter1: AnyIterable[T1]) -> AsyncIterator[Tuple[T1]]:  # pragma: no cover
    pass


@overload
def zip(
    __iter1: AnyIterable[T1], __iter2: AnyIterable[T2]
) -> AsyncIterator[Tuple[T1, T2]]:  # pragma: no cover
    pass


@overload
def zip(
    __iter1: AnyIterable[T1], __iter2: AnyIterable[T2], __iter3: AnyIterable[T3]
) -> AsyncIterator[Tuple[T1, T2, T3]]:  # pragma: no cover
    pass


@overload
def zip(
    __iter1: AnyIterable[T1],
    __iter2: AnyIterable[T2],
    __iter3: AnyIterable[T3],
    __iter4: AnyIterable[T4],
) -> AsyncIterator[Tuple[T1, T2, T3, T4]]:  # pragma: no cover
    pass


@overload
def zip(
    __iter1: AnyIterable[T1],
    __iter2: AnyIterable[T2],
    __iter3: AnyIterable[T3],
    __iter4: AnyIterable[T4],
    __iter5: AnyIterable[T5],
) -> AsyncIterator[Tuple[T1, T2, T3, T4, T5]]:  # pragma: no cover
    pass


@overload
def zip(
    __iter1: AnyIterable[Any],
    __iter2: AnyIterable[Any],
    __iter3: AnyIterable[Any],
    __iter4: AnyIterable[Any],
    __iter5: AnyIterable[Any],
    __iter6: AnyIterable[Any],
    *__iterables: AnyIterable[Any],
) -> AsyncIterator[Tuple[Any, ...]]:  # pragma: no cover
    pass


async def zip(*itrs: AnyIterable[Any]) -> AsyncIterator[Tuple[Any, ...]]:
    """
    Yield a tuple of items from mixed iterables until the shortest is consumed.

    Example::

        async for a, b, c in zip(i, j, k):
            ...

    """
    its: List[AsyncIterator[Any]] = [iter(itr) for itr in itrs]

    while True:
        values = await asyncio.gather(
            *[it.__anext__() for it in its], return_exceptions=True
        )
        if builtins.any(isinstance(v, AnyStop) for v in values):
            break
        yield builtins.tuple(values)
