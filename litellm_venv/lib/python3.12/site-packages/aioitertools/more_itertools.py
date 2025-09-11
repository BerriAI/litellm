# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

import asyncio
from typing import AsyncIterable, List, Tuple, TypeVar

from aioitertools.helpers import maybe_await

from .builtins import iter
from .itertools import islice
from .types import AnyIterable, Predicate


T = TypeVar("T")


async def take(n: int, iterable: AnyIterable[T]) -> List[T]:
    """
    Return the first n items of iterable as a list.

    If there are too few items in iterable, all of them are returned.
    n needs to be at least 0. If it is 0, an empty list is returned.

    Example::

        first_two = await take(2, [1, 2, 3, 4, 5])

    """
    if n < 0:
        raise ValueError("take's first parameter can't be negative")
    return [item async for item in islice(iterable, n)]


async def chunked(iterable: AnyIterable[T], n: int) -> AsyncIterable[List[T]]:
    """
    Break iterable into chunks of length n.

    The last chunk will be shorter if the total number of items is not
    divisible by n.

    Example::

        async for chunk in chunked([1, 2, 3, 4, 5], n=2):
            ...  # first iteration: chunk == [1, 2]; last one: chunk == [5]
    """
    it = iter(iterable)
    chunk = await take(n, it)
    while chunk != []:
        yield chunk
        chunk = await take(n, it)


async def before_and_after(
    predicate: Predicate[T], iterable: AnyIterable[T]
) -> Tuple[AsyncIterable[T], AsyncIterable[T]]:
    """
    A variant of :func:`aioitertools.takewhile` that allows complete access to the
    remainder of the iterator.

         >>> it = iter('ABCdEfGhI')
         >>> all_upper, remainder = await before_and_after(str.isupper, it)
         >>> ''.join([char async for char in all_upper])
         'ABC'
         >>> ''.join([char async for char in remainder])
         'dEfGhI'

    Note that the first iterator must be fully consumed before the second
    iterator can generate valid results.
    """

    it = iter(iterable)

    transition = asyncio.get_event_loop().create_future()

    async def true_iterator():
        async for elem in it:
            if await maybe_await(predicate(elem)):
                yield elem
            else:
                transition.set_result(elem)
                return

        transition.set_exception(StopAsyncIteration)

    async def remainder_iterator():
        try:
            yield (await transition)
        except StopAsyncIteration:
            return

        async for elm in it:
            yield elm

    return true_iterator(), remainder_iterator()
