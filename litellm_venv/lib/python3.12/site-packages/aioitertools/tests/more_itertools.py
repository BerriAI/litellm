# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

from typing import AsyncIterable
from unittest import TestCase

import aioitertools.more_itertools as mit
from .helpers import async_test


async def _gen() -> AsyncIterable[int]:
    for i in range(5):
        yield i


async def _empty() -> AsyncIterable[int]:
    return
    yield 0


class MoreItertoolsTest(TestCase):
    @async_test
    async def test_take(self) -> None:
        self.assertEqual(await mit.take(2, _gen()), [0, 1])
        self.assertEqual(await mit.take(2, range(5)), [0, 1])

    @async_test
    async def test_take_zero(self) -> None:
        self.assertEqual(await mit.take(0, _gen()), [])

    @async_test
    async def test_take_negative(self) -> None:
        with self.assertRaises(ValueError):
            await mit.take(-1, _gen())

    @async_test
    async def test_take_more_than_iterable(self) -> None:
        self.assertEqual(await mit.take(10, _gen()), list(range(5)))

    @async_test
    async def test_take_empty(self) -> None:
        it = _gen()
        self.assertEqual(len(await mit.take(5, it)), 5)
        self.assertEqual(await mit.take(1, it), [])
        self.assertEqual(await mit.take(1, _empty()), [])

    @async_test
    async def test_chunked(self) -> None:
        self.assertEqual(
            [chunk async for chunk in mit.chunked(_gen(), 2)], [[0, 1], [2, 3], [4]]
        )
        self.assertEqual(
            [chunk async for chunk in mit.chunked(range(5), 2)], [[0, 1], [2, 3], [4]]
        )

    @async_test
    async def test_chunked_empty(self) -> None:
        self.assertEqual([], [chunk async for chunk in mit.chunked(_empty(), 2)])

    @async_test
    async def test_before_and_after_split(self) -> None:
        it = _gen()
        before, after = await mit.before_and_after(lambda i: i <= 2, it)
        self.assertEqual([elm async for elm in before], [0, 1, 2])
        self.assertEqual([elm async for elm in after], [3, 4])

    @async_test
    async def test_before_and_after_before_only(self) -> None:
        it = _gen()
        before, after = await mit.before_and_after(lambda i: True, it)
        self.assertEqual([elm async for elm in before], [0, 1, 2, 3, 4])
        self.assertEqual([elm async for elm in after], [])

    @async_test
    async def test_before_and_after_after_only(self) -> None:
        it = _gen()
        before, after = await mit.before_and_after(lambda i: False, it)
        self.assertEqual([elm async for elm in before], [])
        self.assertEqual([elm async for elm in after], [0, 1, 2, 3, 4])

    @async_test
    async def test_before_and_after_async_predicate(self) -> None:
        async def predicate(elm: int) -> bool:
            return elm <= 2

        it = _gen()
        before, after = await mit.before_and_after(predicate, it)
        self.assertEqual([elm async for elm in before], [0, 1, 2])
        self.assertEqual([elm async for elm in after], [3, 4])

    @async_test
    async def test_before_and_after_empty(self) -> None:
        it = _empty()
        before, after = await mit.before_and_after(lambda i: True, it)
        self.assertEqual([elm async for elm in before], [])
        self.assertEqual([elm async for elm in after], [])
