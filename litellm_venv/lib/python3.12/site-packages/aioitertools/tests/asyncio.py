# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

import asyncio
from unittest import TestCase

import aioitertools as ait
import aioitertools.asyncio as aio
from .helpers import async_test

slist = ["A", "B", "C"]
srange = range(3)


class AsyncioTest(TestCase):
    def test_import(self):
        self.assertEqual(ait.asyncio, aio)

    @async_test
    async def test_as_completed(self):
        async def sleepy(number, duration):
            await asyncio.sleep(duration)
            return number

        pairs = [(1, 0.3), (2, 0.1), (3, 0.5)]
        expected = [2, 1, 3]

        futures = [sleepy(*pair) for pair in pairs]
        results = await ait.list(aio.as_completed(futures))
        self.assertEqual(results, expected)

        futures = [sleepy(*pair) for pair in pairs]
        results = []
        async for value in aio.as_completed(futures):
            results.append(value)
        self.assertEqual(results, expected)

    @async_test
    async def test_as_completed_timeout(self):
        calls = [(1.0,), (0.1,)]

        futures = [asyncio.sleep(*args) for args in calls]
        with self.assertRaises(asyncio.TimeoutError):
            await ait.list(aio.as_completed(futures, timeout=0.5))

        futures = [asyncio.sleep(*args) for args in calls]
        results = 0
        with self.assertRaises(asyncio.TimeoutError):
            async for _ in aio.as_completed(futures, timeout=0.5):
                results += 1
        self.assertEqual(results, 1)

    @async_test
    async def test_as_generated(self):
        async def gen():
            for i in range(10):
                yield i
                await asyncio.sleep(0)

        gens = [gen(), gen(), gen()]
        expected = list(range(10)) * 3
        results = []
        async for value in aio.as_generated(gens):
            results.append(value)
        self.assertEqual(30, len(results))
        self.assertListEqual(sorted(expected), sorted(results))

    @async_test
    async def test_as_generated_exception(self):
        async def gen1():
            for i in range(3):
                yield i
                await asyncio.sleep(0)
            raise Exception("fake")

        async def gen2():
            for i in range(10):
                yield i
                await asyncio.sleep(0)

        gens = [gen1(), gen2()]
        results = []
        with self.assertRaisesRegex(Exception, "fake"):
            async for value in aio.as_generated(gens):
                results.append(value)
        self.assertNotIn(10, results)

    @async_test
    async def test_as_generated_return_exception(self):
        async def gen1():
            for i in range(3):
                yield i
                await asyncio.sleep(0)
            raise Exception("fake")

        async def gen2():
            for i in range(10):
                yield i
                await asyncio.sleep(0)

        gens = [gen1(), gen2()]
        expected = list(range(3)) + list(range(10))
        errors = []
        results = []
        async for value in aio.as_generated(gens, return_exceptions=True):
            if isinstance(value, Exception):
                errors.append(value)
            else:
                results.append(value)
        self.assertListEqual(sorted(expected), sorted(results))
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], Exception)

    @async_test
    async def test_as_generated_task_cancelled(self):
        async def gen(max: int = 10):
            for i in range(5):
                if i > max:
                    raise asyncio.CancelledError
                yield i
                await asyncio.sleep(0)

        gens = [gen(2), gen()]
        expected = list(range(3)) + list(range(5))
        results = []
        async for value in aio.as_generated(gens):
            results.append(value)
        self.assertListEqual(sorted(expected), sorted(results))

    @async_test
    async def test_as_generated_cancelled(self):
        async def gen():
            for i in range(5):
                yield i
                await asyncio.sleep(0.1)

        expected = [0, 0, 1, 1]
        results = []

        async def foo():
            gens = [gen(), gen()]
            async for value in aio.as_generated(gens):
                results.append(value)
            return results

        task = asyncio.ensure_future(foo())
        await asyncio.sleep(0.15)
        task.cancel()
        await task

        self.assertListEqual(sorted(expected), sorted(results))

    @async_test
    async def test_gather_input_types(self):
        async def fn(arg):
            await asyncio.sleep(0.001)
            return arg

        fns = [fn(1), asyncio.ensure_future(fn(2))]
        if hasattr(asyncio, "create_task"):
            # 3.7 only
            fns.append(asyncio.create_task(fn(3)))
        else:
            fns.append(fn(3))

        result = await aio.gather(*fns)
        self.assertEqual([1, 2, 3], result)

    @async_test
    async def test_gather_limited(self):
        max_counter = 0
        counter = 0

        async def fn(arg):
            nonlocal counter, max_counter
            counter += 1
            max_counter = max(max_counter, counter)
            await asyncio.sleep(0.001)
            counter -= 1
            return arg

        # Limit of 2
        result = await aio.gather(*[fn(i) for i in range(10)], limit=2)
        self.assertEqual(2, max_counter)
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], result)

        # No limit
        result = await aio.gather(*[fn(i) for i in range(10)])
        self.assertEqual(
            10, max_counter
        )  # TODO: on a loaded machine this might be less?
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], result)

    @async_test
    async def test_gather_limited_dupes(self):
        async def fn(arg):
            await asyncio.sleep(0.001)
            return arg

        f = fn(1)
        g = fn(2)
        result = await aio.gather(f, f, f, g, f, g, limit=2)
        self.assertEqual([1, 1, 1, 2, 1, 2], result)

        f = fn(1)
        g = fn(2)
        result = await aio.gather(f, f, f, g, f, g)
        self.assertEqual([1, 1, 1, 2, 1, 2], result)

    @async_test
    async def test_gather_with_exceptions(self):
        class MyException(Exception):
            pass

        async def fn(arg, fail=False):
            await asyncio.sleep(arg)
            if fail:
                raise MyException(arg)
            return arg

        with self.assertRaises(MyException):
            await aio.gather(fn(0.002, fail=True), fn(0.001))

        result = await aio.gather(
            fn(0.002, fail=True), fn(0.001), return_exceptions=True
        )
        self.assertEqual(result[1], 0.001)
        self.assertIsInstance(result[0], MyException)

    @async_test
    async def test_gather_cancel(self):
        cancelled = False
        started = False

        async def _fn():
            nonlocal started, cancelled
            try:
                started = True
                await asyncio.sleep(10)  # might as well be forever
            except asyncio.CancelledError:
                nonlocal cancelled
                cancelled = True
                raise

        async def _gather():
            await aio.gather(_fn())

        if hasattr(asyncio, "create_task"):
            # 3.7+ only
            task = asyncio.create_task(_gather())
        else:
            task = asyncio.ensure_future(_gather())
        # to insure the gather actually runs
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(started)
        self.assertTrue(cancelled)
