# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

import asyncio
import operator
from unittest import TestCase

import aioitertools as ait
from .helpers import async_test

slist = ["A", "B", "C"]
srange = range(1, 4)


class ItertoolsTest(TestCase):
    @async_test
    async def test_accumulate_range_default(self):
        it = ait.accumulate(srange)
        for k in [1, 3, 6]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_accumulate_range_function(self):
        it = ait.accumulate(srange, func=operator.mul)
        for k in [1, 2, 6]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_accumulate_range_coroutine(self):
        async def mul(a, b):
            return a * b

        it = ait.accumulate(srange, func=mul)
        for k in [1, 2, 6]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_accumulate_gen_function(self):
        async def gen():
            yield 1
            yield 2
            yield 4

        it = ait.accumulate(gen(), func=operator.mul)
        for k in [1, 2, 8]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_accumulate_gen_coroutine(self):
        async def mul(a, b):
            return a * b

        async def gen():
            yield 1
            yield 2
            yield 4

        it = ait.accumulate(gen(), func=mul)
        for k in [1, 2, 8]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_accumulate_empty(self):
        values = []
        async for value in ait.accumulate([]):
            values.append(value)

        self.assertEqual(values, [])

    @async_test
    async def test_batched(self):
        test_matrix = [
            ([], 1, []),
            ([1, 2, 3], 1, [(1,), (2,), (3,)]),
            ([2, 3, 4], 2, [(2, 3), (4,)]),
            ([5, 6], 3, [(5, 6)]),
            (ait.iter([-2, -1, 0, 1, 2]), 2, [(-2, -1), (0, 1), (2,)]),
        ]
        for iterable, batch_size, answer in test_matrix:
            result = [batch async for batch in ait.batched(iterable, batch_size)]

            self.assertEqual(result, answer)

    @async_test
    async def test_batched_errors(self):
        with self.assertRaisesRegex(ValueError, "n must be at least one"):
            [batch async for batch in ait.batched([1], 0)]
        with self.assertRaisesRegex(ValueError, "incomplete batch"):
            [batch async for batch in ait.batched([1, 2, 3], 2, strict=True)]

    @async_test
    async def test_chain_lists(self):
        it = ait.chain(slist, srange)
        for k in ["A", "B", "C", 1, 2, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_chain_list_gens(self):
        async def gen():
            for k in range(2, 9, 2):
                yield k

        it = ait.chain(slist, gen())
        for k in ["A", "B", "C", 2, 4, 6, 8]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_chain_from_iterable(self):
        async def gen():
            for k in range(2, 9, 2):
                yield k

        it = ait.chain.from_iterable([slist, gen()])
        for k in ["A", "B", "C", 2, 4, 6, 8]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_chain_from_iterable_parameter_expansion_gen(self):
        async def gen():
            for k in range(2, 9, 2):
                yield k

        async def parameters_gen():
            yield slist
            yield gen()

        it = ait.chain.from_iterable(parameters_gen())
        for k in ["A", "B", "C", 2, 4, 6, 8]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_combinations(self):
        it = ait.combinations(range(4), 3)
        for k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_combinations_with_replacement(self):
        it = ait.combinations_with_replacement(slist, 2)
        for k in [
            ("A", "A"),
            ("A", "B"),
            ("A", "C"),
            ("B", "B"),
            ("B", "C"),
            ("C", "C"),
        ]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_compress_list(self):
        data = range(10)
        selectors = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0]

        it = ait.compress(data, selectors)
        for k in [1, 2, 6, 8]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_compress_gen(self):
        data = "abcdefghijkl"
        selectors = ait.cycle([1, 0, 0])

        it = ait.compress(data, selectors)
        for k in ["a", "d", "g", "j"]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_count_bare(self):
        it = ait.count()
        for k in [0, 1, 2, 3]:
            self.assertEqual(await ait.next(it), k)

    @async_test
    async def test_count_start(self):
        it = ait.count(42)
        for k in [42, 43, 44, 45]:
            self.assertEqual(await ait.next(it), k)

    @async_test
    async def test_count_start_step(self):
        it = ait.count(42, 3)
        for k in [42, 45, 48, 51]:
            self.assertEqual(await ait.next(it), k)

    @async_test
    async def test_count_negative(self):
        it = ait.count(step=-2)
        for k in [0, -2, -4, -6]:
            self.assertEqual(await ait.next(it), k)

    @async_test
    async def test_cycle_list(self):
        it = ait.cycle(slist)
        for k in ["A", "B", "C", "A", "B", "C", "A", "B"]:
            self.assertEqual(await ait.next(it), k)

    @async_test
    async def test_cycle_gen(self):
        async def gen():
            yield 1
            yield 2
            yield 42

        it = ait.cycle(gen())
        for k in [1, 2, 42, 1, 2, 42, 1, 2]:
            self.assertEqual(await ait.next(it), k)

    @async_test
    async def test_dropwhile_empty(self):
        def pred(x):
            return x < 2

        result = await ait.list(ait.dropwhile(pred, []))
        self.assertEqual(result, [])

    @async_test
    async def test_dropwhile_function_list(self):
        def pred(x):
            return x < 2

        it = ait.dropwhile(pred, srange)
        for k in [2, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_dropwhile_function_gen(self):
        def pred(x):
            return x < 2

        async def gen():
            yield 1
            yield 2
            yield 42

        it = ait.dropwhile(pred, gen())
        for k in [2, 42]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_dropwhile_coroutine_list(self):
        async def pred(x):
            return x < 2

        it = ait.dropwhile(pred, srange)
        for k in [2, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_dropwhile_coroutine_gen(self):
        async def pred(x):
            return x < 2

        async def gen():
            yield 1
            yield 2
            yield 42

        it = ait.dropwhile(pred, gen())
        for k in [2, 42]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_filterfalse_function_list(self):
        def pred(x):
            return x % 2 == 0

        it = ait.filterfalse(pred, srange)
        for k in [1, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_filterfalse_coroutine_list(self):
        async def pred(x):
            return x % 2 == 0

        it = ait.filterfalse(pred, srange)
        for k in [1, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_groupby_list(self):
        data = "aaabba"

        it = ait.groupby(data)
        for k in [("a", ["a", "a", "a"]), ("b", ["b", "b"]), ("a", ["a"])]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_groupby_list_key(self):
        data = "aAabBA"

        it = ait.groupby(data, key=str.lower)
        for k in [("a", ["a", "A", "a"]), ("b", ["b", "B"]), ("a", ["A"])]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_groupby_gen(self):
        async def gen():
            for c in "aaabba":
                yield c

        it = ait.groupby(gen())
        for k in [("a", ["a", "a", "a"]), ("b", ["b", "b"]), ("a", ["a"])]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_groupby_gen_key(self):
        async def gen():
            for c in "aAabBA":
                yield c

        it = ait.groupby(gen(), key=str.lower)
        for k in [("a", ["a", "A", "a"]), ("b", ["b", "B"]), ("a", ["A"])]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_groupby_empty(self):
        async def gen():
            for _ in range(0):
                yield  # Force generator with no actual iteration

        async for _ in ait.groupby(gen()):
            self.fail("No iteration should have happened")

    @async_test
    async def test_islice_bad_range(self):
        with self.assertRaisesRegex(ValueError, "must pass stop index"):
            async for _ in ait.islice([1, 2]):
                pass

        with self.assertRaisesRegex(ValueError, "too many arguments"):
            async for _ in ait.islice([1, 2], 1, 2, 3, 4):
                pass

    @async_test
    async def test_islice_stop_zero(self):
        values = []
        async for value in ait.islice(range(5), 0):
            values.append(value)
        self.assertEqual(values, [])

    @async_test
    async def test_islice_range_stop(self):
        it = ait.islice(srange, 2)
        for k in [1, 2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_islice_range_start_step(self):
        it = ait.islice(srange, 0, None, 2)
        for k in [1, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_islice_range_start_stop(self):
        it = ait.islice(srange, 1, 3)
        for k in [2, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_islice_range_start_stop_step(self):
        it = ait.islice(srange, 1, 3, 2)
        for k in [2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_islice_gen_stop(self):
        async def gen():
            yield 1
            yield 2
            yield 3
            yield 4

        gen_it = gen()
        it = ait.islice(gen_it, 2)
        for k in [1, 2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)
        assert await ait.list(gen_it) == [3, 4]

    @async_test
    async def test_islice_gen_start_step(self):
        async def gen():
            yield 1
            yield 2
            yield 3
            yield 4

        it = ait.islice(gen(), 1, None, 2)
        for k in [2, 4]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_islice_gen_start_stop(self):
        async def gen():
            yield 1
            yield 2
            yield 3
            yield 4

        it = ait.islice(gen(), 1, 3)
        for k in [2, 3]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_islice_gen_start_stop_step(self):
        async def gen():
            yield 1
            yield 2
            yield 3
            yield 4

        gen_it = gen()
        it = ait.islice(gen_it, 1, 3, 2)
        for k in [2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)
        assert await ait.list(gen_it) == [4]

    @async_test
    async def test_permutations_list(self):
        it = ait.permutations(srange, r=2)
        for k in [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_permutations_gen(self):
        async def gen():
            yield 1
            yield 2
            yield 3

        it = ait.permutations(gen(), r=2)
        for k in [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_product_list(self):
        it = ait.product([1, 2], [6, 7])
        for k in [(1, 6), (1, 7), (2, 6), (2, 7)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_product_gen(self):
        async def gen(x):
            yield x
            yield x + 1

        it = ait.product(gen(1), gen(6))
        for k in [(1, 6), (1, 7), (2, 6), (2, 7)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_repeat(self):
        it = ait.repeat(42)
        for k in [42] * 10:
            self.assertEqual(await ait.next(it), k)

    @async_test
    async def test_repeat_limit(self):
        it = ait.repeat(42, 5)
        for k in [42] * 5:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_starmap_function_list(self):
        data = [slist[:2], slist[1:], slist]

        def concat(*args):
            return "".join(args)

        it = ait.starmap(concat, data)
        for k in ["AB", "BC", "ABC"]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_starmap_function_gen(self):
        def gen():
            yield slist[:2]
            yield slist[1:]
            yield slist

        def concat(*args):
            return "".join(args)

        it = ait.starmap(concat, gen())
        for k in ["AB", "BC", "ABC"]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_starmap_coroutine_list(self):
        data = [slist[:2], slist[1:], slist]

        async def concat(*args):
            return "".join(args)

        it = ait.starmap(concat, data)
        for k in ["AB", "BC", "ABC"]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_starmap_coroutine_gen(self):
        async def gen():
            yield slist[:2]
            yield slist[1:]
            yield slist

        async def concat(*args):
            return "".join(args)

        it = ait.starmap(concat, gen())
        for k in ["AB", "BC", "ABC"]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_takewhile_empty(self):
        def pred(x):
            return x < 3

        values = await ait.list(ait.takewhile(pred, []))
        self.assertEqual(values, [])

    @async_test
    async def test_takewhile_function_list(self):
        def pred(x):
            return x < 3

        it = ait.takewhile(pred, srange)
        for k in [1, 2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_takewhile_function_gen(self):
        async def gen():
            yield 1
            yield 2
            yield 3

        def pred(x):
            return x < 3

        it = ait.takewhile(pred, gen())
        for k in [1, 2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_takewhile_coroutine_list(self):
        async def pred(x):
            return x < 3

        it = ait.takewhile(pred, srange)
        for k in [1, 2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_takewhile_coroutine_gen(self):
        def gen():
            yield 1
            yield 2
            yield 3

        async def pred(x):
            return x < 3

        it = ait.takewhile(pred, gen())
        for k in [1, 2]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_tee_list_two(self):
        it1, it2 = ait.tee(slist * 2)

        for k in slist * 2:
            a, b = await asyncio.gather(ait.next(it1), ait.next(it2))
            self.assertEqual(a, b)
            self.assertEqual(a, k)
            self.assertEqual(b, k)
        for it in [it1, it2]:
            with self.assertRaises(StopAsyncIteration):
                await ait.next(it)

    @async_test
    async def test_tee_list_six(self):
        itrs = ait.tee(slist * 2, n=6)

        for k in slist * 2:
            values = await asyncio.gather(*[ait.next(it) for it in itrs])
            for value in values:
                self.assertEqual(value, k)
        for it in itrs:
            with self.assertRaises(StopAsyncIteration):
                await ait.next(it)

    @async_test
    async def test_tee_gen_two(self):
        async def gen():
            yield 1
            yield 4
            yield 9
            yield 16

        it1, it2 = ait.tee(gen())

        for k in [1, 4, 9, 16]:
            a, b = await asyncio.gather(ait.next(it1), ait.next(it2))
            self.assertEqual(a, b)
            self.assertEqual(a, k)
            self.assertEqual(b, k)
        for it in [it1, it2]:
            with self.assertRaises(StopAsyncIteration):
                await ait.next(it)

    @async_test
    async def test_tee_gen_six(self):
        async def gen():
            yield 1
            yield 4
            yield 9
            yield 16

        itrs = ait.tee(gen(), n=6)

        for k in [1, 4, 9, 16]:
            values = await asyncio.gather(*[ait.next(it) for it in itrs])
            for value in values:
                self.assertEqual(value, k)
        for it in itrs:
            with self.assertRaises(StopAsyncIteration):
                await ait.next(it)

    @async_test
    async def test_tee_propagate_exception(self):
        class MyError(Exception):
            pass

        async def gen():
            yield 1
            yield 2
            raise MyError

        async def consumer(it):
            result = 0
            async for item in it:
                result += item
            return result

        it1, it2 = ait.tee(gen())

        values = await asyncio.gather(
            consumer(it1),
            consumer(it2),
            return_exceptions=True,
        )

        for value in values:
            self.assertIsInstance(value, MyError)

    @async_test
    async def test_zip_longest_range(self):
        a = range(3)
        b = range(5)

        it = ait.zip_longest(a, b)

        for k in [(0, 0), (1, 1), (2, 2), (None, 3), (None, 4)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_zip_longest_fillvalue(self):
        async def gen():
            yield 1
            yield 4
            yield 9
            yield 16

        a = gen()
        b = range(5)

        it = ait.zip_longest(a, b, fillvalue=42)

        for k in [(1, 0), (4, 1), (9, 2), (16, 3), (42, 4)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaises(StopAsyncIteration):
            await ait.next(it)

    @async_test
    async def test_zip_longest_exception(self):
        async def gen():
            yield 1
            yield 2
            raise Exception("fake error")

        a = gen()
        b = ait.repeat(5)

        it = ait.zip_longest(a, b)

        for k in [(1, 5), (2, 5)]:
            self.assertEqual(await ait.next(it), k)
        with self.assertRaisesRegex(Exception, "fake error"):
            await ait.next(it)
