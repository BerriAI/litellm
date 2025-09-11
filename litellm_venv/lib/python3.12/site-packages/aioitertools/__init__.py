# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

"""
itertools and builtins for AsyncIO and mixed iterables
"""

__author__ = "Amethyst Reese"
from . import asyncio
from .__version__ import __version__
from .builtins import (
    all,
    any,
    enumerate,
    iter,
    list,
    map,
    max,
    min,
    next,
    set,
    sum,
    tuple,
    zip,
)
from .itertools import (
    accumulate,
    batched,
    chain,
    combinations,
    combinations_with_replacement,
    compress,
    count,
    cycle,
    dropwhile,
    filterfalse,
    groupby,
    islice,
    permutations,
    product,
    repeat,
    starmap,
    takewhile,
    tee,
    zip_longest,
)
