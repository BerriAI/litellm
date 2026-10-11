"""
Memory Leak Detection Tests - Linear Memory Growth

Tests that check for linear/progressive memory growth by running different numbers
of requests (1k, 2k, 4k, 10k, 30k) with the same memory limit. If lower request
count tests pass but higher ones fail, it indicates linear memory growth per request.

These tests will fail if memory leaks are detected, helping catch OOM issues before production.

IMPORTANT: These tests should be run INDIVIDUALLY, not all together. Running them
together causes memory baseline drift between tests, making it difficult to detect
linear growth accurately. Each test should be run in isolation:

    pytest tests/integration/compatibility/test_linear_memory_growth.py::test_memory_baseline_1k -v
    pytest tests/integration/compatibility/test_linear_memory_growth.py::test_memory_baseline_2k -v
    # etc.
"""

import pytest

from tests.integration._support.memory_leak_utils import run_memory_baseline_test

MEMORY_LIMIT = "40 MB"


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel
async def test_memory_baseline_1k(test_router, limit_memory):
    await run_memory_baseline_test(1000, test_router, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel
async def test_memory_baseline_2k(test_router, limit_memory):
    await run_memory_baseline_test(2000, test_router, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel
async def test_memory_baseline_4k(test_router, limit_memory):
    await run_memory_baseline_test(4000, test_router, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel
async def test_memory_baseline_10k(test_router, limit_memory):
    await run_memory_baseline_test(10000, test_router, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel
async def test_memory_baseline_30k(test_router, limit_memory):
    await run_memory_baseline_test(30000, test_router, limit_memory)
