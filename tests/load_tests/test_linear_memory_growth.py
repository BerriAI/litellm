"""
Memory Leak Detection Tests - Linear Memory Growth

Tests that check for linear/progressive memory growth by running different numbers
of requests (1k, 2k, 4k, 10k, 30k) with the same memory limit. If lower request
count tests pass but higher ones fail, it indicates linear memory growth per request.

These tests will fail if memory leaks are detected, helping catch OOM issues before production.

IMPORTANT: These tests should be run INDIVIDUALLY, not all together. Running them
together causes memory baseline drift between tests, making it difficult to detect
linear growth accurately. Each test should be run in isolation:

    pytest tests/load_tests/test_linear_memory_growth.py::test_memory_baseline_1k -v
    pytest tests/load_tests/test_linear_memory_growth.py::test_memory_baseline_2k -v
    # etc.

NOTE: Not recommended for accurate results:
pytest tests/load_tests/test_linear_memory_growth.py -v
"""

import pytest

from tests.load_tests.memory_leak_utils import (
    limit_memory,  # noqa: F401  # pytest fixture used via dependency injection
    mock_server,  # noqa: F401  # pytest fixture used via dependency injection
    run_memory_baseline_test,
    test_router,  # noqa: F401  # pytest fixture used via dependency injection
)

# Memory limit for all linear memory growth tests
MEMORY_LIMIT = "40 MB"


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_memory_baseline_1k(test_router, limit_memory):
    """
    Memory baseline test with 1,000 requests.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If this passes but higher request count tests fail, indicates progressive memory leak.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_linear_memory_growth.py::test_memory_baseline_1k -v
    """
    await run_memory_baseline_test(1000, test_router, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_memory_baseline_2k(test_router, limit_memory):
    """
    Memory baseline test with 2,000 requests.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If this passes but test_memory_baseline_4k fails, indicates progressive memory leak.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_linear_memory_growth.py::test_memory_baseline_2k -v
    """
    await run_memory_baseline_test(2000, test_router, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_memory_baseline_4k(test_router, limit_memory):
    """
    Memory baseline test with 4,000 requests.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_memory_baseline_1k and test_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_linear_memory_growth.py::test_memory_baseline_4k -v
    """
    await run_memory_baseline_test(4000, test_router, limit_memory)



@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_memory_baseline_10k(test_router, limit_memory):
    """
    Memory baseline test with 10,000 requests.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_memory_baseline_1k and test_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_linear_memory_growth.py::test_memory_baseline_10k -v
    """
    await run_memory_baseline_test(10000, test_router, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_memory_baseline_30k(test_router, limit_memory):
    """
    Memory baseline test with 30,000 requests.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_memory_baseline_1k and test_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_linear_memory_growth.py::test_memory_baseline_30k -v
    """
    await run_memory_baseline_test(30000, test_router, limit_memory)
