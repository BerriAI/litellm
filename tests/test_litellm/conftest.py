# conftest.py - IMPROVED VERSION
#
# Key changes:
# 1. Changed module reload from 'module' scope to 'function' scope for better isolation
# 2. Made cache flushing happen per-function instead of per-module
# 3. Removed manual event loop creation (let pytest-asyncio handle it)
# 4. Added proper cleanup in fixtures
# 5. Added worker-specific isolation for parallel execution

import importlib
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio

import litellm


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture (changed from module scope).

    This ensures better isolation when running tests in parallel:
    - Each test function gets a clean litellm state
    - Cache is flushed before each test
    - No module reloading during parallel execution

    Note: Module reloading at function scope is safer for parallel execution
    but adds overhead. Consider removing reload entirely if tests can work without it.
    """
    # Get worker ID if running with pytest-xdist
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')

    # Store original callback state (all callback lists)
    original_state = {}
    if hasattr(litellm, 'callbacks'):
        original_state['callbacks'] = litellm.callbacks.copy() if litellm.callbacks else []
    if hasattr(litellm, 'success_callback'):
        original_state['success_callback'] = litellm.success_callback.copy() if litellm.success_callback else []
    if hasattr(litellm, 'failure_callback'):
        original_state['failure_callback'] = litellm.failure_callback.copy() if litellm.failure_callback else []
    if hasattr(litellm, '_async_success_callback'):
        original_state['_async_success_callback'] = litellm._async_success_callback.copy() if litellm._async_success_callback else []
    if hasattr(litellm, '_async_failure_callback'):
        original_state['_async_failure_callback'] = litellm._async_failure_callback.copy() if litellm._async_failure_callback else []

    # Store transport/network globals — many tests set these without restoring,
    # causing subsequent tests to get None from _create_async_transport()
    for _attr in ('disable_aiohttp_transport', 'force_ipv4'):
        if hasattr(litellm, _attr):
            original_state[_attr] = getattr(litellm, _attr)

    # Flush cache before test (critical for respx mocks)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    # Clear success/failure callbacks to prevent chaining
    if hasattr(litellm, 'success_callback'):
        litellm.success_callback = []
    if hasattr(litellm, 'failure_callback'):
        litellm.failure_callback = []
    if hasattr(litellm, '_async_success_callback'):
        litellm._async_success_callback = []
    if hasattr(litellm, '_async_failure_callback'):
        litellm._async_failure_callback = []

    yield

    # Cleanup after test
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    # Restore all callback lists to original state
    for attr_name, original_value in original_state.items():
        if hasattr(litellm, attr_name):
            setattr(litellm, attr_name, original_value)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup/teardown for heavy initialization.

    Use this sparingly - most state should be handled by isolate_litellm_state.
    Only reload modules here if absolutely necessary.
    """
    sys.path.insert(
        0, os.path.abspath("../..")
    )

    import litellm

    # Only reload if NOT running in parallel (module reload + parallel = bad)
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', None)
    if worker_id is None:
        # Single process mode - safe to reload
        importlib.reload(litellm)

        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server
                importlib.reload(litellm.proxy.proxy_server)
        except Exception as e:
            print(f"Error reloading litellm.proxy.proxy_server: {e}")

        # Flush cache after reload (prevents stale client instances)
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()

    print(f"[conftest] Module setup complete (worker: {worker_id or 'master'})")

    yield

    # Teardown - no need to manually manage event loops with pytest-asyncio auto mode
    print(f"[conftest] Module teardown complete (worker: {worker_id or 'master'})")


def pytest_collection_modifyitems(config, items):
    """
    Customize test collection order.

    - Separate tests marked with 'no_parallel' from parallelizable tests
    - Sort custom_logger tests first (they tend to interfere with other tests)
    """
    # Separate no_parallel tests
    no_parallel_tests = [
        item for item in items
        if any(mark.name == "no_parallel" for mark in item.iter_markers())
    ]

    # Separate custom_logger tests
    custom_logger_tests = [
        item for item in items
        if "custom_logger" in item.parent.name
        and item not in no_parallel_tests
    ]

    # Everything else
    other_tests = [
        item for item in items
        if item not in no_parallel_tests and item not in custom_logger_tests
    ]

    # Sort each group
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)
    no_parallel_tests.sort(key=lambda x: x.name)

    # Reorder: custom_logger first (isolated), then other tests, then no_parallel tests last
    items[:] = custom_logger_tests + other_tests + no_parallel_tests


def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # Add marker for flaky tests (for documentation purposes)
    config.addinivalue_line(
        "markers", "flaky: mark test as potentially flaky (should use --reruns)"
    )

    # Detect if running in CI
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('LITELLM_CI') == 'true'
    if is_ci:
        print("[conftest] Running in CI mode - enabling stricter test isolation")


# Optional: Add a fixture for tests that need even stricter isolation
@pytest.fixture
def strict_isolation():
    """
    Use this fixture for tests that need extra strict isolation.

    Example:
        def test_something(strict_isolation):
            # Test code with guaranteed clean state
            pass
    """
    # Force flush all caches
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    # Reset all global state
    if hasattr(litellm, "disable_aiohttp_transport"):
        original_aiohttp = litellm.disable_aiohttp_transport
        litellm.disable_aiohttp_transport = False
    else:
        original_aiohttp = None

    if hasattr(litellm, "set_verbose"):
        original_verbose = litellm.set_verbose
        litellm.set_verbose = False
    else:
        original_verbose = None

    yield

    # Restore original state
    if original_aiohttp is not None:
        litellm.disable_aiohttp_transport = original_aiohttp
    if original_verbose is not None:
        litellm.set_verbose = original_verbose

    # Final cache flush
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
