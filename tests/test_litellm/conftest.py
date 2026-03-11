# conftest.py - IMPROVED VERSION
#
# Key changes:
# 1. Changed module reload from 'module' scope to 'function' scope for better isolation
# 2. Made cache flushing happen per-function instead of per-module
# 3. Removed manual event loop creation (let pytest-asyncio handle it)
# 4. Added proper cleanup in fixtures
# 5. Added worker-specific isolation for parallel execution

import copy
import importlib
import logging
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

import litellm
from litellm._logging import (
    handler as litellm_default_handler,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)


def _copy_provider_mapping(mapping):
    return {key: value.copy() if isinstance(value, (list, set, dict)) else value for key, value in mapping.items()}


def _refresh_litellm_module_refs():
    global litellm, litellm_default_handler, verbose_logger, verbose_proxy_logger, verbose_router_logger

    litellm = importlib.import_module("litellm")
    litellm_logging = importlib.import_module("litellm._logging")
    litellm_default_handler = litellm_logging.handler
    verbose_logger = litellm_logging.verbose_logger
    verbose_router_logger = litellm_logging.verbose_router_logger
    verbose_proxy_logger = litellm_logging.verbose_proxy_logger


_INITIAL_LITELLM_LOCAL_MODEL_COST_MAP = os.getenv("LITELLM_LOCAL_MODEL_COST_MAP")
_INITIAL_MODEL_COST_MAP_URL = litellm.model_cost_map_url
_INITIAL_DROP_PARAMS = getattr(litellm, "drop_params", False)
_BASE_MODEL_COST = copy.deepcopy(litellm.get_model_cost_map(url=_INITIAL_MODEL_COST_MAP_URL))
_BASE_MODEL_SETS = {
    name: value.copy() for name, value in vars(litellm).items() if name.endswith("_models") and isinstance(value, set)
}
_BASE_MODEL_LIST = list(getattr(litellm, "model_list", []))
_BASE_MODEL_LIST_SET = set(getattr(litellm, "model_list_set", set()))
_BASE_MODELS_BY_PROVIDER = _copy_provider_mapping(litellm.models_by_provider)
_BASE_LITELLM_LOGGER_STATE = {
    verbose_logger.name: {"level": verbose_logger.level, "propagate": verbose_logger.propagate},
    verbose_router_logger.name: {
        "level": verbose_router_logger.level,
        "propagate": verbose_router_logger.propagate,
    },
    verbose_proxy_logger.name: {
        "level": verbose_proxy_logger.level,
        "propagate": verbose_proxy_logger.propagate,
    },
}


def _restore_model_cost_state():
    _refresh_litellm_module_refs()
    from litellm.utils import _invalidate_model_cost_lowercase_map

    if _INITIAL_LITELLM_LOCAL_MODEL_COST_MAP is None:
        os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    else:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = _INITIAL_LITELLM_LOCAL_MODEL_COST_MAP

    litellm.model_cost_map_url = _INITIAL_MODEL_COST_MAP_URL
    litellm.drop_params = _INITIAL_DROP_PARAMS
    litellm.model_cost = copy.deepcopy(_BASE_MODEL_COST)

    for attr_name, base_value in _BASE_MODEL_SETS.items():
        current_value = getattr(litellm, attr_name, None)
        if isinstance(current_value, set):
            current_value.clear()
            current_value.update(base_value)
        else:
            setattr(litellm, attr_name, base_value.copy())

    litellm.model_list = list(_BASE_MODEL_LIST)
    litellm.model_list_set = set(_BASE_MODEL_LIST_SET)
    litellm.models_by_provider = _copy_provider_mapping(_BASE_MODELS_BY_PROVIDER)
    _invalidate_model_cost_lowercase_map()


def _get_litellm_utils_callback_list():
    import litellm.utils as litellm_utils

    callback_list = getattr(litellm_utils, "callback_list", [])
    if isinstance(callback_list, list):
        return litellm_utils, callback_list.copy()
    return litellm_utils, []


def _make_fresh_litellm_handler():
    handler = logging.StreamHandler()
    handler.setLevel(litellm_default_handler.level)
    if litellm_default_handler.formatter is not None:
        handler.setFormatter(litellm_default_handler.formatter)
    return handler


def _restore_litellm_logger_state():
    _refresh_litellm_module_refs()
    logger_map = {
        verbose_logger.name: verbose_logger,
        verbose_router_logger.name: verbose_router_logger,
        verbose_proxy_logger.name: verbose_proxy_logger,
    }

    for logger_name, state in _BASE_LITELLM_LOGGER_STATE.items():
        logger = logger_map[logger_name]
        logger.handlers.clear()
        logger.addHandler(_make_fresh_litellm_handler())
        logger.setLevel(state["level"])
        logger.propagate = state["propagate"]
        logger.disabled = False


def _reset_global_logging_worker():
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    if GLOBAL_LOGGING_WORKER._queue is not None:
        while True:
            try:
                task = GLOBAL_LOGGING_WORKER._queue.get_nowait()
            except Exception:
                break

            coroutine = task.get("coroutine")
            close = getattr(coroutine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            try:
                GLOBAL_LOGGING_WORKER._queue.task_done()
            except Exception:
                pass

    for running_task in list(GLOBAL_LOGGING_WORKER._running_tasks):
        try:
            running_task.cancel()
        except Exception:
            pass

    if GLOBAL_LOGGING_WORKER._worker_task is not None:
        try:
            GLOBAL_LOGGING_WORKER._worker_task.cancel()
        except Exception:
            pass

    GLOBAL_LOGGING_WORKER._running_tasks.clear()
    GLOBAL_LOGGING_WORKER._worker_task = None
    GLOBAL_LOGGING_WORKER._queue = None
    GLOBAL_LOGGING_WORKER._sem = None
    GLOBAL_LOGGING_WORKER._bound_loop = None
    GLOBAL_LOGGING_WORKER._last_aggressive_clear_time = 0.0
    GLOBAL_LOGGING_WORKER._aggressive_clear_in_progress = False


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
    _refresh_litellm_module_refs()
    _restore_model_cost_state()

    # Store original callback state (all callback lists)
    original_state = {}
    if hasattr(litellm, "callbacks"):
        original_state["callbacks"] = litellm.callbacks.copy() if litellm.callbacks else []
    if hasattr(litellm, "input_callback"):
        original_state["input_callback"] = (
            litellm.input_callback.copy() if litellm.input_callback else []
        )
    if hasattr(litellm, "service_callback"):
        original_state["service_callback"] = (
            litellm.service_callback.copy() if litellm.service_callback else []
        )
    if hasattr(litellm, "success_callback"):
        original_state["success_callback"] = litellm.success_callback.copy() if litellm.success_callback else []
    if hasattr(litellm, "failure_callback"):
        original_state["failure_callback"] = litellm.failure_callback.copy() if litellm.failure_callback else []
    if hasattr(litellm, "_async_input_callback"):
        original_state["_async_input_callback"] = (
            litellm._async_input_callback.copy() if litellm._async_input_callback else []
        )
    if hasattr(litellm, "_async_success_callback"):
        original_state["_async_success_callback"] = (
            litellm._async_success_callback.copy() if litellm._async_success_callback else []
        )
    if hasattr(litellm, "_async_failure_callback"):
        original_state["_async_failure_callback"] = (
            litellm._async_failure_callback.copy() if litellm._async_failure_callback else []
        )
    litellm_utils, original_callback_list = _get_litellm_utils_callback_list()
    original_state["litellm_utils_callback_list"] = original_callback_list

    # Store transport/network globals — many tests set these without restoring,
    # causing subsequent tests to get None from _create_async_transport()
    for _attr in ("disable_aiohttp_transport", "force_ipv4"):
        if hasattr(litellm, _attr):
            original_state[_attr] = getattr(litellm, _attr)

    # Flush cache before test (critical for respx mocks)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    _restore_litellm_logger_state()
    _reset_global_logging_worker()

    if hasattr(litellm, "input_callback"):
        litellm.input_callback = []
    if hasattr(litellm, "service_callback"):
        litellm.service_callback = []
    # Clear success/failure callbacks to prevent chaining
    if hasattr(litellm, "callbacks"):
        litellm.callbacks = []
    if hasattr(litellm, "success_callback"):
        litellm.success_callback = []
    if hasattr(litellm, "failure_callback"):
        litellm.failure_callback = []
    if hasattr(litellm, "_async_input_callback"):
        litellm._async_input_callback = []
    if hasattr(litellm, "_async_success_callback"):
        litellm._async_success_callback = []
    if hasattr(litellm, "_async_failure_callback"):
        litellm._async_failure_callback = []
    litellm_utils.callback_list = []

    yield

    # Cleanup after test
    _refresh_litellm_module_refs()
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    _restore_litellm_logger_state()
    _reset_global_logging_worker()

    # Restore all callback lists to original state
    for attr_name, original_value in original_state.items():
        if attr_name == "litellm_utils_callback_list":
            litellm_utils.callback_list = original_value
            continue
        if hasattr(litellm, attr_name):
            setattr(litellm, attr_name, original_value)

    _restore_model_cost_state()


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup/teardown for heavy initialization.

    Use this sparingly - most state should be handled by isolate_litellm_state.
    Only reload modules here if absolutely necessary.
    """
    sys.path.insert(0, os.path.abspath("../.."))

    import litellm

    # Only reload if NOT running in parallel (module reload + parallel = bad)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        # Single process mode - safe to reload
        importlib.reload(litellm)

        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception as e:
            sys.stderr.write(f"Error reloading litellm.proxy.proxy_server: {e}\n")

        # Flush cache after reload (prevents stale client instances)
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()

    yield

    # Teardown - no need to manually manage event loops with pytest-asyncio auto mode


def pytest_collection_modifyitems(config, items):
    """
    Customize test collection order.

    - Separate tests marked with 'no_parallel' from parallelizable tests
    - Sort custom_logger tests first (they tend to interfere with other tests)
    """
    # Separate no_parallel tests
    no_parallel_tests = [item for item in items if any(mark.name == "no_parallel" for mark in item.iter_markers())]

    # Separate custom_logger tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name and item not in no_parallel_tests
    ]

    # Everything else
    other_tests = [item for item in items if item not in no_parallel_tests and item not in custom_logger_tests]

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
    config.addinivalue_line("markers", "flaky: mark test as potentially flaky (should use --reruns)")

    # Detect if running in CI
    _ = os.environ.get("CI") == "true" or os.environ.get("LITELLM_CI") == "true"


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
