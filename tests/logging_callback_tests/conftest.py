# conftest.py
#
# xdist-compatible test isolation for logging callback tests.
#   - Function-scoped fixture saves/restores litellm globals (no reload)
#   - Module-scoped fixture reloads only in single-process mode
#   - Clears _in_memory_loggers to prevent cached logger instance leaks

import importlib
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


_LIST_ATTRS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
    "pre_call_rules",
    "post_call_rules",
)

_SCALAR_ATTRS = (
    "set_verbose",
    "cache",
    "num_retries",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "datadog_params",
    "vector_store_registry",
)


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Saves and restores litellm callback/global state so tests don't leak
    side effects. Works safely under pytest-xdist parallel execution.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging

    original_state = {}

    # Save list-type attrs (callbacks)
    for attr in _LIST_ATTRS:
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if isinstance(val, list) else val

    # Save scalar attrs
    for attr in _SCALAR_ATTRS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)

    # Flush cache and clear internal logger instances before test
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    # Clear cached logger instances (LangsmithLogger, SlackAlerting, etc.)
    ll_logging._in_memory_loggers.clear()

    # Clear callbacks before test
    for attr in _LIST_ATTRS:
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])

    yield

    # Restore all saved state
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    ll_logging._in_memory_loggers.clear()

    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    sys.path.insert(0, os.path.abspath("../.."))

    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)

        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception as e:
            print(f"Error reloading litellm.proxy.proxy_server: {e}")

        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()

    yield


def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests
