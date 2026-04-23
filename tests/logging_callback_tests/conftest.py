# conftest.py
#
# xdist-compatible test isolation for logging callback tests.
#
# Key design: capture litellm's true default values at conftest import time
# (BEFORE test modules are imported) so we can reset to clean defaults before
# each test. This is necessary because some test modules set module-level
# globals like `litellm.num_retries = 3` which pollute state for all tests
# in the same xdist worker.

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
    "num_retries_per_request",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "datadog_params",
    "vector_store_registry",
)

# ---- Capture true defaults at conftest import time ----
# This runs BEFORE any test modules are imported, so values are clean.
_DEFAULTS: dict = {}
for _attr in _LIST_ATTRS:
    if hasattr(litellm, _attr):
        _val = getattr(litellm, _attr)
        _DEFAULTS[_attr] = _val.copy() if isinstance(_val, list) else _val
for _attr in _SCALAR_ATTRS:
    if hasattr(litellm, _attr):
        _DEFAULTS[_attr] = getattr(litellm, _attr)


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm state to the true defaults captured at conftest import time,
    then restores after the test. This prevents module-level mutations (e.g.
    `litellm.num_retries = 3` at the top of test_langfuse_e2e_test.py) from
    leaking across tests within the same xdist worker.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging

    # Flush cache and clear internal logger instances before test
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    # Clear cached logger instances (LangsmithLogger, SlackAlerting, etc.)
    ll_logging._in_memory_loggers.clear()

    # Reset ALL attrs to their true defaults before the test runs.
    # This undoes any module-level mutations from test file imports.
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)

    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])

    yield

    # Teardown: reset back to defaults again (belt-and-suspenders)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    ll_logging._in_memory_loggers.clear()

    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)

    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])


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
