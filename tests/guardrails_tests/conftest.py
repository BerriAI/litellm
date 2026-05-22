# conftest.py
#
# xdist-compatible test isolation for guardrails tests.
# Pattern matches tests/test_litellm/conftest.py:
#   - Function-scoped fixture saves/restores litellm globals (no reload)
#   - Module-scoped fixture reloads only in single-process mode

import importlib
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm

from tests._vcr_conftest_common import (  # noqa: E402
    VerboseReporterState,
    apply_vcr_auto_marker_to_items,
    record_vcr_outcome,
    register_persister_if_enabled,
    vcr_config_dict,
)

_verbose_state = VerboseReporterState()


@pytest.fixture(scope="module")
def vcr_config():
    return vcr_config_dict()


def pytest_recording_configure(config, vcr):
    register_persister_if_enabled(vcr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    yield
    record_vcr_outcome(request, vcr)


def pytest_configure(config):
    _verbose_state.remember_pluginmanager(config)


def pytest_runtest_logreport(report):
    _verbose_state.maybe_emit_verdict(report)


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Saves and restores litellm callback/global state so tests don't leak
    side effects. Works safely under pytest-xdist parallel execution.
    """
    # Save original callback state
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []

    # Save other globals that tests commonly mutate
    for attr in ("set_verbose", "cache", "num_retries"):
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)

    # Flush cache before test
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    # Clear callbacks before test
    for attr in (
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])

    yield

    # Restore all saved state
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

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
    apply_vcr_auto_marker_to_items(items)

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
