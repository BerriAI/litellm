# conftest.py
#
# xdist-compatible test isolation for llm_translation tests.
# Mirrors the pattern in tests/local_testing/conftest.py:
#   - Function-scoped fixture resets litellm globals to true defaults
#   - Module-scoped reload only in single-process mode

import asyncio
import importlib
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm  # noqa: E402

from tests._vcr_conftest_common import (  # noqa: E402
    VerboseReporterState,
    apply_vcr_auto_marker_to_items,
    record_vcr_outcome,
    register_persister_if_enabled,
    vcr_config_dict,
)

# vcrpy and respx both patch the httpx transport — applying both makes one
# silently win, so respx-using files opt out of the auto-marker.
_RESPX_CONFLICTING_FILES = frozenset(
    {
        "test_gpt4o_audio.py",
        "test_nvidia_nim.py",
        "test_openai.py",
        "test_openai_o1.py",
        "test_prompt_caching.py",
        "test_text_completion_unit_tests.py",
        "test_xai.py",
    }
)
_VCR_AUTO_MARKER_SKIP_FILES = _RESPX_CONFLICTING_FILES | frozenset(
    {"test_vcr_redis_persister.py"}
)

# Tests that observe live cross-call provider state (e.g. prompt-cache
# warm-up between two consecutive calls); replay can't reproduce that state.
_VCR_INCOMPATIBLE_NODEID_SUFFIXES = (
    "::test_prompt_caching",
    "TestBedrockInvokeNovaJson::test_json_response_pydantic_obj",
    "::test_bedrock_converse__streaming_passthrough",
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


# ---------------------------------------------------------------------------
# Capture TRUE defaults at conftest import time (before test modules pollute).
# ---------------------------------------------------------------------------
_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):  # Add event_loop as a dependency
    sys.path.insert(0, os.path.abspath("../.."))

    import litellm

    # ---- Save current state (for teardown restore) ----
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

    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)

    # ---- Reset to true defaults before the test ----
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)

    # Set the event loop from the fixture
    asyncio.set_event_loop(event_loop)

    yield

    # ---- Teardown ----
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)

    # Clean up any pending tasks
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()

    # Run the event loop until all tasks are cancelled
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


def pytest_collection_modifyitems(config, items):
    apply_vcr_auto_marker_to_items(
        items,
        skip_files=_VCR_AUTO_MARKER_SKIP_FILES,
        skip_nodeid_suffixes=_VCR_INCOMPATIBLE_NODEID_SUFFIXES,
    )

    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    items[:] = custom_logger_tests + other_tests
