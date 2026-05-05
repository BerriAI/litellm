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

from tests._vcr_redis_persister import (  # noqa: E402
    filter_non_2xx_response,
    format_vcr_verdict,
    make_redis_persister,
    mark_test_outcome_for_cassette,
    patch_vcrpy_aiohttp_record_path,
    vcr_verbose_enabled,
)

_controller_pluginmanager = None
_controller_terminal_reporter = None


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
_VCR_INCOMPATIBLE_NODEID_SUFFIXES = frozenset(
    {
        "::test_prompt_caching",
        "TestBedrockInvokeNovaJson::test_json_response_pydantic_obj",
        "::test_bedrock_converse__streaming_passthrough",
    }
)


def _is_vcr_incompatible(nodeid: str) -> bool:
    return any(nodeid.endswith(suffix) for suffix in _VCR_INCOMPATIBLE_NODEID_SUFFIXES)


_FILTERED_REQUEST_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "anthropic-version",
    "openai-api-key",
    "azure-api-key",
    "api-key",
    "cookie",
    "x-amz-security-token",
    "x-amz-date",
    "x-amz-content-sha256",
    "amz-sdk-invocation-id",
    "amz-sdk-request",
    "x-goog-api-key",
    "x-goog-user-project",
)

_FILTERED_RESPONSE_HEADERS = (
    "set-cookie",
    "x-request-id",
    "request-id",
    "cf-ray",
    "anthropic-organization-id",
    "openai-organization",
    "x-amzn-requestid",
    "x-amzn-trace-id",
    "date",
)


def _scrub_response(response):
    if not isinstance(response, dict):
        return response
    headers = response.get("headers") or {}
    if isinstance(headers, dict):
        for header in list(headers):
            if header.lower() in _FILTERED_RESPONSE_HEADERS:
                headers.pop(header, None)
    return response


def _before_record_response(response):
    return filter_non_2xx_response(_scrub_response(response))


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": list(_FILTERED_REQUEST_HEADERS),
        "decode_compressed_response": True,
        "record_mode": "new_episodes",
        "allow_playback_repeats": True,
        "match_on": (
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "body",
        ),
        "before_record_response": _before_record_response,
    }


def _vcr_disabled() -> bool:
    if os.environ.get("LITELLM_VCR_DISABLE") == "1":
        return True
    return not os.environ.get("CASSETTE_REDIS_URL")


def pytest_recording_configure(config, vcr):
    if _vcr_disabled():
        return
    vcr.register_persister(make_redis_persister())
    patch_vcrpy_aiohttp_record_path()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    yield
    cassette = vcr
    rep_call = getattr(request.node, "rep_call", None)
    test_passed = bool(rep_call and rep_call.passed)
    cassette_path = getattr(cassette, "_path", None) if cassette is not None else None
    if cassette_path:
        mark_test_outcome_for_cassette(cassette_path, test_passed)

    if not vcr_verbose_enabled():
        return
    verdict = format_vcr_verdict(cassette)
    request.node.user_properties.append(("vcr_verdict", verdict))


def pytest_configure(config):
    global _controller_pluginmanager
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    _controller_pluginmanager = config.pluginmanager


def _resolve_terminal_reporter():
    global _controller_terminal_reporter
    if _controller_terminal_reporter is not None:
        return _controller_terminal_reporter
    if _controller_pluginmanager is None:
        return None
    _controller_terminal_reporter = _controller_pluginmanager.getplugin(
        "terminalreporter"
    )
    return _controller_terminal_reporter


def pytest_runtest_logreport(report):
    if report.when != "teardown":
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    if not vcr_verbose_enabled():
        return
    reporter = _resolve_terminal_reporter()
    if reporter is None:
        return
    verdict = next(
        (v for k, v in (report.user_properties or []) if k == "vcr_verdict"),
        None,
    )
    if not verdict:
        return
    reporter.write_line(f"{verdict} :: {report.nodeid}")


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
    if not _vcr_disabled():
        for item in items:
            filename = os.path.basename(str(item.fspath))
            if filename in _VCR_AUTO_MARKER_SKIP_FILES:
                continue
            if _is_vcr_incompatible(item.nodeid):
                continue
            if item.get_closest_marker("vcr") is not None:
                continue
            item.add_marker(pytest.mark.vcr)

    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    items[:] = custom_logger_tests + other_tests
