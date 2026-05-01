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


# Controller-side handles for writing per-test VCR verdicts to the live
# terminal. ``pytest_configure`` stashes the pluginmanager (workers don't get
# a TerminalReporter — their output is captured and aggregated by the
# controller), and ``pytest_runtest_logreport`` resolves the TerminalReporter
# lazily on first use because it isn't registered yet at conftest configure
# time.
_controller_pluginmanager = None
_controller_terminal_reporter = None


# vcrpy and respx both patch the httpx transport — applying both makes one
# silently win. Files in this set use respx and are skipped by the
# auto-marker below.
_RESPX_CONFLICTING_FILES = frozenset(
    {
        "test_azure_o_series.py",
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
# warm-up between two consecutive calls) cannot benefit from cassette
# replay: the second call's "expected" state depends on what the *live*
# provider does between the two calls, not on what was recorded earlier.
# Auto-marking them with @pytest.mark.vcr just wastes cycles and (before
# the outcome gate) used to poison the cache. They go live with their
# existing @pytest.mark.flaky retry logic.
#
# Match by suffix on the pytest nodeid so subclassed/parametrized variants
# are covered: e.g. "::test_prompt_caching" matches all subclasses that
# inherit the base test.
_VCR_INCOMPATIBLE_NODEID_SUFFIXES = frozenset(
    {
        # Provider prompt-cache propagation isn't deterministic between two
        # back-to-back calls; the test is flaky against the live provider.
        "::test_prompt_caching",
        # Bedrock Nova returns tool_call vs JSON nondeterministically; the
        # base assertion expects JSON. Other providers' versions of this
        # test are healthy, so we narrow with a class-name guard below.
        "TestBedrockInvokeNovaJson::test_json_response_pydantic_obj",
        # Bedrock streaming response_cost calc returns None intermittently.
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
    # Cassettes live on a dedicated Redis (CASSETTE_REDIS_URL) so the cache
    # isn't shared with — and accidentally flushed by — tests that exercise
    # the application Redis via REDIS_URL/REDIS_HOST.
    return not os.environ.get("CASSETTE_REDIS_URL")


def pytest_recording_configure(config, vcr):
    if _vcr_disabled():
        return
    vcr.register_persister(make_redis_persister())
    patch_vcrpy_aiohttp_record_path()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach each phase's report to the item so fixture teardown can read it.

    Used by ``_vcr_outcome_gate`` below to skip persisting cassettes for
    failed test runs (incl. failed retries that pytest-rerunfailures will
    re-attempt) so a "bad luck" recording can't poison future replays.
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    """Tell the persister whether the test that owns this cassette passed.

    Runs after ``vcr`` (which yields the active Cassette). At teardown time
    the call-phase report is attached to the item by the makereport hook
    above, so we can mark the cassette key passed/failed before vcrpy's
    Cassette.__exit__ triggers persister.save_cassette.

    Stashes a per-test hit/miss verdict on ``user_properties`` so the
    controller-side ``pytest_runtest_logreport`` hook can surface it to the
    live terminal. xdist serializes ``user_properties`` on each phase's
    report back to the controller, which is the only process that has a
    TerminalReporter wired to CI's live log.
    """
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
    """Stash the pluginmanager so the logreport hook can find TerminalReporter.

    We can't grab TerminalReporter directly here — it's not registered until
    pytest's own ``pytest_configure`` runs, and conftest hooks may run first.
    Stashing the config is enough; the hook resolves on first use.
    """
    global _controller_pluginmanager
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return  # workers don't have a live-log TerminalReporter
    _controller_pluginmanager = config.pluginmanager


def _resolve_terminal_reporter():
    """Lazy-resolve the TerminalReporter once it's been registered."""
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
    """Print VCR verdicts on the controller, alongside PASSED/FAILED markers.

    Runs once per phase per test. We pick teardown so the verdict (appended
    in ``_vcr_outcome_gate`` teardown) is present in ``report.user_properties``.
    """
    if report.when != "teardown":
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return  # only the controller has a live-log TerminalReporter
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
