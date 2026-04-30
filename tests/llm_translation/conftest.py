# conftest.py
#
# xdist-compatible test isolation for llm_translation tests.
# Mirrors the pattern in tests/local_testing/conftest.py:
#   - Function-scoped fixture resets litellm globals to true defaults
#   - Module-scoped reload only in single-process mode

import importlib
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
# Make sibling helper modules under tests/llm_translation/ importable regardless
# of the directory pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import litellm

import asyncio

from _vcr_redis_persister import (  # noqa: E402  (sibling module under conftest dir)
    filter_non_2xx_response,
    make_redis_persister,
)


# ---------------------------------------------------------------------------
# VCR cassette infrastructure (pytest-recording)
# ---------------------------------------------------------------------------
# Tests marked with ``@pytest.mark.vcr`` replay HTTP traffic from a cassette
# under ``cassettes/<test_module>/<test_name>.yaml`` instead of hitting the
# live provider. Default record mode is ``none`` (replay only) so CI never
# accidentally calls a real LLM. To re-record every marked test in one sweep::
#
#     ANTHROPIC_API_KEY=sk-ant-... \
#         uv run pytest tests/llm_translation -m vcr --record-mode=once
#
# See ``tests/llm_translation/cassettes/README.md`` for the full workflow.

# Headers that must never be persisted to a cassette. Matched
# case-insensitively by vcrpy.
_FILTERED_REQUEST_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    # Strip ``anthropic-version`` so live-recorded and mock-recorded cassettes
    # have the same shape (the local mock helper drops it too).
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

# Per-request response headers we strip so cassettes diff cleanly across
# re-records.
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
    """Strip per-request response headers we don't want in the cassette."""
    if not isinstance(response, dict):
        return response
    headers = response.get("headers") or {}
    if isinstance(headers, dict):
        for header in list(headers):
            if header.lower() in _FILTERED_RESPONSE_HEADERS:
                headers.pop(header, None)
    return response


def _before_record_response(response):
    """Compose per-request scrubbing with the 2xx-only cache policy.

    Order matters: we scrub headers first so we don't leak request IDs even
    on responses we end up dropping from the cassette mid-development.
    """
    response = _scrub_response(response)
    return filter_non_2xx_response(response)


@pytest.fixture(scope="module")
def vcr_config():
    """Shared VCR config consumed by ``pytest-recording``.

    Applied to every ``@pytest.mark.vcr`` test in this directory.
    """
    return {
        "filter_headers": list(_FILTERED_REQUEST_HEADERS),
        "decode_compressed_response": True,
        # Match on full request shape so streaming vs non-streaming and
        # different prompts produce distinct cassettes.
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


def pytest_recording_configure(config, vcr):
    """Swap vcrpy's default filesystem persister for a Redis-backed one.

    Opt-in via ``LITELLM_VCR_REDIS=1`` so local dev keeps the YAML-on-disk
    behaviour and CI (which sets the flag) gets a 24h-TTL cache that
    auto-refreshes against live providers without manual ``make`` runs.
    """
    if os.environ.get("LITELLM_VCR_REDIS") != "1":
        return
    vcr.register_persister(make_redis_persister())


# pytest-recording's default cassette dir is
# ``<test_dir>/cassettes/<test_module>``. Keep that — it gives every test its
# own file and avoids name collisions across modules.

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
    curr_dir = os.getcwd()
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


# Number of attempts a vcr-marked test gets when recording against a live
# provider. Replay-only runs never reach the network so this only matters on
# cache miss / record mode. Tenacity-style exponential backoff is provided by
# the underlying provider SDKs (openai, anthropic) when they see 429/5xx, so
# bumping num_retries propagates retry-with-backoff for free.
_VCR_RECORD_RETRIES = 3


@pytest.fixture(autouse=True)
def _vcr_record_retries(setup_and_teardown, request):
    """Configure record-time retries for ``@pytest.mark.vcr`` tests.

    Depends on ``setup_and_teardown`` so this runs *after* the per-test
    ``importlib.reload(litellm)`` resets ``num_retries`` back to None.
    """
    if request.node.get_closest_marker("vcr") is None:
        return
    litellm.num_retries = _VCR_RECORD_RETRIES


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
