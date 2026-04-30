# conftest.py
#
# xdist-compatible test isolation for llm_translation tests.
# Mirrors the pattern in tests/local_testing/conftest.py:
#   - Function-scoped fixture resets litellm globals to true defaults
#   - Module-scoped reload only in single-process mode
#
# Also wires up the Redis-backed VCR cache. Every test in this directory is
# auto-marked with ``@pytest.mark.vcr`` (see ``pytest_collection_modifyitems``)
# unless its file appears in ``_RESPX_CONFLICTING_FILES`` — those use respx,
# which patches the same httpx transport vcrpy does. Cache key naming, TTL,
# and 2xx-only filtering live in ``tests/_vcr_redis_persister.py``.

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
    make_redis_persister,
)


# ---------------------------------------------------------------------------
# VCR cassette infrastructure (pytest-recording + Redis)
# ---------------------------------------------------------------------------
# All tests in tests/llm_translation/ are auto-marked with ``@pytest.mark.vcr``
# (excluding the respx-using files listed below). On cache miss vcrpy records
# the live response into Redis under ``litellm:vcr:cassette:<rel_path>`` with
# a 24h TTL; subsequent runs within that window replay without touching the
# network. Set ``LITELLM_VCR_DISABLE=1`` to skip VCR entirely (e.g. when
# debugging an upstream API change locally).

# Test files that use ``respx`` to patch httpx. vcrpy patches the same
# transport, so applying both to the same test will make one of them silently
# win and the other look like a no-op. Skip auto-marking these.
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

# The persister's own unit tests must not run inside a VCR cassette context —
# they call ``save_cassette`` / ``load_cassette`` directly against fakeredis
# and don't make HTTP calls, but auto-marking them would still wrap each
# test in a Redis lookup we don't want.
_VCR_AUTO_MARKER_SKIP_FILES = _RESPX_CONFLICTING_FILES | frozenset(
    {"test_vcr_redis_persister.py"}
)

# Headers that must never be persisted to a cassette. Matched
# case-insensitively by vcrpy.
_FILTERED_REQUEST_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    # Strip ``anthropic-version`` so cassettes have a stable shape across
    # SDK versions that bump the header.
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

    ``record_mode="once"`` is what makes this a useful daily cache:
    - cassette absent (cache miss) → record the live call into Redis,
    - cassette present (cache hit) → replay only.
    24h TTL on the Redis key means each new day's first run records against
    live providers, surfacing API drift within a day instead of silently
    serving stale responses forever.
    """
    return {
        "filter_headers": list(_FILTERED_REQUEST_HEADERS),
        "decode_compressed_response": True,
        "record_mode": "once",
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


def _vcr_disabled() -> bool:
    """VCR is disabled when explicitly opted out or when Redis isn't wired.

    No Redis means no cache to read from or write to — fall back to live
    calls instead of silently writing YAML to disk (which we don't ship).
    """
    if os.environ.get("LITELLM_VCR_DISABLE") == "1":
        return True
    return not os.environ.get("REDIS_HOST")


def pytest_recording_configure(config, vcr):
    """Register the Redis-backed cassette persister."""
    if _vcr_disabled():
        return
    vcr.register_persister(make_redis_persister())


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
    # 1. Auto-apply ``@pytest.mark.vcr`` to every collected test in this
    #    directory so any provider call lands in the Redis cache. Skip files
    #    that use respx (it patches the same transport vcrpy does) and the
    #    persister's own unit tests. Skip entirely if VCR is disabled (no
    #    REDIS_HOST or LITELLM_VCR_DISABLE=1) so dev runs without Redis
    #    don't go through cassette logic at all.
    if not _vcr_disabled():
        for item in items:
            filename = os.path.basename(str(item.fspath))
            if filename in _VCR_AUTO_MARKER_SKIP_FILES:
                continue
            if item.get_closest_marker("vcr") is not None:
                continue
            item.add_marker(pytest.mark.vcr)

    # 2. Preserve the historical ordering of custom_logger tests vs the rest.
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    items[:] = custom_logger_tests + other_tests
