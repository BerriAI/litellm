# conftest.py

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


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    curr_dir = os.getcwd()  # Get the current working directory
    sys.path.insert(
        0, os.path.abspath("../..")
    )  # Adds the project directory to the system path

    import litellm
    from litellm import Router

    importlib.reload(litellm)

    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            import litellm.proxy.proxy_server

            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")

    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    # from litellm import Router, completion, aembedding, acompletion, embedding
    yield

    # Teardown code (executes after the yield point)
    loop.close()  # Close the loop created earlier
    asyncio.set_event_loop(None)  # Remove the reference to the loop


def pytest_collection_modifyitems(config, items):
    if not _vcr_disabled():
        for item in items:
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
