"""
Shared VCR (Redis-backed) plumbing for test directories.

This module is imported by per-directory ``conftest.py`` files to enable
24-hour HTTP caching against a Redis backend. The first run hits live
provider APIs and records the exchange; subsequent runs within 24h replay
from Redis without touching the network. See
``tests/llm_translation/Readme.md`` for the full design notes.

Each consuming ``conftest.py`` should:

1. Define a ``vcr_config`` fixture that delegates to :func:`vcr_config_dict`.
2. Define ``pytest_recording_configure(config, vcr)`` that calls
   :func:`register_persister_if_enabled`.
3. Define ``pytest_runtest_makereport`` and a ``_vcr_outcome_gate`` autouse
   fixture using :func:`make_outcome_gate_fixture` (or copy the snippet) so
   failed tests don't poison cassettes.
4. Add ``apply_vcr_auto_marker_to_items`` inside
   ``pytest_collection_modifyitems`` so non-respx tests are auto-marked
   with ``pytest.mark.vcr``.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import pytest

from tests._vcr_redis_persister import (
    filter_non_2xx_response,
    format_vcr_verdict,
    make_redis_persister,
    mark_test_outcome_for_cassette,
    patch_vcrpy_aiohttp_record_path,
    vcr_verbose_enabled,
)


SAFE_BODY_MATCHER_NAME = "safe_body"

FILTERED_REQUEST_HEADERS = (
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

FILTERED_RESPONSE_HEADERS = (
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
            if header.lower() in FILTERED_RESPONSE_HEADERS:
                headers.pop(header, None)
    return response


def _before_record_response(response):
    return filter_non_2xx_response(_scrub_response(response))


def _safe_body_matcher(r1, r2) -> None:
    """Body matcher that compares raw bytes and never raises on bad JSON.

    vcrpy's stock ``body`` matcher inspects ``Content-Type`` and runs
    ``json.loads`` on bodies typed ``application/json`` so it can compare
    semantically. That crashes (``json.JSONDecodeError: Extra data``) on
    JSON Lines payloads — which the Bedrock batch S3 PUT and a few other
    upload paths use — before the matcher even gets a chance to return
    "not a match".

    This matcher avoids the JSON normalization step entirely. It first
    tries direct ``==`` equality on the original payloads (so dicts,
    lists, etc. are compared structurally), then falls back to a bytes
    comparison after coercing ``str`` to UTF-8. Anything that's not
    bytes/str/equal-by-default is treated as a mismatch. This is
    strictly more conservative than vcrpy's default — the only thing it
    gives up is "different JSON key order is treated as the same body",
    which doesn't matter for our deterministic litellm-built request
    payloads. It can never produce a false positive that the default
    would have rejected.

    The trade-off is that bodies containing nondeterministic values (UUIDs,
    timestamps) will produce a cache miss; the right fix for those cases
    is a ``before_record_request`` scrubber, not a smarter matcher.
    """
    body1 = getattr(r1, "body", None)
    body2 = getattr(r2, "body", None)
    if body1 == body2:
        return

    def _to_bytes(b):
        if b is None:
            return b""
        if isinstance(b, bytes):
            return b
        if isinstance(b, str):
            return b.encode("utf-8")
        return None

    n1 = _to_bytes(body1)
    n2 = _to_bytes(body2)
    if n1 is not None and n2 is not None:
        if n1 == n2:
            return
        raise AssertionError("request bodies differ")
    raise AssertionError("request bodies differ")


def vcr_config_dict() -> dict:
    """Return the VCR config dict shared across all consuming conftests."""
    return {
        "filter_headers": list(FILTERED_REQUEST_HEADERS),
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
            SAFE_BODY_MATCHER_NAME,
        ),
        "before_record_response": _before_record_response,
    }


def vcr_disabled() -> bool:
    """VCR is disabled when explicitly turned off, or when no Redis is configured."""
    if os.environ.get("LITELLM_VCR_DISABLE") == "1":
        return True
    return not os.environ.get("CASSETTE_REDIS_URL")


def register_persister_if_enabled(vcr) -> None:
    """Wire the Redis persister and custom matchers into vcrpy if VCR is
    enabled.

    Call this from ``pytest_recording_configure(config, vcr)`` in conftest.
    """
    if vcr_disabled():
        return
    vcr.register_persister(make_redis_persister())
    vcr.register_matcher(SAFE_BODY_MATCHER_NAME, _safe_body_matcher)
    patch_vcrpy_aiohttp_record_path()


def apply_vcr_auto_marker_to_items(
    items,
    *,
    skip_files: Iterable[str] = (),
    skip_nodeid_suffixes: Iterable[str] = (),
) -> None:
    """Auto-apply ``pytest.mark.vcr`` to collected items.

    ``skip_files`` is a set of basenames (e.g. ``test_openai.py``) that
    should not be auto-marked — typically files that already use ``respx``,
    since respx and vcrpy both patch the httpx transport and conflict.

    ``skip_nodeid_suffixes`` is a set of node-id suffixes (e.g.
    ``"::test_prompt_caching"``) that observe live cross-call provider
    state which replay can't reproduce.
    """
    if vcr_disabled():
        return
    skip_files = frozenset(skip_files)
    skip_nodeid_suffixes = tuple(skip_nodeid_suffixes)
    for item in items:
        filename = os.path.basename(str(item.path))
        if filename in skip_files:
            continue
        if any(item.nodeid.endswith(suffix) for suffix in skip_nodeid_suffixes):
            continue
        if item.get_closest_marker("vcr") is not None:
            continue
        item.add_marker(pytest.mark.vcr)


def record_vcr_outcome(request, vcr) -> None:
    """Mark the cassette with the test outcome and emit a verbose verdict.

    Call this from a ``yield``-after section of an autouse fixture in
    conftest, after the test has run.
    """
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


# ---------------------------------------------------------------------------
# Verbose-verdict reporter helpers (optional; used by conftests that want to
# print "[VCR HIT]/[VCR MISS]/..." lines next to each test in CI logs).
# ---------------------------------------------------------------------------
class VerboseReporterState:
    """Container for the controller-process plugin manager / terminal reporter.

    A single instance lives in each conftest that wants verbose output.
    """

    def __init__(self) -> None:
        self.pluginmanager = None
        self.terminal_reporter = None

    def remember_pluginmanager(self, config) -> None:
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return
        self.pluginmanager = config.pluginmanager

    def resolve_terminal_reporter(self):
        if self.terminal_reporter is not None:
            return self.terminal_reporter
        if self.pluginmanager is None:
            return None
        self.terminal_reporter = self.pluginmanager.getplugin("terminalreporter")
        return self.terminal_reporter

    def maybe_emit_verdict(self, report) -> None:
        if report.when != "teardown":
            return
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return
        if not vcr_verbose_enabled():
            return
        reporter = self.resolve_terminal_reporter()
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
# Drop-in conftest snippet (copy/paste guidance, not executed).
# ---------------------------------------------------------------------------
# from tests._vcr_conftest_common import (
#     VerboseReporterState,
#     apply_vcr_auto_marker_to_items,
#     record_vcr_outcome,
#     register_persister_if_enabled,
#     vcr_config_dict,
# )
#
# _verbose_state = VerboseReporterState()
# _RESPX_CONFLICTING_FILES = frozenset({...})
#
# @pytest.fixture(scope="module")
# def vcr_config():
#     return vcr_config_dict()
#
# def pytest_recording_configure(config, vcr):
#     register_persister_if_enabled(vcr)
#
# @pytest.hookimpl(hookwrapper=True)
# def pytest_runtest_makereport(item, call):
#     outcome = yield
#     rep = outcome.get_result()
#     setattr(item, f"rep_{rep.when}", rep)
#
# @pytest.fixture(autouse=True)
# def _vcr_outcome_gate(request, vcr):
#     yield
#     record_vcr_outcome(request, vcr)
#
# def pytest_configure(config):
#     _verbose_state.remember_pluginmanager(config)
#
# def pytest_runtest_logreport(report):
#     _verbose_state.maybe_emit_verdict(report)
#
# def pytest_collection_modifyitems(config, items):
#     apply_vcr_auto_marker_to_items(
#         items, skip_files=_RESPX_CONFLICTING_FILES,
#     )
