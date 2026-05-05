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

import atexit
import hashlib
import os
import sys
from typing import Iterable

import pytest

from tests._vcr_redis_persister import (
    cassette_cache_capacity_snapshot,
    cassette_cache_health,
    filter_non_2xx_response,
    format_vcr_verdict,
    make_redis_persister,
    mark_test_outcome_for_cassette,
    patch_vcrpy_aiohttp_record_path,
    vcr_verbose_enabled,
)

# Surface a warning when used memory exceeds this fraction of the cap,
# even if no SET has failed yet. Gives early notice before tests start
# failing on teardown.
CASSETTE_CACHE_HIGH_WATER_FRACTION = 0.85


SAFE_BODY_MATCHER_NAME = "safe_body"
KEY_FINGERPRINT_MATCHER_NAME = "key_fingerprint"

# Synthetic header name we attach to every request before it is recorded.
# The value is a one-way hash of the API-key headers on the request. This
# lets the ``key_fingerprint`` matcher distinguish requests that carry
# different API keys *after* we have scrubbed the real key value out of
# the cassette, without ever leaking the secret. See ``_before_record_request``
# and ``_key_fingerprint_matcher`` below.
KEY_FINGERPRINT_HEADER = "x-litellm-key-fp"

# Headers that carry an API key whose VALUE distinguishes one request from
# another (e.g. "the prior request used a real key, this one uses a bad
# key, so the cassette should not be reused"). We hash these into the
# fingerprint header. Note this is intentionally narrower than
# ``FILTERED_REQUEST_HEADERS`` — AWS SigV4 headers (``x-amz-date`` etc.)
# are *also* secrets-bearing but their values change on every call, so
# fingerprinting them would defeat caching entirely.
API_KEY_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "openai-api-key",
    "azure-api-key",
    "api-key",
    "x-goog-api-key",
)

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


def _iter_header_values(headers, name: str):
    """Yield all values for ``name`` from a vcrpy request headers object.

    vcrpy normalizes request headers into a dict-like (case-insensitive on
    most transports, case-sensitive on a few). Some transports also pass
    through multi-valued headers as lists. We accept either shape so the
    fingerprint stays stable across transports.
    """
    if headers is None:
        return
    target = name.lower()
    try:
        items = headers.items()
    except AttributeError:
        return
    for key, value in items:
        if str(key).lower() != target:
            continue
        if isinstance(value, (list, tuple)):
            for v in value:
                yield v
        else:
            yield value


def _compute_key_fingerprint(request) -> str:
    """Return a short, deterministic hash of the request's API-key headers.

    Empty/missing keys hash to a stable sentinel so requests without auth
    headers all bucket together (and don't accidentally collide with any
    real key). The hash is one-way so cassettes never contain the secret.
    """
    headers = getattr(request, "headers", None)
    parts: list[str] = []
    for header_name in API_KEY_HEADERS:
        for value in _iter_header_values(headers, header_name):
            if value is None:
                continue
            text = value if isinstance(value, str) else str(value)
            text = text.strip()
            if not text:
                continue
            parts.append(f"{header_name}={text}")
    if not parts:
        return "no-key"
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _strip_headers(headers, names: Iterable[str]) -> None:
    """Remove any header in ``names`` from a vcrpy request headers dict.

    Comparison is case-insensitive; we delete *all* keys whose lowercase
    name is in ``names``. We do this in-place because vcrpy already deep-
    copies the request before invoking ``before_record_request``.
    """
    if headers is None:
        return
    targets = {n.lower() for n in names}
    try:
        keys = list(headers.keys())
    except AttributeError:
        return
    for key in keys:
        if str(key).lower() in targets:
            try:
                del headers[key]
            except (KeyError, TypeError):
                pass


def _before_record_request(request):
    """Stamp + scrub auth headers before vcrpy persists the request.

    Two responsibilities, in this order:

    1. Compute a one-way hash of the API-key headers and stash it as the
       synthetic ``KEY_FINGERPRINT_HEADER``. The ``key_fingerprint``
       matcher uses this so a request made with a different API key is
       not silently served from a cassette recorded with the real key.
    2. Strip every ``FILTERED_REQUEST_HEADERS`` value (auth headers, AWS
       SigV4 timestamps, cookies, etc.) so the cassette never contains
       the secret. We do the scrubbing here — instead of via vcrpy's
       ``filter_headers`` knob — because ``filter_headers`` runs *before*
       ``before_record_request`` and would erase the auth value before
       step 1 could read it.

    Without step 1, scrubbing the auth header would cause a "bad-key"
    request to match a "good-key" cassette (because everything else —
    method, URL, body — is identical), so vcrpy would replay the recorded
    200 and any test that expects a 401 (failure callbacks,
    ``check_valid_key`` returning False, etc.) would silently flip.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return request
    fingerprint = _compute_key_fingerprint(request)
    try:
        headers[KEY_FINGERPRINT_HEADER] = fingerprint
    except (TypeError, AttributeError):
        pass
    _strip_headers(headers, FILTERED_REQUEST_HEADERS)
    return request


def _key_fingerprint_matcher(r1, r2) -> None:
    """Match requests by the synthetic key-fingerprint header.

    Two requests are considered equivalent only if they were sent with the
    same API key (or both with no key). Combined with the existing scrub
    of the real auth header, this lets us cache responses for the real
    key without poisoning the cassette for tests that deliberately use a
    bad key.
    """

    def _fp(req):
        for value in _iter_header_values(
            getattr(req, "headers", None), KEY_FINGERPRINT_HEADER
        ):
            if value is None:
                continue
            return value if isinstance(value, str) else str(value)
        return "no-key"

    if _fp(r1) != _fp(r2):
        raise AssertionError("API key fingerprints differ")


def vcr_config_dict() -> dict:
    """Return the VCR config dict shared across all consuming conftests.

    Note: the auth-header scrubbing is intentionally performed inside
    ``_before_record_request`` instead of via vcrpy's ``filter_headers``
    knob. ``filter_headers`` runs *before* ``before_record_request``, and
    we need the raw auth header values available so we can fingerprint
    them for the ``key_fingerprint`` matcher before stripping them.
    """
    return {
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
            KEY_FINGERPRINT_MATCHER_NAME,
            SAFE_BODY_MATCHER_NAME,
        ),
        "before_record_request": _before_record_request,
        "before_record_response": _before_record_response,
    }


def vcr_disabled() -> bool:
    """VCR is disabled when explicitly turned off, or when no Redis is configured."""
    if os.environ.get("LITELLM_VCR_DISABLE") == "1":
        return True
    return not os.environ.get("CASSETTE_REDIS_URL")


_atexit_banner_registered = False


def _print_atexit_banner() -> None:
    """Stderr fallback banner — fires even when no consuming conftest
    wires up ``pytest_terminal_summary``. Skipped on xdist workers so
    the controller is the only emitter.
    """
    if vcr_disabled():
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    health = cassette_cache_health()
    save_failures = int(health.get("save_failures", 0) or 0)
    load_failures = int(health.get("load_failures", 0) or 0)
    snapshot = cassette_cache_capacity_snapshot()

    def _emit(line: str) -> None:
        sys.stderr.write(f"{line}\n")

    if save_failures or load_failures:
        bar = "=" * 60
        _emit(bar)
        _emit("VCR CASSETTE CACHE DEGRADED")
        if save_failures:
            _emit(
                f"  {save_failures} cassette save failure(s); last error: "
                f"{health.get('save_failure_last_error', '')}"
            )
        if load_failures:
            _emit(
                f"  {load_failures} cassette load failure(s); last error: "
                f"{health.get('load_failure_last_error', '')}"
            )
        if snapshot:
            _emit(_format_capacity_line(snapshot))
        _emit(bar)
        return
    if snapshot and snapshot["used_pct"] >= CASSETTE_CACHE_HIGH_WATER_FRACTION * 100:
        bar = "=" * 60
        _emit(bar)
        _emit("VCR CASSETTE CACHE NEAR CAPACITY")
        _emit(_format_capacity_line(snapshot))
        _emit(bar)


def register_persister_if_enabled(vcr) -> None:
    """Wire the Redis persister and custom matchers into vcrpy if VCR is
    enabled.

    Call this from ``pytest_recording_configure(config, vcr)`` in conftest.
    """
    if vcr_disabled():
        return
    vcr.register_persister(make_redis_persister())
    vcr.register_matcher(SAFE_BODY_MATCHER_NAME, _safe_body_matcher)
    vcr.register_matcher(KEY_FINGERPRINT_MATCHER_NAME, _key_fingerprint_matcher)
    patch_vcrpy_aiohttp_record_path()
    global _atexit_banner_registered
    if not _atexit_banner_registered:
        atexit.register(_print_atexit_banner)
        _atexit_banner_registered = True


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


def _format_capacity_line(snapshot: dict) -> str:
    used = int(snapshot.get("used_memory_bytes", 0) or 0)
    cap = int(snapshot.get("maxmemory_bytes", 0) or 0)
    pct = float(snapshot.get("used_pct", 0.0) or 0.0)
    used_mb = used / (1024 * 1024)
    cap_mb = cap / (1024 * 1024)
    return (
        f"  Cassette Redis usage: {used_mb:.1f} MiB / {cap_mb:.1f} MiB "
        f"({pct:.1f}% of maxmemory)"
    )


def emit_cassette_cache_session_banner(terminalreporter) -> None:
    """Write a session-end banner about cassette-cache health.

    - If any save/load failed during the session, prints a loud red banner
      with the failure counts and last error.
    - Otherwise, if Redis is at ≥ ``CASSETTE_CACHE_HIGH_WATER_FRACTION``
      of its maxmemory, prints an orange warning so the next CI run
      knows the cache is close to OOM.
    - No-op when VCR is disabled, no failures occurred, and capacity is
      healthy.

    Call this from ``pytest_terminal_summary(terminalreporter, ...)`` in
    consuming conftests. Safe to call from an xdist controller; xdist
    workers will skip emitting (they'd duplicate output).
    """
    if vcr_disabled():
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    health = cassette_cache_health()
    save_failures = int(health.get("save_failures", 0) or 0)
    load_failures = int(health.get("load_failures", 0) or 0)
    snapshot = cassette_cache_capacity_snapshot()

    if save_failures or load_failures:
        terminalreporter.write_sep(
            "=", "VCR CASSETTE CACHE DEGRADED", red=True, bold=True
        )
        if save_failures:
            terminalreporter.write_line(
                f"  {save_failures} cassette save failure(s); last error: "
                f"{health.get('save_failure_last_error', '')}"
            )
        if load_failures:
            terminalreporter.write_line(
                f"  {load_failures} cassette load failure(s); last error: "
                f"{health.get('load_failure_last_error', '')}"
            )
        terminalreporter.write_line(
            "  Tests still passed because cassette persistence is best-effort, "
            "but the Redis cache may be degraded (e.g. at maxmemory cap, "
            "unreachable, or read-only)."
        )
        if snapshot:
            terminalreporter.write_line(_format_capacity_line(snapshot))
        terminalreporter.write_sep("=", red=True, bold=True)
        return

    if snapshot and snapshot["used_pct"] >= CASSETTE_CACHE_HIGH_WATER_FRACTION * 100:
        terminalreporter.write_sep(
            "=", "VCR CASSETTE CACHE NEAR CAPACITY", yellow=True, bold=True
        )
        terminalreporter.write_line(_format_capacity_line(snapshot))
        terminalreporter.write_line(
            "  No save failures yet, but Redis is approaching maxmemory. "
            "Consider running tests/_flush_vcr_cache.py or letting more "
            "keys age out before the next session."
        )
        terminalreporter.write_sep("=", yellow=True, bold=True)


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
