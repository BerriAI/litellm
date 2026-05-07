"""Shared VCR (Redis-backed) plumbing imported by per-directory conftests.

See ``tests/llm_translation/Readme.md`` for the full design and
``tests/llm_translation/conftest.py`` for the reference wiring."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
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

CASSETTE_CACHE_HIGH_WATER_FRACTION = 0.85


SAFE_BODY_MATCHER_NAME = "safe_body"
KEY_FINGERPRINT_MATCHER_NAME = "key_fingerprint"
KEY_FINGERPRINT_HEADER = "x-litellm-key-fp"

# Intentionally narrower than ``FILTERED_REQUEST_HEADERS``: AWS SigV4 headers
# carry secrets but their values rotate on every call, so fingerprinting them
# would defeat caching.
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

# Tiny placeholder used to replace base64 image payloads in cassettes.
# Decodes to b"test" — short, valid base64 so test code that decodes
# the field still succeeds.
VCR_IMAGE_B64_PLACEHOLDER = "dGVzdA=="

# Fixed boundary substituted into multipart request bodies so the
# ``safe_body`` matcher sees the same bytes across record and replay.
# httpx generates a fresh random boundary per request via os.urandom,
# which otherwise turns every multipart cassette into a permanent miss.
VCR_FIXED_MULTIPART_BOUNDARY = "vcr-static-boundary"


def _scrub_response(response):
    if not isinstance(response, dict):
        return response
    headers = response.get("headers") or {}
    if isinstance(headers, dict):
        for header in list(headers):
            if header.lower() in FILTERED_RESPONSE_HEADERS:
                headers.pop(header, None)
    return response


def _replace_b64_json_in_place(obj) -> bool:
    """Recursively replace ``b64_json`` string values in a JSON tree.

    Returns ``True`` if any value was rewritten. The check on the
    existing value's length keeps the function idempotent — once a
    value has been swapped to the placeholder, subsequent invocations
    are no-ops.
    """
    changed = False
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (
                key == "b64_json"
                and isinstance(value, str)
                and len(value) > len(VCR_IMAGE_B64_PLACEHOLDER)
            ):
                obj[key] = VCR_IMAGE_B64_PLACEHOLDER
                changed = True
            elif _replace_b64_json_in_place(value):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if _replace_b64_json_in_place(item):
                changed = True
    return changed


def _strip_image_b64_payloads(response):
    """Replace ``b64_json`` payloads in image-gen responses before save.

    Image-edit and image-generation responses carry the full base64
    PNG/JPEG (1-10+ MB) in ``data[*].b64_json``. The image_gen tests
    only assert response shape — the field decodes, schema validates —
    they never inspect pixel content. Swapping to a 4-byte placeholder
    preserves all those checks while shrinking cassettes by ~99%.
    """
    if not isinstance(response, dict):
        return response
    body = response.get("body")
    if not isinstance(body, dict):
        return response
    raw = body.get("string")
    if raw is None:
        return response

    if isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            return response
        was_bytes = True
    elif isinstance(raw, str):
        text = raw
        was_bytes = False
    else:
        return response

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return response

    if not _replace_b64_json_in_place(payload):
        return response

    new_text = json.dumps(payload, separators=(",", ":"))
    body["string"] = new_text.encode("utf-8") if was_bytes else new_text

    headers = response.get("headers")
    if isinstance(headers, dict):
        new_len_value = str(len(new_text.encode("utf-8")))
        for key in list(headers):
            if str(key).lower() == "content-length":
                value = headers[key]
                headers[key] = (
                    [new_len_value] if isinstance(value, list) else new_len_value
                )
    return response


def _before_record_response(response):
    return filter_non_2xx_response(_scrub_response(_strip_image_b64_payloads(response)))


def _safe_body_matcher(r1, r2) -> None:
    """Compare request bodies as bytes; never invokes ``json.loads``.

    vcrpy's stock ``body`` matcher unconditionally json-decodes
    ``application/json`` payloads, which raises on JSON Lines bodies
    (e.g. the Bedrock batch S3 PUT) before it can return "no match".
    This matcher is strictly more conservative — the only equivalence
    it gives up vs. the default is "JSON key order doesn't matter".
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
    if n1 is not None and n2 is not None and n1 == n2:
        return
    raise AssertionError("request bodies differ")


def _iter_header_values(headers, name: str):
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


def _normalize_multipart_boundary(request) -> None:
    """Rewrite random multipart boundaries to a fixed string in-place.

    httpx generates a fresh ``boundary=<random hex>`` for every
    multipart request via ``os.urandom``. Without normalization, the
    request body bytes differ across runs even when everything else is
    identical, the ``safe_body`` matcher misses, and the persister
    keeps appending new episodes until ``MAX_EPISODES_PER_CASSETTE``
    refuses the save — leaving audio-transcription tests effectively
    unmocked. Replacing the boundary in both the Content-Type header
    and the body bytes makes the request deterministic.

    Idempotent — vcrpy invokes this hook multiple times per request,
    so the second invocation sees ``boundary=vcr-static-boundary``
    already and short-circuits.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return

    content_type_key = None
    content_type_value = None
    try:
        for key in list(headers.keys()):
            if str(key).lower() == "content-type":
                content_type_key = key
                value = headers[key]
                content_type_value = value if isinstance(value, str) else str(value)
                break
    except AttributeError:
        return

    if not content_type_value or "multipart/" not in content_type_value.lower():
        return

    fixed_param = f"boundary={VCR_FIXED_MULTIPART_BOUNDARY}"
    if fixed_param in content_type_value:
        return

    match = re.search(r"boundary=([^\s;]+)", content_type_value)
    if not match:
        return
    current_boundary = match.group(1).strip('"')
    if current_boundary == VCR_FIXED_MULTIPART_BOUNDARY:
        return

    try:
        headers[content_type_key] = content_type_value.replace(
            match.group(0), fixed_param
        )
    except (TypeError, AttributeError):
        return

    body = getattr(request, "body", None)
    if body is None:
        return

    if isinstance(body, (bytes, bytearray)):
        try:
            new_body = bytes(body).replace(
                current_boundary.encode("utf-8"),
                VCR_FIXED_MULTIPART_BOUNDARY.encode("utf-8"),
            )
        except (TypeError, ValueError):
            return
    elif isinstance(body, str):
        new_body = body.replace(current_boundary, VCR_FIXED_MULTIPART_BOUNDARY)
    else:
        return

    try:
        request.body = new_body
    except (AttributeError, TypeError):
        pass


def _before_record_request(request):
    """Fingerprint API keys, scrub them, and normalize multipart boundaries.

    Order matters in two ways:

    1. vcrpy's ``filter_headers`` config option runs *before*
       ``before_record_request``, so the auth-header scrubbing has to
       live here; otherwise the secret would already be gone when we
       try to hash it.
    2. vcrpy invokes this hook more than once per request (e.g.
       ``can_play_response_for`` calls it, then ``_responses`` calls it
       again on the result). The second invocation sees a request whose
       auth headers we already stripped, so re-hashing would yield
       ``"no-key"`` and the stored vs. incoming fingerprints would
       diverge. Skip the recompute when the header is already set so
       this hook is idempotent. The boundary normalizer is also
       idempotent for the same reason.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return request
    if not any(_iter_header_values(headers, KEY_FINGERPRINT_HEADER)):
        fingerprint = _compute_key_fingerprint(request)
        try:
            headers[KEY_FINGERPRINT_HEADER] = fingerprint
        except (TypeError, AttributeError):
            pass
    _strip_headers(headers, FILTERED_REQUEST_HEADERS)
    _normalize_multipart_boundary(request)
    return request


def _key_fingerprint_matcher(r1, r2) -> None:
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
    if os.environ.get("LITELLM_VCR_DISABLE") == "1":
        return True
    return not os.environ.get("CASSETTE_REDIS_URL")


_atexit_banner_registered = False


def _print_atexit_banner() -> None:
    """Fallback for conftests that don't wire up ``pytest_terminal_summary``."""
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
    """Call from ``pytest_recording_configure(config, vcr)`` in each conftest."""
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

    ``skip_files`` are basenames to leave un-marked (e.g. respx-using
    files, since respx and vcrpy both patch the httpx transport).
    ``skip_nodeid_suffixes`` are node-id suffixes for individual tests
    that depend on live cross-call provider state.
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
    """Call from the post-yield section of an autouse fixture per test."""
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
    """Call from ``pytest_terminal_summary``. No-op on xdist workers."""
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


class VerboseReporterState:
    """Holds the controller's plugin manager / terminal reporter so each
    consuming conftest can print ``[VCR HIT|MISS|...]`` lines next to tests."""

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
