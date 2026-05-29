"""Shared VCR (Redis-backed) plumbing imported by per-directory conftests.

See ``tests/llm_translation/Readme.md`` for the full design and
``tests/llm_translation/conftest.py`` for the reference wiring."""

from __future__ import annotations

import ast
import atexit
import hashlib
import json
import os
import re
import socket
import sys
import threading
from collections import defaultdict
from typing import Iterable

import pytest
import vcr.matchers as _vcr_matchers

from tests._vcr_redis_persister import (
    MAX_EPISODES_PER_CASSETTE,
    VCR_VERBOSE_ENV,
    cassette_cache_capacity_snapshot,
    cassette_cache_health,
    filter_non_2xx_response,
    make_redis_persister,
    mark_test_outcome_for_cassette,
    patch_vcrpy_aiohttp_record_path,
)

# Force litellm to use its bundled model-cost-map backup instead of fetching it
# from raw.githubusercontent.com on import. Several VCR conftests reload litellm
# in an autouse fixture (``importlib.reload(litellm)``); ``litellm.__init__``
# calls ``get_model_cost_map()`` which issues a live ``httpx.get`` unless this is
# set. While a cassette is active that fetch gets *recorded* as an extra episode
# (it was present in ~710 of ~1900 cached cassettes). For tests that then skip,
# it is the only recorded episode, so the persister refuses to save it (skipped
# tests don't persist) and the test re-records it live and is classified
# MISS:NOT_PERSISTED on every run. Pinning to the local backup removes the
# network call entirely, so skip tests record nothing (NOOP) and passing tests
# stop carrying a volatile github episode. This matches the established idiom in
# the unit-test suite, which sets the same flag (see e.g.
# tests/test_litellm/test_cost_calculator.py). ``setdefault`` so an explicit
# override still wins.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

CASSETTE_CACHE_HIGH_WATER_FRACTION = 0.85


SAFE_BODY_MATCHER_NAME = "safe_body"
KEY_FINGERPRINT_MATCHER_NAME = "key_fingerprint"
TOLERANT_QUERY_MATCHER_NAME = "tolerant_query"
KEY_FINGERPRINT_HEADER = "x-litellm-key-fp"

VCR_DIAG_DIR_ENV = "LITELLM_VCR_DIAG_DIR"
VCR_DIAG_DIR_DEFAULT = "test-results/vcr-diagnostics"


def _vcr_diag_dir() -> str:
    return os.environ.get(VCR_DIAG_DIR_ENV) or VCR_DIAG_DIR_DEFAULT


def vcr_diag_write_line(msg: str) -> None:
    try:
        directory = _vcr_diag_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{os.getpid()}.log")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(msg.rstrip("\n") + "\n")
    except OSError:
        pass


def reset_vcr_diag_dir() -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    directory = _vcr_diag_dir()
    if not os.path.isdir(directory):
        return
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if name.endswith(".log"):
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass


# CircleCI truncates a step's retrievable output to the last ~400 KB. The
# diagnostic log is emitted right *before* the final pytest summary line but
# *after* the VCR CLASSIFICATION SUMMARY, so an unbounded dump (the body/key
# matchers log one block per *episode comparison*, even on an eventual HIT)
# pushes the classification summary out of the retrievable window and makes
# misses impossible to read in CI. Dedupe identical blocks (the same mismatch
# is logged against every non-matching episode) and cap the total emitted size
# so the summary always survives.
VCR_DIAG_EMIT_MAX_LINES = 400


def emit_vcr_diagnostic_log(terminalreporter) -> None:
    directory = _vcr_diag_dir()
    if not os.path.isdir(directory):
        return
    try:
        files = sorted(f for f in os.listdir(directory) if f.endswith(".log"))
    except OSError:
        return
    if not files:
        return

    # Collect every line, tagged by source file, deduplicating identical lines
    # (with an occurrence count) so the repeated per-episode mismatch blocks
    # collapse to one representative each.
    seen_counts: dict[str, int] = defaultdict(int)
    ordered: list[tuple[str, str]] = []  # (source_file, line)
    read_errors: list[str] = []
    for name in files:
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except OSError as exc:
            read_errors.append(
                f"  [failed to read {name}: {type(exc).__name__}: {exc}]"
            )
            continue
        for line in content.splitlines():
            if not line.strip():
                continue
            seen_counts[line] += 1
            if seen_counts[line] == 1:
                ordered.append((name, line))

    if not ordered and not read_errors:
        return

    terminalreporter.write_sep("=", "VCR DIAGNOSTIC LOG", bold=True)
    terminalreporter.write_line(
        f"  source dir: {directory}  (deduplicated; full log archived as a CI artifact)"
    )
    for line in read_errors:
        terminalreporter.write_line(line)

    emitted = 0
    last_source = None
    for name, line in ordered:
        if emitted >= VCR_DIAG_EMIT_MAX_LINES:
            terminalreporter.write_line(
                f"  ... {len(ordered) - emitted} more unique diagnostic line(s) "
                "suppressed to keep the classification summary retrievable in CI."
            )
            break
        if name != last_source:
            terminalreporter.write_sep("-", name, bold=False)
            last_source = name
        count = seen_counts.get(line, 1)
        suffix = f"  (x{count})" if count > 1 else ""
        terminalreporter.write_line(line + suffix)
        emitted += 1
    terminalreporter.write_sep("=", bold=True)


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


def pin_httpx_multipart_boundary(monkeypatch) -> None:
    try:
        import httpx._multipart as _httpx_multipart
    except ImportError:
        return

    _original_init = _httpx_multipart.MultipartStream.__init__

    def _init_with_fixed_boundary(self, data, files, boundary=None, **kwargs):
        if boundary is None:
            boundary = VCR_FIXED_MULTIPART_BOUNDARY.encode("ascii")
        return _original_init(self, data=data, files=files, boundary=boundary, **kwargs)

    monkeypatch.setattr(
        _httpx_multipart.MultipartStream, "__init__", _init_with_fixed_boundary
    )


@pytest.fixture(scope="session", autouse=True)
def _pin_multipart_boundary():
    monkeypatch = pytest.MonkeyPatch()
    pin_httpx_multipart_boundary(monkeypatch)
    yield
    monkeypatch.undo()


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
        vcr_diag_write_line(
            f"[vcr-strip-b64] response is {type(response).__name__!r}, not "
            "dict; skipping b64 scrub"
        )
        return response
    body = response.get("body")
    if not isinstance(body, dict):
        vcr_diag_write_line(
            f"[vcr-strip-b64] response['body'] is {type(body).__name__!r}, "
            "not dict; skipping b64 scrub"
        )
        return response
    raw = body.get("string")
    if raw is None:
        return response

    if isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            vcr_diag_write_line(
                "[vcr-strip-b64] response body bytes are not valid UTF-8; "
                "skipping b64 scrub"
            )
            return response
        was_bytes = True
    elif isinstance(raw, str):
        text = raw
        was_bytes = False
    else:
        vcr_diag_write_line(
            f"[vcr-strip-b64] response['body']['string'] is "
            f"{type(raw).__name__!r}, not bytes/str; skipping b64 scrub"
        )
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


def _canonical_body(request) -> tuple[bytes, str]:
    pre_type = type(getattr(request, "body", None)).__name__
    _materialize_iterable_body(request)
    body = getattr(request, "body", None)
    if body is None:
        return b"", pre_type
    if isinstance(body, bytes):
        return body, pre_type
    if isinstance(body, bytearray):
        return bytes(body), pre_type
    if isinstance(body, str):
        return body.encode("utf-8"), pre_type
    if isinstance(body, (dict, list)):
        try:
            return (
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                pre_type,
            )
        except (TypeError, ValueError):
            pass
    method = getattr(request, "method", "?")
    uri = getattr(request, "uri", getattr(request, "url", "?"))
    vcr_diag_write_line(
        f"[vcr-canonical-body] FALLBACK: {method} {uri} body type "
        f"{type(body).__name__!r} not coerced to bytes; comparing as b''"
    )
    return b"", pre_type


# ---------------------------------------------------------------------------
# Volatile-token body normalization (compare-time only).
#
# Many tests append a cache-buster to the request body so the *live* call
# isn't served from an upstream prompt/response cache during recording:
# ``f"...{time.time()}"``, ``f"...{uuid.uuid4()}"``. LiteLLM's own
# observability payloads (langfuse/otel) likewise carry per-call UUIDs and
# ISO-8601 timestamps. None of that affects what the test asserts (response
# shape, cost, caching behaviour), but it makes the request body differ on
# every run, so vcrpy never matches and the cassette keeps appending episodes
# until it overflows ``MAX_EPISODES_PER_CASSETTE`` and re-records live forever.
#
# We canonicalize these volatile substrings to fixed placeholders *only for
# matching* (in ``_safe_body_matcher``), never in what we store — so the
# cassette on disk keeps the real bytes for debuggability, and the
# normalization is applied symmetrically to both the incoming and the stored
# request. Because it's symmetric and compare-time, it can never mask a
# response-level discrepancy; it only changes which recorded episode is
# selected. This mirrors the existing SigV4 / multipart-boundary / b64-image
# normalizations already in this module, and means the already-bloated
# cassettes start replaying immediately without a flush + re-record.
_VCR_UUID_RE = re.compile(
    rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# ISO-8601 timestamps, e.g. ``2026-05-25T03:40:37.262045Z`` /
# ``2026-05-25T03:40:37+00:00``.
_VCR_ISO_TS_RE = re.compile(
    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
# Unix epoch as 13-digit milliseconds, then 10-digit ``time.time()`` float,
# then 10-digit integer seconds. Anchored to ``1`` + 9/12 digits, which keeps
# them inside the 2001-2033 / 2001-2033 epoch windows and avoids matching
# ordinary identifiers. Order matters: the longer/float forms are substituted
# before the bare-integer form so the integer rule can't bite off a prefix.
_VCR_UNIX_MS_RE = re.compile(rb"(?<![\d.])1[0-9]{12}(?![\d.])")
_VCR_UNIX_FLOAT_RE = re.compile(rb"(?<![\d.])1[0-9]{9}\.[0-9]+")
_VCR_UNIX_INT_RE = re.compile(rb"(?<![\d.])1[0-9]{9}(?![\d.])")


def _normalize_volatile_tokens(body: bytes) -> bytes:
    """Replace per-run cache-busters (UUIDs / timestamps) with placeholders.

    Compare-time only — see the module note above. Returns ``body`` unchanged
    when it contains none of these patterns, so deterministic requests are
    unaffected.
    """
    if not body:
        return body
    body = _VCR_UUID_RE.sub(b"<vcr-uuid>", body)
    body = _VCR_ISO_TS_RE.sub(b"<vcr-iso-ts>", body)
    body = _VCR_UNIX_MS_RE.sub(b"<vcr-unix-ms>", body)
    body = _VCR_UNIX_FLOAT_RE.sub(b"<vcr-unix-float>", body)
    body = _VCR_UNIX_INT_RE.sub(b"<vcr-unix-int>", body)
    return body


# Hosts whose request body is a rotating credential exchange (a freshly signed
# JWT ``assertion=...`` or refresh-token grant). The body changes on every run
# and carries no information the test asserts on, so matching on
# method+scheme+host+port+path+query is sufficient — skip the body comparison.
_CREDENTIAL_EXCHANGE_HOSTS = (
    "oauth2.googleapis.com",
    "sts.googleapis.com",
    "accounts.google.com",
    "metadata.google.internal",
    "169.254.169.254",
)


def _request_host(request) -> str:
    uri = getattr(request, "uri", None) or getattr(request, "url", "") or ""
    uri = str(uri)
    if "//" not in uri:
        return ""
    rest = uri.split("//", 1)[1]
    return rest.split("/", 1)[0].split("@")[-1].split(":")[0].lower()


def _is_credential_exchange_request(request) -> bool:
    return _request_host(request) in _CREDENTIAL_EXCHANGE_HOSTS


# Observability / telemetry backends LiteLLM logs to. A telemetry export is a
# snapshot of the *whole* call — fresh span/trace UUIDs, ISO-8601 timestamps,
# durations, token costs, the LiteLLM build SHA (``release``), and the recorded
# LLM response content — and tests often round-trip a fresh ``trace_id`` back
# through the backend's query API to verify logging happened. None of that is
# reproducible under deterministic replay, and none of it is what the test
# asserts on (it checks redaction / presence, or a locally-computed trace id).
# So for these hosts we match on method+scheme+host+port+path only: the
# expensive LLM call still matches normally and stays cached, while the cheap
# telemetry POST/GET replays from the recorded response. This is why the body
# and query matchers below both short-circuit for telemetry hosts.
_TELEMETRY_HOST_SUFFIXES = (
    "langfuse.com",
    "arize.com",
    "phoenix.arize.com",
    "traceloop.com",
    "braintrust.dev",
    "comet.com",
    "wandb.ai",
    "honeycomb.io",
    "signoz.io",
)


def _is_telemetry_request(request) -> bool:
    host = _request_host(request)
    if not host:
        return False
    return any(host == s or host.endswith("." + s) for s in _TELEMETRY_HOST_SUFFIXES)


# Nodeid of the test currently executing, set per-test by
# ``install_live_call_probe`` (runs in the autouse gate at setup). Used to
# decide whether an incidental telemetry POST should be recorded — see
# ``_should_drop_telemetry_record``. xdist workers are separate processes and
# tests run sequentially within a worker, so a plain module global is safe.
_current_test_nodeid: str = ""

# Test files/dirs that legitimately record & replay telemetry HTTP (they assert
# on the outgoing observability payload or query the backend back). Identified
# by a substring of the test path. Everything else is treated as a non-telemetry
# test for which a telemetry call is incidental leakage (see below).
_TELEMETRY_TEST_PATH_MARKERS = (
    "langfuse",
    "arize",
    "phoenix",
    "traceloop",
    "braintrust",
    "comet",
    "wandb",
    "honeycomb",
    "signoz",
    "otel",
    "opentelemetry",
    "telemetry",
    "observability",
    "logging",  # tests/logging_callback_tests, logging_testing dirs
)


def _current_test_records_telemetry() -> bool:
    nodeid = _current_test_nodeid.lower()
    return any(marker in nodeid for marker in _TELEMETRY_TEST_PATH_MARKERS)


# Test paths that legitimately RECORD AND REPLAY a telemetry *export* POST and
# assert on its response. Only the pass-through proxy test does this: it
# forwards a client POST to Langfuse's ``/api/public/ingestion`` and asserts the
# upstream multi-status (207) it replays from the cassette. Every other
# telemetry test either mocks the export client and asserts on the mock (the
# langfuse e2e suite) or asserts on a read-back GET / an in-memory span exporter
# — for those the export POST is fire-and-forget and must not be recorded (see
# ``_should_drop_telemetry_record``).
_TELEMETRY_EXPORT_REPLAY_TEST_MARKERS = ("pass_through",)


def _current_test_replays_telemetry_export() -> bool:
    nodeid = _current_test_nodeid.lower()
    return any(m in nodeid for m in _TELEMETRY_EXPORT_REPLAY_TEST_MARKERS)


def _is_telemetry_export_request(request) -> bool:
    """A telemetry *export* — a span/trace/event ingestion call, always a POST
    to an observability host. Read-backs (verifying a trace landed) are GETs."""
    if not _is_telemetry_request(request):
        return False
    return str(getattr(request, "method", "") or "").upper() == "POST"


# Thread-local "we are inside Cassette._load" flag. vcrpy's ``Cassette._load``
# replays each *stored* interaction through ``Cassette.append``, which runs
# ``before_record_request`` on it; a ``None`` return there silently drops the
# stored episode. ``_should_drop_telemetry_record`` must therefore NOT fire
# during load, or it would delete already-recorded telemetry episodes the
# instant a non-telemetry-named test (or the very first test in a worker, whose
# ``_current_test_nodeid`` is still empty) loads them — forcing an endless live
# re-record (a phantom MISS:RECORDED on a cassette that was present in Redis).
# The drop is only ever meant to stop *new* incidental telemetry from being
# recorded, never to filter the existing cassette on read. ``_load`` and its
# ``append`` calls run synchronously in one thread, so a thread-local correctly
# scopes the guard and never masks a concurrent background-flush record.
_vcr_load_guard = threading.local()


def _vcr_load_in_progress() -> bool:
    return getattr(_vcr_load_guard, "active", False)


def patch_vcrpy_cassette_load_guard() -> None:
    """Wrap ``Cassette._load`` so ``_should_drop_telemetry_record`` is inert
    while stored episodes are being replayed into the in-memory cassette."""
    import vcr.cassette as _cassette_mod

    if getattr(_cassette_mod.Cassette._load, "_litellm_load_guarded", False):
        return
    _orig_load = _cassette_mod.Cassette._load

    def _guarded_load(self):
        _vcr_load_guard.active = True
        try:
            return _orig_load(self)
        finally:
            _vcr_load_guard.active = False

    _guarded_load._litellm_load_guarded = True
    _cassette_mod.Cassette._load = _guarded_load


def _should_drop_telemetry_record(request) -> bool:
    """Whether to refuse to record this request into the active cassette.

    Several test modules set ``litellm.success_callback = ["langfuse"]`` (and
    similar) at *import* time, which globally enables observability logging for
    the whole worker. Unrelated tests then emit telemetry whose async flush
    (litellm's background logging worker) lands in a *later* test's VCR window
    and gets saved as a spurious episode — a non-deterministic MISS:RECORDED on
    whichever test happened to be active (observed on
    ``test_lowest_latency_routing_buffer`` carrying a Langfuse batch from an
    unrelated completion). Refusing to record telemetry for non-telemetry tests
    makes the leak a harmless live fire-and-forget call instead (telemetry hosts
    are not in ``_LIVE_CALL_HOST_SUFFIXES``, so the probe doesn't flag it, and
    vcrpy treats a ``None`` from ``before_record_request`` as "don't record" and
    "can't replay" → the request passes through live and is never stored).
    Tests that actually assert on telemetry keep recording it.

    Crucially, this never fires while ``Cassette._load`` is replaying stored
    interactions (see ``_vcr_load_in_progress``): dropping there would delete an
    already-recorded telemetry episode on read and force a live re-record.

    The async-flush leak also rotates *within* the telemetry test set: litellm's
    observability loggers flush on a background thread, so an export POST
    scheduled by one telemetry test fires mid-way through a *later*
    telemetry-named test (after that test's own ``httpx`` mock has exited) and
    is recorded as a phantom episode — a non-deterministic MISS:RECORDED /
    PARTIAL that lands on a different telemetry test from run to run. Telemetry
    *export* POSTs are fire-and-forget; no test asserts on a recorded export
    response except the pass-through proxy test (which forwards to Langfuse
    ingestion and replays its 207). So drop incidental export POSTs everywhere
    else too — dropping returns ``None`` (live fire-and-forget, never stored),
    which can only turn a phantom miss into a harmless live call, never the
    reverse. Recorded read-back GETs that telemetry tests assert on are matched
    by method and so are left untouched.
    """
    if _vcr_load_in_progress():
        return False
    if not _is_telemetry_request(request):
        return False
    if (
        _is_telemetry_export_request(request)
        and not _current_test_replays_telemetry_export()
    ):
        return True
    return not _current_test_records_telemetry()


def _should_passthrough_credential_exchange(request) -> bool:
    """Force the Google OAuth2/STS token mint to run live, never from cassette.

    The mint returns a short-lived ``ya29.*`` access token. Recording it lets a
    *stale* token replay on a later run; litellm caches it (the recorded
    ``expires_in`` keeps ``credentials.expired`` False, so it is never
    refreshed) and sends it to a live Vertex/Gemini endpoint, which rejects it
    with ``ACCESS_TOKEN_EXPIRED``. The token body carries nothing a test asserts
    on, so always mint it live: returning ``None`` from ``before_record_request``
    makes vcrpy neither store nor replay the call. Inert during
    ``Cassette._load`` for the same reason as ``_should_drop_telemetry_record``.
    """
    if _vcr_load_in_progress():
        return False
    return _is_credential_exchange_request(request)


# Google APIs (Vertex AI, Gemini, OAuth2/STS). Auth is a ``ya29.*`` OAuth2
# access token minted fresh on every run, so the per-request key fingerprint
# rotates and never matches a recording. The logical credential — the GCP
# project — is part of the matched URL path (``/projects/<project>/...``), so
# skipping the fingerprint comparison for these hosts keeps cache isolation by
# project while letting the existing recordings replay without a re-record.
# (We also collapse ``ya29.*`` tokens to one marker in ``_stable_key_value`` so
# *new* recordings store a stable fingerprint; this matcher relaxation is what
# rescues the cassettes already recorded under the old per-token fingerprints.)
_GOOGLE_HOST_SUFFIXES = (
    "googleapis.com",
    "google.internal",
)


def _is_google_host_request(request) -> bool:
    host = _request_host(request)
    if not host:
        return False
    return any(host == s or host.endswith("." + s) for s in _GOOGLE_HOST_SUFFIXES)


def _safe_body_matcher(r1, r2) -> None:
    """Compare request bodies as bytes; never invokes ``json.loads``.

    vcrpy's stock ``body`` matcher unconditionally json-decodes
    ``application/json`` payloads, which raises on JSON Lines bodies
    (e.g. the Bedrock batch S3 PUT) before it can return "no match".
    This matcher is strictly more conservative — the only equivalence
    it gives up vs. the default is "JSON key order doesn't matter".

    Two compare-time relaxations layer on top, both symmetric so they can
    never hide a response-level discrepancy:

    * Requests to a rotating-credential-exchange host (Google OAuth2/STS
      token endpoints) skip the body comparison — the signed-JWT body
      changes every run. The host matcher still gates the overall match.
    * Volatile cache-buster tokens (UUIDs / epoch timestamps) are
      canonicalized away via ``_normalize_volatile_tokens``.
    """
    if _is_credential_exchange_request(r1) or _is_telemetry_request(r1):
        return
    body1, pre1 = _canonical_body(r1)
    body2, pre2 = _canonical_body(r2)
    if body1 == body2:
        return
    if _normalize_volatile_tokens(body1) == _normalize_volatile_tokens(body2):
        return
    _emit_body_mismatch_diagnostic(r1, r2, body1, body2, pre1, pre2)
    raise AssertionError("request bodies differ")


def _emit_body_mismatch_diagnostic(r1, r2, body1, body2, pre1, pre2) -> None:
    def _describe(label, asbytes, pre_type):
        return (
            f"  {label}: pre_canonical_type={pre_type!r} length={len(asbytes)} "
            f"sha256={hashlib.sha256(asbytes).hexdigest()} "
            f"preview={asbytes[:120]!r}"
        )

    method_a = getattr(r1, "method", "?")
    method_b = getattr(r2, "method", "?")
    url_a = getattr(r1, "uri", getattr(r1, "url", "?"))
    url_b = getattr(r2, "uri", getattr(r2, "url", "?"))
    lines = [
        "[vcr-safe-body-matcher] request body mismatch",
        f"  request[a]: {method_a} {url_a}",
        f"  request[b]: {method_b} {url_b}",
        _describe("body[a]", body1, pre1),
        _describe("body[b]", body2, pre2),
    ]
    if body1 != body2:
        offset = next(
            (i for i in range(min(len(body1), len(body2))) if body1[i] != body2[i]),
            min(len(body1), len(body2)),
        )
        start = max(0, offset - 100)
        end_a = min(len(body1), offset + 100)
        end_b = min(len(body2), offset + 100)
        lines.append(f"  first divergent byte offset: {offset}")
        lines.append(f"  window[a] @ {start}..{end_a}: {body1[start:end_a]!r}")
        lines.append(f"  window[b] @ {start}..{end_b}: {body2[start:end_b]!r}")
    vcr_diag_write_line("\n".join(lines))


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


_AWS_SIGV4_CREDENTIAL_RE = re.compile(
    r"AWS4-HMAC-SHA256\s+Credential=([^/\s,]+)/", re.IGNORECASE
)

# Google OAuth2 access tokens always start with ``ya29.`` regardless of how
# they were minted (service account, metadata server, impersonation).
_GOOGLE_OAUTH_BEARER_RE = re.compile(r"^Bearer\s+ya29\.", re.IGNORECASE)


def _stable_key_value(header_name: str, raw: str) -> str:
    """Return a *stable* identifier for a credential header.

    For Bearer / API-key headers the entire value is stable across calls,
    so we hash it as-is. For AWS SigV4 ``Authorization`` headers, only
    the access-key portion of ``Credential=AKIA.../<DATE>/...`` is stable
    — date, region, signed headers, and signature all rotate per request,
    so hashing the full value would push every Bedrock request into a new
    cassette episode. Extract just the access-key id when present.
    """
    if header_name.lower() != "authorization":
        return raw
    match = _AWS_SIGV4_CREDENTIAL_RE.search(raw)
    if match:
        return f"aws-sigv4:{match.group(1)}"
    # Google OAuth2 access tokens (``ya29.*``) are minted fresh from the
    # service-account credentials on every run, so hashing the raw token
    # would push every Vertex/Gemini request into a new cassette episode —
    # exactly the SigV4 failure mode above. The logical credential (the GCP
    # project) is already part of the matched URL path, so collapse all such
    # tokens to one stable marker.
    if _GOOGLE_OAUTH_BEARER_RE.match(raw):
        return "google-oauth2"
    return raw


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
            stable = _stable_key_value(header_name, text)
            parts.append(f"{header_name}={stable}")
    if not parts:
        method = getattr(request, "method", "?")
        uri = getattr(request, "uri", getattr(request, "url", "?"))
        vcr_diag_write_line(
            f"[vcr-key-fingerprint] no API key header found on {method} "
            f"{uri}; falling back to 'no-key'. If this request should have "
            "carried auth, something earlier in the pipeline stripped it."
        )
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
        vcr_diag_write_line(
            f"[vcr-multipart-normalize] body normalization SKIPPED: "
            f"body type {type(body).__name__!r} is not bytes/bytearray/str. "
            f"content-type={content_type_value!r}. "
            f"Recorded body will retain the random boundary substring "
            f"and the safe_body matcher will miss on the next run."
        )
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
    # Refuse to record incidental telemetry leaked from a globally-enabled
    # observability callback into a non-telemetry test (see
    # ``_should_drop_telemetry_record``). Returning ``None`` tells vcrpy not to
    # store the interaction; the request passes through live (fire-and-forget).
    if _should_drop_telemetry_record(request):
        return None
    if _should_passthrough_credential_exchange(request):
        return None
    headers = getattr(request, "headers", None)
    if headers is None:
        return request
    _materialize_iterable_body(request)
    if not any(_iter_header_values(headers, KEY_FINGERPRINT_HEADER)):
        fingerprint = _compute_key_fingerprint(request)
        try:
            headers[KEY_FINGERPRINT_HEADER] = fingerprint
        except (TypeError, AttributeError):
            pass
    _strip_headers(headers, FILTERED_REQUEST_HEADERS)
    _normalize_multipart_boundary(request)
    return request


def _materialize_iterable_body(request) -> None:
    body = getattr(request, "body", None)
    if body is None or isinstance(body, (bytes, bytearray, str)):
        return
    if not hasattr(body, "__next__"):
        return
    try:
        chunks = list(body)
    except TypeError:
        return

    out = _coalesce_chunks_to_bytes(chunks)
    if out is None:
        method = getattr(request, "method", "?")
        uri = getattr(request, "uri", getattr(request, "url", "?"))
        first_type = type(chunks[0]).__name__ if chunks else "empty"
        vcr_diag_write_line(
            f"[vcr-materialize] FALLBACK: {method} {uri} chunk type "
            f"{first_type!r} not coerced to bytes; storing b''"
        )
        out = b""

    try:
        request.body = out
    except (AttributeError, TypeError):
        pass

    for attr in ("_was_iter", "_was_file"):
        try:
            setattr(request, attr, False)
        except (AttributeError, TypeError):
            pass


def _coalesce_chunks_to_bytes(chunks):
    if not chunks:
        return b""
    first = chunks[0]
    try:
        if isinstance(first, int):
            return bytes(chunks)
        if isinstance(first, (bytes, bytearray)):
            return b"".join(c if isinstance(c, bytes) else bytes(c) for c in chunks)
        if isinstance(first, str):
            return "".join(chunks).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return None


def _key_fingerprint_matcher(r1, r2) -> None:
    # Google OAuth2 access tokens rotate every run; the project in the URL
    # path (matched separately) is the stable credential identity, so skip the
    # fingerprint comparison for Google hosts. See ``_is_google_host_request``.
    if _is_google_host_request(r1):
        return

    def _fp(req):
        for value in _iter_header_values(
            getattr(req, "headers", None), KEY_FINGERPRINT_HEADER
        ):
            if value is None:
                continue
            return value if isinstance(value, str) else str(value)
        return "no-key"

    fp1, fp2 = _fp(r1), _fp(r2)
    if fp1 != fp2:
        method_a = getattr(r1, "method", "?")
        method_b = getattr(r2, "method", "?")
        url_a = getattr(r1, "uri", getattr(r1, "url", "?"))
        url_b = getattr(r2, "uri", getattr(r2, "url", "?"))
        vcr_diag_write_line(
            "[vcr-key-fingerprint-matcher] API key fingerprints differ\n"
            f"  request[a]: {method_a} {url_a} fingerprint={fp1!r}\n"
            f"  request[b]: {method_b} {url_b} fingerprint={fp2!r}"
        )
        raise AssertionError("API key fingerprints differ")


def _tolerant_query_matcher(r1, r2) -> None:
    """vcrpy's ``query`` matcher, but tolerant of telemetry round-trips.

    Observability backends are queried back with a freshly-generated
    ``trace_id`` (e.g. ``GET /observations?traceId=litellm-test-<uuid>``).
    Comparing the query string would miss on every run. For telemetry hosts
    we skip the query comparison entirely (the host+path matchers still gate
    the match); every other host uses vcrpy's stock query matcher unchanged.
    """
    if _is_telemetry_request(r1):
        return
    _vcr_matchers.query(r1, r2)


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
            TOLERANT_QUERY_MATCHER_NAME,
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
    vcr.register_matcher(TOLERANT_QUERY_MATCHER_NAME, _tolerant_query_matcher)
    patch_vcrpy_aiohttp_record_path()
    patch_vcrpy_cassette_load_guard()
    global _atexit_banner_registered
    if not _atexit_banner_registered:
        atexit.register(_print_atexit_banner)
        _atexit_banner_registered = True


VCR_SKIP_REASON_USER_ATTR = "vcr_skip_reason"

# Marker reasons recorded per-item / per-test for the session summary.
SKIP_REASON_RESPX = "respx_conflict"
SKIP_REASON_RESPX_MODULE = "respx_conflict_module"
SKIP_REASON_INCOMPATIBLE = "incompatible"
SKIP_REASON_FILE_OPT_OUT = "file_opt_out"
SKIP_REASON_DISABLED = "disabled"
SKIP_REASON_PRE_MARKED = "already_marked"

# Hostnames we consider an "expensive live call" if a non-VCR-marked test
# happens to hit them. Localhost/redis/databases are explicitly excluded.
_LIVE_CALL_HOST_SUFFIXES = (
    ".openai.com",
    ".anthropic.com",
    ".vertexai.googleapis.com",
    ".aiplatform.googleapis.com",
    ".googleapis.com",
    ".x.ai",
    ".cohere.ai",
    ".cohere.com",
    ".voyageai.com",
    ".perplexity.ai",
    ".mistral.ai",
    ".groq.com",
    ".huggingface.co",
    ".azure.com",
    ".tavily.com",
    ".serper.dev",
    ".searchapi.io",
    ".firecrawl.dev",
    ".exa.ai",
)
_LIVE_CALL_LOCAL_PREFIXES = (
    "127.",
    "localhost",
    "::1",
    "0.0.0.0",
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "192.168.",
)


class _RespxUsageVisitor(ast.NodeVisitor):
    """AST visitor that flags real respx wiring in a test module.

    Substring scans of the source text are unreliable: a comment like
    ``# Previously used respx.mock`` or a docstring referencing respx
    would falsely flag the module. We only count:

    * ``@pytest.mark.respx`` / ``@respx.mock`` decorators
    * ``with respx.mock(): ...`` context managers
    * ``respx.mock(...)`` / ``respx.mock`` attribute access
    * function parameters / fixture arguments named ``respx_mock``
    """

    def __init__(self) -> None:
        self.uses_respx = False

    def _decorator_is_respx(self, dec: ast.expr) -> bool:
        # ``@respx.mock`` (Attribute) or ``@respx.mock(...)`` (Call wrapping Attribute)
        if isinstance(dec, ast.Call):
            dec = dec.func
        if isinstance(dec, ast.Attribute):
            return (
                isinstance(dec.value, ast.Name)
                and dec.value.id == "respx"
                and dec.attr == "mock"
            )
        return False

    def _is_pytest_mark_respx(self, dec: ast.expr) -> bool:
        # ``@pytest.mark.respx`` or ``@pytest.mark.respx(...)``.
        if isinstance(dec, ast.Call):
            dec = dec.func
        if (
            isinstance(dec, ast.Attribute)
            and dec.attr == "respx"
            and isinstance(dec.value, ast.Attribute)
            and dec.value.attr == "mark"
            and isinstance(dec.value.value, ast.Name)
            and dec.value.value.id == "pytest"
        ):
            return True
        return False

    def _check_decorators(self, decs: list[ast.expr]) -> None:
        for d in decs:
            if self._decorator_is_respx(d) or self._is_pytest_mark_respx(d):
                self.uses_respx = True

    def _check_args(self, args: ast.arguments) -> None:
        # ``def test_foo(respx_mock): ...`` — pytest supplies the fixture
        # whenever the parameter name appears, regardless of marker.
        all_args = (
            list(args.args)
            + list(args.kwonlyargs)
            + (list(args.posonlyargs) if hasattr(args, "posonlyargs") else [])
        )
        for a in all_args:
            if a.arg == "respx_mock":
                self.uses_respx = True
                return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_decorators(node.decorator_list)
        self._check_args(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_decorators(node.decorator_list)
        self._check_args(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_decorators(node.decorator_list)
        self.generic_visit(node)

    def _is_respx_mock_attr(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "respx"
            and node.attr == "mock"
        )

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                ctx = ctx.func
            if self._is_respx_mock_attr(ctx):
                self.uses_respx = True
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                ctx = ctx.func
            if self._is_respx_mock_attr(ctx):
                self.uses_respx = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # ``respx.mock(...)`` invocation outside a ``with``/decorator —
        # e.g. ``mock = respx.mock()`` at module scope.
        if self._is_respx_mock_attr(node.func):
            self.uses_respx = True
        self.generic_visit(node)


def _module_uses_respx(item) -> bool:
    """Return True if the test's *module* actually wires up respx.

    Uses an ``ast`` walk (not substring matching) so comments and
    docstrings that mention respx don't count as real usage. A bare
    ``from respx import MockRouter`` import with no other respx
    references therefore won't flag the module — that's exactly the
    dead-import case this PR is trying to surface.
    """
    module = getattr(item, "module", None)
    src_file = getattr(module, "__file__", None) or str(getattr(item, "path", "") or "")
    if not src_file or not os.path.isfile(src_file):
        return False
    try:
        with open(src_file, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return False
    try:
        tree = ast.parse(src, filename=src_file)
    except SyntaxError:
        # If the test file itself is broken, fall back to "no respx" —
        # the test will fail collection on its own and we don't want
        # the auto-marker to mask that with a misleading skip reason.
        return False
    visitor = _RespxUsageVisitor()
    visitor.visit(tree)
    return visitor.uses_respx


def _item_uses_respx(item) -> bool:
    """Return True if *this specific item* will trigger respx.

    Two signals: the ``respx`` pytest marker, and the ``respx_mock``
    fixture appearing in the item's resolved fixture chain. Either alone
    causes vcrpy + respx to fight over the httpx transport.
    """
    if item.get_closest_marker("respx") is not None:
        return True
    fixturenames = getattr(item, "fixturenames", None) or ()
    if "respx_mock" in fixturenames:
        return True
    return False


# Cache the source-scan result so we don't reread each module per item.
_RESPX_MODULE_CACHE: dict[str, bool] = {}


def _module_path_uses_respx(item) -> bool:
    src_file = str(getattr(item, "path", "") or "")
    if not src_file:
        return False
    cached = _RESPX_MODULE_CACHE.get(src_file)
    if cached is not None:
        return cached
    result = _module_uses_respx(item)
    _RESPX_MODULE_CACHE[src_file] = result
    return result


def apply_vcr_auto_marker_to_items(
    items,
    *,
    skip_files: Iterable[str] = (),
    skip_nodeid_suffixes: Iterable[str] = (),
) -> None:
    """Auto-apply ``pytest.mark.vcr`` to collected items.

    Skip semantics (in priority order):

    1. ``vcr_disabled()`` — global env-var off-switch (``LITELLM_VCR_DISABLE=1``
       or no ``CASSETTE_REDIS_URL``).
    2. Item already carries ``@pytest.mark.vcr`` — leave it alone.
    3. Item triggers respx (per-item marker / fixture) — vcrpy and respx
       both patch the httpx transport so applying both makes one silently
       no-op. We tag the item ``vcr_skip_reason=respx_conflict``.
    4. Module wires up respx anywhere — even tests in the file that don't
       themselves use respx still inherit the patched transport when
       respx fixtures activate at session level. Tagged
       ``respx_conflict_module``.
    5. ``skip_files`` / ``skip_nodeid_suffixes`` opt-out lists from the
       caller — used for tests that observe live cross-call provider state
       (e.g. prompt-cache warmup) which deterministic replay can't model.
       Tagged ``incompatible``.

    Each skipped item gets a ``vcr_skip_reason`` attribute so the
    session-end summary can show why it isn't cached.
    """
    if vcr_disabled():
        for item in items:
            setattr(item, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_DISABLED)
        return
    skip_files = frozenset(skip_files)
    skip_nodeid_suffixes = tuple(skip_nodeid_suffixes)
    for item in items:
        if item.get_closest_marker("vcr") is not None:
            setattr(item, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_PRE_MARKED)
            continue
        if _item_uses_respx(item):
            setattr(item, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_RESPX)
            continue
        filename = os.path.basename(str(item.path))
        if filename in skip_files:
            # Trust the caller's opt-out, but split by reason: if the
            # module actually uses respx, label the conflict precisely so
            # the summary surfaces dead respx imports vs. real conflicts.
            if _module_path_uses_respx(item):
                setattr(item, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_RESPX_MODULE)
            else:
                setattr(item, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_FILE_OPT_OUT)
            continue
        if any(item.nodeid.endswith(suffix) for suffix in skip_nodeid_suffixes):
            setattr(item, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_INCOMPATIBLE)
            continue
        item.add_marker(pytest.mark.vcr)


# ---------------------------------------------------------------------------
# Per-test stats accumulator + verdict classification.
#
# The session-end summary needs richer signal than the line-level verdict:
# - which tests overflowed ``MAX_EPISODES_PER_CASSETTE`` (cassette refused
#   to save → live calls every CI run);
# - which tests fired live HTTP at a real LLM endpoint while VCR was not
#   active for them (genuine wasted spend, not just "test mocked elsewhere");
# - skip-reason buckets so we can tell respx-conflict from
#   incompatible-by-design from "module imports respx but never uses it".
# ---------------------------------------------------------------------------

# Verdict tags used in the per-test logline AND in the session summary
# breakdown.
VERDICT_HIT = "VCR HIT"
VERDICT_MISS_RECORDED = "VCR MISS:RECORDED"
VERDICT_MISS_OVERFLOW = "VCR MISS:OVERFLOW"
VERDICT_MISS_NOT_PERSISTED = "VCR MISS:NOT_PERSISTED"
VERDICT_PARTIAL = "VCR PARTIAL"
VERDICT_NOOP_NO_TRAFFIC = "VCR NOOP"
VERDICT_UNMARKED_LIVE_CALL = "VCR UNMARKED:LIVE_CALL"
VERDICT_UNMARKED_NO_TRAFFIC = "VCR UNMARKED:NO_TRAFFIC"
VERDICT_DISABLED = "VCR DISABLED"

# Per-session stats. Cleared by ``_reset_session_stats`` for unit tests.
_session_stats = {
    "verdict_counts": defaultdict(int),
    "overflow_tests": [],  # list of nodeids
    "unmarked_live_call_tests": [],  # list of (nodeid, hosts)
    "skip_reason_counts": defaultdict(int),
    "skip_reason_examples": defaultdict(list),
}


def _reset_session_stats() -> None:
    _session_stats["verdict_counts"].clear()
    _session_stats["overflow_tests"].clear()
    _session_stats["unmarked_live_call_tests"].clear()
    _session_stats["skip_reason_counts"].clear()
    _session_stats["skip_reason_examples"].clear()


# user_properties keys used to ship structured outcome data from xdist workers
# back to the controller. ``vcr_verdict`` is the human-readable line that
# ``VerboseReporterState.maybe_emit_verdict`` writes next to each test;
# ``vcr_outcome`` + ``vcr_recorded_by`` are the structured payload that
# ``aggregate_report_outcome`` folds into the controller's ``_session_stats``
# so the session-end summary actually has data in xdist mode.
_USER_PROP_VERDICT_LINE = "vcr_verdict"
_USER_PROP_OUTCOME = "vcr_outcome"
_USER_PROP_RECORDED_BY = "vcr_recorded_by"


def _emit_outcome_payload(
    node,
    verdict: str,
    *,
    skip_reason: str | None = None,
    live_call_hosts: Iterable[str] | None = None,
) -> None:
    """Stash a structured VCR outcome on a pytest node so the xdist
    controller can fold it into ``_session_stats``.

    On a worker, ``record_vcr_outcome`` has already updated the worker-local
    ``_session_stats`` — but in xdist mode that state lives in the worker
    process and never reaches the controller's ``pytest_terminal_summary``.
    We use the report's ``user_properties`` channel (which xdist round-trips
    back to the controller) to ship the outcome, and
    ``aggregate_report_outcome`` rebuilds the controller's stats from there.

    The recorder tags ``vcr_recorded_by`` with ``PYTEST_XDIST_WORKER`` so
    the controller can distinguish "recorded in this same main process —
    already counted" from "recorded in a worker — needs aggregation here".
    """
    node.user_properties.append(
        (
            _USER_PROP_OUTCOME,
            {
                "verdict": verdict,
                "skip_reason": skip_reason,
                "live_call_hosts": list(live_call_hosts) if live_call_hosts else [],
            },
        )
    )
    node.user_properties.append(
        (_USER_PROP_RECORDED_BY, os.environ.get("PYTEST_XDIST_WORKER", ""))
    )


def aggregate_report_outcome(report) -> None:
    """Fold a worker-produced VCR outcome into the controller's session stats.

    No-op outside the xdist controller path:

    * On a worker, ``_session_stats`` was already updated in-process by
      ``record_vcr_outcome`` — and the worker doesn't render the summary
      anyway, so there's nothing for us to aggregate.
    * In single-process mode, ``vcr_recorded_by`` is the empty string,
      which means the same process that ran the test is now handling the
      report — ``_session_stats`` already has the entry, double-counting
      would be a bug.
    * Only when ``vcr_recorded_by`` is a non-empty worker id (``"gw0"``
      etc.) do we know the controller's ``_session_stats`` is missing this
      test and needs the outcome folded in.
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    if report.when != "teardown":
        return

    recorded_by = next(
        (v for k, v in (report.user_properties or []) if k == _USER_PROP_RECORDED_BY),
        None,
    )
    if not recorded_by:
        return

    outcome = next(
        (v for k, v in (report.user_properties or []) if k == _USER_PROP_OUTCOME),
        None,
    )
    if not outcome:
        return

    verdict = outcome.get("verdict")
    if not verdict:
        return

    nodeid = report.nodeid
    _session_stats["verdict_counts"][verdict] += 1

    if verdict == VERDICT_MISS_OVERFLOW:
        _session_stats["overflow_tests"].append(nodeid)
    elif verdict == VERDICT_UNMARKED_LIVE_CALL:
        _session_stats["unmarked_live_call_tests"].append(
            (nodeid, list(outcome.get("live_call_hosts") or []))
        )

    skip_reason = outcome.get("skip_reason")
    if skip_reason:
        _session_stats["skip_reason_counts"][skip_reason] += 1
        examples = _session_stats["skip_reason_examples"][skip_reason]
        if len(examples) < 5:
            examples.append(nodeid)


def session_stats_snapshot() -> dict:
    """Read-only copy of the per-session VCR stats. Used by the summary."""
    return {
        "verdict_counts": dict(_session_stats["verdict_counts"]),
        "overflow_tests": list(_session_stats["overflow_tests"]),
        "unmarked_live_call_tests": list(_session_stats["unmarked_live_call_tests"]),
        "skip_reason_counts": dict(_session_stats["skip_reason_counts"]),
        "skip_reason_examples": {
            k: list(v) for k, v in _session_stats["skip_reason_examples"].items()
        },
    }


def _classify_marked_test(cassette) -> str:
    """Map cassette state → verdict tag for tests that *were* VCR-marked."""
    played = getattr(cassette, "play_count", 0) or 0
    dirty = getattr(cassette, "dirty", False)
    total = len(cassette) if hasattr(cassette, "__len__") else 0

    # "OVERFLOW" mirrors ``_RedisPersister.save_cassette``'s
    # ``> MAX_EPISODES_PER_CASSETTE`` guard. Cassettes that hit this
    # threshold are refused for save, so the test re-records live every
    # run. Only flag when ``dirty=True`` — if a cassette grew past the
    # cap historically but this run replayed it without adding new
    # episodes, the persister never tries to save (no recording
    # happened), so the cache state is stable and the next run will
    # replay too. Flagging that case as OVERFLOW would tag healthy
    # cached tests as cost leaks.
    if total > MAX_EPISODES_PER_CASSETTE and dirty:
        return VERDICT_MISS_OVERFLOW
    if played == 0 and not dirty:
        return VERDICT_NOOP_NO_TRAFFIC
    if played > 0 and not dirty:
        return VERDICT_HIT
    if played == 0 and dirty:
        return VERDICT_MISS_RECORDED
    return VERDICT_PARTIAL


def _format_verdict_line(verdict: str, cassette, extra: str = "") -> str:
    if cassette is None:
        return f"[{verdict}]{(' ' + extra) if extra else ''}"
    played = getattr(cassette, "play_count", 0) or 0
    total = len(cassette) if hasattr(cassette, "__len__") else 0
    base = f"[{verdict}] played={played} entries={total}"
    if extra:
        base = f"{base} {extra}"
    return base


# ---------------------------------------------------------------------------
# Live-call detection for tests that bypass VCR.
#
# When a test isn't VCR-marked (respx_conflict, incompatible, or just
# plain unmarked), we wrap its socket calls inside the autouse
# ``_vcr_outcome_gate`` fixture so we can flag any outbound TCP connection
# to a known LLM provider. This converts "likely live call" into
# "confirmed: this test connected to host X".
# ---------------------------------------------------------------------------

_LIVE_CALL_BUFFER_KEY = "vcr_live_call_hosts"


def _is_live_call_host(host: str) -> bool:
    if not host:
        return False
    host = host.lower()
    if any(host.startswith(p) for p in _LIVE_CALL_LOCAL_PREFIXES):
        return False
    if any(host.endswith(suffix) for suffix in _LIVE_CALL_HOST_SUFFIXES):
        return True
    # AWS Bedrock endpoints are ``bedrock-runtime[-fips].{region}.amazonaws.com``
    # (region between ``bedrock-runtime`` and ``amazonaws.com``), so plain
    # suffix matching can't catch them.
    if host.endswith(".amazonaws.com") and host.split(".", 1)[0].startswith(
        "bedrock-runtime"
    ):
        return True
    return False


class _LiveCallProbe:
    """Context manager that monkeypatches ``socket.create_connection`` and
    ``socket.socket.connect`` for the lifetime of a test, recording any
    outbound TCP connection to a known LLM host.

    We don't intercept HTTP at the application layer because that would
    fight with vcrpy/respx in tests that *do* mock httpx — the socket
    layer is below both, so this probe is safe regardless of what's
    patched above it. We also don't raise: the goal is observability, not
    a hard gate.
    """

    def __init__(self) -> None:
        self.hosts: list[str] = []
        self._orig_create_connection = None
        self._orig_socket_connect = None

    def __enter__(self):
        self._orig_create_connection = socket.create_connection
        self._orig_socket_connect = socket.socket.connect

        def _wrapped_create_connection(address, *args, **kwargs):
            try:
                host = address[0] if isinstance(address, tuple) else None
                if host and _is_live_call_host(host) and host not in self.hosts:
                    self.hosts.append(host)
            except Exception:
                pass
            return self._orig_create_connection(address, *args, **kwargs)

        def _wrapped_socket_connect(sock_self, address):
            try:
                host = address[0] if isinstance(address, tuple) else None
                if host and _is_live_call_host(host) and host not in self.hosts:
                    self.hosts.append(host)
            except Exception:
                pass
            return self._orig_socket_connect(sock_self, address)

        socket.create_connection = _wrapped_create_connection
        socket.socket.connect = _wrapped_socket_connect
        return self

    def __exit__(self, *exc):
        if self._orig_create_connection is not None:
            socket.create_connection = self._orig_create_connection
        if self._orig_socket_connect is not None:
            socket.socket.connect = self._orig_socket_connect
        return False


def vcr_outcome_logging_enabled() -> bool:
    """Verdict logging is on whenever VCR itself is active.

    The old ``LITELLM_VCR_VERBOSE=1`` gate kept logs quiet by default, but
    that hides the very signal we need to know whether a paid test ran
    against a real provider. CI logs already drop a one-line verdict per
    test; that's what makes the cost analysis tractable. Set
    ``LITELLM_VCR_VERBOSE=0`` if you really want the legacy quiet mode.
    """
    if vcr_disabled():
        return False
    if os.environ.get(VCR_VERBOSE_ENV) == "0":
        return False
    return True


def record_vcr_outcome(request, vcr) -> None:
    """Call from the post-yield section of an autouse fixture per test."""
    cassette = vcr
    rep_call = getattr(request.node, "rep_call", None)
    test_passed = bool(rep_call and rep_call.passed)
    cassette_path = getattr(cassette, "_path", None) if cassette is not None else None
    if cassette_path:
        mark_test_outcome_for_cassette(cassette_path, test_passed)

    nodeid = request.node.nodeid

    if cassette is not None:
        verdict = _classify_marked_test(cassette)
        # Track overflow tests even when verbose logging is off — the
        # session summary shows them either way.
        if verdict == VERDICT_MISS_OVERFLOW:
            _session_stats["overflow_tests"].append(nodeid)
        if not test_passed and verdict == VERDICT_MISS_RECORDED:
            verdict = VERDICT_MISS_NOT_PERSISTED
        _session_stats["verdict_counts"][verdict] += 1
        _emit_outcome_payload(request.node, verdict)
        if vcr_outcome_logging_enabled():
            line = _format_verdict_line(verdict, cassette)
            request.node.user_properties.append((_USER_PROP_VERDICT_LINE, line))
        return

    # Cassette is None ⇒ test wasn't VCR-marked. Honor the skip reason
    # we tagged at collection time, and pull live-call hosts captured by
    # the socket probe (if any).
    skip_reason = getattr(
        request.node, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_FILE_OPT_OUT
    )
    _session_stats["skip_reason_counts"][skip_reason] += 1

    hosts = getattr(request.node, _LIVE_CALL_BUFFER_KEY, []) or []
    if hosts:
        verdict = VERDICT_UNMARKED_LIVE_CALL
        _session_stats["unmarked_live_call_tests"].append((nodeid, list(hosts)))
        extra = f"reason={skip_reason} hosts={','.join(hosts)}"
    else:
        verdict = VERDICT_UNMARKED_NO_TRAFFIC
        extra = f"reason={skip_reason}"

    _session_stats["verdict_counts"][verdict] += 1

    examples = _session_stats["skip_reason_examples"][skip_reason]
    if len(examples) < 5:
        examples.append(nodeid)

    _emit_outcome_payload(
        request.node,
        verdict,
        skip_reason=skip_reason,
        live_call_hosts=hosts,
    )
    if vcr_outcome_logging_enabled():
        request.node.user_properties.append(
            (_USER_PROP_VERDICT_LINE, _format_verdict_line(verdict, None, extra))
        )


def install_live_call_probe(request, vcr) -> None:
    """Activate the live-call socket probe for non-VCR-marked tests.

    Call this from inside the per-test autouse ``_vcr_outcome_gate``
    fixture *before* the ``yield``. When ``vcr`` is ``None`` (test isn't
    VCR-marked) we patch ``socket.connect`` for the duration of the test
    and stash any LLM-host connections on ``request.node`` so
    ``record_vcr_outcome`` can include them in the verdict line.

    Tests that *are* VCR-marked don't get the probe — vcrpy itself
    intercepts above the socket layer, so any "outbound" socket would be
    a recording cycle, not real spend.
    """
    # Track the current test for telemetry-leak suppression (applies to every
    # test, VCR-marked or not). See ``_should_drop_telemetry_record``.
    global _current_test_nodeid
    _current_test_nodeid = str(
        getattr(getattr(request, "node", None), "nodeid", "") or ""
    )
    if vcr is not None or vcr_disabled():
        return None
    probe = _LiveCallProbe()
    probe.__enter__()
    setattr(request.node, _LIVE_CALL_BUFFER_KEY, probe.hosts)
    request.addfinalizer(lambda: probe.__exit__(None, None, None))
    return probe


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


def emit_vcr_classification_summary(terminalreporter) -> None:
    """Render the per-classification summary at session end.

    Output sections (only included when non-empty):

    * **Verdict counts** — full breakdown of HIT / MISS:RECORDED /
      MISS:OVERFLOW / MISS:NOT_PERSISTED / PARTIAL / NOOP /
      UNMARKED:LIVE_CALL / UNMARKED:NO_TRAFFIC. The OVERFLOW and
      UNMARKED:LIVE_CALL counts are the cost-leak signals.
    * **Cassette overflow** (>``MAX_EPISODES_PER_CASSETTE``) — these tests
      fire live every CI run because the persister refuses to save them.
      Usually means the request body is non-deterministic (file handle
      consumed, AWS SigV4 timestamp, random UUID).
    * **Unmarked tests with live API calls** — confirmed live HTTP traffic
      to a known LLM host while VCR was *not* active for the test. This
      is the "convert likely → confirmed" signal: each entry is real
      money the cache would otherwise prevent.
    * **Skip-reason breakdown** — how many tests opted out of VCR and
      why (respx_conflict, respx_conflict_module, file_opt_out,
      incompatible). Bare ``file_opt_out`` entries with zero respx usage
      in the module are dead skip-list rows worth pruning.
    """
    if vcr_disabled():
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    snapshot = session_stats_snapshot()
    counts = snapshot["verdict_counts"]
    if not counts:
        return

    terminalreporter.write_sep("=", "VCR CACHE CLASSIFICATION SUMMARY", bold=True)
    for verdict in (
        VERDICT_HIT,
        VERDICT_PARTIAL,
        VERDICT_MISS_RECORDED,
        VERDICT_MISS_OVERFLOW,
        VERDICT_MISS_NOT_PERSISTED,
        VERDICT_NOOP_NO_TRAFFIC,
        VERDICT_UNMARKED_NO_TRAFFIC,
        VERDICT_UNMARKED_LIVE_CALL,
    ):
        n = counts.get(verdict, 0)
        if not n:
            continue
        terminalreporter.write_line(f"  [{verdict}] {n}")

    overflow = snapshot["overflow_tests"]
    if overflow:
        terminalreporter.write_sep(
            "-",
            f"CASSETTE OVERFLOW (>{MAX_EPISODES_PER_CASSETTE} episodes, save refused)",
            red=True,
            bold=True,
        )
        terminalreporter.write_line(
            "  These tests will hit the live provider on every CI run "
            "because the persister won't save cassettes that grew past "
            "the limit. Stabilize the request body (file handle consumed, "
            "SigV4 timestamp, UUID, or boundary leak)."
        )
        for nodeid in overflow:
            terminalreporter.write_line(f"    - {nodeid}")

    live_calls = snapshot["unmarked_live_call_tests"]
    if live_calls:
        terminalreporter.write_sep(
            "-",
            "UNMARKED TESTS WITH LIVE API CALLS",
            red=True,
            bold=True,
        )
        terminalreporter.write_line(
            "  These tests connected to a real LLM provider host while "
            "they were NOT VCR-marked. Either add @pytest.mark.vcr "
            "explicitly, mock with respx, or move them off the "
            "respx_conflict / incompatible skip list."
        )
        for nodeid, hosts in live_calls:
            terminalreporter.write_line(f"    - {nodeid}  →  {','.join(hosts)}")

    reasons = snapshot["skip_reason_counts"]
    if reasons:
        terminalreporter.write_sep("-", "SKIP-REASON BREAKDOWN", bold=True)
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            examples = snapshot["skip_reason_examples"].get(reason, [])
            terminalreporter.write_line(f"  {reason}: {n}")
            for ex in examples:
                terminalreporter.write_line(f"    - {ex}")
    terminalreporter.write_sep("=", bold=True)


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
        # Aggregate xdist-worker stats into the controller's session counters
        # first — this path is independent of verbose logging because the
        # structured outcome payload is always attached when VCR is active,
        # and ``aggregate_report_outcome`` no-ops outside the xdist-controller
        # case on its own.
        aggregate_report_outcome(report)

        if report.when != "teardown":
            return
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return
        if not vcr_outcome_logging_enabled():
            return
        reporter = self.resolve_terminal_reporter()
        if reporter is None:
            return
        verdict = next(
            (
                v
                for k, v in (report.user_properties or [])
                if k == _USER_PROP_VERDICT_LINE
            ),
            None,
        )
        if not verdict:
            return
        reporter.write_line(f"{verdict} :: {report.nodeid}")
