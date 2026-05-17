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
from collections import defaultdict
from typing import Iterable

import pytest

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


_AWS_SIGV4_CREDENTIAL_RE = re.compile(
    r"AWS4-HMAC-SHA256\s+Credential=([^/\s,]+)/", re.IGNORECASE
)


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
