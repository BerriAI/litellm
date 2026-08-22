"""Provider-edge record/replay server for e2e runs (LIT-5745).

Record and replay scope to provider-bound traffic only: the proxy boots for
real, tests hit it for real, and only the hop from the proxy to the provider
is recorded or served from a bundle. Suites opt in per deployment by pointing
``litellm_params.api_base`` at ``provider_edge_api_base(mount)``, which is an
in-process HTTP server mounting each supported provider under a path prefix
(``http://127.0.0.1:<port>/openai`` forwards to ``https://api.openai.com``).
In record mode the edge relays each request verbatim, stores the interaction,
and serves the proxy the same filtered response replay will serve later; in
replay mode it serves straight from the bundle and never opens a provider
connection, so a green replay run with a fake provider key proves the entire
proxy pipeline (auth, routing, spend logging) without provider spend.

Request identity reuses fixture_canonical.py: interactions match by canonical
content key, order-independent across keys and FIFO within one. Edge requests
store no headers at all: SDK telemetry headers vary run to run and credential
headers must never touch disk. An unmatched replay call returns HTTP
``REPLAY_MISS_STATUS`` naming the closest recorded interaction, which the
proxy relays as a provider error the failing test surfaces.

v1 limits: only the mounts in ``EDGE_MOUNTS`` (SigV4 providers like Bedrock
sign the Host header, so a forwarding edge breaks their signatures), streaming
fidelity is LIT-5742, and CI wiring is LIT-5748. Suites that do not wire the
edge keep hitting providers live in every mode.
"""

from __future__ import annotations

import base64
import difflib
import functools
import hashlib
import re
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, assert_never
from urllib.parse import parse_qsl, urlsplit

from pydantic import JsonValue, TypeAdapter

from e2e_http import NetworkError, RawResponse, forward
from fixture_bundle import (
    BundleRecorder,
    Interaction,
    LoadedBundle,
    RecordedHttpResponse,
    RecordedRequest,
    UnreadableBundle,
    UnsafeBundleDir,
    interaction_filename,
    load_bundle,
    prepare_bundle,
    slug_for_test,
)
from fixture_canonical import CanonicalRequest, canonical_string, canonicalize
from fixture_mode import (
    FIXTURE_MODES,
    InvalidFixtureMode,
    ReplayMiss,
    current_test_key,
    parse_fixture_mode,
)

EDGE_MOUNTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai": "https://api.openai.com",
        "anthropic": "https://api.anthropic.com",
    }
)

REPLAY_MISS_STATUS: Final = 599

_HOP_BY_HOP_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_REQUEST_DROPPED_HEADERS: Final[frozenset[str]] = _HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
    "accept-encoding",
}
_RESPONSE_DROPPED_HEADERS: Final[frozenset[str]] = _HOP_BY_HOP_HEADERS | {
    "content-encoding",
    "content-length",
    "set-cookie",
}

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


_BOUNDARY_PATTERN: Final = re.compile(
    r'boundary=(?:"([^"]*)"|([^;,\s]+))', re.IGNORECASE
)
_DISPOSITION_NAME_PATTERN: Final = re.compile(r'(?:^|;)\s*name="([^"]*)"')
_DISPOSITION_FILENAME_PATTERN: Final = re.compile(r'(?:^|;)\s*filename="([^"]*)"')
_UNPARSED_MULTIPART: Final = "<unparsed-multipart>"
_BOUNDARY_PLACEHOLDER: Final = b"--<boundary>"


@dataclass(frozen=True)
class _MultipartPart:
    field_name: str
    filename: str | None
    content: bytes


def _header_value(headers: Mapping[str, str], name: str) -> str:
    wanted: Final = name.lower()
    return next((value for key, value in headers.items() if key.lower() == wanted), "")


def _multipart_boundary(content_type: str) -> str | None:
    if "multipart/form-data" not in content_type.lower():
        return None
    match: Final = _BOUNDARY_PATTERN.search(content_type)
    return None if match is None else match.group(1) or match.group(2)


def _parse_multipart_part(segment: bytes) -> _MultipartPart | None:
    head, separator, content = segment.partition(b"\r\n\r\n")
    if not separator:
        return None
    disposition: Final = "".join(
        value
        for line in head.decode("utf-8", errors="replace").split("\r\n")
        for name, _, value in [line.partition(":")]
        if name.strip().lower() == "content-disposition"
    )
    name_match: Final = _DISPOSITION_NAME_PATTERN.search(disposition)
    if name_match is None:
        return None
    filename_match: Final = _DISPOSITION_FILENAME_PATTERN.search(disposition)
    return _MultipartPart(
        field_name=name_match.group(1),
        filename=None if filename_match is None else filename_match.group(1),
        content=content,
    )


def _multipart_parts(body: bytes, boundary: str) -> tuple[_MultipartPart, ...] | None:
    """The wire body split back into its parts, or None when it does not parse as the
    declared envelope so the caller can fall back to the opaque content digest."""
    segments: Final = body.split(b"--" + boundary.encode())
    if len(segments) < 3 or not segments[-1].startswith(b"--"):
        return None
    parsed: Final = tuple(
        _parse_multipart_part(segment.removeprefix(b"\r\n").removesuffix(b"\r\n"))
        for segment in segments[1:-1]
    )
    if any(part is None for part in parsed):
        return None
    return tuple(part for part in parsed if part is not None)


def _content_digest(content: bytes) -> str:
    """Text is canonicalized before hashing so a per-run marker inside an uploaded JSONL
    does not move the key; anything that is not UTF-8 is hashed byte for byte, since a
    lossy decode collapses every binary payload of one length onto one digest."""
    try:
        text: Final = content.decode("utf-8")
    except UnicodeDecodeError:
        return hashlib.sha256(content).hexdigest()
    return hashlib.sha256(canonical_string(text).encode()).hexdigest()


def _form_fields(fields: tuple[_MultipartPart, ...]) -> dict[str, str]:
    """The ordinary field parts, flattened into the mapping the bundle format stores. A
    name sent more than once takes an index instead of overwriting the earlier value, so
    nothing an upload said is dropped from its key."""
    form: dict[str, str] = {}
    for part in fields:
        name = part.field_name
        occurrence = 1
        while name in form:
            name = f"{part.field_name}[{occurrence}]"
            occurrence += 1
        form[name] = part.content.decode("utf-8", errors="replace")
    return form


def _file_identity(files: tuple[_MultipartPart, ...]) -> tuple[str | None, str | None, int | None]:
    """Name, content digest, and total length for the uploaded file parts. The name
    carries each part's field name as well as its filename, so two uploads sending the
    same bytes under different field names stay apart. A lone file keeps its own content
    digest; several fold into one digest over the per-part identities, which is ordered,
    so parts arriving in a different order key differently."""
    if not files:
        return None, None, None
    names: Final = ", ".join(f"{part.field_name}:{part.filename}" for part in files)
    total: Final = sum(len(part.content) for part in files)
    if len(files) == 1:
        return names, _content_digest(files[0].content), total
    folded: Final = _JSON.dump_json(
        [
            [part.field_name, part.filename, _content_digest(part.content), len(part.content)]
            for part in files
        ]
    )
    return names, hashlib.sha256(folded).hexdigest(), total


def _multipart_request(
    method: str, path: str, params: dict[str, str], parts: tuple[_MultipartPart, ...]
) -> RecordedRequest:
    """A multipart upload keyed by what it says rather than by its wire bytes: every
    ordinary field, plus the identity of the uploaded file. The random per-request
    boundary is envelope, never content, so it never reaches the digest."""
    form: Final = _form_fields(tuple(part for part in parts if part.filename is None))
    file_name, file_sha256, file_bytes = _file_identity(
        tuple(part for part in parts if part.filename is not None)
    )
    return RecordedRequest(
        method=method,
        path=path,
        headers={},
        params=params,
        form=form,
        file_name=file_name,
        file_sha256=file_sha256,
        file_bytes=file_bytes,
    )


def _opaque_request(
    method: str,
    path: str,
    params: dict[str, str],
    body: bytes,
    digested: bytes,
    file_name: str | None = None,
) -> RecordedRequest:
    """A body kept out of the bundle and matched on its digest alone. ``digested`` is
    what the digest runs over, which is the body itself unless something in it has to be
    normalized away first."""
    return RecordedRequest(
        method=method,
        path=path,
        headers={},
        params=params,
        file_name=file_name,
        file_sha256=_content_digest(digested),
        file_bytes=len(body),
    )


def _edge_request(
    method: str, path: str, query: str, body: bytes | None, content_type: str = ""
) -> RecordedRequest:
    """The identity replay matches on: the edge path (mount included), the query as
    params, and the body as parsed JSON, as parsed multipart fields and file identity
    when the content type declares an envelope, or as a content digest otherwise so
    opaque uploads still match across runs. A multipart body that does not parse still
    has its boundary normalized away, because that boundary is fresh every request and
    would otherwise guarantee a miss."""
    params: Final = dict(parse_qsl(query, keep_blank_values=True))
    lowered_method: Final = method.lower()
    if not body:
        return RecordedRequest(method=lowered_method, path=path, headers={}, params=params)
    boundary: Final = _multipart_boundary(content_type)
    if boundary is not None:
        parts = _multipart_parts(body, boundary)
        if parts is not None:
            return _multipart_request(lowered_method, path, params, parts)
        return _opaque_request(
            lowered_method,
            path,
            params,
            body,
            body.replace(b"--" + boundary.encode(), _BOUNDARY_PLACEHOLDER),
            _UNPARSED_MULTIPART,
        )
    try:
        parsed: Final[JsonValue] = _JSON.validate_json(body)
    except ValueError:
        return _opaque_request(lowered_method, path, params, body, body)
    return RecordedRequest(
        method=lowered_method, path=path, headers={}, params=params, body=parsed
    )


def _build_pool(recorded: tuple[Interaction, ...]) -> dict[str, deque[Interaction]]:
    keys: Final = tuple(canonicalize(interaction.request).key for interaction in recorded)
    return {
        key: deque(
            interaction
            for candidate_key, interaction in zip(keys, recorded, strict=True)
            if candidate_key == key
        )
        for key in dict.fromkeys(keys)
    }


def _closest_recorded(
    canonical: CanonicalRequest, recorded: tuple[Interaction, ...]
) -> tuple[CanonicalRequest, str]:
    candidates: Final = tuple(canonicalize(interaction.request) for interaction in recorded)
    ratios: Final = tuple(
        difflib.SequenceMatcher(
            None, f"{canonical.method} {canonical.path}\n{canonical.content}",
            f"{candidate.method} {candidate.path}\n{candidate.content}",
        ).ratio()
        for candidate in candidates
    )
    best: Final = max(range(len(candidates)), key=lambda index: ratios[index])
    return candidates[best], interaction_filename(best, recorded[best].request)


def _miss_message(test_key: str, slug: str, canonical: CanonicalRequest, bundle: LoadedBundle) -> str:
    recorded: Final = bundle.interactions.get(slug, ())
    if not recorded:
        return (
            f"replay miss for {test_key}: computed key {canonical.key} but nothing is recorded "
            f"under {slug}; re-record with E2E_FIXTURE_MODE=record"
        )
    closest, closest_file = _closest_recorded(canonical, recorded)
    diff: Final = "\n".join(
        islice(
            difflib.unified_diff(
                closest.pretty_content().splitlines(),
                canonical.pretty_content().splitlines(),
                fromfile=f"closest recorded ({closest_file})",
                tofile="test made",
                lineterm="",
            ),
            60,
        )
    )
    return (
        f"replay miss for {test_key}: no recorded interaction matches key {canonical.key}; "
        f"closest recorded key is {closest.key} ({closest_file})\n{diff}\n"
        "re-record with E2E_FIXTURE_MODE=record"
    )


@dataclass(slots=True)
class ReplaySource:
    """One shared pool per test over a loaded bundle, so every provider call the
    proxy makes in the session consumes from the same recorded interactions.
    Every pool is built once at construction and per-key consumption is a single
    atomic deque pop, so concurrent replay calls never race. Calls match by
    canonical content key: order-independent across distinct keys (concurrent
    tests interleave calls nondeterministically), FIFO within one key (a retry
    or poll loop replays its recorded responses in recorded order)."""

    bundle: LoadedBundle
    _pools: dict[str, dict[str, deque[Interaction]]] = field(init=False)

    def __post_init__(self) -> None:
        self._pools = {
            slug: _build_pool(recorded) for slug, recorded in self.bundle.interactions.items()
        }

    def _pool(self, slug: str) -> dict[str, deque[Interaction]]:
        return self._pools.get(slug, {})

    def next_interaction(self, request: RecordedRequest) -> Interaction:
        test_key: Final = current_test_key()
        slug: Final = slug_for_test(test_key)
        pool: Final = self._pool(slug)
        canonical: Final = canonicalize(request)
        queue: Final = pool.get(canonical.key)
        if queue is None:
            raise ReplayMiss(_miss_message(test_key, slug, canonical, self.bundle))
        try:
            return queue.popleft()
        except IndexError:
            raise ReplayMiss(
                f"replay exhausted for {test_key}: every recorded interaction for key "
                f"{canonical.key} is already consumed; re-record with E2E_FIXTURE_MODE=record"
            ) from None

    def leftover_error(self, test_key: str) -> str | None:
        """Non-None when the test consumed fewer interactions than were recorded,
        meaning a passing replay proved less than the bundle claims."""
        slug: Final = slug_for_test(test_key)
        recorded: Final = self.bundle.interactions.get(slug, ())
        if not recorded:
            return None
        leftover: Final = tuple(
            interaction for queue in self._pool(slug).values() for interaction in queue
        )
        if not leftover:
            return None
        return (
            f"replay incomplete for {test_key}: {len(leftover)} of {len(recorded)} recorded "
            f"interactions never consumed, e.g. {canonicalize(leftover[0].request).key}; "
            "re-record with E2E_FIXTURE_MODE=record"
        )


@dataclass(frozen=True, slots=True)
class RecordEdge:
    """Record backend: forward to the provider, persist, serve the filtered copy.
    The lock serializes recorder writes because the edge server handles requests
    on concurrent threads."""

    recorder: BundleRecorder
    lock: threading.Lock


@dataclass(frozen=True, slots=True)
class ReplayEdge:
    source: ReplaySource


type EdgeBackend = RecordEdge | ReplayEdge


@dataclass(frozen=True, slots=True)
class EdgeReply:
    status_code: int
    headers: dict[str, str]
    body: bytes


def _text_reply(status_code: int, message: str) -> EdgeReply:
    return EdgeReply(
        status_code=status_code,
        headers={"content-type": "text/plain; charset=utf-8"},
        body=message.encode(),
    )


def _reply_from_recorded(response: RecordedHttpResponse) -> EdgeReply:
    return EdgeReply(
        status_code=response.status_code,
        headers=dict(response.headers),
        body=base64.b64decode(response.body_b64),
    )


def _recorded_response(outcome: RawResponse | NetworkError) -> RecordedHttpResponse:
    match outcome:
        case RawResponse(status_code=status_code, headers=headers, body=body):
            return RecordedHttpResponse(
                status_code=status_code,
                headers={
                    name: value
                    for name, value in headers.items()
                    if name not in _RESPONSE_DROPPED_HEADERS
                },
                body_b64=base64.b64encode(body).decode("ascii"),
            )
        case NetworkError(message=message):
            return RecordedHttpResponse(
                status_code=502,
                headers={"content-type": "text/plain; charset=utf-8"},
                body_b64=base64.b64encode(
                    f"provider edge could not reach the provider: {message}".encode()
                ).decode("ascii"),
            )


def _upstream_url(upstream_base: str, upstream_path: str, query: str) -> str:
    url: Final = f"{upstream_base}/{upstream_path}"
    return f"{url}?{query}" if query else url


def _handle_record(
    backend: RecordEdge,
    request: RecordedRequest,
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> EdgeReply:
    forwarded: Final = {
        name: value for name, value in headers.items() if name.lower() not in _REQUEST_DROPPED_HEADERS
    }
    outcome: Final = forward(method, url, headers=forwarded, body=body, timeout=timeout)
    response: Final = _recorded_response(outcome)
    with backend.lock:
        backend.recorder.record(test_key=current_test_key(), request=request, response=response)
    return _reply_from_recorded(response)


def _handle_replay(source: ReplaySource, request: RecordedRequest) -> EdgeReply:
    try:
        interaction: Final = source.next_interaction(request)
    except ReplayMiss as miss:
        return _text_reply(REPLAY_MISS_STATUS, str(miss))
    return _reply_from_recorded(interaction.response)


def handle_edge_request(
    backend: EdgeBackend,
    mounts: Mapping[str, str],
    method: str,
    raw_path: str,
    headers: Mapping[str, str],
    body: bytes | None,
    *,
    timeout: float,
) -> EdgeReply:
    """The edge's pure core, one HTTP exchange in and out: resolve the mount
    prefix, then record (forward + persist) or replay (serve from the bundle).
    Socket-free so unit tests exercise every branch without a server."""
    split: Final = urlsplit(raw_path)
    mount, _, upstream_path = split.path.lstrip("/").partition("/")
    upstream_base: Final = mounts.get(mount)
    if upstream_base is None:
        return _text_reply(
            404, f"unknown provider mount {mount!r}; known mounts: {', '.join(sorted(mounts))}"
        )
    request: Final = _edge_request(
        method, split.path, split.query, body, _header_value(headers, "content-type")
    )
    match backend:
        case RecordEdge():
            return _handle_record(
                backend,
                request,
                method=method,
                url=_upstream_url(upstream_base, upstream_path, split.query),
                headers=headers,
                body=body,
                timeout=timeout,
            )
        case ReplayEdge(source=source):
            return _handle_replay(source, request)
        case _:
            assert_never(backend)


class _EdgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def _handle(self) -> None:
        edge_server: Final = self.server
        assert isinstance(edge_server, _EdgeHTTPServer)
        length: Final = int(self.headers.get("content-length") or "0")
        body: Final = self.rfile.read(length) if length else None
        reply: Final = handle_edge_request(
            edge_server.backend,
            edge_server.mounts,
            self.command,
            self.path,
            {name.lower(): value for name, value in self.headers.items()},
            body,
            timeout=edge_server.forward_timeout,
        )
        self.send_response(reply.status_code)
        for name, value in reply.headers.items():
            self.send_header(name, value)
        self.send_header("content-length", str(len(reply.body)))
        self.end_headers()
        self.wfile.write(reply.body)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the per-request stderr line BaseHTTPRequestHandler emits."""


class _EdgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        bind: tuple[str, int],
        *,
        backend: EdgeBackend,
        mounts: Mapping[str, str],
        forward_timeout: float,
    ) -> None:
        super().__init__(bind, _EdgeHandler)
        self.backend: Final = backend
        self.mounts: Final = mounts
        self.forward_timeout: Final = forward_timeout


@dataclass(frozen=True, slots=True)
class ProviderEdge:
    port: int
    advertise_host: str

    def api_base(self, mount: str) -> str:
        return f"http://{self.advertise_host}:{self.port}/{mount}"


@dataclass(frozen=True, slots=True)
class RunningEdge:
    edge: ProviderEdge
    server: _EdgeHTTPServer

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def start_provider_edge(
    backend: EdgeBackend,
    *,
    mounts: Mapping[str, str] = EDGE_MOUNTS,
    bind_host: str = "127.0.0.1",
    advertise_host: str | None = None,
    forward_timeout: float = 60.0,
) -> RunningEdge:
    """Boot an edge server on an OS-assigned port in a daemon thread.
    ``advertise_host`` is what api_base URLs name (it differs from the bind
    host when the proxy runs in a container and reaches the host machine via
    a gateway address like host.docker.internal)."""
    server: Final = _EdgeHTTPServer(
        (bind_host, 0), backend=backend, mounts=mounts, forward_timeout=forward_timeout
    )
    thread: Final = threading.Thread(target=server.serve_forever, name="e2e-provider-edge", daemon=True)
    thread.start()
    return RunningEdge(
        edge=ProviderEdge(port=server.server_address[1], advertise_host=advertise_host or bind_host),
        server=server,
    )


@functools.lru_cache(maxsize=8)
def _shared_recorder(root: Path) -> BundleRecorder:
    prepared = prepare_bundle(root)
    if isinstance(prepared, UnsafeBundleDir):
        raise ValueError(f"E2E_FIXTURE_DIR {prepared.path} {prepared.reason}")
    return prepared


@functools.lru_cache(maxsize=8)
def _shared_replay_source(root: Path) -> ReplaySource:
    loaded = load_bundle(root)
    if isinstance(loaded, UnreadableBundle):
        raise ValueError(f"cannot replay from {root}: {loaded.reason}")
    return ReplaySource(bundle=loaded)


@functools.lru_cache(maxsize=8)
def _shared_edge(
    mode: Literal["record", "replay"],
    bundle_dir: Path,
    bind_host: str,
    advertise_host: str,
    forward_timeout: float,
) -> ProviderEdge:
    backend: Final[EdgeBackend] = (
        RecordEdge(recorder=_shared_recorder(bundle_dir), lock=threading.Lock())
        if mode == "record"
        else ReplayEdge(source=_shared_replay_source(bundle_dir))
    )
    return start_provider_edge(
        backend,
        mounts=EDGE_MOUNTS,
        bind_host=bind_host,
        advertise_host=advertise_host,
        forward_timeout=forward_timeout,
    ).edge


def replay_leftover_error(*, mode_raw: str, bundle_dir: Path, test_key: str) -> str | None:
    """Teardown-time completeness check: in replay mode a passed test with
    unconsumed recorded interactions must fail instead of passing against a
    recording it no longer matches. Inert in every other mode."""
    if parse_fixture_mode(mode_raw) != "replay":
        return None
    return _shared_replay_source(bundle_dir).leftover_error(test_key)


def provider_edge_api_base(
    mount: str,
    *,
    mode_raw: str,
    bundle_dir: Path,
    bind_host: str,
    advertise_host: str,
    forward_timeout: float = 60.0,
) -> str | None:
    """The api_base a suite gives an edge-wired deployment: None in live mode
    (the deployment keeps its real provider api_base) and the process-wide edge
    server's mount URL in record and replay, booting the server on first use."""
    mode: Final = parse_fixture_mode(mode_raw)
    match mode:
        case InvalidFixtureMode(value=value):
            raise ValueError(f"E2E_FIXTURE_MODE={value!r} is not one of {', '.join(FIXTURE_MODES)}")
        case "live":
            return None
        case "record" | "replay":
            if mount not in EDGE_MOUNTS:
                raise ValueError(
                    f"unknown provider mount {mount!r}; known mounts: {', '.join(sorted(EDGE_MOUNTS))}"
                )
            return _shared_edge(mode, bundle_dir, bind_host, advertise_host, forward_timeout).api_base(mount)
        case _:
            assert_never(mode)
