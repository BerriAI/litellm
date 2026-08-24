"""Harness coverage for the provider-edge record/replay server (LIT-5745).

No proxy and no ``e2e`` marker. A stdlib http.server stands in for the
provider (dependency injection via the mounts mapping, no monkeypatching):
record mode must forward each edge call to it verbatim, persist one
interaction file, and serve the proxy the same filtered response replay will
serve later; replay mode must serve byte-identical responses from the bundle
alone, with the fake provider's hit log proving nothing leaves the process,
and answer any drifted call with HTTP ``REPLAY_MISS_STATUS`` naming the
computed and closest recorded canonical keys (LIT-5741; the pure canonicalizer
is pinned in test_fixture_canonical.py). Requests are made through
``e2e_http.forward`` so the whole HTTP surface of the edge is exercised; the
pure ``handle_edge_request`` core is pinned socket-free alongside.

Streaming fidelity (LIT-5742) is pinned at the transfer layer, because that is
the only layer where it is visible: a chunked provider sends a known list of
transfer chunks, one of which deliberately splits an SSE event mid-token, and a
raw-socket client reads the edge's own reply back as HTTP chunks. Counting SSE
events at the client would prove nothing, since a coalesced body carries the
same events as a chunk-per-event one.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
from collections.abc import Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from e2e_http import RawResponse, StreamChunk, forward
from fixture_canonical import canonicalize
from fixture_bundle import (
    BundleRecorder,
    Interaction,
    LoadedBundle,
    RecordedHttpResponse,
    RecordedRequest,
    RecordedStreamedResponse,
    load_bundle,
    prepare_bundle,
    slug_for_test,
)
from fixture_mode import current_test_key
from provider_edge import (
    REPLAY_MISS_STATUS,
    EdgeBackend,
    EdgeReply,
    EdgeStream,
    ProviderEdge,
    RecordEdge,
    ReplayEdge,
    ReplaySource,
    edge_request,
    handle_edge_request,
    provider_edge_api_base,
    replay_leftover_error,
    start_provider_edge,
)

CHAT_PATH = "/openai/v1/chat/completions"
UPLOAD_PATH = "/openai/v1/files"
REPLAY_MOUNTS = {"openai": "https://replay.invalid"}
JSON_OBJECT = TypeAdapter(dict[str, object])
BATCH_JSONL = b'{"custom_id":"one"}\n{"custom_id":"two"}\n'


def json_object(body: bytes) -> dict[str, object]:
    return JSON_OBJECT.validate_json(body)


class _FakeProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, bind: tuple[str, int]) -> None:
        super().__init__(bind, _FakeProviderHandler)
        self.hits: list[str] = []


class _FakeProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self._respond()

    def do_GET(self) -> None:
        self._respond()

    def _respond(self) -> None:
        provider = self.server
        assert isinstance(provider, _FakeProvider)
        length = int(self.headers.get("content-length") or "0")
        body = self.rfile.read(length) if length else b""
        provider.hits.append(f"{self.command} {self.path}")
        payload = json.dumps(
            {"echo": body.decode("utf-8"), "path": self.path, "hit": len(provider.hits)}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.send_header("x-upstream", "fake")
        self.send_header("set-cookie", "session=fake-cookie")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the per-request stderr line BaseHTTPRequestHandler emits."""


@contextmanager
def fake_provider() -> Generator[_FakeProvider]:
    server = _FakeProvider(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def provider_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


STREAM_PATH = "/openai/v1/messages"
STREAM_BODY = json.dumps({"model": "claude", "stream": True}).encode()
MID_EVENT_HEAD = b'data: {"type":"content_bl'
MID_EVENT_TAIL = b'ock_delta","delta":{"text":" two"}}\n\n'
SSE_CHUNKS: tuple[bytes, ...] = (
    b'data: {"type":"content_block_delta","delta":{"text":"one"}}\n\n',
    MID_EVENT_HEAD,
    MID_EVENT_TAIL,
    b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n',
    b"data: [DONE]\n\n",
)
JSON_CHUNKS: tuple[bytes, ...] = (b'{"echo":"one",', b'"chunked":true}')


class _ChunkedProvider(ThreadingHTTPServer):
    """A provider that frames its response as a known list of transfer chunks, each
    flushed on its own, and optionally hangs up part way through without writing the
    terminating chunk. The chunk list is what the recording has to reproduce."""

    daemon_threads = True

    def __init__(
        self,
        bind: tuple[str, int],
        *,
        chunks: tuple[bytes, ...],
        content_type: str,
        abort_after: int | None,
    ) -> None:
        super().__init__(bind, _ChunkedProviderHandler)
        self.chunks = chunks
        self.content_type = content_type
        self.abort_after = abort_after
        self.hits: list[str] = []


class _ChunkedProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        provider = self.server
        assert isinstance(provider, _ChunkedProvider)
        length = int(self.headers.get("content-length") or "0")
        if length:
            self.rfile.read(length)
        provider.hits.append(f"{self.command} {self.path}")
        self.send_response(200)
        self.send_header("content-type", provider.content_type)
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        limit = len(provider.chunks) if provider.abort_after is None else provider.abort_after
        for chunk in provider.chunks[:limit]:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        if limit < len(provider.chunks):
            self.close_connection = True
            return
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        """Silence the per-request stderr line BaseHTTPRequestHandler emits."""


@contextmanager
def chunked_provider(
    *,
    chunks: tuple[bytes, ...] = SSE_CHUNKS,
    content_type: str = "text/event-stream",
    abort_after: int | None = None,
) -> Generator[_ChunkedProvider]:
    server = _ChunkedProvider(
        ("127.0.0.1", 0), chunks=chunks, content_type=content_type, abort_after=abort_after
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def response_header(head: str, name: str) -> str | None:
    wanted = f"{name.lower()}:"
    for line in head.splitlines()[1:]:
        if line.lower().startswith(wanted):
            return line.split(":", 1)[1].strip()
    return None


def _read_chunked(sock: socket.socket, buffered: bytes) -> tuple[list[bytes], str]:
    """A chunked body read back one entry per HTTP chunk, plus how the message ended.

    The framing is parsed rather than ``recv`` calls counted, because TCP is free to
    coalesce two chunks into one segment or split one across two, so a read count
    says nothing about how the sender framed the message."""
    chunks: list[bytes] = []
    try:
        while True:
            while b"\r\n" not in buffered:
                piece = sock.recv(65536)
                if not piece:
                    return chunks, "truncated"
                buffered += piece
            line, _, buffered = buffered.partition(b"\r\n")
            size = int(line.split(b";")[0], 16)
            if size == 0:
                return chunks, "terminated"
            while len(buffered) < size + 2:
                piece = sock.recv(65536)
                if not piece:
                    return chunks, "truncated"
                buffered += piece
            chunks.append(buffered[:size])
            buffered = buffered[size + 2 :]
    except ConnectionResetError:
        return chunks, "reset"


def _read_fixed(sock: socket.socket, buffered: bytes, length: int) -> tuple[list[bytes], str]:
    while len(buffered) < length:
        piece = sock.recv(65536)
        if not piece:
            return ([buffered] if buffered else []), "truncated"
        buffered += piece
    return ([buffered[:length]] if length else []), "terminated"


def raw_stream_post(port: int, path: str, body: bytes) -> tuple[str, list[bytes], str]:
    """POST over a raw socket and read the reply at the transfer layer: the response
    head, one entry per HTTP chunk (or the whole body for a content-length reply),
    and how the message ended, ``terminated`` when its terminator arrived,
    ``truncated`` on a graceful close before it, ``reset`` on an abortive one.

    ``call_edge`` goes through ``forward``, which buffers, so it cannot see any of
    this; the streaming tests need the framing itself, so they read the socket."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=15)
    try:
        sock.sendall(
            (
                f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                f"content-type: application/json\r\ncontent-length: {len(body)}\r\n\r\n"
            ).encode()
            + body
        )
        buffered = b""
        while b"\r\n\r\n" not in buffered:
            piece = sock.recv(65536)
            if not piece:
                break
            buffered += piece
        head_bytes, _, rest = buffered.partition(b"\r\n\r\n")
        head = head_bytes.decode("latin-1")
        if (response_header(head, "transfer-encoding") or "").lower() == "chunked":
            chunks, ending = _read_chunked(sock, rest)
        else:
            chunks, ending = _read_fixed(
                sock, rest, int(response_header(head, "content-length") or 0)
            )
        return head, chunks, ending
    finally:
        sock.close()


@contextmanager
def running_edge(backend: EdgeBackend, mounts: Mapping[str, str]) -> Generator[ProviderEdge]:
    running = start_provider_edge(backend, mounts=mounts, bind_host="127.0.0.1")
    try:
        yield running.edge
    finally:
        running.shutdown()


def record_backend(root: Path) -> RecordEdge:
    recorder = prepare_bundle(root)
    assert isinstance(recorder, BundleRecorder)
    return RecordEdge(recorder=recorder, lock=threading.Lock())


def replay_source(root: Path) -> ReplaySource:
    loaded = load_bundle(root)
    assert isinstance(loaded, LoadedBundle)
    return ReplaySource(bundle=loaded)


def call_edge(
    edge: ProviderEdge,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> RawResponse:
    outcome = forward(
        method,
        f"http://{edge.advertise_host}:{edge.port}{path}",
        headers=headers or {},
        body=body,
        timeout=10.0,
    )
    assert isinstance(outcome, RawResponse)
    return outcome


def this_tests_files(root: Path) -> list[Path]:
    slug_dir = root / slug_for_test(current_test_key())
    return sorted(slug_dir.glob("*.json")) if slug_dir.is_dir() else []


def chat_body(prompt: str) -> bytes:
    return json.dumps({"model": "gpt", "messages": [{"role": "user", "content": prompt}]}).encode()


def multipart_body(
    boundary: str,
    fields: tuple[tuple[str, str], ...] = (),
    files: tuple[tuple[str, str, bytes], ...] = (),
) -> bytes:
    """One multipart/form-data body on the wire, exactly as ``requests`` writes it, with
    the boundary under the caller's control instead of randomly generated."""
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        + value.encode()
        for name, value in fields
    ] + [
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
        ).encode()
        + content
        for name, filename, content in files
    ]
    return b"\r\n".join(parts) + f"\r\n--{boundary}--\r\n".encode()


def upload_headers(boundary: str) -> dict[str, str]:
    return {
        "content-type": f"multipart/form-data; boundary={boundary}",
        "authorization": "Bearer sk-upload-secret",
    }


def record_upload(root: Path, body: bytes, boundary: str) -> None:
    with fake_provider() as provider:
        with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
            call_edge(edge, "POST", UPLOAD_PATH, body=body, headers=upload_headers(boundary))


def replay_upload(root: Path, body: bytes, boundary: str) -> RawResponse:
    with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
        return call_edge(edge, "POST", UPLOAD_PATH, body=body, headers=upload_headers(boundary))


class TestRecordMode:
    def test_forwards_to_the_provider_and_writes_one_interaction_file(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                reply = call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
            assert provider.hits == ["POST /v1/chat/completions"]
        assert reply.status_code == 200
        served = json_object(reply.body)
        assert served["echo"] == chat_body("hi").decode()
        files = this_tests_files(root)
        assert [file.name for file in files] == ["0000-post-openai-v1-chat-completions.json"]
        interaction = Interaction.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert interaction.request.method == "post"
        assert interaction.request.path == CHAT_PATH
        assert interaction.request.body == json_object(chat_body("hi"))
        assert interaction.response.status_code == 200

    def test_never_stores_headers_so_credentials_never_touch_disk(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(
                    edge,
                    "POST",
                    CHAT_PATH,
                    body=chat_body("hi"),
                    headers={"authorization": "Bearer sk-live-provider-secret-abc123"},
                )
        raw = this_tests_files(root)[0].read_text(encoding="utf-8")
        assert "sk-live-provider-secret-abc123" not in raw
        interaction = Interaction.model_validate_json(raw)
        assert interaction.request.headers == {}

    def test_strips_volatile_response_headers_and_serves_the_filtered_copy(self, tmp_path: Path) -> None:
        """What record serves the proxy must equal what replay will serve later
        (record/replay parity), so the filtered stored copy is served in both."""
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                reply = call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        assert reply.headers.get("x-upstream") == "fake"
        assert "set-cookie" not in reply.headers
        interaction = Interaction.model_validate_json(
            this_tests_files(root)[0].read_text(encoding="utf-8")
        )
        assert interaction.response.headers.get("x-upstream") == "fake"
        assert "set-cookie" not in interaction.response.headers
        assert "content-length" not in interaction.response.headers

    def test_unreachable_provider_records_and_serves_a_502(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with running_edge(record_backend(root), {"openai": "http://127.0.0.1:9"}) as edge:
            reply = call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        assert reply.status_code == 502
        assert b"could not reach the provider" in reply.body
        interaction = Interaction.model_validate_json(
            this_tests_files(root)[0].read_text(encoding="utf-8")
        )
        assert interaction.response.status_code == 502


class TestReplayMode:
    def test_serves_recorded_bytes_with_zero_provider_hits(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                recorded = call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
            hits_after_record = list(provider.hits)
            with running_edge(
                ReplayEdge(source=replay_source(root)), {"openai": provider_url(provider)}
            ) as edge:
                replayed = call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
            assert provider.hits == hits_after_record
        assert replayed.status_code == recorded.status_code
        assert replayed.body == recorded.body
        assert replayed.headers.get("x-upstream") == "fake"

    def test_request_identity_ignores_auth_headers(self, tmp_path: Path) -> None:
        """The proxy sends different bearer tokens across runs (fresh virtual
        keys, rotated provider keys), so headers are no part of the match."""
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(
                    edge, "POST", CHAT_PATH, body=chat_body("hi"),
                    headers={"authorization": "Bearer sk-first-run"},
                )
        with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
            replayed = call_edge(
                edge, "POST", CHAT_PATH, body=chat_body("hi"),
                headers={"authorization": "Bearer sk-second-run"},
            )
        assert replayed.status_code == 200

    def test_content_drift_returns_the_miss_status_naming_both_keys(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(edge, "POST", CHAT_PATH, body=chat_body("x"))
        with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
            missed = call_edge(edge, "POST", CHAT_PATH, body=chat_body("y"))
        assert missed.status_code == REPLAY_MISS_STATUS
        message = missed.body.decode()
        assert f"no recorded interaction matches key post {CHAT_PATH} #" in message
        assert f"closest recorded key is post {CHAT_PATH} #" in message
        assert '"content": "x"' in message
        assert '"content": "y"' in message
        assert "re-record with E2E_FIXTURE_MODE=record" in message

    def test_query_params_are_part_of_the_identity(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(edge, "GET", "/openai/v1/models?purpose=batch")
            assert provider.hits == ["GET /v1/models?purpose=batch"]
        with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
            missed = call_edge(edge, "GET", "/openai/v1/models?purpose=other")
            matched = call_edge(edge, "GET", "/openai/v1/models?purpose=batch")
        assert missed.status_code == REPLAY_MISS_STATUS
        assert matched.status_code == 200

    def test_identical_requests_replay_their_responses_in_recorded_order(self, tmp_path: Path) -> None:
        """A poll or retry loop repeats the same request and the proxy asserts
        on the progression, so duplicates under one key stay FIFO."""
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
                call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
            first = json_object(call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi")).body)
            second = json_object(call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi")).body)
        assert first["hit"] == 1
        assert second["hit"] == 2

    def test_exhausted_key_returns_the_miss_status(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
            call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
            exhausted = call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        assert exhausted.status_code == REPLAY_MISS_STATUS
        assert b"already consumed" in exhausted.body

    def test_non_json_bodies_match_by_canonical_digest_without_storing_them(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        opaque = b"custom_id one\ncustom_id two\n"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(edge, "POST", "/openai/v1/files", body=opaque)
        raw = this_tests_files(root)[0].read_text(encoding="utf-8")
        interaction = Interaction.model_validate_json(raw)
        assert interaction.request.body is None
        assert interaction.request.file_sha256 is not None
        assert interaction.request.file_bytes == len(opaque)
        assert "custom_id" not in interaction.request.model_dump_json()
        with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
            replayed = call_edge(edge, "POST", "/openai/v1/files", body=opaque)
        assert replayed.status_code == 200


class TestMultipartIdentity:
    """LIT-5974: a multipart upload is keyed by its parsed fields and file identity.
    ``requests`` picks a fresh random boundary per request, so hashing the wire body
    made every upload miss on replay; parsing the envelope keys the upload on what it
    actually says, which is stable across runs and still separates real drift."""

    def test_a_fresh_boundary_replays_the_same_upload(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        recorded = multipart_body(
            "d0a1b2c3d4e5f60718293a4b5c6d7e8f",
            fields=(("purpose", "batch"),),
            files=(("file", "batch.jsonl", BATCH_JSONL),),
        )
        record_upload(root, recorded, "d0a1b2c3d4e5f60718293a4b5c6d7e8f")

        rerun = multipart_body(
            "ffffeeeeddddccccbbbbaaaa99998888",
            fields=(("purpose", "batch"),),
            files=(("file", "batch.jsonl", BATCH_JSONL),),
        )
        assert rerun != recorded
        replayed = replay_upload(root, rerun, "ffffeeeeddddccccbbbbaaaa99998888")
        assert replayed.status_code == 200, replayed.body[:400]

    def test_the_stored_request_carries_fields_and_file_identity_but_no_secrets(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "bundle"
        boundary = "0123456789abcdef0123456789abcdef"
        record_upload(
            root,
            multipart_body(
                boundary,
                fields=(("purpose", "batch"),),
                files=(("file", "batch.jsonl", BATCH_JSONL),),
            ),
            boundary,
        )

        raw = this_tests_files(root)[0].read_text(encoding="utf-8")
        interaction = Interaction.model_validate_json(raw)
        assert interaction.request.form == {"purpose": "batch"}
        assert interaction.request.file_name == json.dumps(
            [["file", "batch.jsonl", "application/octet-stream"]], separators=(",", ":")
        )
        assert interaction.request.file_bytes == len(BATCH_JSONL)
        stored = interaction.request.model_dump_json()
        assert boundary not in stored
        assert "sk-upload-secret" not in stored
        assert "custom_id" not in stored

    @pytest.mark.parametrize(
        ("fields", "files"),
        [
            pytest.param(
                (("purpose", "batch"),),
                (("file", "batch.jsonl", b'{"custom_id":"three"}\n'),),
                id="file-content",
            ),
            pytest.param(
                (("purpose", "batch"),),
                (("file", "other.jsonl", BATCH_JSONL),),
                id="file-name",
            ),
            pytest.param(
                (("purpose", "fine-tune"),),
                (("file", "batch.jsonl", BATCH_JSONL),),
                id="form-field",
            ),
            pytest.param(
                (("purpose", "batch"), ("purpose", "batch")),
                (("file", "batch.jsonl", BATCH_JSONL),),
                id="repeated-form-field",
            ),
            pytest.param(
                (("purpose", "batch"),),
                (
                    ("file", "batch.jsonl", BATCH_JSONL),
                    ("mask", "mask.jsonl", BATCH_JSONL),
                ),
                id="extra-file-part",
            ),
        ],
    )
    def test_a_structurally_different_upload_misses(
        self,
        tmp_path: Path,
        fields: tuple[tuple[str, str], ...],
        files: tuple[tuple[str, str, bytes], ...],
    ) -> None:
        root = tmp_path / "bundle"
        record_upload(
            root,
            multipart_body(
                "aaaaaaaabbbbbbbbccccccccdddddddd",
                fields=(("purpose", "batch"),),
                files=(("file", "batch.jsonl", BATCH_JSONL),),
            ),
            "aaaaaaaabbbbbbbbccccccccdddddddd",
        )

        drifted = replay_upload(
            root,
            multipart_body("11112222333344445555666677778888", fields=fields, files=files),
            "11112222333344445555666677778888",
        )
        assert drifted.status_code == REPLAY_MISS_STATUS

    def test_several_file_parts_separate_when_their_contents_swap(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        image, mask = b"image-bytes", b"mask-bytes"
        record_upload(
            root,
            multipart_body(
                "1a1a1a1a2b2b2b2b3c3c3c3c4d4d4d4d",
                fields=(("prompt", "a cat"),),
                files=(("image", "a.png", image), ("mask", "b.png", mask)),
            ),
            "1a1a1a1a2b2b2b2b3c3c3c3c4d4d4d4d",
        )

        swapped = replay_upload(
            root,
            multipart_body(
                "5e5e5e5e6f6f6f6f7070707081818181",
                fields=(("prompt", "a cat"),),
                files=(("image", "a.png", mask), ("mask", "b.png", image)),
            ),
            "5e5e5e5e6f6f6f6f7070707081818181",
        )
        assert swapped.status_code == REPLAY_MISS_STATUS

        same = replay_upload(
            root,
            multipart_body(
                "9292929203030303a4a4a4a4b5b5b5b5",
                fields=(("prompt", "a cat"),),
                files=(("image", "a.png", image), ("mask", "b.png", mask)),
            ),
            "9292929203030303a4a4a4a4b5b5b5b5",
        )
        assert same.status_code == 200, same.body[:400]

    def test_a_body_that_does_not_match_its_declared_boundary_stays_opaque(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "bundle"
        opaque = b"custom_id one\ncustom_id two\n"
        absent = "boundary-that-is-absent-from-the-body"
        record_upload(root, opaque, absent)

        raw = this_tests_files(root)[0].read_text(encoding="utf-8")
        interaction = Interaction.model_validate_json(raw)
        assert interaction.request.form is None
        assert interaction.request.file_name == "<unparsed-multipart>"
        assert interaction.request.file_bytes == len(opaque)
        assert "custom_id" not in interaction.request.model_dump_json()
        assert replay_upload(root, opaque, absent).status_code == 200


def raw_multipart(boundary: str, *parts: tuple[str, bytes]) -> bytes:
    """A body assembled from literal part headers, so a test can send the shapes a
    well-formed helper cannot: a file part with no filename, a declared per-part content
    type, a repeated or bracketed field name, or a non-UTF-8 value."""
    return (
        b"".join(
            f"--{boundary}\r\n{head}\r\n\r\n".encode() + content + b"\r\n"
            for head, content in parts
        )
        + f"--{boundary}--\r\n".encode()
    )


def upload_key(body: bytes, boundary: str) -> str:
    content_type: Final = f"multipart/form-data; boundary={boundary}"
    return canonicalize(edge_request("POST", UPLOAD_PATH, "", body, content_type)).key


DISPOSITION = 'Content-Disposition: form-data; name="{name}"'
FILE_DISPOSITION = DISPOSITION + '; filename="{filename}"'


class TestMultipartIdentityEdges:
    """The identity a multipart upload keys on, pinned against the ways two materially
    different uploads could otherwise collapse onto one key. A collision here is the
    dangerous failure: replay would answer one request with another's response."""

    def test_a_declared_part_content_type_separates_otherwise_identical_uploads(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"
        as_json = raw_multipart(
            boundary,
            (FILE_DISPOSITION.format(name="file", filename="a") + "\r\nContent-Type: application/json", b"xy"),
        )
        as_csv = raw_multipart(
            boundary,
            (FILE_DISPOSITION.format(name="file", filename="a") + "\r\nContent-Type: text/csv", b"xy"),
        )

        assert upload_key(as_json, boundary) != upload_key(as_csv, boundary)

    def test_a_file_part_without_a_filename_is_not_mistaken_for_a_plain_field(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"
        upload = raw_multipart(
            boundary,
            (DISPOSITION.format(name="file") + "\r\nContent-Type: application/octet-stream", b"CONTENT"),
        )
        plain_field = raw_multipart(boundary, (DISPOSITION.format(name="file"), b"CONTENT"))

        request = edge_request(
            "POST", UPLOAD_PATH, "", upload, f"multipart/form-data; boundary={boundary}"
        )

        assert upload_key(upload, boundary) != upload_key(plain_field, boundary)
        assert request.form == {}
        assert b"CONTENT".decode() not in request.model_dump_json()

    def test_a_filename_carrying_a_per_run_marker_keys_the_same_next_run(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"

        def upload(marker: str) -> str:
            body = raw_multipart(
                boundary,
                (FILE_DISPOSITION.format(name="one", filename=f"{marker}.jsonl"), b"first"),
                (FILE_DISPOSITION.format(name="two", filename="steady.jsonl"), b"second"),
            )
            return upload_key(body, boundary)

        assert upload("a1b2c3d4e5f6") == upload("0f9e8d7c6b5a")

    def test_a_separator_inside_a_filename_cannot_forge_a_different_split(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"
        colon_in_filename = raw_multipart(
            boundary, (FILE_DISPOSITION.format(name="file", filename="a:b.jsonl"), b"same")
        )
        colon_in_field = raw_multipart(
            boundary, (FILE_DISPOSITION.format(name="file:a", filename="b.jsonl"), b"same")
        )

        assert upload_key(colon_in_filename, boundary) != upload_key(colon_in_field, boundary)

    def test_a_repeated_field_cannot_collide_with_a_literal_indexed_name(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"
        repeated = raw_multipart(
            boundary,
            (DISPOSITION.format(name="purpose"), b"x"),
            (DISPOSITION.format(name="purpose"), b"y"),
        )
        literal_index = raw_multipart(
            boundary,
            (DISPOSITION.format(name="purpose"), b"x"),
            (DISPOSITION.format(name="purpose[1]"), b"y"),
        )

        assert upload_key(repeated, boundary) != upload_key(literal_index, boundary)

    def test_two_binary_field_values_of_one_length_stay_apart(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"
        first = raw_multipart(boundary, (DISPOSITION.format(name="blob"), b"\xff\xfe\xfd"))
        second = raw_multipart(boundary, (DISPOSITION.format(name="blob"), b"\xf0\xf1\xf2"))

        assert upload_key(first, boundary) != upload_key(second, boundary)

    def test_a_secret_named_field_never_reaches_the_stored_request(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"
        body = raw_multipart(
            boundary,
            (DISPOSITION.format(name="openai_api_key"), b"sk-live-DEADBEEF-0123456789abcd"),
            (DISPOSITION.format(name="purpose"), b"batch"),
        )

        request = edge_request(
            "POST", UPLOAD_PATH, "", body, f"multipart/form-data; boundary={boundary}"
        )

        assert "sk-live-DEADBEEF-0123456789abcd" not in request.model_dump_json()
        assert request.form == {"openai_api_key": "<secret>", "purpose": "batch"}

    def test_a_redacted_field_still_matches_the_live_request_that_carried_the_secret(
        self,
    ) -> None:
        boundary = "0123456789abcdef0123456789abcdef"

        def upload(secret: str) -> str:
            body = raw_multipart(
                boundary,
                (DISPOSITION.format(name="openai_api_key"), secret.encode()),
                (DISPOSITION.format(name="purpose"), b"batch"),
            )
            return upload_key(body, boundary)

        assert upload("sk-live-DEADBEEF-0123456789abcd") == upload("<secret>")

    def test_a_length_change_the_canonicalizer_absorbs_does_not_move_the_key(self) -> None:
        boundary = "0123456789abcdef0123456789abcdef"

        def upload(created: str) -> str:
            body = raw_multipart(
                boundary,
                (
                    FILE_DISPOSITION.format(name="file", filename="batch.jsonl"),
                    b'{"created_at":"' + created.encode() + b'"}',
                ),
            )
            return upload_key(body, boundary)

        assert upload("2026-08-21T02:08:19Z") == upload("2026-08-21T02:08:19.123456Z")

    @pytest.mark.parametrize(
        "content_type",
        [
            pytest.param("multipart/form-data; myboundary=zzz; boundary={boundary}", id="lookalike-parameter"),
            pytest.param("multipart/form-data; BOUNDARY={boundary}", id="uppercase-parameter"),
        ],
    )
    def test_the_boundary_parameter_is_read_the_way_the_client_meant_it(
        self, content_type: str
    ) -> None:
        boundary = "0123456789abcdef0123456789abcdef"
        body = raw_multipart(
            boundary, (FILE_DISPOSITION.format(name="file", filename="batch.jsonl"), BATCH_JSONL)
        )

        request = edge_request(
            "POST", UPLOAD_PATH, "", body, content_type.format(boundary=boundary)
        )

        assert request.form == {}
        assert request.file_name is not None
        assert "batch.jsonl" in request.file_name

    def test_an_empty_declared_boundary_falls_back_instead_of_splitting_on_dashes(self) -> None:
        body = b'--\r\nContent-Disposition: form-data; name="a"\r\n\r\nvalue\r\n----\r\n'

        request = edge_request(
            "POST", UPLOAD_PATH, "", body, 'multipart/form-data; boundary=""'
        )

        assert request.form is None
        assert request.file_sha256 is not None


class TestReplayLeftover:
    def test_partially_consumed_recording_names_the_leftover(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
                call_edge(edge, "GET", "/openai/v1/models")
        source = replay_source(root)
        with running_edge(ReplayEdge(source=source), REPLAY_MOUNTS) as edge:
            call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        error = source.leftover_error(current_test_key())
        assert error is not None
        assert "1 of 2 recorded interactions never consumed" in error
        assert "e.g. get /openai/v1/models #" in error
        assert "re-record with E2E_FIXTURE_MODE=record" in error

    def test_fully_consumed_recording_leaves_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        source = replay_source(root)
        with running_edge(ReplayEdge(source=source), REPLAY_MOUNTS) as edge:
            call_edge(edge, "POST", CHAT_PATH, body=chat_body("hi"))
        assert source.leftover_error(current_test_key()) is None

    def test_test_without_recordings_has_no_leftover(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        assert isinstance(prepare_bundle(root), BundleRecorder)
        assert replay_source(root).leftover_error("suite.py::test_never_recorded") is None

    def test_inert_outside_replay_mode(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        assert replay_leftover_error(mode_raw="", bundle_dir=missing, test_key="k") is None
        assert replay_leftover_error(mode_raw="record", bundle_dir=missing, test_key="k") is None


class TestConcurrentReplay:
    def test_parallel_identical_calls_serve_each_recording_exactly_once(self, tmp_path: Path) -> None:
        """The edge server handles requests on concurrent threads and a burst
        of parallel identical calls consumes one shared pool: no response
        duplicated, none forgotten, nothing left over at teardown."""
        root = tmp_path / "bundle"
        recorder = prepare_bundle(root)
        assert isinstance(recorder, BundleRecorder)
        for ordinal in range(32):
            recorder.record(
                test_key=current_test_key(),
                request=RecordedRequest(method="post", path=CHAT_PATH, headers={}, body={"n": "same"}),
                response=RecordedHttpResponse(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    body_b64=base64.b64encode(json.dumps({"value": f"v{ordinal:02d}"}).encode()).decode(),
                ),
            )
        source = replay_source(root)
        body = json.dumps({"n": "same"}).encode()
        barrier = threading.Barrier(8)
        with running_edge(ReplayEdge(source=source), REPLAY_MOUNTS) as edge:

            def consume(_: int) -> tuple[str, ...]:
                barrier.wait()
                return tuple(
                    str(json_object(call_edge(edge, "POST", CHAT_PATH, body=body).body)["value"])
                    for _call in range(4)
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                served = sorted(value for values in executor.map(consume, range(8)) for value in values)
        assert served == [f"v{ordinal:02d}" for ordinal in range(32)]
        assert source.leftover_error(current_test_key()) is None


def record_stream(root: Path, *, abort_after: int | None = None) -> tuple[str, list[bytes], str]:
    with chunked_provider(abort_after=abort_after) as provider:
        with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
            return raw_stream_post(edge.port, STREAM_PATH, STREAM_BODY)


def replay_stream(root: Path) -> tuple[str, list[bytes], str]:
    with running_edge(ReplayEdge(source=replay_source(root)), REPLAY_MOUNTS) as edge:
        return raw_stream_post(edge.port, STREAM_PATH, STREAM_BODY)


def only_recorded_response(root: Path) -> RecordedHttpResponse | RecordedStreamedResponse:
    files = this_tests_files(root)
    assert len(files) == 1, [file.name for file in files]
    return Interaction.model_validate_json(files[0].read_text(encoding="utf-8")).response


def recorded_stream(root: Path) -> RecordedStreamedResponse:
    response = only_recorded_response(root)
    assert isinstance(response, RecordedStreamedResponse), response
    return response


def stream_chunks(response: RecordedStreamedResponse) -> list[bytes]:
    return [base64.b64decode(chunk) for chunk in response.chunks_b64]


class TestStreamingFidelity:
    """LIT-5742: a streamed response records and replays as the chunk sequence the
    provider actually sent, not as one coalesced body. The unit of fidelity is the
    HTTP transfer chunk, so every assertion here is made at the transfer layer."""

    def test_a_streamed_response_records_its_chunk_boundaries(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        record_stream(root)

        recorded = recorded_stream(root)
        assert recorded.status_code == 200
        assert stream_chunks(recorded) == list(SSE_CHUNKS)
        assert recorded.truncated is None

    def test_replay_reproduces_the_recorded_split_points(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        record_stream(root)

        head, chunks, ending = replay_stream(root)
        assert head.startswith("HTTP/1.1 200 OK")
        assert response_header(head, "transfer-encoding") == "chunked"
        assert response_header(head, "content-type") == "text/event-stream"
        assert len(chunks) > 1
        assert chunks == list(SSE_CHUNKS)
        assert ending == "terminated"

    def test_record_mode_relays_the_stream_chunked_like_replay_will(self, tmp_path: Path) -> None:
        """Record/replay parity at the framing level: what record serves the proxy
        must be what replay serves it later, chunk for chunk."""
        root = tmp_path / "bundle"
        recorded_head, recorded_chunks, recorded_ending = record_stream(root)
        replayed_head, replayed_chunks, replayed_ending = replay_stream(root)

        assert response_header(recorded_head, "transfer-encoding") == "chunked"
        assert recorded_chunks == list(SSE_CHUNKS)
        assert recorded_chunks == replayed_chunks
        assert recorded_ending == replayed_ending == "terminated"
        assert response_header(recorded_head, "transfer-encoding") == response_header(
            replayed_head, "transfer-encoding"
        )

    def test_a_chunk_split_inside_an_event_survives_replay(self, tmp_path: Path) -> None:
        """The anti-tautology test. One provider chunk ends mid-token, so the two
        halves of that SSE event must arrive as two chunks; an implementation that
        joins the body and re-splits it on event boundaries cannot pass this."""
        root = tmp_path / "bundle"
        record_stream(root)

        _, chunks, _ = replay_stream(root)
        split_at = SSE_CHUNKS.index(MID_EVENT_HEAD)
        assert chunks[split_at] == MID_EVENT_HEAD
        assert chunks[split_at + 1] == MID_EVENT_TAIL
        assert b"content_block_delta" not in chunks[split_at]
        assert b"content_block_delta" in chunks[split_at] + chunks[split_at + 1]

    def test_the_usage_chunk_replays_in_its_recorded_position(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        record_stream(root)
        recorded = stream_chunks(recorded_stream(root))

        _, replayed, _ = replay_stream(root)
        usage_positions = [
            index for index, chunk in enumerate(recorded) if b"output_tokens" in chunk
        ]
        assert usage_positions == [
            index for index, chunk in enumerate(replayed) if b"output_tokens" in chunk
        ]
        assert usage_positions == [len(replayed) - 2]
        assert replayed[-1] == SSE_CHUNKS[-1]

    def test_a_mid_stream_upstream_failure_records_the_delivered_chunks_and_the_truncation(
        self, tmp_path: Path
    ) -> None:
        """The provider delivers two chunks and hangs up. The deltas it did send are
        the difference between a stream that died and a request that never streamed,
        so they are recorded, and the recording says the stream never terminated."""
        root = tmp_path / "bundle"
        head, chunks, ending = record_stream(root, abort_after=2)

        assert head.startswith("HTTP/1.1 200 OK")
        assert chunks == list(SSE_CHUNKS[:2])
        assert ending == "truncated"
        recorded = recorded_stream(root)
        assert recorded.status_code == 200
        assert stream_chunks(recorded) == list(SSE_CHUNKS[:2])
        assert recorded.truncated is not None
        assert recorded.truncated.startswith("upstream: ")

    def test_a_downstream_disconnect_mid_relay_records_only_the_delivered_chunks(
        self, tmp_path: Path
    ) -> None:
        """The provider keeps sending, but the proxy the edge relays to hangs up after
        two chunks. The chunk whose downstream write never landed must stay out of the
        recording, or replay would hand back a byte the record run never delivered.

        Driven through the pure ``handle_edge_request`` core because a socket client
        cannot force these tiny chunks to block mid-write, so closing the relay
        generator is the faithful stand-in for the downstream write raising: it lands
        the generator on the same suspended yield a broken pipe would."""
        root = tmp_path / "bundle"
        with chunked_provider() as provider:
            outcome = handle_edge_request(
                record_backend(root),
                {"openai": provider_url(provider)},
                "POST",
                STREAM_PATH,
                {"content-type": "application/json"},
                STREAM_BODY,
                timeout=10.0,
            )
            assert isinstance(outcome, EdgeStream)
            steps = outcome.steps
            first = next(steps)
            second = next(steps)
            assert isinstance(first, StreamChunk) and isinstance(second, StreamChunk)
            assert (first.data, second.data) == (SSE_CHUNKS[0], SSE_CHUNKS[1])
            steps.close()

        recorded = recorded_stream(root)
        assert recorded.status_code == 200
        assert stream_chunks(recorded) == [SSE_CHUNKS[0]]
        assert recorded.truncated == "downstream: relay closed after 1 chunks"

    def test_a_truncated_recording_replays_as_a_truncated_stream(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        record_stream(root, abort_after=2)

        head, chunks, ending = replay_stream(root)
        assert head.startswith("HTTP/1.1 200 OK")
        assert response_header(head, "transfer-encoding") == "chunked"
        assert chunks == list(SSE_CHUNKS[:2])
        assert ending == "truncated"

    def test_a_non_streamed_response_keeps_the_buffered_shape(self, tmp_path: Path) -> None:
        """No-churn guard: an ordinary JSON response records and is framed exactly as
        it was before streaming existed."""
        root = tmp_path / "bundle"
        with fake_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                head, chunks, ending = raw_stream_post(edge.port, CHAT_PATH, chat_body("hi"))

        response = only_recorded_response(root)
        assert isinstance(response, RecordedHttpResponse)
        assert response_header(head, "transfer-encoding") is None
        assert response_header(head, "content-length") is not None
        assert ending == "terminated"
        assert json_object(b"".join(chunks))["echo"] == chat_body("hi").decode()

    def test_a_chunked_non_sse_response_stays_buffered(self, tmp_path: Path) -> None:
        """Detection keys off the content type, not the transfer encoding: providers
        chunk ordinary JSON freely, and treating that as streamed would move nearly
        every recording to the chunk-list shape for no gain."""
        root = tmp_path / "bundle"
        with chunked_provider(chunks=JSON_CHUNKS, content_type="application/json") as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                head, chunks, _ = raw_stream_post(edge.port, CHAT_PATH, chat_body("hi"))

        response = only_recorded_response(root)
        assert isinstance(response, RecordedHttpResponse)
        assert base64.b64decode(response.body_b64) == b"".join(JSON_CHUNKS)
        assert response_header(head, "transfer-encoding") is None
        assert b"".join(chunks) == b"".join(JSON_CHUNKS)

    def test_replay_of_a_stream_makes_no_provider_connection(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        with chunked_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:
                raw_stream_post(edge.port, STREAM_PATH, STREAM_BODY)
            hits_after_record = list(provider.hits)
            with running_edge(
                ReplayEdge(source=replay_source(root)), {"openai": provider_url(provider)}
            ) as edge:
                _, chunks, ending = raw_stream_post(edge.port, STREAM_PATH, STREAM_BODY)
            assert provider.hits == hits_after_record == ["POST /v1/messages"]
        assert chunks == list(SSE_CHUNKS)
        assert ending == "terminated"

    def test_concurrent_streams_each_record_their_own_chunks(self, tmp_path: Path) -> None:
        """The edge relays streams on concurrent threads and each one takes the
        recorder lock once, at the end, so neither recording loses or borrows a chunk
        from the other."""
        root = tmp_path / "bundle"
        bodies = [
            json.dumps({"model": "claude", "stream": True, "n": index}).encode()
            for index in range(2)
        ]
        barrier = threading.Barrier(len(bodies))
        with chunked_provider() as provider:
            with running_edge(record_backend(root), {"openai": provider_url(provider)}) as edge:

                def consume(body: bytes) -> tuple[list[bytes], str]:
                    barrier.wait()
                    _, chunks, ending = raw_stream_post(edge.port, STREAM_PATH, body)
                    return chunks, ending

                with ThreadPoolExecutor(max_workers=len(bodies)) as executor:
                    served = list(executor.map(consume, bodies))

        assert served == [(list(SSE_CHUNKS), "terminated")] * len(bodies)
        files = this_tests_files(root)
        assert len(files) == len(bodies)
        for file in files:
            response = Interaction.model_validate_json(
                file.read_text(encoding="utf-8")
            ).response
            assert isinstance(response, RecordedStreamedResponse), response
            assert stream_chunks(response) == list(SSE_CHUNKS)


class TestHandleEdgeRequestPure:
    def test_unknown_mount_404s_naming_the_known_mounts(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        assert isinstance(prepare_bundle(root), BundleRecorder)
        reply = handle_edge_request(
            ReplayEdge(source=replay_source(root)),
            {"openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com"},
            "POST",
            "/bedrock/model/invoke",
            {},
            b"{}",
            timeout=1.0,
        )
        assert isinstance(reply, EdgeReply)
        assert reply.status_code == 404
        assert b"unknown provider mount 'bedrock'" in reply.body
        assert b"anthropic, openai" in reply.body

    def test_replay_serves_a_directly_recorded_interaction(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        recorder = prepare_bundle(root)
        assert isinstance(recorder, BundleRecorder)
        recorder.record(
            test_key=current_test_key(),
            request=RecordedRequest(method="post", path=CHAT_PATH, headers={}, body={"prompt": "x"}),
            response=RecordedHttpResponse(
                status_code=201, headers={"x-upstream": "fake"}, body_b64=base64.b64encode(b"ok").decode()
            ),
        )
        reply = handle_edge_request(
            ReplayEdge(source=replay_source(root)),
            {"openai": "https://api.openai.com"},
            "POST",
            CHAT_PATH,
            {"authorization": "Bearer sk-anything"},
            json.dumps({"prompt": "x"}).encode(),
            timeout=1.0,
        )
        assert isinstance(reply, EdgeReply)
        assert reply.status_code == 201
        assert reply.body == b"ok"
        assert reply.headers == {"x-upstream": "fake"}


class TestApiBaseSeam:
    def test_live_mode_returns_none(self, tmp_path: Path) -> None:
        for mode_raw in ("live", ""):
            assert (
                provider_edge_api_base(
                    "openai",
                    mode_raw=mode_raw,
                    bundle_dir=tmp_path / "bundle",
                    bind_host="127.0.0.1",
                    advertise_host="127.0.0.1",
                )
                is None
            )

    def test_invalid_mode_raises_naming_the_value(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cached"):
            provider_edge_api_base(
                "openai",
                mode_raw="cached",
                bundle_dir=tmp_path / "bundle",
                bind_host="127.0.0.1",
                advertise_host="127.0.0.1",
            )

    def test_unknown_mount_raises_naming_the_known_mounts(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown provider mount 'bedrock'"):
            provider_edge_api_base(
                "bedrock",
                mode_raw="record",
                bundle_dir=tmp_path / "bundle",
                bind_host="127.0.0.1",
                advertise_host="127.0.0.1",
            )

    def test_record_mode_boots_one_shared_edge_and_prepares_the_bundle(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        first = provider_edge_api_base(
            "openai", mode_raw="record", bundle_dir=root, bind_host="127.0.0.1", advertise_host="127.0.0.1"
        )
        second = provider_edge_api_base(
            "anthropic", mode_raw="record", bundle_dir=root, bind_host="127.0.0.1", advertise_host="127.0.0.1"
        )
        assert first is not None and second is not None
        assert first.endswith("/openai")
        assert second.endswith("/anthropic")
        assert first.rsplit("/", 1)[0] == second.rsplit("/", 1)[0]
        assert (root / "manifest.json").is_file()
