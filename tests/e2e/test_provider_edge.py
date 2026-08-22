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
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from e2e_http import RawResponse, forward
from fixture_bundle import (
    BundleRecorder,
    Interaction,
    LoadedBundle,
    RecordedHttpResponse,
    RecordedRequest,
    load_bundle,
    prepare_bundle,
    slug_for_test,
)
from fixture_mode import current_test_key
from provider_edge import (
    REPLAY_MISS_STATUS,
    EdgeBackend,
    ProviderEdge,
    RecordEdge,
    ReplayEdge,
    ReplaySource,
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


def provider_url(server: _FakeProvider) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


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
        assert interaction.request.file_name == "file:batch.jsonl"
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
