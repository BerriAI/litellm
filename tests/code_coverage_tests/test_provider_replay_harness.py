from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Final

import pytest
from fixture_bundle import BundleRecorder, LoadedBundle, load_bundle, prepare_bundle
from fixture_mode import current_test_key
from fixture_profile import MatchProfile
from provider_edge import REPLAY_MISS_STATUS, RecordEdge, ReplayEdge, ReplaySource
from test_provider_edge import (
    CHAT_PATH,
    SSE_CHUNKS,
    STREAM_BODY,
    UPLOAD_PATH,
    call_edge,
    chunked_provider,
    fake_provider,
    json_object,
    provider_url,
    raw_stream_post,
    running_edge,
    this_tests_files,
)


class TestStrictIdentity:
    @pytest.mark.parametrize("path", [CHAT_PATH, "/anthropic/v1/messages"])
    def test_roundtrip_rejects_semantic_changes(self, tmp_path: Path, path: str) -> None:
        recorder: Final = prepare_bundle(tmp_path / "strict", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        original: Final = (
            b'{"model":"synthetic","messages":[{"role":"user",'
            b'"content":"2031-04-05 00000000-0000-0000-0000-000000000001"}],"options":[1,2]}'
        )
        headers: Final = {
            "content-type": "application/json",
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "feature-a",
            "openai-beta": "feature-b",
            "authorization": "Bearer synthetic-secret-one",
        }
        query: Final = "?part=one&part=two&blank="
        with fake_provider() as provider:
            mounts: Final = {"openai": provider_url(provider), "anthropic": provider_url(provider)}
            with running_edge(RecordEdge(recorder, threading.Lock()), mounts) as edge:
                captured: Final = call_edge(edge, "POST", path + query, body=original, headers=headers)
                assert captured.status_code == 200
                assert json_object(captured.body)["echo"] == original.decode()
            loaded: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(loaded, LoadedBundle)
            assert loaded.manifest.match_profile == "stateless_v1"
            source: Final = ReplaySource(loaded)
            with running_edge(ReplayEdge(source), mounts) as edge:
                cases: Final = (
                    (original.replace(b"2031-04-05", b"2032-06-07"), headers, query, "body"),
                    (original.replace(b"000000000001", b"000000000002"), headers, query, "body"),
                    (original.replace(b"[1,2]", b"[2,1]"), headers, query, "body"),
                    (original.replace(b"synthetic", b"other"), headers, query, "body"),
                    (original, headers, "?part=three&part=two&blank=", "query"),
                    (original, headers, "?part=two&part=one&blank=", "query"),
                    *(
                        (original, {k: v for k, v in headers.items() if k != name}, query, "headers")
                        for name in ("accept", "anthropic-version", "anthropic-beta", "openai-beta")
                    ),
                    *(
                        (original, {**headers, name: value}, query, "headers")
                        for name in ("accept", "anthropic-version", "anthropic-beta", "openai-beta")
                        for value in ("different", "")
                    ),
                    (original, {**headers, "authorization": "Basic synthetic-secret-two"}, query, "auth"),
                    (original, {k: v for k, v in headers.items() if k != "authorization"}, query, "auth"),
                )
                for rejected, reason in (
                    (call_edge(edge, "POST", path + changed_query, body=body, headers=changed_headers), reason)
                    for body, changed_headers, changed_query, reason in cases
                ):
                    assert rejected.status_code == REPLAY_MISS_STATUS
                    assert reason in rejected.body.decode()
                    assert b"synthetic-secret" not in rejected.body
                reordered: Final = json.dumps(dict(reversed(list(json_object(original).items())))).encode()
                accepted: Final = call_edge(
                    edge, "POST", path + query, body=reordered, headers={k.upper(): v for k, v in headers.items()}
                )
                assert accepted.status_code == 200
                assert accepted.body == captured.body
            assert source.leftover_error(current_test_key()) is None
            assert len(provider.hits) == 1
        assert "synthetic-secret" not in "".join(file.read_text() for file in recorder.root.rglob("*.json"))

    @pytest.mark.parametrize(
        "body",
        [
            b'{"value":null}',
            b'{"value":""}',
            b'{"value":false}',
            b'{"value":0}',
            b'{"value":[]}',
            b'{"value":{}}',
            b'{"value":0.123456789012345678901}',
            b'{"value":0.123456789012345678902}',
            b'{"value":1e400}',
            b'{"value":1}',
            b'{"value":1e0}',
            b'{"value":-0}',
            b'{"value":1e9999999999999999999}',
        ],
    )
    def test_json_values_remain_distinct(self, tmp_path: Path, body: bytes) -> None:
        recorder: Final = prepare_bundle(tmp_path / "strict", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        with fake_provider() as provider:
            mounts: Final = {"openai": provider_url(provider)}
            with running_edge(RecordEdge(recorder, threading.Lock()), mounts) as edge:
                assert (
                    call_edge(
                        edge, "POST", CHAT_PATH, body=body, headers={"content-type": "application/json"}
                    ).status_code
                    == 200
                )
            loaded: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(loaded, LoadedBundle)
            with running_edge(ReplayEdge(ReplaySource(loaded)), mounts) as edge:
                values: Final = (
                    b"{}",
                    b'{"value":null}',
                    b'{"value":""}',
                    b'{"value":false}',
                    b'{"value":0}',
                    b'{"value":[]}',
                    b'{"value":{}}',
                    b'{"value":0.123456789012345678901}',
                    b'{"value":0.123456789012345678902}',
                    b'{"value":1e400}',
                    b'{"value":1}',
                    b'{"value":1e0}',
                    b'{"value":-0}',
                    b'{"value":1e9999999999999999999}',
                )
                for rejected in (
                    call_edge(edge, "POST", CHAT_PATH, body=value, headers={"content-type": "application/json"})
                    for value in values
                    if value != body
                ):
                    assert rejected.status_code == REPLAY_MISS_STATUS
                    assert b"body" in rejected.body
                assert (
                    call_edge(
                        edge, "POST", CHAT_PATH, body=body, headers={"content-type": "application/json"}
                    ).status_code
                    == 200
                )
            assert len(provider.hits) == 1

    @pytest.mark.parametrize(
        "path,body,headers",
        [
            (UPLOAD_PATH, b"{}", {"content-type": "application/json"}),
            (CHAT_PATH + "?part=%FF", b"{}", {"content-type": "application/json"}),
            (CHAT_PATH + "?part=%FE", b"{}", {"content-type": "application/json"}),
            (CHAT_PATH, b"opaque", {"content-type": "application/octet-stream"}),
            (CHAT_PATH, b"--boundary", {"content-type": "multipart/form-data; boundary=boundary"}),
            (CHAT_PATH, b'{"x":1,"x":2}', {"content-type": "application/json"}),
            (CHAT_PATH, b"{}", {"content-type": "application/json", "x-custom-behavior": "synthetic-private-value"}),
        ],
    )
    def test_ineligible_capture_never_calls_provider(
        self, tmp_path: Path, path: str, body: bytes, headers: dict[str, str]
    ) -> None:
        recorder: Final = prepare_bundle(tmp_path / "strict", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        with fake_provider() as provider:
            with running_edge(RecordEdge(recorder, threading.Lock()), {"openai": provider_url(provider)}) as edge:
                result: Final = call_edge(edge, "POST", path, body=body, headers=headers)
                assert result.status_code == REPLAY_MISS_STATUS
                assert b"eligibility error" in result.body
                assert b"synthetic-private-value" not in result.body
            assert provider.hits == []
        assert this_tests_files(recorder.root) == []

    def test_destination_is_part_of_actual_http_identity(self, tmp_path: Path) -> None:
        recorder: Final = prepare_bundle(tmp_path / "strict", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        with fake_provider() as provider:
            with running_edge(RecordEdge(recorder, threading.Lock()), {"openai": provider_url(provider)}) as edge:
                assert (
                    call_edge(
                        edge, "POST", CHAT_PATH, body=b"{}", headers={"content-type": "application/json"}
                    ).status_code
                    == 200
                )
            loaded: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(loaded, LoadedBundle)
            with running_edge(ReplayEdge(ReplaySource(loaded)), {"openai": provider_url(provider) + "/other"}) as edge:
                result: Final = call_edge(
                    edge, "POST", CHAT_PATH, body=b"{}", headers={"content-type": "application/json"}
                )
                assert result.status_code == REPLAY_MISS_STATUS
                assert b"upstream" in result.body
            assert len(provider.hits) == 1

    def test_credentials_are_not_identity_and_fresh_process_replays(self, tmp_path: Path) -> None:
        recorder: Final = prepare_bundle(tmp_path / "strict", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        headers: Final = {
            "content-type": "application/json",
            "authorization": "bEaReR synthetic-token",
            "x-api-key": "synthetic-api-key",
            "cookie": "synthetic-cookie",
        }
        path: Final = CHAT_PATH + "?api_key=synthetic-query-secret&part=one&part=two"
        body: Final = b'{"model":"synthetic","messages":[]}'
        with fake_provider(echo_request=False) as provider:
            mounts: Final = {"openai": provider_url(provider)}
            with running_edge(RecordEdge(recorder, threading.Lock()), mounts) as edge:
                captured: Final = call_edge(edge, "POST", path, body=body, headers=headers)
                assert captured.status_code == 200
            seen_headers, seen_body = provider.requests[0]
            assert {k.lower(): v for k, v in seen_headers.items()}.items() >= headers.items()
            assert seen_body == body
            assert provider.hits == ["POST " + path.removeprefix("/openai")]
            artifacts: Final = "".join(file.read_text() for file in recorder.root.rglob("*.json"))
            for secret in ("synthetic-token", "synthetic-api-key", "synthetic-cookie", "synthetic-query-secret"):
                assert secret not in artifacts
            child: Final = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import json, sys
from pathlib import Path
from fixture_bundle import LoadedBundle, load_bundle
from provider_edge import ProviderRequestObservation, observed_provider_edge, replay_leftover_error
from test_provider_edge import call_edge
from fixture_profile import MatchProfile
from fixture_mode import current_test_key
loaded = load_bundle(Path(sys.argv[1]), profile="stateless_v1")
assert isinstance(loaded, LoadedBundle)
with observed_provider_edge(ProviderRequestObservation("synthetic"), mode_raw="replay", bundle_dir=Path(sys.argv[1]), bind_host="127.0.0.1", advertise_host="127.0.0.1", mounts={"openai": sys.argv[2]}) as edge:
    response = call_edge(edge, "POST", sys.argv[3], body=sys.argv[4].encode(), headers=json.loads(sys.argv[5]))
    assert response.status_code == 200
    print(response.body.decode())
assert replay_leftover_error(mode_raw="replay", bundle_dir=Path(sys.argv[1]), test_key=current_test_key()) is None
""",
                    str(recorder.root),
                    provider_url(provider),
                    path.replace("synthetic-query-secret", "new-query-credential"),
                    body.decode(),
                    json.dumps({**headers, "authorization": "Bearer another-credential", "x-api-key": "another-key"}),
                ],
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "e2e"),
                    "E2E_REPLAY_MATCH_PROFILE": "stateless_v1",
                },
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert child.returncode == 0, child.stderr
            assert child.stdout.strip().encode() == captured.body
            assert len(provider.hits) == 1

    @pytest.mark.parametrize("profile,other", [("legacy", "stateless_v1"), ("stateless_v1", "legacy")])
    def test_profiles_cannot_load_each_others_bundles(
        self, tmp_path: Path, profile: MatchProfile, other: MatchProfile
    ) -> None:
        from fixture_bundle import UnreadableBundle

        recorder: Final = prepare_bundle(tmp_path / profile, profile=profile)
        assert isinstance(recorder, BundleRecorder)
        mismatch: Final = load_bundle(recorder.root, profile=other)
        assert isinstance(mismatch, UnreadableBundle)
        assert "profile mismatch" in mismatch.reason
        assert "re-record" in mismatch.reason

    @pytest.mark.parametrize("abort_after", [None, 2])
    def test_strict_stream_preserves_chunks_and_truncation(self, tmp_path: Path, abort_after: int | None) -> None:
        recorder: Final = prepare_bundle(tmp_path / "strict", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        with chunked_provider(abort_after=abort_after) as provider:
            mounts: Final = {"anthropic": provider_url(provider)}
            with running_edge(RecordEdge(recorder, threading.Lock()), mounts) as edge:
                _, captured, captured_ending = raw_stream_post(edge.port, "/anthropic/v1/messages", STREAM_BODY)
            loaded: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(loaded, LoadedBundle)
            source: Final = ReplaySource(loaded)
            with running_edge(ReplayEdge(source), mounts) as edge:
                _, replayed, ending = raw_stream_post(edge.port, "/anthropic/v1/messages", STREAM_BODY)
            assert captured == replayed == list(SSE_CHUNKS[:abort_after])
            assert ending == captured_ending
            assert (ending == "terminated") == (abort_after is None)
            assert source.leftover_error(current_test_key()) is None
            assert len(provider.hits) == 1

    def test_auth_scheme_survives_missing_credentials(self, tmp_path: Path) -> None:
        recorder: Final = prepare_bundle(tmp_path / "strict", profile="stateless_v1")
        assert isinstance(recorder, BundleRecorder)
        headers: Final = {"content-type": "application/json", "authorization": "Bearer"}
        with fake_provider() as provider:
            mounts: Final = {"openai": provider_url(provider)}
            with running_edge(RecordEdge(recorder, threading.Lock()), mounts) as edge:
                assert call_edge(edge, "POST", CHAT_PATH, body=b"{}", headers=headers).status_code == 200
            loaded: Final = load_bundle(recorder.root, profile="stateless_v1")
            assert isinstance(loaded, LoadedBundle)
            with running_edge(ReplayEdge(ReplaySource(loaded)), mounts) as edge:
                for result in (
                    call_edge(edge, "POST", CHAT_PATH, body=b"{}", headers={**headers, "authorization": scheme})
                    for scheme in ("Basic", "Digest")
                ):
                    assert result.status_code == REPLAY_MISS_STATUS
                    assert b"auth" in result.body
                assert (
                    call_edge(
                        edge,
                        "POST",
                        CHAT_PATH,
                        body=b"{}",
                        headers={**headers, "authorization": "bEaReR synthetic-token"},
                    ).status_code
                    == 200
                )
            assert len(provider.hits) == 1
