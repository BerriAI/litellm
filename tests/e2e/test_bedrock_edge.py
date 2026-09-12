import base64
import threading
from pathlib import Path
from typing import Final

import pytest
from bedrock_edge import (
    TEST_ACCESS_KEY,
    TEST_SECRET_KEY,
    bedrock_upstream,
    sign_bedrock_forward,
    valid_bedrock_signature,
)
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from fixture_bundle import BundleRecorder, LoadedBundle, RecordedHttpResponse, load_bundle, prepare_bundle
from fixture_mode import current_test_key
from provider_edge import EdgeReply, RecordEdge, ReplayEdge, ReplaySource, edge_request, handle_edge_request

_PATH: Final = "/bedrock-us-east-1/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/converse"
_BODY: Final = b'{"messages":[{"role":"user","content":[{"text":"Hello"}]}]}'


def _signed_headers() -> dict[str, str]:
    request: Final = AWSRequest(
        method="POST",
        url=f"http://127.0.0.1:1234{_PATH}",
        data=_BODY,
        headers={"content-type": "application/json", "host": "127.0.0.1:1234"},
    )
    SigV4Auth(Credentials(TEST_ACCESS_KEY, TEST_SECRET_KEY), "bedrock", "us-east-1").add_auth(request)
    return {key.lower(): value for key, value in request.headers.items()}


class TestBedrockSigningBoundary:
    def test_accepts_a_valid_signature_from_the_proxy(self) -> None:
        assert valid_bedrock_signature("POST", _PATH, _signed_headers(), _BODY, "us-east-1")

    @pytest.mark.parametrize("change", ["body", "path", "region", "host", "authorization"])
    def test_rejects_changes_to_signed_request_inputs(self, change: str) -> None:
        headers: Final = _signed_headers()
        assert not valid_bedrock_signature(
            "POST",
            _PATH + "/tampered" if change == "path" else _PATH,
            {**headers, change: "tampered"} if change in {"host", "authorization"} else headers,
            b"{}" if change == "body" else _BODY,
            "us-west-2" if change == "region" else "us-east-1",
        )

    def test_forward_is_resigned_for_aws_and_preserves_the_body_and_feature_headers(self) -> None:
        url: Final = "https://bedrock-runtime.us-east-1.amazonaws.com/model/example/converse"
        credentials: Final = Credentials("recording-access", "recording-secret", "recording-session")
        incoming: Final = {**_signed_headers(), "anthropic-beta": "feature-flag"}
        forwarded: Final = sign_bedrock_forward("POST", url, incoming, _BODY, "us-east-1", credentials)
        assert "Credential=recording-access/" in forwarded["Authorization"]
        assert TEST_ACCESS_KEY not in forwarded["Authorization"]
        assert forwarded["X-Amz-Security-Token"] == "recording-session"
        assert forwarded["anthropic-beta"] == "feature-flag"
        assert "127.0.0.1" not in str(forwarded)

    @pytest.mark.parametrize(
        "mount", ["bedrock-localhost", "bedrock-us-east-1.evil.test", "bedrock-us-east-1/anything"]
    )
    def test_mount_cannot_select_an_arbitrary_upstream_host(self, mount: str) -> None:
        assert bedrock_upstream(mount) is None

    def test_bad_signature_cannot_reach_aws_or_write_a_recording(self, tmp_path: Path) -> None:
        recorder: Final = prepare_bundle(tmp_path / "bundle")
        assert isinstance(recorder, BundleRecorder)
        result: Final = handle_edge_request(
            RecordEdge(recorder=recorder, lock=threading.Lock()),
            {},
            "POST",
            _PATH,
            _signed_headers(),
            b"tampered",
            timeout=0.1,
        )
        assert isinstance(result, EdgeReply) and result.status_code == 403
        bundle: Final = load_bundle(tmp_path / "bundle")
        assert isinstance(bundle, LoadedBundle) and not bundle.interactions

    def test_replay_validates_signature_before_consuming_the_recording(self, tmp_path: Path) -> None:
        root: Final = tmp_path / "bundle"
        recorder: Final = prepare_bundle(root)
        assert isinstance(recorder, BundleRecorder)
        recorder.record(
            test_key=current_test_key(),
            request=edge_request("POST", _PATH, "", _BODY, "application/json"),
            response=RecordedHttpResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body_b64=base64.b64encode(b'{"output":{"message":{"role":"assistant"}}}').decode(),
            ),
        )
        loaded: Final = load_bundle(root)
        assert isinstance(loaded, LoadedBundle)
        source: Final = ReplaySource(loaded)
        rejected: Final = handle_edge_request(
            ReplayEdge(source),
            {},
            "POST",
            _PATH,
            {},
            _BODY,
            timeout=0.1,
        )
        assert isinstance(rejected, EdgeReply) and rejected.status_code == 403
        result: Final = handle_edge_request(
            ReplayEdge(source),
            {},
            "POST",
            _PATH,
            _signed_headers(),
            _BODY,
            timeout=0.1,
        )
        assert isinstance(result, EdgeReply) and result.status_code == 200
        assert result.body == b'{"output":{"message":{"role":"assistant"}}}'
        assert source.leftover_error(current_test_key()) is None
