"""GCS bucket and SQS sink contracts: object names and bodies on Google Cloud Storage, SendMessage bodies on SQS.

GCS: the `gcs_bucket` callback is a premium feature and always uploads to `https://storage.googleapis.com/upload/
storage/v1/b/<bucket>/o?uploadType=media&name=<object>` (litellm/integrations/gcs_bucket/gcs_bucket_base.py). The
rig diverts that host the way `test_batch_line_items_cold_storage.py` does: `HTTPS_PROXY` points at a tunnelling
forward proxy that hands every CONNECT to a TLS peer presenting a self signed `storage.googleapis.com` certificate
trusted through `SSL_CERT_FILE`, and the service account's token URI points at a plain scripted token endpoint.
With batching on (the default) a flush writes `<YYYY-MM-DD>/batch-<id>.ndjson` holding one payload per line; with
`GCS_USE_BATCHED_LOGGING=false` each success is `<YYYY-MM-DD>/<response id>`, each failure
`<YYYY-MM-DD>/failure-<hex>`, and `metadata.gcs_log_id` names `<YYYY-MM-DD>/custom-<hex>-<id>`. A bucket name with a
folder suffix moves the folder into the object name.

SQS: the `aws_sqs` callback signs a form encoded `Action=SendMessage&Version=2012-11-05&MessageBody=<json>` POST to the
configured queue URL with SigV4 for service `sqs` (litellm/integrations/sqs.py), one message per payload, so the
queue URL can point straight at a scripted peer. With application level encryption the body is
`{"__encrypted__": true, "payload": ...}` and no prompt text leaves the proxy.

Both are skipped without what they need (LITELLM_LICENSE for GCS). A sink outage must never fail the caller.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import owned_proxy
from integration._support.timing import assert_within_factor_of_control, timed
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from integration.observability.dec1_callbacks_support import (
    CHAT_REJECTED,
    PROVIDER_REJECTION,
    REPLY_TEXT,
    ScriptedProvider,
    write_config,
)
from integration.observability.dec1_sinks_support import (
    SQS_ACCESS_KEY,
    SQS_QUEUE_PATH,
    SinkRig,
    assert_not_held,
    chat,
    healthy_seconds,
    marker,
    new_call_id,
    sinks_rig,
    timed_chat,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

GCS_BUCKET: Final = "dec1-sinks-gcs"
GCS_FOLDER: Final = "dec1-folder"
ENCRYPTION_KEY: Final = base64.b64encode(b"d" * 32).decode()
NEEDS_LICENSE: Final = "gcs_bucket logging is a premium feature and needs LITELLM_LICENSE"


# ---------------------------------------------------------------- GCS rig


@dataclass
class Bucket:
    """Objects uploaded to the diverted GCS peer by name, plus each upload's headers."""

    objects: dict[str, bytes] = field(default_factory=dict)
    headers: dict[str, dict[str, str]] = field(default_factory=dict)
    failing: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, bytes]:
        with self.lock:
            return dict(self.objects)


def _gcs_peer(bucket: Bucket) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "POST" and path == "/_oauth/token":
            return Reply(body=b'{"access_token":"dec1-scripted-token","expires_in":3600,"token_type":"Bearer"}')
        if request.method == "POST" and path == f"/upload/storage/v1/b/{GCS_BUCKET}/o":
            if bucket.failing:
                return Reply(status=503, body=b'{"error":"dec1 scripted gcs outage"}')
            name: Final = unquote(parse_qs(urlsplit(request.target).query)["name"][0])
            with bucket.lock:
                bucket.objects[name] = request.body
                bucket.headers[name] = dict(request.headers)
            return Reply(body=json.dumps({"name": name, "bucket": GCS_BUCKET}).encode())
        return Reply(status=404, body=b'{"error":"not scripted"}')

    return respond


@dataclass(frozen=True)
class GcsRig:
    proxy: Gateway
    bucket: Bucket

    def object_with(self, needle: str, *, seconds: float = 60) -> tuple[str, bytes]:
        found: Final = eventually(
            lambda: tuple((name, body) for name, body in self.bucket.snapshot().items() if needle.encode() in body),
            lambda objects: len(objects) >= 1,
            seconds=seconds,
        )
        assert len(found) == 1, [name for name, _ in found]
        return found[0]


def _gcs_rig(directory: Path, batched: bool, bucket_name: str) -> Iterator[GcsRig]:
    if not os.environ.get("LITELLM_LICENSE"):
        pytest.skip(NEEDS_LICENSE)
    certificate, key = write_self_signed_cert(directory, names=("storage.googleapis.com",))
    bucket: Final = Bucket()
    with (
        wire_server(ScriptedProvider()) as provider,
        wire_server(_gcs_peer(bucket)) as token_wire,
        wire_server(_gcs_peer(bucket), tls=server_context(certificate, key)) as gcs_wire,
        tunnelling_forward_proxy(int(urlsplit(gcs_wire.url).port or 0)) as tunnel,
        gateway_from_environment() as shared,
    ):
        account: Final = directory / "service-account.json"
        account.write_text(service_account_json("dec1-sinks-project", token_wire.url))
        config: Final = write_config(directory / "gcs.yaml", provider.url, {"callbacks": ["gcs_bucket"]})
        overrides: Final = {
            "GCS_BUCKET_NAME": bucket_name,
            "GCS_PATH_SERVICE_ACCOUNT": str(account),
            "GCS_FLUSH_INTERVAL": "1",
            "GCS_USE_BATCHED_LOGGING": "true" if batched else "false",
            "HTTPS_PROXY": tunnel.url,
            "https_proxy": tunnel.url,
            "SSL_CERT_FILE": str(certificate),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "STORE_MODEL_IN_DB": "False",
        }
        with owned_proxy(shared, directory, overrides, config=config) as proxy:
            yield GcsRig(proxy, bucket)


@pytest.fixture(scope="module")
def gcs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[GcsRig]:
    yield from _gcs_rig(tmp_path_factory.mktemp("dec1-sinks-gcs"), False, f"{GCS_BUCKET}/{GCS_FOLDER}")


@pytest.fixture(scope="module")
def gcs_batched(tmp_path_factory: pytest.TempPathFactory) -> Iterator[GcsRig]:
    yield from _gcs_rig(tmp_path_factory.mktemp("dec1-sinks-gcs-batched"), True, GCS_BUCKET)


def _gcs_chat(rig: GcsRig, text: str, *, model: str = "dec1-chat", **extra: JsonValue) -> dict[str, JsonValue]:
    response: Final = rig.proxy.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    )
    return {"status": response.status_code, "body": response.json()}


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def test_gcs_success_object_is_named_by_date_and_response_id_under_the_folder(gcs: GcsRig) -> None:
    """With batching off and GCS_BUCKET_NAME "dec1-sinks-gcs/dec1-folder", a successful chat must upload to bucket
    "dec1-sinks-gcs" one object named "dec1-folder/<UTC date>/<response id>"."""
    text: Final = marker()
    result: Final = _gcs_chat(gcs, text)
    assert result["status"] == 200
    name, _ = gcs.object_with(text)
    assert name == f"{GCS_FOLDER}/{_today()}/{result['body']['id']}"  # type: ignore[index]


def test_gcs_object_body_is_the_standard_logging_payload(gcs: GcsRig) -> None:
    """The uploaded object must be one JSON StandardLoggingPayload with status "success", the user text in messages
    and the reply in response."""
    text: Final = marker()
    assert _gcs_chat(gcs, text)["status"] == 200
    _, body = gcs.object_with(text)
    payload: Final = json.loads(body)
    assert payload["status"] == "success"
    assert REPLY_TEXT in json.dumps(payload["response"])


def test_gcs_upload_carries_the_service_account_bearer_token(gcs: GcsRig) -> None:
    """The upload must carry `Authorization: Bearer dec1-scripted-token`, the token the service account exchanged at
    its token URI, and content-type application/json."""
    text: Final = marker()
    assert _gcs_chat(gcs, text)["status"] == 200
    name, _ = gcs.object_with(text)
    headers: Final = gcs.bucket.headers[name]
    assert headers["authorization"] == "Bearer dec1-scripted-token"
    assert headers["content-type"] == "application/json"


def test_gcs_failure_object_is_named_failure_hex(gcs: GcsRig) -> None:
    """A provider 400 must upload an object named "dec1-folder/<UTC date>/failure-<32 hex>" whose payload has status
    "failure" and the provider's error text in error_str."""
    text: Final = marker()
    assert _gcs_chat(gcs, text, model=CHAT_REJECTED)["status"] == 400
    name, body = gcs.object_with(text)
    prefix: Final = f"{GCS_FOLDER}/{_today()}/failure-"
    assert name.startswith(prefix)
    assert len(name.removeprefix(prefix)) == 32
    payload: Final = json.loads(body)
    assert payload["status"] == "failure"
    assert PROVIDER_REJECTION in str(payload["error_str"])


def test_gcs_log_id_metadata_names_a_custom_object(gcs: GcsRig) -> None:
    """Request metadata gcs_log_id "dec1-log" must name the object "dec1-folder/<UTC date>/custom-<32 hex>-dec1-log"."""
    text: Final = marker()
    assert _gcs_chat(gcs, text, metadata={"gcs_log_id": "dec1-log"})["status"] == 200
    name, _ = gcs.object_with(text)
    head: Final = f"{GCS_FOLDER}/{_today()}/custom-"
    assert name.startswith(head)
    assert name.endswith("-dec1-log")


def test_gcs_outage_never_fails_or_slows_the_request(gcs: GcsRig) -> None:
    """With the GCS peer answering 503 to uploads, a chat must still return 200 and take no longer than a healthy
    control call times 3 plus one second, since uploads run from the periodic flush task."""
    control: Final = timed(lambda: _gcs_chat(gcs, marker()))
    assert control.value["status"] == 200
    gcs.bucket.failing = True
    try:
        outage = timed(lambda: _gcs_chat(gcs, marker()))
        assert outage.value["status"] == 200
        assert_within_factor_of_control(outage.seconds, control=control.seconds + 1, factor=3)
    finally:
        gcs.bucket.failing = False


def test_gcs_batched_logging_writes_ndjson_batch_objects(gcs_batched: GcsRig) -> None:
    """With batching on (the default), two chats must land in object(s) named "<UTC date>/batch-<id>.ndjson" whose
    lines each parse to one payload, one line per call."""
    first, second = marker(), marker()
    assert _gcs_chat(gcs_batched, first)["status"] == 200
    assert _gcs_chat(gcs_batched, second)["status"] == 200
    for text in (first, second):
        name, body = gcs_batched.object_with(text)
        assert name.startswith(f"{_today()}/batch-")
        assert name.endswith(".ndjson")
        lines = [json.loads(line) for line in body.decode().splitlines() if line.strip()]
        assert sum(text in json.dumps(line["messages"]) for line in lines) == 1


# ---------------------------------------------------------------- SQS


@pytest.fixture(scope="module")
def sqs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-sqs"),
        {
            "callbacks": ["aws_sqs"],
            "aws_sqs_callback_params": {
                "sqs_queue_url": "os.environ/SQS_QUEUE_URL",
                "sqs_region_name": "us-east-1",
                "sqs_strip_base64_files": True,
            },
        },
    ) as created:
        yield created


@pytest.fixture(scope="module")
def sqs_encrypted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-sqs-encrypted"),
        {
            "callbacks": ["aws_sqs"],
            "aws_sqs_callback_params": {
                "sqs_queue_url": "os.environ/SQS_QUEUE_URL",
                "sqs_region_name": "us-east-1",
                "sqs_aws_use_application_level_encryption": True,
                "sqs_app_encryption_key_b64": ENCRYPTION_KEY,
            },
        },
    ) as created:
        yield created


def _sqs_message(rig: SinkRig, call_id: str) -> dict[str, JsonValue]:
    found: Final = eventually(
        lambda: tuple(message for message in rig.sqs_messages() if message.get("litellm_call_id") == call_id),
        lambda messages: len(messages) >= 1,
        seconds=60,
    )
    assert len(found) == 1
    return found[0]


def test_sqs_send_message_is_form_encoded_with_action_and_version(sqs: SinkRig) -> None:
    """Each SQS post must be `application/x-www-form-urlencoded` with Action "SendMessage", Version "2012-11-05" and a
    MessageBody, sent to the queue URL path."""
    call_id: Final = new_call_id()
    assert chat(sqs, call_id, marker()).status_code == 200
    _sqs_message(sqs, call_id)
    for request in sqs.peer("sqs").posts(SQS_QUEUE_PATH):
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        form = parse_qs(request.body.decode())
        assert form["Action"] == ["SendMessage"]
        assert form["Version"] == ["2012-11-05"]
        assert "MessageBody" in form


def test_sqs_request_is_sigv4_signed_for_the_sqs_service(sqs: SinkRig) -> None:
    """Each SQS post must carry `Authorization: AWS4-HMAC-SHA256 Credential=<access key>/<date>/us-east-1/sqs/
    aws4_request, ...` and an x-amz-date header, and must never carry the secret key."""
    call_id: Final = new_call_id()
    assert chat(sqs, call_id, marker()).status_code == 200
    _sqs_message(sqs, call_id)
    for request in sqs.peer("sqs").posts(SQS_QUEUE_PATH):
        authorization = request.headers["authorization"]
        assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={SQS_ACCESS_KEY}/")
        assert "/us-east-1/sqs/aws4_request" in authorization
        assert "x-amz-date" in request.headers


def test_sqs_sends_one_message_per_payload(sqs: SinkRig) -> None:
    """Two chats must produce exactly one SQS message each, each MessageBody a single StandardLoggingPayload object
    with that call's litellm_call_id and status "success"."""
    calls: Final = (new_call_id(), new_call_id())
    for call_id in calls:
        assert chat(sqs, call_id, marker()).status_code == 200
    for call_id in calls:
        assert _sqs_message(sqs, call_id)["status"] == "success"


def test_sqs_failure_message_has_failure_status_and_error(sqs: SinkRig) -> None:
    """A provider 400 must produce one SQS message with status "failure" and the provider text in error_str."""
    call_id: Final = new_call_id()
    assert chat(sqs, call_id, marker(), model=CHAT_REJECTED).status_code == 400
    message: Final = _sqs_message(sqs, call_id)
    assert message["status"] == "failure"
    assert PROVIDER_REJECTION in str(message["error_str"])


def test_sqs_strip_base64_files_removes_inline_data_urls(sqs: SinkRig) -> None:
    """With `sqs_strip_base64_files: true`, a message with an inline `data:image/png;base64,...` image must reach SQS
    without the base64 data, while the request itself succeeds. UNVERIFIED: the replacement text left in place."""
    call_id: Final = new_call_id()
    data: Final = "iVBORw0KGgo" + "A" * 200
    content: Final = [
        {"type": "text", "text": marker()},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
    ]
    response: Final = sqs.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": "dec1-chat", "messages": [{"role": "user", "content": content}]},
        headers={"x-litellm-call-id": call_id},
    )
    assert response.status_code == 200
    assert data not in json.dumps(_sqs_message(sqs, call_id))


def test_sqs_application_encryption_hides_the_payload(sqs_encrypted: SinkRig) -> None:
    """With application level encryption on, every MessageBody must be {"__encrypted__": true, "payload": {...}}
    where the payload is the AES-GCM envelope of three base64 strings `nonce`, `ciphertext` and `tag`
    (`AppCrypto.encrypt_json`, litellm/litellm_core_utils/app_crypto.py:18-28, sent by litellm/integrations/sqs.py:
    280-283), and neither the user text nor the call id may appear in any SQS post."""
    call_id, text = new_call_id(), marker()
    assert chat(sqs_encrypted, call_id, text).status_code == 200
    messages: Final = eventually(sqs_encrypted.sqs_messages, lambda found: len(found) >= 1, seconds=60)
    for message in messages:
        assert message["__encrypted__"] is True
        payload = message["payload"]
        assert isinstance(payload, dict), message
        assert sorted(payload) == ["ciphertext", "nonce", "tag"], message
        assert all(isinstance(value, str) and value for value in payload.values()), message
    held(
        lambda: tuple(
            r for r in sqs_encrypted.peer("sqs").posts() if text.encode() in r.body or call_id.encode() in r.body
        ),
        lambda found: not found,
        holding=3,
        seconds=10,
    )


def test_sqs_outage_never_fails_or_slows_the_request(sqs: SinkRig) -> None:
    """With the SQS peer answering 500, and then holding every post, a chat must still return 200 within the healthy control time plus the 30 second stall."""
    control: Final = healthy_seconds(sqs)
    peer: Final = sqs.peer("sqs")
    peer.switch.status = 500
    try:
        response, elapsed = timed_chat(sqs, new_call_id(), marker())
        assert response.status_code == 200
        assert_not_held(elapsed, control)
    finally:
        peer.switch.status = 200
    peer.switch.stall()
    try:
        response, elapsed = timed_chat(sqs, new_call_id(), marker())
        assert response.status_code == 200
        assert_not_held(elapsed, control)
    finally:
        peer.switch.release()
