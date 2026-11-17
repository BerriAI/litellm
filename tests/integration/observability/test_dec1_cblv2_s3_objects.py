"""S3 logging (`s3_v2`, now served by the Rust callback behind CustomLoggerV2, #51539 and #51893) must keep its
object keys, bodies, headers and batching exactly as the Python S3Logger wrote them.

Contract, checked against a scripted path-style S3 peer:
- one object per logged attempt at `<s3_path>/<YYYY-MM-DD>/time-HH-MM-SS-ffffff_<payload id>.json`, the date and
  clock taken in UTC from the call start, `/` and `:` in the id replaced by `_`
- the body is the whole StandardLoggingPayload as JSON, so readers (cold storage lookups, Athena tables) parse it
- each PUT carries Content-Type application/json, Content-MD5, x-amz-content-sha256, Content-Language en,
  Content-Disposition inline with a `time-<date>T<clock>_<id>.json` filename, the immutable Cache-Control, the
  SSE headers when configured, and a SigV4 Authorization for the configured access key
- `s3_use_team_prefix` and `s3_use_key_prefix` put the team alias then the key alias before the date, with
  Python `bool()` semantics, so the string "false" still turns the prefix on (spec decision D10)
- `s3_log_prompts_only` nulls `response`; `s3_strip_base64_files` removes inline base64 data
- `s3_batch_file_upload` writes one NDJSON object per folder per flush
- failures are logged too; `no-log` and `x-litellm-disable-callbacks: s3_v2` (licensed) skip the object
- audit logs go to `<s3_path>/audit_logs/<date>/<HH-MM-SS>_<id>.json`

Line numbers refer to litellm/rust/crates/callbacks-s3-log/src (key.rs, upload.rs, settings.rs, sink.rs).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually, held, object_value
from integration.observability.dec1_cblv2_support import (
    CHAT,
    CHAT_COST,
    CHAT_REJECTED,
    S3_ACCESS_KEY,
    S3_PATH,
    Rig,
    S3Put,
    cblv2_rig,
    chat,
    marker,
    stream_chat,
)
from pydantic import JsonValue

LICENSED: Final = bool(os.environ.get("LITELLM_LICENSE"))
KMS_KEY: Final = "synthetic-cblv2-kms-key"
CACHE_CONTROL: Final = "private, immutable, max-age=31536000, s-maxage=0"
DATE: Final = r"\d{4}-\d{2}-\d{2}"
CLOCK: Final = r"\d{2}-\d{2}-\d{2}-\d{6}"
PIXEL: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """Default `s3_v2` settings, plus audit logs routed to the same logger."""
    directory: Final = tmp_path_factory.mktemp("cblv2-s3-plain")
    with cblv2_rig(
        directory,
        {"callbacks": ["s3_v2"], "audit_log_callbacks": ["s3_v2"], "store_audit_logs": True},
        s3_params={},
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def shaped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """Team prefix given as the string "false", key prefix on, SSE with a KMS key, prompts only, base64 stripped."""
    directory: Final = tmp_path_factory.mktemp("cblv2-s3-shaped")
    with cblv2_rig(
        directory,
        {"callbacks": ["s3_v2"]},
        s3_params={
            "s3_use_team_prefix": "false",
            "s3_use_key_prefix": True,
            "s3_server_side_encryption": "aws:kms",
            "s3_sse_kms_key_id": KMS_KEY,
            "s3_log_prompts_only": True,
            "s3_strip_base64_files": True,
        },
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def batched(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """`s3_batch_file_upload: true` with global `turn_off_message_logging`."""
    directory: Final = tmp_path_factory.mktemp("cblv2-s3-batched")
    with cblv2_rig(
        directory,
        {"callbacks": ["s3_v2"], "turn_off_message_logging": True},
        s3_params={"s3_batch_file_upload": True},
    ) as booted:
        yield booted


def _object(rig: Rig, response_id: str, seconds: float = 60) -> S3Put:
    """The single object whose body holds the payload with this id."""
    found: Final = eventually(lambda: rig.s3.objects_for(response_id), bool, seconds=seconds)
    assert len(found) == 1, [put.key for put in found]
    return found[0]


def _payload(put: S3Put, response_id: str) -> dict[str, JsonValue]:
    matching: Final = [line for line in put.lines() if line.get("id") == response_id]
    assert len(matching) == 1, put.key
    return matching[0]


def _ok(rig: Rig, text: str | None = None, **extra: JsonValue) -> str:
    response: Final = chat(rig, text or marker(), **extra)
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def test_successful_chat_writes_one_object_under_the_daily_key(plain: Rig) -> None:
    """One chat must produce exactly one PUT whose key is `cblv2-logs/<YYYY-MM-DD>/time-HH-MM-SS-ffffff_<id>.json`
    where <id> is the chat completion id the caller received (key.rs:94-127 and logging_id at key.rs:90)."""
    response_id: Final = _ok(plain)
    put: Final = _object(plain, response_id)
    assert re.fullmatch(rf"{S3_PATH}/{DATE}/time-{CLOCK}_{re.escape(response_id)}\.json", put.key), put.key


def test_object_is_put_path_style_on_the_configured_bucket(plain: Rig) -> None:
    """With `s3_endpoint_url` set and virtual-hosted style off, the PUT target is `/<bucket>/<key>` on the endpoint
    (object_url, key.rs:220-232)."""
    response_id: Final = _ok(plain)
    put: Final = _object(plain, response_id)
    assert put.target.startswith("/cblv2-bucket/cblv2-logs/"), put.target


def test_object_body_is_the_standard_logging_payload(plain: Rig) -> None:
    """The object body must be one JSON StandardLoggingPayload: id equals the response id, call_type is
    "acompletion", status "success", model_group "cblv2-chat", total_tokens 15 and response_cost 0.019."""
    response_id: Final = _ok(plain)
    payload: Final = _payload(_object(plain, response_id), response_id)
    assert payload["call_type"] == "acompletion"
    assert payload["status"] == "success"
    assert payload["model_group"] == CHAT
    assert payload["total_tokens"] == 15
    assert payload["response_cost"] == pytest.approx(CHAT_COST)


def test_object_body_keeps_the_request_messages_and_the_response(plain: Rig) -> None:
    """Without prompts-only, the payload must carry `messages` with the caller's text and `response` with the reply
    the caller received, byte compatible with what downstream readers parse."""
    text: Final = marker()
    response_id: Final = _ok(plain, text)
    payload: Final = _payload(_object(plain, response_id), response_id)
    assert text in json.dumps(payload["messages"])
    assert object_value(payload["response"])["id"] == response_id


def test_object_body_carries_the_call_id_from_the_response_header(plain: Rig) -> None:
    """The payload's `litellm_call_id` must equal the `x-litellm-call-id` response header of the call."""
    response: Final = chat(plain, marker())
    assert response.status_code == 200
    payload: Final = _payload(_object(plain, response.json()["id"]), response.json()["id"])
    assert payload["litellm_call_id"] == response.headers["x-litellm-call-id"]


def test_put_headers_are_the_python_header_set(plain: Rig) -> None:
    """The PUT must carry Content-Type application/json, Content-Language en and Cache-Control
    "private, immutable, max-age=31536000, s-maxage=0" (upload.rs:205-218)."""
    response_id: Final = _ok(plain)
    headers: Final = _object(plain, response_id).headers
    assert headers["content-type"] == "application/json"
    assert headers["content-language"] == "en"
    assert headers["cache-control"] == CACHE_CONTROL


def test_put_digests_match_the_body(plain: Rig) -> None:
    """`Content-MD5` must be the base64 MD5 of the exact body bytes and `x-amz-content-sha256` their hex SHA-256,
    so S3 can verify the upload."""
    response_id: Final = _ok(plain)
    put: Final = _object(plain, response_id)
    assert put.headers["content-md5"] == base64.b64encode(hashlib.md5(put.body).digest()).decode()
    assert put.headers["x-amz-content-sha256"] == hashlib.sha256(put.body).hexdigest()


def test_content_disposition_names_the_download_with_date_clock_and_id(plain: Rig) -> None:
    """`Content-Disposition` must be `inline; filename="time-<YYYY-MM-DD>T<HH-MM-SS-ffffff>_<id>.json"`
    (download_name, key.rs:130-136)."""
    response_id: Final = _ok(plain)
    disposition: Final = _object(plain, response_id).headers["content-disposition"]
    assert re.fullmatch(rf'inline; filename="time-{DATE}T{CLOCK}_{re.escape(response_id)}\.json"', disposition), (
        disposition
    )


def test_put_is_signed_with_the_configured_access_key(plain: Rig) -> None:
    """The PUT's Authorization header must be a SigV4 signature for the configured access key id in us-east-1 for
    the s3 service: it starts with `AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/` and names `us-east-1/s3`."""
    response_id: Final = _ok(plain)
    authorization: Final = _object(plain, response_id).headers["authorization"]
    assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={S3_ACCESS_KEY}/"), authorization
    assert "/us-east-1/s3/aws4_request" in authorization


def test_default_settings_send_no_server_side_encryption_headers(plain: Rig) -> None:
    """Without SSE settings the PUT must carry neither `x-amz-server-side-encryption` nor the KMS key id header."""
    response_id: Final = _ok(plain)
    headers: Final = _object(plain, response_id).headers
    assert "x-amz-server-side-encryption" not in headers
    assert "x-amz-server-side-encryption-aws-kms-key-id" not in headers


def test_default_settings_put_no_team_or_key_prefix(plain: Rig) -> None:
    """With prefixes off, a call on a key with alias A in a team with alias T must still be written directly under
    `cblv2-logs/<date>/`, with neither alias in the key."""
    with plain.proxy.scenario() as scenario:
        team_alias: Final = f"cblv2-team-{uuid.uuid4().hex[:8]}"
        key_alias: Final = f"cblv2-key-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(key_alias=key_alias, team_id=scenario.team(team_alias=team_alias))
        response_id: Final = _ok(plain, key=key)
        put: Final = _object(plain, response_id)
    assert team_alias not in put.key and key_alias not in put.key
    assert re.match(rf"{S3_PATH}/{DATE}/time-", put.key), put.key


def test_streamed_chat_writes_one_object_with_the_assembled_payload(plain: Rig) -> None:
    """A streamed chat read to the end must produce exactly one object (not one per chunk) whose payload has
    stream true and total_tokens 15. The payload id is the chunk id."""
    status, _, sse = stream_chat(plain, marker())
    assert status == 200
    chunk_id: Final = json.loads(sse.splitlines()[0].removeprefix("data: "))["id"]
    payload: Final = _payload(_object(plain, chunk_id), chunk_id)
    assert payload["stream"] is True
    assert payload["total_tokens"] == 15


def test_provider_failure_writes_one_failure_object(plain: Rig) -> None:
    """A chat the provider answers with 400 must produce one object whose payload has status "failure" and
    error_information.error_code "400". Its id falls back to the call id, so the key ends with
    `_<x-litellm-call-id>.json`."""
    response: Final = chat(plain, marker(), model=CHAT_REJECTED)
    assert response.status_code == 400
    call_id: Final = response.headers["x-litellm-call-id"]
    put: Final = _object(plain, call_id)
    payload: Final = _payload(put, call_id)
    assert payload["status"] == "failure"
    assert object_value(payload["error_information"])["error_code"] == "400"
    assert put.key.endswith(f"_{call_id}.json"), put.key


def test_no_log_request_writes_no_object(plain: Rig) -> None:
    """A chat with body `no-log: true` must produce no object for its response id, while a normal chat right after
    it is written. s3_v2 declares `honors_no_log=True` in its V2 manifest (s3-log-v3 spec section 3)."""
    skipped: Final = _ok(plain, **{"no-log": True})
    logged: Final = _ok(plain)
    _object(plain, logged)
    held(lambda: plain.s3.objects_for(skipped), lambda found: found == (), holding=3)


def test_disable_callbacks_header_skips_the_object(plain: Rig) -> None:
    """On a licensed proxy, `x-litellm-disable-callbacks: s3_v2` on one chat must produce no object for it, while
    the next chat without the header is written."""
    if not LICENSED:
        pytest.skip("dynamic callback disabling is honoured only with LITELLM_LICENSE set")
    skipped: Final = _ok(plain, headers={"x-litellm-disable-callbacks": "s3_v2"})
    logged: Final = _ok(plain)
    _object(plain, logged)
    held(lambda: plain.s3.objects_for(skipped), lambda found: found == (), holding=3)


def test_two_identical_requests_write_two_distinct_objects(plain: Rig) -> None:
    """Two chats with the same text must produce two objects with different keys, one per response id, never one
    object overwritten by the other."""
    text: Final = marker()
    first: Final = _object(plain, _ok(plain, text))
    second: Final = _object(plain, _ok(plain, text))
    assert first.key != second.key


def test_key_update_writes_an_audit_object_under_audit_logs(plain: Rig) -> None:
    """With `audit_log_callbacks: [s3_v2]` and `store_audit_logs: true`, updating a key must write an object at
    `cblv2-logs/audit_logs/<date>/<HH-MM-SS>_<audit id>.json` whose body names the updated object
    (submit_audit, sink.rs:354-372). UNVERIFIED: audit events flush on `s3_batch_size` (default 512) or the
    periodic flush; the 1 s flush interval is assumed to cover them, and the body field that holds the key token
    hash is assumed to be `object_id`."""
    with plain.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=f"cblv2-audit-{uuid.uuid4().hex[:8]}")
        plain.proxy.post("/key/update", {"key": key, "metadata": {"cblv2": "audit"}})

        def audit_objects() -> tuple[S3Put, ...]:
            return tuple(put for put in plain.s3.recorded() if put.key.startswith(f"{S3_PATH}/audit_logs/"))

        found: Final = eventually(audit_objects, bool, seconds=60)
    assert re.fullmatch(rf"{S3_PATH}/audit_logs/{DATE}/\d{{2}}-\d{{2}}-\d{{2}}_[^/]+\.json", found[-1].key), found[
        -1
    ].key
    assert found[-1].headers["content-disposition"].startswith('inline; filename="audit-')


def test_team_prefix_given_as_the_string_false_still_prefixes_the_team_alias(shaped: Rig) -> None:
    """`s3_use_team_prefix: "false"` is read with Python `bool()`, so it is true, and the key starts with
    `cblv2-logs/<team alias>/<key alias>/<date>/` (settings.rs:342 and its test at :535). This pins owner decision
    D10 of the s3-log-v3 spec: a port that parses the string as false moves every team's objects."""
    with shaped.proxy.scenario() as scenario:
        team_alias: Final = f"cblv2-team-{uuid.uuid4().hex[:8]}"
        key_alias: Final = f"cblv2-key-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(key_alias=key_alias, team_id=scenario.team(team_alias=team_alias))
        response_id: Final = _ok(shaped, key=key)
        put: Final = _object(shaped, response_id)
    assert re.match(rf"{S3_PATH}/{re.escape(team_alias)}/{re.escape(key_alias)}/{DATE}/time-", put.key), put.key


def test_key_prefix_is_skipped_for_a_key_without_an_alias(shaped: Rig) -> None:
    """A key with no alias in no team must be written at `cblv2-logs/<date>/...`: a missing alias adds no empty
    folder and no literal "None" segment."""
    with shaped.proxy.scenario() as scenario:
        response_id: Final = _ok(shaped, key=scenario.key())
        put: Final = _object(shaped, response_id)
    assert re.match(rf"{S3_PATH}/{DATE}/time-", put.key), put.key


def test_kms_settings_add_both_server_side_encryption_headers(shaped: Rig) -> None:
    """With `s3_server_side_encryption: aws:kms` and `s3_sse_kms_key_id` set, the PUT must carry
    `x-amz-server-side-encryption: aws:kms` and `x-amz-server-side-encryption-aws-kms-key-id: <key id>`."""
    response_id: Final = _ok(shaped)
    headers: Final = _object(shaped, response_id).headers
    assert headers["x-amz-server-side-encryption"] == "aws:kms"
    assert headers["x-amz-server-side-encryption-aws-kms-key-id"] == KMS_KEY


def test_prompts_only_nulls_the_response_and_keeps_the_messages(shaped: Rig) -> None:
    """With `s3_log_prompts_only: true` the payload's `response` must be null while `messages` still holds the
    caller's text and the usage and cost fields are unchanged."""
    text: Final = marker()
    response_id: Final = _ok(shaped, text)
    payload: Final = _payload(_object(shaped, response_id), response_id)
    assert payload["response"] is None
    assert text in json.dumps(payload["messages"])
    assert payload["total_tokens"] == 15


def test_strip_base64_files_removes_inline_image_data(shaped: Rig) -> None:
    """With `s3_strip_base64_files: true`, a chat whose message carries an inline base64 PNG data URL must be
    written without the base64 data anywhere in the object body, while the text part survives. UNVERIFIED: whether
    the image part is dropped or replaced by a placeholder; only the absence of the data is asserted."""
    text: Final = marker()
    response: Final = shaped.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": CHAT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PIXEL}"}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    put: Final = _object(shaped, response.json()["id"])
    assert PIXEL not in put.body.decode()
    assert text in put.body.decode()


def test_batch_file_upload_writes_ndjson_lines_for_several_calls(batched: Rig) -> None:
    """With `s3_batch_file_upload: true`, calls flushed together are written as one object of Content-Type
    application/x-ndjson with one payload per line, under the same daily folder (sink.rs:385)."""
    ids: Final = [_ok(batched) for _ in range(3)]
    puts: Final = [_object(batched, response_id) for response_id in ids]
    for put in puts:
        assert put.headers["content-type"] == "application/x-ndjson", put.key
        assert re.match(rf"{S3_PATH}/{DATE}/", put.key), put.key
    assert sum(len(put.lines()) for put in {put.key: put for put in puts}.values()) >= 3


def test_global_message_redaction_applies_to_the_s3_body(batched: Rig) -> None:
    """With `turn_off_message_logging: true` the object body must not contain the caller's text or the provider
    reply: messages and response are replaced by the redaction placeholder, while total_tokens stays 15."""
    text: Final = marker()
    response: Final = chat(batched, text)
    assert response.status_code == 200
    response_id: Final = str(response.json()["id"])
    put: Final = _object(batched, response_id)
    assert text not in put.body.decode()
    assert "cblv2 scripted reply" not in put.body.decode()
    assert _payload(put, response_id)["total_tokens"] == 15


def test_a_logging_sink_outage_never_fails_the_request(batched: Rig) -> None:
    """While the S3 peer answers 503 to the next PUTs, a chat must still return 200 promptly, and its payload is
    delivered once the peer recovers: S3 logging is off the request path."""
    batched.s3.fail_next = 2
    response: Final = chat(batched, marker())
    assert response.status_code == 200
    _object(batched, str(response.json()["id"]), seconds=90)
