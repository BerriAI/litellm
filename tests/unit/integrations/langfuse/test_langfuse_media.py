"""Covers the media step of the langfuse callback: base64 payloads leave the trace as Langfuse media
references and reach Langfuse media storage through the REST API, as they did under the v2 SDK."""

import json
import logging
import threading
from base64 import b64encode
from collections.abc import Callable, Iterator, Sequence
from time import monotonic, sleep
from typing import Final

import httpx
import pytest
from langfuse.api import LangfuseAPI

from litellm.integrations.langfuse.langfuse_media import MediaField, MediaUpload, MediaUploader, extract_media

Transport = Callable[[httpx.Request], httpx.Response]

HELLO_BASE64 = b64encode(b"hello").decode()
HELLO_SHA256 = "LPJNul+wow4m6DsqxbninhsWHlwfp0JecwQzYpOLmCQ="
HELLO_PNG_REFERENCE = "@@@langfuseMedia:type=image/png|id=LPJNul-wow4m6Dsqxbninh|source=base64_data_uri@@@"
HELLO_PNG_URI = "data:image/png;base64," + HELLO_BASE64
DECODABLE_REFERENCE = "@@@langfuseMedia:type=image/png|id=LPJNul-wow4m6DsqxbninhA|source=base64_data_uri@@@"
TRACE_ID = "a" * 32
OBSERVATION_ID = "b" * 16
STORAGE_URL = "https://bucket.s3.amazonaws.com/media/hello?X-Amz-Signature=signed"


def test_data_uri_in_a_chat_message_becomes_the_reference_the_v2_sdk_wrote():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": HELLO_PNG_URI}},
            ],
        }
    ]

    replaced, found = extract_media(json.dumps(messages))

    assert json.loads(replaced)[0]["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": HELLO_PNG_REFERENCE}},
    ]
    assert [(media.content_type.value, media.content, media.sha256_hash) for media in found] == [
        ("image/png", b"hello", HELLO_SHA256)
    ]
    assert HELLO_BASE64 not in replaced


@pytest.mark.parametrize(
    "block",
    [
        {"type": "base64", "media_type": "image/png", "data": HELLO_BASE64},
        {"type": "media", "mime_type": "image/png", "data": HELLO_BASE64},
    ],
    ids=["anthropic", "vertex"],
)
def test_typed_provider_blocks_keep_their_shape_and_carry_the_reference(block):
    replaced, found = extract_media(json.dumps({"content": [{"type": "image", "source": block}]}))

    assert json.loads(replaced)["content"][0]["source"] == {**block, "data": HELLO_PNG_REFERENCE}
    assert [media.content for media in found] == [b"hello"]


def test_a_value_that_is_one_data_uri_is_replaced_whole():
    replaced, found = extract_media(HELLO_PNG_URI)

    assert replaced == HELLO_PNG_REFERENCE
    assert [(media.media_id, media.reference) for media in found] == [("LPJNul-wow4m6Dsqxbninh", HELLO_PNG_REFERENCE)]


def test_url_safe_base64_decodes_to_the_same_bytes_as_the_standard_alphabet():
    payload = bytes(range(248, 256)) * 3
    standard = b64encode(payload).decode()
    url_safe = standard.replace("+", "-").replace("/", "_")
    assert standard != url_safe

    (from_standard,) = extract_media("data:image/png;base64," + standard)[1]
    (from_url_safe,) = extract_media("data:image/png;base64," + url_safe)[1]

    assert from_url_safe == from_standard
    assert from_standard.content == payload


def test_every_payload_in_one_value_is_replaced_and_returned():
    other_uri = "data:image/jpeg;base64," + b64encode(b"other").decode()

    replaced, found = extract_media(json.dumps({"images": [HELLO_PNG_URI, other_uri, HELLO_PNG_URI]}))

    assert "base64," not in replaced
    assert [media.content for media in found] == [b"hello", b"other", b"hello"]


@pytest.mark.parametrize(
    "serialized",
    [
        json.dumps([{"role": "user", "content": "plain text"}]),
        json.dumps({"note": "the word base64 alone is not media"}),
        json.dumps({"url": "data:image/png,base64ab"}),
        json.dumps({"url": "data:application/x-not-stored;base64," + HELLO_BASE64}),
        json.dumps({"url": "data:image/png;base64,"}),
        json.dumps({"type": "base64", "media_type": "image/png", "data": DECODABLE_REFERENCE}),
        json.dumps({"type": "base64", "media_type": "image/png", "data": 7}),
        json.dumps({"type": ["base64"], "media_type": "image/png", "data": HELLO_BASE64}),
        '{"truncated": "data:image/png;base64,' + HELLO_BASE64,
        "base64 in a sentence that is not JSON",
    ],
    ids=[
        "text-only",
        "base64-word",
        "not-base64-uri",
        "unsupported-content-type",
        "empty-payload",
        "already-a-reference",
        "non-string-data",
        "non-string-type",
        "invalid-json",
        "plain-string",
    ],
)
def test_values_without_storable_media_are_returned_untouched(serialized):
    assert extract_media(serialized) == (serialized, ())


def test_payloads_nested_past_the_depth_the_sdk_walked_stay_inline():
    nested = HELLO_PNG_URI
    for _ in range(12):
        nested = [nested]
    serialized = json.dumps(nested)

    assert extract_media(serialized) == (serialized, ())


def test_nesting_too_deep_to_parse_is_returned_untouched():
    serialized = "[" * 100_000 + '"data:image/png;base64,' + HELLO_BASE64 + '"' + "]" * 100_000

    assert extract_media(serialized) == (serialized, ())


def test_a_data_uri_followed_by_prose_is_left_inline_with_its_text():
    serialized: Final = json.dumps({"content": HELLO_PNG_URI + " and this is what I see in it"})

    assert extract_media(serialized) == (serialized, ())


def _upload(field: MediaField = "input", observation_id: str | None = OBSERVATION_ID) -> MediaUpload:
    return MediaUpload(
        media=extract_media(HELLO_PNG_URI)[1][0], trace_id=TRACE_ID, observation_id=observation_id, field=field
    )


def _uploader(
    transport: Transport,
    *,
    delays: Sequence[float] = (),
    max_queued_bytes: int = 1_000,
    validate_upload_url: bool = False,
) -> MediaUploader:
    client: Final = httpx.Client(transport=httpx.MockTransport(transport))
    return MediaUploader(
        api=LangfuseAPI(base_url="https://langfuse.example", username="pk", password="sk", httpx_client=client),
        client=client,
        timeout=5.0,
        delays=delays,
        max_queued_bytes=max_queued_bytes,
        validate_upload_url=validate_upload_url,
    )


def _langfuse(
    requests: list[httpx.Request],
    *,
    upload_url: str | None = STORAGE_URL,
    storage: Sequence[httpx.Response] = (httpx.Response(200, text="stored"),),
) -> Transport:
    responses: Final = iter(storage)

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/api/public/media":
            return httpx.Response(201, json={"mediaId": "LPJNul-wow4m6Dsqxbninh", "uploadUrl": upload_url})
        if request.method == "PUT":
            return next(responses)
        return httpx.Response(200, json={})

    return transport


def test_new_media_is_registered_with_its_trace_put_to_storage_and_confirmed():
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests))

    assert uploader.submit((_upload(),)) is True

    assert uploader.drain(5_000) is True
    assert uploader.failed_uploads == 0
    register, put, confirm = requests
    assert (register.method, register.url.path) == ("POST", "/api/public/media")
    assert json.loads(register.content) == {
        "traceId": TRACE_ID,
        "observationId": OBSERVATION_ID,
        "contentType": "image/png",
        "contentLength": 5,
        "sha256Hash": HELLO_SHA256,
        "field": "input",
    }
    assert (put.method, str(put.url), put.content) == ("PUT", STORAGE_URL, b"hello")
    assert put.headers["content-type"] == "image/png"
    assert put.headers["x-amz-checksum-sha256"] == HELLO_SHA256
    assert put.headers["x-ms-blob-type"] == "BlockBlob"
    assert (confirm.method, confirm.url.path) == ("PATCH", "/api/public/media/LPJNul-wow4m6Dsqxbninh")
    confirmed = json.loads(confirm.content)
    assert (confirmed["uploadHttpStatus"], confirmed.get("uploadHttpError")) == (200, None)
    assert confirmed["uploadTimeMs"] >= 0 and confirmed["uploadedAt"]


def test_media_langfuse_already_holds_is_linked_to_the_trace_without_a_second_upload(caplog):
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests, upload_url=None))

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        uploader.submit((_upload(field="metadata"),))
        assert uploader.drain(5_000) is True

    assert caplog.text == ""
    assert [(request.method, request.url.path) for request in requests] == [("POST", "/api/public/media")]
    assert json.loads(requests[0].content)["field"] == "metadata"


def test_trace_level_media_is_registered_without_an_observation():
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests, upload_url=None))

    uploader.submit((_upload(observation_id=None),))

    assert uploader.drain(5_000) is True
    registered = json.loads(requests[0].content)
    assert registered["traceId"] == TRACE_ID
    assert registered.get("observationId") is None


def test_google_cloud_storage_upload_leaves_out_the_s3_and_azure_headers():
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests, upload_url="https://storage.googleapis.com/bucket/hello?sig=1"))

    uploader.submit((_upload(),))

    assert uploader.drain(5_000) is True
    put = requests[1]
    assert put.headers["content-type"] == "image/png"
    assert "x-amz-checksum-sha256" not in put.headers and "x-ms-blob-type" not in put.headers


def test_a_storage_outage_is_retried_and_the_final_status_is_what_langfuse_is_told():
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests, storage=(httpx.Response(503), httpx.Response(200))), delays=(0.0,))

    uploader.submit((_upload(),))

    assert uploader.drain(5_000) is True
    assert [request.method for request in requests] == ["POST", "PUT", "PUT", "PATCH"]
    assert json.loads(requests[-1].content)["uploadHttpStatus"] == 200
    assert uploader.failed_uploads == 0


def test_a_refused_upload_is_reported_to_langfuse_by_status_only_and_logged(caplog):
    """The storage URL comes from the Langfuse host, so what the storage answers is never sent back to that host."""
    requests: list[httpx.Request] = []
    refusal = httpx.Response(403, text="internal-service-body")
    uploader = _uploader(_langfuse(requests, storage=(refusal,)), delays=(0.0,))

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        uploader.submit((_upload(),))
        assert uploader.drain(5_000) is True

    assert [request.method for request in requests] == ["POST", "PUT", "PATCH"]
    confirmed = json.loads(requests[-1].content)
    assert (confirmed["uploadHttpStatus"], confirmed["uploadHttpError"]) == (403, "Forbidden")
    assert b"internal-service-body" not in requests[-1].content
    assert "the media storage refused it with HTTP 403" in caplog.text
    assert uploader.failed_uploads == 1


def test_the_storage_response_body_is_never_read():
    requests: list[httpx.Request] = []

    class _Unreadable(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            raise AssertionError("the storage response body was read")

    uploader = _uploader(_langfuse(requests, storage=(httpx.Response(200, stream=_Unreadable()),)))

    uploader.submit((_upload(),))

    assert uploader.drain(5_000) is True
    assert uploader.failed_uploads == 0
    assert json.loads(requests[-1].content)["uploadHttpStatus"] == 200


def test_a_storage_redirect_is_not_followed():
    requests: list[httpx.Request] = []
    redirect = httpx.Response(307, headers={"location": "http://169.254.169.254/latest/meta-data"})
    uploader = _uploader(_langfuse(requests, storage=(redirect,)))

    uploader.submit((_upload(),))

    assert uploader.drain(5_000) is True
    assert [(request.method, request.url.host) for request in requests] == [
        ("POST", "langfuse.example"),
        ("PUT", "bucket.s3.amazonaws.com"),
        ("PATCH", "langfuse.example"),
    ]
    assert json.loads(requests[-1].content)["uploadHttpStatus"] == 307
    assert uploader.failed_uploads == 1


@pytest.mark.parametrize(
    "upload_url", ["http://169.254.169.254/latest/meta-data", "http://127.0.0.1:8080/admin", "file:///etc/passwd"]
)
def test_a_host_from_request_credentials_cannot_send_the_upload_to_an_internal_address(upload_url: str, caplog):
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests, upload_url=upload_url), validate_upload_url=True)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        uploader.submit((_upload(),))
        assert uploader.drain(5_000) is True

    assert [(request.method, request.url.host) for request in requests] == [("POST", "langfuse.example")]
    assert "its upload URL was refused" in caplog.text
    assert uploader.failed_uploads == 1


def test_an_operator_configured_host_may_keep_its_storage_on_an_internal_address():
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests, upload_url="http://127.0.0.1:9000/bucket/hello"))

    uploader.submit((_upload(),))

    assert uploader.drain(5_000) is True
    assert [(request.method, request.url.host) for request in requests] == [
        ("POST", "langfuse.example"),
        ("PUT", "127.0.0.1"),
        ("PATCH", "langfuse.example"),
    ]
    assert uploader.failed_uploads == 0


def test_storage_that_is_never_reached_is_not_reported_as_uploaded(caplog):
    requests: list[httpx.Request] = []
    langfuse = _langfuse(requests)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            requests.append(request)
            raise httpx.ConnectError("storage unreachable")
        return langfuse(request)

    uploader = _uploader(transport, delays=(0.0,))

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        uploader.submit((_upload(),))
        assert uploader.drain(5_000) is True

    assert [request.method for request in requests] == ["POST", "PUT", "PUT"]
    assert "storage unreachable" in caplog.text
    assert uploader.failed_uploads == 1


def test_a_registration_that_does_not_reach_langfuse_is_sent_again_after_each_delay():
    requests: list[httpx.Request] = []
    langfuse = _langfuse(requests)
    outage = iter((True, True))

    def transport(request: httpx.Request) -> httpx.Response:
        if next(outage, False):
            requests.append(request)
            raise httpx.ConnectError("langfuse unreachable")
        return langfuse(request)

    uploader = _uploader(transport, delays=(0.0, 0.0))

    assert uploader.submit((_upload(),)) is True
    assert uploader.drain(5_000) is True

    assert [request.method for request in requests] == ["POST", "POST", "POST", "PUT", "PATCH"]
    assert uploader.failed_uploads == 0
    assert uploader.submit((_upload(),)) is True


def test_a_registration_that_never_reaches_langfuse_fails_that_upload_only(caplog):
    requests: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("langfuse unreachable")

    uploader = _uploader(transport, delays=(0.0,))

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        uploader.submit((_upload(),))
        assert uploader.drain(5_000) is True

    assert [request.method for request in requests] == ["POST", "POST"]
    assert uploader.failed_uploads == 1
    assert "langfuse unreachable" in caplog.text
    assert uploader.submit((_upload(),)) is True


def test_a_failed_registration_is_logged_without_its_headers_and_the_next_upload_still_runs(caplog):
    requests: list[httpx.Request] = []
    langfuse = _langfuse(requests)
    refusals = iter((True,))

    def transport(request: httpx.Request) -> httpx.Response:
        if next(refusals, False):
            return httpx.Response(400, json={"message": "bad"}, headers={"x-secret-header": "do-not-log"})
        return langfuse(request)

    uploader = _uploader(transport)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        uploader.submit((_upload(),))
        uploader.submit((_upload(),))
        assert uploader.drain(5_000) is True

    assert [request.method for request in requests] == ["POST", "PUT", "PATCH"]
    assert "HTTP 400 from the media API" in caplog.text
    assert "do-not-log" not in caplog.text
    assert uploader.failed_uploads == 1


def _blocked_uploader(release: threading.Event, started: threading.Event, **limits: int) -> MediaUploader:
    def transport(request: httpx.Request) -> httpx.Response:
        started.set()
        release.wait(10)
        return httpx.Response(201, json={"mediaId": "LPJNul-wow4m6Dsqxbninh", "uploadUrl": None})

    return _uploader(transport, **limits)


def test_drain_reports_uploads_still_running_at_the_deadline():
    release, started = threading.Event(), threading.Event()
    uploader = _blocked_uploader(release, started)
    uploader.submit((_upload(),))
    assert started.wait(5)

    assert uploader.drain(50) is False
    release.set()
    assert uploader.drain(5_000) is True


def test_a_group_that_does_not_fit_as_a_whole_queues_none_of_its_uploads():
    requests: list[httpx.Request] = []
    uploader = _uploader(_langfuse(requests, upload_url=None), max_queued_bytes=9)

    assert uploader.submit((_upload(), _upload())) is False

    assert uploader.drain(5_000) is True
    assert requests == []
    assert uploader.submit((_upload(),)) is True
    assert uploader.drain(5_000) is True
    assert len(requests) == 1


def test_media_that_does_not_fit_next_to_the_waiting_uploads_is_turned_away_until_they_finish():
    """Each waiting upload holds its decoded payload, so the bytes are what is capped."""
    release, started = threading.Event(), threading.Event()
    uploader = _blocked_uploader(release, started, max_queued_bytes=5)
    assert uploader.submit((_upload(),)) is True
    assert started.wait(5)

    assert uploader.submit((_upload(),)) is False

    release.set()
    assert uploader.drain(5_000) is True
    assert uploader.submit((_upload(),)) is True
    assert uploader.drain(5_000) is True


def test_shutdown_ends_the_upload_thread_after_the_queued_uploads():
    requests: list[httpx.Request] = []
    before = {thread.ident for thread in threading.enumerate()}
    uploader = _uploader(_langfuse(requests, upload_url=None))
    thread = next(thread for thread in threading.enumerate() if thread.ident not in before)
    assert (thread.name, thread.daemon) == ("langfuse-media-upload", True)

    uploader.submit((_upload(),))
    uploader.shutdown()

    deadline = monotonic() + 5
    while thread.is_alive() and monotonic() < deadline:
        sleep(0.01)
    assert not thread.is_alive()
    assert len(requests) == 1
