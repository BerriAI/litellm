"""`gemini/` Files API uploads and context caching (`cachedContents`) through the proxy.

Files. `POST /v1/files` with `model=<a gemini/ deployment>` uploads with that deployment's key in
Google's two step resumable protocol (`GoogleAIStudioFilesHandler`, `llm_http_handler.create_file`):
first `POST {api_base}/upload/v1beta/files` with `X-Goog-Upload-Protocol: resumable`,
`X-Goog-Upload-Command: start`, the byte count and content type in `X-Goog-Upload-Header-*` and a JSON
body `{"file": {"display_name": <filename>}}`; Google answers with an `X-Goog-Upload-URL` header, and
the bytes are then posted to that URL with `X-Goog-Upload-Command: upload, finalize`. The returned
OpenAI file object uses Google's file `uri` as its id, encoded with the model name so later reads find
the deployment again. Note the Files API path carries its own `v1beta`, unlike chat, so these tests
give the deployment a bare host `api_base`.

Context caching. When a chat request marks a long prefix with `cache_control`, the proxy looks the
prefix up in `{api_base}/cachedContents` by `displayName` (a hash of the cached messages), creates it
with `POST {api_base}/cachedContents` when missing, and sends only the rest of the conversation to
`generateContent` with `cachedContent: <name>` (`ContextCachingEndpoints.async_check_and_create_cache`).
A created cache with a future `expireTime` is remembered in process, so the next identical request
skips both list and create (`KnownCachedContents`). The listing and creation of a cache from a
`v1beta` base is already pinned by `test_gemini_context_caching_custom_api_base_wire.py`; this file pins
the rest.
"""

from __future__ import annotations

import base64
import json
from typing import Final

from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_gemini_support import (
    CACHE_NAME,
    FLASH,
    GEMINI_KEY,
    at,
    body_of,
    cached_contents_responder,
    cached_messages,
    gemini,
    gemini_echo,
    gemini_reply,
    google_error,
    headers_of,
    json_of,
    marker,
    path_of,
    reply_json,
    text_of,
    usage_metadata,
)
from pydantic import JsonValue

_FILE_URI: Final = "https://generativelanguage.googleapis.com/v1beta/files/dec1file"
_PAYLOAD: Final = b"%PDF-1.4 synthetic dec1 document"


def _files_responder(peer_url: list[str], state: str = "ACTIVE") -> object:
    """A peer that speaks the resumable upload and the file resource paths; `peer_url` is filled in after
    the peer starts so the upload URL can point back at it."""

    def respond(request: Request) -> Reply:
        path: Final = path_of(request)
        file_resource: Final[dict[str, JsonValue]] = {
            "name": "files/dec1file",
            "displayName": "dec1.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": str(len(_PAYLOAD)),
            "createTime": "2026-11-17T10:00:00.123456789Z",
            "updateTime": "2026-11-17T10:00:00.123456789Z",
            "uri": _FILE_URI,
            "state": state,
        }
        if path == "/upload/v1beta/files" and "upload_id" not in request.target:
            return Reply(
                body=b"{}",
                headers={
                    "X-Goog-Upload-URL": f"{peer_url[0]}/upload/v1beta/files?upload_id=dec1&upload_protocol=resumable"
                },
            )
        if path == "/upload/v1beta/files":
            return reply_json({"file": file_resource})
        if path == "/v1beta/files/dec1file" and request.method == "GET":
            return reply_json(file_resource)
        if path == "/v1beta/files/dec1file" and request.method == "DELETE":
            return reply_json({})
        return gemini_echo(request)

    return respond


def _upload(gateway: Gateway, model: str, filename: str = "dec1.pdf") -> object:
    return gateway.request_multipart(
        "/v1/files",
        {"purpose": "user_data", "model": model},
        {"file": (filename, _PAYLOAD, "application/pdf")},
    )


def _decoded_id(file_id: str) -> str:
    encoded: Final = file_id.removeprefix("file-")
    return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()


# ---------------------------------------------------------------- Files API


def test_an_upload_starts_a_resumable_session_with_the_deployment_key(gateway: Gateway) -> None:
    """The first upstream request is `POST /upload/v1beta/files` with `x-goog-api-key`, `X-Goog-Upload-Protocol:
    resumable`, `X-Goog-Upload-Command: start`, `X-Goog-Upload-Header-Content-Length` equal to the byte count,
    `X-Goog-Upload-Header-Content-Type: application/pdf`, and the JSON body `{"file": {"display_name": "dec1.pdf"}}`."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        response: Final = _upload(gateway, rig.model)
        assert response.status_code == 200, response.text  # pyright: ignore[reportAttributeAccessIssue]
        start: Final = rig.peer.to("/upload/v1beta/files")[0]
        headers: Final = headers_of(start)
        assert headers["x-goog-api-key"] == GEMINI_KEY
        assert headers["x-goog-upload-protocol"] == "resumable"
        assert headers["x-goog-upload-command"] == "start"
        assert headers["x-goog-upload-header-content-length"] == str(len(_PAYLOAD))
        assert headers["x-goog-upload-header-content-type"] == "application/pdf"
        assert body_of(start) == {"file": {"display_name": "dec1.pdf"}}


def test_the_bytes_go_to_the_returned_upload_url_with_finalize(gateway: Gateway) -> None:
    """The second request goes to the `X-Goog-Upload-URL` Google returned (query `upload_id=dec1` kept), with
    `X-Goog-Upload-Command: upload, finalize`, `X-Goog-Upload-Offset: 0` and exactly the uploaded bytes."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        assert _upload(gateway, rig.model).status_code == 200  # pyright: ignore[reportAttributeAccessIssue]
        uploads: Final = rig.peer.to("/upload/v1beta/files")
        assert len(uploads) == 2
        finalize: Final = uploads[1]
        assert "upload_id=dec1" in finalize.target
        assert headers_of(finalize)["x-goog-upload-command"] == "upload, finalize"
        assert headers_of(finalize)["x-goog-upload-offset"] == "0"
        assert finalize.body == _PAYLOAD


def test_the_finalize_request_carries_no_api_key(gateway: Gateway) -> None:
    """The upload URL is a pre-authorised session URL, and the proxy sends only the upload headers to it
    (`upload_request.headers`), so the finalize request has no `x-goog-api-key`. A rewrite that reuses the
    first request's headers would leak the key to whatever host the URL names."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        assert _upload(gateway, rig.model).status_code == 200  # pyright: ignore[reportAttributeAccessIssue]
        assert "x-goog-api-key" not in headers_of(rig.peer.to("/upload/v1beta/files")[1])


def test_the_file_object_names_google_s_uri_inside_a_model_encoded_id(gateway: Gateway) -> None:
    """The caller gets an OpenAI file object with `object: "file"`, `purpose: "user_data"`, `bytes` equal to
    Google's `sizeBytes`, `filename: "dec1.pdf"`, `status: "uploaded"`, and an `id` that starts with `file-`
    and decodes to `litellm:<google uri>;model,<model name>` (`encode_file_id_with_model`)."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        response: Final = _upload(gateway, rig.model)
        body: Final = json_of(response)  # pyright: ignore[reportArgumentType]
        assert (body["object"], body["purpose"], body["filename"], body["status"]) == (
            "file",
            "user_data",
            "dec1.pdf",
            "uploaded",
        )
        assert body["bytes"] == len(_PAYLOAD)
        file_id: Final = string_value(body["id"])
        assert file_id.startswith("file-")
        assert _decoded_id(file_id) == f"litellm:{_FILE_URI};model,{rig.model}"


def test_google_nanosecond_timestamps_become_unix_seconds(gateway: Gateway) -> None:
    """`createTime: "2026-11-17T10:00:00.123456789Z"` (nine fractional digits, which Python's parser
    rejects) becomes `created_at: 1794909600` (`_gemini_timestamp` trims to microseconds and truncates)."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        body: Final = json_of(_upload(gateway, rig.model))  # pyright: ignore[reportArgumentType]
        assert body["created_at"] == 1794909600


def test_retrieving_the_file_reads_the_v1beta_resource_with_the_key(gateway: Gateway) -> None:
    """`GET /v1/files/<encoded id>` decodes the model, uses its key and reads `GET /v1beta/files/dec1file`
    (`gemini_file_name` turns the full URI into `files/dec1file`); an `ACTIVE` Google state is reported as
    `status: "processed"`. UNVERIFIED: that retrieve on a model-encoded id routes to the deployment's
    `api_base` rather than `GEMINI_API_BASE`."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        file_id: Final = string_value(json_of(_upload(gateway, rig.model))["id"])  # pyright: ignore[reportArgumentType]
        response: Final = gateway.request("GET", f"/v1/files/{file_id}")
        assert response.status_code == 200, response.text
        read: Final = rig.peer.to("/v1beta/files/dec1file", method="GET")
        assert len(read) == 1
        assert headers_of(read[0])["x-goog-api-key"] == GEMINI_KEY
        assert json_of(response)["status"] == "processed"


def test_a_failed_google_file_is_reported_as_error(gateway: Gateway) -> None:
    """A file resource whose `state` is `FAILED` is reported as `status: "error"` on retrieve, and
    `PROCESSING` as `uploaded` (`transform_retrieve_file_response`). UNVERIFIED routing as above."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box, state="FAILED"), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        file_id: Final = string_value(json_of(_upload(gateway, rig.model))["id"])  # pyright: ignore[reportArgumentType]
        response: Final = gateway.request("GET", f"/v1/files/{file_id}")
        assert json_of(response)["status"] == "error"


def test_deleting_the_file_sends_delete_to_the_v1beta_resource(gateway: Gateway) -> None:
    """`DELETE /v1/files/<encoded id>` becomes `DELETE /v1beta/files/dec1file` with the key header.
    UNVERIFIED routing as above."""
    url_box: Final[list[str]] = []
    with gemini(gateway, _files_responder(url_box), base_suffix="") as rig:  # pyright: ignore[reportArgumentType]
        url_box.append(rig.peer.url)
        file_id: Final = string_value(json_of(_upload(gateway, rig.model))["id"])  # pyright: ignore[reportArgumentType]
        response: Final = gateway.request("DELETE", f"/v1/files/{file_id}")
        assert response.status_code == 200, response.text
        deleted: Final = rig.peer.to("/v1beta/files/dec1file", method="DELETE")
        assert len(deleted) == 1
        assert headers_of(deleted[0])["x-goog-api-key"] == GEMINI_KEY


def test_an_upload_refused_by_google_keeps_the_status(gateway: Gateway) -> None:
    """When the start request is answered 403 `PERMISSION_DENIED`, the caller gets an error status (no
    file object) and no bytes are uploaded. UNVERIFIED: whether the status is 403 or remapped to 500 by
    the files error handler."""
    with gemini(gateway, lambda _r: google_error(403, "PERMISSION_DENIED", "upload forbidden"), base_suffix="") as rig:
        response: Final = _upload(gateway, rig.model)
        assert response.status_code >= 400  # pyright: ignore[reportAttributeAccessIssue]
        assert len(rig.peer.to("/upload/v1beta/files")) == 1


def test_an_uploaded_file_uri_is_sent_as_file_data_in_chat(gateway: Gateway) -> None:
    """Chat on the same deployment with a `file` block whose `file_id` is the Google URI and `format:
    application/pdf` sends `{"file_data": {"mime_type": "application/pdf", "file_uri": <uri>}}` and does not
    try to fetch the URI."""
    with gemini(gateway, gemini_echo) as rig:
        messages: Final[list[JsonValue]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"summarise {marker()}"},
                    {"type": "file", "file": {"file_id": _FILE_URI, "format": "application/pdf"}},
                ],
            }
        ]
        assert rig.chat(messages=messages).status_code == 200
        parts: Final = list_value(object_value(list_value(rig.peer.one_body()["contents"])[0])["parts"])
        assert {"file_data": {"mime_type": "application/pdf", "file_uri": _FILE_URI}} in parts
        assert rig.peer.to("/v1beta/files/dec1file", method="GET") == ()


# ---------------------------------------------------------------- context caching


def test_the_cache_create_body_names_the_model_and_the_hashed_display_name(gateway: Gateway) -> None:
    """`POST /v1beta/cachedContents` carries `model: "models/gemini-2.5-flash"` (no project path, unlike
    Vertex), a non empty `displayName` (the cache key hash), and `contents` holding only the cached user
    block; the uncached question is not in the cache."""
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=cached_messages()).status_code == 200
        created: Final = body_of(rig.peer.to("/v1beta/cachedContents")[0])
        assert created["model"] == f"models/{FLASH}"
        assert string_value(created["displayName"])
        contents: Final = list_value(created["contents"])
        assert len(contents) == 1
        assert "How many clauses" not in text_of(contents[0])


def test_the_generate_call_sends_only_the_uncached_turn_with_cached_content(gateway: Gateway) -> None:
    """After the cache is created, `generateContent` carries `cachedContent: "cachedContents/dec1-cache"`
    and `contents` with only "How many clauses are there?"; the long prefix is not sent twice."""
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=cached_messages()).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["cachedContent"] == CACHE_NAME
        contents: Final = list_value(body["contents"])
        assert [text_of(item) for item in contents] == ["How many clauses are there?"]


def test_cache_control_ttl_is_sent_in_seconds(gateway: Gateway) -> None:
    """`cache_control: {"type": "ephemeral", "ttl": "5m"}` is sent on the create body as `ttl: "300s"`
    (`_normalize_ttl_to_seconds`); Google only takes seconds."""
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=cached_messages("5m")).status_code == 200
        assert body_of(rig.peer.to("/v1beta/cachedContents")[0])["ttl"] == "300s"


def test_an_invalid_ttl_is_left_out_of_the_create_body(gateway: Gateway) -> None:
    """A ttl that does not match `<number><s|m|h>` (here `"forever"`) is dropped, so the create body has no
    `ttl` and Google's one hour default applies, instead of a 400 from Google."""
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=cached_messages("forever")).status_code == 200
        assert "ttl" not in body_of(rig.peer.to("/v1beta/cachedContents")[0])


def test_system_messages_move_into_the_cache(gateway: Gateway) -> None:
    """Once any message is cached, every system message goes into the cache as `system_instruction` and the
    generateContent body has no `system_instruction`, because Google rejects one next to `cachedContent`
    (`separate_cached_messages`)."""
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=cached_messages(system="Answer briefly.")).status_code == 200
        created: Final = body_of(rig.peer.to("/v1beta/cachedContents")[0])
        assert "Answer briefly." in json.dumps(created["system_instruction"])
        assert "system_instruction" not in rig.peer.one_body()


def test_tools_move_into_the_cache_and_out_of_the_generate_call(gateway: Gateway) -> None:
    """With a cached prefix and a `get_weather` tool, the tool is put on the cache create body under
    `tools` and the generateContent body has neither `tools` nor `toolConfig`."""
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
    }
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=cached_messages(), tools=[tool]).status_code == 200
        assert "get_weather" in json.dumps(body_of(rig.peer.to("/v1beta/cachedContents")[0])["tools"])
        body: Final = rig.peer.one_body()
        assert "tools" not in body and "toolConfig" not in body


def test_a_listed_cache_with_the_same_display_name_is_reused_without_create(gateway: Gateway) -> None:
    """When the list already has an item whose `displayName` equals the request's cache key, the proxy uses
    its `name` and never POSTs `cachedContents`. The key is learnt from a first request whose create body
    the peer records, then a second proxy-wide fresh scenario lists it back."""
    with gemini(gateway, cached_contents_responder(created={"name": CACHE_NAME})) as first:
        messages: Final = cached_messages()
        assert first.chat(messages=messages).status_code == 200
        display_name: Final = body_of(first.peer.to("/v1beta/cachedContents")[0])["displayName"]
    listed: Final = [{"name": "cachedContents/listed", "displayName": display_name}]
    with gemini(gateway, cached_contents_responder(listed=listed)) as rig:
        assert rig.chat(messages=messages).status_code == 200
        assert rig.peer.to("/v1beta/cachedContents", method="POST") == ()
        assert rig.peer.one_body()["cachedContent"] == "cachedContents/listed"


def test_a_cache_with_a_future_expiry_is_reused_without_list_or_create(gateway: Gateway) -> None:
    """A created cache with `expireTime` in 2999 is remembered, so a second identical request on the same
    deployment sends only `generateContent` with the same `cachedContent`: no second `GET` or `POST` to
    `cachedContents`. UNVERIFIED: holds per proxy worker; with several workers the second request may land
    on a worker that lists again (still no second create)."""
    with gemini(gateway, cached_contents_responder()) as rig:
        messages: Final = cached_messages()
        assert rig.chat(messages=messages).status_code == 200
        assert rig.chat(messages=messages, temperature=0.1).status_code == 200
        assert len(rig.peer.to("/v1beta/cachedContents", method="POST")) == 1
        assert all(body_of(call)["cachedContent"] == CACHE_NAME for call in rig.peer.generate_calls())


def test_a_cache_without_expire_time_is_listed_again_next_time(gateway: Gateway) -> None:
    """A create answer with no `expireTime` is not remembered (`KnownCachedContents.remember` needs one),
    so the next identical request lists `cachedContents` again."""
    with gemini(gateway, cached_contents_responder(created={"name": CACHE_NAME})) as rig:
        messages: Final = cached_messages()
        assert rig.chat(messages=messages).status_code == 200
        assert rig.chat(messages=messages, temperature=0.1).status_code == 200
        assert len(rig.peer.to("/v1beta/cachedContents", method="GET")) == 2


def test_a_cached_content_from_the_caller_skips_list_and_create(gateway: Gateway) -> None:
    """A caller passing `cached_content: "cachedContents/mine"` gets that name in the generateContent body
    and the proxy neither lists nor creates caches, even with `cache_control` on a message."""
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=cached_messages(), cached_content="cachedContents/mine").status_code == 200
        assert rig.peer.one_body()["cachedContent"] == "cachedContents/mine"
        assert [r for r in rig.peer.requests() if "cachedContents" in path_of(r)] == []


def test_a_cached_contents_404_skips_caching_and_sends_the_whole_prompt(gateway: Gateway) -> None:
    """A gateway that answers 404 on `cachedContents` turns caching off for the request: the call still
    succeeds and generateContent carries the full conversation and no `cachedContent`
    (`_CachedContentsNotServedError`)."""
    with gemini(gateway, cached_contents_responder(list_status=404)) as rig:
        response: Final = rig.chat(messages=cached_messages())
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert "cachedContent" not in body
        assert len(list_value(body["contents"])) >= 1
        assert "Contract clause 0" in json.dumps(body["contents"])


def test_a_short_prefix_is_never_cached(gateway: Gateway) -> None:
    """A `cache_control` block of a few words is below the 1024 token minimum, so no `cachedContents`
    request is made and the whole prompt goes to generateContent."""
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"short {marker()}", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=messages).status_code == 200
        assert [r for r in rig.peer.requests() if "cachedContents" in path_of(r)] == []
        assert "cachedContent" not in rig.peer.one_body()


def test_a_cached_block_ending_on_an_assistant_turn_is_not_cached(gateway: Gateway) -> None:
    """A cached block whose last message is an assistant turn would be refused by Google with "Requests
    ending with a model turn are not supported", so the proxy skips caching for it
    (`cached_messages_end_on_supported_turn`)."""
    control: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": [{"type": "text", "text": f"{marker()} context", "cache_control": control}]},
        {"role": "assistant", "content": [{"type": "text", "text": "x " * 3000, "cache_control": control}]},
        {"role": "user", "content": "go on"},
    ]
    with gemini(gateway, cached_contents_responder()) as rig:
        assert rig.chat(messages=messages).status_code == 200
        assert rig.peer.to("/v1beta/cachedContents", method="POST") == ()


def test_a_rejected_cache_is_recreated_and_the_call_retried_once(gateway: Gateway) -> None:
    """When generateContent answers 404 "CachedContent not found (or permission denied)" for the cache it was
    given, the proxy forgets that cache, creates a fresh one and sends generateContent again; the caller
    gets 200 (`post_with_fresh_cache_async`)."""
    generated: Final[list[int]] = []

    def generate(request: Request) -> Reply:
        generated.append(1)
        if len(generated) == 1:
            return google_error(404, "NOT_FOUND", "CachedContent not found (or permission denied)")
        return gemini_reply(usage=usage_metadata(5129, 2, cachedContentTokenCount=5120))

    with gemini(gateway, cached_contents_responder(generate=generate)) as rig:
        response: Final = rig.chat(messages=cached_messages())
        assert response.status_code == 200, response.text
        assert len(rig.peer.generate_calls()) == 2
        assert len(rig.peer.to("/v1beta/cachedContents", method="POST")) == 2


def test_cached_tokens_are_reported_on_the_chat_usage(gateway: Gateway) -> None:
    """`cachedContentTokenCount: 5120` inside `promptTokenCount: 5129` is reported as `prompt_tokens: 5129`
    and `prompt_tokens_details.cached_tokens: 5120`; the cached part is still counted in the prompt."""
    with gemini(gateway, cached_contents_responder()) as rig:
        body: Final = json_of(rig.chat(messages=cached_messages()))
        assert at(body, "usage", "prompt_tokens") == 5129
        assert at(body, "usage", "prompt_tokens_details", "cached_tokens") == 5120


def test_cache_creation_tokens_are_added_to_the_prompt_of_the_creating_call(gateway: Gateway) -> None:
    """When the create answer reports `usageMetadata.totalTokenCount: 5000`, the request that created the
    cache reports `cache_creation_input_tokens: 5000` and `prompt_tokens = 9 + 5000` for a generate answer
    of `promptTokenCount: 9` (`_record_cache_creation_tokens`, `_calculate_usage(cache_creation_tokens)`)."""
    created: Final[dict[str, JsonValue]] = {
        "name": CACHE_NAME,
        "expireTime": "2999-01-01T00:00:00Z",
        "usageMetadata": {"totalTokenCount": 5000},
    }
    responder: Final = cached_contents_responder(
        created=created, generate=lambda _r: gemini_reply(usage=usage_metadata(9, 2))
    )
    with gemini(gateway, responder) as rig:
        body: Final = json_of(rig.chat(messages=cached_messages()))
        assert at(body, "usage", "cache_creation_input_tokens") == 5000
        assert at(body, "usage", "prompt_tokens") == 5009
