"""Operator-defined pass-through endpoints: what the proxy sends to the target and what it hands back.

A pass-through endpoint maps a proxy path to a target URL. The proxy is meant to be a transparent relay for it:
the method, JSON body, query string, multipart parts and binary bytes go to the target as the caller sent them,
the target's status, headers and body come back as the target sent them, and the only things the proxy adds or
removes are credentials and hop-by-hop framing. Every test here registers an endpoint through
`POST /config/pass_through_endpoint` on the shared proxy, points it at a scripted vendor, and compares the request
the vendor received and the response the caller received against literals.

A rewrite of the HTTP front door or of the relay can silently change any of these: re-serialising a body,
dropping a query parameter, buffering a stream, forwarding the caller's virtual key, or swallowing an upstream
error status. The options that only a config file can set (`forward_headers`, `merge_query_params`,
`os.environ/` headers) live in `test_dec1_native_custom_passthrough_config.py`.
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_native_support import (
    JSON_LIST,
    always,
    assert_key_absent,
    first_sse_frame,
    granted_key,
    json_body,
    json_reply,
    multipart_parts,
    only_request,
    path_of,
    query_of,
    register_pass_through,
)
from pydantic import JsonValue

_BODY: Final[dict[str, JsonValue]] = {"ticket_id": "T-1", "nested": {"values": [1, 2.5, None, True]}, "text": "héllo"}
_REPLY: Final[dict[str, JsonValue]] = {"summary": "scripted", "items": [1, 2, 3]}
_SSE_FRAMES: Final = (
    b'event: delta\ndata: {"text":"first"}\n\n',
    b": vendor comment\n\n",
    b'id: 7\ndata: {"text":"second"}\n\n',
    b"data: [DONE]\n\n",
)
_ALL_BYTES: Final = bytes(range(256)) * 8


def test_post_body_reaches_the_exact_target_url_with_the_same_json(gateway: Gateway) -> None:
    """An endpoint targets `<vendor>/v1/summarize` and a granted key POSTs a nested JSON body to its proxy path.

    The vendor must receive exactly one `POST /v1/summarize` whose JSON equals the caller's body key for key,
    including the null, the float and the non-ASCII string, and the caller must get the vendor's JSON back with 200.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/summarize")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert response.json() == _REPLY, response.text
        upstream: Final = only_request(wire)
        assert (upstream.method, upstream.target) == ("POST", "/v1/summarize"), upstream
        assert json_body(upstream) == _BODY, upstream.body


def test_the_callers_virtual_key_never_reaches_the_target(gateway: Gateway) -> None:
    """A granted key calls an endpoint that has no configured headers.

    The vendor request must not contain the virtual key in any header, in the URL or in the body, and must not
    carry an `authorization` header at all, because header forwarding is off unless the operator turns it on.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/summarize")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert_key_absent(upstream, key)
        assert "authorization" not in upstream.headers, dict(upstream.headers)


def test_configured_headers_are_sent_and_ordinary_client_headers_are_not(gateway: Gateway) -> None:
    """The endpoint is created with `headers: {x-vendor-key: ..., x-vendor-org: ...}`.

    The caller adds its own `x-client-secret` and `cookie` headers. The vendor must see both configured headers
    with their literal values and must see neither of the caller's headers.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(
            gateway,
            scenario,
            f"{wire.url}/v1/summarize",
            headers={"x-vendor-key": "synthetic-vendor-key", "x-vendor-org": "org-7"},
        )
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request(
            "POST", path, _BODY, key=key, headers={"x-client-secret": "client-only", "cookie": "session=abc"}
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert upstream.headers["x-vendor-key"] == "synthetic-vendor-key", dict(upstream.headers)
        assert upstream.headers["x-vendor-org"] == "org-7", dict(upstream.headers)
        assert "x-client-secret" not in upstream.headers, dict(upstream.headers)
        assert "cookie" not in upstream.headers, dict(upstream.headers)


def test_an_x_pass_header_is_forwarded_with_the_prefix_removed(gateway: Gateway) -> None:
    """The caller sends `x-pass-anthropic-beta: tools-2099` to an endpoint with header forwarding off.

    The vendor must receive `anthropic-beta: tools-2099`. The `x-pass-` prefix is the caller's only way to hand a
    header to the target on a default endpoint, so the stripped name and the unchanged value both matter.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/summarize")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key, headers={"x-pass-anthropic-beta": "tools-2099"})
        assert response.status_code == 200, response.text
        assert only_request(wire).headers["anthropic-beta"] == "tools-2099"


@pytest.mark.parametrize(
    "protected",
    ["authorization", "api-key", "x-api-key", "x-goog-api-key", "ocp-apim-subscription-key", "x-amz-security-token"],
)
def test_an_x_pass_header_cannot_set_a_credential_header(gateway: Gateway, protected: str) -> None:
    """The endpoint is configured with a vendor credential in the protected header and the caller sends
    `x-pass-<that header>: attacker-value`.

    The vendor must receive the configured credential, never `attacker-value`. The `x-pass-` escape hatch must
    not let a caller replace the operator's credential or inject AWS signing headers.
    """
    configured: Final = {} if protected.startswith("x-amz-") else {protected: "operator-credential"}
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/summarize", headers=configured)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request(
            "POST", path, _BODY, key=key, headers={f"x-pass-{protected}": "attacker-value"}
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert "attacker-value" not in upstream.headers.values(), dict(upstream.headers)
        assert upstream.headers.get(protected) == configured.get(protected), dict(upstream.headers)


def test_the_callers_query_string_is_forwarded_to_the_target(gateway: Gateway) -> None:
    """A granted key calls the endpoint with `?page=2&filter=open&empty=`.

    The vendor must receive the same three query parameters with the same values on the target path.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/tickets")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("GET", path, key=key, params={"page": "2", "filter": "open", "empty": ""})
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert path_of(upstream) == "/v1/tickets", upstream.target
        assert query_of(upstream) == {"page": "2", "filter": "open", "empty": ""}, upstream.target


def test_a_query_string_on_the_target_url_is_kept_when_the_caller_sends_none(gateway: Gateway) -> None:
    """The endpoint's target is `<vendor>/v1/tickets?api-version=2024-10-21` and the caller sends no query.

    The vendor must receive `api-version=2024-10-21`: a fixed query on the configured target is part of the target.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/tickets?api-version=2024-10-21")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert query_of(only_request(wire)) == {"api-version": "2024-10-21"}


def test_default_query_params_are_sent_when_the_caller_sends_none(gateway: Gateway) -> None:
    """The endpoint has `default_query_params: {api-version: 2024-01, region: eu}` and the caller sends no query.

    The vendor must receive exactly those two parameters.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(
            gateway,
            scenario,
            f"{wire.url}/v1/tickets",
            default_query_params={"api-version": "2024-01", "region": "eu"},
        )
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert query_of(only_request(wire)) == {"api-version": "2024-01", "region": "eu"}


def test_query_precedence_is_caller_over_target_over_defaults(gateway: Gateway) -> None:
    """Defaults are `{a: default, b: default, c: default}`, the target URL carries `?b=target&c=target`, and the
    caller sends `?c=caller&d=caller`.

    The vendor must receive `a=default, b=target, c=caller, d=caller`: defaults are the weakest source, the
    target's own query overrides them, and the caller overrides both.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(
            gateway,
            scenario,
            f"{wire.url}/v1/tickets?b=target&c=target",
            default_query_params={"a": "default", "b": "default", "c": "default"},
        )
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key, params={"c": "caller", "d": "caller"})
        assert response.status_code == 200, response.text
        assert query_of(only_request(wire)) == {"a": "default", "b": "target", "c": "caller", "d": "caller"}


def test_include_subpath_appends_the_rest_of_the_path_to_the_target(gateway: Gateway) -> None:
    """The endpoint has `include_subpath: true` and targets `<vendor>/api`.

    A call to `<path>/v1/text-to-image/base/2.3` must reach the vendor as `POST /api/v1/text-to-image/base/2.3`,
    and a call to the bare path must still reach `POST /api`.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/api", include_subpath=True)
        key: Final = granted_key(scenario, path)
        deep: Final = gateway.request("POST", f"{path}/v1/text-to-image/base/2.3", _BODY, key=key)
        assert deep.status_code == 200, deep.text
        assert only_request(wire).target == "/api/v1/text-to-image/base/2.3"
        bare: Final = gateway.request("POST", path, _BODY, key=key)
        assert bare.status_code == 200, bare.text
        assert only_request(wire).target == "/api"


def test_include_subpath_keeps_a_trailing_slash(gateway: Gateway) -> None:
    """With `include_subpath: true`, a call to `<path>/v1/items/` (trailing slash) is made.

    The vendor must receive `/api/v1/items/` with the slash intact, because some vendors treat the two spellings
    as different resources.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/api", include_subpath=True)
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", f"{path}/v1/items/", _BODY, key=key)
        assert response.status_code == 200, response.text
        assert only_request(wire).target == "/api/v1/items/"


def test_without_include_subpath_a_deeper_path_is_not_served(gateway: Gateway) -> None:
    """The endpoint is created without `include_subpath` and the caller requests `<path>/extra`.

    The proxy must answer 404 or 405 and the vendor must receive nothing: an exact-path endpoint does not
    silently cover everything beneath it.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/api")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", f"{path}/extra", _BODY, key=key)
        assert response.status_code in (404, 405), response.text
        assert wire.drain() == ()


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_put_patch_and_delete_are_forwarded_with_their_method_and_body(gateway: Gateway, method: str) -> None:
    """An endpoint with no `methods` restriction is called with PUT, PATCH and DELETE carrying a JSON body.

    The vendor must receive the same method on the target path with the same JSON body.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/tickets/7")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request(method, path, _BODY, key=key)
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert (upstream.method, upstream.target) == (method, "/v1/tickets/7"), upstream
        assert json_body(upstream) == _BODY, upstream.body


def test_a_get_is_forwarded_without_a_request_body(gateway: Gateway) -> None:
    """A granted key sends a GET to the endpoint.

    The vendor must receive `GET` on the target path with an empty body, not a JSON `null` or `{}` body that a
    strict vendor would reject.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/tickets")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("GET", path, key=key)
        assert response.status_code == 200, response.text
        assert response.json() == _REPLY, response.text
        upstream: Final = only_request(wire)
        assert (upstream.method, upstream.body) == ("GET", b""), upstream


def test_methods_option_limits_the_endpoint_to_the_listed_methods(gateway: Gateway) -> None:
    """The endpoint is created with `methods: ["POST"]`.

    A POST must reach the vendor and return 200. A GET to the same path must be refused with 405 and must not
    reach the vendor.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/tickets", methods=["POST"])
        key: Final = granted_key(scenario, path)
        allowed: Final = gateway.request("POST", path, _BODY, key=key)
        assert allowed.status_code == 200, allowed.text
        assert only_request(wire).method == "POST"
        refused: Final = gateway.request("GET", path, key=key)
        assert refused.status_code == 405, refused.text
        assert wire.drain() == ()


def test_one_path_can_send_get_and_post_to_different_targets(gateway: Gateway) -> None:
    """Two endpoints share one proxy path: `methods: ["GET"]` targets `/read` and `methods: ["POST"]` targets
    `/write` on the same vendor.

    A GET must reach `/read` and a POST must reach `/write`. The method is part of the route's identity, so a
    registry keyed on path alone would send both to one target.
    """
    shared_path: Final = f"/dec1-native-split-{uuid.uuid4().hex}"
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        register_pass_through(gateway, scenario, f"{wire.url}/read", path=shared_path, methods=["GET"])
        register_pass_through(gateway, scenario, f"{wire.url}/write", path=shared_path, methods=["POST"])
        key: Final = granted_key(scenario, shared_path)
        read: Final = gateway.request("GET", shared_path, key=key)
        assert read.status_code == 200, read.text
        assert [(r.method, r.target) for r in wire.drain()] == [("GET", "/read")]
        write: Final = gateway.request("POST", shared_path, _BODY, key=key)
        assert write.status_code == 200, write.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/write")]


@pytest.mark.parametrize("status", [201, 202, 204])
def test_a_non_200_success_status_from_the_target_is_returned_unchanged(gateway: Gateway, status: int) -> None:
    """The vendor answers 201, 202 or 204.

    The caller must receive the same status code, and for 201 and 202 the same JSON body. A relay that
    normalises every success to 200 breaks clients that branch on Created and Accepted.
    """
    body: Final = b"" if status == 204 else json.dumps(_REPLY).encode()
    with wire_server(always(Reply(status=status, body=body))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == status, response.text
        assert response.content == body, response.content


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 429, 500, 502, 503])
def test_an_upstream_json_error_is_returned_with_its_status_and_body(gateway: Gateway, status: int) -> None:
    """The vendor answers an error status with its own JSON error body.

    The caller must receive the same status and a body that parses to exactly the vendor's JSON. The proxy must
    not wrap the body in its own error envelope, turn a 4xx into a 500, or retry the call.
    """
    error: Final[dict[str, JsonValue]] = {"error": {"code": f"vendor_{status}", "message": "scripted failure"}}
    with wire_server(always(json_reply(error, status=status))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == status, response.text
        assert response.json() == error, response.text
        assert len(wire.drain()) == 1


def test_an_upstream_plain_text_error_is_returned_byte_for_byte(gateway: Gateway) -> None:
    """The vendor answers 503 with a `text/plain` body that is not JSON.

    The caller must receive 503, the same bytes and a `text/plain` content type.
    """
    reply: Final = Reply(status=503, body=b"upstream is draining\n", content_type="text/plain")
    with wire_server(always(reply)) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 503, response.text
        assert response.content == b"upstream is draining\n", response.content
        assert response.headers["content-type"].startswith("text/plain"), response.headers


def test_upstream_response_headers_are_relayed_to_the_caller(gateway: Gateway) -> None:
    """The vendor answers with `x-vendor-request-id`, `retry-after` and `x-ratelimit-remaining` headers.

    The caller must receive all three with the vendor's values. Clients use these for support tickets and backoff.
    """
    headers: Final = {"x-vendor-request-id": "req-123", "retry-after": "7", "x-ratelimit-remaining": "41"}
    with wire_server(always(json_reply(_REPLY, headers=headers))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert {name: response.headers[name] for name in headers} == headers, dict(response.headers)


def test_the_upstream_server_header_is_not_duplicated(gateway: Gateway) -> None:
    """The vendor answers with `server: vendor-edge`.

    The caller's response must carry at most one `server` header and it must not be `vendor-edge`. Strict HTTP
    clients reject a response with two `Server` headers, which is what happens when the vendor's is relayed
    next to the proxy's own.
    """
    with (
        wire_server(always(json_reply(_REPLY, headers={"server": "vendor-edge"}))) as wire,
        gateway.scenario() as scenario,
    ):
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 200, response.text
        assert response.headers.get_list("server") != ["vendor-edge"], response.headers
        assert len(response.headers.get_list("server")) <= 1, response.headers


def test_the_response_carries_a_litellm_call_id_header(gateway: Gateway) -> None:
    """Two calls are made to the same endpoint.

    Each response must carry an `x-litellm-call-id` header holding a UUID, and the two values must differ. The
    header is how an operator ties a caller's complaint to a log line and a spend row.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        first: Final = gateway.request("POST", path, _BODY, key=key)
        second: Final = gateway.request("POST", path, _BODY, key=key)
        assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
        identities: Final = (first.headers["x-litellm-call-id"], second.headers["x-litellm-call-id"])
        assert [str(uuid.UUID(identity)) for identity in identities] == list(identities), identities
        assert identities[0] != identities[1], identities
        assert len(wire.drain()) == 2


def test_an_upstream_error_response_still_carries_the_litellm_call_id(gateway: Gateway) -> None:
    """The vendor answers 500.

    The relayed 500 must still carry `x-litellm-call-id`, so a failed call can be traced as easily as a good one.
    """
    with (
        wire_server(always(json_reply({"error": "boom"}, status=500))) as wire,
        gateway.scenario() as scenario,
    ):
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code == 500, response.text
        assert str(uuid.UUID(response.headers["x-litellm-call-id"])) == response.headers["x-litellm-call-id"]
        assert len(wire.drain()) == 1


def test_a_binary_response_is_returned_byte_for_byte(gateway: Gateway) -> None:
    """The vendor answers with 2048 bytes of `application/octet-stream` covering every byte value.

    The caller must receive the identical bytes and the same content type. A relay that decodes the body as
    text or JSON corrupts audio, images and archive downloads.
    """
    reply: Final = Reply(body=_ALL_BYTES, content_type="application/octet-stream")
    with wire_server(always(reply)) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/files/f1/content")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("GET", path, key=key)
        assert response.status_code == 200, response.text
        assert response.content == _ALL_BYTES
        assert response.headers["content-type"] == "application/octet-stream", response.headers
        assert len(wire.drain()) == 1


def test_a_jsonl_response_is_returned_without_being_reparsed(gateway: Gateway) -> None:
    """The vendor answers `application/jsonl` with two lines whose key order and spacing are unusual.

    The caller must receive the exact bytes. Batch result files are line-oriented, so re-serialising them as a
    single JSON document or reordering keys would break downstream parsers.
    """
    lines: Final = b'{"b":1,  "a":2}\n{"custom_id":"x","response":{"status_code":200}}\n'
    with (
        wire_server(always(Reply(body=lines, content_type="application/jsonl"))) as wire,
        gateway.scenario() as scenario,
    ):
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/files/f2/content")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("GET", path, key=key)
        assert response.status_code == 200, response.text
        assert response.content == lines, response.content
        assert len(wire.drain()) == 1


def test_an_event_stream_response_is_relayed_frame_for_frame(gateway: Gateway) -> None:
    """The caller sends `stream: true` and the vendor answers `text/event-stream` with named events, a comment
    line, an `id:` field and a `[DONE]` sentinel.

    The caller must receive a `text/event-stream` response whose bytes equal the vendor's frames joined in
    order: no frame dropped, rewritten into another dialect, or given an extra terminator.
    """
    reply: Final = Reply(content_type="text/event-stream", chunks=_SSE_FRAMES)
    with wire_server(always(reply)) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/stream")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, {"stream": True, "prompt": "go"}, key=key)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        assert response.content == b"".join(_SSE_FRAMES), response.content
        assert json_body(only_request(wire)) == {"stream": True, "prompt": "go"}


def test_an_event_stream_is_relayed_even_when_the_request_did_not_ask_to_stream(gateway: Gateway) -> None:
    """The caller's body has no `stream` field but the vendor still answers `text/event-stream`.

    The caller must receive the frames unchanged with an event-stream content type. The relay decides from the
    vendor's response, so a vendor that always streams keeps working.
    """
    reply: Final = Reply(content_type="text/event-stream", chunks=_SSE_FRAMES)
    with wire_server(always(reply)) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/stream")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, {"prompt": "go"}, key=key)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        assert response.content == b"".join(_SSE_FRAMES), response.content
        assert len(wire.drain()) == 1


def test_the_first_stream_frame_reaches_the_caller_before_the_upstream_finishes(gateway: Gateway) -> None:
    """The vendor sends one SSE frame and then blocks until the test releases it.

    While the vendor is blocked, the caller must already be able to read the complete first frame. Only then is
    the vendor released and the rest must follow unchanged. A relay that buffers the whole body would deadlock
    here until the harness timeout.
    """
    gate: Final = threading.Event()
    reply: Final = Reply(content_type="text/event-stream", chunks=_SSE_FRAMES, gate_after_first=gate)
    with wire_server(always(reply)) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/stream")
        key: Final = granted_key(scenario, path)
        with gateway.client.stream(
            "POST", path, json={"stream": True}, headers={"Authorization": f"Bearer {key}"}
        ) as response:
            raw: Final = iter(response.iter_raw())
            try:
                assert response.status_code == 200, response.status_code
                first: Final = first_sse_frame(raw)
                assert first == _SSE_FRAMES[0], first
            finally:
                gate.set()
            streamed: Final = first + b"".join(raw)
        assert streamed == b"".join(_SSE_FRAMES), streamed
        assert len(wire.drain()) == 1


def test_an_upstream_error_on_a_streaming_request_keeps_its_status_and_body(gateway: Gateway) -> None:
    """The caller sends `stream: true` and the vendor refuses with 429 and a JSON error body.

    The caller must receive 429 and a body that parses to the vendor's JSON, not a 200 event stream that carries
    the error inside it.
    """
    error: Final[dict[str, JsonValue]] = {"error": {"type": "rate_limit", "message": "slow down"}}
    with wire_server(always(json_reply(error, status=429))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/stream")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, {"stream": True}, key=key)
        assert response.status_code == 429, response.text
        assert response.json() == error, response.text
        assert len(wire.drain()) == 1


def test_a_multipart_upload_reaches_the_target_with_its_file_bytes_and_fields(gateway: Gateway) -> None:
    """A granted key uploads a multipart form with one binary file part and two text fields.

    The vendor must receive a `multipart/form-data` request whose parts are the same field names, the same file
    name and the identical file bytes. The boundary may differ, the content may not.
    """
    payload: Final = b"\x00\x01RIFF-scripted-audio\xff\xfe"
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/audio")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request_multipart(
            path,
            {"model": "scribe-1", "language": "en"},
            {"file": ("clip.wav", payload, "audio/wav")},
            key=key,
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert upstream.headers["content-type"].startswith("multipart/form-data; boundary="), dict(upstream.headers)
        assert sorted(multipart_parts(upstream), key=lambda part: part[0]) == [
            ("file", "clip.wav", payload),
            ("language", None, b"en"),
            ("model", None, b"scribe-1"),
        ]


def test_a_multipart_form_without_files_stays_multipart(gateway: Gateway) -> None:
    """A granted key posts a multipart form that has only text fields.

    The vendor must still receive `multipart/form-data` with both fields. An HTTP client library downgrades a
    file-less form to `application/x-www-form-urlencoded` unless the relay prevents it, and vendors that only
    parse multipart then see an empty form.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/forms")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.client.post(
            path,
            files={"prompt": (None, "draw a cat"), "size": (None, "512x512")},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert upstream.headers["content-type"].startswith("multipart/form-data; boundary="), dict(upstream.headers)
        assert sorted(multipart_parts(upstream), key=lambda part: part[0]) == [
            ("prompt", None, b"draw a cat"),
            ("size", None, b"512x512"),
        ]


def test_repeated_multipart_file_fields_are_all_forwarded(gateway: Gateway) -> None:
    """A granted key uploads two files under the same field name `image`.

    The vendor must receive both parts with their own file names and bytes, in order. Collapsing a repeated
    field to its last value silently drops the first image.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/images/edits")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.client.post(
            path,
            files=[("image", ("a.png", b"\x89PNG-a", "image/png")), ("image", ("b.png", b"\x89PNG-b", "image/png"))],
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        assert multipart_parts(only_request(wire)) == (
            ("image", "a.png", b"\x89PNG-a"),
            ("image", "b.png", b"\x89PNG-b"),
        )


def test_a_json_envelope_with_custom_body_and_query_params_replaces_what_is_forwarded(gateway: Gateway) -> None:
    """The caller posts `{"custom_body": {...}, "query_params": {...}}` to a configured endpoint.

    The vendor must receive only the inner `custom_body` object as its JSON body and the inner `query_params` as
    its query string. This envelope is a long-standing way for clients that cannot set a query string to drive
    a pass-through endpoint, and nothing else documents it.
    """
    inner: Final[dict[str, JsonValue]] = {"documents": ["a", "b"], "top_n": 1}
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/rerank")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request(
            "POST", path, {"custom_body": inner, "query_params": {"version": "3", "dry_run": "true"}}, key=key
        )
        assert response.status_code == 200, response.text
        upstream: Final = only_request(wire)
        assert json_body(upstream) == inner, upstream.body
        assert query_of(upstream) == {"version": "3", "dry_run": "true"}, upstream.target


def test_a_deleted_endpoint_stops_forwarding(gateway: Gateway) -> None:
    """An endpoint is created, called once successfully, then removed with
    `DELETE /config/pass_through_endpoint?endpoint_id=...`.

    The next call to its path must not reach the vendor and must answer 404.
    """
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        path: Final = f"/dec1-native-deleted-{uuid.uuid4().hex}"
        created: Final = gateway.post(
            "/config/pass_through_endpoint", {"path": path, "target": f"{wire.url}/v1/jobs", "auth": True}
        )
        endpoint_id: Final = str(object_value(JSON_LIST.validate_python(created["endpoints"])[0])["id"])
        key: Final = granted_key(scenario, path)
        assert gateway.request("POST", path, _BODY, key=key).status_code == 200
        assert len(wire.drain()) == 1
        deleted: Final = gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": endpoint_id})
        assert deleted.status_code == 200, deleted.text
        after: Final = gateway.request("POST", path, _BODY, key=key)
        assert after.status_code == 404, after.text
        assert wire.drain() == ()


def test_a_created_endpoint_is_listed_with_the_options_it_was_created_with(gateway: Gateway) -> None:
    """An endpoint is created with a target, headers, `include_subpath`, `cost_per_request`, `methods` and
    `default_query_params`.

    `GET /config/pass_through_endpoint` must list one entry for its path whose fields equal what was sent, with
    `auth` true and `is_from_config` false. The admin UI edits endpoints from this listing.
    """
    options: Final[dict[str, JsonValue]] = {
        "headers": {"x-vendor-key": "synthetic-vendor-key"},
        "include_subpath": True,
        "cost_per_request": 0.125,
        "methods": ["GET", "POST"],
        "default_query_params": {"api-version": "2024-01"},
    }
    with wire_server(always(json_reply(_REPLY))) as wire, gateway.scenario() as scenario:
        target: Final = f"{wire.url}/api"
        path: Final = register_pass_through(gateway, scenario, target, **options)
        listed: Final = JSON_LIST.validate_python(gateway.get("/config/pass_through_endpoint")["endpoints"])
        mine: Final = [object_value(entry) for entry in listed if object_value(entry)["path"] == path]
        assert len(mine) == 1, listed
        shown: Final = {name: mine[0][name] for name in (*options, "target", "auth", "is_from_config")}
        assert shown == {**options, "target": target, "auth": True, "is_from_config": False}, mine[0]


def test_a_request_to_an_unreachable_target_fails_without_hanging(gateway: Gateway) -> None:
    """The endpoint targets a port nothing listens on.

    The caller must get an error status of 500 or above with a JSON `error` object, and the response must still
    carry `x-litellm-call-id`. UNVERIFIED: the exact status for a connection failure was not confirmed; see the
    final `except Exception` branch of `pass_through_request` in `pass_through_endpoints.py`.
    """
    with wire_server(always(json_reply(_REPLY))) as wire:
        closed_port_url: Final = wire.url
    with gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{closed_port_url}/v1/jobs")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, _BODY, key=key)
        assert response.status_code >= 500, response.text
        assert isinstance(response.json()["error"], dict), response.text
        assert "x-litellm-call-id" in response.headers, dict(response.headers)


def _upstream_body_is_untouched(request: Request) -> Reply:
    return Reply(body=request.body)


def test_litellm_only_body_fields_are_not_added_to_the_forwarded_json(gateway: Gateway) -> None:
    """The vendor echoes the request body it received and the caller sends a three-key JSON object.

    The echoed body must have exactly the caller's three keys. The relay keeps a logging object and guardrail
    flags next to the parsed body internally, and none of that may be serialised to the vendor.
    """
    sent: Final[dict[str, JsonValue]] = {"engine": "vendor-engine", "input": "hello", "options": {"k": 1}}
    with wire_server(_upstream_body_is_untouched) as wire, gateway.scenario() as scenario:
        path: Final = register_pass_through(gateway, scenario, f"{wire.url}/v1/echo")
        key: Final = granted_key(scenario, path)
        response: Final = gateway.request("POST", path, sent, key=key)
        assert response.status_code == 200, response.text
        assert response.json() == sent, response.text
        assert len(wire.drain()) == 1
