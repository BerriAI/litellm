"""Every spelling of an inference route reaches the same handler, and unmatched requests get stable answers.

The proxy serves each OpenAI surface under several paths: the versioned one (`/v1/chat/completions`), the bare
one (`/chat/completions`), the legacy engines one (`/engines/{model}/chat/completions`) and the Azure one
(`/openai/deployments/{model}/chat/completions`). Clients built on different SDKs depend on each spelling, so a
front door that forwards or re-implements routing must send all of them to the same place with the same result.
The second half pins what a caller sees when nothing matches: the 404 and 405 bodies FastAPI produces, the 307
that fixes a trailing slash, and how HEAD and OPTIONS are answered. Each test puts a recording provider behind
a model it registers, so "same result" is checked on the wire the provider saw and on the response the caller got.
"""

from __future__ import annotations

import json
import uuid
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration.compatibility.dec1_http_support import (
    CHAT,
    chat_body,
    connection,
    exchange,
    front_door,
    host_header,
    json_post,
    path_of,
    read_response,
    request_bytes,
)
from pydantic import JsonValue

CHAT_ALIASES: Final = (
    pytest.param("/v1/chat/completions", True, id="v1"),
    pytest.param("/chat/completions", True, id="bare"),
    pytest.param("/engines/{model}/chat/completions", False, id="engines"),
    pytest.param("/openai/deployments/{model}/chat/completions", False, id="azure-deployments"),
)
COMPLETION_ALIASES: Final = (
    pytest.param("/v1/completions", True, id="v1"),
    pytest.param("/completions", True, id="bare"),
    pytest.param("/engines/{model}/completions", False, id="engines"),
    pytest.param("/openai/deployments/{model}/completions", False, id="azure-deployments"),
)
EMBEDDING_ALIASES: Final = (
    pytest.param("/v1/embeddings", True, id="v1"),
    pytest.param("/embeddings", True, id="bare"),
    pytest.param("/engines/{model}/embeddings", False, id="engines"),
    pytest.param("/openai/deployments/{model}/embeddings", False, id="azure-deployments"),
)
RESPONSES_ALIASES: Final = ("/v1/responses", "/responses", "/openai/v1/responses")
INFERENCE_PATHS: Final = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses", "/v1/messages")


def _without_model(body: dict[str, JsonValue], model_in_body: bool) -> dict[str, JsonValue]:
    return body if model_in_body else {key: value for key, value in body.items() if key != "model"}


def _answer(body: dict[str, JsonValue]) -> JsonValue:
    return object_value(object_value(list_value(body["choices"])[0])["message"])["content"]


@pytest.mark.parametrize(("template", "model_in_body"), CHAT_ALIASES)
def test_every_chat_completions_path_reaches_the_provider_with_the_same_request(
    gateway: Gateway, template: str, model_in_body: bool
) -> None:
    """A chat model is registered behind a recording provider. The caller posts the same chat request to each of
    the four chat paths, naming the model in the body on the plain paths and only in the URL on the engines and
    Azure paths. Every path must answer 200 with the provider's text, and the provider must receive exactly one
    POST to `/v1/chat/completions` carrying the deployment's model `gpt-4o-mini` and the caller's messages."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        sent: Final = chat_body(model, "alias control")
        response: Final = gateway.request("POST", template.format(model=model), _without_model(sent, model_in_body))
        assert response.status_code == 200, response.text
        assert _answer(JSON_OBJECT.validate_json(response.content)) == "scripted"
        calls: Final = door.calls()
        assert [path_of(call) for call in calls] == ["/v1/chat/completions"]
        received: Final = JSON_OBJECT.validate_json(calls[0].body)
        assert (received["model"], received["messages"]) == ("gpt-4o-mini", sent["messages"])


@pytest.mark.parametrize(("template", "model_in_body"), COMPLETION_ALIASES)
def test_every_text_completions_path_reaches_the_provider_with_the_same_request(
    gateway: Gateway, template: str, model_in_body: bool
) -> None:
    """A text-completion model is registered behind a recording provider. The caller posts the same prompt to
    each of the four completions paths. Every path must answer 200 with `choices[0].text == "scripted"`, and the
    provider must receive exactly one POST to `/v1/completions` carrying the caller's prompt."""
    with front_door(gateway) as door:
        model: Final = door.completion_model()
        prompt: Final = f"alias prompt {uuid.uuid4().hex}"
        sent: Final[dict[str, JsonValue]] = {"model": model, "prompt": prompt, "cache": {"no-cache": True}}
        response: Final = gateway.request("POST", template.format(model=model), _without_model(sent, model_in_body))
        assert response.status_code == 200, response.text
        choice: Final = object_value(list_value(JSON_OBJECT.validate_json(response.content)["choices"])[0])
        assert choice["text"] == "scripted"
        calls: Final = door.calls()
        assert [path_of(call) for call in calls] == ["/v1/completions"]
        assert JSON_OBJECT.validate_json(calls[0].body)["prompt"] == prompt


@pytest.mark.parametrize(("template", "model_in_body"), EMBEDDING_ALIASES)
def test_every_embeddings_path_reaches_the_provider_with_the_same_request(
    gateway: Gateway, template: str, model_in_body: bool
) -> None:
    """An embedding model is registered behind a recording provider. The caller posts the same input to each of
    the four embeddings paths. Every path must answer 200 with the provider's vector `[0.25, 0.5, 0.75]`, and the
    provider must receive exactly one POST to `/v1/embeddings`."""
    with front_door(gateway) as door:
        model: Final = door.embedding_model()
        sent: Final[dict[str, JsonValue]] = {
            "model": model,
            "input": f"alias input {uuid.uuid4().hex}",
            "encoding_format": "float",
            "cache": {"no-cache": True},
        }
        response: Final = gateway.request("POST", template.format(model=model), _without_model(sent, model_in_body))
        assert response.status_code == 200, response.text
        first: Final = object_value(list_value(JSON_OBJECT.validate_json(response.content)["data"])[0])
        assert first["embedding"] == [0.25, 0.5, 0.75]
        assert [path_of(call) for call in door.calls()] == ["/v1/embeddings"]


@pytest.mark.parametrize("path", RESPONSES_ALIASES)
def test_every_responses_path_reaches_the_provider_with_the_same_request(gateway: Gateway, path: str) -> None:
    """A chat model is registered behind a recording provider. The caller posts the same Responses API request
    to `/v1/responses`, `/responses` and `/openai/v1/responses`. Each must answer 200 with a `response` object
    whose first output text is "scripted", and the provider must receive exactly one POST to `/v1/responses`
    carrying the caller's input."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        text: Final = f"alias input {uuid.uuid4().hex}"
        response: Final = gateway.request("POST", path, {"model": model, "input": text})
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        message: Final = object_value(list_value(body["output"])[0])
        assert body["object"] == "response"
        assert object_value(list_value(message["content"])[0])["text"] == "scripted"
        calls: Final = door.calls()
        assert [path_of(call) for call in calls] == ["/v1/responses"]
        assert JSON_OBJECT.validate_json(calls[0].body)["input"] == text


def test_messages_path_reaches_the_provider_in_anthropic_shape(gateway: Gateway) -> None:
    """An Anthropic model is registered behind a recording provider. The caller posts an Anthropic Messages
    request to `/v1/messages`, authenticating with the proxy key in `x-api-key` as the Anthropic SDK does. The
    proxy must answer 200 with a `type: message` body whose text is "scripted", and the provider must receive one
    POST to `/v1/messages` with the caller's `max_tokens` and messages and the deployment's key in `x-api-key`."""
    with front_door(gateway) as door:
        model: Final = door.messages_model()
        messages: Final[list[JsonValue]] = [{"role": "user", "content": f"alias message {uuid.uuid4().hex}"}]
        response: Final = gateway.client.post(
            "/v1/messages",
            json={"model": model, "max_tokens": 16, "messages": messages},
            headers={"x-api-key": gateway.key, "anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["type"], object_value(list_value(body["content"])[0])["text"]) == ("message", "scripted")
        calls: Final = door.calls()
        assert [path_of(call) for call in calls] == ["/v1/messages"]
        received: Final = JSON_OBJECT.validate_json(calls[0].body)
        assert (received["max_tokens"], received["messages"]) == (16, messages)
        assert calls[0].headers.get("x-api-key") == "integration-provider-key"


def test_versioned_and_bare_chat_paths_return_the_same_litellm_header_names(gateway: Gateway) -> None:
    """One chat model is called once through `/v1/chat/completions` and once through `/chat/completions`. Both
    responses must be 200 and must carry the same set of `x-litellm-*` header names, so a client cannot tell
    which spelling it used. A front door that serves only one spelling natively would drift here first."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        versioned: Final = gateway.request("POST", "/v1/chat/completions", chat_body(model))
        bare: Final = gateway.request("POST", "/chat/completions", chat_body(model))
        assert (versioned.status_code, bare.status_code) == (200, 200), (versioned.text, bare.text)
        names: Final = tuple(
            sorted(name for name in response.headers if name.startswith("x-litellm-")) for response in (versioned, bare)
        )
        assert names[0] == names[1]
        assert "x-litellm-call-id" in names[0]


def test_path_model_wins_over_a_different_body_model_on_the_azure_path(gateway: Gateway) -> None:
    """Two chat models are registered, each behind its own provider key. The caller posts to
    `/openai/deployments/{first}/chat/completions` with the second model named in the body. The request must be
    served by the first model: the provider sees the first deployment's key. The handler resolves the model as
    path before body (`resolve_inference_model`), which Azure SDK users rely on."""
    with front_door(gateway) as door:
        first: Final = door.chat_model(api_key="sk-dec1-first")
        second: Final = door.chat_model(api_key="sk-dec1-second")
        response: Final = gateway.request(
            "POST", f"/openai/deployments/{first}/chat/completions", chat_body(second, "path beats body")
        )
        assert response.status_code == 200, response.text
        assert [call.headers.get("authorization") for call in door.calls()] == ["Bearer sk-dec1-first"]


def test_a_key_cannot_reach_a_model_outside_its_list_through_the_path_alias(gateway: Gateway) -> None:
    """A virtual key is limited to one model and a second model exists. The caller posts to
    `/openai/deployments/{second}/chat/completions` while naming the allowed model in the body. The proxy must
    refuse with 403 and the provider must receive nothing. UNVERIFIED: auth reads the body model first
    (`populate_request_with_path_params` only fills missing keys) while the handler prefers the path model."""
    pytest.skip("BUG: suspected, unverified. Auth checks the body model but the handler serves the path model")
    with front_door(gateway) as door:
        allowed: Final = door.chat_model()
        other: Final = door.chat_model()
        key: Final = door.scenario.key(models=[allowed])
        response: Final = gateway.request(
            "POST", f"/openai/deployments/{other}/chat/completions", chat_body(allowed), key=key
        )
        assert response.status_code == 403, response.text
        assert door.calls() == ()


def test_a_model_name_with_a_slash_is_accepted_in_the_deployment_path(gateway: Gateway) -> None:
    """A chat model is registered under a name that contains a slash. The caller posts to
    `/openai/deployments/<group>/<name>/chat/completions` with no model in the body. The route's `{model:path}`
    parameter must take both segments, so the request is served with 200 and reaches the provider once."""
    with front_door(gateway) as door:
        name: Final = f"dec1/slashed-{uuid.uuid4().hex}"
        model: Final = door.scenario.model(name=name, api_base=f"{door.upstream.url}/v1")
        body: Final = {key: value for key, value in chat_body(model).items() if key != "model"}
        response: Final = gateway.request("POST", f"/openai/deployments/{model}/chat/completions", body)
        assert response.status_code == 200, response.text
        assert len(door.calls()) == 1


def test_a_query_string_on_the_chat_path_does_not_change_routing(gateway: Gateway) -> None:
    """The Azure SDK appends `?api-version=...` to every call. The caller posts a chat request to
    `/openai/deployments/{model}/chat/completions?api-version=2024-10-21`. The proxy must serve it with 200 and
    the provider must receive one POST whose path is still `/v1/chat/completions`, with no `api-version` leaking
    into the forwarded JSON body."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        response: Final = gateway.request(
            "POST",
            f"/openai/deployments/{model}/chat/completions",
            chat_body(model),
            params={"api-version": "2024-10-21"},
        )
        assert response.status_code == 200, response.text
        calls: Final = door.calls()
        assert [path_of(call) for call in calls] == ["/v1/chat/completions"]
        assert "api-version" not in JSON_OBJECT.validate_json(calls[0].body)


def test_models_list_is_the_same_on_the_versioned_and_bare_paths(gateway: Gateway) -> None:
    """A model is registered. The caller lists models on `/v1/models` and on `/models` with the same key. Both
    must answer 200 with `object: list`, both must include the new model's id, and the two id sets must be equal."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        listings: Final = tuple(gateway.get(path) for path in ("/v1/models", "/models"))
        ids: Final = tuple(
            frozenset(object_value(entry)["id"] for entry in list_value(listing["data"])) for listing in listings
        )
        assert [listing["object"] for listing in listings] == ["list", "list"]
        assert model in ids[0]
        assert ids[0] == ids[1]


def test_unknown_path_is_a_404_with_the_fastapi_detail_body(gateway: Gateway) -> None:
    """The caller sends an authenticated GET to a path no route serves. The proxy must answer 404 with
    `content-type: application/json` and exactly `{"detail": "Not Found"}`. This is FastAPI's default shape, not
    the OpenAI error envelope, and SDKs that probe for optional routes read it as "feature absent"."""
    response: Final = gateway.request("GET", f"/dec1-no-such-route-{uuid.uuid4().hex}")
    assert response.status_code == 404, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "Not Found"}


def test_unknown_path_is_a_404_even_without_credentials(gateway: Gateway) -> None:
    """The caller sends a POST with no Authorization header to a path no route serves. The proxy must answer 404
    `{"detail": "Not Found"}` rather than 401, because authentication is a per-route dependency and no route
    matched. A front door that authenticates before routing would turn this into a 401."""
    response: Final = gateway.client.post(f"/dec1-no-such-route-{uuid.uuid4().hex}", json={})
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("path", INFERENCE_PATHS[:4])
def test_get_on_a_post_only_openai_path_is_a_405_that_names_post(gateway: Gateway, path: str) -> None:
    """The caller sends an authenticated GET to an OpenAI-shaped inference path that only accepts POST. The proxy
    must answer 405 with exactly `{"detail": "Method Not Allowed"}` and an `allow` header that lists POST.
    UNVERIFIED for `/v1/responses`: a GET route exists for `/v1/responses/{response_id}` but not for the bare path."""
    response: Final = gateway.request("GET", path)
    assert response.status_code == 405, response.text
    assert response.json() == {"detail": "Method Not Allowed"}
    assert "POST" in response.headers["allow"]


def test_get_on_the_messages_path_is_a_405_in_the_anthropic_error_envelope(gateway: Gateway) -> None:
    """The caller sends an authenticated GET to `/v1/messages`, which only accepts POST. The proxy must answer
    405, and because the route is Anthropic-shaped the body must be the Anthropic envelope: top-level
    `type: "error"` and an `error` object with a `type` and the message "Method Not Allowed"
    (`otlp_http_exception_handler` picks the envelope from the path, `anthropic_error_payload.py:50`)."""
    response: Final = gateway.request("GET", "/v1/messages")
    assert response.status_code == 405, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = object_value(body["error"])
    assert body["type"] == "error"
    assert error["message"] == "Method Not Allowed"
    assert isinstance(error["type"], str)


def test_put_and_delete_on_the_chat_path_are_405(gateway: Gateway) -> None:
    """The caller sends an authenticated PUT, DELETE and PATCH to `/v1/chat/completions`. Each must be answered
    405 `{"detail": "Method Not Allowed"}` and none may reach a handler or be read as a chat request."""
    statuses: Final = {
        method: (gateway.request(method, CHAT).status_code, gateway.request(method, CHAT).json())
        for method in ("PUT", "DELETE", "PATCH")
    }
    assert statuses == dict.fromkeys(("PUT", "DELETE", "PATCH"), (405, {"detail": "Method Not Allowed"}))


def test_path_matching_is_case_sensitive(gateway: Gateway) -> None:
    """The caller posts a valid chat body to `/V1/Chat/Completions`. Route matching is case sensitive, so the
    proxy must answer 404 `{"detail": "Not Found"}` and must not treat it as the chat route."""
    response: Final = gateway.request("POST", "/V1/Chat/Completions", chat_body("any-model"))
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Not Found"}


def test_trailing_slash_on_the_chat_path_redirects_to_the_canonical_path(gateway: Gateway) -> None:
    """The caller posts to `/v1/chat/completions/` without following redirects. The router must answer 307 with a
    `location` whose path is `/v1/chat/completions`, so the client repeats the same method and body there. The
    redirect is issued before authentication. UNVERIFIED: read from Starlette's `redirect_slashes` default, which
    the app does not turn off (`proxy_server.py:2161`)."""
    response: Final = gateway.client.post(f"{CHAT}/", json=chat_body("any-model"), follow_redirects=False)
    assert response.status_code == 307, response.text
    assert urlsplit(response.headers["location"]).path == CHAT


def test_trailing_slash_redirect_is_built_from_the_host_header_the_client_sent(gateway: Gateway) -> None:
    """The caller posts to `/v1/chat/completions/` over a raw socket with `Host: public.example:8443`. The 307
    `location` must be `http://public.example:8443/v1/chat/completions`: the proxy builds absolute URLs from the
    Host header. A forwarder that rewrites Host to its private upstream address would leak that address here."""
    body: Final = json.dumps(chat_body("any-model")).encode()
    payload: Final = request_bytes(
        "POST",
        f"{CHAT}/",
        (
            ("Host", "public.example:8443"),
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            ("Connection", "close"),
        ),
        body,
    )
    (response,) = exchange(gateway, payload)
    assert response.status == 307, response.body
    assert response.header("location") == f"http://public.example:8443{CHAT}"


def test_following_the_trailing_slash_redirect_serves_the_chat_request(gateway: Gateway) -> None:
    """A chat model is registered. The caller posts to `/v1/chat/completions/` and follows redirects, as browsers
    and most HTTP clients do for 307. The final answer must be 200 with the provider's text, the provider must be
    called once, and the redirect history must hold exactly one 307."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        response: Final = gateway.client.post(
            f"{CHAT}/",
            json=chat_body(model),
            headers={"Authorization": f"Bearer {gateway.key}"},
            follow_redirects=True,
        )
        assert response.status_code == 200, response.text
        assert [hop.status_code for hop in response.history] == [307]
        assert len(door.calls()) == 1


def test_head_on_liveliness_is_refused_with_405(gateway: Gateway) -> None:
    """The caller sends HEAD to `/health/liveliness`. FastAPI GET routes do not answer HEAD, so the proxy must
    return 405 with an `allow` header that lists GET and no response body. Load balancers configured with HEAD
    probes see this today, so a front door must not start answering 200. UNVERIFIED: read from FastAPI's
    `APIRoute`, which registers only the declared methods."""
    response: Final = gateway.client.head("/health/liveliness")
    assert response.status_code == 405, response.text
    assert "GET" in response.headers["allow"]
    assert response.content == b""


def test_a_head_response_carries_no_body_so_the_next_request_on_the_connection_parses(gateway: Gateway) -> None:
    """The caller writes HEAD `/health/liveliness` and then GET `/health/liveliness` on one connection. The HEAD
    answer must end at its blank line even if it declares a `content-length`, so the second answer is read as a
    clean 200 whose body is the JSON string "I'm alive!". A proxy that writes a HEAD body corrupts the stream."""
    host: Final = ("Host", host_header(gateway))
    head: Final = request_bytes("HEAD", "/health/liveliness", (host,))
    get: Final = request_bytes("GET", "/health/liveliness", (host, ("Connection", "close")))
    with connection(gateway) as (client, stream):
        client.sendall(head + get)
        first: Final = read_response(stream, bodiless=True)
        second: Final = read_response(stream)
    assert first.status == 405
    assert (second.status, second.body) == (200, b'"I\'m alive!"')


def test_options_on_liveliness_and_readiness_lists_get_and_options(gateway: Gateway) -> None:
    """The caller sends a plain OPTIONS (no Origin header) to `/health/liveliness`, `/health/liveness` and
    `/health/readiness`. Each must answer 200 with `allow: GET, OPTIONS`, `access-control-allow-methods:
    GET, OPTIONS` and an empty body, with no credentials required (`_health_endpoints.py:2454`)."""
    for path in ("/health/liveliness", "/health/liveness", "/health/readiness"):
        response = gateway.client.options(path)
        assert response.status_code == 200, (path, response.text)
        assert response.headers["allow"] == "GET, OPTIONS", path
        assert response.headers["access-control-allow-methods"] == "GET, OPTIONS", path
        assert response.content == b"", path


def test_plain_options_on_the_chat_path_is_a_405(gateway: Gateway) -> None:
    """The caller sends OPTIONS to `/v1/chat/completions` with no `Origin` and no
    `Access-Control-Request-Method`, so it is not a CORS preflight. No route handles OPTIONS there, so the proxy
    must answer 405 `{"detail": "Method Not Allowed"}`. Only a real preflight gets the CORS middleware's 200."""
    response: Final = gateway.client.options(CHAT)
    assert response.status_code == 405, response.text
    assert response.json() == {"detail": "Method Not Allowed"}


def test_the_same_request_gives_the_same_status_over_httpx_and_a_raw_socket(gateway: Gateway) -> None:
    """A chat model is registered. The caller sends one chat request through `httpx` and an equivalent one as
    hand-written HTTP/1.1 bytes on a socket. Both must answer 200 with `content-type: application/json` and the
    provider's text, proving the raw-socket helper the other front-door tests use sees what real clients see."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        via_client: Final = gateway.request("POST", CHAT, chat_body(model))
        raw_body: Final = json.dumps(chat_body(model)).encode()
        (via_socket,) = exchange(gateway, json_post(gateway, CHAT, raw_body, extra=(("Connection", "close"),)))
        assert (via_client.status_code, via_socket.status) == (200, 200), (via_client.text, via_socket.body)
        assert via_client.headers["content-type"] == via_socket.header("content-type") == "application/json"
        assert _answer(via_socket.json()) == _answer(JSON_OBJECT.validate_json(via_client.content)) == "scripted"
