"""Vertex AI context caching (`cachedContents`) driven by `cache_control` on chat messages.

When a chat request to a Vertex Gemini deployment marks a leading block of messages with
`cache_control: {"type": "ephemeral"}`, `ContextCachingEndpoints.async_check_and_create_cache`
(`litellm/llms/vertex_ai/context_caching/vertex_ai_context_caching.py:561`) first lists
`{api_base}/v1/projects/{project}/locations/{location}/cachedContents` (GET, following `nextPageToken`), looks
for an entry whose `displayName` is the hash of the cached block, creates one with POST when none is found, and
then sends `generateContent` with `cachedContent: <name>` and only the uncached turns. The create body names the
model as the full Vertex resource `projects/{p}/locations/{l}/publishers/google/models/{model}`
(`transformation.py:172`), carries the cached system prompt and tools, and a `ttl` normalised to seconds.

Differences from Google AI Studio that a Rust port must keep: the Vertex collection is per project and
location, authenticated with the minted bearer rather than `x-goog-api-key`; a 403 on the list is treated as
"nothing cached" and the cache is created anyway; a 404 on the list is NOT skipped the way Gemini's is. The AI
Studio side is pinned in `test_dec1_gemini_files_caching_wire.py`, and the custom `api_base` naming a model
resource in `test_gemini_context_caching_custom_api_base_wire.py`.

Each test uses its own peer (an "open" peer that records GETs too) and a cached text holding a fresh marker,
so the in-process known-cache map and the shared response cache never answer for another test.
"""

from __future__ import annotations

import json
import threading
from typing import Final

import httpx
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_vertex3_support import (
    OpenPeer,
    is_token_from,
    json_of,
    location_path,
    open_peer,
    register_open_vertex,
)
from integration.providers.dec1_vertex_support import (
    GEMINI,
    LOCATION,
    PROJECT,
    body_of,
    gemini_json,
    google_error,
    marker,
    models_path,
    path_of,
    query_of,
    usage_metadata,
)
from pydantic import JsonValue

_COLLECTION: Final = f"{location_path(LOCATION)}/cachedContents"
_GENERATE: Final = f"{models_path()}:generateContent"
_MODEL_RESOURCE: Final = f"projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{GEMINI}"
_CACHED_TOKENS: Final = 5120


def _cached_text() -> str:
    """About 10k tokens of policy text with a fresh marker, well above Vertex's minimum cacheable size."""
    tag: Final = marker()
    return " ".join(f"Policy clause {index} ({tag}): the gateway keeps this clause cached." for index in range(700))


def _cached_messages(
    text: str, question: str, *, ttl: str | None = None, system: str | None = None
) -> list[dict[str, JsonValue]]:
    control: Final[dict[str, JsonValue]] = {"type": "ephemeral", **({"ttl": ttl} if ttl is not None else {})}
    lead: Final[list[dict[str, JsonValue]]] = (
        [{"role": "system", "content": [{"type": "text", "text": system, "cache_control": control}]}]
        if system is not None
        else []
    )
    return [
        *lead,
        {"role": "user", "content": [{"type": "text", "text": text, "cache_control": control}]},
        {"role": "user", "content": question},
    ]


class _Google:
    """A scripted Vertex host: the `cachedContents` collection plus `generateContent`."""

    def __init__(
        self,
        *,
        pages: tuple[dict[str, JsonValue], ...] = ({"cachedContents": []},),
        list_error: Reply | None = None,
        create_error: Reply | None = None,
        create_expire: str | None = "2999-01-01T00:00:00Z",
    ) -> None:
        self.pages = pages
        self.list_error = list_error
        self.create_error = create_error
        self.create_expire = create_expire
        self.created: list[dict[str, JsonValue]] = []
        self.listed: list[dict[str, JsonValue]] = []
        self.list_calls = 0
        self._lock = threading.Lock()

    def name(self) -> str:
        return f"projects/{PROJECT}/locations/{LOCATION}/cachedContents/dec1-cache-{len(self.created)}"

    def __call__(self, request: Request) -> Reply:
        path: Final = path_of(request)
        if path == _COLLECTION and request.method == "GET":
            if self.list_error is not None:
                return self.list_error
            with self._lock:
                index: Final = self.list_calls
                self.list_calls += 1
            page: Final = (
                self.pages[min(index, len(self.pages) - 1)] if not self.created else {"cachedContents": self.listed}
            )
            return Reply(body=json.dumps(page).encode())
        if path == _COLLECTION and request.method == "POST":
            if self.create_error is not None:
                return self.create_error
            with self._lock:
                self.created.append(body_of(request))
                entry: dict[str, JsonValue] = {
                    "name": self.name(),
                    "displayName": body_of(request)["displayName"],
                    "model": _MODEL_RESOURCE,
                    "usageMetadata": {"totalTokenCount": _CACHED_TOKENS},
                }
                if self.create_expire is not None:
                    entry["expireTime"] = self.create_expire
                self.listed.append(entry)
            return Reply(body=json.dumps(entry).encode())
        if request.method == "POST" and path.endswith(":generateContent"):
            return Reply(
                body=json.dumps(
                    gemini_json(usage=usage_metadata(_CACHED_TOKENS + 9, 2, cachedContentTokenCount=_CACHED_TOKENS))
                ).encode()
            )
        return Reply(body=b"{}")


def _chat(gateway: Gateway, model: str, messages: list[dict[str, JsonValue]], **fields: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": messages, "max_tokens": 16, **fields}
    )


def _calls(peer: OpenPeer, method: str, path: str) -> list[Request]:
    return [request for request in peer.calls() if request.method == method and path_of(request) == path]


def test_a_cached_block_lists_then_creates_then_generates_on_the_vertex_collection(gateway: Gateway) -> None:
    """A first request with a cached block makes exactly three calls in order: `GET
    /v1/projects/dec1-vertex-project/locations/us-central1/cachedContents`, `POST` to the same collection, and
    `POST .../publishers/google/models/gemini-2.5-flash:generateContent`."""
    google: Final = _Google()
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        response: Final = _chat(gateway, model, _cached_messages(_cached_text(), f"How many clauses? {marker()}"))
        assert response.status_code == 200, response.text
        assert peer.shape() == [("GET", _COLLECTION), ("POST", _COLLECTION), ("POST", _GENERATE)]


def test_the_vertex_cache_create_body_names_the_full_model_resource(gateway: Gateway) -> None:
    """The create body's `model` is `projects/dec1-vertex-project/locations/us-central1/publishers/google/
    models/gemini-2.5-flash` (AI Studio uses `models/<model>`), its `displayName` is a non-empty cache key
    and its `contents` hold the cached text."""
    google: Final = _Google()
    text: Final = _cached_text()
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(text, f"q {marker()}")).status_code == 200
        created: Final = google.created[0]
        assert created["model"] == _MODEL_RESOURCE
        assert isinstance(created["displayName"], str) and created["displayName"]
        assert text in json.dumps(created["contents"])


def test_the_generate_call_sends_only_the_uncached_turn_with_the_cache_name(gateway: Gateway) -> None:
    """The `generateContent` body carries `cachedContent` equal to the `name` the create call returned
    and its `contents` hold only the trailing question, never the cached text, which is what makes the
    cache save money."""
    google: Final = _Google()
    text: Final = _cached_text()
    question: Final = f"How many clauses are there? {marker()}"
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(text, question)).status_code == 200
        generated: Final = body_of(_calls(peer, "POST", _GENERATE)[0])
        assert generated["cachedContent"] == f"projects/{PROJECT}/locations/{LOCATION}/cachedContents/dec1-cache-1"
        sent: Final = json.dumps(generated["contents"])
        assert question in sent and text not in sent


def test_every_cache_call_carries_the_minted_bearer(gateway: Gateway) -> None:
    """The list, the create and the generate call all carry `Authorization: Bearer <token>` minted from
    the deployment's service account, and none carries `x-goog-api-key` (the AI Studio scheme)."""
    google: Final = _Google()
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}")).status_code == 200
        calls: Final = peer.calls()
        assert len(calls) == 3
        assert all(is_token_from(peer.plan, call) for call in calls), [dict(call.headers) for call in calls]
        assert all("x-goog-api-key" not in call.headers for call in calls)


def test_a_v1_api_base_grafts_the_collection_path(gateway: Gateway) -> None:
    """An `api_base` of `<peer>/v1` has the default collection path grafted under its version, so the list
    and create go to `/v1/projects/dec1-vertex-project/locations/us-central1/cachedContents` and not
    `/v1/v1/...` or `/v1:cachedContents` (`collection_endpoint=True` in `_check_custom_proxy`)."""
    google: Final = _Google()
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI, base_suffix="/v1")
        assert _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}")).status_code == 200
        assert [path for _method, path in peer.shape()][:2] == [_COLLECTION, _COLLECTION]


def test_an_hour_ttl_is_sent_to_vertex_in_seconds(gateway: Gateway) -> None:
    """`cache_control.ttl: "1h"` becomes `ttl: "3600s"` in the create body (`_normalize_ttl_to_seconds`)."""
    google: Final = _Google()
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}", ttl="1h")).status_code == 200
        assert google.created[0]["ttl"] == "3600s"


def test_a_cached_system_prompt_moves_into_the_cache(gateway: Gateway) -> None:
    """A cached system message is sent as the create body's `system_instruction` and the generate call
    then has no `system_instruction` of its own: the system prompt lives in the cache."""
    google: Final = _Google()
    system: Final = f"You are a policy assistant {marker()}"
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        messages: Final = _cached_messages(_cached_text(), f"q {marker()}", system=system)
        assert _chat(gateway, model, messages).status_code == 200
        assert system in json.dumps(google.created[0]["system_instruction"])
        generated: Final = body_of(_calls(peer, "POST", _GENERATE)[0])
        assert "system_instruction" not in generated and "systemInstruction" not in generated


def test_tools_move_into_the_vertex_cache_and_out_of_generate(gateway: Gateway) -> None:
    """With a cached block and an OpenAI tool, the create body carries `tools` (Vertex refuses a
    `generateContent` that sets both `cachedContent` and `tools`) and the generate body has no `tools`."""
    google: Final = _Google()
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {
            "name": "lookup_clause",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
        },
    }
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}"), tools=[tool]).status_code == 200
        assert "lookup_clause" in json.dumps(google.created[0]["tools"])
        assert "tools" not in body_of(_calls(peer, "POST", _GENERATE)[0])


def test_the_list_follows_next_page_token_before_creating(gateway: Gateway) -> None:
    """When the first list page answers `{"cachedContents": [], "nextPageToken": "dec1-page-2"}` the proxy
    asks for `cachedContents?pageToken=dec1-page-2` before it gives up and creates the cache, so the calls
    are GET, GET (with that query), POST create, POST generate."""
    google: Final = _Google(pages=({"cachedContents": [], "nextPageToken": "dec1-page-2"}, {"cachedContents": []}))
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}")).status_code == 200
        calls: Final = peer.calls()
        assert [(call.method, path_of(call)) for call in calls] == [
            ("GET", _COLLECTION),
            ("GET", _COLLECTION),
            ("POST", _COLLECTION),
            ("POST", _GENERATE),
        ]
        assert (query_of(calls[0]), query_of(calls[1])) == ("", "pageToken=dec1-page-2")


def test_a_listed_cache_is_reused_on_the_next_request_without_create(gateway: Gateway) -> None:
    """A cache created without `expireTime` is not remembered in process, so the next request with the
    same cached block lists again, finds the entry by `displayName` (now listed with a future expiry) and
    sends `generateContent` with that `name` without a second create."""
    google: Final = _Google(create_expire=None)
    text: Final = _cached_text()
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(text, f"first {marker()}")).status_code == 200
        google.listed[0] = {**object_value(google.listed[0]), "expireTime": "2999-01-01T00:00:00Z"}
        assert _chat(gateway, model, _cached_messages(text, f"second {marker()}")).status_code == 200
        assert len(google.created) == 1
        generates: Final = _calls(peer, "POST", _GENERATE)
        assert [body_of(call)["cachedContent"] for call in generates] == [google.listed[0]["name"]] * 2


def test_a_forbidden_list_still_creates_the_cache(gateway: Gateway) -> None:
    """A 403 on the Vertex list (a service account allowed to create but not list caches) is read as
    "nothing cached" (`check_cache` returns None on 403), so the cache is still created and used."""
    google: Final = _Google(list_error=google_error(403, "PERMISSION_DENIED", "list denied"))
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        assert _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}")).status_code == 200
        assert peer.shape() == [("GET", _COLLECTION), ("POST", _COLLECTION), ("POST", _GENERATE)]


def test_a_missing_vertex_collection_fails_the_call_without_generating(gateway: Gateway) -> None:
    """Unlike Gemini (where a 404 on `cachedContents` skips caching), a 404 on the Vertex list is raised as
    `VertexAIError(404)`, so the caller gets an error status and `generateContent` is never called. A port
    that copies Gemini's skip would change this. UNVERIFIED: the caller status (404 expected through
    exception mapping); look at `vertex_ai_context_caching.py:280`."""
    google: Final = _Google(list_error=google_error(404, "NOT_FOUND", "cachedContents not served"))
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        response: Final = _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}"))
        assert response.status_code == 404, response.text
        assert _calls(peer, "POST", _GENERATE) == []


def test_a_refused_create_fails_the_call_with_its_status(gateway: Gateway) -> None:
    """When the create answers 400 `INVALID_ARGUMENT`, the caller gets 400 with Google's message and no
    `generateContent` is sent (`VertexAIError(status_code=error_code)` in `lookup_or_create`)."""
    google: Final = _Google(create_error=google_error(400, "INVALID_ARGUMENT", "cached content too small"))
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        response: Final = _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}"))
        assert response.status_code == 400, response.text
        assert "cached content too small" in response.text
        assert _calls(peer, "POST", _GENERATE) == []


def test_a_caller_cached_content_skips_list_and_create(gateway: Gateway) -> None:
    """A request with `cached_content: "projects/.../cachedContents/caller-cache"` makes no list and no
    create: the only call is `generateContent` with `cachedContent` set to that name
    (`async_check_and_create_cache` returns early when `cached_content` is given)."""
    google: Final = _Google()
    name: Final = f"projects/{PROJECT}/locations/{LOCATION}/cachedContents/caller-cache"
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        messages: Final = [{"role": "user", "content": f"q {marker()}"}]
        assert _chat(gateway, model, messages, cached_content=name).status_code == 200
        assert peer.shape() == [("POST", _GENERATE)]
        assert body_of(peer.calls()[0])["cachedContent"] == name


def test_vertex_cached_tokens_are_reported_on_the_chat_usage(gateway: Gateway) -> None:
    """`usageMetadata.cachedContentTokenCount: 5120` on the generate answer is returned as
    `usage.prompt_tokens_details.cached_tokens: 5120`."""
    google: Final = _Google()
    with open_peer(google) as peer, gateway.scenario() as scenario:
        model: Final = register_open_vertex(scenario, peer, GEMINI)
        response: Final = _chat(gateway, model, _cached_messages(_cached_text(), f"q {marker()}"))
        assert response.status_code == 200, response.text
        details: Final = object_value(object_value(json_of(response)["usage"])["prompt_tokens_details"])
        assert details["cached_tokens"] == _CACHED_TOKENS
        assert list_value(json_of(response)["choices"])
