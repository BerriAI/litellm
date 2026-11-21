"""`gemini/` (Google AI Studio) deployments: where a request lands, how the key travels, how errors come back.

AI Studio is the API-key flavour of Gemini. The proxy builds the URL itself
(`litellm/llms/vertex_ai/vertex_llm_base.py::_check_custom_proxy`): with a custom `api_base` the
request goes to `{api_base}/models/{model}:generateContent`, or `:streamGenerateContent` with the
query `alt=sse` when streaming, and the deployment key is sent ONLY as the `x-goog-api-key` header.
It is never put in a `?key=` query parameter (`get_gemini_url` says so explicitly: a key in the URL
leaks into error tracebacks), never sent as `Authorization`, and there is no OAuth token exchange as on
Vertex. The `v1beta` segment is NOT added by the proxy when an `api_base` is set, so the version is
whatever the operator puts in the base; Gemini 3 models, which default to `v1alpha` on the Google
host, follow the base too.

Errors come back through `_map_vertex_exception` (shared with Vertex): Google's
`{"error": {"code", "message", "status"}}` keeps its status for 400, 404 and 429; the AI Studio
specific `API key not valid.` text becomes a 401; and the string `The model is overloaded.` becomes a
500 regardless of the 503 Google sends. A Rust port has to keep the header-only key, the exact path
shapes and these status rules; every test uses a fresh prompt marker because the shared proxy caches
responses.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_gemini_support import (
    FLASH,
    GEMINI_3_FLASH,
    GEMINI_KEY,
    gemini,
    gemini_echo,
    gemini_reply,
    gemini_sse,
    generate_path,
    google_error,
    headers_of,
    json_of,
    marker,
    message_of,
    path_of,
    query_of,
    streamed_text,
)

_OVERLOADED: Final = "The model is overloaded. Please try again later."
_BAD_KEY: Final = "API key not valid. Please pass a valid API key."


def _error_message(response_body: dict[str, object]) -> str:
    error: Final = response_body.get("error")
    assert isinstance(error, dict), response_body
    return str(error.get("message"))


# ---------------------------------------------------------------- URL shape


def test_a_chat_posts_to_v1beta_models_generate_content_under_the_api_base(gateway: Gateway) -> None:
    """A `gemini/gemini-2.5-flash` deployment with `api_base: http://<peer>/v1beta` receives exactly one
    `POST /v1beta/models/gemini-2.5-flash:generateContent`. The model id is put after `models/` with the
    `gemini/` prefix stripped, and nothing else (no project, no location) is in the path."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert call.method == "POST"
        assert path_of(call) == generate_path(FLASH)


def test_a_bare_host_api_base_gets_no_version_segment(gateway: Gateway) -> None:
    """With `api_base: http://<peer>` (no path) the proxy joins `models/gemini-2.5-flash:generateContent`
    straight onto the host, giving `/models/gemini-2.5-flash:generateContent` without `v1beta`. The
    version only comes from the default Google URL, which a custom base replaces entirely
    (`join_api_base(api_base, f"models/{model}:{endpoint}")`)."""
    with gemini(gateway, gemini_echo, base_suffix="") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == f"/models/{FLASH}:generateContent"


def test_a_trailing_slash_on_the_api_base_is_not_doubled(gateway: Gateway) -> None:
    """`api_base: http://<peer>/v1beta/` reaches `/v1beta/models/gemini-2.5-flash:generateContent`;
    `join_api_base` strips trailing slashes so the path never contains `//models`."""
    with gemini(gateway, gemini_echo, base_suffix="/v1beta/") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == generate_path(FLASH)


def test_a_gemini_3_model_follows_the_api_base_version_not_v1alpha(gateway: Gateway) -> None:
    """Gemini 3 models default to `v1alpha` on the Google host (`get_gemini_url`), but with
    `api_base: http://<peer>/v1beta` the call goes to `/v1beta/models/gemini-3-flash-preview:generateContent`:
    the custom base path replaces the default URL, version included."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == generate_path(GEMINI_3_FLASH)


def test_a_non_streaming_chat_has_no_query_string(gateway: Gateway) -> None:
    """A non-streaming call carries no query at all: no `alt`, and in particular no `key=` parameter."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        assert query_of(rig.peer.one_call()) == ""


def test_a_streaming_chat_posts_to_stream_generate_content_with_alt_sse(gateway: Gateway) -> None:
    """`stream: true` switches the method to `:streamGenerateContent` and adds exactly the query
    `alt=sse`, the only query the proxy ever puts on an AI Studio URL."""
    with gemini(gateway, gemini_echo) as rig:
        response, _text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert path_of(call) == generate_path(FLASH, method="streamGenerateContent")
        assert query_of(call) == "alt=sse"


def test_a_streamed_answer_is_relayed_as_chat_chunks(gateway: Gateway) -> None:
    """Two SSE frames with "hello " and "stream" come back as `chat.completion.chunk` deltas joining to
    "hello stream", ending with `data: [DONE]`; the AI Studio stream uses the same SSE framing as Vertex."""
    with gemini(gateway, gemini_echo) as rig:
        response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 200
        assert streamed_text(text) == "hello stream"
        assert text.rstrip().endswith("data: [DONE]")


# ---------------------------------------------------------------- the key


def test_the_deployment_key_travels_only_as_x_goog_api_key(gateway: Gateway) -> None:
    """The model call carries `x-goog-api-key: dec1-ai-studio-key` and no `Authorization` header, so the
    key is never mistaken for an OAuth bearer and never reaches a log line as a URL."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        headers: Final = headers_of(rig.peer.one_call())
        assert headers.get("x-goog-api-key") == GEMINI_KEY
        assert "authorization" not in headers


def test_the_deployment_key_is_never_in_the_url(gateway: Gateway) -> None:
    """Neither the streaming nor the non-streaming request target contains `key=` or the key itself.
    Older releases sent `?key=`; `get_gemini_url` moved it to the header to keep it out of tracebacks."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        calls: Final = rig.peer.generate_calls()
        assert len(calls) == 2
        assert all("key=" not in call.target and GEMINI_KEY not in call.target for call in calls)


def test_the_callers_virtual_key_never_reaches_google(gateway: Gateway) -> None:
    """The proxy key the caller authenticated with appears in no header, query or body byte of the
    upstream call; only the deployment's own AI Studio key does."""
    with gemini(gateway, gemini_echo) as rig:
        caller_key: Final = rig.scenario.key()
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": rig.model, "messages": [{"role": "user", "content": marker()}]},
            key=caller_key,
        )
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert caller_key not in call.target
        assert caller_key not in call.headers.values()
        assert caller_key.encode() not in call.body


def test_a_caller_api_key_override_is_sent_as_the_header(gateway: Gateway) -> None:
    """A deployment carrying `configurable_clientside_auth_params: ["api_key"]` lets the caller send
    `api_key: caller-studio-key` in the body; the upstream header becomes `x-goog-api-key: caller-studio-key`
    and the deployment key is not sent. UNVERIFIED: assumes the clientside auth allowlist applies to the
    gemini route the same way as to openai (`litellm/proxy/auth/auth_utils.py`)."""
    with gemini(gateway, gemini_echo, configurable_clientside_auth_params=["api_key"]) as rig:
        assert rig.chat(api_key="caller-studio-key").status_code == 200
        assert headers_of(rig.peer.one_call()).get("x-goog-api-key") == "caller-studio-key"


def test_the_body_never_carries_the_model_or_the_key(gateway: Gateway) -> None:
    """The generateContent body has `contents` and no `model`, `api_key`, `key` or `messages` key: AI
    Studio takes the model from the path and rejects unknown top-level fields."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(user="dec1-user", metadata={"note": "x"}).status_code == 200
        body: Final = rig.peer.one_body()
        assert "contents" in body
        assert not {"model", "api_key", "key", "messages", "user", "metadata"} & set(body)


def test_labels_are_not_sent_to_ai_studio(gateway: Gateway) -> None:
    """A caller's `labels: {"team": "search"}` reaches Vertex as top-level `labels`, but AI Studio has no
    such field, so the `gemini/` body carries no `labels` key (`transformation.py`: labels only when
    `custom_llm_provider != gemini`)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(labels={"team": "search"}).status_code == 200
        assert "labels" not in rig.peer.one_body()


@pytest.mark.parametrize(("tier", "expected"), [("priority", "priority"), ("flex", "flex"), ("default", "standard")])
def test_service_tier_is_sent_as_service_tier(gateway: Gateway, tier: str, expected: str) -> None:
    """OpenAI's `service_tier` is sent as top-level `serviceTier`, lower-cased, with OpenAI's `default`
    renamed to Google's `standard` (`transformation.py` service_tier branch)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(service_tier=tier).status_code == 200
        assert rig.peer.one_body()["serviceTier"] == expected


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "google_status", "message"),
    [
        (400, "INVALID_ARGUMENT", 'Invalid JSON payload received. Unknown name "foo"'),
        (404, "NOT_FOUND", "models/gemini-9 is not found for API version v1beta"),
        (429, "RESOURCE_EXHAUSTED", "You exceeded your current quota, please check your plan"),
    ],
)
def test_a_google_error_keeps_its_status_and_message(
    gateway: Gateway, status: int, google_status: str, message: str
) -> None:
    """Google's error envelope with status 400, 404 or 429 leaves the proxy with the same status and the
    Google message inside `error.message`, prefixed with the provider name `Gemini`."""
    with gemini(gateway, lambda _r: google_error(status, google_status, message)) as rig:
        response: Final = rig.chat()
        assert response.status_code == status, response.text
        text: Final = _error_message(json_of(response))
        assert message.split(" ")[0] in text
        assert "Gemini" in text


def test_an_invalid_api_key_message_is_answered_as_401(gateway: Gateway) -> None:
    """AI Studio answers a wrong key with HTTP 400 `INVALID_ARGUMENT` and the text `API key not valid.`;
    `_map_vertex_exception` matches that text before the status branches and raises
    `AuthenticationError`, so the caller sees 401, not 400."""
    with gemini(gateway, lambda _r: google_error(400, "INVALID_ARGUMENT", _BAD_KEY)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 401, response.text
        assert "API key not valid" in _error_message(json_of(response))


def test_an_overloaded_503_is_answered_as_500(gateway: Gateway) -> None:
    """Google's usual 503 `UNAVAILABLE` text `The model is overloaded. Please try again later.` matches
    the `The model is overloaded.` string check, which comes before the status branches and raises
    `InternalServerError`, so the caller sees 500 rather than 503. Pinned as today's behaviour."""
    with gemini(gateway, lambda _r: google_error(503, "UNAVAILABLE", _OVERLOADED)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 500, response.text
        assert "overloaded" in _error_message(json_of(response))


def test_a_503_with_another_message_stays_503(gateway: Gateway) -> None:
    """A 503 whose message is not the overload text falls through to the status branch and is answered
    as 503 (`ServiceUnavailableError`)."""
    with gemini(gateway, lambda _r: google_error(503, "UNAVAILABLE", "Backend is warming up")) as rig:
        assert rig.chat().status_code == 503


def test_a_429_wrapped_in_a_500_envelope_is_answered_as_429(gateway: Gateway) -> None:
    """A gateway in front of Google that answers HTTP 500 with `{"error": {"code": 429, ...}}` is mapped
    to `RateLimitError` (the 5xx-with-body-429 branch), so the caller can back off."""
    body: Final = json.dumps({"error": {"code": 429, "message": "upstream throttled", "status": "RESOURCE_EXHAUSTED"}})
    with gemini(gateway, lambda _r: Reply(status=500, body=body.encode())) as rig:
        assert rig.chat().status_code == 429


def test_a_streaming_request_refused_with_429_answers_429_before_any_chunk(gateway: Gateway) -> None:
    """When `:streamGenerateContent` is refused with 429 the caller's stream request is answered with
    status 429 and an error body, not a 200 stream that later carries an error frame."""
    with gemini(gateway, lambda _r: google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for metric")) as rig:
        response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert response.status_code == 429, text


def test_the_deployment_keeps_serving_after_an_upstream_error(gateway: Gateway) -> None:
    """A first call answered 500 by Google and a second answered normally: the second caller gets 200 and
    "pong", since cooldowns are disabled on the shared proxy and a failure must not poison the deployment."""
    calls: Final[list[int]] = []

    def respond(request: Request) -> Reply:
        calls.append(1)
        if len(calls) == 1:
            return google_error(500, "INTERNAL", "An internal error has occurred")
        return gemini_echo(request)

    with gemini(gateway, respond, num_retries=0) as rig:
        assert rig.chat().status_code >= 500
        second: Final = rig.chat()
        assert second.status_code == 200, second.text
        assert message_of(second)["content"] == "pong"


def test_a_prompt_blocked_by_safety_answers_200_with_content_filter(gateway: Gateway) -> None:
    """A `promptFeedback.blockReason: SAFETY` answer with no candidates is a 200 with one choice whose
    `finish_reason` is `content_filter` and whose content is empty, not an exception. UNVERIFIED: the
    exact empty content value (None vs "") was taken from the Vertex suite's equivalent test."""
    blocked: Final = {
        "promptFeedback": {"blockReason": "SAFETY", "safetyRatings": []},
        "usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9},
        "modelVersion": FLASH,
    }
    with gemini(gateway, lambda _r: Reply(body=json.dumps(blocked).encode())) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        choices: Final = list_value(json_of(response)["choices"])
        assert object_value(choices[0])["finish_reason"] == "content_filter"
        assert not object_value(object_value(choices[0])["message"]).get("content")


def test_a_recitation_finish_is_reported_as_content_filter(gateway: Gateway) -> None:
    """`finishReason: RECITATION` maps to the OpenAI `content_filter` finish reason on AI Studio as on Vertex."""
    with gemini(gateway, lambda _r: gemini_reply(finish="RECITATION")) as rig:
        response: Final = rig.chat()
        assert string_value(object_value(list_value(json_of(response)["choices"])[0])["finish_reason"]) == (
            "content_filter"
        )


def test_a_stream_ended_with_max_tokens_finishes_with_length(gateway: Gateway) -> None:
    """A stream whose last frame says `MAX_TOKENS` ends with a chunk whose `finish_reason` is `length`."""
    with gemini(gateway, lambda _r: gemini_sse(finish="MAX_TOKENS")) as rig:
        _response, text = rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert '"finish_reason": "length"' in text or '"finish_reason":"length"' in text
