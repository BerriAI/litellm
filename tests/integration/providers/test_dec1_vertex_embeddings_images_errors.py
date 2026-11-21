"""Vertex embeddings, Vertex image generation and Vertex error mapping through the proxy.

Embeddings on `text-embedding-005` go to `:predict` with `instances: [{"content": <text>}, ...]`; the
answer's per-prediction `statistics.token_count` values are summed into `usage.prompt_tokens` and
`metadata.billableCharacterCount` becomes `usage.prompt_tokens_details.character_count`
(`vertex_embeddings/transformation.py`).

Image generation picks a config by model route (`image_generation/__init__.py`). Imagen models send one
`:predict` call with `parameters.sampleCount = n`. Gemini image models return one image per call, so
since #52103 (2026-11-16) `n` is served by `n` identical concurrent `generateContent` calls whose
images and usage are merged (`litellm/images/fan_out.py`), capped at `MAX_IMAGE_FAN_OUT_REQUESTS`
(10). With an `api_base` both image configs post to the `api_base` verbatim and skip credentials.
Spend for image generation is not asserted here, because custom image prices are currently lost.

Errors: Vertex answers `{"error": {"code", "message", "status"}}` and `_map_vertex_exception`
(`litellm/litellm_core_utils/exception_mapping_utils.py`) picks the exception, whose status is what
the caller sees.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_vertex_support import (
    EMBEDDING,
    GEMINI_IMAGE,
    IMAGEN,
    PNG_B64,
    SA_PROJECT,
    at,
    body_of,
    counter,
    embedding_peer,
    gemini_echo,
    gemini_image_reply,
    google_error,
    imagen_reply,
    json_of,
    marker,
    models_path,
    path_of,
    vertex,
)

_PREDICT: Final = models_path(EMBEDDING) + ":predict"
_IMAGEN_PATH: Final = models_path(IMAGEN) + ":predict"
_GEMINI_IMAGE_PATH: Final = models_path(GEMINI_IMAGE) + ":generateContent"


# ---------------------------------------------------------------- embeddings


def test_an_embedding_request_posts_instances_to_predict(gateway: Gateway) -> None:
    """Two inputs reach `POST .../models/text-embedding-005:predict` as `instances: [{"content": a},
    {"content": b}]` in order, with a minted bearer token."""
    texts: Final = [f"alpha {marker()}", f"bravo {marker()}"]
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": texts})
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert path_of(call) == _PREDICT
        assert body_of(call)["instances"] == [{"content": text} for text in texts]
        assert call.headers["authorization"].removeprefix("Bearer ") in rig.peer.plan.issued


def test_embedding_vectors_come_back_in_input_order_with_summed_token_usage(gateway: Gateway) -> None:
    """The two predictions are returned as `data[0]` and `data[1]` with their `values`, and
    `usage.prompt_tokens = 3 + 4 = 7` (the sum of `statistics.token_count`) with `total_tokens = 7`."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        body: Final = json_of(rig.post("/v1/embeddings", {"input": [f"a {marker()}", f"b {marker()}"]}))
        data: Final = [object_value(item) for item in list_value(body["data"])]
        assert [(item["index"], item["embedding"]) for item in data] == [(0, [0.5, 0.0]), (1, [0.5, 1.0])]
        usage: Final = object_value(body["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (7, 7)


def test_billable_characters_are_reported_in_prompt_tokens_details(gateway: Gateway) -> None:
    """Vertex's `metadata.billableCharacterCount` (here the input length) is returned as
    `usage.prompt_tokens_details.character_count`, which per-character embedding pricing bills on."""
    text: Final = f"count my characters {marker()}"
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        usage: Final = object_value(json_of(rig.post("/v1/embeddings", {"input": [text]}))["usage"])
        assert at(usage, "prompt_tokens_details", "character_count") == len(text)


def test_embedding_dimensions_become_output_dimensionality(gateway: Gateway) -> None:
    """`dimensions: 128` is sent as `parameters.outputDimensionality: 128`."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"d {marker()}"], "dimensions": 128}).status_code == 200
        assert at(rig.peer.one_body(), "parameters", "outputDimensionality") == 128


def test_embedding_task_type_and_title_are_put_on_each_instance(gateway: Gateway) -> None:
    """Vertex-specific `task_type: "RETRIEVAL_DOCUMENT"` and `title` are copied onto every instance
    (`create_embedding_input`). UNVERIFIED: assumes the proxy passes both through as provider params."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        response: Final = rig.post(
            "/v1/embeddings", {"input": [f"t {marker()}"], "task_type": "RETRIEVAL_DOCUMENT", "title": "handbook"}
        )
        assert response.status_code == 200, response.text
        instance: Final = object_value(list_value(rig.peer.one_body()["instances"])[0])
        assert (instance["task_type"], instance["title"]) == ("RETRIEVAL_DOCUMENT", "handbook")


def test_embedding_project_comes_from_the_service_account_when_unset(gateway: Gateway) -> None:
    """Without `vertex_project` the embedding path names the service account's `project_id`."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING, vertex_project=None) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"p {marker()}"]}).status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(EMBEDDING, project=SA_PROJECT) + ":predict"


def test_embedding_caller_labels_reach_the_predict_body(gateway: Gateway) -> None:
    """A caller's `labels` on an embedding request is sent as the top-level `labels` of the predict body
    and not inside `parameters`."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"l {marker()}"], "labels": {"app": "search"}}).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["labels"] == {"app": "search"}
        assert "labels" not in object_value(body.get("parameters") or {})


# ---------------------------------------------------------------- image generation


def test_imagen_n_is_one_predict_call_with_sample_count(gateway: Gateway) -> None:
    """On an Imagen model `n: 3` is one `:predict` call with `instances: [{"prompt": <prompt>}]` and
    `parameters.sampleCount: 3`; Imagen returns all images from one call so nothing fans out."""
    prompt: Final = f"a lighthouse {marker()}"
    with vertex(gateway, lambda _r: imagen_reply(3), backend=IMAGEN, base_suffix=_IMAGEN_PATH) as rig:
        response: Final = rig.post("/v1/images/generations", {"prompt": prompt, "n": 3})
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert body["instances"] == [{"prompt": prompt}]
        assert object_value(body["parameters"])["sampleCount"] == 3


def test_imagen_predictions_come_back_as_b64_json_images(gateway: Gateway) -> None:
    """Each `bytesBase64Encoded` prediction is returned as one `data[]` item with that `b64_json`."""
    with vertex(gateway, lambda _r: imagen_reply(2), backend=IMAGEN, base_suffix=_IMAGEN_PATH) as rig:
        body: Final = json_of(rig.post("/v1/images/generations", {"prompt": f"x {marker()}", "n": 2}))
        assert [object_value(item)["b64_json"] for item in list_value(body["data"])] == [PNG_B64, PNG_B64]


def test_an_image_api_base_is_posted_to_verbatim_without_a_token(gateway: Gateway) -> None:
    """With an `api_base`, Vertex image generation posts to exactly that URL (no `:predict` appended)
    and `validate_environment` returns before minting, so no `Authorization` header is sent and the
    token endpoint is never called, even though the deployment has `vertex_credentials`. Pinned as
    current behaviour; a gateway that needs the Google token in front of Vertex cannot get one."""
    with vertex(gateway, lambda _r: imagen_reply(1), backend=IMAGEN, base_suffix="/custom/imagen") as rig:
        assert rig.post("/v1/images/generations", {"prompt": f"x {marker()}"}).status_code == 200
        call: Final = rig.peer.one_call()
        assert path_of(call) == "/custom/imagen"
        assert "authorization" not in call.headers
        assert rig.peer.token_calls() == ()


def test_gemini_image_n_fans_out_to_n_single_image_calls(gateway: Gateway) -> None:
    """Since #52103, `n: 3` on a Gemini image model sends three identical `generateContent` calls, each
    with `generationConfig: {"responseModalities": ["IMAGE"]}` and no `candidateCount`, because Gemini
    image models refuse a candidate count above one."""
    prompt: Final = f"three cats {marker()}"
    with vertex(gateway, lambda _r: gemini_image_reply(), backend=GEMINI_IMAGE, base_suffix=_GEMINI_IMAGE_PATH) as rig:
        response: Final = rig.post("/v1/images/generations", {"prompt": prompt, "n": 3})
        assert response.status_code == 200, response.text
        calls: Final = rig.peer.model_calls()
        assert len(calls) == 3
        for call in calls:
            assert body_of(call) == {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"]},
            }


def test_gemini_image_fan_out_merges_the_images_and_sums_usage(gateway: Gateway) -> None:
    """The three single-image answers are merged into one response with three `data` items and usage
    summed across calls: `total_tokens = 3 * (10 + 1290) = 3900`. UNVERIFIED: the usage field names come
    from `transform_gemini_image_usage`."""
    with vertex(gateway, lambda _r: gemini_image_reply(), backend=GEMINI_IMAGE, base_suffix=_GEMINI_IMAGE_PATH) as rig:
        body: Final = json_of(rig.post("/v1/images/generations", {"prompt": f"x {marker()}", "n": 3}))
        assert [object_value(item)["b64_json"] for item in list_value(body["data"])] == [PNG_B64] * 3
        assert object_value(body["usage"])["total_tokens"] == 3900


def test_gemini_image_n_above_ten_is_refused_without_a_call(gateway: Gateway) -> None:
    """`n: 11` is above `MAX_IMAGE_FAN_OUT_REQUESTS` (10), so the caller gets HTTP 400 with
    `n=11 is above the maximum of 10 images per request.` and the peer sees no model call."""
    with vertex(gateway, lambda _r: gemini_image_reply(), backend=GEMINI_IMAGE, base_suffix=_GEMINI_IMAGE_PATH) as rig:
        response: Final = rig.post("/v1/images/generations", {"prompt": f"x {marker()}", "n": 11})
        assert response.status_code == 400, response.text
        assert "n=11 is above the maximum of 10 images per request." in response.text
        assert rig.peer.model_calls() == ()


def test_gemini_image_without_n_sends_one_call(gateway: Gateway) -> None:
    """A Gemini image request without `n` sends exactly one `generateContent` call and returns one image."""
    with vertex(gateway, lambda _r: gemini_image_reply(), backend=GEMINI_IMAGE, base_suffix=_GEMINI_IMAGE_PATH) as rig:
        body: Final = json_of(rig.post("/v1/images/generations", {"prompt": f"x {marker()}"}))
        assert len(rig.peer.model_calls()) == 1
        assert len(list_value(body["data"])) == 1


def test_one_failed_fan_out_call_fails_the_whole_request(gateway: Gateway) -> None:
    """When the second of three fan-out calls answers 429, every call still runs to completion, the
    caller gets an error rather than a partial image list, and the peer saw all three calls
    (`run_concurrently` / `split_outcomes`). UNVERIFIED: the caller status is expected to be 429."""
    take: Final = counter()

    def respond(_request: Request) -> Reply:
        if take() == 2:
            return google_error(429, "RESOURCE_EXHAUSTED", "Resource exhausted. Please try again later.")
        return gemini_image_reply()

    with vertex(gateway, respond, backend=GEMINI_IMAGE, base_suffix=_GEMINI_IMAGE_PATH) as rig:
        response: Final = rig.post("/v1/images/generations", {"prompt": f"x {marker()}", "n": 3})
        assert response.status_code == 429, response.text
        assert len(rig.peer.model_calls()) >= 3


# ---------------------------------------------------------------- error mapping (Gemini chat)


@pytest.mark.parametrize(
    ("status", "google_status", "message", "expected"),
    (
        (429, "RESOURCE_EXHAUSTED", "Resource exhausted. Please try again later.", 429),
        (400, "INVALID_ARGUMENT", "Request contains an invalid argument.", 400),
        (404, "NOT_FOUND", "Publisher Model was not found or your project does not have access to it.", 404),
        (503, "UNAVAILABLE", "The service is currently unavailable.", 503),
        (500, "INTERNAL", "Internal error encountered.", 500),
    ),
    ids=("resource_exhausted", "invalid_argument", "not_found", "unavailable", "internal"),
)
def test_a_google_error_keeps_its_status_and_message(
    gateway: Gateway, status: int, google_status: str, message: str, expected: int
) -> None:
    """A Gemini chat call that Vertex answers with Google's error envelope leaves the proxy with the
    mapped status (429 RateLimitError, 400 BadRequestError, 404 NotFoundError, 503 ServiceUnavailable,
    500 InternalServerError) and Google's message text in the body."""
    with vertex(gateway, lambda _r: google_error(status, google_status, message)) as rig:
        response: Final = rig.chat()
        assert response.status_code == expected, response.text
        assert message in response.text


def test_a_google_quota_message_on_a_400_is_still_a_rate_limit(gateway: Gateway) -> None:
    """`_map_vertex_exception` matches `Quota exceeded for` in the text before looking at the status, so a
    400 whose message is a quota message reaches the caller as 429."""
    message: Final = "Quota exceeded for aiplatform.googleapis.com/generate_content_requests_per_minute."
    with vertex(gateway, lambda _r: google_error(400, "FAILED_PRECONDITION", message)) as rig:
        assert rig.chat().status_code == 429


def test_a_google_403_permission_denied_reaches_the_caller_as_403(gateway: Gateway) -> None:
    """Google's 403 `PERMISSION_DENIED` envelope reaches the caller as HTTP 403 with Google's message.

    Round 2 (run 44176682961) showed the proxy now answers 403 with `"type": "permission_error"` for this
    body, so the earlier BUG skip and the companion test that pinned HTTP 400 were dropped. The message text
    still starts with `litellm.BadRequestError: Vertex_aiException BadRequestError`, so only the status and
    Google's message are asserted.
    """
    message: Final = "Permission 'aiplatform.endpoints.predict' denied on resource."
    with vertex(gateway, lambda _r: google_error(403, "PERMISSION_DENIED", message)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 403, response.text
        assert message in response.text


def test_a_streaming_request_refused_with_429_answers_429(gateway: Gateway) -> None:
    """When `:streamGenerateContent` answers 429 before any frame, the caller's streaming request gets
    an HTTP 429 response rather than a 200 stream with an error inside. UNVERIFIED: assumes the proxy
    opens the upstream stream before sending its own status line."""
    with vertex(
        gateway, lambda _r: google_error(429, "RESOURCE_EXHAUSTED", "Resource exhausted. Please try again later.")
    ) as rig:
        response, _text = rig.stream(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]}
        )
        assert response.status_code == 429


def test_the_deployment_keeps_serving_after_an_upstream_error(gateway: Gateway) -> None:
    """After a 503 from Vertex the next request to the same deployment is answered normally (the shared
    proxy disables cooldowns), so one Google blip does not take the model group down."""
    take: Final = counter()

    def respond(request: Request) -> Reply:
        if take() == 1:
            return google_error(503, "UNAVAILABLE", "The service is currently unavailable.")
        return gemini_echo(request)

    with vertex(gateway, respond, num_retries=0) as rig:
        assert rig.chat().status_code == 503
        assert rig.chat().status_code == 200


def test_an_embedding_429_is_answered_as_429(gateway: Gateway) -> None:
    """A `RESOURCE_EXHAUSTED` answer to an embedding `:predict` call leaves the proxy as HTTP 429."""
    reply: Final = google_error(429, "RESOURCE_EXHAUSTED", "Resource exhausted. Please try again later.")
    with vertex(gateway, lambda _r: reply, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"e {marker()}"]}).status_code == 429


def test_an_invalid_location_is_refused_before_any_request(gateway: Gateway) -> None:
    """`vertex_location: "evil.example#"` fails `validate_vertex_location` (lowercase letters, digits and
    hyphens only, or `global`), so the call fails before any model call is made;
    otherwise a client-controlled location could steer the admin's token to another host.
    UNVERIFIED: the caller status (expected 500, a ValueError inside the provider)."""
    with vertex(gateway, gemini_echo, vertex_location="evil.example#") as rig:
        response: Final = rig.chat()
        assert response.status_code >= 400, response.text
        assert "Invalid vertex_location format" in response.text
        assert rig.peer.model_calls() == ()
