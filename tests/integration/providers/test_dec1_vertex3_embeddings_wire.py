"""Vertex AI embeddings beyond the basics: text embedding task types, truncation, Gemini Embedding 2 config,
and multimodal (image and video) embeddings.

Three Vertex embedding wires sit behind `/v1/embeddings`:

- Text embedding models (`text-embedding-005`, `text-multilingual-embedding-002`) post `:predict` with one
  `instances[]` entry per input. `VertexAITextEmbeddingConfig` (`litellm/llms/vertex_ai/vertex_embeddings/
  transformation.py`) puts `task_type` and `title` on EVERY instance, maps OpenAI's `input_type` to `task_type`,
  and passes the remaining provider params (such as `auto_truncate`) as `parameters`.
- Models whose cost map entry sets `uses_embed_content` (`gemini-embedding-2`) post `:embedContent` with one
  `content.parts[]` and Gemini's camel case config (`taskType`, `outputDimensionality`); since #51686 (landed
  2026-11-13) `auto_truncate`, `document_ocr` and `audio_track_extraction` go under `embedContentConfig`
  instead of being dropped (`gemini_embeddings/batch_embed_content_transformation.py:_embed_content_config`).
- `multimodalembedding@001` posts `:predict` with `text`, `image` (`gcsUri` or `bytesBase64Encoded`) and
  `video` (`gcsUri`) instances; a text followed by a media element is merged into one instance, `dimensions`
  becomes `parameters.dimension`, and usage is reported as characters, images and video seconds with zero
  tokens (`multimodal_embeddings/transformation.py`).

Basic text embedding order, token sums and billable characters are pinned in
`test_dec1_vertex_embeddings_images_errors.py`; file blocks on `embedContent` in
`test_vertex_embedding_file_block_wire.py`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_vertex3_support import GEMINI_EMBEDDING_2, MULTIMODAL_EMBEDDING, json_of
from integration.providers.dec1_vertex_support import (
    EMBEDDING,
    PNG_B64,
    body_of,
    embedding_peer,
    marker,
    models_path,
    path_of,
    vertex,
)
from pydantic import JsonValue

_IMAGE_URI: Final = "gs://dec1-bucket/images/cat.png"
_VIDEO_URI: Final = "gs://dec1-bucket/clips/cat.mp4"


def _instances(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [object_value(item) for item in list_value(body["instances"])]


# ---------------------------------------------------------------- text embedding task types


def test_task_type_and_title_are_set_on_every_text_instance(gateway: Gateway) -> None:
    """`task_type: "RETRIEVAL_DOCUMENT"` and `title: "Handbook"` on a two input request are copied onto
    both `instances[]` entries next to `content` (`create_embedding_input`), since Vertex reads them per
    instance."""
    first: Final = f"first {marker()}"
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        response: Final = rig.post(
            "/v1/embeddings", {"input": [first, "second"], "task_type": "RETRIEVAL_DOCUMENT", "title": "Handbook"}
        )
        assert response.status_code == 200, response.text
        assert _instances(rig.peer.one_body()) == [
            {"content": first, "task_type": "RETRIEVAL_DOCUMENT", "title": "Handbook"},
            {"content": "second", "task_type": "RETRIEVAL_DOCUMENT", "title": "Handbook"},
        ]


@pytest.mark.parametrize(
    "task_type", ["RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CODE_RETRIEVAL_QUERY"]
)
def test_each_vertex_task_type_is_forwarded_as_given(gateway: Gateway, task_type: str) -> None:
    """Any Vertex task type string is forwarded unchanged on the instance; the proxy does not validate or
    rename it."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"t {marker()}"], "task_type": task_type}).status_code == 200
        assert _instances(rig.peer.one_body())[0]["task_type"] == task_type


def test_openai_input_type_becomes_the_instance_task_type(gateway: Gateway) -> None:
    """`input_type: "RETRIEVAL_QUERY"` (the Cohere style name some SDKs send) is mapped to the instance's
    `task_type: "RETRIEVAL_QUERY"` (`VertexAITextEmbeddingConfig.map_openai_params`)."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        assert (
            rig.post("/v1/embeddings", {"input": [f"t {marker()}"], "input_type": "RETRIEVAL_QUERY"}).status_code == 200
        )
        instance: Final = _instances(rig.peer.one_body())[0]
        assert instance["task_type"] == "RETRIEVAL_QUERY"
        assert "input_type" not in instance


def test_without_a_task_type_the_instance_has_only_content(gateway: Gateway) -> None:
    """A plain request sends instances of exactly `{"content": <text>}`, with no `task_type` or `title`
    (Vertex then applies its own default)."""
    text: Final = f"plain {marker()}"
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [text]}).status_code == 200
        assert _instances(rig.peer.one_body()) == [{"content": text}]


def test_auto_truncate_false_is_sent_as_a_predict_parameter(gateway: Gateway) -> None:
    """`auto_truncate: false` is sent as `parameters.auto_truncate: false` so Vertex errors on an over long
    input instead of silently embedding a prefix. UNVERIFIED: the parameter key spelling (`auto_truncate`, as
    given, rather than `autoTruncate`); it comes from passing provider params through `parameters`."""
    with vertex(gateway, embedding_peer, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"t {marker()}"], "auto_truncate": False}).status_code == 200
        assert object_value(rig.peer.one_body()["parameters"])["auto_truncate"] is False


# ---------------------------------------------------------------- Gemini Embedding 2 (`:embedContent`)


def _embed_content_peer(request: Request) -> Reply:
    return Reply(body=json.dumps({"embedding": {"values": [0.125, 0.875]}}).encode())


def test_gemini_embedding_2_posts_embed_content(gateway: Gateway) -> None:
    """`vertex_ai/gemini-embedding-2` (cost map `uses_embed_content: true`) posts once to
    `/v1/projects/dec1-vertex-project/locations/us-central1/publishers/google/models/gemini-embedding-2:
    embedContent` with `content.parts: [{"text": <input>}]`, and the single vector comes back as
    `data[0].embedding`."""
    text: Final = f"embed me {marker()}"
    with vertex(gateway, _embed_content_peer, backend=GEMINI_EMBEDDING_2) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": text})
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert path_of(call) == f"{models_path(GEMINI_EMBEDDING_2)}:embedContent"
        assert object_value(body_of(call)["content"])["parts"] == [{"text": text}]
        assert object_value(list_value(json_of(response)["data"])[0])["embedding"] == [0.125, 0.875]


def test_gemini_embedding_2_maps_task_type_title_and_dimensions_to_camel_case(gateway: Gateway) -> None:
    """`task_type`, `title` and `dimensions: 256` are sent as top level `taskType`, `title` and
    `outputDimensionality: 256` on the `embedContent` body (`_filter_embed_params`), not as snake case."""
    with vertex(gateway, _embed_content_peer, backend=GEMINI_EMBEDDING_2) as rig:
        response: Final = rig.post(
            "/v1/embeddings",
            {"input": f"t {marker()}", "task_type": "RETRIEVAL_DOCUMENT", "title": "Doc", "dimensions": 256},
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert (body["taskType"], body["title"], body["outputDimensionality"]) == ("RETRIEVAL_DOCUMENT", "Doc", 256)
        assert "task_type" not in body


def test_gemini_embedding_2_sends_auto_truncate_in_embed_content_config(gateway: Gateway) -> None:
    """Since #51686 `auto_truncate: false` is sent as `embedContentConfig: {"autoTruncate": false}`; before
    it the flag was dropped without an error."""
    with vertex(gateway, _embed_content_peer, backend=GEMINI_EMBEDDING_2) as rig:
        assert rig.post("/v1/embeddings", {"input": f"t {marker()}", "auto_truncate": False}).status_code == 200
        assert rig.peer.one_body()["embedContentConfig"] == {"autoTruncate": False}


def test_gemini_embedding_2_sends_document_ocr_and_audio_track_extraction(gateway: Gateway) -> None:
    """`document_ocr: true` and `audio_track_extraction: true` land under `embedContentConfig` as
    `documentOcr` and `audioTrackExtraction` (#51686), next to `autoTruncate` when that is given too."""
    with vertex(gateway, _embed_content_peer, backend=GEMINI_EMBEDDING_2) as rig:
        response: Final = rig.post(
            "/v1/embeddings",
            {"input": f"t {marker()}", "document_ocr": True, "audio_track_extraction": True, "auto_truncate": True},
        )
        assert response.status_code == 200, response.text
        assert rig.peer.one_body()["embedContentConfig"] == {
            "documentOcr": True,
            "audioTrackExtraction": True,
            "autoTruncate": True,
        }


def test_gemini_embedding_2_without_config_params_sends_no_embed_content_config(gateway: Gateway) -> None:
    """A plain request sends no `embedContentConfig` key at all (`_embed_content_config` returns `{}`), so
    Google's defaults apply."""
    with vertex(gateway, _embed_content_peer, backend=GEMINI_EMBEDDING_2) as rig:
        assert rig.post("/v1/embeddings", {"input": f"t {marker()}"}).status_code == 200
        assert "embedContentConfig" not in rig.peer.one_body()


# ---------------------------------------------------------------- multimodal embeddings


def _multimodal_peer(predictions: list[dict[str, JsonValue]]) -> Callable[[Request], Reply]:
    return lambda _request: Reply(body=json.dumps({"predictions": predictions}).encode())


_TEXT_PREDICTION: Final[dict[str, JsonValue]] = {"textEmbedding": [0.1, 0.2]}
_IMAGE_PREDICTION: Final[dict[str, JsonValue]] = {"imageEmbedding": [0.3, 0.4]}


def test_multimodal_text_posts_predict_with_a_text_instance(gateway: Gateway) -> None:
    """`vertex_ai/multimodalembedding@001` with a text input posts to `.../models/multimodalembedding@001:
    predict` with `instances: [{"text": <input>}]`."""
    text: Final = f"a cat {marker()}"
    with vertex(gateway, _multimodal_peer([_TEXT_PREDICTION]), backend=MULTIMODAL_EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [text]}).status_code == 200
        call: Final = rig.peer.one_call()
        assert path_of(call) == f"{models_path(MULTIMODAL_EMBEDDING)}:predict"
        assert _instances(body_of(call)) == [{"text": text}]


def test_a_gcs_image_becomes_an_image_instance_with_gcs_uri(gateway: Gateway) -> None:
    """A `gs://...png` input is sent as `{"image": {"gcsUri": "gs://dec1-bucket/images/cat.png"}}`."""
    with vertex(gateway, _multimodal_peer([_IMAGE_PREDICTION]), backend=MULTIMODAL_EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [_IMAGE_URI]}).status_code == 200
        assert _instances(rig.peer.one_body()) == [{"image": {"gcsUri": _IMAGE_URI}}]


def test_a_data_uri_image_is_sent_as_base64_bytes_without_the_prefix(gateway: Gateway) -> None:
    """A `data:image/png;base64,<b64>` input is sent as `{"image": {"bytesBase64Encoded": <b64>}}` with the
    `data:...,` prefix cut off (`_create_image_instance`)."""
    with vertex(gateway, _multimodal_peer([_IMAGE_PREDICTION]), backend=MULTIMODAL_EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"data:image/png;base64,{PNG_B64}"]}).status_code == 200
        assert _instances(rig.peer.one_body()) == [{"image": {"bytesBase64Encoded": PNG_B64}}]


def test_a_gcs_mp4_becomes_a_video_instance(gateway: Gateway) -> None:
    """A `gs://...mp4` input is sent as `{"video": {"gcsUri": "gs://dec1-bucket/clips/cat.mp4"}}` (any URI
    containing `mp4` is treated as video)."""
    reply: Final = [{"videoEmbeddings": [{"embedding": [0.5], "startOffsetSec": 0, "endOffsetSec": 4}]}]
    with vertex(gateway, _multimodal_peer(reply), backend=MULTIMODAL_EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [_VIDEO_URI]}).status_code == 200
        assert _instances(rig.peer.one_body()) == [{"video": {"gcsUri": _VIDEO_URI}}]


def test_a_text_followed_by_an_image_is_merged_into_one_instance(gateway: Gateway) -> None:
    """`["a cat", "gs://...cat.png"]` becomes ONE instance `{"text": "a cat ...", "image": {"gcsUri": ...}}`
    (`_try_merge_text_with_media`), not two, so the two inputs share one prediction."""
    text: Final = f"a cat {marker()}"
    with vertex(
        gateway, _multimodal_peer([{**_TEXT_PREDICTION, **_IMAGE_PREDICTION}]), backend=MULTIMODAL_EMBEDDING
    ) as rig:
        assert rig.post("/v1/embeddings", {"input": [text, _IMAGE_URI]}).status_code == 200
        assert _instances(rig.peer.one_body()) == [{"text": text, "image": {"gcsUri": _IMAGE_URI}}]


def test_a_merged_text_and_image_prediction_returns_only_the_text_vector(gateway: Gateway) -> None:
    """When one prediction holds both `textEmbedding` and `imageEmbedding`, only the text vector is
    returned (`transform_embedding_response_to_openai` checks `textEmbedding` first, `elif` the image), so
    the caller gets one `data` entry for two inputs. Surprising, but it is today's contract."""
    with vertex(
        gateway, _multimodal_peer([{**_TEXT_PREDICTION, **_IMAGE_PREDICTION}]), backend=MULTIMODAL_EMBEDDING
    ) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [f"a cat {marker()}", _IMAGE_URI]})
        assert response.status_code == 200, response.text
        data: Final = [object_value(item) for item in list_value(json_of(response)["data"])]
        assert [item["embedding"] for item in data] == [[0.1, 0.2]]


def test_dimensions_become_the_multimodal_dimension_parameter(gateway: Gateway) -> None:
    """`dimensions: 128` is sent as `parameters: {"dimension": 128}` (singular, the multimodal model's
    name for it)."""
    with vertex(gateway, _multimodal_peer([_TEXT_PREDICTION]), backend=MULTIMODAL_EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [f"t {marker()}"], "dimensions": 128}).status_code == 200
        assert rig.peer.one_body()["parameters"] == {"dimension": 128}


def test_multimodal_predictions_come_back_in_order_and_video_segments_each_get_an_entry(gateway: Gateway) -> None:
    """For inputs `[image, video]` with predictions `[{imageEmbedding}, {videoEmbeddings: [two segments]}]`
    the response `data` holds the image vector at index 0 and one entry per video segment, both at index 1."""
    predictions: Final[list[dict[str, JsonValue]]] = [
        _IMAGE_PREDICTION,
        {
            "videoEmbeddings": [
                {"embedding": [0.6], "startOffsetSec": 0, "endOffsetSec": 4},
                {"embedding": [0.7], "startOffsetSec": 4, "endOffsetSec": 10},
            ]
        },
    ]
    with vertex(gateway, _multimodal_peer(predictions), backend=MULTIMODAL_EMBEDDING) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [_IMAGE_URI, _VIDEO_URI]})
        assert response.status_code == 200, response.text
        data: Final = [object_value(item) for item in list_value(json_of(response)["data"])]
        assert [(item["index"], item["embedding"]) for item in data] == [(0, [0.3, 0.4]), (1, [0.6]), (1, [0.7])]


def test_multimodal_usage_counts_characters_images_and_video_seconds(gateway: Gateway) -> None:
    """Usage is reported with `prompt_tokens: 0` and `prompt_tokens_details` holding `character_count`
    (the text length), `image_count: 1` and `video_length_seconds: 10.0` (the summed segment spans), which
    the per-character, per-image and per-second prices bill on. UNVERIFIED: `count_characters` counts the
    raw length (no whitespace stripping); the text here has no spaces to keep it exact."""
    text: Final = f"cat{marker()}"
    predictions: Final[list[dict[str, JsonValue]]] = [
        {**_TEXT_PREDICTION, **_IMAGE_PREDICTION},
        {
            "videoEmbeddings": [
                {"embedding": [0.6], "startOffsetSec": 0, "endOffsetSec": 4},
                {"embedding": [0.7], "startOffsetSec": 4, "endOffsetSec": 10},
            ]
        },
    ]
    with vertex(gateway, _multimodal_peer(predictions), backend=MULTIMODAL_EMBEDDING) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [text, _IMAGE_URI, _VIDEO_URI]})
        assert response.status_code == 200, response.text
        usage: Final = object_value(json_of(response)["usage"])
        assert usage["prompt_tokens"] == 0
        details: Final = object_value(usage["prompt_tokens_details"])
        assert (details["character_count"], details["image_count"], details["video_length_seconds"]) == (
            len(text),
            1,
            10.0,
        )


def test_a_multimodal_answer_without_predictions_is_a_server_error(gateway: Gateway) -> None:
    """An answer without `predictions` is refused with `embedding response does not contain 'predictions'`
    (an `InternalServerError`, status 500) instead of an empty list. UNVERIFIED: the status after proxy
    exception mapping."""
    with vertex(gateway, lambda _r: Reply(body=b'{"unexpected": true}'), backend=MULTIMODAL_EMBEDDING) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [f"t {marker()}"]})
        assert response.status_code == 500, response.text
        assert "does not contain 'predictions'" in response.text
