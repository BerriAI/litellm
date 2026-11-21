"""`gemini/` embeddings (`batchEmbedContents`) and provider token counting (`countTokens`).

Embeddings. `/v1/embeddings` on a `gemini/` deployment always uses Google's batch method, one
`POST {api_base}/models/{model}:batchEmbedContents` per call whose body is `{"requests": [...]}`, one
`EmbedContentRequest` per input with `model: "models/<model>"` and `content.parts`; OpenAI
`dimensions` becomes `outputDimensionality`, `task_type` (or the OpenAI style `input_type`) becomes
`taskType`, and `title` passes through (`transform_openai_input_gemini_content`). Vertex embeddings use
`embedContent`/`predict` instead (`batch_embed_content_handler.py`: `use_embed_content` only for
vertex_ai). Since #51686 (2026-11-13) `document_ocr`, `audio_track_extraction` and `auto_truncate` go
under `embedContentConfig`. The answer's `embeddings[].values` come back as `data[i].embedding` in
input order. File block and multimodal inputs are pinned by `test_gemini_embedding_file_block_wire.py`.

Token counting. `POST /utils/token_counter?call_endpoint=true` and `/v1/messages/count_tokens` on a
`gemini/` deployment ask Google: `POST {api_base}/v1beta/models/{model}:countTokens` with the key header
(`GoogleAIStudioTokenCounter`). Unlike chat, the count path appends its own `v1beta`, so these tests give
the deployment a bare host `api_base`. A body is `{"contents": [...]}`, or
`{"generateContentRequest": {...}}` when a system instruction or tools must be counted.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_gemini_support import (
    EMBEDDING,
    FLASH,
    GEMINI_KEY,
    body_of,
    gemini,
    gemini_echo,
    google_error,
    headers_of,
    json_of,
    marker,
    path_of,
    reply_json,
)
from pydantic import JsonValue

_BATCH: Final = f"/v1beta/models/{EMBEDDING}:batchEmbedContents"


def _embed(request: Request) -> Reply:
    if path_of(request).endswith(":batchEmbedContents"):
        count: Final = len(list_value(body_of(request)["requests"]))
        return reply_json({"embeddings": [{"values": [0.25, float(index)]} for index in range(count)]})
    if path_of(request).endswith(":countTokens"):
        return reply_json({"totalTokens": 31, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 31}]})
    return gemini_echo(request)


def _requests(rig_body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [object_value(item) for item in list_value(rig_body["requests"])]


def _one_embed_body(rig: object) -> dict[str, JsonValue]:
    calls: Final = rig.peer.to(":batchEmbedContents")  # pyright: ignore[reportAttributeAccessIssue]
    assert len(calls) == 1, [call.target for call in calls]
    return body_of(calls[0])


# ---------------------------------------------------------------- embeddings


def test_an_embedding_posts_one_batch_embed_contents_with_the_key_header(gateway: Gateway) -> None:
    """Two inputs reach the peer as ONE `POST /v1beta/models/gemini-embedding-001:batchEmbedContents` with
    `x-goog-api-key` and no `key=` query, not as two `embedContent` calls."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [f"a {marker()}", "b"]})
        assert response.status_code == 200, response.text
        calls: Final = rig.peer.to(":batchEmbedContents")
        assert [path_of(call) for call in calls] == [_BATCH]
        assert headers_of(calls[0])["x-goog-api-key"] == GEMINI_KEY
        assert "key=" not in calls[0].target
        assert rig.peer.to(":embedContent") == ()


def test_each_input_is_one_request_naming_the_model(gateway: Gateway) -> None:
    """The body is `{"requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": a}]}},
    {... b ...}]}`, in input order, one request per input string."""
    first: Final = f"alpha {marker()}"
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [first, "beta"]}).status_code == 200
        requests: Final = _requests(_one_embed_body(rig))
        assert [item["model"] for item in requests] == [f"models/{EMBEDDING}"] * 2
        assert [item["content"] for item in requests] == [{"parts": [{"text": first}]}, {"parts": [{"text": "beta"}]}]


def test_a_single_string_input_is_one_request(gateway: Gateway) -> None:
    """`input: "<text>"` (not a list) is still sent through `batchEmbedContents` as one request."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": f"solo {marker()}"}).status_code == 200
        assert len(_requests(_one_embed_body(rig))) == 1


def test_dimensions_become_output_dimensionality_on_every_request(gateway: Gateway) -> None:
    """`dimensions: 256` is put on each request as `outputDimensionality: 256` and no `dimensions` key is sent."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [marker(), "x"], "dimensions": 256}).status_code == 200
        requests: Final = _requests(_one_embed_body(rig))
        assert all(item.get("outputDimensionality") == 256 and "dimensions" not in item for item in requests)


def test_task_type_and_title_are_put_on_every_request(gateway: Gateway) -> None:
    """Provider params `task_type: "RETRIEVAL_DOCUMENT"` and `title: "Handbook"` become `taskType` and
    `title` on each request (`_GEMINI_PARAM_BY_OPENAI_PARAM`, `_SUPPORTED_EMBED_PARAMS`)."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        response: Final = rig.post(
            "/v1/embeddings", {"input": [marker()], "task_type": "RETRIEVAL_DOCUMENT", "title": "Handbook"}
        )
        assert response.status_code == 200, response.text
        request: Final = _requests(_one_embed_body(rig))[0]
        assert (request.get("taskType"), request.get("title")) == ("RETRIEVAL_DOCUMENT", "Handbook")
        assert "task_type" not in request


def test_openai_input_type_becomes_task_type(gateway: Gateway) -> None:
    """`input_type: "RETRIEVAL_QUERY"` (the cross provider spelling) is mapped to `task_type` by
    `VertexAITextEmbeddingConfig.map_openai_params` and reaches Google as `taskType: "RETRIEVAL_QUERY"`."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [marker()], "input_type": "RETRIEVAL_QUERY"}).status_code == 200
        assert _requests(_one_embed_body(rig))[0].get("taskType") == "RETRIEVAL_QUERY"


def test_embed_content_config_options_are_nested(gateway: Gateway) -> None:
    """`auto_truncate: false` and `document_ocr: true` are sent on each request as
    `embedContentConfig: {"autoTruncate": false, "documentOcr": true}` (#51686)."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        response: Final = rig.post(
            "/v1/embeddings", {"input": [marker()], "auto_truncate": False, "document_ocr": True}
        )
        assert response.status_code == 200, response.text
        assert _requests(_one_embed_body(rig))[0].get("embedContentConfig") == {
            "autoTruncate": False,
            "documentOcr": True,
        }


def test_float_encoding_format_is_accepted_and_not_forwarded(gateway: Gateway) -> None:
    """The OpenAI SDK always sends `encoding_format: "float"`; for gemini it is dropped (float lists are
    what Google returns) instead of being refused, and no `encoding_format` key reaches Google."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        assert rig.post("/v1/embeddings", {"input": [marker()], "encoding_format": "float"}).status_code == 200
        assert "encoding_format" not in json.dumps(_one_embed_body(rig))


def test_base64_encoding_format_is_refused(gateway: Gateway) -> None:
    """`encoding_format: "base64"` stays on the unsupported param path and is answered 400 without a
    Google call. UNVERIFIED: the shared proxy may set `drop_params`, in which case it is dropped."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        response: Final = rig.post("/v1/embeddings", {"input": [marker()], "encoding_format": "base64"})
        assert response.status_code == 400, response.text
        assert rig.peer.to(":batchEmbedContents") == ()


def test_vectors_come_back_in_input_order(gateway: Gateway) -> None:
    """Google's `embeddings[i].values` come back as `data[i].embedding` with `index: i` and `object:
    "embedding"`, and the list object is `"list"`."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        body: Final = json_of(rig.post("/v1/embeddings", {"input": [marker(), "b", "c"]}))
        assert body["object"] == "list"
        data: Final = [object_value(item) for item in list_value(body["data"])]
        assert [item["index"] for item in data] == [0, 1, 2]
        assert [item["embedding"] for item in data] == [[0.25, 0.0], [0.25, 1.0], [0.25, 2.0]]


def test_text_embedding_usage_is_counted_locally(gateway: Gateway) -> None:
    """`batchEmbedContents` answers carry no token counts, so `usage.prompt_tokens` is computed by the
    proxy's tokenizer and equals `total_tokens` (both greater than 0) (`process_response`)."""
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        usage: Final = object_value(json_of(rig.post("/v1/embeddings", {"input": [f"some words {marker()}"]}))["usage"])
        assert int(str(usage["prompt_tokens"])) > 0
        assert usage["prompt_tokens"] == usage["total_tokens"]


def test_an_embedding_429_is_answered_as_429(gateway: Gateway) -> None:
    """Google's 429 `RESOURCE_EXHAUSTED` on `batchEmbedContents` leaves the proxy as 429."""
    with gemini(
        gateway, lambda _r: google_error(429, "RESOURCE_EXHAUSTED", "Resource exhausted"), backend=EMBEDDING
    ) as rig:
        assert rig.post("/v1/embeddings", {"input": [marker()]}).status_code == 429


def test_a_files_reference_input_is_resolved_against_the_api_base(gateway: Gateway) -> None:
    """An embedding input `files/dec1doc` must be resolved with `GET {api_base}/v1beta/files/dec1doc` on the
    deployment's own base; today `_async_resolve_file_references` hard codes
    `https://generativelanguage.googleapis.com/v1beta/...`, so a deployment behind a gateway reaches Google
    directly with its key."""
    pytest.skip(
        "BUG: Gemini embedding files/ references are resolved on generativelanguage.googleapis.com, ignoring api_base"
    )
    with gemini(gateway, _embed, backend=EMBEDDING) as rig:
        rig.post("/v1/embeddings", {"input": ["files/dec1doc"]})
        assert len(rig.peer.to("/v1beta/files/dec1doc", method="GET")) == 1


# ---------------------------------------------------------------- token counting


def _count(rig: object, **fields: JsonValue) -> object:
    return rig.gateway.client.post(  # pyright: ignore[reportAttributeAccessIssue]
        "/utils/token_counter",
        params={"call_endpoint": "true"},
        json={"model": rig.model, **fields},  # pyright: ignore[reportAttributeAccessIssue]
        headers={"Authorization": f"Bearer {rig.gateway.key}"},  # pyright: ignore[reportAttributeAccessIssue]
    )


def test_token_counter_asks_google_count_tokens_with_the_key(gateway: Gateway) -> None:
    """`POST /utils/token_counter?call_endpoint=true` on a bare host deployment sends
    `POST /v1beta/models/gemini-2.5-flash:countTokens` with `x-goog-api-key`, and answers `total_tokens: 31`
    from Google's `totalTokens`."""
    with gemini(gateway, _embed, base_suffix="") as rig:
        response: Final = _count(rig, messages=[{"role": "user", "content": f"count me {marker()}"}])
        assert response.status_code == 200, response.text  # pyright: ignore[reportAttributeAccessIssue]
        calls: Final = rig.peer.to(":countTokens")
        assert [path_of(call) for call in calls] == [f"/v1beta/models/{FLASH}:countTokens"]
        assert headers_of(calls[0])["x-goog-api-key"] == GEMINI_KEY
        assert json_of(response)["total_tokens"] == 31  # pyright: ignore[reportArgumentType]


def test_a_plain_count_sends_only_contents(gateway: Gateway) -> None:
    """Without a system message or tools the countTokens body is exactly `{"contents": [{"role": "user",
    "parts": [{"text": ...}]}]}` (`_build_count_tokens_request_body`)."""
    text: Final = f"count me {marker()}"
    with gemini(gateway, _embed, base_suffix="") as rig:
        _count(rig, messages=[{"role": "user", "content": text}])
        body: Final = body_of(rig.peer.to(":countTokens")[0])
        assert set(body) == {"contents"}
        assert text in json.dumps(body["contents"])


def test_a_count_with_tools_wraps_a_generate_content_request(gateway: Gateway) -> None:
    """With tools the body becomes `{"generateContentRequest": {"model": "models/gemini-2.5-flash",
    "contents": [...], "tools": [...]}}`, so the tool declarations are counted too."""
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
    }
    with gemini(gateway, _embed, base_suffix="") as rig:
        _count(rig, messages=[{"role": "user", "content": marker()}], tools=[tool])
        body: Final = body_of(rig.peer.to(":countTokens")[0])
        request: Final = object_value(body["generateContentRequest"])
        assert request["model"] == f"models/{FLASH}"
        assert "get_weather" in json.dumps(request["tools"])


def test_messages_count_tokens_counts_the_system_prompt_through_google(gateway: Gateway) -> None:
    """`POST /v1/messages/count_tokens` with `system: "Be terse"` on the gemini deployment sends a
    `generateContentRequest` whose `systemInstruction` holds "Be terse", and answers Anthropic's
    `{"input_tokens": 31}`."""
    with gemini(gateway, _embed, base_suffix="") as rig:
        response: Final = rig.post(
            "/v1/messages/count_tokens",
            {"system": "Be terse", "messages": [{"role": "user", "content": marker()}]},
        )
        assert response.status_code == 200, response.text
        assert json_of(response)["input_tokens"] == 31
        request: Final = object_value(body_of(rig.peer.to(":countTokens")[0])["generateContentRequest"])
        assert "Be terse" in json.dumps(request["systemInstruction"])


def test_a_count_with_a_versioned_api_base_doubles_the_version(gateway: Gateway) -> None:
    """With the `api_base: http://<peer>/v1beta` that chat and context caching expect, the count path should
    be `/v1beta/models/gemini-2.5-flash:countTokens`; `_construct_url` appends its own `v1beta/` and posts to
    `/v1beta/v1beta/...` instead, so one deployment cannot serve both chat and counting."""
    pytest.skip("BUG: gemini countTokens appends v1beta to an api_base that already ends in /v1beta")
    with gemini(gateway, _embed) as rig:
        _count(rig, messages=[{"role": "user", "content": marker()}])
        assert [path_of(call) for call in rig.peer.to(":countTokens")] == [f"/v1beta/models/{FLASH}:countTokens"]


def test_a_count_tokens_error_from_google_reaches_the_caller(gateway: Gateway) -> None:
    """A 400 from `countTokens` is answered as an error with Google's message rather than a silent local
    estimate. UNVERIFIED: the status the proxy answers (`provider_error_response`), expected 400."""
    with gemini(
        gateway, lambda _r: google_error(400, "INVALID_ARGUMENT", "contents is not specified"), base_suffix=""
    ) as rig:
        response: Final = _count(rig, messages=[{"role": "user", "content": marker()}])
        assert response.status_code == 400, response.text  # pyright: ignore[reportAttributeAccessIssue]
        assert "contents is not specified" in response.text  # pyright: ignore[reportAttributeAccessIssue]


def test_a_count_without_call_endpoint_never_reaches_google(gateway: Gateway) -> None:
    """`/utils/token_counter` without `call_endpoint=true` counts locally: the answer has a positive
    `total_tokens` and no `countTokens` request is made."""
    with gemini(gateway, _embed, base_suffix="") as rig:
        response: Final = rig.gateway.request(
            "POST", "/utils/token_counter", {"model": rig.model, "messages": [{"role": "user", "content": marker()}]}
        )
        assert response.status_code == 200, response.text
        assert int(str(json_of(response)["total_tokens"])) > 0
        assert rig.peer.to(":countTokens") == ()
