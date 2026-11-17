"""`/v1/embeddings` on Bedrock: Titan and Cohere bodies, `extra_body` merging, regions, usage and cost.

Bedrock embeddings go to `POST /model/<url-encoded id>/invoke` (`litellm/llms/bedrock/embed/embedding.py`).
Titan models take one input per request (`{"inputText": ...}`), so a list of inputs fans out into one call per
item and the reported usage is the sum of each reply's `inputTextTokenCount`; Titan v2 also maps
`dimensions` and `encoding_format` (`float` to `embeddingTypes: ["float"]`, `base64` to `["binary"]`).
Cohere takes the whole list in one call (`{"texts": [...], "input_type": "search_document", ...}`).

Since #51710 (Nov 16) a caller's `extra_body` is merged into the provider body: its fields are added at the
top level, its values replace fields the transform only fills with a default (Cohere `input_type`), an
object on both sides is combined field by field, and generated content fields (`inputText`, `texts`) always
win. The basic Titan v2 fan-out, Titan usage sum, Titan v1 `dimensions` refusal and Cohere v3 renames are
pinned in `test_dec1_textapis_embeddings_providers.py` and `test_bedrock_embedding_wire.py`; these tests add
the rest.
"""

from __future__ import annotations

import base64
import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock_support import (
    ACCESS_KEY,
    COHERE_EMBED_V3,
    COHERE_EMBED_V4,
    TITAN_EMBED_V2,
    aws_error,
    bedrock,
    body_of,
    error_message,
    json_reply,
    marker,
    path_of,
    sigv4_params,
    sigv4_scope,
)

VECTOR: Final = [0.25, -0.5, 0.75]


def _titan_reply(request: Request) -> Reply:
    sent: Final = body_of(request)
    if sent.get("embeddingTypes") == ["binary"]:
        return json_reply({"embeddingsByType": {"binary": [1, 0, 1]}, "inputTextTokenCount": 4})
    return json_reply({"embedding": VECTOR, "inputTextTokenCount": 4})


def _cohere_reply(request: Request) -> Reply:
    texts: Final = body_of(request)["texts"]
    assert isinstance(texts, list)
    return json_reply(
        {
            "embeddings": {"float": [VECTOR for _ in texts]},
            "id": "cohere-dec1",
            "response_type": "embeddings_by_type",
            "texts": texts,
        }
    )


def test_titan_v2_dimensions_and_float_encoding_reach_the_body(gateway: Gateway) -> None:
    """`dimensions: 256` and `encoding_format: "float"` on `bedrock/amazon.titan-embed-text-v2:0`.

    Bedrock must receive `POST /model/amazon.titan-embed-text-v2%3A0/invoke` with the body
    `{"inputText": <text>, "dimensions": 256, "embeddingTypes": ["float"]}`
    (`amazon_titan_v2_transformation.py:62`), and the caller gets the `embedding` vector back.
    """
    text: Final = marker()
    with bedrock(gateway, _titan_reply, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed(text, dimensions=256, encoding_format="float")
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == "/model/amazon.titan-embed-text-v2%3A0/invoke", sent.target
        assert body_of(sent) == {"inputText": text, "dimensions": 256, "embeddingTypes": ["float"]}
        assert response.json()["data"][0]["embedding"] == VECTOR


def test_titan_v2_base64_encoding_asks_for_binary_and_returns_the_binary_vector(gateway: Gateway) -> None:
    """`encoding_format: "base64"` maps to Titan v2 `embeddingTypes: ["binary"]`.

    Bedrock answers `embeddingsByType.binary: [1, 0, 1]`, and the caller's `data[0].embedding` must be that
    list (`_transform_response` prefers `embeddingsByType.binary`, `amazon_titan_v2_transformation.py:94`).
    UNVERIFIED whether the proxy re-encodes the list as a base64 string for OpenAI compatibility.
    """
    with bedrock(gateway, _titan_reply, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed(marker(), encoding_format="base64")
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["embeddingTypes"] == ["binary"]
        assert response.json()["data"][0]["embedding"] == [1, 0, 1]


def test_titan_v2_extra_body_fields_are_added_to_the_body_but_cannot_replace_the_input(gateway: Gateway) -> None:
    """`extra_body: {"normalize": false, "inputText": "injected"}` on Titan v2 (#51710).

    Bedrock must receive `normalize: false` next to the generated fields, while `inputText` stays the
    caller's `input` because generated fields win over `extra_body` (`_with_extra_body`, `embedding.py:211`).
    """
    text: Final = marker()
    with bedrock(gateway, _titan_reply, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed(text, extra_body={"normalize": False, "inputText": "injected"})
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["normalize"] is False
        assert sent["inputText"] == text


def test_titan_v2_extra_body_is_merged_into_every_fanned_out_request(gateway: Gateway) -> None:
    """Two inputs on Titan v2 with `extra_body: {"normalize": false}`.

    Titan takes one input per call, so Bedrock must receive two POSTs, one `inputText` each in input order,
    and both must carry `normalize: false`; the caller gets two embeddings with `index` 0 and 1.
    """
    first, second = marker(), marker()
    with bedrock(gateway, _titan_reply, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed([first, second], extra_body={"normalize": False})
        assert response.status_code == 200, response.text
        sent: Final = [body_of(request) for request in rig.posts()]
        assert [body["inputText"] for body in sent] == [first, second]
        assert all(body["normalize"] is False for body in sent)
        assert [item["index"] for item in response.json()["data"]] == [0, 1]


def test_titan_v2_embedding_is_priced_from_the_summed_input_token_counts(gateway: Gateway) -> None:
    """Two inputs, each answered with `inputTextTokenCount: 4`, on Titan v2 with no deployment price.

    `usage.prompt_tokens` must be 8 and `x-litellm-response-cost` must be 8 * 2e-08 = 1.6e-07 from the
    `amazon.titan-embed-text-v2:0` cost map row.
    """
    with bedrock(gateway, _titan_reply, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed([marker(), marker()])
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["prompt_tokens"] == 8
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(8 * 2e-08)


def test_cohere_sends_every_input_in_one_request_in_order(gateway: Gateway) -> None:
    """Three inputs on `bedrock/cohere.embed-english-v3` go in one call: `texts` holds all three in input order.

    The caller gets three embeddings with `index` 0, 1 and 2.
    """
    inputs: Final = [marker(), marker(), marker()]
    with bedrock(gateway, _cohere_reply, model=f"bedrock/{COHERE_EMBED_V3}") as rig:
        response: Final = rig.embed(inputs)
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["texts"] == inputs
        assert [item["index"] for item in response.json()["data"]] == [0, 1, 2]


def test_cohere_extra_body_input_type_replaces_the_search_document_default(gateway: Gateway) -> None:
    """`extra_body: {"input_type": "search_query", "truncate": "END"}` on Cohere (#51710).

    `input_type` is a field the transform only defaults, so the caller's value must replace
    `search_document` (`_caller_values_for_defaults`, `embedding.py:155`), and `truncate: END` must be added
    at the top level.
    """
    with bedrock(gateway, _cohere_reply, model=f"bedrock/{COHERE_EMBED_V3}") as rig:
        response: Final = rig.embed([marker()], extra_body={"input_type": "search_query", "truncate": "END"})
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["input_type"] == "search_query"
        assert sent["truncate"] == "END"


def test_cohere_extra_body_cannot_replace_the_texts(gateway: Gateway) -> None:
    """`extra_body: {"texts": ["injected"]}` on Cohere: the generated `texts` from `input` must win.

    Bedrock must receive `texts == [<caller input>]`, so `extra_body` can never smuggle a different input.
    """
    text: Final = marker()
    with bedrock(gateway, _cohere_reply, model=f"bedrock/{COHERE_EMBED_V3}") as rig:
        response: Final = rig.embed([text], extra_body={"texts": ["injected"]})
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["texts"] == [text]


def test_cohere_embed_v4_goes_to_its_own_invoke_path(gateway: Gateway) -> None:
    """`bedrock/cohere.embed-v4:0` uses the Cohere transform at `/model/cohere.embed-v4%3A0/invoke`.

    The body must carry `texts` and `input_type: search_document`, and `usage.prompt_tokens` must be
    positive so the call is billed (`cohere.embed-v4:0` costs 1.2e-07 per token).
    UNVERIFIED: whether v4 adds `embedding_types` by default.
    """
    text: Final = marker()
    with bedrock(gateway, _cohere_reply, model=f"bedrock/{COHERE_EMBED_V4}") as rig:
        response: Final = rig.embed([text])
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == "/model/cohere.embed-v4%3A0/invoke", sent.target
        body: Final = body_of(sent)
        assert body["texts"] == [text]
        assert body["input_type"] == "search_document"
        assert response.json()["usage"]["prompt_tokens"] > 0


def test_a_region_segment_in_the_embedding_model_path_is_the_signing_region(gateway: Gateway) -> None:
    """`bedrock/eu-west-1/amazon.titan-embed-text-v2:0` with AWS keys and no `aws_region_name`.

    The embedding handler splits the region path (`split_bedrock_region_path`, `embedding.py:602`), so
    Bedrock must see `/model/amazon.titan-embed-text-v2:0/invoke` without the region and a SigV4 scope of
    `eu-west-1/bedrock`.
    """
    with bedrock(
        gateway, _titan_reply, model=f"bedrock/eu-west-1/{TITAN_EMBED_V2}", **sigv4_params(aws_region_name=None)
    ) as rig:
        response: Final = rig.embed(marker())
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{TITAN_EMBED_V2}/invoke"
        assert sigv4_scope(sent) == (ACCESS_KEY, "eu-west-1", "bedrock")


def test_a_throttled_embedding_is_a_429_with_the_bedrock_text(gateway: Gateway) -> None:
    """Bedrock answers the Titan embedding call with HTTP 429 ThrottlingException `Rate exceeded`.

    With `num_retries: 0` the caller must get HTTP 429 whose message contains `Rate exceeded`.
    """

    def respond(_: Request) -> Reply:
        return aws_error(429, "ThrottlingException", "Rate exceeded")

    with bedrock(gateway, respond, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed(marker(), num_retries=0)
        assert response.status_code == 429, response.text
        assert "Rate exceeded" in error_message(response), response.text


def test_an_embedding_validation_error_is_a_400(gateway: Gateway) -> None:
    """Bedrock answers HTTP 400 ValidationException `Malformed input request: #/dimensions: 300 is not a valid enum value`.

    The caller must get HTTP 400 carrying the Bedrock text, so clients see a bad request, not a server error.
    """

    def respond(_: Request) -> Reply:
        return aws_error(
            400, "ValidationException", "Malformed input request: #/dimensions: 300 is not a valid enum value"
        )

    with bedrock(gateway, respond, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed(marker(), dimensions=300, num_retries=0)
        assert response.status_code == 400, response.text
        assert "is not a valid enum value" in error_message(response), response.text


def test_an_embedding_spend_row_names_bedrock_and_the_input_tokens(gateway: Gateway) -> None:
    """A Titan v2 embedding with one input answered with `inputTextTokenCount: 4` and an input price of 0.001.

    One SpendLogs row must land with `call_type` `aembedding`, `custom_llm_provider` `bedrock`,
    `prompt_tokens` 4 and `spend` 0.004. The row is found through the caller's `x-litellm-call-id`.
    UNVERIFIED: the embedding row's `request_id` equals the response header `x-litellm-call-id`.
    """
    from integration.providers.dec1_bedrock_support import spend_row

    with bedrock(gateway, _titan_reply, model=f"bedrock/{TITAN_EMBED_V2}", input_cost_per_token=0.001) as rig:
        response: Final = rig.embed(marker())
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.headers["x-litellm-call-id"])
        assert row["call_type"] == "aembedding"
        assert row["custom_llm_provider"] == "bedrock"
        assert row["prompt_tokens"] == 4
        assert float(str(row["spend"])) == pytest.approx(0.004)


def test_a_base64_image_input_on_titan_multimodal_is_sent_as_input_image(gateway: Gateway) -> None:
    """A data URL image as the `input` of `bedrock/amazon.titan-embed-image-v1`.

    Bedrock must receive `inputImage` holding the raw base64 (no `data:` prefix) instead of `inputText`.
    UNVERIFIED: read from the provider name only (`amazon_titan_multimodal_transformation.py`); the exact
    key set beyond `inputImage` was not traced.
    """
    raw: Final = base64.b64encode(b"\x89PNG\r\n\x1a\nsynthetic").decode()

    def respond(_: Request) -> Reply:
        return json_reply({"embedding": VECTOR, "inputTextTokenCount": 0})

    with bedrock(gateway, respond, model="bedrock/amazon.titan-embed-image-v1") as rig:
        response: Final = rig.embed(f"data:image/png;base64,{raw}")
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent.get("inputImage") == raw, json.dumps(sent)[:400]
        assert "inputText" not in sent
