"""Contract: `/v1/embeddings` on Cohere, Voyage and Jina deployments, beyond the basics already pinned elsewhere.

The caller always speaks the OpenAI embeddings shape; the proxy translates it per provider and translates the
answer back. These tests pin the less common shapes that a Rust port of the embedding path has to keep:

    cohere      a `data:` URL input is sent as `images` with `input_type: "image"`; a dict of typed embeddings
                (`{"float": [...], "int8": [...]}`) is flattened into one OpenAI list; usage counts billed text
                plus image tokens; the deployment `api_base` is used verbatim; nested token arrays are decoded to
                text by the proxy before Cohere sees them, a flat token array is refused
    voyage      a `*context*` model uses `/contextualizedembeddings` with `inputs` and auto chunking unless the
                caller asks for `input_type: "query"`; `api_base` gets `/embeddings` appended unless it already
                ends there; usage is `total_tokens` for both prompt and total
    jina_ai     a mix of `data:` image URLs and text becomes `[{"image": <base64>}, {"text": ...}]`; a keyless
                deployment sends no Authorization header; provider fields such as `task` reach the body

Basic Cohere/Voyage/Jina bodies, param renames and unsupported-param refusals are in
`test_dec1_textapis_embeddings_providers.py`; this file targets what that file does not.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.providers.dec1_embedrerank_support import (
    DECODED_TOKEN_TEXTS,
    FLAT_TOKEN_INPUT,
    INPUT_RATE,
    PROVIDER_KEY,
    TOKEN_INPUT,
    VECTOR,
    Call,
    Peer,
    cohere_embed,
    embed,
    error_message,
    indexes,
    json_reply,
    marker,
    openai_list,
    peer,
    served,
    vectors,
)
from pydantic import JsonValue

from tests.integration._support.client import Scenario

_COHERE: Final = "embed-english-v3.0"
_VOYAGE: Final = "voyage-3.5"
_VOYAGE_CONTEXT: Final = "voyage-context-3"
_JINA: Final = "jina-embeddings-v4"
_PNG_DATA_URL: Final = "data:image/png;base64,iVBORw0KGgo="
_PNG_BASE64: Final = "iVBORw0KGgo="


def _cohere(scenario: Scenario, provider: Peer, suffix: str = "/v2/embed", **parameters: JsonValue) -> str:
    return scenario.model(
        model=f"cohere/{_COHERE}",
        api_base=f"{provider.url}{suffix}",
        api_key=PROVIDER_KEY,
        input_cost_per_token=INPUT_RATE,
        **parameters,
    )


def _voyage(scenario: Scenario, provider: Peer, upstream: str = _VOYAGE, suffix: str = "/v1") -> str:
    return scenario.model(
        model=f"voyage/{upstream}",
        api_base=f"{provider.url}{suffix}",
        api_key=PROVIDER_KEY,
        input_cost_per_token=INPUT_RATE,
    )


def _jina(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    return scenario.model(model=f"jina_ai/{_JINA}", api_base=f"{provider.url}/v1", **parameters)


def _voyage_contextual(call: Call) -> Reply:
    return json_reply(
        {
            "object": "list",
            "data": [
                {"object": "list", "index": 0, "data": [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}]}
            ],
            "model": _VOYAGE_CONTEXT,
            "usage": {"total_tokens": 9},
        }
    )


def test_cohere_api_base_is_used_verbatim_as_the_embed_url(gateway: Gateway) -> None:
    """A Cohere embedding deployment whose `api_base` is `<peer>/gateway/custom-embed` (no `/v2/embed` suffix).

    `CohereEmbeddingConfig.get_complete_url` returns an explicit `api_base` unchanged, so the one POST must land on
    `/gateway/custom-embed`. Only the env fallback `COHERE_API_BASE` (#50249) gets `/v2/embed` joined; a port that
    joins the route onto a deployment `api_base` too would break gateways that already expose the full URL.
    """
    with peer(cohere_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider, "/gateway/custom-embed")
        served(embed(gateway, model, [marker()]))
        assert provider.only().path == "/gateway/custom-embed"


def test_cohere_sends_the_deployment_key_as_a_bearer_with_json_content_type(gateway: Gateway) -> None:
    """A keyed Cohere embedding deployment serves one call.

    Cohere must receive `Authorization: Bearer dec1-embedrerank-provider-key` and `Content-Type:
    application/json`, never the caller's virtual key.
    """
    with peer(cohere_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        served(embed(gateway, model, [marker()]))
        call: Final = provider.only()
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.headers["content-type"].startswith("application/json")


def test_cohere_image_data_url_is_sent_as_images_with_the_image_input_type(gateway: Gateway) -> None:
    """The caller embeds one `data:image/png;base64,...` string on a Cohere v3 deployment.

    `_transform_request` detects base64 and must send `{"model": "embed-english-v3.0", "images": [<the data URL>],
    "input_type": "image"}` with no `texts` field. The data URL is sent whole, prefix included.
    """
    with peer(cohere_embed({"images": 1})) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        served(embed(gateway, model, [_PNG_DATA_URL]))
        assert provider.only().body == {"model": _COHERE, "images": [_PNG_DATA_URL], "input_type": "image"}


def test_cohere_billed_image_tokens_are_added_to_prompt_tokens(gateway: Gateway) -> None:
    """Cohere answers `meta.billed_units: {"input_tokens": 3, "image_tokens": 10}`.

    With `SEPARATE_FROM_INPUT_TOKENS` billing the caller's usage must be `prompt_tokens: 13`, `total_tokens: 13`,
    and `prompt_tokens_details` must carry `text_tokens: 3` and `image_tokens: 10`.
    """
    with peer(cohere_embed({"input_tokens": 3, "image_tokens": 10})) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        usage: Final = object_value(served(embed(gateway, model, [_PNG_DATA_URL]))["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (13, 13), usage
        details: Final = object_value(usage["prompt_tokens_details"])
        assert (details["text_tokens"], details["image_tokens"]) == (3, 10), details


def test_cohere_typed_embeddings_are_flattened_into_one_list_per_type(gateway: Gateway) -> None:
    """The caller embeds two strings with `encoding_format: "float"`; Cohere answers
    `embeddings: {"float": [v, v]}` (the typed v2 shape).

    The caller must get an OpenAI list with two entries indexed 0 and 1, each holding the float vector, and no
    `float` wrapper object in the response.
    """
    two: Final[JsonValue] = {"float": [list(VECTOR), list(VECTOR)]}
    with peer(cohere_embed(embeddings=two)) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        body: Final = served(embed(gateway, model, [marker(), marker()], encoding_format="float"))
        assert indexes(body) == [0, 1]
        assert vectors(body) == [list(VECTOR), list(VECTOR)]


def test_cohere_two_embedding_types_restart_the_index_for_each_type(gateway: Gateway) -> None:
    """Cohere answers one input with `embeddings: {"float": [v], "int8": [[1, 2, 3]]}`.

    `_response_rows` with the OPENAI row shape chains every type's rows and numbers each type from 0, so the caller
    gets two entries both with `index: 0`, the float vector first and then `[1, 2, 3]`. Pinned because an OpenAI
    client keyed on `index` sees a duplicate; a port must change this on purpose, not by accident.
    """
    typed: Final[JsonValue] = {"float": [list(VECTOR)], "int8": [[1, 2, 3]]}
    with peer(cohere_embed(embeddings=typed)) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        body: Final = served(embed(gateway, model, [marker()]))
        assert indexes(body) == [0, 0]
        assert vectors(body) == [list(VECTOR), [1, 2, 3]]


def test_cohere_nested_token_arrays_are_decoded_to_text_before_cohere(gateway: Gateway) -> None:
    """The caller sends `input: [[791, 4062, 14198, 2385], [40, 1093]]` to a Cohere deployment.

    Cohere is not in `LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS`, so the proxy decodes each array
    with the cl100k tokenizer before routing and Cohere must receive `texts: ["The quick brown base", "I like"]`.
    """
    with peer(cohere_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        served(embed(gateway, model, TOKEN_INPUT))
        assert provider.only().body["texts"] == DECODED_TOKEN_TEXTS


def test_cohere_refuses_a_flat_token_array_without_calling_cohere(gateway: Gateway) -> None:
    """The caller sends one flat token array `input: [791, 4062, 14198, 2385]` to a Cohere deployment.

    The proxy only decodes nested arrays, and `CohereEmbeddingConfig.transform_embedding_request` raises
    `Input must be a list of strings` for integer items. The caller must get an error carrying that text and Cohere
    must receive nothing. UNVERIFIED: the status (a bare ValueError, expected 500 today; 400 would be better).
    """
    with peer(cohere_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        response: Final = embed(gateway, model, FLAT_TOKEN_INPUT)
        assert response.status_code >= 400, response.text
        assert "Input must be a list of strings" in error_message(response), response.text
        assert provider.posts() == ()


def test_cohere_refuses_an_empty_input_list_with_400_and_no_provider_call(gateway: Gateway) -> None:
    """The caller sends `input: []` to a Cohere deployment.

    An empty list is a caller error, so the caller should get 400 and Cohere should receive nothing.
    """
    pytest.skip("BUG: Cohere transform_embedding_request reads input[0] of an empty list and raises IndexError (500)")
    with peer(cohere_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        response: Final = embed(gateway, model, [])
        assert response.status_code == 400, response.text
        assert provider.posts() == ()


def test_cohere_provider_500_reaches_the_caller_as_500_with_the_cohere_text(gateway: Gateway) -> None:
    """Cohere answers HTTP 500 with body `{"message": "dec1 cohere internal"}`.

    The caller must get HTTP 500 and the error message must contain `dec1 cohere internal` (CohereError keeps the
    provider's status and text).
    """

    def respond(call: Call) -> Reply:
        return json_reply({"message": "dec1 cohere internal"}, 500)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        response: Final = embed(gateway, model, [marker()])
        assert response.status_code == 500, response.text
        assert "dec1 cohere internal" in error_message(response)


def test_voyage_appends_embeddings_to_a_bare_host_api_base(gateway: Gateway) -> None:
    """A Voyage deployment whose `api_base` is the bare peer URL with no path.

    `ensure_api_route(api_base, "embeddings")` joins the route, so the POST must land on `/embeddings`.
    """
    with peer(openai_list(4)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider, suffix="")
        served(embed(gateway, model, [marker()]))
        assert provider.only().path == "/embeddings"


def test_voyage_keeps_an_api_base_that_already_ends_in_embeddings(gateway: Gateway) -> None:
    """A Voyage deployment whose `api_base` is `<peer>/v1/embeddings`.

    The route is already present, so the POST must land on `/v1/embeddings`, not `/v1/embeddings/embeddings`.
    """
    with peer(openai_list(4)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider, suffix="/v1/embeddings")
        served(embed(gateway, model, [marker()]))
        assert provider.only().path == "/v1/embeddings"


def test_voyage_sends_the_deployment_key_as_bearer(gateway: Gateway) -> None:
    """A keyed Voyage deployment serves one embedding.

    `VoyageEmbeddingConfig.validate_environment` sends exactly `Authorization: Bearer <deployment key>`.
    """
    with peer(openai_list(4)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider)
        served(embed(gateway, model, [marker()]))
        assert provider.only().headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_voyage_nested_token_arrays_are_decoded_to_text(gateway: Gateway) -> None:
    """The caller sends two token arrays to a Voyage deployment.

    Voyage does not accept token ids, so the proxy decodes them and Voyage must receive
    `input: ["The quick brown base", "I like"]`.
    """
    with peer(openai_list(4)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider)
        served(embed(gateway, model, TOKEN_INPUT))
        assert provider.only().body["input"] == DECODED_TOKEN_TEXTS


def test_voyage_usage_prompt_tokens_are_the_reported_total_tokens(gateway: Gateway) -> None:
    """Voyage answers `usage: {"total_tokens": 11}` with no `prompt_tokens`.

    The caller's usage must be `prompt_tokens: 11` and `total_tokens: 11`, and the cost header must be
    11 * 0.001 = 0.011.
    """

    def respond(call: Call) -> Reply:
        return json_reply(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}],
                "model": _VOYAGE,
                "usage": {"total_tokens": 11},
            }
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider)
        response: Final = embed(gateway, model, [marker()])
        usage: Final = object_value(served(response)["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (11, 11), usage
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.011)


def test_voyage_contextual_model_posts_inputs_with_auto_chunking(gateway: Gateway) -> None:
    """The caller embeds two strings on `voyage/voyage-context-3`.

    Any model name containing `context` uses `VoyageContextualEmbeddingConfig`: one POST to
    `/v1/contextualizedembeddings` with body `{"inputs": [a, b], "model": "voyage-context-3",
    "enable_auto_chunking": true, "chunk_size": 32000, "input_type": "document"}`.
    """
    first, second = marker(), marker()
    with peer(_voyage_contextual) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider, upstream=_VOYAGE_CONTEXT)
        served(embed(gateway, model, [first, second]))
        call: Final = provider.only()
        assert call.path == "/v1/contextualizedembeddings"
        assert call.body == {
            "inputs": [first, second],
            "model": _VOYAGE_CONTEXT,
            "enable_auto_chunking": True,
            "chunk_size": 32000,
            "input_type": "document",
        }


def test_voyage_contextual_query_input_type_turns_auto_chunking_off(gateway: Gateway) -> None:
    """The caller embeds one string on `voyage/voyage-context-3` with the Voyage field `input_type: "query"`.

    Queries are never chunked, so the body must carry `input_type: "query"` and no `enable_auto_chunking` or
    `chunk_size`. UNVERIFIED: that the caller's `input_type` reaches `optional_params` for Voyage through
    `add_provider_specific_params_to_optional_params`.
    """
    with peer(_voyage_contextual) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider, upstream=_VOYAGE_CONTEXT)
        served(embed(gateway, model, marker(), input_type="query"))
        body: Final = provider.only().body
        assert body["input_type"] == "query"
        assert "enable_auto_chunking" not in body and "chunk_size" not in body, body


def test_voyage_contextual_answer_is_relayed_with_total_tokens_usage(gateway: Gateway) -> None:
    """Voyage's contextual endpoint answers a nested `data` list and `usage.total_tokens: 9`.

    The caller must get `data` exactly as Voyage sent it (the nested per-document list is not re-shaped) and usage
    `prompt_tokens: 9`, `total_tokens: 9`.
    """
    with peer(_voyage_contextual) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider, upstream=_VOYAGE_CONTEXT)
        body: Final = served(embed(gateway, model, [marker()]))
        assert body["data"] == [
            {"object": "list", "index": 0, "data": [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}]}
        ]
        usage: Final = object_value(body["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (9, 9)


def test_jina_mixed_image_and_text_inputs_become_typed_objects(gateway: Gateway) -> None:
    """The caller embeds `[<png data URL>, "plain text"]` on a Jina deployment.

    Because one input is a `data:` base64 URL, `JinaAIEmbeddingConfig` sends `input: [{"image": "iVBORw0KGgo="},
    {"text": "plain text"}]`: the image loses its `data:image/png;base64,` prefix, the text is wrapped.
    """
    text: Final = marker()
    with peer(openai_list(6)) as provider, gateway.scenario() as scenario:
        model: Final = _jina(scenario, provider, api_key=PROVIDER_KEY)
        served(embed(gateway, model, [_PNG_DATA_URL, text]))
        assert provider.only().body["input"] == [{"image": _PNG_BASE64}, {"text": text}]


def test_jina_text_only_inputs_stay_plain_strings(gateway: Gateway) -> None:
    """The caller embeds two plain strings on a Jina deployment.

    Without any `data:` input the list is sent unchanged: `input: [a, b]` of strings, not `{"text": ...}` objects.
    """
    inputs: Final = [marker(), marker()]
    with peer(openai_list(6)) as provider, gateway.scenario() as scenario:
        model: Final = _jina(scenario, provider, api_key=PROVIDER_KEY)
        served(embed(gateway, model, inputs))
        assert provider.only().body["input"] == inputs


def test_jina_keyless_deployment_sends_no_authorization_header(gateway: Gateway) -> None:
    """A Jina embedding deployment with no `api_key` (a self-hosted Jina server).

    `JinaAIEmbeddingConfig.validate_environment` adds Authorization only when a key exists, so the request must
    carry no `authorization` header at all. UNVERIFIED: that no `JINA_AI_API_KEY` is set in the CI environment.
    """
    with peer(openai_list(6)) as provider, gateway.scenario() as scenario:
        model: Final = _jina(scenario, provider)
        served(embed(gateway, model, [marker()]))
        assert "authorization" not in provider.only().headers


def test_jina_provider_specific_task_field_reaches_the_body(gateway: Gateway) -> None:
    """The caller sends the Jina-only field `task: "retrieval.passage"` next to `input`.

    Jina is not called through the OpenAI SDK, so unknown caller fields are added to the provider body: Jina must
    receive `task: "retrieval.passage"`. UNVERIFIED: read from `add_provider_specific_params_to_optional_params`.
    """
    with peer(openai_list(6)) as provider, gateway.scenario() as scenario:
        model: Final = _jina(scenario, provider, api_key=PROVIDER_KEY)
        served(embed(gateway, model, [marker()], task="retrieval.passage"))
        assert provider.only().body["task"] == "retrieval.passage"


def test_jina_answer_is_returned_with_its_usage(gateway: Gateway) -> None:
    """Jina answers an OpenAI-shaped list with `usage.prompt_tokens: 6` and `total_tokens: 6`.

    The caller must get `object: "list"`, one vector at index 0 and usage `prompt_tokens: 6`.
    """
    with peer(openai_list(6)) as provider, gateway.scenario() as scenario:
        model: Final = _jina(scenario, provider, api_key=PROVIDER_KEY)
        body: Final = served(embed(gateway, model, [marker()]))
        assert body["object"] == "list"
        assert indexes(body) == [0]
        assert object_value(body["usage"])["prompt_tokens"] == 6
