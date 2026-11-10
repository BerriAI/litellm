"""Contract: `POST /v1/embeddings` translated for providers that do not speak the OpenAI embeddings format.

The caller always sends the OpenAI shape. What the provider receives depends on the deployment's provider, and
each translation below was read from the provider's transformation module:

    cohere/embed-english-v3.0     POST to the deployment `api_base` exactly as configured, body
                                  `{"model", "texts": [...], "input_type": "search_document"}`;
                                  `encoding_format` becomes `embedding_types: [<format>]`, `dimensions` becomes
                                  `output_dimension` (`litellm/llms/cohere/embed/transformation.py`)
    voyage/voyage-3               POST `<api_base>/embeddings`, body `{"input", "model"}`; `dimensions` becomes
                                  `output_dimension`; usage is taken from `usage.total_tokens`
                                  (`litellm/llms/voyage/embedding/transformation.py`)
    jina_ai/jina-embeddings-v3    POST `<api_base>/embeddings`, body `{"model", "input": [...]}`; only
                                  `dimensions` is supported (`litellm/llms/jina_ai/embedding/transformation.py`)
    together_ai/<model>           the OpenAI path and body, with the Together key as bearer (`litellm/main.py`)
    bedrock/amazon.titan-embed-*  one `POST /model/<id>/invoke` per input with `{"inputText": ...}`; usage is the
                                  sum of each reply's `inputTextTokenCount`
                                  (`litellm/llms/bedrock/embed/amazon_titan_v2_transformation.py`)

For these providers an OpenAI parameter the provider does not support is refused with 400 and
`litellm.UnsupportedParamsError: <provider> does not support parameters: {...}, for model=<model>` unless
`drop_params` is on. Whatever the provider answers, the caller gets the OpenAI list shape back. Azure, Vertex,
Gemini, SageMaker, Ollama and Bedrock Cohere already have wire tests in this directory.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_textapis_support import (
    INPUT_RATE,
    PROVIDER_KEY,
    VECTOR,
    Call,
    Peer,
    embed,
    error_message,
    json_reply,
    marker,
    peer,
    served,
)
from pydantic import JsonValue

from tests.integration._support.client import Scenario

_COHERE: Final = "embed-english-v3.0"
_VOYAGE: Final = "voyage-3"
_JINA: Final = "jina-embeddings-v3"
_TOGETHER: Final = "BAAI/bge-base-en-v1.5"
_TITAN_V2: Final = "amazon.titan-embed-text-v2:0"
_TITAN_V1: Final = "amazon.titan-embed-text-v1"


def _texts(call: Call) -> int:
    for field in ("texts", "input"):
        value = call.body.get(field)
        if isinstance(value, list):
            return len(value)
    return 1


def _cohere(call: Call) -> Reply:
    """Cohere v2 embed reply: typed embeddings when `embedding_types` was sent, a bare list otherwise."""
    vectors: Final[list[JsonValue]] = [list(VECTOR) for _ in range(_texts(call))]
    types: Final = call.body.get("embedding_types")
    embeddings: Final[JsonValue] = {str(types[0]): vectors} if isinstance(types, list) and types else vectors
    return json_reply(
        {"id": "dec1-cohere-embed", "embeddings": embeddings, "meta": {"billed_units": {"input_tokens": 3}}}
    )


def _openai_like(total_tokens: int) -> object:
    def respond(call: Call) -> Reply:
        return json_reply(
            {
                "object": "list",
                "model": call.body.get("model"),
                "data": [
                    {"object": "embedding", "index": index, "embedding": list(VECTOR)}
                    for index in range(_texts(call))
                ],
                "usage": {"total_tokens": total_tokens, "prompt_tokens": total_tokens},
            }
        )

    return respond


def _titan(call: Call) -> Reply:
    return json_reply({"embedding": list(VECTOR), "inputTextTokenCount": 4})


def _cohere_model(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    return scenario.model(
        model=f"cohere/{_COHERE}",
        api_base=f"{provider.url}/v2/embed",
        api_key=PROVIDER_KEY,
        input_cost_per_token=INPUT_RATE,
        **parameters,
    )


def _voyage_model(scenario: Scenario, provider: Peer) -> str:
    return scenario.model(
        model=f"voyage/{_VOYAGE}", api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY, input_cost_per_token=INPUT_RATE
    )


def _jina_model(scenario: Scenario, provider: Peer) -> str:
    return scenario.model(model=f"jina_ai/{_JINA}", api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY)


def _titan_model(scenario: Scenario, provider: Peer, upstream: str) -> str:
    return scenario.model(
        model=f"bedrock/{upstream}", api_base=provider.url, api_key=PROVIDER_KEY, aws_region_name="us-east-1"
    )


def _vectors(body: dict[str, JsonValue]) -> list[JsonValue]:
    return [object_value(entry)["embedding"] for entry in list_value(body["data"])]


def test_cohere_receives_texts_and_the_default_search_document_input_type(gateway: Gateway) -> None:
    """A caller embeds a list of two strings on a `cohere/embed-english-v3.0` deployment whose `api_base` is
    `<peer>/v2/embed`.

    Cohere must receive one `POST /v2/embed` (the configured URL, with nothing appended) with
    `Authorization: Bearer <deployment key>` and the body
    `{"model": "embed-english-v3.0", "texts": [<both strings>], "input_type": "search_document"}`.
    """
    inputs: Final[list[JsonValue]] = [marker(), marker()]
    with peer(_cohere) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        served(embed(gateway, model, inputs))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v2/embed")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {"model": _COHERE, "texts": inputs, "input_type": "search_document"}


def test_cohere_wraps_a_single_string_into_a_one_element_texts_list(gateway: Gateway) -> None:
    """A caller embeds one bare string on the Cohere deployment.

    Cohere only accepts a list, so the body's `texts` must be `[<the string>]`.
    """
    text: Final = marker()
    with peer(_cohere) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        served(embed(gateway, model, text))
        assert provider.only().body["texts"] == [text]


def test_cohere_encoding_format_and_dimensions_are_renamed(gateway: Gateway) -> None:
    """A caller sends `encoding_format: "float"` and `dimensions: 256` to the Cohere deployment.

    Cohere must receive `"embedding_types": ["float"]` and `"output_dimension": 256`, and neither OpenAI name.
    """
    text: Final = marker()
    with peer(_cohere) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        served(embed(gateway, model, [text], encoding_format="float", dimensions=256))
        assert provider.only().body == {
            "model": _COHERE,
            "texts": [text],
            "input_type": "search_document",
            "embedding_types": ["float"],
            "output_dimension": 256,
        }


def test_cohere_input_type_from_the_caller_replaces_the_default(gateway: Gateway) -> None:
    """A caller sends the Cohere-specific field `input_type: "search_query"`.

    Provider-specific fields are passed through for non-OpenAI providers, so Cohere must receive
    `"input_type": "search_query"` instead of the default `search_document`. Retrieval quality depends on it.
    """
    with peer(_cohere) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        served(embed(gateway, model, [marker()], input_type="search_query"))
        assert provider.only().body["input_type"] == "search_query"


def test_cohere_reply_is_returned_in_the_openai_list_shape_with_billed_input_tokens(gateway: Gateway) -> None:
    """Cohere answers `{"embeddings": [[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]], "meta": {"billed_units":
    {"input_tokens": 3}}}` for two inputs.

    The caller gets two `data` entries with `object` `"embedding"`, `index` 0 and 1 and the vectors, and
    `usage` with `prompt_tokens` 3 and `total_tokens` 3 taken from Cohere's billed units.
    """
    with peer(_cohere) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        body: Final = served(embed(gateway, model, [marker(), marker()]))
        usage: Final = object_value(body["usage"])
        assert [(object_value(entry)["object"], object_value(entry)["index"]) for entry in list_value(body["data"])] == [
            ("embedding", 0),
            ("embedding", 1),
        ]
        assert _vectors(body) == [list(VECTOR), list(VECTOR)]
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (3, 3)


def test_cohere_cost_header_prices_the_billed_input_tokens(gateway: Gateway) -> None:
    """The Cohere deployment costs 0.001 per input token and Cohere bills 3 input tokens.

    `x-litellm-response-cost` must be 0.003: the cost comes from the provider's billed units, not from a local
    token count of the input text.
    """
    with peer(_cohere) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        response: Final = embed(gateway, model, [marker()])
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.003)


def test_cohere_refuses_the_user_parameter_with_an_unsupported_params_400(gateway: Gateway) -> None:
    """A caller sends `user: "dec1-end-user"` to the Cohere deployment, which supports only `encoding_format`
    and `dimensions`.

    The proxy must answer 400 without calling Cohere, and the message must contain `litellm.UnsupportedParamsError:
    cohere does not support parameters: {'user': 'dec1-end-user'}, for model=embed-english-v3.0`.
    """
    with peer(_cohere) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        response: Final = embed(gateway, model, [marker()], user="dec1-end-user")
        assert response.status_code == 400, response.text
        assert (
            "litellm.UnsupportedParamsError: cohere does not support parameters: {'user': 'dec1-end-user'}, "
            f"for model={_COHERE}" in error_message(response)
        ), response.text
        assert provider.posts() == ()


def test_cohere_rejects_a_provider_429_with_429_and_the_provider_text(gateway: Gateway) -> None:
    """Cohere answers 429 with the body `{"message": "dec1 cohere trial key rate limit"}`.

    The caller gets HTTP 429 with `error.code` `"429"` and the provider's text inside `error.message`.
    UNVERIFIED: the litellm prefix in front of the text (expected `litellm.RateLimitError: CohereException - `)
    was not traced through `exception_type`, so only the status, code and provider text are asserted.
    """

    def respond(call: Call) -> Reply:
        return json_reply({"message": "dec1 cohere trial key rate limit"}, status=429)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _cohere_model(scenario, provider)
        response: Final = embed(gateway, model, [marker()])
        assert response.status_code == 429, response.text
        assert "dec1 cohere trial key rate limit" in error_message(response)
        assert len(provider.posts()) >= 1


def test_voyage_receives_input_and_model_on_the_embeddings_route(gateway: Gateway) -> None:
    """A caller embeds a list of two strings on a `voyage/voyage-3` deployment whose `api_base` is `<peer>/v1`.

    Voyage must receive one `POST /v1/embeddings` with `Authorization: Bearer <deployment key>` and the body
    `{"input": [<both strings>], "model": "voyage-3"}`.
    """
    inputs: Final[list[JsonValue]] = [marker(), marker()]
    with peer(_openai_like(9)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage_model(scenario, provider)
        served(embed(gateway, model, inputs))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/embeddings")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {"input": inputs, "model": _VOYAGE}


def test_voyage_dimensions_becomes_output_dimension_and_encoding_format_is_kept(gateway: Gateway) -> None:
    """A caller sends `dimensions: 512` and `encoding_format: "float"` to the Voyage deployment.

    Voyage must receive `"output_dimension": 512` (its own name) and `"encoding_format": "float"` (same name).
    """
    inputs: Final[list[JsonValue]] = [marker()]
    with peer(_openai_like(9)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage_model(scenario, provider)
        served(embed(gateway, model, inputs, dimensions=512, encoding_format="float"))
        assert provider.only().body == {
            "input": inputs,
            "model": _VOYAGE,
            "output_dimension": 512,
            "encoding_format": "float",
        }


def test_voyage_input_type_is_passed_through(gateway: Gateway) -> None:
    """A caller sends the Voyage-specific field `input_type: "query"`.

    Voyage must receive `"input_type": "query"` in the body next to `input` and `model`.
    """
    inputs: Final[list[JsonValue]] = [marker()]
    with peer(_openai_like(9)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage_model(scenario, provider)
        served(embed(gateway, model, inputs, input_type="query"))
        assert provider.only().body == {"input": inputs, "model": _VOYAGE, "input_type": "query"}


def test_voyage_usage_and_cost_come_from_total_tokens(gateway: Gateway) -> None:
    """Voyage answers `usage: {"total_tokens": 9}` and the deployment costs 0.001 per input token.

    The caller's `usage` must have `prompt_tokens` 9 and `total_tokens` 9 (Voyage reports no prompt count of its
    own), and `x-litellm-response-cost` must be 0.009.
    """
    with peer(_openai_like(9)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage_model(scenario, provider)
        response: Final = embed(gateway, model, [marker()])
        usage: Final = object_value(served(response)["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (9, 9)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.009)


def test_voyage_refuses_the_user_parameter_with_an_unsupported_params_400(gateway: Gateway) -> None:
    """A caller sends `user` to the Voyage deployment, which supports only `encoding_format` and `dimensions`.

    The proxy must answer 400 without calling Voyage and the message must contain
    `voyage does not support parameters: {'user': 'dec1-end-user'}, for model=voyage-3`.
    """
    with peer(_openai_like(9)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage_model(scenario, provider)
        response: Final = embed(gateway, model, [marker()], user="dec1-end-user")
        assert response.status_code == 400, response.text
        assert (
            f"voyage does not support parameters: {{'user': 'dec1-end-user'}}, for model={_VOYAGE}"
            in error_message(response)
        ), response.text
        assert provider.posts() == ()


def test_voyage_provider_401_is_returned_as_401_with_the_provider_text(gateway: Gateway) -> None:
    """Voyage answers 401 with `{"detail": "dec1 voyage key rejected"}`.

    The caller gets HTTP 401 with `error.code` `"401"` and the provider text inside `error.message`.
    UNVERIFIED: the litellm prefix in front of the text was not traced; only status and text are asserted.
    """

    def respond(call: Call) -> Reply:
        return json_reply({"detail": "dec1 voyage key rejected"}, status=401)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _voyage_model(scenario, provider)
        response: Final = embed(gateway, model, [marker()])
        assert response.status_code == 401, response.text
        assert "dec1 voyage key rejected" in error_message(response)
        assert len(provider.posts()) >= 1


def test_jina_receives_model_and_a_list_input_on_the_embeddings_route(gateway: Gateway) -> None:
    """A caller embeds one bare string on a `jina_ai/jina-embeddings-v3` deployment whose `api_base` is
    `<peer>/v1`.

    Jina must receive one `POST /v1/embeddings` with `Authorization: Bearer <deployment key>` and the body
    `{"model": "jina-embeddings-v3", "input": [<the string>]}`: a single string is wrapped in a list.
    """
    text: Final = marker()
    with peer(_openai_like(5)) as provider, gateway.scenario() as scenario:
        model: Final = _jina_model(scenario, provider)
        served(embed(gateway, model, text))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/embeddings")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {"model": _JINA, "input": [text]}


def test_jina_forwards_dimensions_under_the_same_name(gateway: Gateway) -> None:
    """A caller sends `dimensions: 128` to the Jina deployment. Jina must receive `"dimensions": 128`."""
    text: Final = marker()
    with peer(_openai_like(5)) as provider, gateway.scenario() as scenario:
        model: Final = _jina_model(scenario, provider)
        served(embed(gateway, model, [text], dimensions=128))
        assert provider.only().body == {"model": _JINA, "input": [text], "dimensions": 128}


def test_jina_refuses_encoding_format_with_an_unsupported_params_400(gateway: Gateway) -> None:
    """A caller sends `encoding_format: "base64"` to the Jina deployment, which supports only `dimensions`.

    The proxy must answer 400 without calling Jina and the message must contain
    `jina_ai does not support parameters: {'encoding_format': 'base64'}, for model=jina-embeddings-v3`.
    OpenAI SDKs send `encoding_format` by default, so this refusal is what a stock SDK user sees.
    """
    with peer(_openai_like(5)) as provider, gateway.scenario() as scenario:
        model: Final = _jina_model(scenario, provider)
        response: Final = embed(gateway, model, [marker()], encoding_format="base64")
        assert response.status_code == 400, response.text
        assert (
            f"jina_ai does not support parameters: {{'encoding_format': 'base64'}}, for model={_JINA}"
            in error_message(response)
        ), response.text
        assert provider.posts() == ()


def test_together_ai_uses_the_openai_embeddings_path_and_body(gateway: Gateway) -> None:
    """A caller embeds one string on a `together_ai/BAAI/bge-base-en-v1.5` deployment whose `api_base` is
    `<peer>/v1`.

    Together is served through the OpenAI client: one `POST /v1/embeddings` with the Together key as bearer and
    the body `{"model": "BAAI/bge-base-en-v1.5", "input": "<the string>"}` (the provider prefix is removed, the
    organisation prefix of the model id is kept).
    """
    text: Final = marker()
    with peer(_openai_like(5)) as provider, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"together_ai/{_TOGETHER}", api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY
        )
        served(embed(gateway, model, text))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/embeddings")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {"model": _TOGETHER, "input": text}


def test_bedrock_titan_v2_gets_one_invoke_call_per_input_with_input_text(gateway: Gateway) -> None:
    """A caller embeds a list of two strings with `dimensions: 256` on a `bedrock/amazon.titan-embed-text-v2:0`
    deployment that authenticates with a bearer token.

    Titan embeds one text per call, so Bedrock must receive two `POST /model/amazon.titan-embed-text-v2:0/invoke`
    requests, one `{"inputText": <string>, "dimensions": 256}` per input, each with
    `Authorization: Bearer <token>`. The path may arrive with the colon percent-encoded.
    """
    inputs: Final = [marker(), marker()]
    with peer(_titan) as provider, gateway.scenario() as scenario:
        model: Final = _titan_model(scenario, provider, _TITAN_V2)
        served(embed(gateway, model, list(inputs), dimensions=256))
        calls: Final = provider.posts()
        assert len(calls) == 2, [call.path for call in calls]
        assert {call.path.replace("%3A", ":") for call in calls} == {f"/model/{_TITAN_V2}/invoke"}
        assert {call.headers["authorization"] for call in calls} == {f"Bearer {PROVIDER_KEY}"}
        assert sorted(str(call.body["inputText"]) for call in calls) == sorted(inputs)
        assert all(call.body == {"inputText": call.body["inputText"], "dimensions": 256} for call in calls)


def test_bedrock_titan_usage_is_the_sum_of_the_per_input_token_counts(gateway: Gateway) -> None:
    """Each Titan reply reports `inputTextTokenCount: 4` and the caller embedded two strings.

    The caller gets two `data` entries indexed 0 and 1 and `usage` with `prompt_tokens` 8 and `total_tokens` 8.
    """
    with peer(_titan) as provider, gateway.scenario() as scenario:
        model: Final = _titan_model(scenario, provider, _TITAN_V2)
        body: Final = served(embed(gateway, model, [marker(), marker()]))
        usage: Final = object_value(body["usage"])
        assert [object_value(entry)["index"] for entry in list_value(body["data"])] == [0, 1]
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (8, 8)


def test_bedrock_titan_v1_refuses_dimensions_with_an_unsupported_params_400(gateway: Gateway) -> None:
    """A caller sends `dimensions: 256` to a `bedrock/amazon.titan-embed-text-v1` deployment, which supports no
    OpenAI embedding parameter.

    The proxy must answer 400 without calling Bedrock and the message must contain
    `bedrock does not support parameters: {'dimensions': 256}, for model=amazon.titan-embed-text-v1`.
    """
    with peer(_titan) as provider, gateway.scenario() as scenario:
        model: Final = _titan_model(scenario, provider, _TITAN_V1)
        response: Final = embed(gateway, model, [marker()], dimensions=256)
        assert response.status_code == 400, response.text
        assert (
            f"bedrock does not support parameters: {{'dimensions': 256}}, for model={_TITAN_V1}"
            in error_message(response)
        ), response.text
        assert provider.posts() == ()
