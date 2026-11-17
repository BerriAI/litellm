"""Contract: `/v1/embeddings` on OpenAI-compatible and self-hosted embedding servers.

These providers are the ones most often run inside a customer network, so their URL joining and header rules
matter more than for hosted APIs. Each translation below was read from the provider module:

    mistral/<m>             OpenAI client; only `encoding_format` is supported, other OpenAI params are refused
    nvidia_nim/<org>/<m>    OpenAI client; `input_type` and `truncate` travel in `extra_body`, so they arrive as
                            top-level body fields (`litellm/llms/nvidia_nim/embed.py`)
    together_ai/<org>/<m>   OpenAI client; OpenAI params are passed through
    infinity/<m>            POST `<api_base>/embeddings` (joined unless present), `{"input", "model", ...}`,
                            `dimensions` renamed `output_dimension` (`litellm/llms/infinity/embedding`)
    hosted_vllm/<m>         POST `<api_base>/embeddings`, a string input is wrapped into a list, a keyless
                            deployment sends no Authorization (`litellm/llms/hosted_vllm/embedding`)
    huggingface/<m>         POST to the deployment `api_base` verbatim (a TEI server), `{"inputs": [...]}`; the
                            caller's `input_type` names the HF task and skips the Hub lookup
                            (`litellm/llms/huggingface/embedding/handler.py`)
    ollama/<m>              POST `<api_base>/api/embed`, `{"model", "input": [...]}`; `truncate`, `keep_alive`,
                            `dimensions` and `options` stay top level, any other provider field goes into
                            `options`; usage is `prompt_eval_count` (`litellm/llms/ollama/completion/handler.py`)

Nested token arrays are decoded to text by the proxy for every provider here except `hosted_vllm`, which accepts
token ids (`LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS`).
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
    embed,
    error_message,
    indexes,
    json_reply,
    marker,
    ollama_embed,
    openai_list,
    peer,
    served,
    vectors,
)
from pydantic import JsonValue

from tests.integration._support.client import Scenario

_NIM: Final = "nvidia/nv-embedqa-e5-v5"
_TOGETHER: Final = "BAAI/bge-large-en-v1.5"
_INFINITY: Final = "BAAI/bge-small-en-v1.5"
_VLLM: Final = "intfloat/e5-mistral-7b-instruct"
_HF: Final = "BAAI/bge-base-en-v1.5"
_OLLAMA: Final = "nomic-embed-text"


def _model(scenario: Scenario, litellm_model: str, api_base: str, **parameters: JsonValue) -> str:
    return scenario.model(model=litellm_model, api_base=api_base, **parameters)


def _tei(call: Call) -> Reply:
    """A Text Embeddings Inference `/embed` answer: one bare float vector per input."""
    count: Final = len(call.body["inputs"]) if isinstance(call.body.get("inputs"), list) else 1
    return json_reply([list(VECTOR) for _ in range(count)])


def _hf(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    return _model(scenario, f"huggingface/{_HF}", f"{provider.url}/embed", api_key=PROVIDER_KEY, **parameters)


def test_mistral_refuses_dimensions_without_calling_mistral(gateway: Gateway) -> None:
    """The caller sends `dimensions: 256` to a `mistral/mistral-embed` deployment.

    `MistralEmbeddingConfig` supports only `encoding_format`, so the proxy must answer 400 with
    `mistral does not support parameters: {'dimensions': 256}, for model=mistral-embed` and Mistral must receive
    nothing. UNVERIFIED: the 400 status (UnsupportedParamsError is built with 500 and mapped to 400 by the proxy,
    as the sibling Cohere/Voyage refusals show).
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, "mistral/mistral-embed", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        response: Final = embed(gateway, model, [marker()], dimensions=256)
        assert response.status_code == 400, response.text
        assert "mistral does not support parameters: {'dimensions': 256}, for model=mistral-embed" in error_message(
            response
        )
        assert provider.posts() == ()


def test_mistral_refuses_the_user_parameter(gateway: Gateway) -> None:
    """The caller sends `user: "dec1-end-user"` to a Mistral embedding deployment.

    `user` is not in Mistral's supported list, so the caller must get 400 naming
    `{'user': 'dec1-end-user'}` and no request reaches Mistral. Callers that always send `user` need `drop_params`.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, "mistral/mistral-embed", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        response: Final = embed(gateway, model, [marker()], user="dec1-end-user")
        assert response.status_code == 400, response.text
        assert "{'user': 'dec1-end-user'}" in error_message(response)
        assert provider.posts() == ()


def test_mistral_drops_an_unsupported_parameter_when_the_deployment_sets_drop_params(gateway: Gateway) -> None:
    """The same `dimensions: 256` on a Mistral deployment with `drop_params: true`.

    The call must succeed and Mistral must receive a body without `dimensions`.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(
            scenario, "mistral/mistral-embed", f"{provider.url}/v1", api_key=PROVIDER_KEY, drop_params=True
        )
        served(embed(gateway, model, [marker()], dimensions=256))
        assert "dimensions" not in provider.only().body


def test_nvidia_nim_puts_input_type_and_truncate_at_the_top_of_the_body(gateway: Gateway) -> None:
    """The caller sends NIM retrieval fields `input_type: "query"` and `truncate: "END"` on an
    `nvidia_nim/nvidia/nv-embedqa-e5-v5` deployment.

    NIM goes through the OpenAI client with these fields in `extra_body`, so NIM must receive one
    `POST /v1/embeddings` whose body has `model: "nvidia/nv-embedqa-e5-v5"`, the input, `input_type: "query"` and
    `truncate: "END"` as top-level fields (not nested under `extra_body`). NIM e5 models refuse a request without
    `input_type`.
    """
    text: Final = marker()
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"nvidia_nim/{_NIM}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, [text], input_type="query", truncate="END"))
        call: Final = provider.only()
        assert call.path == "/v1/embeddings"
        assert (call.body["model"], call.body["input"]) == (_NIM, [text])
        assert (call.body["input_type"], call.body["truncate"]) == ("query", "END")
        assert "extra_body" not in call.body


def test_nvidia_nim_forwards_user_and_dimensions(gateway: Gateway) -> None:
    """The caller sends `user: "dec1-nim-user"` and `dimensions: 384` to a NIM embedding deployment.

    Both are in NIM's supported list (`encoding_format`, `user`, `dimensions`), so NIM must receive them with the
    same names and values.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"nvidia_nim/{_NIM}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, [marker()], user="dec1-nim-user", dimensions=384))
        body: Final = provider.only().body
        assert (body["user"], body["dimensions"]) == ("dec1-nim-user", 384)


def test_nvidia_nim_nested_token_arrays_are_decoded_to_text(gateway: Gateway) -> None:
    """The caller sends two token arrays to a NIM deployment.

    `nvidia_nim` is not a token-array provider, so NIM must receive `input: ["The quick brown base", "I like"]`.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"nvidia_nim/{_NIM}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, TOKEN_INPUT, input_type="passage"))
        assert provider.only().body["input"] == DECODED_TOKEN_TEXTS


def test_together_forwards_dimensions_and_decodes_token_arrays(gateway: Gateway) -> None:
    """The caller sends two token arrays with `dimensions: 512` to a `together_ai/BAAI/bge-large-en-v1.5` deployment.

    Together must receive `input: ["The quick brown base", "I like"]` (decoded by the proxy) and `dimensions: 512`
    (OpenAI-compatible providers pass OpenAI params through). UNVERIFIED: the dimensions pass-through, read from the
    `openai_compatible_providers` branch of `get_optional_params_embeddings`.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"together_ai/{_TOGETHER}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, TOKEN_INPUT, dimensions=512))
        body: Final = provider.only().body
        assert (body["input"], body["dimensions"]) == (DECODED_TOKEN_TEXTS, 512)


def test_infinity_joins_embeddings_onto_the_api_base(gateway: Gateway) -> None:
    """An `infinity/BAAI/bge-small-en-v1.5` deployment with `api_base: <peer>/v1` embeds one string.

    Infinity must receive `POST /v1/embeddings` with body exactly `{"input": "<the string>", "model":
    "BAAI/bge-small-en-v1.5"}` (a single string is not wrapped into a list) and the deployment key as bearer.
    """
    text: Final = marker()
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"infinity/{_INFINITY}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, text))
        call: Final = provider.only()
        assert call.path == "/v1/embeddings"
        assert call.body == {"input": text, "model": _INFINITY}
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_infinity_renames_dimensions_and_keeps_encoding_format(gateway: Gateway) -> None:
    """The caller sends `dimensions: 128` and `encoding_format: "float"` to an Infinity deployment.

    `InfinityEmbeddingConfig.map_openai_params` sends `output_dimension: 128` and `encoding_format: "float"`, and no
    `dimensions` field.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"infinity/{_INFINITY}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, [marker()], dimensions=128, encoding_format="float"))
        body: Final = provider.only().body
        assert (body["output_dimension"], body["encoding_format"]) == (128, "float")
        assert "dimensions" not in body


def test_infinity_forwards_the_modality_field(gateway: Gateway) -> None:
    """The caller sends the Infinity field `modality: "text"`.

    `modality` is in Infinity's supported list, so the body must carry `modality: "text"` at the top level.
    UNVERIFIED: whether it arrives top level or inside an `extra_body` that the handler unwraps (#51335).
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"infinity/{_INFINITY}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, [marker()], modality="text"))
        assert provider.only().body["modality"] == "text"


def test_infinity_keyless_deployment_sends_no_bearer_none(gateway: Gateway) -> None:
    """An Infinity deployment with no `api_key` (the usual self-hosted setup).

    The provider should receive no Authorization header, or at least never the literal `Bearer None`.
    """
    pytest.skip(
        "BUG: InfinityEmbeddingConfig.validate_environment formats a missing key as 'Authorization: Bearer None'"
    )
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"infinity/{_INFINITY}", f"{provider.url}/v1", api_key=None)
        served(embed(gateway, model, [marker()]))
        assert provider.only().headers.get("authorization") != "Bearer None"


def test_infinity_usage_is_read_from_prompt_and_total_tokens(gateway: Gateway) -> None:
    """Infinity answers `usage: {"prompt_tokens": 8, "total_tokens": 8}`.

    The caller's usage must be `prompt_tokens: 8`, `total_tokens: 8`, and the `data` list is relayed as sent.
    """
    with peer(openai_list(8)) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"infinity/{_INFINITY}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        body: Final = served(embed(gateway, model, [marker(), marker()]))
        usage: Final = object_value(body["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (8, 8)
        assert indexes(body) == [0, 1]


def test_hosted_vllm_wraps_a_string_input_and_strips_the_provider_prefix(gateway: Gateway) -> None:
    """A `hosted_vllm/intfloat/e5-mistral-7b-instruct` deployment with `api_base: <peer>/v1` embeds one string.

    vLLM must receive `POST /v1/embeddings` with body `{"model": "intfloat/e5-mistral-7b-instruct", "input":
    ["<the string>"]}`: the string is wrapped in a list and the `hosted_vllm/` prefix is gone.
    """
    text: Final = marker()
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"hosted_vllm/{_VLLM}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, text))
        call: Final = provider.only()
        assert call.path == "/v1/embeddings"
        assert call.body == {"model": _VLLM, "input": [text]}
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_hosted_vllm_keyless_deployment_sends_no_authorization(gateway: Gateway) -> None:
    """A hosted vLLM embedding deployment with no `api_key`.

    `HostedVLLMEmbeddingConfig` substitutes `fake-api-key` and then leaves Authorization out, so vLLM must receive no
    `authorization` header. UNVERIFIED: that `HOSTED_VLLM_API_KEY` is unset in the CI environment.

    Test mistake fixed (run 44176682961): `Scenario.model` fills in `api_key: integration-provider-key` unless told
    otherwise (tests/integration/_support/client.py:293), so the deployment was not keyless; it now sets
    `api_key=None`.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"hosted_vllm/{_VLLM}", f"{provider.url}/v1", api_key=None)
        served(embed(gateway, model, [marker()]))
        assert "authorization" not in provider.only().headers


def test_hosted_vllm_forwards_dimensions_user_and_encoding_format(gateway: Gateway) -> None:
    """The caller sends `dimensions: 64`, `user: "dec1-vllm-user"` and `encoding_format: "float"`.

    All three are in the hosted vLLM supported list and are forwarded unchanged under the same names.
    """
    with peer(openai_list()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"hosted_vllm/{_VLLM}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        served(embed(gateway, model, [marker()], dimensions=64, user="dec1-vllm-user", encoding_format="float"))
        body: Final = provider.only().body
        assert (body["dimensions"], body["user"], body["encoding_format"]) == (64, "dec1-vllm-user", "float")


def test_huggingface_tei_receives_inputs_at_the_api_base_verbatim(gateway: Gateway) -> None:
    """A `huggingface/BAAI/bge-base-en-v1.5` deployment with `api_base: <peer>/embed` (a TEI server); the caller
    sends two strings and `input_type: "feature-extraction"`.

    The HF handler posts to the `api_base` unchanged with body exactly `{"inputs": [a, b]}`; `input_type` names the
    task, is removed from the body, and avoids a lookup on huggingface.co. The bearer is the deployment key.
    """
    inputs: Final = [marker(), marker()]
    with peer(_tei) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        served(embed(gateway, model, inputs, input_type="feature-extraction"))
        call: Final = provider.only()
        assert call.path == "/embed"
        assert call.body == {"inputs": inputs}
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_huggingface_tei_vectors_are_returned_in_input_order(gateway: Gateway) -> None:
    """TEI answers a bare list of two float vectors.

    The caller must get `object: "list"` with entries indexed 0 and 1 holding those vectors.
    """
    with peer(_tei) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        body: Final = served(embed(gateway, model, [marker(), marker()], input_type="feature-extraction"))
        assert body["object"] == "list"
        assert indexes(body) == [0, 1]
        assert vectors(body) == [list(VECTOR), list(VECTOR)]


def test_huggingface_sentence_similarity_task_sends_source_and_sentences(gateway: Gateway) -> None:
    """The caller sends three strings with `input_type: "sentence-similarity"`.

    The first string is the source: HF must receive `{"inputs": {"source_sentence": a, "sentences": [b, c]}}`.
    """
    first, second, third = marker(), marker(), marker()

    def respond(call: Call) -> Reply:
        return json_reply([0.9, 0.1])

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        served(embed(gateway, model, [first, second, third], input_type="sentence-similarity"))
        assert provider.only().body == {"inputs": {"source_sentence": first, "sentences": [second, third]}}


def test_huggingface_sentence_similarity_with_one_input_is_refused(gateway: Gateway) -> None:
    """The caller sends a single string with `input_type: "sentence-similarity"`.

    `_transform_input_on_pipeline_tag` raises HuggingFaceError 400 `sentence-similarity requires 2+ sentences`
    before any request; the caller must get 400 with that text and the server receives nothing.
    """
    with peer(_tei) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        response: Final = embed(gateway, model, [marker()], input_type="sentence-similarity")
        assert response.status_code == 400, response.text
        assert "sentence-similarity requires 2+ sentences" in error_message(response)
        assert provider.posts() == ()


def test_huggingface_unknown_task_name_is_refused_before_any_request(gateway: Gateway) -> None:
    """The caller sends `input_type: "summarization"`, which is not an embedding task.

    `get_hf_task_embedding_for_model` raises `Invalid task_type=summarization` and nothing is sent. UNVERIFIED: the
    status (a bare Exception, expected 500).
    """
    with peer(_tei) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        response: Final = embed(gateway, model, [marker()], input_type="summarization")
        assert response.status_code >= 400, response.text
        assert "Invalid task_type=summarization" in error_message(response)
        assert provider.posts() == ()


def test_huggingface_refuses_the_user_parameter(gateway: Gateway) -> None:
    """The caller sends `user: "dec1-hf-user"` to a HuggingFace embedding deployment.

    HuggingFace has no embedding param mapping and is not OpenAI-compatible, so the proxy must answer 400 with
    `Setting {'user': 'dec1-hf-user'} is not supported by huggingface` and send nothing.
    """
    with peer(_tei) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        response: Final = embed(gateway, model, [marker()], input_type="feature-extraction", user="dec1-hf-user")
        assert response.status_code == 400, response.text
        assert "Setting {'user': 'dec1-hf-user'} is not supported by huggingface" in error_message(response)
        assert provider.posts() == ()


def test_huggingface_error_body_keeps_the_tei_status(gateway: Gateway) -> None:
    """TEI answers HTTP 413 `{"error": "Input validation error: inputs too long", "error_type": "Validation"}`.

    The caller should get 413 with that text.
    """
    pytest.skip("BUG: the HF embedding handler raises HuggingFaceError(500) for any body with an `error` key")

    def respond(call: Call) -> Reply:
        return json_reply({"error": "Input validation error: inputs too long", "error_type": "Validation"}, 413)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        response: Final = embed(gateway, model, [marker()], input_type="feature-extraction")
        assert response.status_code == 413, response.text
        assert "Input validation error: inputs too long" in error_message(response)


def test_huggingface_embedding_usage_has_no_completion_tokens(gateway: Gateway) -> None:
    """A TEI embedding answer; the proxy counts prompt tokens with its tokenizer because TEI reports none.

    An embedding has no output, so `usage.completion_tokens` should be 0 and `total_tokens` equal `prompt_tokens`.
    """
    pytest.skip("BUG: _process_embedding_response sets completion_tokens to the prompt token count")
    with peer(_tei) as provider, gateway.scenario() as scenario:
        model: Final = _hf(scenario, provider)
        usage: Final = object_value(served(embed(gateway, model, [marker()], input_type="feature-extraction"))["usage"])
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == usage["prompt_tokens"]


def test_ollama_posts_a_list_input_to_api_embed(gateway: Gateway) -> None:
    """An `ollama/nomic-embed-text` deployment at the bare peer URL embeds one string.

    Ollama must receive `POST /api/embed` with body exactly `{"model": "nomic-embed-text", "input": ["<string>"]}`.
    """
    text: Final = marker()
    with peer(ollama_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"ollama/{_OLLAMA}", provider.url)
        served(embed(gateway, model, text))
        call: Final = provider.only()
        assert call.path == "/api/embed"
        assert call.body == {"model": _OLLAMA, "input": [text]}


def test_ollama_keeps_keep_alive_and_dimensions_top_level_and_nests_other_fields_in_options(
    gateway: Gateway,
) -> None:
    """The caller sends `dimensions: 256`, `keep_alive: "5m"` and the runtime field `num_ctx: 2048`.

    `_prepare_ollama_embedding_payload` keeps `dimensions` and `keep_alive` at the top and moves any other field
    into `options`, so Ollama must receive `dimensions: 256`, `keep_alive: "5m"`, `options: {"num_ctx": 2048}`.
    """
    with peer(ollama_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"ollama/{_OLLAMA}", provider.url)
        served(embed(gateway, model, [marker()], dimensions=256, keep_alive="5m", num_ctx=2048))
        body: Final = provider.only().body
        assert (body["dimensions"], body["keep_alive"]) == (256, "5m")
        assert body["options"] == {"num_ctx": 2048}


def test_ollama_refuses_encoding_format(gateway: Gateway) -> None:
    """The caller sends `encoding_format: "float"` (what OpenAI SDKs send by default) to an Ollama deployment.

    The Ollama branch accepts only `dimensions` among OpenAI params, so the caller must get 400 containing
    `Setting {'encoding_format': 'float'} is not supported by ollama` and Ollama receives nothing.
    """
    with peer(ollama_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"ollama/{_OLLAMA}", provider.url)
        response: Final = embed(gateway, model, [marker()], encoding_format="float")
        assert response.status_code == 400, response.text
        assert "Setting {'encoding_format': 'float'} is not supported by ollama" in error_message(response)
        assert provider.posts() == ()


def test_ollama_refuses_a_flat_token_array_with_400(gateway: Gateway) -> None:
    """The caller sends one flat token array to an Ollama deployment.

    The proxy decodes only nested arrays, so `main.embedding` raises BadRequestError `Invalid input for ollama
    embeddings. input=[791, 4062, 14198, 2385]`; the caller must get 400 with that text and Ollama receives nothing.
    """
    with peer(ollama_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"ollama/{_OLLAMA}", provider.url)
        response: Final = embed(gateway, model, FLAT_TOKEN_INPUT)
        assert response.status_code == 400, response.text
        assert "Invalid input for ollama embeddings. input=[791, 4062, 14198, 2385]" in error_message(response)
        assert provider.posts() == ()


def test_ollama_nested_token_arrays_are_decoded_to_text(gateway: Gateway) -> None:
    """The caller sends two nested token arrays to an Ollama deployment.

    The proxy decodes them first, so Ollama must receive `input: ["The quick brown base", "I like"]`.
    """
    with peer(ollama_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"ollama/{_OLLAMA}", provider.url)
        served(embed(gateway, model, TOKEN_INPUT))
        assert provider.only().body["input"] == DECODED_TOKEN_TEXTS


def test_ollama_usage_is_prompt_eval_count_and_vectors_keep_order(gateway: Gateway) -> None:
    """Ollama answers two vectors and `prompt_eval_count: 4`.

    The caller must get entries indexed 0 and 1 and usage `prompt_tokens: 4`, `completion_tokens: 0`,
    `total_tokens: 4`; the cost header on a deployment priced 0.001 per input token is 0.004.
    """
    with peer(ollama_embed(4)) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"ollama/{_OLLAMA}", provider.url, input_cost_per_token=INPUT_RATE)
        response: Final = embed(gateway, model, [marker(), marker()])
        body: Final = served(response)
        assert indexes(body) == [0, 1]
        usage: Final = object_value(body["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (4, 0, 4)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.004)


def test_ollama_without_prompt_eval_count_estimates_prompt_tokens(gateway: Gateway) -> None:
    """Ollama answers without `prompt_eval_count` (older servers).

    The handler estimates with the proxy tokenizer over the joined inputs, so the caller's `prompt_tokens` must be
    a positive integer equal to `total_tokens`, never 0 and never missing. UNVERIFIED: the exact count depends on
    the tokenizer, so only positivity is pinned.

    BUG (run 44176682961): the call fails with 400 `'Logging' object has no attribute 'debug'`. When
    `prompt_eval_count` is missing and an encoding is available, `_process_ollama_embedding_response` calls
    `.debug(...)` on the litellm `Logging` object it was handed (litellm/llms/ollama/completion/handler.py:79-82 on
    main), which has no such method, so the estimate is never returned.
    """
    pytest.skip(
        "BUG: an Ollama embedding without prompt_eval_count fails with 'Logging' object has no attribute 'debug'"
    )
    with peer(ollama_embed(None)) as provider, gateway.scenario() as scenario:
        model: Final = _model(scenario, f"ollama/{_OLLAMA}", provider.url)
        usage: Final = object_value(served(embed(gateway, model, [f"hello world {marker()}"]))["usage"])
        prompt_tokens: Final = usage["prompt_tokens"]
        assert isinstance(prompt_tokens, int) and prompt_tokens > 0, usage
        assert usage["total_tokens"] == prompt_tokens
