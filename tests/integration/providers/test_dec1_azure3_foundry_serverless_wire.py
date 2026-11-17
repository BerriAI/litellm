"""Azure AI Foundry serverless (MaaS) endpoints and Azure AI Inference api-version handling.

Serverless deployments of Llama, Mistral, Cohere and Phi are `azure_ai/<model>` deployments whose `api_base`
is the endpoint's own host (`https://<name>.<region>.models.ai.azure.com` in production; a loopback peer here).
`AzureAIStudioConfig` (`litellm/llms/azure_ai/chat/transformation.py`) builds the call:

- URL: `{api_base}/chat/completions` (only `*.services.ai.azure.com` gets `/models/chat/completions`).
  `api-version` is added from `api_version` only when `api_base` has none already (`get_complete_url:468`).
- Auth: the key as `Authorization: Bearer <key>` on any host that is not an Azure OpenAI host
  (`validate_environment`, `api_key_header_for_base`).
- Body: OpenAI chat shape with the bare model name, list content flattened to a string.
- Params: `tool_choice` is unsupported for a model whose row lacks `supports_tool_choice` (Phi-4-mini-instruct).
- 422 recovery: Mistral's "unknown field: parameter index is not a valid field" is retried once with `index`
  removed from replayed `tool_calls` (`transform_request_on_unprocessable_entity_error`).
- Embeddings (`AzureAIEmbedding`): text goes to `{api_base}/embeddings` through the OpenAI client, base64 images
  to `{api_base}/images/embeddings` with a Bearer header, and results are re-assembled in input order.
- An `azure_ai/<openai model name>` on a non Foundry base is re-routed as Azure OpenAI
  (`_is_azure_openai_model`).

Prices come from the cost map rows `azure_ai/<model>`.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure_support import (
    FOUNDRY_KEY,
    WEATHER_TOOL,
    always,
    body_of,
    chat_json,
    error_message,
    failing_then,
    foundry,
    json_reply,
    path_of,
    query_of,
    spend_row,
)

FAMILIES: Final = (
    pytest.param("Llama-3.3-70B-Instruct", 7.1e-07, 7.1e-07, id="llama"),
    pytest.param("mistral-small-2503", 1e-07, 3e-07, id="mistral"),
    pytest.param("cohere-command-a", 2.5e-06, 1e-05, id="cohere"),
    pytest.param("Phi-4", 1.25e-07, 5e-07, id="phi"),
)
"""(model, input cost per token, output cost per token) from model_prices_and_context_window.json."""
PNG_DATA_URL: Final = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _answer(model: str) -> Reply:
    return json_reply(chat_json("pong", model=model))


@pytest.mark.parametrize(("model", "input_cost", "output_cost"), FAMILIES)
def test_a_serverless_endpoint_gets_chat_completions_with_a_bearer_key_and_the_bare_model(
    gateway: Gateway, model: str, input_cost: float, output_cost: float
) -> None:
    """An `azure_ai/<family model>` deployment whose `api_base` is the serverless endpoint root (loopback).

    The endpoint must see exactly one `POST /chat/completions` with no query, `Authorization: Bearer
    synthetic-foundry-key`, no `api-key` header, and body `model` equal to the bare model name.
    """
    del input_cost, output_cost
    with foundry(gateway, always(_answer(model)), model=f"azure_ai/{model}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/chat/completions", {})
        assert request.headers.get("authorization") == f"Bearer {FOUNDRY_KEY}"
        assert "api-key" not in request.headers
        assert body_of(request)["model"] == model


@pytest.mark.parametrize(("model", "input_cost", "output_cost"), FAMILIES)
def test_a_serverless_call_is_priced_from_its_family_cost_map_row(
    gateway: Gateway, model: str, input_cost: float, output_cost: float
) -> None:
    """The endpoint reports 11 prompt and 7 completion tokens; the deployment has no custom prices.

    The SpendLogs row must have `spend == 11 * input + 7 * output` from the `azure_ai/<model>` row (for example
    Llama-3.3-70B-Instruct `18 * 7.1e-07 = 1.278e-05`) and `custom_llm_provider == "azure_ai"`.
    """
    with foundry(gateway, always(_answer(model)), model=f"azure_ai/{model}") as rig:
        body: Final = rig.chat().json()
        row: Final = spend_row(body["id"])
        assert float(str(row["spend"])) == pytest.approx(11 * input_cost + 7 * output_cost)
        assert row["custom_llm_provider"] == "azure_ai"


def test_text_content_parts_are_flattened_for_a_serverless_model(gateway: Gateway) -> None:
    """The caller sends user content `[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]` to Llama.

    Foundry serverless rejects list content, so the user message content Llama gets must be the string `ab`.
    UNVERIFIED: the join separator of `convert_content_list_to_str` (empty string assumed).
    """
    model: Final = "Llama-3.3-70B-Instruct"
    with foundry(gateway, always(_answer(model)), model=f"azure_ai/{model}") as rig:
        rig.chat_body(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}
        )
        messages: Final = body_of(rig.one_post())["messages"]
        assert messages == [{"role": "user", "content": "ab"}]


def test_tool_choice_on_a_model_without_tool_choice_support_is_refused_before_the_endpoint(gateway: Gateway) -> None:
    """An `azure_ai/Phi-4-mini-instruct` deployment (cost map row has no `supports_tool_choice`); the caller sends
    `tools` and `tool_choice: auto` without `drop_params`.

    `get_supported_openai_params` leaves `tool_choice` out, so the caller must get HTTP 400 whose message names
    `tool_choice`, and the endpoint must receive no request.
    """
    model: Final = "Phi-4-mini-instruct"
    with foundry(gateway, always(_answer(model)), model=f"azure_ai/{model}") as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="auto")
        assert response.status_code == 400, response.text
        assert "tool_choice" in error_message(response)
        assert rig.posts() == ()


def test_tool_choice_is_dropped_for_that_model_under_drop_params(gateway: Gateway) -> None:
    """Same Phi-4-mini-instruct call with `drop_params: true` in the request.

    The endpoint must get the `tools` unchanged and no `tool_choice` key at all.
    """
    model: Final = "Phi-4-mini-instruct"
    with foundry(gateway, always(_answer(model)), model=f"azure_ai/{model}") as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], tool_choice="auto", drop_params=True)
        assert response.status_code == 200, response.text
        body: Final = body_of(rig.one_post())
        assert body["tools"] == [WEATHER_TOOL] and "tool_choice" not in body


def test_a_mistral_index_rejection_is_retried_without_tool_call_indexes(gateway: Gateway) -> None:
    """A Mistral serverless endpoint answers the first call 422 "unknown field: parameter index is not a valid
    field"; the caller replays an assistant tool call carrying `index: 0`.

    The proxy must retry exactly once, and the second body's assistant `tool_calls[0]` must have no `index` key
    while keeping `id`, `type` and `function`.
    """
    model: Final = "mistral-small-2503"
    rejection: Final = json_reply(
        {"detail": [{"msg": "unknown field: parameter index is not a valid field"}]}, status=422
    )
    tool_call: Final = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
    }
    with foundry(gateway, failing_then([rejection], _answer(model)), model=f"azure_ai/{model}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                ],
                "tools": [WEATHER_TOOL],
            }
        )
        assert response.status_code == 200, response.text
        posts: Final = rig.posts()
        assert len(posts) == 2, posts
        retried: Final = body_of(posts[1])["messages"][1]["tool_calls"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportArgumentType, reportCallIssue]
        assert retried == {key: value for key, value in tool_call.items() if key != "index"}


def test_an_api_version_in_the_api_base_wins_over_the_configured_one(gateway: Gateway) -> None:
    """An Azure AI Inference style `api_base` of `<peer>?api-version=2024-05-01-preview` and `api_version:
    2025-05-01` on an `azure_ai/Llama-3.3-70B-Instruct` deployment.

    `get_complete_url` only adds `api-version` when the base has none, so the query must be exactly
    `{"api-version": "2024-05-01-preview"}` and the path `/chat/completions`.
    """
    model: Final = "Llama-3.3-70B-Instruct"
    with foundry(
        gateway,
        always(_answer(model)),
        model=f"azure_ai/{model}",
        base_path="?api-version=2024-05-01-preview",
        api_version="2025-05-01",
    ) as rig:
        rig.chat()
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/chat/completions", {"api-version": "2024-05-01-preview"})


def test_an_inference_models_base_keeps_the_models_segment(gateway: Gateway) -> None:
    """`api_base` is `<peer>/models` with `api_version: 2024-05-01-preview` (the Azure AI Inference layout).

    The call must go to `POST /models/chat/completions?api-version=2024-05-01-preview` (`add_path_to_api_base`
    appends `/chat/completions` to the given path).
    """
    model: Final = "Phi-4"
    with foundry(
        gateway,
        always(_answer(model)),
        model=f"azure_ai/{model}",
        base_path="/models",
        api_version="2024-05-01-preview",
    ) as rig:
        rig.chat()
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == (
            "/models/chat/completions",
            {"api-version": "2024-05-01-preview"},
        )


def _embedding_peer(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    count: Final = len(body["input"]) if isinstance(body["input"], list) else 1
    first: Final = 0.5 if request.target.split("?", 1)[0].endswith("/images/embeddings") else 0.1
    return json_reply(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": [first, float(i)]} for i in range(count)],
            "model": "embed-v-4-0",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
    )


def test_foundry_text_embeddings_post_to_embeddings_without_an_api_version(gateway: Gateway) -> None:
    """An `azure_ai/embed-v-4-0` deployment with `api_version: 2024-05-01-preview` embeds `["hello"]`.

    `azure_ai_embedding.embedding` never receives `api_version`, so the endpoint must see `POST /embeddings`
    with no query at all (unlike Foundry chat, which adds it), and `Authorization: Bearer synthetic-foundry-key`.
    A Rust port that "fixes" this changes the wire; pinned as current behaviour.
    """
    with foundry(gateway, _embedding_peer, model="azure_ai/embed-v-4-0", api_version="2024-05-01-preview") as rig:
        response: Final = rig.embed(["hello"])
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/embeddings", {})
        assert request.headers.get("authorization") == f"Bearer {FOUNDRY_KEY}"


def test_foundry_image_inputs_go_to_images_embeddings_and_results_keep_input_order(gateway: Gateway) -> None:
    """The caller embeds `["text one", <png data url>, "text two"]` on `azure_ai/embed-v-4-0`.

    Base64 inputs are split off (`AzureAICohereConfig._transform_request`): the endpoint must get one
    `POST /embeddings` with `input == ["text one", "text two"]` and one `POST /images/embeddings` whose `input`
    holds the image, and the caller's `data` must have indexes 0, 1, 2 with the image vector (first value 0.5)
    at index 1. UNVERIFIED: the image request shape (`[{"image": <url>}]` assumed) and the recombined indexes.
    """
    with foundry(gateway, _embedding_peer, model="azure_ai/embed-v-4-0") as rig:
        response: Final = rig.embed(["text one", PNG_DATA_URL, "text two"])
        assert response.status_code == 200, response.text
        posts: Final = {path_of(r): body_of(r) for r in rig.posts()}
        assert set(posts) == {"/embeddings", "/images/embeddings"}, posts
        assert posts["/embeddings"]["input"] == ["text one", "text two"]
        data: Final = response.json()["data"]
        assert [item["index"] for item in data] == [0, 1, 2]
        assert data[1]["embedding"][0] == 0.5


def test_an_openai_model_name_under_azure_ai_on_a_plain_base_is_served_as_azure_openai(gateway: Gateway) -> None:
    """An `azure_ai/gpt-4o` deployment on a loopback base (not `*.services.ai.azure.com`) with `api_version:
    2024-10-21`.

    `_is_azure_openai_model` switches the provider to `azure`, so the peer must see
    `POST /openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21` with the key in `api-key`, not a
    Foundry `/chat/completions` with a Bearer. UNVERIFIED: depends on `get_llm_provider` consulting
    `get_openai_compatible_provider_info` for `azure_ai`.
    """
    with foundry(gateway, always(_answer("gpt-4o")), model="azure_ai/gpt-4o", api_version="2024-10-21") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == (
            "/openai/deployments/gpt-4o/chat/completions",
            {"api-version": "2024-10-21"},
        )
        assert request.headers.get("api-key") == FOUNDRY_KEY
