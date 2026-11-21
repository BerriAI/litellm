"""Azure OpenAI `/v1/responses` and `/v1/embeddings` on the wire.

Responses: `AzureOpenAIResponsesAPIConfig` (`litellm/llms/azure/responses/transformation.py`) picks the URL
from the deployment's versions. `responses_api_version` wins; a dated version older than 2025-03-01, a dated
GA version, or no version at all resolve to `AZURE_DEFAULT_RESPONSES_API_VERSION = "preview"`
(`litellm/constants.py:54`), which selects `/openai/v1/responses?api-version=preview`; a dated preview from
2025-03-01 on keeps the classic `/openai/responses?api-version=<v>` route; the opt-out `none` posts to
`/openai/v1/responses` with no `api-version`. The body `model` is the deployment name with litellm routing
markers (`responses/`, `o_series/`, `gpt5_series/`) stripped (`transformation.py:161`, the `gpt5_series/`
strip landed on 2026-11-12 in #51438), chat-shaped function tools are flattened, and `status` is removed
from message input items Azure rejects. The existing `test_azure_responses_api_version_wire.py` already
pins dated GA, `responses_api_version` and Foundry project bases, so those are not repeated.

Embeddings: the AzureOpenAI SDK posts `/openai/deployments/{deployment}/embeddings?api-version=<v>`;
`test_azure_embedding_wire.py` already pins the dated route, `dimensions` and `encoding_format`. This file
adds credentials, the v1 route, errors, headers and spend.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure_support import (
    API_KEY,
    EMBEDDING_DEPLOYMENT,
    ENTRA_TOKEN,
    GPT5_DEPLOYMENT,
    INPUT_COST,
    OUTPUT_COST,
    PREVIEW_API_VERSION,
    PRICES,
    WEATHER_TOOL,
    always,
    azure,
    azure_error,
    body_of,
    embedding_json,
    error_message,
    json_reply,
    marker,
    path_of,
    query_of,
    responses_json,
    spend_rows_for_group,
)


def _responses(_: Request) -> Reply:
    return json_reply(responses_json("azure responded"))


def _embeddings(_: Request) -> Reply:
    return json_reply(embedding_json([[0.25, 0.5, 0.75]]))


_RESPONSES_MODEL: Final = f"azure/{GPT5_DEPLOYMENT}"
_EMBEDDING_MODEL: Final = f"azure/{EMBEDDING_DEPLOYMENT}"


def test_responses_without_any_api_version_posts_to_the_v1_preview_route(gateway: Gateway) -> None:
    """The deployment has no `api_version` and no `responses_api_version`.

    Azure must see `POST /openai/v1/responses` with query exactly `{"api-version": "preview"}`.
    UNVERIFIED: assumes `AZURE_DEFAULT_RESPONSES_API_VERSION` is unset in CI.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL, api_version=None) as rig:
        assert rig.responses().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/responses", {"api-version": "preview"})


def test_responses_on_a_2025_preview_version_keeps_the_classic_route(gateway: Gateway) -> None:
    """The deployment sets `api_version: 2025-04-01-preview` (not before 2025-03-01, not dated GA).

    `_resolve_responses_api_version` returns it unchanged (`transformation.py:69`), so Azure must see
    `POST /openai/responses?api-version=2025-04-01-preview`, not the v1 route.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL, api_version=PREVIEW_API_VERSION) as rig:
        assert rig.responses().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/responses", {"api-version": PREVIEW_API_VERSION})


def test_responses_on_a_version_older_than_the_responses_api_uses_v1_preview(gateway: Gateway) -> None:
    """The deployment sets `api_version: 2024-08-01-preview`, which predates the Responses API (2025-03-01).

    The old version is discarded (`_predates_responses_api`, `transformation.py:45`), so Azure must see
    `POST /openai/v1/responses?api-version=preview`, never `api-version=2024-08-01-preview`.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL, api_version="2024-08-01-preview") as rig:
        assert rig.responses().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/responses", {"api-version": "preview"})


def test_responses_with_the_api_version_opt_out_sends_no_api_version(gateway: Gateway) -> None:
    """The deployment sets `api_version: none`.

    The opt-out removes the query (`_without_rejected_api_version`, `transformation.py:90`), so Azure must see
    `POST /openai/v1/responses` with no query string at all.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL, api_version="none") as rig:
        assert rig.responses().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/responses", {})


def test_responses_send_the_key_in_the_api_key_header(gateway: Gateway) -> None:
    """A keyed deployment serves one Responses call.

    `base_validate_azure_environment` puts the key in `api-key` (`common_utils.py:1044`); Azure must receive
    `api-key: synthetic-azure-key` and no `Authorization` header.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL) as rig:
        assert rig.responses().status_code == 200
        headers: Final = rig.one_post().headers
        assert headers.get("api-key") == API_KEY
        assert "authorization" not in headers, headers


def test_responses_with_only_an_azure_ad_token_send_a_bearer(gateway: Gateway) -> None:
    """The deployment has no key and `azure_ad_token: synthetic-entra-access-token`.

    With no key `get_azure_ad_token` is used (`common_utils.py:1052`), so Azure must receive
    `Authorization: Bearer synthetic-entra-access-token` and no `api-key`.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL, api_key=None, azure_ad_token=ENTRA_TOKEN) as rig:
        assert rig.responses().status_code == 200
        headers: Final = rig.one_post().headers
        assert headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"
        assert "api-key" not in headers, headers


@pytest.mark.parametrize(
    "model",
    [f"azure/{GPT5_DEPLOYMENT}", f"azure/responses/{GPT5_DEPLOYMENT}", f"azure/gpt5_series/{GPT5_DEPLOYMENT}"],
    ids=["plain", "responses_marker", "gpt5_series_marker"],
)
def test_responses_body_model_is_the_deployment_name_without_routing_markers(gateway: Gateway, model: str) -> None:
    """The deployment is `azure/<marker>gpt-5.4-nano` for no marker, `responses/` and `gpt5_series/`.

    The body `model` Azure receives must be exactly `gpt-5.4-nano`. The `gpt5_series/` strip is new
    (#51438, 2026-11-12); before it Azure got `gpt5_series/gpt-5.4-nano` and answered DeploymentNotFound.
    """
    with azure(gateway, _responses, model=model) as rig:
        assert rig.responses().status_code == 200
        assert body_of(rig.one_post())["model"] == GPT5_DEPLOYMENT


def test_responses_flatten_chat_shaped_function_tools(gateway: Gateway) -> None:
    """The caller sends a chat-shaped tool `{"type": "function", "function": {"name", "description", "parameters"}}`.

    Azure Responses needs flat tools (`transformation.py:192`), so it must receive
    `{"type": "function", "name": "get_weather", "description": ..., "parameters": ...}` with no nested `function`.
    """
    function: Final = WEATHER_TOOL["function"]
    with azure(gateway, _responses, model=_RESPONSES_MODEL) as rig:
        assert rig.responses(tools=[WEATHER_TOOL]).status_code == 200
        tools: Final = body_of(rig.one_post())["tools"]
        assert tools == [{"type": "function", **function}]  # pyright: ignore[reportGeneralTypeIssues]


def test_responses_strip_status_from_message_input_items(gateway: Gateway) -> None:
    """The caller replays a message item `{"type": "message", "role": "user", "content": <text>, "status": "completed"}`.

    Azure rejects `status` on message items (`_AZURE_REJECTED_INPUT_FIELDS`, `transformation.py:99`), so the item
    Azure receives must have no `status` key and keep `type`, `role` and `content`.
    """
    text: Final = marker()
    item: Final = {"type": "message", "role": "user", "content": text, "status": "completed"}
    with azure(gateway, _responses, model=_RESPONSES_MODEL) as rig:
        assert rig.responses(input=[item]).status_code == 200
        sent: Final = body_of(rig.one_post())["input"]
        assert sent == [{"type": "message", "role": "user", "content": text}], sent


def test_responses_answer_returns_the_output_text_and_usage(gateway: Gateway) -> None:
    """Azure answers a completed response with output text `azure responded` and 11 input, 7 output tokens.

    The caller must get `object: response`, `status: completed`, that text in `output[0].content[0].text`, and
    `usage.input_tokens == 11`, `usage.output_tokens == 7`.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL) as rig:
        response: Final = rig.responses()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert (body["object"], body["status"]) == ("response", "completed")
        assert body["output"][0]["content"][0]["text"] == "azure responded"
        assert (body["usage"]["input_tokens"], body["usage"]["output_tokens"]) == (11, 7)


def test_responses_invalid_encrypted_content_explains_encrypted_content_affinity(gateway: Gateway) -> None:
    """Azure answers 400 `code: invalid_encrypted_content` (reasoning items from another resource).

    The caller must get HTTP 400 whose message names `encrypted_content_affinity`, the router check that keeps
    follow-ups on the resource that encrypted them (`exception_mapping_utils.py:2131`). UNVERIFIED: that the
    Responses path maps through `_map_azure_exception` rather than a generic mapper.
    """
    reply: Final = azure_error(400, "invalid_encrypted_content", "The encrypted content could not be verified.")
    with azure(gateway, always(reply), model=_RESPONSES_MODEL, max_retries=0) as rig:
        response: Final = rig.responses(num_retries=0)
        assert response.status_code == 400, response.text
        assert "encrypted_content_affinity" in error_message(response)


def test_a_responses_call_writes_one_aresponses_spend_row(gateway: Gateway) -> None:
    """A Responses call on a deployment priced 0.001 input and 0.002 output per token; Azure reports 11/7 tokens.

    One SpendLogs row must land for the group with `call_type: aresponses`, `custom_llm_provider: azure`,
    11 prompt and 7 completion tokens and `spend = 0.025`.
    """
    with azure(gateway, _responses, model=_RESPONSES_MODEL, **PRICES) as rig:
        assert rig.responses().status_code == 200
        row: Final = spend_rows_for_group(rig.model)[0]
        assert row["call_type"] == "aresponses"
        assert row["custom_llm_provider"] == "azure"
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_embeddings_with_only_an_azure_ad_token_send_a_bearer(gateway: Gateway) -> None:
    """An embedding deployment with no key and `azure_ad_token: synthetic-entra-access-token`.

    Azure must receive `POST /openai/deployments/text-embedding-3-small/embeddings` with
    `Authorization: Bearer synthetic-entra-access-token` and no `api-key`.
    """
    with azure(gateway, _embeddings, model=_EMBEDDING_MODEL, api_key=None, azure_ad_token=ENTRA_TOKEN) as rig:
        response: Final = rig.embed([marker()])
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert path_of(request) == f"/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings"
        assert request.headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"
        assert "api-key" not in request.headers, request.headers


def test_embeddings_on_the_v1_api_version_post_to_openai_v1_embeddings(gateway: Gateway) -> None:
    """An embedding deployment with `api_version: v1`.

    `aembedding` builds the client with `opt_out_client_api_version` (`azure.py:698`), so a v1 version uses the
    plain OpenAI client: Azure must see `POST /openai/v1/embeddings` with no `api-version` query and the
    deployment name as body `model`.
    """
    with azure(gateway, _embeddings, model=_EMBEDDING_MODEL, api_version="v1") as rig:
        assert rig.embed([marker()]).status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/openai/v1/embeddings", {})
        assert body_of(request)["model"] == EMBEDDING_DEPLOYMENT


def test_embeddings_return_the_vectors_and_prompt_token_usage(gateway: Gateway) -> None:
    """Azure answers one 3 float vector and `usage.prompt_tokens: 6`.

    The caller must get `object: list`, one `embedding` item at index 0 with `[0.25, 0.5, 0.75]`, and
    `usage.prompt_tokens == 6`.
    """
    with azure(gateway, _embeddings, model=_EMBEDDING_MODEL) as rig:
        response: Final = rig.embed([marker()])
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["object"] == "list"
        assert body["data"][0]["embedding"] == [0.25, 0.5, 0.75]
        assert body["usage"]["prompt_tokens"] == 6


def test_an_embedding_429_reaches_the_caller_as_429(gateway: Gateway) -> None:
    """Azure answers the embedding call with 429; no retries at either layer.

    The caller must get HTTP 429 and Azure exactly one POST.
    """
    with azure(gateway, always(azure_error(429, "429", "rate limited")), model=_EMBEDDING_MODEL, max_retries=0) as rig:
        response: Final = rig.embed([marker()], num_retries=0)
        assert response.status_code == 429, response.text
        assert len(rig.posts()) == 1


def test_embedding_responses_carry_the_azure_region_header(gateway: Gateway) -> None:
    """Azure answers the embedding call with `x-ms-region: Sweden Central`.

    `process_azure_headers` (`common_utils.py:252`) keeps provider headers, so the caller must get
    `llm_provider-x-ms-region: Sweden Central`.
    """
    reply: Final = json_reply(embedding_json([[0.1]]), headers={"x-ms-region": "Sweden Central"})
    with azure(gateway, always(reply), model=_EMBEDDING_MODEL) as rig:
        response: Final = rig.embed([marker()])
        assert response.status_code == 200, response.text
        assert response.headers.get("llm_provider-x-ms-region") == "Sweden Central"


def test_an_embedding_call_writes_one_priced_spend_row(gateway: Gateway) -> None:
    """An embedding deployment priced at 0.001 per input token; Azure reports 6 prompt tokens.

    One SpendLogs row must land with `call_type: aembedding`, `custom_llm_provider: azure`, `prompt_tokens: 6`,
    `completion_tokens: 0` and `spend = 6 * 0.001 = 0.006`.
    """
    with azure(gateway, _embeddings, model=_EMBEDDING_MODEL, input_cost_per_token=INPUT_COST) as rig:
        response: Final = rig.embed([marker()])
        assert response.status_code == 200, response.text
        row: Final = spend_rows_for_group(rig.model)[0]
        assert row["call_type"] == "aembedding"
        assert row["custom_llm_provider"] == "azure"
        assert (row["prompt_tokens"], row["completion_tokens"]) == (6, 0)
        assert float(str(row["spend"])) == pytest.approx(6 * INPUT_COST)
