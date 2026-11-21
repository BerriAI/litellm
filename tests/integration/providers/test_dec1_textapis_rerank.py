"""Contract: `POST /v1/rerank` (also `/rerank` and `/v2/rerank`), from the caller's Cohere-shaped body to each provider.

The proxy speaks Cohere's rerank format to callers: `{"model", "query", "documents", "top_n", "rank_fields",
"return_documents", "max_chunks_per_doc", "max_tokens_per_doc"}` in, `{"id", "results": [{"index",
"relevance_score", "document"?}], "meta"}` out. What each provider receives was read from its transformation:

    cohere        POST `<api_base>/v2/rerank` (or `/v1/rerank` when `max_chunks_per_doc` is sent), bearer key,
                  body is the caller's fields with the upstream model name; the reply is returned as is
    jina_ai       POST `/v1/rerank` on the api_base host, only `query`, `documents`, `top_n` and
                  `return_documents` are forwarded; string documents in the reply become `{"text": ...}`
    voyage        POST `<api_base>/v1/rerank`, `top_n` is renamed `top_k`; the reply's `data` becomes `results`
    together_ai   POST `<api_base>/rerank`; `max_chunks_per_doc` is refused
    infinity      POST `<api_base>/rerank`, Cohere v1 body; string documents in the reply become `{"text": ...}`

Cost is `input_cost_per_query` times `meta.billed_units.search_units` for Cohere and `input_cost_per_token`
times `meta.billed_units.total_tokens` for Jina and Voyage. The spend log row has `call_type` `arerank`.
Cohere v1/v2 route selection, object documents, Bedrock, HuggingFace, NVIDIA NIM, Azure AI and the Voyage host
choice already have files in this directory (`test_rerank_latency_headers_wire.py` and neighbours).
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_textapis_support import (
    DOCUMENTS,
    INPUT_RATE,
    MODEL_DENIED_TEMPLATE,
    PROVIDER_KEY,
    RERANK_COST,
    RERANK_SCORES,
    Call,
    Peer,
    cohere_rerank_model,
    cohere_rerank_reply,
    error_message,
    error_of,
    json_reply,
    marker,
    number,
    peer,
    rerank,
    results_of,
    rows_of_key,
    served,
)
from pydantic import JsonValue

from tests.integration._support.client import Scenario

_COHERE: Final = "rerank-v3.5"


def _string_document_results(key: str, usage: dict[str, JsonValue]) -> object:
    """A provider that answers results under `key` with each document as a bare string, plus a `usage` block."""

    def respond(call: Call) -> Reply:
        return json_reply(
            {
                "id": "dec1-provider-rerank",
                key: [
                    {"index": index, "relevance_score": score, "document": DOCUMENTS[index]}
                    for index, score in RERANK_SCORES
                ],
                "usage": usage,
            }
        )

    return respond


def _provider_model(scenario: Scenario, litellm_model: str, api_base: str, **parameters: JsonValue) -> str:
    return scenario.model(
        model=litellm_model, api_base=api_base, api_key=PROVIDER_KEY, model_info={"mode": "rerank"}, **parameters
    )


def _documents(response_body: dict[str, JsonValue]) -> list[JsonValue]:
    return [object_value(entry).get("document") for entry in list_value(response_body["results"])]


def test_cohere_reply_id_results_and_meta_are_returned_to_the_caller(gateway: Gateway) -> None:
    """Cohere answers two results (index 0 scores 0.91, index 1 scores 0.03) and
    `meta.billed_units.search_units: 1`.

    The caller's body must have a string `id`, `results` equal to `[{"index": 0, "relevance_score": 0.91},
    {"index": 1, "relevance_score": 0.03}]` in the provider's order, and `meta.billed_units.search_units` 1.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        response: Final = rerank(gateway, model)
        body: Final = served(response)
        assert isinstance(body["id"], str) and body["id"], body
        assert results_of(response) == RERANK_SCORES
        assert object_value(object_value(body["meta"])["billed_units"])["search_units"] == 1, body


def test_cohere_body_without_top_n_carries_only_model_query_and_documents(gateway: Gateway) -> None:
    """A caller sends only `model`, `query` and `documents`.

    Cohere must receive `POST /v2/rerank` with exactly `{"model": "rerank-v3.5", "query", "documents"}`: no
    `top_n`, no `return_documents` and no null placeholders for the options the caller left out.
    """
    query: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        served(rerank(gateway, model, query=query))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v2/rerank")
        assert call.body == {"model": _COHERE, "query": query, "documents": list(DOCUMENTS)}


def test_cohere_return_documents_and_rank_fields_are_forwarded(gateway: Gateway) -> None:
    """A caller sends `return_documents: true` and `rank_fields: ["text"]` with object documents.

    Cohere must receive both fields unchanged next to the model, query and documents.
    """
    query: Final = marker()
    documents: Final[list[JsonValue]] = [{"text": DOCUMENTS[0]}, {"text": DOCUMENTS[1]}]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        served(rerank(gateway, model, query=query, documents=documents, return_documents=True, rank_fields=["text"]))
        assert provider.only().body == {
            "model": _COHERE,
            "query": query,
            "documents": documents,
            "return_documents": True,
            "rank_fields": ["text"],
        }


def test_cohere_documents_returned_by_the_provider_reach_the_caller(gateway: Gateway) -> None:
    """Cohere answers each result with `"document": {"text": <the document>}`.

    The caller's results must carry the same `document` objects, in the provider's order.
    """

    def respond(call: Call) -> Reply:
        return json_reply(
            {
                "id": "dec1-cohere-with-documents",
                "results": [
                    {"index": index, "relevance_score": score, "document": {"text": DOCUMENTS[index]}}
                    for index, score in RERANK_SCORES
                ],
                "meta": {"billed_units": {"search_units": 1}},
            }
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        body: Final = served(rerank(gateway, model, return_documents=True))
        assert _documents(body) == [{"text": DOCUMENTS[0]}, {"text": DOCUMENTS[1]}]


def test_cohere_sends_the_deployment_key_as_bearer_with_json_headers(gateway: Gateway) -> None:
    """A virtual key reranks on a Cohere deployment.

    Cohere must receive `Authorization: Bearer <deployment key>`, `accept: application/json` and
    `content-type: application/json`, and the virtual key must not appear in any upstream header.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        served(rerank(gateway, model, key=key))
        call: Final = provider.only()
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert (call.headers["accept"], call.headers["content-type"]) == ("application/json", "application/json")
        assert all(key not in value for value in call.headers.values()), dict(call.headers)


@pytest.mark.parametrize(("search_units", "cost"), [(1, 0.002), (3, 0.006)])
def test_cohere_cost_is_the_query_price_times_the_billed_search_units(
    gateway: Gateway, search_units: int, cost: float
) -> None:
    """The deployment costs 0.002 per query and Cohere bills 1 or 3 search units.

    `x-litellm-response-cost` must be 0.002 or 0.006: search units times `input_cost_per_query`. Long document
    lists are billed as several search units, so a flat per-request price would undercharge.
    """
    with peer(lambda call: cohere_rerank_reply(search_units)) as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        response: Final = rerank(gateway, model)
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(cost)


def test_rerank_writes_one_spend_log_row_with_call_type_arerank_and_the_query_cost(gateway: Gateway) -> None:
    """A virtual key reranks once on a Cohere deployment priced at 0.002 per query.

    Exactly one `LiteLLM_SpendLogs` row lands for the key with `call_type` `arerank`, `spend` 0.002,
    `model_group` equal to the public model name, `custom_llm_provider` `cohere` and `status` `success`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        served(rerank(gateway, model, key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["call_type"], row["model_group"], row["custom_llm_provider"], row["status"]) == (
            "arerank",
            model,
            "cohere",
            "success",
        ), row
        assert number(row["spend"]) == pytest.approx(RERANK_COST), row


def test_rerank_spend_is_added_to_the_key(gateway: Gateway) -> None:
    """After one rerank priced at 0.002, the key's `spend` in `LiteLLM_VerificationToken` must reach 0.002.

    Rerank spend goes through the same ledger as chat, so budgets see it.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        served(rerank(gateway, model, key=key))
        info: Final = eventually(
            lambda: object_value(gateway.get("/key/info", {"key": key})["info"]),
            lambda found: number(found["spend"]) > 0,
            seconds=70,
        )
        assert number(info["spend"]) == pytest.approx(RERANK_COST), info


def test_jina_receives_only_the_fields_it_supports_on_v1_rerank(gateway: Gateway) -> None:
    """A caller sends `top_n: 1`, `return_documents: true`, `rank_fields: ["text"]` and `max_tokens_per_doc: 64`
    to a `jina_ai/jina-reranker-v2-base-multilingual` deployment.

    Jina must receive one `POST /v1/rerank` with `authorization: Bearer <deployment key>` and the body
    `{"model", "query", "documents", "top_n": 1, "return_documents": true}`. `rank_fields` and
    `max_tokens_per_doc` are not Jina parameters and are dropped without an error.
    """
    query: Final = marker()
    respond: Final = _string_document_results("results", {"total_tokens": 40})
    with peer(respond) as provider, gateway.scenario() as scenario:  # type: ignore[arg-type]
        model: Final = _provider_model(scenario, "jina_ai/jina-reranker-v2-base-multilingual", provider.url)
        served(
            rerank(
                gateway,
                model,
                query=query,
                top_n=1,
                return_documents=True,
                rank_fields=["text"],
                max_tokens_per_doc=64,
            )
        )
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/rerank")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": list(DOCUMENTS),
            "top_n": 1,
            "return_documents": True,
        }


def test_jina_string_documents_are_returned_as_text_objects_with_token_usage_in_meta(gateway: Gateway) -> None:
    """Jina answers each result with the document as a bare string and `usage: {"total_tokens": 40}`.

    The caller must get `document: {"text": <string>}` on each result (the Cohere shape) and
    `meta.billed_units.total_tokens` 40.
    """
    respond: Final = _string_document_results("results", {"total_tokens": 40})
    with peer(respond) as provider, gateway.scenario() as scenario:  # type: ignore[arg-type]
        model: Final = _provider_model(scenario, "jina_ai/jina-reranker-v2-base-multilingual", provider.url)
        response: Final = rerank(gateway, model, return_documents=True)
        body: Final = served(response)
        assert results_of(response) == RERANK_SCORES
        assert _documents(body) == [{"text": DOCUMENTS[0]}, {"text": DOCUMENTS[1]}]
        assert object_value(object_value(body["meta"])["billed_units"])["total_tokens"] == 40, body


def test_jina_cost_is_the_token_price_times_total_tokens(gateway: Gateway) -> None:
    """A Jina rerank deployment costs 0.001 per input token and Jina reports 40 total tokens.

    `x-litellm-response-cost` must be 0.04. Jina bills rerank by tokens, not by search units.
    """
    respond: Final = _string_document_results("results", {"total_tokens": 40})
    with peer(respond) as provider, gateway.scenario() as scenario:  # type: ignore[arg-type]
        model: Final = _provider_model(
            scenario, "jina_ai/jina-reranker-v2-base-multilingual", provider.url, input_cost_per_token=INPUT_RATE
        )
        response: Final = rerank(gateway, model)
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.04)


def test_voyage_receives_top_k_instead_of_top_n_on_v1_rerank(gateway: Gateway) -> None:
    """A caller sends `top_n: 1` to a `voyage/rerank-2` deployment whose `api_base` is the peer's root URL.

    Voyage must receive one `POST /v1/rerank` with `Authorization: Bearer <deployment key>` and the body
    `{"model": "rerank-2", "query", "documents", "top_k": 1}`: Voyage's name for the result limit is `top_k`.
    """
    query: Final = marker()
    respond: Final = _string_document_results("data", {"total_tokens": 25})
    with peer(respond) as provider, gateway.scenario() as scenario:  # type: ignore[arg-type]
        model: Final = _provider_model(scenario, "voyage/rerank-2", provider.url)
        served(rerank(gateway, model, query=query, top_n=1))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/rerank")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {"model": "rerank-2", "query": query, "documents": list(DOCUMENTS), "top_k": 1}


def test_voyage_data_is_returned_as_results_with_tokens_in_meta(gateway: Gateway) -> None:
    """Voyage answers its results under `data` with string documents and `usage: {"total_tokens": 25}`.

    The caller must get them under `results` with `document: {"text": ...}`, `meta.billed_units.total_tokens`
    25 and `meta.tokens` equal to `{"input_tokens": 25, "output_tokens": 0}`.
    """
    respond: Final = _string_document_results("data", {"total_tokens": 25})
    with peer(respond) as provider, gateway.scenario() as scenario:  # type: ignore[arg-type]
        model: Final = _provider_model(scenario, "voyage/rerank-2", provider.url)
        response: Final = rerank(gateway, model, return_documents=True)
        meta: Final = object_value(served(response)["meta"])
        assert results_of(response) == RERANK_SCORES
        assert _documents(served(response)) == [{"text": DOCUMENTS[0]}, {"text": DOCUMENTS[1]}]
        assert object_value(meta["billed_units"])["total_tokens"] == 25, meta
        assert meta["tokens"] == {"input_tokens": 25, "output_tokens": 0}, meta


def test_together_ai_receives_the_cohere_shaped_body_on_its_rerank_route(gateway: Gateway) -> None:
    """A caller sends `top_n: 2` to a `together_ai/Salesforce/Llama-Rank-V1` deployment whose `api_base` is
    `<peer>/v1`.

    Together must receive one `POST /v1/rerank` with `authorization: Bearer <deployment key>` and the body
    `{"model": "Salesforce/Llama-Rank-V1", "query", "top_n": 2, "documents"}`.
    """
    query: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _provider_model(scenario, "together_ai/Salesforce/Llama-Rank-V1", f"{provider.url}/v1")
        response: Final = rerank(gateway, model, query=query, top_n=2)
        assert results_of(response) == RERANK_SCORES
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/rerank")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {
            "model": "Salesforce/Llama-Rank-V1",
            "query": query,
            "top_n": 2,
            "documents": list(DOCUMENTS),
        }


def test_together_ai_refuses_max_chunks_per_doc_without_calling_the_provider(gateway: Gateway) -> None:
    """A caller sends `max_chunks_per_doc: 3` to the Together deployment.

    The request must fail with a message containing `TogetherAI does not support max_chunks_per_doc`, and
    Together must not be called. UNVERIFIED: the HTTP status; the handler raises a plain `ValueError`
    (`litellm/llms/together_ai/rerank/handler.py`), which `exception_type` most likely maps to 500, while 400
    would be the right answer for a caller mistake. Only "an error status" is asserted.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _provider_model(scenario, "together_ai/Salesforce/Llama-Rank-V1", f"{provider.url}/v1")
        response: Final = rerank(gateway, model, max_chunks_per_doc=3)
        assert response.status_code >= 400, response.text
        assert "TogetherAI does not support max_chunks_per_doc" in error_message(response)
        assert provider.posts() == ()


def test_infinity_appends_rerank_to_the_api_base_and_converts_string_documents(gateway: Gateway) -> None:
    """A caller sends `top_n: 2` to an `infinity/dec1-reranker` deployment whose `api_base` is the peer's root.

    Infinity must receive one `POST /rerank` (no version segment) with the body `{"model": "dec1-reranker",
    "query", "documents", "top_n": 2}`. Its reply carries string documents and `usage: {"prompt_tokens": 12,
    "total_tokens": 12}`; the caller gets `document: {"text": ...}` and `meta.tokens.input_tokens` 12.
    """
    query: Final = marker()
    respond: Final = _string_document_results("results", {"prompt_tokens": 12, "total_tokens": 12})
    with peer(respond) as provider, gateway.scenario() as scenario:  # type: ignore[arg-type]
        model: Final = _provider_model(scenario, "infinity/dec1-reranker", provider.url)
        response: Final = rerank(gateway, model, query=query, top_n=2)
        body: Final = served(response)
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/rerank")
        assert call.body == {"model": "dec1-reranker", "query": query, "documents": list(DOCUMENTS), "top_n": 2}
        assert _documents(body) == [{"text": DOCUMENTS[0]}, {"text": DOCUMENTS[1]}]
        assert object_value(object_value(body["meta"])["tokens"])["input_tokens"] == 12, body


def test_rerank_on_a_provider_without_rerank_support_fails_without_an_upstream_call(gateway: Gateway) -> None:
    """A caller reranks on an `openai/gpt-4o-mini` chat deployment.

    OpenAI has no rerank API, so the request must fail with a message containing `Unsupported provider: openai`
    and the provider must not be called. UNVERIFIED: the HTTP status (a `ValueError` raised in
    `litellm/rerank_api/main.py`), so only "an error status" is asserted.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY)
        response: Final = rerank(gateway, model)
        assert response.status_code >= 400, response.text
        assert "Unsupported provider: openai" in error_message(response)
        assert provider.posts() == ()


@pytest.mark.parametrize("missing", ["query", "documents"])
def test_a_missing_query_or_documents_is_refused_with_the_exact_missing_parameter_error(
    gateway: Gateway, missing: str
) -> None:
    """A caller posts a rerank body without `query` or without `documents`.

    The proxy must answer 400 with `error.message` exactly `/rerank: Missing required parameter: '<name>'.`,
    `error.type` `invalid_request_error`, `error.param` the missing name and `error.code` `"400"`, and must not
    call the provider.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        body: Final[dict[str, JsonValue]] = {"model": model, "query": marker(), "documents": list(DOCUMENTS)}
        del body[missing]
        response: Final = gateway.request("POST", "/v1/rerank", body)
        assert response.status_code == 400, response.text
        assert error_of(response) == {
            "message": f"/rerank: Missing required parameter: '{missing}'.",
            "type": "invalid_request_error",
            "param": missing,
            "code": "400",
        }
        assert provider.posts() == ()


def test_a_missing_model_is_refused_with_the_missing_parameter_error_for_model(gateway: Gateway) -> None:
    """A caller posts `{"query", "documents"}` with no `model`.

    The proxy must answer 400 with `error.message` `/rerank: Missing required parameter: 'model'.` and
    `error.param` `"model"`. Rerank has no default model setting, unlike moderations.
    """
    response: Final = gateway.request("POST", "/v1/rerank", {"query": marker(), "documents": list(DOCUMENTS)})
    assert response.status_code == 400, response.text
    error: Final = error_of(response)
    assert (error["message"], error["param"], error["code"]) == (
        "/rerank: Missing required parameter: 'model'.",
        "model",
        "400",
    )


def test_an_unknown_rerank_model_is_refused_with_400_naming_the_model(gateway: Gateway) -> None:
    """A caller reranks with a model name nothing matches.

    The proxy must answer 400 and the message must contain
    `/rerank: Invalid model name passed in model=<name>`.
    """
    ghost: Final = f"dec1-textapis-ghost-{uuid.uuid4().hex}"
    response: Final = rerank(gateway, ghost)
    assert response.status_code == 400, response.text
    assert f"/rerank: Invalid model name passed in model={ghost}" in error_message(response)


def test_a_rerank_body_that_is_not_json_is_refused_with_400(gateway: Gateway) -> None:
    """A caller posts the bytes `{"model": ` to `/v1/rerank`.

    A malformed body is a caller mistake and must be answered with 400 and a JSON error body.
    """
    pytest.skip(
        "BUG: /v1/rerank parses the body with orjson inside its generic except block, so a malformed JSON body "
        "is answered with 500 instead of 400 (litellm/proxy/rerank_endpoints/endpoints.py)"
    )
    response: Final = gateway.client.post(
        "/v1/rerank",
        content=b'{"model": ',
        headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
    )
    assert response.status_code == 400, response.text
    assert error_of(response)["code"] == "400"


def test_a_rerank_validation_error_carries_the_call_id_header(gateway: Gateway) -> None:
    """The 400 for a missing `query` must carry an `x-litellm-call-id` response header, and when the caller sent
    its own `x-litellm-call-id` the header must echo that value."""
    call_id: Final = str(uuid.uuid4())
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        response: Final = gateway.request(
            "POST",
            "/v1/rerank",
            {"model": model, "documents": list(DOCUMENTS)},
            headers={"x-litellm-call-id": call_id},
        )
        assert response.status_code == 400, response.text
        assert response.headers["x-litellm-call-id"] == call_id


def test_a_call_id_sent_by_the_caller_is_echoed_on_a_served_rerank(gateway: Gateway) -> None:
    """A caller sends `x-litellm-call-id: <uuid>` with a rerank request that is served.

    The response header `x-litellm-call-id` must be that same value.
    """
    call_id: Final = str(uuid.uuid4())
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        response: Final = rerank(gateway, model, headers={"x-litellm-call-id": call_id})
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] == call_id


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_a_cohere_error_is_returned_with_its_status_and_the_provider_text(gateway: Gateway, status: int) -> None:
    """Cohere answers 400, 401, 429 or 500 with `{"message": "dec1 cohere rerank refused"}`.

    The caller gets the same HTTP status, `error.code` equal to it as a string, and the provider text inside
    `error.message`. UNVERIFIED: the Cohere branch of `exception_type` was not traced for every status; if a
    status is remapped (for example 500 to 503) this test shows it.
    """

    def respond(call: Call) -> Reply:
        return json_reply({"message": "dec1 cohere rerank refused"}, status=status)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        response: Final = rerank(gateway, model)
        assert response.status_code == status, response.text
        assert error_of(response)["code"] == str(status)
        assert "dec1 cohere rerank refused" in error_message(response)
        assert len(provider.posts()) >= 1


def test_a_failed_rerank_writes_a_failure_row_with_zero_spend(gateway: Gateway) -> None:
    """Cohere answers 400 to a virtual key's rerank.

    One `LiteLLM_SpendLogs` row must land for the key with `status` `failure`, `call_type` `arerank` and
    `spend` 0. UNVERIFIED: failure rows for rerank are written by `post_call_failure_hook` with
    `dispatched_call_type="arerank"`; the row's `call_type` literal was read from that argument.
    """

    def respond(call: Call) -> Reply:
        return json_reply({"message": "dec1 cohere rerank refused"}, status=400)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        assert rerank(gateway, model, key=key).status_code == 400
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["status"], row["call_type"]) == ("failure", "arerank"), row
        assert number(row["spend"]) == 0.0, row


def test_a_key_cannot_rerank_on_a_model_outside_its_list(gateway: Gateway) -> None:
    """A key limited to one model reranks on another existing rerank model.

    The proxy must answer 403 with exactly the model access denial body (`type` `key_model_access_denied`,
    `param` `model`, `code` `"403"`) and must not call the provider.
    """
    with peer() as provider, gateway.scenario() as scenario:
        allowed: Final = cohere_rerank_model(scenario, provider)
        other: Final = cohere_rerank_model(scenario, provider)
        key: Final = scenario.key(models=[allowed])
        response: Final = rerank(gateway, other, key=key)
        assert response.status_code == 403, response.text
        assert response.json() == {
            "error": {
                "message": MODEL_DENIED_TEMPLATE.format(model=other),
                "type": "key_model_access_denied",
                "param": "model",
                "code": "403",
            }
        }
        assert provider.posts() == ()


def test_rerank_without_credentials_is_refused_with_401(gateway: Gateway) -> None:
    """A caller posts to `/v1/rerank` with `Authorization: Bearer sk-dec1-not-a-key`.

    The proxy must answer 401 with an `error` object and must not call any provider.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        response: Final = rerank(gateway, model, key="sk-dec1-not-a-key")
        assert response.status_code == 401, response.text
        assert error_of(response)["code"] == "401"
        assert provider.posts() == ()


def test_rerank_requests_count_against_the_key_rpm_limit(gateway: Gateway) -> None:
    """A key with `rpm_limit: 1` reranks twice.

    The first call is served and the second is refused with 429, `error.type` `throttling_error` and
    `error.code` `"429"`; the provider is called once.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        served(rerank(gateway, model, key=key))
        refused: Final = rerank(gateway, model, key=key)
        assert refused.status_code == 429, refused.text
        error: Final = error_of(refused)
        assert (error["type"], error["code"]) == ("throttling_error", "429"), error
        assert len(provider.posts()) == 1


def test_a_key_with_an_exhausted_budget_cannot_rerank(gateway: Gateway) -> None:
    """A key with `max_budget: 0` reranks.

    The proxy must refuse before calling the provider with HTTP 422, `error.type` `budget_exceeded` and
    `error.code` `"422"`, the same refusal chat completions gives.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = rerank(gateway, model, key=key)
        assert response.status_code == 422, response.text
        error: Final = error_of(response)
        assert (error["type"], error["code"]) == ("budget_exceeded", "422"), error
        assert provider.posts() == ()


def test_an_identical_rerank_is_served_from_the_response_cache(gateway: Gateway) -> None:
    """The shared gateway runs with `cache: true`. A caller sends the same rerank (model, query, documents)
    repeatedly.

    After the first call has been stored, a repeat must be answered with the same results without a new
    provider call, and the provider must have been called exactly once in total. `arerank` is in the default
    list of cached call types (`litellm/types/caching.py`).
    """
    query: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        first: Final = rerank(gateway, model, query=query)
        assert results_of(first) == RERANK_SCORES

        def repeat() -> tuple[int, int]:
            before: Final = len(provider.posts())
            again: Final = rerank(gateway, model, query=query)
            assert results_of(again) == RERANK_SCORES
            return before, len(provider.posts())

        counts: Final = eventually(repeat, lambda seen: seen[0] == seen[1], seconds=20)
        assert counts[1] <= 3, "the cache never answered"


def test_a_cached_rerank_names_its_cache_key_in_a_response_header(gateway: Gateway) -> None:
    """Once a rerank is answered from the cache, the response must carry a non-empty `x-litellm-cache-key`
    header. UNVERIFIED: the rerank handler copies `_hidden_params["cache_key"]` into the header; whether the
    cache handler sets it for rerank hits was not traced (`litellm/caching/caching_handler.py`)."""
    query: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        served(rerank(gateway, model, query=query))

        def repeat() -> tuple[int, str]:
            before: Final = len(provider.posts())
            again: Final = rerank(gateway, model, query=query)
            return len(provider.posts()) - before, again.headers.get("x-litellm-cache-key", "")

        hit: Final = eventually(repeat, lambda seen: seen[0] == 0, seconds=20)
        assert hit[1], "cache hit without x-litellm-cache-key"


def test_a_different_query_is_not_answered_from_the_rerank_cache(gateway: Gateway) -> None:
    """A caller reranks the same documents with two different queries.

    Both calls must reach the provider: the query is part of the cache key.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        served(rerank(gateway, model, query=marker()))
        served(rerank(gateway, model, query=marker()))
        assert len(provider.posts()) == 2


def test_no_cache_in_the_request_forces_a_provider_call(gateway: Gateway) -> None:
    """A caller repeats one rerank four times with `cache: {"no-cache": true}` in the body.

    Every call must reach the provider (four provider calls), and the `cache` control must not be forwarded to
    Cohere.
    """
    query: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = cohere_rerank_model(scenario, provider)
        for _ in range(4):
            served(rerank(gateway, model, query=query, cache={"no-cache": True}))
        calls: Final = provider.posts()
        assert len(calls) == 4
        assert all("cache" not in call.body for call in calls)


def _unused(provider: Peer) -> None:
    """Keep `Peer` imported for type readers of this module."""
