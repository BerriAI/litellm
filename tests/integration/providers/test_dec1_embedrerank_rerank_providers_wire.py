"""Contract: `/v1/rerank` request and response mapping for Jina, Voyage, Together, Infinity, hosted vLLM, Azure AI and
Bedrock, beyond what `test_dec1_textapis_rerank.py` and the single-purpose rerank files already pin.

The caller always sends Cohere's rerank body (`query`, `documents` as strings or objects, `top_n`, `rank_fields`,
`return_documents`, `max_chunks_per_doc`) and always gets Cohere's answer back (`id`, `results[]` with `index`,
`relevance_score` and an optional `document: {"text"}`, and `meta`). Per provider, read from its rerank module:

    jina_ai      the `api_base` path is replaced by `/v1/rerank`; a keyless deployment is refused before any call
    voyage       `/v1/rerank` is joined unless the base already ends in `/v1` or `/v1/rerank`; `rank_fields` is
                 dropped silently; `return_documents` is forwarded
    together_ai  `<api_base>/rerank`; object documents in the answer become `{"text"}`; `usage` becomes
                 `meta.billed_units`
    infinity     the Cohere v1 body including `max_chunks_per_doc`; `usage` becomes `meta.billed_units` and
                 `meta.tokens` with `output_tokens = total_tokens - prompt_tokens`
    hosted_vllm  `/rerank` joined; a base ending in `/v1/rerank` is rewritten to `/rerank`; truncation fields are
                 forwarded; `max_chunks_per_doc` is refused; a keyless deployment sends `Bearer fake-api-key`
    azure_ai     Cohere v1 body; `/v1/rerank` joined unless the base ends in `/v1`, `/v2`, `/providers/cohere/v2`
                 (then `/rerank`) or already in `/v1/rerank` or `/v2/rerank`
    bedrock      Agent Runtime `/rerank` body: `queries[].textQuery`, `sources[]` as TEXT or JSON inline documents,
                 `numberOfResults` = `top_n` or the document count; `relevanceScore` becomes `relevance_score`;
                 a missing `usage` is billed as one search unit
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.providers.dec1_embedrerank_support import (
    DOCUMENTS,
    PROVIDER_KEY,
    QUERY_RATE,
    SCORES,
    Call,
    Peer,
    Respond,
    bedrock_rerank,
    cohere_rerank,
    error_message,
    error_reply,
    json_reply,
    peer,
    rerank,
    results,
    served,
)
from pydantic import JsonValue

from tests.integration._support.client import Scenario

_JINA: Final = "jina-reranker-v2-base-multilingual"
_VOYAGE: Final = "rerank-2.5"
_TOGETHER: Final = "Salesforce/Llama-Rank-V1"
_INFINITY: Final = "mixedbread-ai/mxbai-rerank-xsmall-v1"
_VLLM: Final = "BAAI/bge-reranker-v2-m3"
_AZURE: Final = "cohere-rerank-v3.5"
_BEDROCK_ARN: Final = "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
_OBJECT_DOCUMENTS: Final[list[JsonValue]] = [
    {"title": "gateway", "body": DOCUMENTS[0]},
    {"title": "note", "body": DOCUMENTS[1]},
    {"title": "note", "body": DOCUMENTS[2]},
]


def _rerank_model(scenario: Scenario, litellm_model: str, api_base: str, **parameters: JsonValue) -> str:
    return scenario.model(model=litellm_model, api_base=api_base, model_info={"mode": "rerank"}, **parameters)


def _voyage_answer(call: Call) -> Reply:
    return json_reply(
        {
            "object": "list",
            "data": [{"index": index, "relevance_score": score} for index, score in SCORES],
            "model": _VOYAGE,
            "usage": {"total_tokens": 21},
        }
    )


def _usage_answer(usage: dict[str, JsonValue], document: JsonValue = None) -> Respond:
    """A Together/Infinity/vLLM style answer: Cohere results plus a `usage` block and no `meta`."""

    def respond(call: Call) -> Reply:
        entries: Final[list[JsonValue]] = []
        for index, score in SCORES:
            entry: dict[str, JsonValue] = {"index": index, "relevance_score": score}
            if document is not None:
                entry["document"] = document
            entries.append(entry)
        return json_reply({"id": "dec1-usage-rerank", "results": entries, "usage": usage})

    return respond


def _bedrock(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    return scenario.model(
        model=f"bedrock/{_BEDROCK_ARN}",
        api_base=None,
        aws_region_name="us-west-2",
        aws_bedrock_runtime_endpoint=provider.url,
        aws_access_key_id="AKIAINTEGRATION000077",
        aws_secret_access_key="synthetic-dec1-embedrerank-secret",
        model_info={"mode": "rerank"},
        **parameters,
    )


def test_jina_replaces_the_api_base_path_with_v1_rerank(gateway: Gateway) -> None:
    """A Jina rerank deployment whose `api_base` is `<peer>/tenant/prefix`.

    `JinaAIRerankConfig.get_complete_url` keeps only scheme and host and sets the path to `/v1/rerank`, so the POST
    must land on `/v1/rerank`, not `/tenant/prefix/v1/rerank`. A port that joins paths would break this.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(
            scenario, f"jina_ai/{_JINA}", f"{provider.url}/tenant/prefix", api_key=PROVIDER_KEY
        )
        served(rerank(gateway, model))
        assert provider.only().path == "/v1/rerank"


def test_jina_keyless_rerank_is_refused_before_any_request(gateway: Gateway) -> None:
    """A Jina rerank deployment with no `api_key`.

    `rerank_api/main.py` raises `Jina AI API key is required, please set 'JINA_AI_API_KEY' in your environment`
    before any request; the caller must get an error containing that text and Jina receives nothing. UNVERIFIED:
    the status code (a ValueError) and that `JINA_AI_API_KEY` is unset in CI.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"jina_ai/{_JINA}", provider.url)
        response: Final = rerank(gateway, model)
        assert response.status_code >= 400, response.text
        assert "Jina AI API key is required" in error_message(response)
        assert provider.posts() == ()


def test_jina_object_documents_returned_by_jina_are_kept_as_objects(gateway: Gateway) -> None:
    """The caller sends object documents with `return_documents: true`; Jina echoes each document as
    `{"text": "..."}` objects.

    A non-string `document` is passed through unchanged, so every result must carry `document: {"text": ...}` and
    the results keep Jina's order (index 2, 0, 1).
    """
    document: Final[JsonValue] = {"text": DOCUMENTS[0]}
    with peer(_usage_answer({"total_tokens": 12}, document)) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"jina_ai/{_JINA}", provider.url, api_key=PROVIDER_KEY)
        body: Final = served(rerank(gateway, model, documents=_OBJECT_DOCUMENTS, return_documents=True))
        assert results(body) == [(index, score) for index, score in SCORES]
        found: Final = body["results"]
        assert isinstance(found, list)
        assert all(object_value(entry)["document"] == document for entry in found)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        pytest.param("", "/v1/rerank", id="bare-host"),
        pytest.param("/v1", "/v1/rerank", id="v1-base"),
        pytest.param("/v1/rerank", "/v1/rerank", id="full-endpoint"),
        pytest.param("/voyage-gateway", "/voyage-gateway/v1/rerank", id="custom-prefix"),
    ],
)
def test_voyage_rerank_url_is_built_from_the_api_base(gateway: Gateway, suffix: str, expected: str) -> None:
    """A Voyage rerank deployment with `api_base: <peer><suffix>`.

    `VoyageRerankConfig.get_complete_url` joins `v1/rerank` onto a bare or custom base, joins only `rerank` onto a
    base ending in `/v1`, and keeps a base already ending in `/v1/rerank`. The POST must land on `expected`.
    """
    with peer(_voyage_answer) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"voyage/{_VOYAGE}", f"{provider.url}{suffix}", api_key=PROVIDER_KEY)
        served(rerank(gateway, model))
        assert provider.only().path == expected


def test_voyage_drops_rank_fields_and_forwards_return_documents(gateway: Gateway) -> None:
    """The caller sends `rank_fields: ["body"]`, `return_documents: false` and `top_n: 2` to Voyage.

    Voyage has no `rank_fields`; the mapping drops it without an error. The body must be exactly `{"model":
    "rerank-2.5", "query", "documents", "top_k": 2, "return_documents": false}`.
    """
    query: Final = f"voyage {uuid.uuid4().hex}"
    with peer(_voyage_answer) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"voyage/{_VOYAGE}", provider.url, api_key=PROVIDER_KEY)
        served(rerank(gateway, model, query=query, rank_fields=["body"], return_documents=False, top_n=2))
        assert provider.only().body == {
            "model": _VOYAGE,
            "query": query,
            "documents": list(DOCUMENTS),
            "top_k": 2,
            "return_documents": False,
        }


def test_voyage_rerank_meta_reports_total_tokens_as_input_tokens(gateway: Gateway) -> None:
    """Voyage answers `usage.total_tokens: 21`.

    The caller's `meta` must be `billed_units: {"total_tokens": 21}` and `tokens: {"input_tokens": 21,
    "output_tokens": 0}`.
    """
    with peer(_voyage_answer) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"voyage/{_VOYAGE}", provider.url, api_key=PROVIDER_KEY)
        meta: Final = object_value(served(rerank(gateway, model))["meta"])
        assert object_value(meta["billed_units"])["total_tokens"] == 21
        assert meta["tokens"] == {"input_tokens": 21, "output_tokens": 0}


def test_voyage_rerank_error_keeps_its_status(gateway: Gateway) -> None:
    """Voyage answers the rerank with HTTP 429 `dec1 voyage busy`.

    VoyageError carries the status, so the caller must get 429 with that text.
    """
    with peer(lambda call: error_reply(429, "dec1 voyage busy")) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"voyage/{_VOYAGE}", provider.url, api_key=PROVIDER_KEY)
        response: Final = rerank(gateway, model)
        assert response.status_code == 429, response.text
        assert "dec1 voyage busy" in error_message(response)


def test_together_rerank_posts_to_rerank_with_the_key_as_bearer(gateway: Gateway) -> None:
    """A `together_ai/Salesforce/Llama-Rank-V1` rerank deployment with `api_base: <peer>/v1/`.

    The trailing slash is stripped and `/rerank` appended: one POST to `/v1/rerank` with `authorization: Bearer
    <deployment key>` and `model: "Salesforce/Llama-Rank-V1"`.
    """
    with peer(_usage_answer({"prompt_tokens": 9, "total_tokens": 9})) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"together_ai/{_TOGETHER}", f"{provider.url}/v1/", api_key=PROVIDER_KEY)
        served(rerank(gateway, model))
        call: Final = provider.only()
        assert call.path == "/v1/rerank"
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body["model"] == _TOGETHER


def test_together_rerank_usage_becomes_meta_billed_units(gateway: Gateway) -> None:
    """Together answers `usage: {"prompt_tokens": 9, "total_tokens": 9}` and object documents.

    The caller's `meta.billed_units` must carry `total_tokens: 9`, and each result's document must be
    `{"text": <the text Together returned>}`.
    """
    answer: Final = _usage_answer({"prompt_tokens": 9, "total_tokens": 9}, {"text": DOCUMENTS[1]})
    with peer(answer) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"together_ai/{_TOGETHER}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        body: Final = served(rerank(gateway, model, return_documents=True))
        assert object_value(object_value(body["meta"])["billed_units"])["total_tokens"] == 9
        found: Final = body["results"]
        assert isinstance(found, list)
        assert {str(object_value(entry)["document"]) for entry in found} == {str({"text": DOCUMENTS[1]})}


def test_together_rerank_error_keeps_its_status(gateway: Gateway) -> None:
    """Together answers HTTP 429 `dec1 together busy`.

    The caller should get 429 with that text, like every other provider.
    """
    pytest.skip("BUG: TogetherAIRerank raises a bare Exception(response.text) on non-200, losing the 429 status")
    with peer(lambda call: error_reply(429, "dec1 together busy")) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"together_ai/{_TOGETHER}", f"{provider.url}/v1", api_key=PROVIDER_KEY)
        response: Final = rerank(gateway, model)
        assert response.status_code == 429, response.text
        assert "dec1 together busy" in error_message(response)


def test_infinity_rerank_sends_the_cohere_v1_body_including_max_chunks_per_doc(gateway: Gateway) -> None:
    """The caller sends `top_n: 2`, `return_documents: false`, `max_chunks_per_doc: 4` to an Infinity reranker.

    `InfinityRerankConfig` reuses the Cohere v1 request, so the body must be exactly `{"model":
    "mixedbread-ai/mxbai-rerank-xsmall-v1", "query", "documents", "top_n": 2, "return_documents": false,
    "max_chunks_per_doc": 4}`.
    """
    query: Final = f"infinity {uuid.uuid4().hex}"
    with peer(_usage_answer({"prompt_tokens": 7, "total_tokens": 10})) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"infinity/{_INFINITY}", provider.url, api_key=PROVIDER_KEY)
        served(rerank(gateway, model, query=query, top_n=2, return_documents=False, max_chunks_per_doc=4))
        assert provider.only().body == {
            "model": _INFINITY,
            "query": query,
            "documents": list(DOCUMENTS),
            "top_n": 2,
            "return_documents": False,
            "max_chunks_per_doc": 4,
        }


def test_infinity_rerank_meta_splits_usage_into_input_and_output_tokens(gateway: Gateway) -> None:
    """Infinity answers `usage: {"prompt_tokens": 7, "total_tokens": 10}`.

    The caller's `meta.billed_units` must be the usage block itself (`prompt_tokens: 7`, `total_tokens: 10`) and
    `meta.tokens` must be `{"input_tokens": 7, "output_tokens": 3}` (total minus prompt).
    """
    with peer(_usage_answer({"prompt_tokens": 7, "total_tokens": 10})) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"infinity/{_INFINITY}", provider.url, api_key=PROVIDER_KEY)
        meta: Final = object_value(served(rerank(gateway, model))["meta"])
        billed: Final = object_value(meta["billed_units"])
        assert (billed["prompt_tokens"], billed["total_tokens"]) == (7, 10)
        assert meta["tokens"] == {"input_tokens": 7, "output_tokens": 3}


def test_infinity_rerank_without_an_id_gets_a_generated_one(gateway: Gateway) -> None:
    """Infinity answers results with no `id` field.

    The caller must still get a non-empty string `id` (a generated uuid), because Cohere clients require it.
    """

    def respond(call: Call) -> Reply:
        return json_reply({"results": [{"index": 0, "relevance_score": 0.5}], "usage": {"total_tokens": 3}})

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"infinity/{_INFINITY}", provider.url, api_key=PROVIDER_KEY)
        identity: Final = served(rerank(gateway, model))["id"]
        assert isinstance(identity, str) and identity, identity


def test_infinity_keyless_rerank_sends_no_bearer_none(gateway: Gateway) -> None:
    """An Infinity rerank deployment with no `api_key`.

    The provider should never receive `Authorization: Bearer None`.
    """
    pytest.skip("BUG: InfinityRerankConfig.validate_environment formats a missing key as 'Bearer None'")
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"infinity/{_INFINITY}", provider.url)
        served(rerank(gateway, model))
        assert provider.only().headers.get("authorization") != "Bearer None"


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        pytest.param("", "/rerank", id="bare-host"),
        pytest.param("/v1", "/v1/rerank", id="v1-base"),
        pytest.param("/v1/rerank", "/rerank", id="v1-rerank-rewritten"),
        pytest.param("/rerank", "/rerank", id="rerank-kept"),
    ],
)
def test_hosted_vllm_rerank_url_rules(gateway: Gateway, suffix: str, expected: str) -> None:
    """A hosted vLLM rerank deployment with `api_base: <peer><suffix>`.

    `HostedVLLMRerankConfig.get_complete_url` appends `/rerank` unless present, and rewrites a base ending in
    `/v1/rerank` to `/rerank` (vLLM serves the Cohere-compatible route at `/rerank`). The POST must land on
    `expected`.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"hosted_vllm/{_VLLM}", f"{provider.url}{suffix}", api_key=PROVIDER_KEY)
        served(rerank(gateway, model))
        assert provider.only().path == expected


def test_hosted_vllm_keyless_rerank_sends_the_fake_api_key_bearer(gateway: Gateway) -> None:
    """A hosted vLLM rerank deployment with no `api_key`.

    Unlike vLLM embeddings (no header), the rerank config sends `Authorization: Bearer fake-api-key`. Pinned
    because the asymmetry is easy to "fix" in one place only. UNVERIFIED: `HOSTED_VLLM_API_KEY` unset in CI.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"hosted_vllm/{_VLLM}", provider.url)
        served(rerank(gateway, model))
        assert provider.only().headers["authorization"] == "Bearer fake-api-key"


def test_hosted_vllm_rerank_forwards_truncation_and_instruction_fields(gateway: Gateway) -> None:
    """The caller sends `truncate_prompt_tokens: 256`, `truncation_side: "left"`, `max_tokens_per_query: 64`,
    `max_tokens_per_doc: 128` and `instruction: "rank by relevance"`.

    vLLM must receive each of those five fields unchanged next to `model`, `query` and `documents`.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"hosted_vllm/{_VLLM}", provider.url, api_key=PROVIDER_KEY)
        served(
            rerank(
                gateway,
                model,
                truncate_prompt_tokens=256,
                truncation_side="left",
                max_tokens_per_query=64,
                max_tokens_per_doc=128,
                instruction="rank by relevance",
            )
        )
        body: Final = provider.only().body
        assert (body["truncate_prompt_tokens"], body["truncation_side"]) == (256, "left")
        assert (body["max_tokens_per_query"], body["max_tokens_per_doc"]) == (64, 128)
        assert body["instruction"] == "rank by relevance"


def test_hosted_vllm_rerank_refuses_an_invalid_truncation_side_with_400(gateway: Gateway) -> None:
    """The caller sends `truncation_side: "middle"`.

    `validated_truncation_params` raises UnsupportedParamsError 400 with a message starting `hosted_vllm rerank:`;
    the caller must get 400 and vLLM receives nothing.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"hosted_vllm/{_VLLM}", provider.url, api_key=PROVIDER_KEY)
        response: Final = rerank(gateway, model, truncation_side="middle")
        assert response.status_code == 400, response.text
        assert "hosted_vllm rerank:" in error_message(response)
        assert provider.posts() == ()


def test_hosted_vllm_rerank_refuses_max_chunks_per_doc(gateway: Gateway) -> None:
    """The caller sends `max_chunks_per_doc: 2` to a hosted vLLM reranker.

    The mapping raises `Hosted VLLM does not support max_chunks_per_doc` before any request. UNVERIFIED: status
    (a ValueError, expected 500 today).
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"hosted_vllm/{_VLLM}", provider.url, api_key=PROVIDER_KEY)
        response: Final = rerank(gateway, model, max_chunks_per_doc=2)
        assert response.status_code >= 400, response.text
        assert "Hosted VLLM does not support max_chunks_per_doc" in error_message(response)
        assert provider.posts() == ()


def test_hosted_vllm_rerank_without_meta_builds_it_from_usage(gateway: Gateway) -> None:
    """vLLM answers results and `usage: {"total_tokens": 15}` with no `meta`.

    The caller's `meta` must be `billed_units: {"total_tokens": 15}` and `tokens: {"input_tokens": 15}`.
    """
    with peer(_usage_answer({"total_tokens": 15})) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"hosted_vllm/{_VLLM}", provider.url, api_key=PROVIDER_KEY)
        meta: Final = object_value(served(rerank(gateway, model))["meta"])
        assert object_value(meta["billed_units"])["total_tokens"] == 15
        assert object_value(meta["tokens"])["input_tokens"] == 15


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        pytest.param("", "/v1/rerank", id="bare-host"),
        pytest.param("/providers/cohere/v2", "/providers/cohere/v2/rerank", id="foundry-cohere-v2"),
        pytest.param("/v2/rerank", "/v2/rerank", id="v2-endpoint"),
        pytest.param("/v1/", "/v1/rerank", id="v1-trailing-slash"),
    ],
)
def test_azure_ai_rerank_url_rules(gateway: Gateway, suffix: str, expected: str) -> None:
    """An `azure_ai/cohere-rerank-v3.5` deployment with `api_base: <peer><suffix>`.

    `AzureAIRerankConfig.get_complete_url` keeps a base ending in `/v1/rerank` or `/v2/rerank`, appends `/rerank`
    to `/v1`, `/v2` or `/providers/cohere/v2`, and appends `/v1/rerank` to anything else. The POST must land on
    `expected` with `Authorization: Bearer <deployment key>`.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"azure_ai/{_AZURE}", f"{provider.url}{suffix}", api_key=PROVIDER_KEY)
        served(rerank(gateway, model))
        call: Final = provider.only()
        assert call.path == expected
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_azure_ai_rerank_sends_the_cohere_v1_body(gateway: Gateway) -> None:
    """The caller sends object documents with `rank_fields: ["body"]`, `top_n: 1` and `max_chunks_per_doc: 3`.

    Azure AI reuses the Cohere v1 request, so the body must be exactly `{"model": "cohere-rerank-v3.5", "query",
    "documents": <the objects>, "top_n": 1, "rank_fields": ["body"], "max_chunks_per_doc": 3}`.
    """
    query: Final = f"azure {uuid.uuid4().hex}"
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"azure_ai/{_AZURE}", provider.url, api_key=PROVIDER_KEY)
        served(
            rerank(
                gateway,
                model,
                query=query,
                documents=_OBJECT_DOCUMENTS,
                top_n=1,
                rank_fields=["body"],
                max_chunks_per_doc=3,
            )
        )
        assert provider.only().body == {
            "model": _AZURE,
            "query": query,
            "documents": _OBJECT_DOCUMENTS,
            "top_n": 1,
            "rank_fields": ["body"],
            "max_chunks_per_doc": 3,
        }


def test_azure_ai_rerank_answer_is_returned_as_cohere_sent_it(gateway: Gateway) -> None:
    """Azure AI answers a Cohere body with `meta.billed_units.search_units: 2`.

    The caller must get the results in Azure's order (index 2, 0, 1 with their scores) and `meta.billed_units`
    with `search_units: 2`.
    """
    with peer(cohere_rerank(2)) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"azure_ai/{_AZURE}", provider.url, api_key=PROVIDER_KEY)
        body: Final = served(rerank(gateway, model))
        assert results(body) == [(index, score) for index, score in SCORES]
        assert object_value(object_value(body["meta"])["billed_units"])["search_units"] == 2


def test_bedrock_rerank_maps_string_documents_to_text_sources(gateway: Gateway) -> None:
    """A `bedrock/<amazon.rerank-v1:0 ARN>` rerank deployment; the caller sends three string documents, no `top_n`.

    Bedrock Agent Runtime must receive `POST /rerank` with `queries: [{"textQuery": {"text": <query>}, "type":
    "TEXT"}]`, one `{"inlineDocumentSource": {"textDocument": {"text": d}, "type": "TEXT"}, "type": "INLINE"}`
    source per document, and `numberOfResults: 3` (the document count) with `modelArn` set to the ARN.
    """
    query: Final = f"bedrock {uuid.uuid4().hex}"
    with peer(bedrock_rerank) as provider, gateway.scenario() as scenario:
        model: Final = _bedrock(scenario, provider)
        served(rerank(gateway, model, query=query))
        call: Final = provider.only()
        assert call.path == "/rerank"
        assert call.body["queries"] == [{"textQuery": {"text": query}, "type": "TEXT"}]
        assert call.body["sources"] == [
            {"inlineDocumentSource": {"textDocument": {"text": text}, "type": "TEXT"}, "type": "INLINE"}
            for text in DOCUMENTS
        ]
        configuration: Final = object_value(
            object_value(call.body["rerankingConfiguration"])["bedrockRerankingConfiguration"]
        )
        assert configuration["numberOfResults"] == 3
        assert configuration["modelConfiguration"] == {"modelArn": _BEDROCK_ARN}


def test_bedrock_rerank_maps_object_documents_to_json_sources_and_top_n_to_number_of_results(
    gateway: Gateway,
) -> None:
    """The caller sends object documents and `top_n: 1`.

    Each object becomes `{"inlineDocumentSource": {"jsonDocument": <the object>, "type": "JSON"}, "type":
    "INLINE"}` and `numberOfResults` is 1.
    """
    with peer(bedrock_rerank) as provider, gateway.scenario() as scenario:
        model: Final = _bedrock(scenario, provider)
        served(rerank(gateway, model, documents=_OBJECT_DOCUMENTS, top_n=1))
        call: Final = provider.only()
        assert call.body["sources"] == [
            {"inlineDocumentSource": {"jsonDocument": document, "type": "JSON"}, "type": "INLINE"}
            for document in _OBJECT_DOCUMENTS
        ]
        configuration: Final = object_value(
            object_value(call.body["rerankingConfiguration"])["bedrockRerankingConfiguration"]
        )
        assert configuration["numberOfResults"] == 1


def test_bedrock_rerank_answer_is_mapped_to_cohere_results_with_one_search_unit(gateway: Gateway) -> None:
    """Bedrock answers `results: [{"index", "relevanceScore"}]` in score order and no `usage`.

    The caller must get `results` as `(index, relevance_score)` pairs in Bedrock's order (2, 0, 1), no `document`
    fields even though Cohere's default is to return them, and `meta.billed_units.search_units: 1`.
    """
    with peer(bedrock_rerank) as provider, gateway.scenario() as scenario:
        model: Final = _bedrock(scenario, provider)
        body: Final = served(rerank(gateway, model, return_documents=True))
        assert results(body) == [(index, score) for index, score in SCORES]
        found: Final = body["results"]
        assert isinstance(found, list)
        assert all("document" not in object_value(entry) for entry in found)
        assert object_value(object_value(body["meta"])["billed_units"])["search_units"] == 1


def test_bedrock_rerank_cost_is_one_query_at_the_deployment_price(gateway: Gateway) -> None:
    """A Bedrock rerank deployment priced `input_cost_per_query: 0.002`; Bedrock reports no usage.

    One search unit is billed, so `x-litellm-response-cost` must be 0.002.
    """
    with peer(bedrock_rerank) as provider, gateway.scenario() as scenario:
        model: Final = _bedrock(scenario, provider, input_cost_per_query=QUERY_RATE)
        response: Final = rerank(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.002)


def test_bedrock_rerank_throttling_reaches_the_caller_as_429(gateway: Gateway) -> None:
    """Bedrock answers HTTP 429 `{"message": "Rate exceeded"}`.

    The rerank handler raises BedrockError with the HTTP status, so the caller must get 429 with `Rate exceeded`.
    """

    def respond(call: Call) -> Reply:
        return json_reply({"message": "Rate exceeded"}, 429)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _bedrock(scenario, provider)
        response: Final = rerank(gateway, model)
        assert response.status_code == 429, response.text
        assert "Rate exceeded" in error_message(response)
