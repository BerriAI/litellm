"""Contract: what the proxy itself does around embeddings and rerank on non-OpenAI providers: cost and spend rows,
per-input embedding cache, rerank cache keys, route aliases, timeouts, and routing across providers.

These behaviours live in the proxy and router rather than in a provider module, which is exactly the code moving to
the Rust router core and Rust HTTP path. Each test puts a non-OpenAI provider behind the proxy so the provider
translation and the proxy layer are exercised together:

    cost        `x-litellm-response-cost` and `LiteLLM_SpendLogs.spend` come from the provider's own usage field
                (Cohere `billed_units.input_tokens`, Voyage `total_tokens`, Ollama `prompt_eval_count`) times the
                deployment's `input_cost_per_token`; rerank bills `input_cost_per_query` times
                `meta.billed_units.search_units`, and providers that report no search units bill 0
    spend row   `call_type` is `aembedding` or `arerank`, `completion_tokens` is 0, `custom_llm_provider` names
                the provider, `model_group` is the public name
    cache       the shared gateway caches in `default_on` mode: each embedding input has its own cache entry, so a
                list that repeats a cached input only sends the new inputs to the provider; rerank cache keys
                include `top_n`
    routing     ordered deployments and fallbacks move a request from one provider to another (owned proxy with
                `num_retries` set explicitly)

Spend for OpenAI/Azure/Bedrock/Vertex embeddings and the Cohere rerank spend row are pinned by sibling files.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.providers.dec1_embedrerank_support import (
    CACHE_KEY_HEADER,
    COST_HEADER,
    INPUT_RATE,
    PROVIDER_KEY,
    QUERY_RATE,
    Call,
    Peer,
    Respond,
    Rig,
    call_header,
    cohere_embed,
    cohere_rerank,
    deployment,
    embed,
    error_reply,
    failing,
    held,
    json_reply,
    marker,
    new_call_id,
    number,
    ollama_embed,
    openai_list,
    owned_rig,
    peer,
    rerank,
    results,
    served,
    spend_row,
)
from pydantic import JsonValue

from tests.integration._support.client import Scenario

_COHERE: Final = "embed-english-v3.0"
_VOYAGE: Final = "voyage-3.5"
_JINA_RERANK: Final = "jina-reranker-v2-base-multilingual"
_COHERE_RERANK: Final = "rerank-v3.5"
_AZURE_RERANK: Final = "cohere-rerank-v3.5"
_VLLM_RERANK: Final = "BAAI/bge-reranker-v2-m3"


def _voyage_usage(call: Call) -> Reply:
    texts: Final = call.body.get("input")
    count: Final = len(texts) if isinstance(texts, list) else 1
    return json_reply(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": [0.5, float(index)]} for index in range(count)
            ],
            "model": _VOYAGE,
            "usage": {"total_tokens": 11},
        }
    )


def _voyage(scenario: Scenario, provider: Peer) -> str:
    return scenario.model(
        model=f"voyage/{_VOYAGE}", api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY, input_cost_per_token=INPUT_RATE
    )


def _cohere(scenario: Scenario, provider: Peer) -> str:
    return scenario.model(
        model=f"cohere/{_COHERE}",
        api_base=f"{provider.url}/v2/embed",
        api_key=PROVIDER_KEY,
        input_cost_per_token=INPUT_RATE,
    )


def _rerank_model(scenario: Scenario, litellm_model: str, provider: Peer, **parameters: JsonValue) -> str:
    return scenario.model(
        model=litellm_model, api_base=provider.url, api_key=PROVIDER_KEY, model_info={"mode": "rerank"}, **parameters
    )


@pytest.mark.parametrize(
    ("litellm_model", "suffix", "respond", "provider_name", "tokens"),
    [
        pytest.param(f"cohere/{_COHERE}", "/v2/embed", cohere_embed({"input_tokens": 3}), "cohere", 3, id="cohere"),
        pytest.param(f"voyage/{_VOYAGE}", "/v1", _voyage_usage, "voyage", 11, id="voyage"),
        pytest.param("ollama/nomic-embed-text", "", ollama_embed(4), "ollama", 4, id="ollama"),
        pytest.param("nvidia_nim/nvidia/nv-embedqa-e5-v5", "/v1", openai_list(5), "nvidia_nim", 5, id="nim"),
        pytest.param("infinity/BAAI/bge-small-en-v1.5", "/v1", openai_list(6), "infinity", 6, id="infinity"),
    ],
)
def test_an_embedding_writes_one_spend_row_priced_from_the_provider_usage(
    gateway: Gateway, litellm_model: str, suffix: str, respond: Respond, provider_name: str, tokens: int
) -> None:
    """One embedding on a deployment priced 0.001 per input token, sent with a fixed `x-litellm-call-id`.

    Exactly one `LiteLLM_SpendLogs` row must land for that call id with `call_type: "aembedding"`,
    `custom_llm_provider` naming the provider, `prompt_tokens` and `total_tokens` equal to the provider-reported
    count, `completion_tokens: 0` and `spend` = tokens * 0.001. UNVERIFIED for infinity: its handler passes
    `litellm_params={}`, which may hide the deployment price.
    """
    call_id: Final = new_call_id()
    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=litellm_model,
            api_base=f"{provider.url}{suffix}",
            api_key=PROVIDER_KEY,
            input_cost_per_token=INPUT_RATE,
        )
        served(embed(gateway, model, [marker()], headers=call_header(call_id)))
        row: Final = spend_row(gateway, call_id)
        assert (row["call_type"], row["custom_llm_provider"], row["model_group"]) == (
            "aembedding",
            provider_name,
            model,
        )
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (tokens, 0, tokens)
        assert number(row["spend"]) == pytest.approx(tokens * INPUT_RATE)


def test_an_embedding_cost_header_matches_its_spend_row(gateway: Gateway) -> None:
    """A Voyage embedding (11 tokens at 0.001) sent with a fixed call id.

    The response header `x-litellm-response-cost` must be 0.011 and the spend row for that call id must hold the
    same 0.011, so dashboards and callers never disagree.
    """
    call_id: Final = new_call_id()
    with peer(_voyage_usage) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider)
        response: Final = embed(gateway, model, [marker()], headers=call_header(call_id))
        served(response)
        assert float(response.headers[COST_HEADER]) == pytest.approx(0.011)
        assert number(spend_row(gateway, call_id)["spend"]) == pytest.approx(0.011)


def test_an_embedding_spend_row_records_the_caller_user_as_end_user(gateway: Gateway) -> None:
    """A NIM embedding sent with `user: "dec1-embed-end-user"` (NIM supports and forwards `user`).

    NIM must receive `user: "dec1-embed-end-user"`, and once the spend row for the call id has landed its
    `end_user` column must hold `dec1-embed-end-user`. UNVERIFIED: read from how chat end users are logged
    (`get_logging_payload` in `litellm/proxy/spend_tracking/spend_tracking_utils.py`).
    """
    call_id: Final = new_call_id()
    with peer(openai_list(5)) as provider, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="nvidia_nim/nvidia/nv-embedqa-e5-v5", api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY
        )
        served(embed(gateway, model, [marker()], user="dec1-embed-end-user", headers=call_header(call_id)))
        spend_row(gateway, call_id)
        rows: Final = gateway.rows('SELECT end_user FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (call_id,))
        assert rows[0]["end_user"] == "dec1-embed-end-user"
        assert provider.only().body["user"] == "dec1-embed-end-user"


def test_a_failed_embedding_writes_a_failure_row_with_zero_spend(gateway: Gateway) -> None:
    """A Voyage embedding that Voyage answers with HTTP 500.

    The caller gets 500, and one spend row for the call id must land with `status: "failure"` and `spend: 0`.
    """
    call_id: Final = new_call_id()
    with peer(failing(500)) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider)
        response: Final = embed(gateway, model, [marker()], headers=call_header(call_id))
        assert response.status_code == 500, response.text
        row: Final = spend_row(gateway, call_id)
        assert row["status"] == "failure"
        assert number(row["spend"]) == 0.0


def test_a_repeated_voyage_embedding_is_served_from_the_cache(gateway: Gateway) -> None:
    """The same single-input Voyage embedding is sent twice on the shared gateway (cache `default_on`).

    Voyage must be called once; the second answer must carry `x-litellm-cache-key` and
    `x-litellm-response-cost: 0`.
    """
    text: Final = marker()
    with peer(_voyage_usage) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider)
        served(embed(gateway, model, [text]))
        second: Final = embed(gateway, model, [text])
        served(second)
        assert len(provider.posts()) == 1
        assert second.headers.get(CACHE_KEY_HEADER)
        assert float(second.headers[COST_HEADER]) == 0.0


def test_voyage_receives_only_the_inputs_that_are_not_cached(gateway: Gateway) -> None:
    """The caller embeds `[a]` on Voyage, then `[a, b]`.

    Embedding inputs are cached one by one, so the second call must send only `input: [b]` to Voyage, and the caller
    must still get two entries indexed 0 and 1 in request order.
    """
    first, second = marker(), marker()
    with peer(_voyage_usage) as provider, gateway.scenario() as scenario:
        model: Final = _voyage(scenario, provider)
        served(embed(gateway, model, [first]))
        body: Final = served(embed(gateway, model, [first, second]))
        calls: Final = provider.posts()
        assert len(calls) == 2
        assert calls[1].body["input"] == [second]
        data: Final = body["data"]
        assert isinstance(data, list)
        assert [object_value(entry)["index"] for entry in data] == [0, 1]


def test_cohere_receives_only_the_uncached_texts(gateway: Gateway) -> None:
    """The caller embeds `[a]` on Cohere, then `[b, a]`.

    The second Cohere request must carry `texts: [b]` only, while the caller gets two entries in request order.
    UNVERIFIED: per-input caching was read for the OpenAI path; Cohere goes through the same
    `_async_get_cache` list handling in `litellm/caching/caching_handler.py`.
    """
    first, second = marker(), marker()
    with peer(cohere_embed()) as provider, gateway.scenario() as scenario:
        model: Final = _cohere(scenario, provider)
        served(embed(gateway, model, [first]))
        body: Final = served(embed(gateway, model, [second, first]))
        calls: Final = provider.posts()
        assert len(calls) == 2
        assert calls[1].body["texts"] == [second]
        data: Final = body["data"]
        assert isinstance(data, list) and len(data) == 2


def test_the_same_input_on_two_providers_is_cached_separately(gateway: Gateway) -> None:
    """The same string is embedded on a Voyage deployment and then on a Cohere deployment.

    The cache key includes the model, so Cohere must still be called once and the Cohere answer carries no
    cache key header.
    """
    text: Final = marker()
    with peer(_voyage_usage) as voyage, peer(cohere_embed()) as cohere, gateway.scenario() as scenario:
        served(embed(gateway, _voyage(scenario, voyage), [text]))
        answer: Final = embed(gateway, _cohere(scenario, cohere), [text])
        served(answer)
        assert len(cohere.posts()) == 1
        assert not answer.headers.get(CACHE_KEY_HEADER)


def test_a_rerank_with_a_different_top_n_is_not_served_from_the_cache(gateway: Gateway) -> None:
    """The same Jina rerank (same query and documents) is sent with `top_n: 1` and then `top_n: 2`.

    `top_n` is part of the rerank cache key, so Jina must be called twice.
    """
    query: Final = marker()
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"jina_ai/{_JINA_RERANK}", provider)
        served(rerank(gateway, model, query=query, top_n=1))
        served(rerank(gateway, model, query=query, top_n=2))
        assert len(provider.posts()) == 2


def test_a_rerank_with_reordered_documents_is_not_served_from_the_cache(gateway: Gateway) -> None:
    """The same query is reranked over `[d0, d1, d2]` and then `[d2, d1, d0]`.

    Result indexes refer to document positions, so a cached answer would point at the wrong documents: the provider
    must be called twice.
    """
    query: Final = marker()
    documents: Final[list[JsonValue]] = [marker(), marker(), marker()]
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"cohere/{_COHERE_RERANK}", provider)
        served(rerank(gateway, model, query=query, documents=documents))
        served(rerank(gateway, model, query=query, documents=list(reversed(documents))))
        assert len(provider.posts()) == 2


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank", "/v2/rerank"])
def test_every_rerank_route_reaches_a_non_cohere_provider_the_same_way(gateway: Gateway, path: str) -> None:
    """A hosted vLLM reranker is called through each of the three public rerank routes.

    The route only names the caller's API flavour: vLLM must receive one `POST /rerank` with the same body each
    time, and the caller gets the results in vLLM's order.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"hosted_vllm/{_VLLM_RERANK}", provider)
        body: Final = served(rerank(gateway, model, path=path))
        assert provider.only().path == "/rerank"
        assert [index for index, _ in results(body)] == [2, 0, 1]


def test_rerank_results_keep_the_provider_order_and_top_n_is_not_applied_by_the_proxy(gateway: Gateway) -> None:
    """The caller asks `top_n: 1`; the provider (ignoring it) answers three results ordered 2, 0, 1.

    The proxy does not re-sort or truncate results: the caller must get all three in the provider's order. Pinned
    so a port does not start trimming silently.
    """
    with peer(cohere_rerank()) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"cohere/{_COHERE_RERANK}", provider)
        body: Final = served(rerank(gateway, model, top_n=1))
        assert [index for index, _ in results(body)] == [2, 0, 1]
        assert provider.only().body["top_n"] == 1


def test_azure_ai_rerank_cost_is_the_query_price_times_search_units(gateway: Gateway) -> None:
    """An Azure AI rerank deployment priced `input_cost_per_query: 0.002`; Azure bills `search_units: 2`.

    `x-litellm-response-cost` must be 0.004 and the spend row for the call id must have `call_type: "arerank"`,
    `custom_llm_provider: "azure_ai"` and `spend` 0.004.
    """
    call_id: Final = new_call_id()
    with peer(cohere_rerank(2)) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"azure_ai/{_AZURE_RERANK}", provider, input_cost_per_query=QUERY_RATE)
        response: Final = rerank(gateway, model, headers=call_header(call_id))
        served(response)
        assert float(response.headers[COST_HEADER]) == pytest.approx(0.004)
        row: Final = spend_row(gateway, call_id)
        assert (row["call_type"], row["custom_llm_provider"]) == ("arerank", "azure_ai")
        assert number(row["spend"]) == pytest.approx(0.004)


@pytest.mark.parametrize(
    ("litellm_model", "usage"),
    [
        pytest.param(f"hosted_vllm/{_VLLM_RERANK}", {"total_tokens": 15}, id="hosted-vllm"),
        pytest.param(
            "infinity/mixedbread-ai/mxbai-rerank-xsmall-v1", {"prompt_tokens": 7, "total_tokens": 7}, id="infinity"
        ),
        pytest.param("together_ai/Salesforce/Llama-Rank-V1", {"prompt_tokens": 7, "total_tokens": 7}, id="together"),
    ],
)
def test_a_per_query_price_bills_nothing_when_the_provider_reports_no_search_units(
    gateway: Gateway, litellm_model: str, usage: dict[str, JsonValue]
) -> None:
    """A rerank deployment priced `input_cost_per_query: 0.002` on a provider that reports only token `usage`.

    `BaseRerankConfig.calculate_rerank_cost` returns 0 when `billed_units.search_units` is missing, so today the
    cost header and spend are 0 even though one query was served. Pinned as current behaviour (see Suspected bugs
    in the index): a port that starts billing one query here changes customer invoices.
    """

    def respond(call: Call) -> Reply:
        return json_reply({"id": "dec1-usage", "results": [{"index": 0, "relevance_score": 0.7}], "usage": usage})

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, litellm_model, provider, input_cost_per_query=QUERY_RATE)
        response: Final = rerank(gateway, model)
        served(response)
        assert float(response.headers[COST_HEADER]) == 0.0


def test_a_failed_rerank_on_a_non_cohere_provider_is_answered_with_its_status(gateway: Gateway) -> None:
    """Azure AI answers the rerank with HTTP 503 `dec1 azure rerank overloaded`.

    The caller must get 503 with that text and an `x-litellm-call-id` header.
    """
    with peer(failing(503, "dec1 azure rerank overloaded")) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, f"azure_ai/{_AZURE_RERANK}", provider)
        response: Final = rerank(gateway, model)
        assert response.status_code == 503, response.text
        assert "dec1 azure rerank overloaded" in response.text
        assert response.headers.get("x-litellm-call-id")


def test_a_voyage_rerank_past_its_deployment_timeout_is_answered_408(gateway: Gateway) -> None:
    """A Voyage rerank deployment with `timeout: 0.5`; Voyage holds its answer for up to 5 seconds.

    The caller must get 408 well before the 5 seconds pass. UNVERIFIED: that the rerank http handler maps the
    httpx read timeout to 408 for Voyage as it does for chat.
    """
    release: Final = threading.Event()
    with peer(held(release, cohere_rerank())) as provider, gateway.scenario() as scenario:
        model: Final = _rerank_model(scenario, "voyage/rerank-2.5", provider, timeout=0.5)
        try:
            response: Final = rerank(gateway, model)
        finally:
            release.set()
        assert response.status_code == 408, response.text
        assert provider.posts()


def test_a_cohere_embedding_past_its_deployment_timeout_is_answered_408(gateway: Gateway) -> None:
    """A Cohere embedding deployment with `timeout: 0.5`; Cohere holds its answer for up to 5 seconds.

    The caller must get 408 before the hold ends.
    """
    release: Final = threading.Event()
    with peer(held(release, cohere_embed())) as provider, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"cohere/{_COHERE}", api_base=f"{provider.url}/v2/embed", api_key=PROVIDER_KEY, timeout=0.5
        )
        try:
            response: Final = embed(gateway, model, [marker()])
        finally:
            release.set()
        assert response.status_code == 408, response.text


def test_request_body_fallbacks_move_a_failed_rerank_to_another_provider(gateway: Gateway) -> None:
    """A Cohere rerank group whose provider answers 500, and a Jina rerank group; the caller sends
    `fallbacks: [<jina group>]` in the rerank body.

    The caller must get 200 with Jina's results, Cohere must have been tried, and the response header
    `x-litellm-model-group` must name the Jina group. UNVERIFIED: the header value after a fallback.
    """
    with (
        peer(failing(500)) as cohere,
        peer(cohere_rerank()) as jina,
        gateway.scenario() as scenario,
    ):
        primary: Final = _rerank_model(scenario, f"cohere/{_COHERE_RERANK}", cohere)
        backup: Final = _rerank_model(scenario, f"jina_ai/{_JINA_RERANK}", jina)
        response: Final = rerank(gateway, primary, fallbacks=[backup])
        body: Final = served(response)
        assert [index for index, _ in results(body)] == [2, 0, 1]
        assert cohere.posts() and len(jina.posts()) == 1
        assert response.headers.get("x-litellm-model-group") == backup


def test_request_body_fallbacks_move_a_failed_embedding_to_another_provider(gateway: Gateway) -> None:
    """A Cohere embedding group whose provider answers 503, and a Voyage group; the caller sends
    `fallbacks: [<voyage group>]`.

    The caller must get 200 with one vector, and Voyage must receive `input: [<the text>]`.
    """
    text: Final = marker()
    with peer(failing(503)) as cohere, peer(_voyage_usage) as voyage, gateway.scenario() as scenario:
        primary: Final = _cohere(scenario, cohere)
        backup: Final = _voyage(scenario, voyage)
        served(embed(gateway, primary, [text], fallbacks=[backup]))
        assert cohere.posts()
        assert voyage.only().body["input"] == [text]


@pytest.fixture(scope="module")
def ordered(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """An owned proxy with group `dec1-er-embed`: a Cohere deployment `order: 1` whose peer answers 500 and a
    Voyage deployment `order: 2` that answers; group `dec1-er-rerank`: Cohere rerank `order: 1` failing and
    Voyage rerank `order: 2`; a `model_group_alias` `dec1-er-embed-alias` for the embedding group; one retry."""

    def models(first: str, second: str) -> list[dict[str, JsonValue]]:
        return [
            deployment("dec1-er-embed", f"cohere/{_COHERE}", f"{first}/v2/embed", order=1),
            deployment("dec1-er-embed", f"voyage/{_VOYAGE}", f"{second}/v1", order=2),
            deployment("dec1-er-rerank", f"cohere/{_COHERE_RERANK}", first, order=1, model_info={"mode": "rerank"}),
            deployment("dec1-er-rerank", "voyage/rerank-2.5", second, order=2, model_info={"mode": "rerank"}),
        ]

    def secondary(call: Call) -> Reply:
        if call.path.endswith("/rerank"):
            return json_reply(
                {"data": [{"index": 1, "relevance_score": 0.8}], "model": "rerank-2.5", "usage": {"total_tokens": 4}}
            )
        return _voyage_usage(call)

    with owned_rig(
        Path(tmp_path_factory.mktemp("dec1-embedrerank-ordered")),
        models,
        primary=lambda call: error_reply(500, "dec1 cohere down"),
        secondary=secondary,
        router_settings={"num_retries": 1, "model_group_alias": {"dec1-er-embed-alias": "dec1-er-embed"}},
    ) as rig:
        yield rig


def test_an_embedding_moves_from_a_failing_cohere_deployment_to_the_voyage_deployment_of_its_group(
    ordered: Rig,
) -> None:
    """On the owned proxy, `dec1-er-embed` has Cohere at `order: 1` (answers 500) and Voyage at `order: 2`.

    With one retry the caller must get 200, and the Voyage peer must have received the input after Cohere was
    tried. The answer's usage is Voyage's (`prompt_tokens: 11`).
    """
    text: Final = marker()
    body: Final = served(embed(ordered.gateway, "dec1-er-embed", [text]))
    assert object_value(body["usage"])["prompt_tokens"] == 11
    assert any(call.body.get("texts") == [text] for call in ordered.primary.posts())
    assert any(call.body.get("input") == [text] for call in ordered.secondary.posts())


def test_a_rerank_moves_from_a_failing_cohere_deployment_to_the_voyage_deployment_of_its_group(
    ordered: Rig,
) -> None:
    """On the owned proxy, `dec1-er-rerank` has Cohere rerank at `order: 1` (answers 500) and Voyage at `order: 2`.

    The caller must get 200 with Voyage's single result `(1, 0.8)` mapped into Cohere's `results` shape.
    """
    query: Final = marker()
    body: Final = served(rerank(ordered.gateway, "dec1-er-rerank", query=query))
    assert results(body) == [(1, 0.8)]
    assert any(call.body.get("query") == query for call in ordered.secondary.posts())


def test_a_model_group_alias_resolves_to_the_embedding_group(ordered: Rig) -> None:
    """The caller embeds on `dec1-er-embed-alias`, a `router_settings.model_group_alias` for `dec1-er-embed`.

    The request must be served by the aliased group (Voyage answers after Cohere fails), so the caller gets 200
    and the Voyage peer receives the text.
    """
    text: Final = marker()
    served(embed(ordered.gateway, "dec1-er-embed-alias", [text]))
    assert any(call.body.get("input") == [text] for call in ordered.secondary.posts())
