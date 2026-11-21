"""Grounding and retrieval on Vertex AI: Gemini grounding tools, RAG Engine search, and Vertex AI Search.

Three surfaces put Google retrieval behind the proxy:

- Gemini grounding tools on `/v1/chat/completions`. `VertexGeminiConfig._map_function`
  (`litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:1316`) turns each built-in tool into its
  own Vertex `Tool` object (`enterpriseWebSearch`, `googleSearchRetrieval`, ...) and drops search tools when
  function declarations are present. Google Search, URL context and code execution are pinned for AI Studio in
  `test_dec1_gemini_tools_thinking_media_wire.py`; this file adds the Vertex-only Enterprise Web Search, the
  legacy `googleSearchRetrieval`, and Vertex AI Search datastore grounding (`retrieval.vertexAiSearch`).
- Vertex AI RAG Engine as a vector store (`custom_llm_provider: vertex_ai/rag_api`).
  `/v1/vector_stores/{id}/search` posts `{api_base}:retrieveContexts` with a `vertex_rag_store` naming the
  corpus and maps `contexts.contexts[]` to OpenAI search results
  (`litellm/llms/vertex_ai/vector_stores/rag_api/transformation.py`).
- Vertex AI Search (Discovery Engine) as a vector store (`vertex_ai/search_api`): `{api_base}:search` with
  `query` and `pageSize`, an `extra_body` allowlist that refuses target-selecting fields, and two hit shapes
  (documents and chunks) (`litellm/llms/vertex_ai/vector_stores/search_api/transformation.py`).

Token reuse across searches is pinned in `test_vector_store_search_wire.py`; this file pins the request and
response mapping, which is what a Rust port of vector store search would have to reproduce.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_vertex3_support import (
    is_token_from,
    json_of,
    open_peer,
    register_vector_store,
)
from integration.providers.dec1_vertex_support import (
    LOCATION,
    PROJECT,
    body_of,
    gemini_echo,
    marker,
    path_of,
    vertex,
)
from pydantic import JsonValue

_FUNCTION_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
}
_RAG_BASE: Final = f"/v1/projects/{PROJECT}/locations/{LOCATION}"
_SEARCH_BASE: Final = f"/v1/projects/{PROJECT}/locations/global/collections/default_collection/dataStores/dec1-ds/servingConfigs/default_config"


def _tools_sent(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [object_value(tool) for tool in list_value(body["tools"])]


# ---------------------------------------------------------------- Gemini grounding tools on Vertex


def test_enterprise_web_search_is_sent_as_its_own_vertex_tool(gateway: Gateway) -> None:
    """A tool `{"enterpriseWebSearch": {}}` reaches `generateContent` as exactly one tool object
    `{"enterpriseWebSearch": {}}` (Vertex requires each built-in tool in its own `Tool`)."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"enterpriseWebSearch": {}}]).status_code == 200
        assert _tools_sent(rig.peer.one_body()) == [{"enterpriseWebSearch": {}}]


def test_the_snake_case_enterprise_web_search_spelling_is_accepted(gateway: Gateway) -> None:
    """`{"enterprise_web_search": {"excludeDomains": ["example.com"]}}` is read through `get_tool_value`'s
    snake case fallback and sent as `{"enterpriseWebSearch": {"excludeDomains": ["example.com"]}}`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"enterprise_web_search": {"excludeDomains": ["example.com"]}}]).status_code == 200
        assert _tools_sent(rig.peer.one_body()) == [{"enterpriseWebSearch": {"excludeDomains": ["example.com"]}}]


def test_google_search_retrieval_config_is_forwarded_verbatim(gateway: Gateway) -> None:
    """The legacy `googleSearchRetrieval` tool with a `dynamicRetrievalConfig` is forwarded as one tool
    object with the same config, mode and threshold untouched."""
    config: Final[dict[str, JsonValue]] = {"dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC", "dynamicThreshold": 0.7}}
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"googleSearchRetrieval": config}]).status_code == 200
        assert _tools_sent(rig.peer.one_body()) == [{"googleSearchRetrieval": config}]


def test_enterprise_web_search_next_to_a_function_is_dropped(gateway: Gateway) -> None:
    """Vertex rejects search tools mixed with function declarations, so with both an
    `enterpriseWebSearch` tool and a function tool only `function_declarations` is sent
    (`_drop_search_tools_mixed_with_functions`) and the call still succeeds."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"enterpriseWebSearch": {}}, _FUNCTION_TOOL]).status_code == 200
        tools: Final = _tools_sent(rig.peer.one_body())
        assert [list(tool) for tool in tools] == [["function_declarations"]]


def test_a_vertex_ai_search_datastore_tool_reaches_vertex(gateway: Gateway) -> None:
    """Grounding with a Vertex AI Search datastore is a `retrieval` tool:
    `{"retrieval": {"vertexAiSearch": {"datastore": "projects/.../dataStores/dec1-ds"}}}` should reach
    `generateContent` as that same tool object. Today `_map_function` has no `retrieval` branch, so the tool
    falls into the "Invalid tool" warning and is silently dropped, leaving an ungrounded call."""
    pytest.skip("BUG: a retrieval.vertexAiSearch grounding tool is dropped with only a warning log")
    datastore: Final = f"projects/{PROJECT}/locations/global/collections/default_collection/dataStores/dec1-ds"
    tool: Final[dict[str, JsonValue]] = {"retrieval": {"vertexAiSearch": {"datastore": datastore}}}
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[tool]).status_code == 200
        assert _tools_sent(rig.peer.one_body()) == [tool]


def test_a_vertex_rag_store_retrieval_tool_reaches_vertex(gateway: Gateway) -> None:
    """Grounding a Gemini call on a RAG Engine corpus is the `retrieval.vertexRagStore` tool; it should be
    forwarded unchanged. Like `vertexAiSearch` it has no branch in `_map_function` and is dropped today."""
    pytest.skip("BUG: a retrieval.vertexRagStore grounding tool is dropped with only a warning log")
    corpus: Final = f"projects/{PROJECT}/locations/{LOCATION}/ragCorpora/dec1-corpus"
    tool: Final[dict[str, JsonValue]] = {
        "retrieval": {"vertexRagStore": {"ragResources": [{"ragCorpus": corpus}], "similarityTopK": 3}}
    }
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[tool]).status_code == 200
        assert _tools_sent(rig.peer.one_body()) == [tool]


# ---------------------------------------------------------------- RAG Engine (`vertex_ai/rag_api`)


def _rag_reply(contexts: list[dict[str, JsonValue]]) -> Callable[[Request], Reply]:
    return lambda _request: Reply(body=json.dumps({"contexts": {"contexts": contexts}}).encode())


def _store_id() -> str:
    return f"dec1-corpus-{uuid.uuid4().hex[:12]}"


def _search(gateway: Gateway, store_id: str, body: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", f"/v1/vector_stores/{store_id}/search", body)


def test_a_rag_search_posts_retrieve_contexts_naming_the_corpus(gateway: Gateway) -> None:
    """`/v1/vector_stores/<id>/search` on a `vertex_ai/rag_api` store posts once to
    `<api_base>:retrieveContexts` with the minted bearer and a body whose
    `vertex_rag_store.rag_resources[0].rag_corpus` is `projects/dec1-vertex-project/locations/us-central1/
    ragCorpora/<id>` (the store id expanded under the configured project and location) and `query.text` is
    the caller's query."""
    store: Final = _store_id()
    query: Final = f"refund policy {marker()}"
    with open_peer(_rag_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "rag_api", f"{peer.url}{_RAG_BASE}", peer.url)
        response: Final = _search(gateway, store, {"query": query})
        assert response.status_code == 200, response.text
        calls: Final = peer.calls()
        assert [(call.method, path_of(call)) for call in calls] == [("POST", f"{_RAG_BASE}:retrieveContexts")]
        assert is_token_from(peer.plan, calls[0])
        body: Final = body_of(calls[0])
        assert body["vertex_rag_store"] == {
            "rag_resources": [{"rag_corpus": f"projects/{PROJECT}/locations/{LOCATION}/ragCorpora/{store}"}]
        }
        assert object_value(body["query"])["text"] == query


def test_a_rag_search_joins_a_list_query_with_spaces(gateway: Gateway) -> None:
    """A list query `["refund", "policy"]` is sent as the single `query.text` `"refund policy"`, since
    `retrieveContexts` takes one text."""
    store: Final = _store_id()
    with open_peer(_rag_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "rag_api", f"{peer.url}{_RAG_BASE}", peer.url)
        assert _search(gateway, store, {"query": ["refund", "policy"]}).status_code == 200
        assert object_value(body_of(peer.calls()[0])["query"])["text"] == "refund policy"


def test_rag_search_options_become_the_rag_retrieval_config(gateway: Gateway) -> None:
    """`max_num_results: 4`, `filters` and `ranking_options` are sent under `query.rag_retrieval_config` as
    `top_k: 4`, `filter` and `ranking` (each passed through as given); without them the key is absent."""
    store: Final = _store_id()
    filters: Final[dict[str, JsonValue]] = {"type": "eq", "key": "tenant", "value": "dec1"}
    ranking: Final[dict[str, JsonValue]] = {"ranker": "auto", "score_threshold": 0.3}
    with open_peer(_rag_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "rag_api", f"{peer.url}{_RAG_BASE}", peer.url)
        assert _search(gateway, store, {"query": f"q {marker()}"}).status_code == 200
        assert (
            _search(
                gateway,
                store,
                {"query": f"q {marker()}", "max_num_results": 4, "filters": filters, "ranking_options": ranking},
            ).status_code
            == 200
        )
        bare, tuned = (object_value(body_of(call)["query"]) for call in peer.calls())
        assert "rag_retrieval_config" not in bare
        assert tuned["rag_retrieval_config"] == {"top_k": 4, "filter": filters, "ranking": ranking}


def test_rag_contexts_become_openai_search_results(gateway: Gateway) -> None:
    """A context `{text, score: 0.82, sourceUri, sourceDisplayName, pageSpan}` comes back as a result with
    `score: 0.82`, `content: [{"type": "text", "text": <text>}]`, `file_id` the `sourceUri`, `filename` the
    `sourceDisplayName`, and `attributes` holding the three source fields; the page object is
    `vector_store.search_results.page` and echoes the query as `search_query`."""
    store: Final = _store_id()
    query: Final = f"refunds {marker()}"
    context: Final[dict[str, JsonValue]] = {
        "text": "Refunds are issued within 14 days.",
        "score": 0.82,
        "sourceUri": "gs://dec1-bucket/policy.pdf",
        "sourceDisplayName": "policy.pdf",
        "pageSpan": {"firstPage": 2, "lastPage": 3},
    }
    with open_peer(_rag_reply([context])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "rag_api", f"{peer.url}{_RAG_BASE}", peer.url)
        response: Final = _search(gateway, store, {"query": query})
        assert response.status_code == 200, response.text
        page: Final = json_of(response)
        assert (page["object"], page["search_query"]) == ("vector_store.search_results.page", query)
        result: Final = object_value(list_value(page["data"])[0])
        assert result["score"] == 0.82
        assert result["content"] == [{"type": "text", "text": "Refunds are issued within 14 days."}]
        assert (result["file_id"], result["filename"]) == ("gs://dec1-bucket/policy.pdf", "policy.pdf")
        assert result["attributes"] == {
            "sourceUri": "gs://dec1-bucket/policy.pdf",
            "sourceDisplayName": "policy.pdf",
            "pageSpan": {"firstPage": 2, "lastPage": 3},
        }


def test_a_rag_context_without_a_source_is_an_unknown_document(gateway: Gateway) -> None:
    """A context with only `text` (no score, no source) maps to `score: 0.0`, `file_id: ""`,
    `filename: "Unknown Document"` and empty `attributes`."""
    store: Final = _store_id()
    with open_peer(_rag_reply([{"text": "orphan chunk"}])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "rag_api", f"{peer.url}{_RAG_BASE}", peer.url)
        response: Final = _search(gateway, store, {"query": f"q {marker()}"})
        assert response.status_code == 200, response.text
        result: Final = object_value(list_value(json_of(response)["data"])[0])
        assert (result["score"], result["file_id"], result["filename"], result["attributes"]) == (
            0.0,
            "",
            "Unknown Document",
            {},
        )


# ---------------------------------------------------------------- Vertex AI Search (`vertex_ai/search_api`)


def _search_reply(results: list[dict[str, JsonValue]]) -> Callable[[Request], Reply]:
    return lambda _request: Reply(body=json.dumps({"results": results}).encode())


def test_a_vertex_ai_search_posts_query_and_default_page_size(gateway: Gateway) -> None:
    """A `vertex_ai/search_api` store posts once to `<api_base>:search` with the minted bearer and the body
    `{"query": <query>, "pageSize": 10}` (10 is the default when `max_num_results` is not given)."""
    store: Final = _store_id()
    query: Final = f"shipping {marker()}"
    with open_peer(_search_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url)
        assert _search(gateway, store, {"query": query}).status_code == 200
        call: Final = peer.calls()[0]
        assert path_of(call) == f"{_SEARCH_BASE}:search"
        assert is_token_from(peer.plan, call)
        assert body_of(call) == {"query": query, "pageSize": 10}


def test_vertex_ai_search_max_num_results_becomes_page_size(gateway: Gateway) -> None:
    """`max_num_results: 3` is sent as `pageSize: 3`."""
    store: Final = _store_id()
    with open_peer(_search_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url)
        assert _search(gateway, store, {"query": f"q {marker()}", "max_num_results": 3}).status_code == 200
        assert body_of(peer.calls()[0])["pageSize"] == 3


def test_vertex_ai_search_extra_body_fields_on_the_allowlist_are_merged(gateway: Gateway) -> None:
    """`extra_body` with `filter` and `contentSearchSpec` (both on the data store allowlist,
    `VertexSearchDataStoreExtraBody`) are merged into the `:search` body next to `query` and `pageSize`.
    UNVERIFIED: that the proxy hands the caller's top level `extra_body` to the transformation
    (`with_caller_query_params` in `litellm/proxy/vector_store_endpoints/endpoints.py:196`)."""
    store: Final = _store_id()
    spec: Final[dict[str, JsonValue]] = {"searchResultMode": "CHUNKS"}
    with open_peer(_search_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url)
        response: Final = _search(
            gateway,
            store,
            {"query": f"q {marker()}", "extra_body": {"filter": 'lang: ANY("en")', "contentSearchSpec": spec}},
        )
        assert response.status_code == 200, response.text
        body: Final = body_of(peer.calls()[0])
        assert (body["filter"], body["contentSearchSpec"]) == ('lang: ANY("en")', spec)


def test_vertex_ai_search_refuses_a_target_selecting_extra_body_field(gateway: Gateway) -> None:
    """`extra_body: {"servingConfig": ...}` would search another target than the registered store, so the
    proxy answers 400 naming "target-selecting fields" and never calls Google."""
    store: Final = _store_id()
    with open_peer(_search_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url)
        response: Final = _search(
            gateway,
            store,
            {"query": f"q {marker()}", "extra_body": {"servingConfig": "projects/other/servingConfigs/x"}},
        )
        assert response.status_code == 400, response.text
        assert "target-selecting fields" in response.text
        assert peer.calls() == ()


def test_vertex_ai_search_refuses_engine_only_fields_on_a_data_store(gateway: Gateway) -> None:
    """`dataStoreSpecs` is only meaningful for an engine (search app); on a data store store it is answered
    with 400 `Unsupported Vertex AI Search extra_body fields ['dataStoreSpecs'] for data store mode` and no
    upstream call."""
    store: Final = _store_id()
    with open_peer(_search_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url)
        response: Final = _search(gateway, store, {"query": f"q {marker()}", "extra_body": {"dataStoreSpecs": []}})
        assert response.status_code == 400, response.text
        assert "Unsupported Vertex AI Search extra_body fields ['dataStoreSpecs'] for data store mode" in response.text
        assert peer.calls() == ()


def test_vertex_ai_search_accepts_engine_only_fields_when_an_engine_is_configured(gateway: Gateway) -> None:
    """With `vertex_engine_id` in the store's params the engine allowlist applies, so
    `numResultsPerDataStore: 2` is forwarded in the `:search` body."""
    store: Final = _store_id()
    with open_peer(_search_reply([])) as peer, gateway.scenario() as scenario:
        register_vector_store(
            gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url, vertex_engine_id="dec1-app"
        )
        response: Final = _search(
            gateway, store, {"query": f"q {marker()}", "extra_body": {"numResultsPerDataStore": 2}}
        )
        assert response.status_code == 200, response.text
        assert body_of(peer.calls()[0])["numResultsPerDataStore"] == 2


def test_vertex_ai_search_document_hits_are_scored_by_position(gateway: Gateway) -> None:
    """Document hits carry no relevance score, so the first hit is returned with `score: 1.0` and the
    second with `0.5` (`1 / (position + 1)`); text comes from `extractive_segments` first, `file_id` is the
    `link`, `filename` the `title`, and `attributes` hold `document_id`, `link` and `title`."""
    store: Final = _store_id()
    hits: Final[list[dict[str, JsonValue]]] = [
        {
            "id": "doc-1",
            "document": {
                "id": "doc-1",
                "derivedStructData": {
                    "title": "Shipping",
                    "link": "gs://dec1-bucket/shipping.pdf",
                    "extractive_segments": [{"content": "Ships in 2 days."}],
                    "snippets": [{"snippet": "not this one"}],
                },
            },
        },
        {"id": "doc-2", "document": {"id": "doc-2", "derivedStructData": {"title": "Returns"}}},
    ]
    with open_peer(_search_reply(hits)) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url)
        response: Final = _search(gateway, store, {"query": f"q {marker()}"})
        assert response.status_code == 200, response.text
        first, second = (object_value(item) for item in list_value(json_of(response)["data"]))
        assert (first["score"], second["score"]) == (1.0, 0.5)
        assert first["content"] == [{"type": "text", "text": "Ships in 2 days."}]
        assert (first["file_id"], first["filename"]) == ("gs://dec1-bucket/shipping.pdf", "Shipping")
        assert first["attributes"] == {
            "document_id": "doc-1",
            "link": "gs://dec1-bucket/shipping.pdf",
            "title": "Shipping",
        }
        assert (second["file_id"], second["content"]) == ("doc-2", [{"type": "text", "text": "Returns"}])


def test_vertex_ai_search_chunk_hits_use_their_relevance_score_and_document(gateway: Gateway) -> None:
    """A chunk hit (`searchResultMode: CHUNKS`) keeps its `relevanceScore` (0.66), takes its text from
    `chunk.content`, its `file_id` from `documentMetadata.uri`, its `filename` from
    `documentMetadata.title`, and its `document_id` from the chunk `name` segment after `/documents/`."""
    store: Final = _store_id()
    chunk: Final[dict[str, JsonValue]] = {
        "chunk": {
            "id": "c-9",
            "name": f"projects/{PROJECT}/locations/global/collections/default_collection/dataStores/dec1-ds/branches/0/documents/doc-7/chunks/c-9",
            "content": "Chunk body text.",
            "relevanceScore": 0.66,
            "documentMetadata": {"uri": "gs://dec1-bucket/guide.pdf", "title": "Guide"},
        }
    }
    with open_peer(_search_reply([chunk])) as peer, gateway.scenario() as scenario:
        register_vector_store(gateway, scenario, store, "search_api", f"{peer.url}{_SEARCH_BASE}", peer.url)
        response: Final = _search(gateway, store, {"query": f"q {marker()}"})
        assert response.status_code == 200, response.text
        result: Final = object_value(list_value(json_of(response)["data"])[0])
        assert result["score"] == 0.66
        assert result["content"] == [{"type": "text", "text": "Chunk body text."}]
        assert (result["file_id"], result["filename"]) == ("gs://dec1-bucket/guide.pdf", "Guide")
        attributes: Final = object_value(result["attributes"])
        assert (attributes["document_id"], attributes["chunk_id"]) == ("doc-7", "c-9")
