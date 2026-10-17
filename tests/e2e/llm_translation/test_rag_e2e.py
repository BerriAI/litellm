"""/v1/rag/ingest and /v1/rag/query through the gateway against real vector stores."""

from __future__ import annotations

import base64
import re
import time
from typing import Final

import pytest
from e2e_config import CHEAP_OPENAI_MODEL, POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import NoBody, unwrap
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    ChatMessage,
    LiteLLMParamsBody,
    OpenAIRagVectorStore,
    RagIngestBody,
    RagIngestOptions,
    RagInlineFile,
    RagQueryBody,
    RagQueryResponse,
    RagRerankConfig,
    RagRetrievalConfig,
    RagSearchHit,
    RagVectorStore,
    VectorStoreDeleteBody,
    VectorStoreFileStatus,
    VertexRagVectorStore,
)
from proxy_client import ProxyClient

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

COHERE_RERANK_MODEL: Final = "cohere/rerank-v4.0-fast"
VERTEX_CORPUS_ID: Final = re.compile(r"^projects/[^/]+/locations/[^/]+/ragCorpora/[^/]+$")


def _document(codename: str) -> RagInlineFile:
    text: Final = f"LiteLLM e2e RAG document.\nThe secret project codename is {codename}.\n"
    return RagInlineFile(
        filename=f"rag-{codename}.txt",
        content=base64.b64encode(text.encode()).decode(),
        content_type="text/plain",
    )


def _ingest(
    proxy: ProxyClient, resources: ResourceManager, key: str, codename: str, vector_store: RagVectorStore
) -> tuple[str, str]:
    ingested: Final = unwrap(
        proxy.rag_ingest(
            key,
            RagIngestBody(
                file=_document(codename),
                ingest_options=RagIngestOptions(name=f"e2e-rag-{codename}", vector_store=vector_store),
            ),
        )
    )
    if ingested.vector_store_id:
        resources.defer(
            lambda: proxy.transport.post(
                "/vector_store/delete",
                headers=proxy.transport.master,
                json=VectorStoreDeleteBody(vector_store_id=ingested.vector_store_id),
                response_type=NoBody,
            )
        )
    assert ingested.status == "completed", f"ingest did not complete: {ingested}"
    assert ingested.file_id, f"ingest returned no file id: {ingested}"
    return ingested.vector_store_id, ingested.file_id


def _ingest_into_openai_store(proxy: ProxyClient, resources: ResourceManager, key: str, codename: str) -> str:
    store_id, file_id = _ingest(proxy, resources, key, codename, OpenAIRagVectorStore(ttl_days=1))
    resources.defer(
        lambda: proxy.transport.delete(
            f"/v1/vector_stores/{store_id}", headers=proxy.transport.bearer(key), json=NoBody(), response_type=NoBody
        )
    )
    resources.defer(
        lambda: proxy.transport.delete(
            f"/v1/files/{file_id}", headers=proxy.transport.bearer(key), json=NoBody(), response_type=NoBody
        )
    )
    _await_indexed(proxy, key, store_id, file_id)
    return store_id


def _await_indexed(proxy: ProxyClient, key: str, store_id: str, file_id: str) -> None:
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        indexed = unwrap(
            proxy.transport.get(
                f"/v1/vector_stores/{store_id}/files/{file_id}",
                headers=proxy.transport.bearer(key),
                params=NoBody(),
                response_type=VectorStoreFileStatus,
            )
        )
        if indexed.status == "completed":
            return
        assert indexed.status == "in_progress", f"file {file_id} stopped indexing in {store_id}: {indexed}"
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"file {file_id} was not indexed in {store_id} within {POLL_TIMEOUT}s")


def _chat_model(proxy: ProxyClient, resources: ResourceManager) -> str:
    model: Final = f"e2e-rag-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model, LiteLLMParamsBody(model=f"openai/{CHEAP_OPENAI_MODEL}", api_key="os.environ/OPENAI_API_KEY")
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _question(store_id: str, model: str, rerank: RagRerankConfig | None = None) -> RagQueryBody:
    return RagQueryBody(
        model=model,
        messages=(
            ChatMessage(role="user", content="What is the secret project codename? Reply with only the codename."),
        ),
        retrieval_config=RagRetrievalConfig(vector_store_id=store_id, top_k=5),
        rerank=rerank,
    )


def _retrieved_text(response: RagQueryResponse) -> str:
    hits: Final = response.choices[0].message.provider_specific_fields.search_results.data
    return " ".join(_hit_text(hit) for hit in hits)


def _hit_text(hit: RagSearchHit) -> str:
    return " ".join(part.text for part in hit.content)


class TestRagOpenAI:
    @pytest.mark.covers("llm.rag.openai.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RAG,
            providers=(Provider.OPENAI,),
            models=(CHEAP_OPENAI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_query_answers_from_the_ingested_document(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key: Final = resources.key()
        codename: Final = f"heron-{unique_marker()}"
        store_id: Final = _ingest_into_openai_store(proxy, resources, key, codename)

        answer: Final = unwrap(proxy.rag_query(key, _question(store_id, _chat_model(proxy, resources))))

        assert codename in _retrieved_text(answer), f"search_results do not carry the ingested document: {answer}"
        assert codename in (answer.choices[0].message.content or ""), f"the answer ignored the context: {answer}"

    @pytest.mark.covers("llm.rag.openai.rerank.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RAG,
            providers=(Provider.OPENAI, Provider.COHERE),
            models=(CHEAP_OPENAI_MODEL, COHERE_RERANK_MODEL),
            mode=Mode.NONSTREAM,
        )
    )
    def test_query_reranks_the_ingested_document(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key: Final = resources.key()
        codename: Final = f"osprey-{unique_marker()}"
        store_id: Final = _ingest_into_openai_store(proxy, resources, key, codename)

        answer: Final = unwrap(
            proxy.rag_query(
                key,
                _question(store_id, _chat_model(proxy, resources), RagRerankConfig(model=COHERE_RERANK_MODEL, top_n=3)),
            )
        )

        fields: Final = answer.choices[0].message.provider_specific_fields
        assert codename in _retrieved_text(answer), f"search_results do not carry the ingested document: {answer}"
        assert fields.rerank_results and fields.rerank_results.results, f"no rerank_results returned: {answer}"
        assert codename in (answer.choices[0].message.content or ""), f"the answer ignored the context: {answer}"


class TestRagVertexAI:
    @pytest.mark.covers("llm.rag.vertex.basic.nonstream.works")
    @meta(Subject(domain=Domain.LLM_TRANSLATION, route=Route.RAG, providers=(Provider.VERTEX_AI,), mode=Mode.NONSTREAM))
    def test_ingest_creates_a_corpus_holding_the_file(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key: Final = resources.key()

        corpus_id, _ = _ingest(proxy, resources, key, f"kestrel-{unique_marker()}", VertexRagVectorStore())

        assert VERTEX_CORPUS_ID.match(corpus_id), f"ingest returned no Vertex corpus resource name: {corpus_id!r}"
