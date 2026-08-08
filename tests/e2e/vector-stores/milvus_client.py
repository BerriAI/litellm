"""Client for the Milvus vector-store e2e suite.

The suite exercises the full managed-vector-store flow against a live proxy plus a
live Milvus / Zilliz Cloud cluster:

  1. create a Milvus collection and seed it (Milvus REST v2, embeddings from OpenAI)
  2. register the collection as a litellm managed vector store (POST /vector_store/new)
  3. register a chat model on the proxy (POST /model/new) for the retrieval test
  4. search the store through the proxy (POST /v1/vector_stores/{id}/search)
  5. use it as retrieval context in chat (POST /chat/completions with vector_store_ids)
  6. tear it all down (delete model + managed store, drop the Milvus collection)

Only the three credentials are configurable; every other value (embedding model,
dimensionality, chat model, field names) is fixed here so a run is deterministic.
Milvus-native and OpenAI-embedding calls go through their own HttpTransport so
every HTTP request still funnels through the one requests-owning module
(e2e_http). Milvus answers 200 with a non-zero body ``code`` on failure, so those
are checked explicitly and retried, since Zilliz serverless collections cold-start.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from pydantic import BaseModel

from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, Result, StreamingResponse, is_ok, unwrap
from models import ChatMessage, ChatResponse
from transport import HttpTransport

MILVUS_API_BASE = os.environ.get("MILVUS_API_BASE", "")
MILVUS_API_KEY = os.environ.get("MILVUS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Fixed so the seed vectors, the proxy's query embedding, and the collection
# dimension all agree; text-embedding-3-small is 1536-dimensional.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# A chat model the proxy already serves (with its own working provider key) for
# the retrieval test. Overridable for a proxy configured with different models.
CHAT_MODEL = os.environ.get("E2E_VS_CHAT_MODEL", "gpt-5.5")

# Milvus quick setup names its vector field "vector"; the managed store points
# search at it and reads the seeded text back out of a "text" dynamic field.
VECTOR_FIELD = "vector"
TEXT_FIELD = "text"

MILVUS_MAX_ATTEMPTS = 4
MILVUS_RETRY_SLEEP_SECONDS = 3.0


def credentials_reason() -> str | None:
    """None when the suite has everything it needs, else why it must skip."""
    missing = [
        name
        for name, value in (
            ("MILVUS_API_BASE", MILVUS_API_BASE),
            ("MILVUS_API_KEY", MILVUS_API_KEY),
            ("OPENAI_API_KEY", OPENAI_API_KEY),
        )
        if not value
    ]
    if missing:
        return f"missing env for milvus vector-store e2e: {', '.join(missing)}"
    return None


# ---- Milvus REST v2 models ----------------------------------------------


class MilvusCreateCollectionBody(BaseModel):
    collectionName: str
    dimension: int = EMBEDDING_DIM
    metricType: str = "COSINE"
    autoID: bool = False


class MilvusEntity(BaseModel):
    """One row: ``id`` primary key, ``vector`` embedding, ``text`` + ``category``
    dynamic fields (Milvus quick setup enables dynamic fields). ``category`` gives
    the grouping / multi-output-field tests something to work with."""

    id: int
    vector: list[float]
    text: str
    category: str


class MilvusInsertBody(BaseModel):
    collectionName: str
    data: list[MilvusEntity]


class MilvusDropCollectionBody(BaseModel):
    collectionName: str


class MilvusReply(BaseModel):
    """Milvus wraps every REST reply as ``{code, data?, message?}`` and returns
    HTTP 200 even for logical errors, so ``code == 0`` is the real success test."""

    code: int
    message: str | None = None


class MilvusInsertData(BaseModel):
    insertCount: int = 0


class MilvusInsertReply(MilvusReply):
    data: MilvusInsertData = MilvusInsertData()


# ---- OpenAI embeddings models -------------------------------------------


class OpenAIEmbedBody(BaseModel):
    input: list[str]
    model: str = EMBEDDING_MODEL


class OpenAIEmbeddingItem(BaseModel):
    embedding: list[float]


class OpenAIEmbedResponse(BaseModel):
    data: list[OpenAIEmbeddingItem]


# ---- proxy managed-store + search models --------------------------------


class RegisterStoreBody(BaseModel):
    vector_store_id: str
    custom_llm_provider: str
    vector_store_name: str | None = None
    litellm_params: dict[str, object] | None = None


class RegisterStoreResponse(BaseModel):
    status: str


class DeleteStoreBody(BaseModel):
    vector_store_id: str


class SearchBody(BaseModel):
    query: str | list[str]
    limit: int | None = None
    offset: int | None = None
    filter: str | None = None
    outputFields: list[str] | None = None
    groupingField: str | None = None
    consistencyLevel: str | None = None


class SearchContent(BaseModel):
    type: str
    text: str


class SearchResult(BaseModel):
    score: float
    content: list[SearchContent] = []
    file_id: str | None = None
    filename: str | None = None
    attributes: dict[str, object] = {}


class SearchResponse(BaseModel):
    object: str
    search_query: str = ""
    data: list[SearchResult] = []


class ChatWithVectorStoreBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    vector_store_ids: list[str]
    max_tokens: int | None = None


# ---- CRUD models ---------------------------------------------------------


class InfoBody(BaseModel):
    vector_store_id: str


class UpdateBody(BaseModel):
    vector_store_id: str
    vector_store_name: str | None = None
    vector_store_description: str | None = None
    vector_store_metadata: dict[str, object] | None = None


class ManagedStoreEntry(BaseModel):
    """A row in the managed-store list / info response. Only the fields the tests
    read are declared; pydantic ignores the rest."""

    vector_store_id: str
    custom_llm_provider: str | None = None
    vector_store_name: str | None = None
    vector_store_description: str | None = None
    vector_store_metadata: dict[str, object] | None = None


class ManagedStoreListResponse(BaseModel):
    data: list[ManagedStoreEntry] = []


class ManagedStoreInfoResponse(BaseModel):
    """POST /vector_store/info wraps the entry under a ``vector_store`` key."""

    vector_store: ManagedStoreEntry


class OpenAICompatCreateBody(BaseModel):
    """The OpenAI-compat create request the proxy exposes at
    ``POST /v1/vector_stores``. Milvus's config raises NotImplementedError, so
    this is only used to prove the error surfaces cleanly (no 5xx crash)."""

    name: str


# ---- seeded corpus -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeededDoc:
    text: str
    category: str


@dataclass(frozen=True, slots=True)
class SeededStore:
    """A live, populated managed vector store handed to a test: the store id, the
    exact documents seeded into Milvus (with their category), and the retrieval
    probe for the chat test (a made-up code present in only one document, so an
    answer echoing it proves the store was actually consulted)."""

    vector_store_id: str
    docs: dict[int, SeededDoc]
    secret_code: str
    code_query: str
    code_doc_id: int

    def text(self, doc_id: int) -> str:
        return self.docs[doc_id].text

    def category(self, doc_id: int) -> str:
        return self.docs[doc_id].category

    def ids_in_category(self, category: str) -> set[int]:
        return {doc_id for doc_id, doc in self.docs.items() if doc.category == category}


def seed_probe_collection(
    client: "VectorStoreClient", vector_store_id: str
) -> None:
    """Create a tiny collection and populate it with two rows, one per category,
    for tests that need to observe extra output fields or grouping semantics.
    The caller is responsible for cleanup."""
    client.create_collection(vector_store_id)
    vectors = client.embed(["a coral reef under the sea", "green plants absorb sunlight"])
    client.insert(
        vector_store_id,
        [
            MilvusEntity(id=1, vector=vectors[0], text="a coral reef under the sea", category="geo"),
            MilvusEntity(id=2, vector=vectors[1], text="green plants absorb sunlight", category="science"),
        ],
    )


def build_corpus(marker: str) -> tuple[dict[int, SeededDoc], str, int]:
    """Five short documents keyed by Milvus primary id, split across two categories
    (``geo`` and ``science``, plus one ``secret``). Exactly one holds a unique
    access code; the rest are unrelated so retrieval has to discriminate. Returns
    (docs, secret_code, code_doc_id)."""
    secret_code = f"NIMBUS-{marker[:6].upper()}"
    code_doc_id = 5
    docs = {
        1: SeededDoc("The Great Barrier Reef is the world's largest coral reef system, off the coast of Australia.", "geo"),
        2: SeededDoc("Mount Everest is the highest mountain above sea level, in the Himalayas.", "geo"),
        3: SeededDoc("The Amazon rainforest is the largest tropical rainforest on Earth.", "geo"),
        4: SeededDoc("Photosynthesis is how green plants convert sunlight into chemical energy.", "science"),
        code_doc_id: SeededDoc(f"The Project Nimbus access code is {secret_code}. Nimbus is the internal logistics platform.", "secret"),
    }
    return docs, secret_code, code_doc_id


# ---- client --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VectorStoreClient:
    """Drives the proxy for model + managed-store + search + chat, and Milvus /
    OpenAI directly for seeding. Holds the shared Gateway so the resources fixture
    can clean up any keys the suite creates."""

    gateway: Gateway
    milvus: HttpTransport
    openai: HttpTransport

    # ---- Milvus native (seeding) ----------------------------------------

    def create_collection(self, name: str) -> None:
        self._milvus_call(
            "/v2/vectordb/collections/create",
            MilvusCreateCollectionBody(collectionName=name),
            MilvusReply,
            f"create collection {name}",
        )

    def insert(self, name: str, entities: list[MilvusEntity]) -> int:
        reply = self._milvus_call(
            "/v2/vectordb/entities/insert",
            MilvusInsertBody(collectionName=name, data=entities),
            MilvusInsertReply,
            f"insert into {name}",
        )
        return reply.data.insertCount if reply else 0

    def drop_collection(self, name: str) -> None:
        # Best-effort: a slow Zilliz drop must not error the test run at teardown.
        self._milvus_call(
            "/v2/vectordb/collections/drop",
            MilvusDropCollectionBody(collectionName=name),
            MilvusReply,
            f"drop collection {name}",
            required=False,
        )

    def _milvus_call[R: MilvusReply](
        self,
        path: str,
        body: BaseModel,
        response_type: type[R],
        what: str,
        *,
        required: bool = True,
    ) -> R | None:
        last: object = None
        for _ in range(MILVUS_MAX_ATTEMPTS):
            result = self.milvus.post(
                path, headers=self.milvus.master, json=body, response_type=response_type
            )
            if is_ok(result):
                reply = unwrap(result)
                if reply.code == 0:
                    return reply
                last = reply.model_dump()
            else:
                last = result
            time.sleep(MILVUS_RETRY_SLEEP_SECONDS)
        if required:
            raise AssertionError(
                f"milvus {what} failed after {MILVUS_MAX_ATTEMPTS} attempts: {last}"
            )
        return None

    # ---- OpenAI embeddings (seeding) ------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.openai.post(
            "/v1/embeddings",
            headers=self.openai.master,
            json=OpenAIEmbedBody(input=texts),
            response_type=OpenAIEmbedResponse,
        )
        return [item.embedding for item in unwrap(result).data]

    # ---- proxy managed store --------------------------------------------

    def store_litellm_params(
        self, *, output_fields: list[str] | None = None
    ) -> dict[str, object]:
        """The registration params that make the proxy embed queries with the same
        model used for seeding and pull the stored text back into results.

        The proxy's request-data merge order for managed stores applies the store's
        ``litellm_params`` AFTER the caller's per-request params (see
        ``_update_request_data_with_litellm_managed_vector_store_registry`` in
        ``litellm/proxy/vector_store_endpoints/endpoints.py``), so a store-level
        ``outputFields`` overrides the per-search value. Tests that need extra
        output fields register their own store with the desired list."""
        params: dict[str, object] = {
            "api_base": MILVUS_API_BASE,
            "api_key": MILVUS_API_KEY,
            "litellm_embedding_model": EMBEDDING_MODEL,
            "litellm_embedding_config": {"api_key": OPENAI_API_KEY},
            "annsField": VECTOR_FIELD,
            "milvus_text_field": TEXT_FIELD,
            "outputFields": output_fields if output_fields is not None else [TEXT_FIELD],
        }
        return params

    def register_store(
        self, vector_store_id: str, litellm_params: dict[str, object]
    ) -> Result[RegisterStoreResponse]:
        return self.gateway.transport.post(
            "/vector_store/new",
            headers=self.gateway.transport.master,
            json=RegisterStoreBody(
                vector_store_id=vector_store_id,
                custom_llm_provider="milvus",
                vector_store_name=vector_store_id,
                litellm_params=litellm_params,
            ),
            response_type=RegisterStoreResponse,
        )

    def delete_store(self, vector_store_id: str) -> None:
        _ = self.gateway.transport.post(
            "/vector_store/delete",
            headers=self.gateway.transport.master,
            json=DeleteStoreBody(vector_store_id=vector_store_id),
            response_type=NoBody,
        )

    def list_stores(self) -> Result[ManagedStoreListResponse]:
        return self.gateway.transport.get(
            "/vector_store/list",
            headers=self.gateway.transport.master,
            params=NoBody(),
            response_type=ManagedStoreListResponse,
        )

    def store_info(self, vector_store_id: str) -> Result[ManagedStoreInfoResponse]:
        return self.gateway.transport.post(
            "/vector_store/info",
            headers=self.gateway.transport.master,
            json=InfoBody(vector_store_id=vector_store_id),
            response_type=ManagedStoreInfoResponse,
        )

    def update_store(
        self,
        vector_store_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StreamingResponse:
        """Update returns the touched row, whose shape differs a bit across
        versions, so this returns the raw HTTP outcome and callers parse only
        what they assert on."""
        return self.gateway.transport.send(
            "/vector_store/update",
            headers=self.gateway.transport.master,
            json=UpdateBody(
                vector_store_id=vector_store_id,
                vector_store_name=name,
                vector_store_description=description,
                vector_store_metadata=metadata,
            ),
        )

    # ---- OpenAI-compat create (unsupported on Milvus, must fail cleanly) --

    def openai_compat_create(self, name: str) -> StreamingResponse:
        return self.gateway.transport.send(
            "/v1/vector_stores",
            headers=self.gateway.transport.master,
            json=OpenAICompatCreateBody(name=name),
        )

    # ---- proxy search / chat --------------------------------------------

    def search(
        self,
        vector_store_id: str,
        query: str | list[str],
        *,
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
        output_fields: list[str] | None = None,
        grouping_field: str | None = None,
        consistency_level: str | None = None,
    ) -> Result[SearchResponse]:
        # Every search needs the text field back for the response transform to
        # populate content; tests that want the extras override with their own list.
        effective_output_fields = output_fields if output_fields is not None else [TEXT_FIELD]
        return self.gateway.transport.post(
            f"/v1/vector_stores/{vector_store_id}/search",
            headers=self.gateway.transport.master,
            json=SearchBody(
                query=query,
                limit=limit,
                offset=offset,
                filter=filter,
                outputFields=effective_output_fields,
                groupingField=grouping_field,
                consistencyLevel=consistency_level,
            ),
            response_type=SearchResponse,
        )

    def search_raw(self, vector_store_id: str, query: str) -> StreamingResponse:
        """Search returning the unparsed HTTP outcome, for the negative path where
        the collection does not exist."""
        return self.gateway.transport.send(
            f"/v1/vector_stores/{vector_store_id}/search",
            headers=self.gateway.transport.master,
            json=SearchBody(query=query),
        )

    def search_raw_with_params(
        self,
        vector_store_id: str,
        query: str,
        *,
        consistency_level: str | None = None,
    ) -> StreamingResponse:
        """Search returning the unparsed HTTP outcome for tests that need to
        assert on non-2xx responses without triggering pydantic validation."""
        return self.gateway.transport.send(
            f"/v1/vector_stores/{vector_store_id}/search",
            headers=self.gateway.transport.master,
            json=SearchBody(
                query=query,
                outputFields=[TEXT_FIELD],
                consistencyLevel=consistency_level,
            ),
        )

    def chat_with_store(
        self, vector_store_id: str, question: str, model: str = CHAT_MODEL
    ) -> Result[ChatResponse]:
        # A generous max_tokens because reasoning models (gpt-5.5) consume the
        # completion budget on hidden reasoning tokens first; a tight cap can
        # exhaust the budget before any visible content is emitted.
        return self.gateway.transport.post(
            "/chat/completions",
            headers=self.gateway.transport.master,
            json=ChatWithVectorStoreBody(
                model=model,
                messages=[ChatMessage(role="user", content=question)],
                vector_store_ids=[vector_store_id],
                max_tokens=2048,
            ),
            response_type=ChatResponse,
        )


def build_client() -> VectorStoreClient:
    return VectorStoreClient(
        gateway=build_gateway(),
        milvus=HttpTransport(base_url=MILVUS_API_BASE, master_key=MILVUS_API_KEY),
        openai=HttpTransport(base_url="https://api.openai.com", master_key=OPENAI_API_KEY),
    )
