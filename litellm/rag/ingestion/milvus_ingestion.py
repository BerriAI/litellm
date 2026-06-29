"""
Milvus-specific RAG Ingestion implementation.

Milvus is an open-source, self-hostable vector database. This implementation
adds write/ingest support to complement the existing Milvus vector store
search provider (litellm/llms/milvus/vector_stores).

This implementation:
1. Generates embeddings using LiteLLM's embedding API (supports any provider)
2. Auto-creates the target collection via the Milvus REST "quick setup" API
   when it does not exist (dynamic fields enabled so chunk text + metadata are
   stored alongside the vector)
3. Inserts chunks + embeddings via the Milvus REST `entities/insert` API

It talks to Milvus over the REST API v2 (`/v2/vectordb/...`) using httpx, so it
does not add a `pymilvus` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litellm._logging import verbose_logger
from litellm.constants import (
    MILVUS_DEFAULT_METRIC_TYPE,
    MILVUS_DEFAULT_TEXT_FIELD,
    MILVUS_DEFAULT_VECTOR_FIELD,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


class MilvusRAGIngestion(BaseRAGIngestion):
    """
    Milvus RAG ingestion using the Milvus REST API v2 + httpx.

    Workflow:
    1. Generate embeddings using LiteLLM (supports any embedding provider)
    2. Auto-create the collection if needed (quick setup, dynamic fields on)
    3. Insert chunks + embeddings via `entities/insert`

    Configuration (vector_store config):
    - collection_name / vector_store_id: target Milvus collection (required)
    - api_base: Milvus REST base URL (or MILVUS_API_BASE env)
    - api_key: Milvus token/credential. The MILVUS_API_KEY env fallback applies
      only when api_base also comes from the MILVUS_API_BASE env, so the server
      token is never sent to a config/credential-supplied endpoint. Optional for
      a self-hosted Milvus without auth.
    - vector_field: embedding field name (default: "vector")
    - text_field: chunk text field name (default: "text")
    - metric_type: distance metric for auto-created collection (default: "COSINE")
    - db_name: Milvus database namespace, server-side only via MILVUS_DB_NAME env.
      Not accepted from the request: it selects the write target's database and is
      outside the per-collection authorization boundary.
    - partition_name: target partition, server-side only via MILVUS_PARTITION_NAME
      env. Not accepted from the request: it selects the write target's partition
      and is outside the per-collection authorization boundary.
    - auto_create_collection: create the collection if missing (default: True)
    """

    def __init__(
        self,
        ingest_options: RAGIngestOptions,
        router: Router | None = None,
    ):
        BaseRAGIngestion.__init__(self, ingest_options=ingest_options, router=router)

        if not self.embedding_config:
            self.embedding_config = {"model": "text-embedding-3-small"}

        self.collection_name = self.vector_store_config.get("collection_name") or self.vector_store_config.get(
            "vector_store_id"
        )
        if not self.collection_name:
            raise ValueError(
                "Milvus RAG ingestion requires 'collection_name' (or 'vector_store_id') in the vector_store config."
            )

        config_api_base = self.vector_store_config.get("api_base")
        self.api_base = config_api_base or get_secret_str("MILVUS_API_BASE")
        if not self.api_base:
            raise ValueError(
                "Milvus API base URL is required. Set the MILVUS_API_BASE environment "
                "variable or pass 'api_base' in the vector_store config."
            )
        self.api_base = self.api_base.rstrip("/")

        config_api_key = self.vector_store_config.get("api_key")
        self.api_key = config_api_key if config_api_base else config_api_key or get_secret_str("MILVUS_API_KEY")
        self.vector_field = self.vector_store_config.get("vector_field", MILVUS_DEFAULT_VECTOR_FIELD)
        self.text_field = self.vector_store_config.get("text_field", MILVUS_DEFAULT_TEXT_FIELD)
        self.metric_type = self.vector_store_config.get("metric_type", MILVUS_DEFAULT_METRIC_TYPE)
        self.db_name = get_secret_str("MILVUS_DB_NAME")
        self.partition_name = get_secret_str("MILVUS_PARTITION_NAME")
        self.auto_create_collection = self.vector_store_config.get("auto_create_collection", True)

        self.async_httpx_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.RAG)

    @classmethod
    def normalize_authorized_vector_store_id(cls, vector_store_opts: dict[str, object]) -> None:
        """
        Milvus resolves its write target from `collection_name` first (falling
        back to `vector_store_id`). Always mirror `collection_name` onto
        `vector_store_id` so authorization covers the collection that will be
        written to - even when the caller supplies a different `vector_store_id`
        they happen to have access to.
        """
        collection_name = vector_store_opts.get("collection_name")
        if collection_name:
            vector_store_opts["vector_store_id"] = collection_name

    @classmethod
    def credential_protected_fields(cls) -> frozenset[str]:
        """
        Milvus selects its write target from `collection_name` (mirrored onto
        `vector_store_id` for authorization), so both must be shielded from
        credential hydration to keep the authorized target intact.
        """
        return super().credential_protected_fields() | {"collection_name"}

    @classmethod
    def can_auto_create_vector_store(cls, vector_store_opts: dict[str, object]) -> bool:
        """
        Milvus is capable of creating the target collection on ingest, so the
        view-only guard must always require the target to resolve to a managed
        vector store. This reports the provider's capability, not the
        request-supplied `auto_create_collection` flag: that flag is caller
        controlled, and trusting it would let a view-only key set it to false,
        name any existing collection, and skip the managed-store check.
        """
        return True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _with_db(self, body: dict[str, object]) -> dict[str, object]:
        if self.db_name:
            body["dbName"] = self.db_name
        return body

    async def _post(self, path: str, body: dict[str, object]) -> dict[str, object]:
        url = f"{self.api_base}{path}"
        response = await self.async_httpx_client.post(url, json=body, headers=self._headers())
        response.raise_for_status()
        data = response.json()
        # Milvus REST returns {"code": 0, "data": ...} on success,
        # {"code": <non-zero>, "message": "..."} on error.
        if isinstance(data, dict) and data.get("code") not in (0, None):
            raise RuntimeError(
                f"Milvus API call to {path} failed: code={data.get('code')} message={data.get('message')}"
            )
        return data

    async def _collection_exists(self) -> bool:
        try:
            data = await self._post(
                "/v2/vectordb/collections/has",
                self._with_db({"collectionName": self.collection_name}),
            )
            inner = data.get("data")
            return bool(inner.get("has")) if isinstance(inner, dict) else False
        except Exception as e:  # noqa: BLE001
            verbose_logger.debug(f"Milvus collection 'has' check failed: {e}")
            return False

    async def _ensure_collection_exists(self, dimension: int) -> None:
        if not self.auto_create_collection:
            return
        if await self._collection_exists():
            return

        verbose_logger.debug(
            f"Creating Milvus collection '{self.collection_name}' (dimension={dimension}, metric={self.metric_type})"
        )
        # Quick-setup create: enables a dynamic field so chunk text + metadata
        # are stored alongside the vector without declaring a full schema.
        body = self._with_db(
            {
                "collectionName": self.collection_name,
                "dimension": dimension,
                "metricType": self.metric_type,
                "vectorFieldName": self.vector_field,
                "autoId": True,
                "enableDynamicField": True,
            }
        )
        await self._post("/v2/vectordb/collections/create", body)

    async def store(
        self,
        chunks: list[str],
        embeddings: list[list[float]] | None,
        filename: str | None = None,
        **_: object,
    ) -> tuple[str | None, str | None]:
        """
        Insert chunks + embeddings into a Milvus collection.

        Steps:
        1. Validate chunks/embeddings were produced
        2. Ensure the collection exists (auto-create quick setup if needed)
        3. Insert rows via `entities/insert`

        Returns:
            Tuple of (collection_name, filename)
        """
        if not embeddings or not chunks:
            raise ValueError(
                "No text content could be extracted from the file for embedding. "
                "Possible causes:\n"
                "  1. PDF files require OCR - add an 'ocr' config with a vision model "
                "(e.g., 'anthropic/claude-3-5-sonnet-20241022')\n"
                "  2. Binary files cannot be processed - convert to text first\n"
                "  3. File is empty or contains no extractable text"
            )

        await self._ensure_collection_exists(dimension=len(embeddings[0]))

        rows: list[dict[str, object]] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            row: dict[str, object] = {
                self.vector_field: embedding,
                self.text_field: chunk,
                "chunk_index": i,
            }
            if filename:
                row["filename"] = filename
            rows.append(row)

        body = self._with_db({"collectionName": self.collection_name, "data": rows})
        if self.partition_name:
            body["partitionName"] = self.partition_name

        await self._post("/v2/vectordb/entities/insert", body)
        verbose_logger.info(f"Inserted {len(rows)} vectors into Milvus collection '{self.collection_name}'")

        return self.collection_name, filename
