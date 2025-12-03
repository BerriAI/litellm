"""
OpenAI-specific RAG Ingestion implementation.

OpenAI handles embedding internally when files are attached to vector stores,
so this implementation skips the embedding step and directly uploads files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.vector_store_files.main import acreate as vector_store_file_acreate
from litellm.vector_stores.main import acreate as vector_store_acreate

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


class OpenAIRAGIngestion(BaseRAGIngestion):
    """
    OpenAI-specific RAG ingestion.

    Key differences from base:
    - Embedding is handled by OpenAI when attaching files to vector stores
    - Files are uploaded and attached to vector stores directly
    - Chunking is done by OpenAI's vector store (uses 'auto' strategy)
    """

    def __init__(
        self,
        ingest_options: "RAGIngestOptions",
        router: Optional["Router"] = None,
    ):
        super().__init__(ingest_options=ingest_options, router=router)

    async def embed(
        self,
        chunks: List[str],
    ) -> Optional[List[List[float]]]:
        """
        OpenAI handles embedding internally - skip this step.

        Returns:
            None (OpenAI embeds when files are attached to vector store)
        """
        # OpenAI handles embedding when files are attached to vector stores
        return None

    async def store(
        self,
        file_content: Optional[bytes],
        filename: Optional[str],
        content_type: Optional[str],
        chunks: List[str],
        embeddings: Optional[List[List[float]]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Store content in OpenAI vector store.

        OpenAI workflow:
        1. Create vector store (if not provided)
        2. Upload file to OpenAI
        3. Attach file to vector store (OpenAI handles chunking/embedding)

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Ignored - OpenAI handles chunking
            embeddings: Ignored - OpenAI handles embedding

        Returns:
            Tuple of (vector_store_id, file_id)
        """
        vector_store_id = self.vector_store_config.get("vector_store_id")
        ttl_days = self.vector_store_config.get("ttl_days")

        # Get credentials from vector_store_config (loaded from litellm_credential_name if provided)
        api_key = self.vector_store_config.get("api_key")
        api_base = self.vector_store_config.get("api_base")

        # Create vector store if not provided
        if not vector_store_id:
            expires_after = {"anchor": "last_active_at", "days": ttl_days} if ttl_days else None
            create_response = await vector_store_acreate(
                name=self.ingest_name or "litellm-rag-ingest",
                custom_llm_provider="openai",
                expires_after=expires_after,
                api_key=api_key,
                api_base=api_base,
            )
            vector_store_id = create_response.get("id")

        # Upload file and attach to vector store
        result_file_id = None
        if file_content and filename and vector_store_id:
            # Upload file to OpenAI
            file_response = await litellm.acreate_file(
                file=(filename, file_content, content_type or "application/octet-stream"),
                purpose="assistants",
                custom_llm_provider="openai",
                api_key=api_key,
                api_base=api_base,
            )
            result_file_id = file_response.id

            # Attach file to vector store (OpenAI handles chunking/embedding)
            await vector_store_file_acreate(
                vector_store_id=vector_store_id,
                file_id=result_file_id,
                custom_llm_provider="openai",
                chunking_strategy=cast(Optional[Dict[str, Any]], self.chunking_strategy),
                api_key=api_key,
                api_base=api_base,
            )

        return vector_store_id, result_file_id

