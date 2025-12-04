"""
RAGFlow-specific RAG Ingestion implementation.

RAGFlow handles embedding and chunking internally when documents are uploaded to datasets,
so this implementation skips the embedding step and directly uploads documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


class RAGFlowRAGIngestion(BaseRAGIngestion):
    """
    RAGFlow-specific RAG ingestion.

    Key differences from base:
    - Embedding is handled by RAGFlow when documents are parsed
    - Documents are uploaded using multipart/form-data
    - Chunking is done by RAGFlow's parser (configurable via chunk_method and parser_config)
    - Supports automatic parsing trigger after upload
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
        RAGFlow handles embedding internally - skip this step.

        Returns:
            None (RAGFlow embeds when documents are parsed)
        """
        # RAGFlow handles embedding when documents are parsed
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
        Store content in RAGFlow dataset.

        RAGFlow workflow:
        1. Upload document using POST /api/v1/datasets/{dataset_id}/documents
        2. Optionally update document configuration (chunk_method, parser_config)
        3. Optionally trigger parsing using POST /api/v1/datasets/{dataset_id}/chunks

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Ignored - RAGFlow handles chunking
            embeddings: Ignored - RAGFlow handles embedding

        Returns:
            Tuple of (dataset_id, document_id)
        """
        vector_store_config = cast(Dict[str, Any], self.vector_store_config)
        
        # Validate required fields
        dataset_id = vector_store_config.get("vector_store_id")
        if not dataset_id:
            raise ValueError("vector_store_id (dataset_id) is required for RAGFlow ingestion")

        # Get API credentials
        api_key = (
            vector_store_config.get("api_key")
            or get_secret_str("RAGFLOW_API_KEY")
        )
        if not api_key:
            raise ValueError("RAGFLOW_API_KEY is required (set env var or pass in vector_store config)")

        api_base = (
            vector_store_config.get("api_base")
            or get_secret_str("RAGFLOW_API_BASE")
            or "http://localhost:9380"
        )
        api_base = api_base.rstrip("/")

        # Get RAGFlow-specific options
        chunk_method = vector_store_config.get("chunk_method")
        parser_config = vector_store_config.get("parser_config")
        auto_parse = vector_store_config.get("auto_parse", True)  # Default to True

        # Upload document
        if not file_content or not filename:
            raise ValueError("file_content and filename are required for RAGFlow document upload")

        document_id = await self._upload_document(
            api_base=api_base,
            api_key=api_key,
            dataset_id=dataset_id,
            filename=filename,
            file_content=file_content,
            content_type=content_type,
        )

        # Update document configuration if chunk_method or parser_config is provided
        if chunk_method or parser_config:
            await self._update_document_config(
                api_base=api_base,
                api_key=api_key,
                dataset_id=dataset_id,
                document_id=document_id,
                chunk_method=chunk_method,
                parser_config=parser_config,
            )

        # Trigger parsing if auto_parse is True
        if auto_parse:
            await self._trigger_parsing(
                api_base=api_base,
                api_key=api_key,
                dataset_id=dataset_id,
                document_ids=[document_id],
            )

        return dataset_id, document_id

    async def _upload_document(
        self,
        api_base: str,
        api_key: str,
        dataset_id: str,
        filename: str,
        file_content: bytes,
        content_type: Optional[str],
    ) -> str:
        """
        Upload a document to RAGFlow dataset.

        Args:
            api_base: RAGFlow API base URL
            api_key: RAGFlow API key
            dataset_id: Dataset ID
            filename: Name of the file
            file_content: File content bytes
            content_type: MIME type

        Returns:
            Document ID
        """
        url = f"{api_base}/api/v1/datasets/{dataset_id}/documents"

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        # Prepare multipart/form-data
        files = {
            "file": (filename, file_content, content_type or "application/octet-stream")
        }

        verbose_logger.debug(f"Uploading document to RAGFlow: {url}")

        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 300.0},  # Longer timeout for large files
        )

        try:
            response = await client.post(
                url,
                files=files,
                headers=headers,
            )

            if response.status_code != 200:
                error_msg = f"Failed to upload document: {response.text}"
                verbose_logger.error(error_msg)
                raise Exception(error_msg)

            response_data = response.json()

            # Check for RAGFlow error response
            if response_data.get("code") != 0:
                error_message = response_data.get("message", "Unknown error")
                raise Exception(f"RAGFlow error: {error_message}")

            # Extract document ID from response
            data = response_data.get("data", [])
            if not data or not isinstance(data, list) or len(data) == 0:
                raise Exception("RAGFlow response missing document data")

            document_id = data[0].get("id")
            if not document_id:
                raise Exception("RAGFlow response missing document id")

            verbose_logger.debug(f"Document uploaded successfully. Document ID: {document_id}")
            return document_id

        except Exception as e:
            verbose_logger.exception(f"Error uploading document to RAGFlow: {e}")
            raise

    async def _update_document_config(
        self,
        api_base: str,
        api_key: str,
        dataset_id: str,
        document_id: str,
        chunk_method: Optional[str],
        parser_config: Optional[Dict[str, Any]],
    ) -> None:
        """
        Update document configuration (chunk_method, parser_config).

        Args:
            api_base: RAGFlow API base URL
            api_key: RAGFlow API key
            dataset_id: Dataset ID
            document_id: Document ID
            chunk_method: Chunking method
            parser_config: Parser configuration
        """
        url = f"{api_base}/api/v1/datasets/{dataset_id}/documents/{document_id}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build request body
        request_body: Dict[str, Any] = {}
        if chunk_method:
            request_body["chunk_method"] = chunk_method
        if parser_config:
            request_body["parser_config"] = parser_config

        if not request_body:
            return  # Nothing to update

        verbose_logger.debug(f"Updating document configuration: {url}")

        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 60.0},
        )

        try:
            response = await client.put(
                url,
                json=request_body,
                headers=headers,
            )

            if response.status_code != 200:
                error_msg = f"Failed to update document configuration: {response.text}"
                verbose_logger.error(error_msg)
                raise Exception(error_msg)

            response_data = response.json()

            # Check for RAGFlow error response
            if response_data.get("code") != 0:
                error_message = response_data.get("message", "Unknown error")
                raise Exception(f"RAGFlow error: {error_message}")

            verbose_logger.debug("Document configuration updated successfully")

        except Exception as e:
            verbose_logger.exception(f"Error updating document configuration: {e}")
            raise

    async def _trigger_parsing(
        self,
        api_base: str,
        api_key: str,
        dataset_id: str,
        document_ids: List[str],
    ) -> None:
        """
        Trigger parsing for documents.

        Args:
            api_base: RAGFlow API base URL
            api_key: RAGFlow API key
            dataset_id: Dataset ID
            document_ids: List of document IDs to parse
        """
        url = f"{api_base}/api/v1/datasets/{dataset_id}/chunks"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_body = {
            "document_ids": document_ids,
        }

        verbose_logger.debug(f"Triggering parsing for documents: {url}")

        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 60.0},
        )

        try:
            response = await client.post(
                url,
                json=request_body,
                headers=headers,
            )

            if response.status_code != 200:
                error_msg = f"Failed to trigger parsing: {response.text}"
                verbose_logger.error(error_msg)
                raise Exception(error_msg)

            response_data = response.json()

            # Check for RAGFlow error response
            if response_data.get("code") != 0:
                error_message = response_data.get("message", "Unknown error")
                raise Exception(f"RAGFlow error: {error_message}")

            verbose_logger.debug("Parsing triggered successfully")

        except Exception as e:
            verbose_logger.exception(f"Error triggering parsing: {e}")
            raise

