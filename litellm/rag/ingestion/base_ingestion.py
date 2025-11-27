"""
Base RAG Ingestion class.

Provides abstract methods for:
- OCR
- Chunking
- Embedding
- Vector Store operations

Providers can inherit and override methods as needed.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid4
from litellm.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.rag.text_splitters import RecursiveCharacterTextSplitter
from litellm.types.rag import RAGIngestOptions, RAGIngestResponse

if TYPE_CHECKING:
    from litellm import Router


class BaseRAGIngestion(ABC):
    """
    Base class for RAG ingestion.

    Providers should inherit from this class and override methods as needed.
    For example, OpenAI handles embedding internally when attaching files to
    vector stores, so it overrides the embedding step to be a no-op.
    """

    def __init__(
        self,
        ingest_options: RAGIngestOptions,
        router: Optional["Router"] = None,
    ):
        self.ingest_options = ingest_options
        self.router = router
        self.ingest_id = f"ingest_{uuid4()}"

        # Extract configs from options
        self.ocr_config = ingest_options.get("ocr")
        self.chunking_strategy: Dict[str, Any] = cast(
            Dict[str, Any],
            ingest_options.get("chunking_strategy") or {"type": "auto"},
        )
        self.embedding_config = ingest_options.get("embedding")
        self.vector_store_config: Dict[str, Any] = cast(
            Dict[str, Any], ingest_options.get("vector_store") or {}
        )
        self.ingest_name = ingest_options.get("name")

        # Load credentials from litellm_credential_name if provided in vector_store config
        self._load_credentials_from_config()

    def _load_credentials_from_config(self) -> None:
        """
        Load credentials from litellm_credential_name if provided in vector_store config.

        This allows users to specify a credential name in the vector_store config
        which will be resolved from litellm.credential_list.
        """
        from litellm.litellm_core_utils.credential_accessor import CredentialAccessor

        credential_name = self.vector_store_config.get("litellm_credential_name")
        if credential_name and litellm.credential_list:
            credential_values = CredentialAccessor.get_credential_values(credential_name)
            # Merge credentials into vector_store_config (don't overwrite existing values)
            for key, value in credential_values.items():
                if key not in self.vector_store_config:
                    self.vector_store_config[key] = value

    @property
    def custom_llm_provider(self) -> str:
        """Get the vector store provider."""
        return self.vector_store_config.get("custom_llm_provider", "openai")

    async def upload(
        self,
        file_data: Optional[Tuple[str, bytes, str]] = None,
        file_url: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[bytes], Optional[str], Optional[str]]:
        """
        Upload / prepare file for ingestion.

        Args:
            file_data: Tuple of (filename, content_bytes, content_type)
            file_url: URL to fetch file from
            file_id: Existing file ID to use

        Returns:
            Tuple of (filename, file_content, content_type, existing_file_id)
        """
        if file_data:
            filename, file_content, content_type = file_data
            return filename, file_content, content_type, None

        if file_url:
            http_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.RAG)
            response = await http_client.get(file_url)
            response.raise_for_status()
            file_content = response.content
            filename = file_url.split("/")[-1] or "document"
            content_type = response.headers.get("content-type", "application/octet-stream")
            return filename, file_content, content_type, None

        if file_id:
            return None, None, None, file_id

        raise ValueError("Must provide file_data, file_url, or file_id")

    async def ocr(
        self,
        file_content: Optional[bytes],
        content_type: Optional[str],
    ) -> Optional[str]:
        """
        Perform OCR on file content to extract text.

        Args:
            file_content: Raw file bytes
            content_type: MIME type of the file

        Returns:
            Extracted text or None if OCR not configured/needed
        """
        if not self.ocr_config or not file_content:
            return None

        ocr_model = self.ocr_config.get("model", "mistral/mistral-ocr-latest")

        # Determine document type
        if content_type and "image" in content_type:
            doc_type, url_key = "image_url", "image_url"
        else:
            doc_type, url_key = "document_url", "document_url"

        # Encode as base64 data URL
        b64_content = base64.b64encode(file_content).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64_content}"

        # Use router if available
        if self.router is not None:
            ocr_response = await self.router.aocr(
                model=ocr_model,
                document={"type": doc_type, url_key: data_url},
            )
        else:
            ocr_response = await litellm.aocr(
                model=ocr_model,
                document={"type": doc_type, url_key: data_url},
            )

        # Extract text from pages
        if hasattr(ocr_response, "pages") and ocr_response.pages:  # type: ignore
            return "\n\n".join(
                page.markdown for page in ocr_response.pages if hasattr(page, "markdown")  # type: ignore
            )

        return None

    def chunk(
        self,
        text: Optional[str],
        file_content: Optional[bytes],
        ocr_was_used: bool,
    ) -> List[str]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter.

        Args:
            text: Text from OCR (if used)
            file_content: Raw file content bytes
            ocr_was_used: Whether OCR was performed

        Returns:
            List of text chunks
        """
        # Get text to chunk
        text_to_chunk: Optional[str] = None
        if text:
            text_to_chunk = text
        elif file_content and not ocr_was_used:
            try:
                text_to_chunk = file_content.decode("utf-8")
            except UnicodeDecodeError:
                verbose_logger.debug("Binary file detected, skipping text chunking")
                return []

        if not text_to_chunk:
            return []

        # Extract RecursiveCharacterTextSplitter args
        splitter_args = self.chunking_strategy or {}
        chunk_size = splitter_args.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = splitter_args.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        separators = splitter_args.get("separators", None)

        # Build splitter kwargs
        splitter_kwargs: Dict[str, Any] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        if separators:
            splitter_kwargs["separators"] = separators

        text_splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
        return text_splitter.split_text(text_to_chunk)

    async def embed(
        self,
        chunks: List[str],
    ) -> Optional[List[List[float]]]:
        """
        Generate embeddings for text chunks.

        Args:
            chunks: List of text chunks

        Returns:
            List of embeddings or None
        """
        if not self.embedding_config or not chunks:
            return None

        embedding_model = self.embedding_config.get("model", "text-embedding-3-small")

        if self.router is not None:
            response = await self.router.aembedding(model=embedding_model, input=chunks)
        else:
            response = await litellm.aembedding(model=embedding_model, input=chunks)

        return [item["embedding"] for item in response.data]

    @abstractmethod
    async def store(
        self,
        file_content: Optional[bytes],
        filename: Optional[str],
        content_type: Optional[str],
        chunks: List[str],
        embeddings: Optional[List[List[float]]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Store content in vector store.

        This method must be implemented by provider-specific subclasses.

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Text chunks (if chunking was done locally)
            embeddings: Embeddings (if embedding was done locally)

        Returns:
            Tuple of (vector_store_id, file_id)
        """
        pass

    async def ingest(
        self,
        file_data: Optional[Tuple[str, bytes, str]] = None,
        file_url: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> RAGIngestResponse:
        """
        Execute the full ingestion pipeline.

        Args:
            file_data: Tuple of (filename, content_bytes, content_type)
            file_url: URL to fetch file from
            file_id: Existing file ID to use

        Returns:
            RAGIngestResponse with status and IDs

        Raises:
            ValueError: If no input source is provided
        """
        # Step 1: Upload (raises ValueError if no input provided)
        filename, file_content, content_type, existing_file_id = await self.upload(
            file_data=file_data,
            file_url=file_url,
            file_id=file_id,
        )

        try:
            # Step 2: OCR (optional)
            extracted_text = await self.ocr(
                file_content=file_content,
                content_type=content_type,
            )

            # Step 3: Chunking
            chunks = self.chunk(
                text=extracted_text,
                file_content=file_content,
                ocr_was_used=self.ocr_config is not None,
            )

            # Step 4: Embedding (optional - some providers handle this internally)
            embeddings = await self.embed(chunks=chunks)

            # Step 5: Store in vector store
            vector_store_id, result_file_id = await self.store(
                file_content=file_content,
                filename=filename,
                content_type=content_type,
                chunks=chunks,
                embeddings=embeddings,
            )

            return RAGIngestResponse(
                id=self.ingest_id,
                status="completed",
                vector_store_id=vector_store_id or "",
                file_id=result_file_id or existing_file_id,
            )

        except Exception as e:
            verbose_logger.exception(f"RAG Pipeline failed: {e}")
            return RAGIngestResponse(
                id=self.ingest_id,
                status="failed",
                vector_store_id="",
                file_id=None,
                error=str(e),
            )

