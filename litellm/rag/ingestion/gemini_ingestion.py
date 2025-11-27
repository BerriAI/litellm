"""
Gemini-specific RAG Ingestion implementation.

Gemini handles embedding and chunking internally when files are uploaded to File Search stores,
so this implementation skips the embedding step and directly uploads files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.llms.gemini.common_utils import GeminiModelInfo
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


class GeminiRAGIngestion(BaseRAGIngestion):
    """
    Gemini-specific RAG ingestion using File Search API.

    Key differences from base:
    - Embedding is handled by Gemini when files are uploaded to File Search stores
    - Files are uploaded using uploadToFileSearchStore API
    - Chunking is done by Gemini's File Search (supports custom white_space_config)
    - Supports custom metadata attachment
    """

    def __init__(
        self,
        ingest_options: "RAGIngestOptions",
        router: Optional["Router"] = None,
    ):
        super().__init__(ingest_options=ingest_options, router=router)
        self.model_info = GeminiModelInfo()

    async def embed(
        self,
        chunks: List[str],
    ) -> Optional[List[List[float]]]:
        """
        Gemini handles embedding internally - skip this step.

        Returns:
            None (Gemini embeds when files are uploaded to File Search store)
        """
        # Gemini handles embedding when files are uploaded to File Search stores
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
        Store content in Gemini File Search store.

        Gemini workflow:
        1. Create File Search store (if not provided)
        2. Upload file using uploadToFileSearchStore (Gemini handles chunking/embedding)

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Ignored - Gemini handles chunking
            embeddings: Ignored - Gemini handles embedding

        Returns:
            Tuple of (vector_store_id, file_id)
        """
        vector_store_id = self.vector_store_config.get("vector_store_id")
        
        vector_store_config = cast(Dict[str, Any], self.vector_store_config)

        # Get API credentials
        api_key = cast(Optional[str], vector_store_config.get("api_key")) or GeminiModelInfo.get_api_key()
        api_base = cast(Optional[str], vector_store_config.get("api_base")) or GeminiModelInfo.get_api_base()
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini File Search")
        
        if not api_base:
            raise ValueError("GEMINI_API_BASE is required")

        api_version = "v1beta"
        base_url = f"{api_base}/{api_version}"

        # Create File Search store if not provided
        if not vector_store_id:
            vector_store_id = await self._create_file_search_store(
                api_key=api_key,
                base_url=base_url,
                display_name=self.ingest_name or "litellm-rag-ingest",
            )

        # Upload file to File Search store
        result_file_id = None
        if file_content and filename and vector_store_id:
            result_file_id = await self._upload_to_file_search_store(
                api_key=api_key,
                base_url=base_url,
                vector_store_id=vector_store_id,
                filename=filename,
                file_content=file_content,
                content_type=content_type,
            )

        return vector_store_id, result_file_id

    async def _create_file_search_store(
        self,
        api_key: str,
        base_url: str,
        display_name: str,
    ) -> str:
        """
        Create a Gemini File Search store.

        Args:
            api_key: Gemini API key
            base_url: Base URL for Gemini API
            display_name: Display name for the store

        Returns:
            Store name (format: fileSearchStores/xxxxxxx)
        """
        url = f"{base_url}/fileSearchStores?key={api_key}"
        
        request_body = {
            "displayName": display_name
        }
        
        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 60.0},
        )
        response = await client.post(
            url,
            json=request_body,
            headers={"Content-Type": "application/json"},
        )
        
        if response.status_code != 200:
            error_msg = f"Failed to create File Search store: {response.text}"
            verbose_logger.error(error_msg)
            raise Exception(error_msg)
        
        response_data = response.json()
        store_name = response_data.get("name", "")
        
        verbose_logger.debug(f"Created File Search store: {store_name}")
        return store_name

    async def _upload_to_file_search_store(
        self,
        api_key: str,
        base_url: str,
        vector_store_id: str,
        filename: str,
        file_content: bytes,
        content_type: Optional[str],
    ) -> str:
        """
        Upload a file to Gemini File Search store using resumable upload.

        Args:
            api_key: Gemini API key
            base_url: Base URL for Gemini API
            vector_store_id: File Search store name
            filename: Name of the file
            file_content: File content bytes
            content_type: MIME type

        Returns:
            File ID or document name
        """
        # Step 1: Initiate resumable upload
        upload_url = await self._initiate_resumable_upload(
            api_key=api_key,
            base_url=base_url,
            vector_store_id=vector_store_id,
            filename=filename,
            file_size=len(file_content),
            content_type=content_type or "application/octet-stream",
        )

        # Step 2: Upload the file content
        file_id = await self._upload_file_content(
            upload_url=upload_url,
            file_content=file_content,
        )

        return file_id

    async def _initiate_resumable_upload(
        self,
        api_key: str,
        base_url: str,
        vector_store_id: str,
        filename: str,
        file_size: int,
        content_type: str,
    ) -> str:
        """
        Initiate a resumable upload session.

        Returns:
            Upload URL for the resumable session
        """
        # Construct the upload URL - need to use the full upload endpoint
        # base_url is like: https://generativelanguage.googleapis.com/v1beta
        # We need: https://generativelanguage.googleapis.com/upload/v1beta/{store_id}:uploadToFileSearchStore
        api_base = base_url.replace("/v1beta", "")  # Get base without version
        url = f"{api_base}/upload/v1beta/{vector_store_id}:uploadToFileSearchStore?key={api_key}"
        
        # Build request body with chunking config and metadata if provided
        request_body: Dict[str, Any] = {
            "displayName": filename
        }

        # Add chunking configuration if provided
        chunking_strategy = self.chunking_strategy
        if chunking_strategy and isinstance(chunking_strategy, dict):
            white_space_config = chunking_strategy.get("white_space_config")
            if white_space_config:
                request_body["chunkingConfig"] = {
                    "whiteSpaceConfig": {
                        "maxTokensPerChunk": white_space_config.get("max_tokens_per_chunk", 800),
                        "maxOverlapTokens": white_space_config.get("max_overlap_tokens", 400),
                    }
                }

        # Add custom metadata if provided in vector_store_config
        custom_metadata = cast(Optional[List[Dict[str, Any]]], self.vector_store_config.get("custom_metadata"))
        if custom_metadata:
            request_body["customMetadata"] = custom_metadata

        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": content_type,
            "Content-Type": "application/json",
        }

        verbose_logger.debug(f"Initiating resumable upload: {url}")

        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 60.0},
        )
        response = await client.post(
            url,
            json=request_body,
            headers=headers,
        )

        if response.status_code not in [200, 201]:
            error_msg = f"Failed to initiate upload: {response.text}"
            verbose_logger.error(error_msg)
            raise Exception(error_msg)
        verbose_logger.debug(f"Initiate resumable upload response: {response.headers}")
        # Extract upload URL from response headers
        upload_url = response.headers.get("x-goog-upload-url")
        if not upload_url:
            raise Exception("No upload URL returned in response headers")

        verbose_logger.debug(f"Got upload URL: {upload_url}")
        return upload_url

    async def _upload_file_content(
        self,
        upload_url: str,
        file_content: bytes,
    ) -> str:
        """
        Upload file content to the resumable upload URL.

        Returns:
            File ID or document name from the response
        """
        headers = {
            "Content-Length": str(len(file_content)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }

        verbose_logger.debug(f"Uploading file content ({len(file_content)} bytes)")

        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 300.0},  # Longer timeout for large files
        )
        response = await client.put(
            upload_url,
            content=file_content,
            headers=headers,
        )

        if response.status_code not in [200, 201]:
            error_msg = f"Failed to upload file: {response.text}"
            verbose_logger.error(error_msg)
            raise Exception(error_msg)

        # Parse response to get file/document ID
        try:
            response_data = response.json()
            # The response should contain the document name or file reference
            file_id = response_data.get("name", "") or response_data.get("file", {}).get("name", "")
            verbose_logger.debug(f"Upload complete. File ID: {file_id}")
            return file_id
        except Exception as e:
            verbose_logger.warning(f"Could not parse upload response: {e}")
            # Return a placeholder if we can't get the ID
            return "uploaded"

