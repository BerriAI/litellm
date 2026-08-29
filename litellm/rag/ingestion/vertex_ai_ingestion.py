"""
Vertex AI-specific RAG Ingestion implementation.

Vertex AI RAG Engine handles embedding and chunking internally when files are uploaded,
so this implementation skips the embedding step and directly uploads files to RAG corpora.

Based on: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api-v1
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


def _present_fields(fields: tuple[tuple[str, object], ...]) -> Mapping[str, object]:
    return {name: value for name, value in fields if value}


class VertexRagResourceName(TypedDict, total=False):
    name: ReadOnly[str]


class VertexRagOperation(TypedDict, total=False):
    """A Vertex AI long-running operation resource, as the RAG Engine API returns it."""

    done: ReadOnly[bool]
    name: ReadOnly[str]
    error: ReadOnly[object]
    response: ReadOnly[VertexRagResourceName]


class VertexRagFileUpload(TypedDict, total=False):
    """Body of a ``ragFiles:upload`` response."""

    name: ReadOnly[str]
    ragFile: ReadOnly[VertexRagResourceName]


class _RagOperationView(TypedDict):
    """Holds one decoded long-running operation so the JSON body reads back typed."""

    operation: ReadOnly[VertexRagOperation]


class _RagFileUploadView(TypedDict):
    """Holds one decoded ``ragFiles:upload`` body so the JSON body reads back typed."""

    upload: ReadOnly[VertexRagFileUpload]


class VertexAIRAGIngestion(BaseRAGIngestion, VertexBase):
    """
    Vertex AI RAG Engine ingestion implementation.

    Key differences from base:
    - Embedding is handled by Vertex AI RAG Engine when files are uploaded
    - Files are uploaded using the RAG API (import or upload)
    - Chunking is done by Vertex AI RAG Engine (supports custom chunking config)
    - Supports Google Cloud Storage (GCS) and Google Drive sources
    - Supports custom parsing configurations (layout parser, LLM parser)
    """

    def __init__(
        self,
        ingest_options: RAGIngestOptions,
        router: Router | None = None,
    ):
        BaseRAGIngestion.__init__(self, ingest_options=ingest_options, router=router)
        VertexBase.__init__(self)

        # Extract Vertex AI specific configs from vector_store_config
        litellm_params: Final = dict(self.vector_store_config)

        # Get project, location, and credentials using VertexBase methods
        self.project_id = self.safe_get_vertex_ai_project(litellm_params)
        self.location = self.get_vertex_ai_location(litellm_params) or "us-central1"
        self.vertex_credentials = self.safe_get_vertex_ai_credentials(litellm_params)

    async def embed(
        self,
        chunks: list[str],
    ) -> list[list[float]] | None:
        """
        Vertex AI RAG Engine handles embedding internally - skip this step.

        Returns:
            None (Vertex AI embeds when files are uploaded to RAG corpus)
        """
        # Vertex AI RAG Engine handles embedding when files are uploaded
        return None

    async def store(
        self,
        file_content: bytes | None,
        filename: str | None,
        content_type: str | None,
        chunks: list[str],
        embeddings: list[list[float]] | None,
        existing_file_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Store content in Vertex AI RAG corpus.

        Vertex AI workflow:
        1. Create RAG corpus (if not provided)
        2. Upload file using RAG API (Vertex AI handles chunking/embedding)

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Ignored - Vertex AI handles chunking
            embeddings: Ignored - Vertex AI handles embedding
            existing_file_id: Existing provider file ID, unsupported for Vertex AI

        Returns:
            Tuple of (rag_corpus_id, file_id)
        """
        if not self.project_id:
            raise ValueError("vertex_project is required for Vertex AI RAG ingestion. Set it in vector_store config.")

        # Get or create RAG corpus
        rag_corpus_id = self.vector_store_config.get("vector_store_id")
        if not rag_corpus_id:
            rag_corpus_id = await self._create_rag_corpus(
                display_name=self.ingest_name or "litellm-rag-corpus",
                description=self.vector_store_config.get("description"),
            )

        # Upload file to RAG corpus
        result_file_id = None
        if file_content and filename and rag_corpus_id:
            result_file_id = await self._upload_file_to_corpus(
                rag_corpus_id=rag_corpus_id,
                filename=filename,
                file_content=file_content,
                content_type=content_type,
            )

        return rag_corpus_id, result_file_id

    async def _create_rag_corpus(
        self,
        display_name: str,
        description: str | None = None,
    ) -> str:
        """
        Create a Vertex AI RAG corpus.

        Args:
            display_name: Display name for the corpus
            description: Optional description

        Returns:
            RAG corpus ID (format: projects/{project}/locations/{location}/ragCorpora/{corpus_id})
        """
        # Get access token using VertexBase method
        access_token, project_id = self._ensure_access_token(
            credentials=self.vertex_credentials,
            project_id=self.project_id,
            custom_llm_provider="vertex_ai",
        )

        # Use the project_id from token if not set
        if not self.project_id:
            self.project_id = project_id

        # Construct URL using vertex base URL helper
        base_url: Final = get_vertex_base_url(self.location)
        url: Final = f"{base_url}/v1beta1/projects/{self.project_id}/locations/{self.location}/ragCorpora"

        # Build request body with camelCase keys (Vertex AI API format)
        vector_db_config: Final = self.vector_store_config.get("vector_db_config")
        embedding_model: Final = self.vector_store_config.get("embedding_model")
        embedding_model_config: Final = (
            {"ragEmbeddingModelConfig": {"vertexPredictionEndpoint": {"endpoint": embedding_model}}}
            if embedding_model
            else None
        )
        vector_db_section: Final = (
            {**(vector_db_config or {}), **embedding_model_config} if embedding_model_config else vector_db_config
        )
        request_body: Final = {
            "displayName": display_name,
            **_present_fields((("description", description), ("vectorDbConfig", vector_db_section))),
        }

        verbose_logger.debug("Creating RAG corpus: %s", url)
        verbose_logger.debug("Request body: %s", json.dumps(request_body, indent=2))

        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 60.0},
        )

        response: Final = await client.post(
            url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code not in [200, 201]:
            error_msg: Final = f"Failed to create RAG corpus: {response.text}"
            verbose_logger.error(error_msg)
            raise Exception(error_msg)

        operation_view: Final[_RagOperationView] = {"operation": response.json()}
        response_data: Final = operation_view["operation"]
        verbose_logger.debug("Create corpus response: %s", json.dumps(response_data, indent=2))

        # The response is a long-running operation
        # Check if it's already done or if we need to poll
        if response_data.get("done"):
            # Operation completed immediately
            corpus_name = response_data.get("response", {}).get("name", "")
        else:
            # Need to poll the operation
            operation_name: Final = response_data.get("name", "")
            verbose_logger.debug("Polling operation: %s", operation_name)
            corpus_name = await self._poll_operation(
                operation_name=operation_name,
                access_token=access_token,
            )

        verbose_logger.debug("Created RAG corpus: %s", corpus_name)
        return corpus_name

    async def _poll_operation(
        self,
        operation_name: str,
        access_token: str,
        max_retries: int = 30,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Poll a long-running operation until it completes.

        Args:
            operation_name: The operation name (e.g., "operations/123456")
            access_token: Access token for authentication
            max_retries: Maximum number of polling attempts
            retry_delay: Delay between polling attempts in seconds

        Returns:
            The corpus name from the completed operation

        Raises:
            Exception: If operation fails or times out
        """
        import asyncio

        base_url: Final = get_vertex_base_url(self.location)
        # Operation name is like: projects/{project}/locations/{location}/operations/{operation_id}
        # We need to construct the full URL
        url: Final = f"{base_url}/v1beta1/{operation_name}"

        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 60.0},
        )

        for attempt in range(max_retries):
            response = await client.get(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                },
            )

            if response.status_code != 200:
                error_msg = f"Failed to poll operation: {response.text}"
                verbose_logger.error(error_msg)
                raise Exception(error_msg)

            operation_view: _RagOperationView = {"operation": response.json()}
            operation_data: VertexRagOperation = operation_view["operation"]

            if operation_data.get("done"):
                # Check for errors
                if "error" in operation_data:
                    error = operation_data.get("error")
                    raise Exception(f"Operation failed: {error}")

                # Extract corpus name from response
                corpus_name = operation_data.get("response", {}).get("name", "")
                if corpus_name:
                    return corpus_name
                else:
                    raise Exception(f"No corpus name in operation response: {operation_data}")

            verbose_logger.debug("Operation not done yet, attempt %s/%s", attempt + 1, max_retries)
            await asyncio.sleep(retry_delay)

        raise Exception(f"Operation timed out after {max_retries} attempts")

    async def _upload_file_to_corpus(
        self,
        rag_corpus_id: str,
        filename: str,
        file_content: bytes,
        content_type: str | None,
    ) -> str:
        """
        Upload a file to Vertex AI RAG corpus using multipart upload.

        Args:
            rag_corpus_id: RAG corpus resource name
            filename: Name of the file
            file_content: File content bytes
            content_type: MIME type

        Returns:
            File ID or resource name
        """
        # Get access token using VertexBase method
        access_token, _ = self._ensure_access_token(
            credentials=self.vertex_credentials,
            project_id=self.project_id,
            custom_llm_provider="vertex_ai",
        )

        # Construct upload URL using vertex base URL helper
        base_url: Final = get_vertex_base_url(self.location)
        url: Final = f"{base_url}/upload/v1beta1/{rag_corpus_id}/ragFiles:upload"

        # Build metadata for the file with snake_case keys (as per upload API docs)
        description: Final = self.vector_store_config.get("file_description")
        rag_file: Final = {
            "display_name": filename,
            **_present_fields((("description", description),)),
        }

        # Add chunking configuration if provided
        chunking_strategy: Final[Mapping[str, object]] = self.chunking_strategy
        chunk_size: Final = chunking_strategy.get("chunk_size")
        chunk_overlap: Final = chunking_strategy.get("chunk_overlap")
        fixed_length_chunking: Final = _present_fields((("chunk_size", chunk_size), ("chunk_overlap", chunk_overlap)))
        upload_rag_file_config: Final = (
            {
                "rag_file_transformation_config": {
                    "rag_file_chunking_config": {"fixed_length_chunking": fixed_length_chunking}
                }
            }
            if fixed_length_chunking
            else None
        )
        metadata: Final = {
            "rag_file": rag_file,
            **_present_fields((("upload_rag_file_config", upload_rag_file_config),)),
        }

        verbose_logger.debug("Uploading file to RAG corpus: %s", url)
        verbose_logger.debug("Metadata: %s", json.dumps(metadata, indent=2))

        # Prepare multipart form data
        files: Final = {
            "metadata": (None, json.dumps(metadata), "application/json"),
            "file": (
                filename,
                file_content,
                content_type or "application/octet-stream",
            ),
        }
        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 300.0},  # Longer timeout for large files
        )

        response: Final = await client.post(
            url,
            files=files,
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-Goog-Upload-Protocol": "multipart",
            },
        )

        if response.status_code not in [200, 201]:
            error_msg: Final = f"Failed to upload file: {response.text}"
            verbose_logger.error(error_msg)
            raise Exception(error_msg)

        # Parse response to get file ID
        try:
            upload_view: Final[_RagFileUploadView] = {"upload": response.json()}
            response_data: Final = upload_view["upload"]
            # The response should contain the rag_file resource name
            rag_file_name: Final = response_data.get("ragFile", {}).get("name", "")
            file_id: Final = rag_file_name or response_data.get("name", "")

            verbose_logger.debug("Upload complete. File ID: %s", file_id)
            return file_id
        except Exception as e:
            verbose_logger.warning("Could not parse upload response: %s", e)
            return "uploaded"

    async def _import_files_from_gcs(
        self,
        rag_corpus_id: str,
        gcs_uris: list[str],
    ) -> str:
        """
        Import files from Google Cloud Storage into RAG corpus.

        Args:
            rag_corpus_id: RAG corpus resource name
            gcs_uris: List of GCS URIs (e.g., ["gs://bucket/file.pdf"])

        Returns:
            Operation name for tracking import progress
        """
        # Get access token using VertexBase method
        access_token, _ = self._ensure_access_token(
            credentials=self.vertex_credentials,
            project_id=self.project_id,
            custom_llm_provider="vertex_ai",
        )

        # Construct import URL using vertex base URL helper
        base_url: Final = get_vertex_base_url(self.location)
        url: Final = f"{base_url}/v1beta1/{rag_corpus_id}/ragFiles:import"

        # Add chunking configuration if provided
        chunking_strategy: Final[Mapping[str, object]] = self.chunking_strategy
        chunk_size: Final = chunking_strategy.get("chunk_size")
        chunk_overlap: Final = chunking_strategy.get("chunk_overlap")

        # Add max embedding requests per minute if specified
        max_embedding_qpm: Final = self.vector_store_config.get("max_embedding_requests_per_min")

        # Build request body with camelCase keys (Vertex AI API format)
        chunking_config: Final = (
            {"chunkSize": chunk_size or 1024, "chunkOverlap": chunk_overlap or 200}
            if chunk_size or chunk_overlap
            else None
        )
        import_config: Final = {
            "gcsSource": {"uris": gcs_uris},
            **_present_fields(
                (
                    ("ragFileChunkingConfig", chunking_config),
                    ("maxEmbeddingRequestsPerMin", max_embedding_qpm),
                )
            ),
        }
        request_body: Final = {"importRagFilesConfig": import_config}

        verbose_logger.debug("Importing files from GCS: %s", url)
        verbose_logger.debug("Request body: %s", json.dumps(request_body, indent=2))

        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.RAG,
            params={"timeout": 60.0},
        )

        response: Final = await client.post(
            url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code not in [200, 201]:
            error_msg: Final = f"Failed to import files: {response.text}"
            verbose_logger.error(error_msg)
            raise Exception(error_msg)

        operation_view: Final[_RagOperationView] = {"operation": response.json()}
        response_data: Final = operation_view["operation"]
        operation_name: Final = response_data.get("name", "")

        verbose_logger.debug("Import operation started: %s", operation_name)
        return operation_name
