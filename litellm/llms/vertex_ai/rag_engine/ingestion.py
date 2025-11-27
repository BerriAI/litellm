"""
Vertex AI RAG Engine Ingestion implementation.

Uses:
- litellm.files.acreate_file for uploading files to GCS
- Vertex AI RAG Engine REST API for importing files into corpus (via httpx)

Key differences from OpenAI:
- Files must be uploaded to GCS first (via litellm.files.acreate_file)
- Embedding is handled internally using text-embedding-005 by default
- Chunking configured via unified chunking_strategy in ingest_options
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from litellm import get_secret_str
from litellm._logging import verbose_logger
from litellm.llms.vertex_ai.rag_engine.transformation import VertexAIRAGTransformation
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.rag import RAGIngestOptions


def _get_str_or_none(value: Any) -> Optional[str]:
    """Cast config value to Optional[str]."""
    return str(value) if value is not None else None


def _get_int(value: Any, default: int) -> int:
    """Cast config value to int with default."""
    if value is None:
        return default
    return int(value)


class VertexAIRAGIngestion(BaseRAGIngestion):
    """
    Vertex AI RAG Engine ingestion.

    Uses litellm.files.acreate_file for GCS upload, then imports into RAG corpus.

    Required config in vector_store:
    - vector_store_id: RAG corpus ID (required)

    Optional config in vector_store:
    - vertex_project: GCP project ID (uses env VERTEXAI_PROJECT if not set)
    - vertex_location: GCP region (default: us-central1)
    - vertex_credentials: Path to credentials JSON (uses ADC if not set)
    - wait_for_import: Wait for import to complete (default: True)
    - import_timeout: Timeout in seconds (default: 600)

    Chunking is configured via ingest_options["chunking_strategy"]:
    - chunk_size: Maximum size of chunks (default: 1000)
    - chunk_overlap: Overlap between chunks (default: 200)

    Authentication:
    - Uses Application Default Credentials (ADC)
    - Run: gcloud auth application-default login
    """

    def __init__(
        self,
        ingest_options: "RAGIngestOptions",
        router: Optional["Router"] = None,
    ):
        super().__init__(ingest_options=ingest_options, router=router)

        # Get corpus ID (required for Vertex AI)
        self.corpus_id = self.vector_store_config.get("vector_store_id")
        if not self.corpus_id:
            raise ValueError(
                "vector_store_id (corpus ID) is required for Vertex AI RAG ingestion. "
                "Please provide an existing RAG corpus ID."
            )

        # GCP config
        self.vertex_project = (
            self.vector_store_config.get("vertex_project")
            or get_secret_str("VERTEXAI_PROJECT")
        )
        self.vertex_location = (
            self.vector_store_config.get("vertex_location")
            or get_secret_str("VERTEXAI_LOCATION")
            or "us-central1"
        )
        self.vertex_credentials = self.vector_store_config.get("vertex_credentials")

        # GCS bucket for file uploads
        self.gcs_bucket = (
            self.vector_store_config.get("gcs_bucket")
            or os.environ.get("GCS_BUCKET_NAME")
        )
        if not self.gcs_bucket:
            raise ValueError(
                "gcs_bucket is required for Vertex AI RAG ingestion. "
                "Set via vector_store config or GCS_BUCKET_NAME env var."
            )

        # Import settings
        self.wait_for_import = self.vector_store_config.get("wait_for_import", True)
        self.import_timeout = _get_int(
            self.vector_store_config.get("import_timeout"), 600
        )

        # Validate required config
        if not self.vertex_project:
            raise ValueError(
                "vertex_project is required for Vertex AI RAG ingestion. "
                "Set via vector_store config or VERTEXAI_PROJECT env var."
            )

    def _get_corpus_name(self) -> str:
        """Get full corpus resource name."""
        return f"projects/{self.vertex_project}/locations/{self.vertex_location}/ragCorpora/{self.corpus_id}"

    async def _upload_file_to_gcs(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        """
        Upload file to GCS using litellm.files.acreate_file.

        Returns:
            GCS URI of the uploaded file (gs://bucket/path/file)
        """
        import litellm

        # Set GCS_BUCKET_NAME env var for litellm.files.create_file
        # The handler uses this to determine where to upload
        original_bucket = os.environ.get("GCS_BUCKET_NAME")
        if self.gcs_bucket:
            os.environ["GCS_BUCKET_NAME"] = self.gcs_bucket

        try:
            # Create file tuple for litellm.files.acreate_file
            file_tuple = (filename, file_content, content_type)

            verbose_logger.debug(
                f"Uploading file to GCS via litellm.files.acreate_file: {filename} "
                f"(bucket: {self.gcs_bucket})"
            )

            # Upload to GCS using LiteLLM's file upload
            response = await litellm.acreate_file(
                file=file_tuple,
                purpose="assistants",  # Purpose for file storage
                custom_llm_provider="vertex_ai",
                vertex_project=self.vertex_project,
                vertex_location=self.vertex_location,
                vertex_credentials=self.vertex_credentials,
            )

            # The response.id should be the GCS URI
            gcs_uri = response.id
            verbose_logger.info(f"Uploaded file to GCS: {gcs_uri}")

            return gcs_uri
        finally:
            # Restore original env var
            if original_bucket is not None:
                os.environ["GCS_BUCKET_NAME"] = original_bucket
            elif "GCS_BUCKET_NAME" in os.environ:
                del os.environ["GCS_BUCKET_NAME"]

    async def _import_file_to_corpus_via_sdk(
        self,
        gcs_uri: str,
    ) -> None:
        """
        Import file into RAG corpus using the Vertex AI SDK.

        The REST API endpoint for importRagFiles is not publicly available,
        so we use the Python SDK.
        """
        try:
            from vertexai import init as vertexai_init
            from vertexai import rag  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "vertexai.rag module not found. Vertex AI RAG requires "
                "google-cloud-aiplatform>=1.60.0. Install with: "
                "pip install 'google-cloud-aiplatform>=1.60.0'"
            )

        # Initialize Vertex AI
        vertexai_init(project=self.vertex_project, location=self.vertex_location)

        # Get chunking config from ingest_options (unified interface)
        transformation_config = self._build_transformation_config()

        corpus_name = self._get_corpus_name()
        verbose_logger.debug(f"Importing {gcs_uri} into corpus {self.corpus_id}")

        if self.wait_for_import:
            # Synchronous import - wait for completion
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[gcs_uri],
                transformation_config=transformation_config,
                timeout=self.import_timeout,
            )
            verbose_logger.info(
                f"Import complete: {response.imported_rag_files_count} files imported"
            )
        else:
            # Async import - don't wait
            _ = rag.import_files_async(
                corpus_name=corpus_name,
                paths=[gcs_uri],
                transformation_config=transformation_config,
            )
            verbose_logger.info("Import started asynchronously")

    def _build_transformation_config(self) -> Any:
        """
        Build Vertex AI TransformationConfig from unified chunking_strategy.

        Uses chunking_strategy from ingest_options (not vector_store).
        """
        try:
            from vertexai import rag  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "vertexai.rag module not found. Vertex AI RAG requires "
                "google-cloud-aiplatform>=1.60.0. Install with: "
                "pip install 'google-cloud-aiplatform>=1.60.0'"
            )

        # Get chunking config from ingest_options using transformation class
        from typing import cast

        from litellm.types.rag import RAGChunkingStrategy

        transformation = VertexAIRAGTransformation()
        chunking_config = transformation.transform_chunking_strategy_to_vertex_format(
            cast(Optional[RAGChunkingStrategy], self.chunking_strategy)
        )

        chunk_size = chunking_config["chunking_config"]["chunk_size"]
        chunk_overlap = chunking_config["chunking_config"]["chunk_overlap"]

        return rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
        )

    async def embed(
        self,
        chunks: List[str],
    ) -> Optional[List[List[float]]]:
        """
        Vertex AI handles embedding internally - skip this step.

        Returns:
            None (Vertex AI embeds when files are imported)
        """
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
        Store content in Vertex AI RAG corpus.

        Vertex AI workflow:
        1. Upload file to GCS via litellm.files.acreate_file
        2. Import file into RAG corpus via SDK
        3. (Optional) Wait for import to complete

        Args:
            file_content: Raw file bytes
            filename: Name of the file
            content_type: MIME type
            chunks: Ignored - Vertex AI handles chunking
            embeddings: Ignored - Vertex AI handles embedding

        Returns:
            Tuple of (corpus_id, gcs_uri)
        """
        if not file_content or not filename:
            verbose_logger.warning(
                "No file content or filename provided for Vertex AI ingestion"
            )
            return _get_str_or_none(self.corpus_id), None

        # Step 1: Upload file to GCS
        gcs_uri = await self._upload_file_to_gcs(
            file_content=file_content,
            filename=filename,
            content_type=content_type or "application/octet-stream",
        )

        # Step 2: Import file into RAG corpus
        try:
            await self._import_file_to_corpus_via_sdk(gcs_uri=gcs_uri)
        except Exception as e:
            verbose_logger.error(f"Failed to import file into RAG corpus: {e}")
            raise RuntimeError(f"Failed to import file into RAG corpus: {e}") from e

        return str(self.corpus_id), gcs_uri

