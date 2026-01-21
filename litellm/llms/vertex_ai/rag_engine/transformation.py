"""
Transformation utilities for Vertex AI RAG Engine.

Handles transforming LiteLLM's unified formats to Vertex AI RAG Engine API format.
"""

from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.rag import RAGChunkingStrategy


class VertexAIRAGTransformation(VertexBase):
    """
    Transformation class for Vertex AI RAG Engine API.

    Handles:
    - Converting unified chunking_strategy to Vertex AI format
    - Building import request payloads
    - Transforming responses
    """

    def __init__(self):
        super().__init__()

    def get_import_rag_files_url(
        self,
        vertex_project: str,
        vertex_location: str,
        corpus_id: str,
    ) -> str:
        """
        Get the URL for importing RAG files.

        Note: The REST endpoint for importRagFiles may not be publicly available.
        Vertex AI RAG Engine primarily uses gRPC-based SDK.
        """
        base_url = get_vertex_base_url(vertex_location)
        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/ragCorpora/{corpus_id}:importRagFiles"

    def get_retrieve_contexts_url(
        self,
        vertex_project: str,
        vertex_location: str,
    ) -> str:
        """Get the URL for retrieving contexts (search)."""
        base_url = get_vertex_base_url(vertex_location)
        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}:retrieveContexts"

    def transform_chunking_strategy_to_vertex_format(
        self,
        chunking_strategy: Optional[RAGChunkingStrategy],
    ) -> Dict[str, Any]:
        """
        Transform LiteLLM's unified chunking_strategy to Vertex AI RAG format.

        LiteLLM format (RAGChunkingStrategy):
            {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]
            }

        Vertex AI RAG format (TransformationConfig):
            {
                "chunking_config": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
            }

        Note: Vertex AI doesn't support custom separators in the same way,
        so we only transform chunk_size and chunk_overlap.
        """
        if not chunking_strategy:
            return {
                "chunking_config": {
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                }
            }

        chunk_size = chunking_strategy.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = chunking_strategy.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

        # Log if separators are provided (not supported by Vertex AI)
        if chunking_strategy.get("separators"):
            verbose_logger.warning(
                "Vertex AI RAG Engine does not support custom separators. "
                "The 'separators' parameter will be ignored."
            )

        return {
            "chunking_config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        }

    def build_import_rag_files_request(
        self,
        gcs_uri: str,
        chunking_strategy: Optional[RAGChunkingStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Build the request payload for importing RAG files.

        Args:
            gcs_uri: GCS URI of the file to import (e.g., gs://bucket/path/file.txt)
            chunking_strategy: LiteLLM unified chunking config

        Returns:
            Request payload dict for importRagFiles API
        """
        transformation_config = self.transform_chunking_strategy_to_vertex_format(
            chunking_strategy
        )

        return {
            "import_rag_files_config": {
                "gcs_source": {
                    "uris": [gcs_uri]
                },
                "rag_file_transformation_config": transformation_config,
            }
        }

    def get_auth_headers(
        self,
        vertex_credentials: Optional[str] = None,
        vertex_project: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Get authentication headers for Vertex AI API calls.

        Uses the base class method to get credentials.
        """
        credentials = self.get_vertex_ai_credentials(
            {"vertex_credentials": vertex_credentials}
        )
        project = vertex_project or self.get_vertex_ai_project({})

        access_token, _ = self._ensure_access_token(
            credentials=credentials,
            project_id=project,
            custom_llm_provider="vertex_ai",
        )

        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

