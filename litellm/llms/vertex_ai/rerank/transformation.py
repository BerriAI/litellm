"""
Translates from Cohere's `/v1/rerank` input format to Vertex AI Discovery Engine's `/rank` input format.

Why separate file? Make it easy to see how transformation works
"""

import math
import uuid
from collections.abc import Mapping
from typing import Final

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.vertex_ai.common_utils import (
    vertex_request_labels_from_litellm_params,
)
from litellm.llms.vertex_ai.vertex_llm_base import (
    VertexBase,
    _VertexCredentialsObject,  # pyright: ignore[reportPrivateUsage]  # exported vertex type
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
)


class VertexAIRerankConfig(BaseRerankConfig, VertexBase):
    """
    Configuration for Vertex AI Discovery Engine Rerank API

    Reference: https://cloud.google.com/generative-ai-app-builder/docs/ranking#rank_or_rerank_a_set_of_records_according_to_a_query
    """

    MAX_RECORDS_PER_SEARCH_UNIT = 100

    def __init__(self) -> None:
        super().__init__()
        self._rerank_token_cache: dict[  # mutable-ok: token cache keyed per (credentials, project)
            tuple[VERTEX_CREDENTIALS_TYPES | None, str | None],
            tuple[str, str, _VertexCredentialsObject | None],
        ] = {}

    def _get_cached_rerank_token(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
    ) -> tuple[str, str] | None:
        cache_key: Final = (credentials, project_id)
        cached: Final = self._rerank_token_cache.get(cache_key)
        if cached is None:
            return None
        token, cached_project, creds = cached
        # Check if underlying credentials are expired; if so, invalidate cache
        if creds is not None and getattr(creds, "expired", False):
            self._rerank_token_cache.pop(cache_key, None)
            return None
        return token, cached_project

    def _set_cached_rerank_token(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
        token: str,
        cached_project: str,
    ) -> None:
        cache_key: Final = (credentials, project_id)
        # Get the underlying credentials object from the base class cache
        creds, _ = self._unpack_cached_credentials(cache_key)  # pyright: ignore[reportUnknownMemberType]  # base helper
        if creds is not None:
            self._rerank_token_cache[cache_key] = (token, cached_project, creds)
        else:
            # Fallback: cache without expiration tracking (e.g., when mocked)
            self._rerank_token_cache[cache_key] = (token, cached_project, None)

    def _resolve_rerank_token(self, params: dict[str, object]) -> tuple[str, str]:  # mutable-ok: safe_get takes dict
        vertex_credentials: Final = self.safe_get_vertex_ai_credentials(params)
        vertex_project_hint: Final = self.safe_get_vertex_ai_project(params)

        cached: Final = self._get_cached_rerank_token(vertex_credentials, vertex_project_hint)
        if cached is not None:
            return cached

        token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project_hint,
            custom_llm_provider="vertex_ai",
        )
        self._set_cached_rerank_token(vertex_credentials, vertex_project_hint, token, project_id)
        return token, project_id

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,
    ) -> str:
        """
        Get the complete URL for the Vertex AI Discovery Engine ranking API
        """
        # Try to get project ID from optional_params first (e.g., vertex_project parameter)
        params: Final[dict[str, object]] = optional_params or {}  # mutable-ok: reuses caller-owned optional_params

        _, vertex_project = self._resolve_rerank_token(params)

        # Fallback to environment or litellm config
        project_id: Final = vertex_project or get_secret_str("VERTEXAI_PROJECT") or litellm.vertex_project

        if not project_id:
            raise ValueError(
                "Vertex AI project ID is required. Please set 'VERTEXAI_PROJECT', 'litellm.vertex_project', or pass 'vertex_project' parameter"
            )

        return f"https://discoveryengine.googleapis.com/v1/projects/{project_id}/locations/global/rankingConfigs/default_ranking_config:rank"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        optional_params: dict | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict:
        """
        Validate and set up authentication for Vertex AI Discovery Engine API
        """
        # Get credentials and project info from optional_params (which contains vertex_credentials, etc.)
        vertex_params: Final[dict[str, object]] = (  # mutable-ok: read-only lookup input from caller
            optional_params.copy() if optional_params else {}
        )
        access_token, project_id = self._resolve_rerank_token(vertex_params)

        default_headers: Final = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": project_id,
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict,
        headers: dict,
        litellm_params: dict | None = None,
    ) -> dict:
        """
        Transform the request from Cohere format to Vertex AI Discovery Engine format
        """
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Vertex AI rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Vertex AI rerank")

        query: Final = optional_rerank_params["query"]
        documents: Final = optional_rerank_params["documents"]
        top_n: Final = optional_rerank_params.get("top_n", None)
        return_documents: Final = optional_rerank_params.get("return_documents", True)

        # Convert documents to records format
        records: Final = []
        for idx, document in enumerate(documents):
            if isinstance(document, str):
                content = document
                title = " ".join(document.split()[:3])  # First 3 words as title
            else:
                # Handle dict format
                content = document.get("text", str(document))
                title = document.get("title", " ".join(content.split()[:3]))

            records.append({"id": str(idx), "title": title, "content": content})

        request_data: Final = {"model": model, "query": query, "records": records}

        if top_n is not None:
            request_data["topN"] = top_n

        # Map return_documents to ignoreRecordDetailsInResponse
        # When return_documents is False, we want to ignore record details (return only IDs)
        request_data["ignoreRecordDetailsInResponse"] = not return_documents

        user_labels: Final = vertex_request_labels_from_litellm_params(litellm_params)
        if user_labels:
            request_data["userLabels"] = user_labels

        return request_data

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        """
        Transform Vertex AI Discovery Engine response to Cohere format
        """
        try:
            raw_response_json: Final = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")

        # Extract records from response
        records: Final = raw_response_json.get("records", [])

        # Convert to Cohere format
        results: Final = []
        for record in records:
            # Handle both cases: with full details and with only IDs
            if "score" in record:
                # Full response with score and details
                results.append(
                    {
                        "index": int(record["id"]),
                        "relevance_score": record.get("score", 0.0),
                    }
                )
            else:
                # Response with only IDs (when ignoreRecordDetailsInResponse=true)
                # We can't provide a relevance score, so we'll use a default
                results.append(
                    {
                        "index": int(record["id"]),
                        "relevance_score": 1.0,  # Default score when details are ignored
                    }
                )

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Create response in Cohere format
        # Convert results to proper RerankResponseResult objects
        rerank_results: Final = []
        for result in results:
            rerank_results.append(
                RerankResponseResult(index=result["index"], relevance_score=result["relevance_score"])
            )

        input_record_count: Final = len(request_data.get("records", ()))
        search_units: Final = math.ceil(input_record_count / self.MAX_RECORDS_PER_SEARCH_UNIT)
        meta: Final = RerankResponseMeta(billed_units=RerankBilledUnits(search_units=search_units))

        return RerankResponse(id=f"vertex_ai_rerank_{uuid.uuid4()}", results=rerank_results, meta=meta)

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: list[str | dict[str, object]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: list[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict:
        """
        Map Cohere rerank params to Vertex AI format
        """
        result: Final = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }
        result.update(non_default_params)
        return result
