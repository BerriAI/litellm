"""
Vertex AI Imagen authentication hook.

Uses the existing VertexBase class for authentication.
"""

from typing import Any, Dict

from litellm.experimental.endpoint_definitions.hooks import GenericEndpointHooks


class VertexAIImagenAuthHook(GenericEndpointHooks):
    """
    Hook that handles Vertex AI authentication using the existing VertexBase class.
    
    Uses the same auth flow as the rest of LiteLLM's Vertex AI integration.
    """
    
    def __init__(self):
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
        self._vertex_base = VertexBase()
    
    async def async_pre_call_hook(
        self,
        operation_name: str,
        headers: Dict[str, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Get Vertex AI auth token and add to headers.
        
        Looks for credentials in kwargs or environment:
        - vertex_credentials / VERTEXAI_CREDENTIALS
        - vertex_project / VERTEXAI_PROJECT (uses project_id from kwargs if not set)
        """
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

        # Get credentials from kwargs or environment
        vertex_credentials = kwargs.pop("vertex_credentials", None)
        if vertex_credentials is None:
            vertex_credentials = VertexBase.safe_get_vertex_ai_credentials({})
        
        # Get project_id - prefer from kwargs, fall back to env
        project_id = kwargs.get("project_id")
        if project_id is None:
            project_id = VertexBase.safe_get_vertex_ai_project({})
        
        # Get access token using VertexBase
        access_token, resolved_project_id = await self._vertex_base._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=project_id,
            custom_llm_provider="vertex_ai",
        )
        
        # Update project_id in kwargs if it was resolved
        if kwargs.get("project_id") is None and resolved_project_id:
            kwargs["project_id"] = resolved_project_id
        
        # Add auth header
        headers["Authorization"] = f"Bearer {access_token}"
        
        return headers
    
    def sync_pre_call_hook(
        self,
        operation_name: str,
        headers: Dict[str, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Sync version - get Vertex AI auth token and add to headers.
        """
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

        # Get credentials from kwargs or environment
        vertex_credentials = kwargs.pop("vertex_credentials", None)
        if vertex_credentials is None:
            vertex_credentials = VertexBase.safe_get_vertex_ai_credentials({})
        
        # Get project_id - prefer from kwargs, fall back to env
        project_id = kwargs.get("project_id")
        if project_id is None:
            project_id = VertexBase.safe_get_vertex_ai_project({})
        
        # Get access token using VertexBase
        access_token, resolved_project_id = self._vertex_base.get_access_token(
            credentials=vertex_credentials,
            project_id=project_id,
        )
        
        # Update project_id in kwargs if it was resolved
        if kwargs.get("project_id") is None and resolved_project_id:
            kwargs["project_id"] = resolved_project_id
        
        # Add auth header
        headers["Authorization"] = f"Bearer {access_token}"
        
        return headers

