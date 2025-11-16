"""
Token counter for Vertex AI Partner Models (Anthropic Claude, Mistral, etc.)

This handler provides token counting for partner models hosted on Vertex AI.
Unlike Gemini models which use Google's token counting API, partner models use
their respective publisher-specific count-tokens endpoints.
"""
from typing import Any, Dict, Optional

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class VertexAIPartnerModelsTokenCounter(VertexBase):
    """
    Token counter for Vertex AI Partner Models.

    Handles token counting for models like Claude (Anthropic), Mistral, etc.
    that are available through Vertex AI's partner model program.
    """

    def _get_publisher_for_model(self, model: str) -> str:
        """
        Determine the publisher name for the given model.

        Args:
            model: The model name (e.g., "claude-3-5-sonnet-20241022")

        Returns:
            Publisher name to use in the Vertex AI endpoint URL

        Raises:
            ValueError: If the model is not a recognized partner model
        """
        if "claude" in model:
            return "anthropic"
        elif "mistral" in model or "codestral" in model:
            return "mistralai"
        elif "llama" in model or "meta/" in model:
            return "meta"
        else:
            raise ValueError(f"Unknown partner model: {model}")

    def _build_count_tokens_endpoint(
        self,
        model: str,
        project_id: str,
        vertex_location: str,
        api_base: Optional[str] = None,
    ) -> str:
        """
        Build the count-tokens endpoint URL for a partner model.

        Args:
            model: The model name
            project_id: Google Cloud project ID
            vertex_location: Vertex AI location (e.g., "us-east5")
            api_base: Optional custom API base URL

        Returns:
            Full endpoint URL for the count-tokens API
        """
        publisher = self._get_publisher_for_model(model)

        # Use custom api_base if provided, otherwise construct default
        if api_base:
            base_url = api_base
        else:
            base_url = f"https://{vertex_location}-aiplatform.googleapis.com"

        # Construct the count-tokens endpoint
        # Format: /v1/projects/{project}/locations/{location}/publishers/{publisher}/models/count-tokens:rawPredict
        endpoint = (
            f"{base_url}/v1/projects/{project_id}/locations/{vertex_location}/"
            f"publishers/{publisher}/models/count-tokens:rawPredict"
        )

        return endpoint

    async def handle_count_tokens_request(
        self,
        model: str,
        request_data: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle token counting request for a Vertex AI partner model.

        Args:
            model: The model name
            request_data: Request payload (Anthropic Messages API format)
            litellm_params: LiteLLM parameters containing credentials, project, location

        Returns:
            Dict containing token count information

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate request
        if "messages" not in request_data:
            raise ValueError("messages required for token counting")

        # Extract Vertex AI credentials and settings
        vertex_credentials = self.get_vertex_ai_credentials(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        vertex_location = self.get_vertex_ai_location(litellm_params)

        # Get access token and resolved project ID
        access_token, project_id = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        # Build the endpoint URL
        endpoint_url = self._build_count_tokens_endpoint(
            model=model,
            project_id=project_id,
            vertex_location=vertex_location or "us-central1",
            api_base=litellm_params.get("api_base"),
        )

        # Prepare headers
        headers = {"Authorization": f"Bearer {access_token}"}

        # Get async HTTP client
        from litellm import LlmProviders

        async_client = get_async_httpx_client(llm_provider=LlmProviders.VERTEX_AI)

        # Make the request
        # Note: Partner models (especially Claude) accept Anthropic Messages API format directly
        response = await async_client.post(
            endpoint_url,
            headers=headers,
            json=request_data,
            timeout=30.0,
        )

        # Check for errors
        if response.status_code != 200:
            error_text = response.text
            raise ValueError(
                f"Token counting request failed with status {response.status_code}: {error_text}"
            )

        # Parse response
        result = response.json()

        # Return token count
        # Vertex AI Anthropic returns: {"input_tokens": 123}
        return {
            "input_tokens": result.get("input_tokens", 0),
            "tokenizer_used": "vertex_ai_partner_models",
        }
