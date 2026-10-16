"""
Token counter for Vertex AI Partner Models (Anthropic Claude, Mistral, etc.)

This handler provides token counting for partner models hosted on Vertex AI.
Unlike Gemini models which use Google's token counting API, partner models use
their respective publisher-specific count-tokens endpoints.
"""

from collections.abc import Mapping
from typing import Any, Final

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


@with_config(ConfigDict(strict=True))
class _VertexCountTokensRequest(TypedDict, total=False, extra_items=object):
    model: ReadOnly[NotRequired[str]]
    messages: ReadOnly[NotRequired[object]]


_VERTEX_COUNT_TOKENS_REQUEST_ADAPTER: Final = TypeAdapter(
    _VertexCountTokensRequest,
)


@with_config(ConfigDict(extra="allow", strict=True))
class _VertexCountTokensResponse(TypedDict):
    input_tokens: ReadOnly[NotRequired[int | float]]


_VERTEX_COUNT_TOKENS_RESPONSE_ADAPTER: Final = TypeAdapter(_VertexCountTokensResponse)


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
        api_base: str | None = None,
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
        publisher: Final = self._get_publisher_for_model(model)

        # Use custom api_base if provided, otherwise construct default
        if api_base:
            base_url = api_base
        else:
            base_url = get_vertex_base_url(vertex_location)

        # Construct the count-tokens endpoint
        # Format: /v1/projects/{project}/locations/{location}/publishers/{publisher}/models/count-tokens:rawPredict
        endpoint: Final = (
            f"{base_url}/v1/projects/{project_id}/locations/{vertex_location}/"
            f"publishers/{publisher}/models/count-tokens:rawPredict"
        )

        return endpoint

    @staticmethod
    def _strip_version_suffix(model: str) -> str:
        """
        Strip version suffixes (e.g. @default, @20251001) from model names.

        The Vertex AI count-tokens endpoint rejects model names that include
        version suffixes — for example, "claude-sonnet-4-6@default" returns
        "not supported for token counting" while "claude-sonnet-4-6" works.
        """
        if "@" in model:
            return model.split("@")[0]
        return model

    async def handle_count_tokens_request(
        self,
        model: str,
        request_data: Mapping[str, object],
        litellm_params: dict[str, Any],
    ) -> dict[str, Any]:
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
        # Strip version suffixes (@default, @20251001, etc.) — the Vertex AI
        # count-tokens endpoint does not accept versioned model names.
        normalized_model: Final = self._strip_version_suffix(model)
        validated_request_payload: Final = _VERTEX_COUNT_TOKENS_REQUEST_ADAPTER.validate_python(request_data)
        request_payload: Final = (
            {
                **validated_request_payload,
                "model": self._strip_version_suffix(validated_request_payload["model"]),
            }
            if "model" in validated_request_payload
            else validated_request_payload
        )

        # Validate request
        if "messages" not in request_payload:
            raise ValueError("messages required for token counting")

        # Extract Vertex AI credentials and settings
        vertex_credentials: Final = self.get_vertex_ai_credentials(litellm_params)
        vertex_project: Final = self.get_vertex_ai_project(litellm_params)

        # Check for count_tokens specific location override
        vertex_count_tokens_location: Final = litellm_params.get("vertex_count_tokens_location")
        vertex_location_raw: Final = self.get_vertex_ai_location(litellm_params)

        # Determine final location with precedence:
        # 1. vertex_count_tokens_location (if provided)
        # 2. vertex_location (if provided)
        # 3. Default to us-east5 for Claude models when no location is set
        # Supported regions: us-east5, europe-west1, asia-southeast1
        # https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/count-tokens
        vertex_location: Final = vertex_count_tokens_location or vertex_location_raw or "us-east5"

        # Get access token and resolved project ID
        access_token, project_id = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        # Build the endpoint URL
        endpoint_url: Final = self._build_count_tokens_endpoint(
            model=normalized_model,
            project_id=project_id,
            vertex_location=vertex_location,
            api_base=litellm_params.get("api_base"),
        )

        # Prepare headers
        headers: Final = {"Authorization": f"Bearer {access_token}"}
        request_json: Final = dict(request_payload)

        # Get async HTTP client
        from litellm import LlmProviders

        async_client: Final = get_async_httpx_client(llm_provider=LlmProviders.VERTEX_AI)

        # Make the request
        # Note: Partner models (especially Claude) accept Anthropic Messages API format directly
        response: Final = await async_client.post(
            endpoint_url,
            headers=headers,
            json=request_json,
            timeout=30.0,
        )

        # Check for errors
        if response.status_code != 200:
            error_text: Final = response.text
            raise ValueError(f"Token counting request failed with status {response.status_code}: {error_text}")

        # Parse response
        result: Final = _VERTEX_COUNT_TOKENS_RESPONSE_ADAPTER.validate_python(response.json())

        # Return token count
        # Vertex AI Anthropic returns: {"input_tokens": 123}
        return {
            "input_tokens": result.get("input_tokens", 0),
            "tokenizer_used": "vertex_ai_partner_models",
        }
