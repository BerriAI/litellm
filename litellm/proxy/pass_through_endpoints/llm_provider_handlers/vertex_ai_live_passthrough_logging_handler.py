"""
Vertex AI Live API WebSocket Passthrough Logging Handler

Handles cost tracking and logging for Vertex AI Live API WebSocket passthrough endpoints.
Supports different modalities: text, audio, video, and web search.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    PassThroughEndpointLoggingTypedDict,
)
from litellm.types.utils import LlmProviders, ModelResponse, Usage
from litellm.utils import get_model_info


class VertexAILivePassthroughLoggingHandler(BasePassthroughLoggingHandler):
    """
    Handles cost tracking and logging for Vertex AI Live API WebSocket passthrough.

    Supports:
    - Text tokens (input/output)
    - Audio tokens (input/output)
    - Video tokens (input/output)
    - Web search requests
    - Tool use tokens
    """

    def _build_complete_streaming_response(self, *args, **kwargs):
        """Not applicable for WebSocket passthrough."""
        return None

    def get_provider_config(self, model: str):
        """Return Vertex AI provider configuration."""
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        return VertexGeminiConfig()

    @property
    def llm_provider_name(self) -> LlmProviders:
        """Return the LLM provider name."""
        return LlmProviders.VERTEX_AI

    @staticmethod
    def _extract_usage_metadata_from_websocket_messages(
        websocket_messages: List[Dict],
    ) -> Optional[Dict]:
        """
        Extract and aggregate usage metadata from a list of WebSocket messages.

        Args:
            websocket_messages: List of WebSocket messages from the Live API

        Returns:
            Dictionary containing aggregated usage metadata, or None if not found
        """
        all_usage_metadata = []

        # Collect all usage metadata messages
        for message in websocket_messages:
            if isinstance(message, dict) and "usageMetadata" in message:
                all_usage_metadata.append(message["usageMetadata"])

        if not all_usage_metadata:
            return None

        # If only one usage metadata, return it as-is
        if len(all_usage_metadata) == 1:
            return all_usage_metadata[0]

        # Aggregate multiple usage metadata messages
        aggregated: Dict[str, Any] = {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
            "promptTokensDetails": [],
            "candidatesTokensDetails": [],
        }

        # Aggregate token counts
        for usage in all_usage_metadata:
            aggregated["promptTokenCount"] += usage.get("promptTokenCount", 0)
            aggregated["candidatesTokenCount"] += usage.get("candidatesTokenCount", 0)
            aggregated["totalTokenCount"] += usage.get("totalTokenCount", 0)

        # Aggregate token details by modality
        modality_totals = {}

        for usage in all_usage_metadata:
            # Process prompt tokens details
            for detail in usage.get("promptTokensDetails", []):
                modality = detail.get("modality", "TEXT")
                token_count = detail.get("tokenCount", 0)

                if modality not in modality_totals:
                    modality_totals[modality] = {"prompt": 0, "candidate": 0}
                modality_totals[modality]["prompt"] += token_count

            # Process candidate tokens details
            for detail in usage.get("candidatesTokensDetails", []):
                modality = detail.get("modality", "TEXT")
                token_count = detail.get("tokenCount", 0)

                if modality not in modality_totals:
                    modality_totals[modality] = {"prompt": 0, "candidate": 0}
                modality_totals[modality]["candidate"] += token_count

        # Convert aggregated modality totals back to details format
        for modality, totals in modality_totals.items():
            if totals["prompt"] > 0:
                aggregated["promptTokensDetails"].append(
                    {"modality": modality, "tokenCount": totals["prompt"]}
                )
            if totals["candidate"] > 0:
                aggregated["candidatesTokensDetails"].append(
                    {"modality": modality, "tokenCount": totals["candidate"]}
                )

        # Add any additional fields from the first usage metadata
        first_usage = all_usage_metadata[0]
        for key, value in first_usage.items():
            if key not in aggregated:
                aggregated[key] = value

        return aggregated

    @staticmethod
    def _calculate_live_api_cost(
        model: str,
        usage_metadata: Dict,
        custom_llm_provider: str = "vertex_ai",
    ) -> float:
        """
        Calculate cost for Vertex AI Live API based on usage metadata.

        Args:
            model: The model name (e.g., "gemini-2.0-flash-live-preview-04-09")
            usage_metadata: Usage metadata from the Live API response
            custom_llm_provider: The LLM provider (default: "vertex_ai")

        Returns:
            Total cost in USD
        """
        try:
            # Get model pricing information
            model_info = get_model_info(
                model=model, custom_llm_provider=custom_llm_provider
            )

            verbose_proxy_logger.debug(
                f"Vertex AI Live API model info for '{model}': {model_info}"
            )

            # Check if pricing info is available
            if not model_info or not model_info.get("input_cost_per_token"):
                verbose_proxy_logger.error(
                    f"No pricing info found for {model} in local model pricing database"
                )
                return 0.0

            total_cost = 0.0

            # Extract token counts from usage metadata
            prompt_token_count = usage_metadata.get("promptTokenCount", 0)
            candidates_token_count = usage_metadata.get("candidatesTokenCount", 0)

            # Calculate base text token costs
            input_cost_per_token = model_info.get("input_cost_per_token", 0.0)
            output_cost_per_token = model_info.get("output_cost_per_token", 0.0)

            total_cost += prompt_token_count * input_cost_per_token
            total_cost += candidates_token_count * output_cost_per_token

            # Handle modality-specific costs if present
            prompt_tokens_details = usage_metadata.get("promptTokensDetails", [])
            candidates_tokens_details = usage_metadata.get(
                "candidatesTokensDetails", []
            )

            # Process prompt tokens by modality
            for detail in prompt_tokens_details:
                modality = detail.get("modality", "TEXT")
                token_count = detail.get("tokenCount", 0)

                if modality == "AUDIO":
                    audio_cost_per_token = model_info.get(
                        "input_cost_per_audio_token", 0.0
                    )
                    total_cost += token_count * audio_cost_per_token
                elif modality == "VIDEO":
                    # Video tokens are typically per second, but we'll treat as per token for now
                    video_cost_per_token = model_info.get(
                        "input_cost_per_video_per_second", 0.0
                    )
                    total_cost += token_count * video_cost_per_token
                # TEXT tokens are already handled above

            # Process candidate tokens by modality
            for detail in candidates_tokens_details:
                modality = detail.get("modality", "TEXT")
                token_count = detail.get("tokenCount", 0)

                if modality == "AUDIO":
                    audio_cost_per_token = model_info.get(
                        "output_cost_per_audio_token", 0.0
                    )
                    total_cost += token_count * audio_cost_per_token
                elif modality == "VIDEO":
                    # Video tokens are typically per second, but we'll treat as per token for now
                    video_cost_per_token = model_info.get(
                        "output_cost_per_video_per_second", 0.0
                    )
                    total_cost += token_count * video_cost_per_token
                # TEXT tokens are already handled above

            # Handle web search costs if present
            tool_use_prompt_token_count = usage_metadata.get(
                "toolUsePromptTokenCount", 0
            )
            if tool_use_prompt_token_count > 0:
                # Web search typically has a fixed cost per request
                web_search_cost = model_info.get("web_search_cost_per_request", 0.0)
                if isinstance(web_search_cost, (int, float)) and web_search_cost > 0:
                    total_cost += web_search_cost
                else:
                    # Fallback to token-based pricing for tool use
                    total_cost += tool_use_prompt_token_count * input_cost_per_token

            verbose_proxy_logger.debug(
                f"Vertex AI Live API cost calculation - Model: {model}, "
                f"Prompt tokens: {prompt_token_count}, "
                f"Candidate tokens: {candidates_token_count}, "
                f"Total cost: ${total_cost:.6f}"
            )

            return total_cost

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error calculating Vertex AI Live API cost: {e}"
            )
            return 0.0

    @staticmethod
    def _create_usage_object_from_metadata(
        usage_metadata: Dict,
        model: str,
    ) -> Usage:
        """
        Create a LiteLLM Usage object from Live API usage metadata.

        Args:
            usage_metadata: Usage metadata from the Live API response
            model: The model name

        Returns:
            LiteLLM Usage object
        """
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", 0)

        # Create modality-specific token details if available
        prompt_tokens_details = usage_metadata.get("promptTokensDetails", [])
        candidates_tokens_details = usage_metadata.get("candidatesTokensDetails", [])

        # Extract text tokens from details
        text_prompt_tokens = 0
        text_completion_tokens = 0

        for detail in prompt_tokens_details:
            if detail.get("modality") == "TEXT":
                text_prompt_tokens = detail.get("tokenCount", 0)
                break

        for detail in candidates_tokens_details:
            if detail.get("modality") == "TEXT":
                text_completion_tokens = detail.get("tokenCount", 0)
                break

        # If no text tokens found in details, use total counts
        if text_prompt_tokens == 0:
            text_prompt_tokens = prompt_tokens
        if text_completion_tokens == 0:
            text_completion_tokens = completion_tokens

        return Usage(
            prompt_tokens=text_prompt_tokens,
            completion_tokens=text_completion_tokens,
            total_tokens=total_tokens,
        )

    def vertex_ai_live_passthrough_handler(
        self,
        websocket_messages: List[Dict],
        logging_obj,
        url_route: str,
        start_time: datetime,
        end_time: datetime,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle cost tracking and logging for Vertex AI Live API WebSocket passthrough.

        Args:
            websocket_messages: List of WebSocket messages from the Live API
            logging_obj: LiteLLM logging object
            url_route: The URL route that was called
            start_time: Request start time
            end_time: Request end time
            request_body: The original request body
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing the result and kwargs for logging
        """
        try:
            # Extract model from request body or kwargs
            model = kwargs.get("model", "gemini-2.0-flash-live-preview-04-09")
            custom_llm_provider = kwargs.get("custom_llm_provider", "vertex_ai")
            verbose_proxy_logger.debug(
                f"Vertex AI Live API model: {model}, custom_llm_provider: {custom_llm_provider}"
            )

            # Extract usage metadata from WebSocket messages
            usage_metadata = self._extract_usage_metadata_from_websocket_messages(
                websocket_messages
            )

            if not usage_metadata:
                verbose_proxy_logger.warning(
                    "No usage metadata found in Vertex AI Live API WebSocket messages"
                )
                return {
                    "result": None,
                    "kwargs": kwargs,
                }

            # Calculate cost using Live API specific pricing
            response_cost = self._calculate_live_api_cost(
                model=model,
                usage_metadata=usage_metadata,
                custom_llm_provider=custom_llm_provider,
            )

            # Create Usage object for standard LiteLLM logging
            usage = self._create_usage_object_from_metadata(
                usage_metadata=usage_metadata,
                model=model,
            )

            # Create a mock ModelResponse for standard logging
            litellm_model_response = ModelResponse(
                id=f"vertex-ai-live-{start_time.timestamp()}",
                object="chat.completion",
                created=int(start_time.timestamp()),
                model=model,
                usage=usage,
                choices=[],
            )

            # Update kwargs with cost information
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = custom_llm_provider

            # Safely log the model name: only allow known safe formats, redact otherwise.
            import re
            allowed_pattern = re.compile(r"^[A-Za-z0-9._\-:]+$")
            safe_model = model if isinstance(model, str) and allowed_pattern.match(model) else "[REDACTED]"
            verbose_proxy_logger.debug(
                f"Vertex AI Live API passthrough cost tracking - "
                f"Model: {safe_model}, Cost: ${response_cost:.6f}, "
                f"Prompt tokens: {usage.prompt_tokens}, "
                f"Completion tokens: {usage.completion_tokens}"
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in Vertex AI Live API passthrough handler: {e}"
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }
