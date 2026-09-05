"""
Vertex AI Live API WebSocket Passthrough Logging Handler

Handles cost tracking and logging for Vertex AI Live API WebSocket passthrough endpoints.
Supports different modalities: text, audio, video, and web search.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    PassThroughEndpointLoggingTypedDict,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    LlmProviders,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


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
        return

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
        websocket_messages: list[dict],
    ) -> dict | None:
        """
        Extract and aggregate usage metadata from a list of WebSocket messages.

        Args:
            websocket_messages: List of WebSocket messages from the Live API

        Returns:
            Dictionary containing aggregated usage metadata, or None if not found
        """
        all_usage_metadata: Final = []

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
        aggregated: Final[dict[str, Any]] = {
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
        modality_totals: Final = {}

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
                aggregated["promptTokensDetails"].append({"modality": modality, "tokenCount": totals["prompt"]})
            if totals["candidate"] > 0:
                aggregated["candidatesTokensDetails"].append({"modality": modality, "tokenCount": totals["candidate"]})

        # Add any additional fields from the first usage metadata
        first_usage: Final = all_usage_metadata[0]
        for key, value in first_usage.items():
            if key not in aggregated:
                aggregated[key] = value

        return aggregated

    @staticmethod
    def _tokens_by_modality(details: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
        """Sum a Live API ``*TokensDetails`` list into ``{modality: tokenCount}``."""
        return MappingProxyType(
            {
                modality: sum(d.get("tokenCount", 0) for d in details if d.get("modality", "TEXT") == modality)
                for modality in {d.get("modality", "TEXT") for d in details}
            }
        )

    @staticmethod
    def _create_usage_object_from_metadata(
        usage_metadata: dict,
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
        _ = model

        prompt_by_modality: Final = VertexAILivePassthroughLoggingHandler._tokens_by_modality(
            usage_metadata.get("promptTokensDetails") or []
        )
        candidates_by_modality: Final = VertexAILivePassthroughLoggingHandler._tokens_by_modality(
            usage_metadata.get("candidatesTokensDetails") or []
        )

        prompt_tokens: Final = usage_metadata.get("promptTokenCount", 0) or sum(prompt_by_modality.values())
        completion_tokens: Final = usage_metadata.get("candidatesTokenCount", 0) or sum(candidates_by_modality.values())

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage_metadata.get("totalTokenCount", 0) or (prompt_tokens + completion_tokens),
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=prompt_by_modality.get("TEXT"),
                audio_tokens=prompt_by_modality.get("AUDIO"),
                image_tokens=prompt_by_modality.get("IMAGE"),
                video_tokens=prompt_by_modality.get("VIDEO"),
                cached_tokens=usage_metadata.get("cachedContentTokenCount", 0) or 0,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                text_tokens=candidates_by_modality.get("TEXT"),
                audio_tokens=candidates_by_modality.get("AUDIO"),
                image_tokens=candidates_by_modality.get("IMAGE"),
                video_tokens=candidates_by_modality.get("VIDEO"),
            ),
        )

    def vertex_ai_live_passthrough_handler(
        self,
        websocket_messages: list[dict],
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
            model: Final = kwargs.get("model", "gemini-2.0-flash-live-preview-04-09")
            custom_llm_provider: Final = kwargs.get("custom_llm_provider", "vertex_ai")
            verbose_proxy_logger.debug(
                "Vertex AI Live API model: %s, custom_llm_provider: %s", model, custom_llm_provider
            )

            # Extract usage metadata from WebSocket messages
            usage_metadata: Final = self._extract_usage_metadata_from_websocket_messages(websocket_messages)

            if not usage_metadata:
                verbose_proxy_logger.warning("No usage metadata found in Vertex AI Live API WebSocket messages")
                return {
                    "result": None,
                    "kwargs": kwargs,
                }

            usage: Final = self._create_usage_object_from_metadata(
                usage_metadata=usage_metadata,
                model=model,
            )

            # Create a mock ModelResponse for standard logging
            litellm_model_response: Final = ModelResponse(
                id=f"vertex-ai-live-{start_time.timestamp()}",
                object="chat.completion",
                created=int(start_time.timestamp()),
                model=model,
                usage=usage,
                choices=[],
            )

            kwargs["model"] = model
            kwargs["custom_llm_provider"] = custom_llm_provider

            # Safely log the model name: only allow known safe formats, redact otherwise.
            import re

            allowed_pattern: Final = re.compile(r"^[A-Za-z0-9._\-:]+$")
            safe_model: Final = model if isinstance(model, str) and allowed_pattern.match(model) else "[REDACTED]"
            verbose_proxy_logger.debug(
                "Vertex AI Live API passthrough cost tracking - Model: %s, "
                "Prompt tokens: %s %s, Completion tokens: %s %s",
                safe_model,
                usage.prompt_tokens,
                usage.prompt_tokens_details,
                usage.completion_tokens,
                usage.completion_tokens_details,
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error("Error in Vertex AI Live API passthrough handler: %s", e)
            return {
                "result": None,
                "kwargs": kwargs,
            }
