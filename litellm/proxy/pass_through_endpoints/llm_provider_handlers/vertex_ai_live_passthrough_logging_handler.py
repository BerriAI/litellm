"""
Vertex AI Live API WebSocket Passthrough Logging Handler

Handles cost tracking and logging for Vertex AI Live API WebSocket passthrough endpoints.
Supports different modalities: text, audio, video, and web search.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import chain
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

_AGGREGATED_FIELDS: Final = frozenset(
    {
        "promptTokenCount",
        "candidatesTokenCount",
        "totalTokenCount",
        "promptTokensDetails",
        "candidatesTokensDetails",
    }
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
    def _resolve_detail_counts(
        details: Sequence[Mapping[str, Any]],
        declared_total: object,
    ) -> tuple[tuple[str, int], ...]:
        """
        Pair each of one turn's ``*TokensDetails`` entries with its token count.

        Live sometimes names the modality that carries the rest of a turn without a
        ``tokenCount``, and reading the absent key as zero drops those tokens from the
        breakdown, so real audio ends up priced as text. A lone unpriced entry therefore takes
        whatever the turn's declared count leaves over. Two or more cannot be told apart, so
        they are left out and the cost calculator charges the remainder as text.
        """
        priced: Final = tuple(
            (str(detail.get("modality", "TEXT")), count)
            for detail in details
            if isinstance(count := detail.get("tokenCount"), int)
        )
        unpriced: Final = tuple(
            str(detail.get("modality", "TEXT")) for detail in details if not isinstance(detail.get("tokenCount"), int)
        )
        if len(unpriced) != 1 or not isinstance(declared_total, int):
            return priced
        residual: Final = declared_total - sum(count for _, count in priced)
        return priced if residual <= 0 else (*priced, (unpriced[0], residual))

    @staticmethod
    def _sum_by_modality(counts: Sequence[tuple[str, int]]) -> Mapping[str, int]:
        """Total the (modality, tokenCount) pairs of one or more turns per modality."""
        return MappingProxyType({modality: sum(c for m, c in counts if m == modality) for modality, _ in counts})

    @staticmethod
    def _merged_modality_totals(
        snapshots: Sequence[Mapping[str, Any]],
        count_key: str,
        details_key: str,
    ) -> Mapping[str, int]:
        """Total every turn's per-modality counts, so the breakdown adds up the way the totals do."""
        return VertexAILivePassthroughLoggingHandler._sum_by_modality(
            tuple(
                chain.from_iterable(
                    VertexAILivePassthroughLoggingHandler._resolve_detail_counts(
                        snapshot.get(details_key) or [], snapshot.get(count_key)
                    )
                    for snapshot in snapshots
                )
            )
        )

    @staticmethod
    def _extract_usage_metadata_from_websocket_messages(
        websocket_messages: list[dict],
    ) -> dict | None:
        """
        Extract and aggregate usage metadata from a list of WebSocket messages.

        Live emits one ``usageMetadata`` per turn and Google charges per turn for every token in
        the session context window, which is the current turn's tokens plus all accumulated
        tokens from previous turns, so the turns add up rather than restating each other. See
        the Live API note under https://cloud.google.com/vertex-ai/generative-ai/pricing.

        Args:
            websocket_messages: List of WebSocket messages from the Live API

        Returns:
            Dictionary containing aggregated usage metadata, or None if not found
        """
        snapshots: Final = tuple(
            message["usageMetadata"]
            for message in websocket_messages
            if isinstance(message, dict) and isinstance(message.get("usageMetadata"), dict)
        )

        if not snapshots:
            return None

        prompt_totals: Final = VertexAILivePassthroughLoggingHandler._merged_modality_totals(
            snapshots, "promptTokenCount", "promptTokensDetails"
        )
        candidate_totals: Final = VertexAILivePassthroughLoggingHandler._merged_modality_totals(
            snapshots, "candidatesTokenCount", "candidatesTokensDetails"
        )
        return {
            **{key: value for key, value in snapshots[0].items() if key not in _AGGREGATED_FIELDS},
            "promptTokenCount": sum(snapshot.get("promptTokenCount", 0) for snapshot in snapshots),
            "candidatesTokenCount": sum(snapshot.get("candidatesTokenCount", 0) for snapshot in snapshots),
            "totalTokenCount": sum(snapshot.get("totalTokenCount", 0) for snapshot in snapshots),
            "promptTokensDetails": [
                {"modality": modality, "tokenCount": count} for modality, count in prompt_totals.items() if count > 0
            ],
            "candidatesTokensDetails": [
                {"modality": modality, "tokenCount": count} for modality, count in candidate_totals.items() if count > 0
            ],
        }

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
        prompt_by_modality: Final = VertexAILivePassthroughLoggingHandler._sum_by_modality(
            VertexAILivePassthroughLoggingHandler._resolve_detail_counts(
                usage_metadata.get("promptTokensDetails") or [], usage_metadata.get("promptTokenCount")
            )
        )
        candidates_by_modality: Final = VertexAILivePassthroughLoggingHandler._sum_by_modality(
            VertexAILivePassthroughLoggingHandler._resolve_detail_counts(
                usage_metadata.get("candidatesTokensDetails") or [], usage_metadata.get("candidatesTokenCount")
            )
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
                tool_use_tokens=usage_metadata.get("toolUsePromptTokenCount") or None,
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

            # Create Usage object for standard LiteLLM logging
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
