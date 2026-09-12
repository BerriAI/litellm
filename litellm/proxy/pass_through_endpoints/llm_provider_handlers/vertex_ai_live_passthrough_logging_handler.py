"""
Vertex AI Live API WebSocket Passthrough Logging Handler

Handles cost tracking and logging for Vertex AI Live API WebSocket passthrough endpoints.
Supports different modalities: text, audio, video, and web search.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import chain, pairwise
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    PassThroughEndpointLoggingTypedDict,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    CostBreakdown,
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
        "toolUsePromptTokenCount",
        "promptTokensDetails",
        "candidatesTokensDetails",
    }
)


def _detail_entries(raw: object) -> tuple[Mapping[str, object], ...]:
    """Narrow one turn's ``*TokensDetails`` value to the entries that are actually shaped like one."""
    return tuple(entry for entry in raw if isinstance(entry, Mapping)) if isinstance(raw, Sequence) else ()


def _grounding_metadata(websocket_messages: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    """Collect every ``serverContent.groundingMetadata`` a session emitted.

    Live reports grounding in the server frames, never in ``usageMetadata``, so the per-query
    charge has to be counted here rather than derived from the token totals.
    """
    return tuple(
        metadata
        for message in websocket_messages
        if isinstance(message, Mapping)
        for server_content in (message.get("serverContent"),)
        if isinstance(server_content, Mapping)
        for metadata in (server_content.get("groundingMetadata"),)
        if isinstance(metadata, Mapping)
    )


def _turns(websocket_messages: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    """Split a session at every ``usageMetadata`` frame; frames after the last one never got their usage."""
    closes: Final = tuple(
        index + 1
        for index, message in enumerate(websocket_messages)
        if isinstance(message, Mapping) and isinstance(message.get("usageMetadata"), dict)
    )
    return tuple(tuple(websocket_messages[start:end]) for start, end in pairwise((0, *closes)))


_SummedField: TypeAlias = Literal[
    "input_cost",
    "output_cost",
    "tool_usage_cost",
    "cache_read_cost",
    "cache_creation_cost",
    "reasoning_cost",
    "original_cost",
    "discount_amount",
    "margin_fixed_amount",
    "margin_total_amount",
]


def _summed(breakdowns: Sequence[CostBreakdown], field: _SummedField) -> float | None:
    values: Final = tuple(value for breakdown in breakdowns if (value := breakdown.get(field)) is not None)
    return sum(values) if values else None


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
        details: Sequence[Mapping[str, object]],
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
        snapshots: Sequence[Mapping[str, object]],
        count_key: str,
        details_key: str,
    ) -> Mapping[str, int]:
        """Total every turn's per-modality counts, so the breakdown adds up the way the totals do."""
        return VertexAILivePassthroughLoggingHandler._sum_by_modality(
            tuple(
                chain.from_iterable(
                    VertexAILivePassthroughLoggingHandler._resolve_detail_counts(
                        _detail_entries(snapshot.get(details_key)), snapshot.get(count_key)
                    )
                    for snapshot in snapshots
                )
            )
        )

    @staticmethod
    def _extract_usage_metadata_from_websocket_messages(
        websocket_messages: Sequence[object],
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
            metadata
            for message in websocket_messages
            if isinstance(message, Mapping)
            for metadata in (message.get("usageMetadata"),)
            if isinstance(metadata, dict)
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
            "toolUsePromptTokenCount": sum(snapshot.get("toolUsePromptTokenCount", 0) for snapshot in snapshots),
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
        grounding_metadata: Sequence[Mapping[str, object]] = (),
    ) -> Usage:
        """
        Create a LiteLLM Usage object from Live API usage metadata.

        Args:
            usage_metadata: Usage metadata from the Live API response
            model: The model name
            grounding_metadata: Every ``serverContent.groundingMetadata`` the session emitted, so
                Search and Maps grounding carry their per-query charge

        Returns:
            LiteLLM Usage object
        """
        prompt_by_modality: Final = VertexAILivePassthroughLoggingHandler._sum_by_modality(
            VertexAILivePassthroughLoggingHandler._resolve_detail_counts(
                _detail_entries(usage_metadata.get("promptTokensDetails")), usage_metadata.get("promptTokenCount")
            )
        )
        candidates_by_modality: Final = VertexAILivePassthroughLoggingHandler._sum_by_modality(
            VertexAILivePassthroughLoggingHandler._resolve_detail_counts(
                _detail_entries(usage_metadata.get("candidatesTokensDetails")),
                usage_metadata.get("candidatesTokenCount"),
            )
        )

        prompt_tokens: Final = usage_metadata.get("promptTokenCount", 0) or sum(prompt_by_modality.values())
        completion_tokens: Final = usage_metadata.get("candidatesTokenCount", 0) or sum(candidates_by_modality.values())

        usage: Final = Usage(
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
        if grounding_metadata:
            from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
                VertexGeminiConfig,
            )

            VertexGeminiConfig._set_grounding_usage_counters(  # pyright: ignore[reportPrivateUsage]  # shared with the chat path; no public alias exists yet
                usage, grounding_metadata
            )
        return usage

    def _session_usage(self, websocket_messages: Sequence[object], model: str) -> Usage | None:
        usage_metadata: Final = self._extract_usage_metadata_from_websocket_messages(websocket_messages)
        if usage_metadata is None:
            return None
        return self._create_usage_object_from_metadata(
            usage_metadata=usage_metadata,
            grounding_metadata=_grounding_metadata(websocket_messages),
            model=model,
        )

    def _turn_cost(
        self,
        turn: Sequence[object],
        model: str,
        logging_obj: LiteLLMLoggingObj,
    ) -> tuple[float, CostBreakdown] | None:
        usage: Final = self._session_usage(turn, model)
        if usage is None:
            return None
        cost: Final = logging_obj._response_cost_calculator(  # pyright: ignore[reportPrivateUsage]  # the call's own calculator keeps custom pricing and the deployment's region in step with the spend row
            result=ModelResponse(model=model, usage=usage),
            litellm_model_name=model,
        )
        if cost is None:
            return None
        breakdown: Final = logging_obj.cost_breakdown
        return None if breakdown is None else (cost, breakdown)

    def _session_cost(
        self,
        websocket_messages: Sequence[object],
        model: str,
        logging_obj: LiteLLMLoggingObj,
    ) -> float | None:
        """Price each turn on its own tokens and grounding, so two grounded turns pay the query fee twice."""
        turn_costs: Final = tuple(self._turn_cost(turn, model, logging_obj) for turn in _turns(websocket_messages))
        priced: Final = tuple(turn_cost for turn_cost in turn_costs if turn_cost is not None)
        if not priced or len(priced) != len(turn_costs):
            return None
        breakdowns: Final = tuple(breakdown for _, breakdown in priced)
        first: Final = breakdowns[0]
        total_cost: Final = sum(cost for cost, _ in priced)
        logging_obj.set_cost_breakdown(
            input_cost=_summed(breakdowns, "input_cost") or 0.0,
            output_cost=_summed(breakdowns, "output_cost") or 0.0,
            total_cost=total_cost,
            cost_for_built_in_tools_cost_usd_dollar=_summed(breakdowns, "tool_usage_cost") or 0.0,
            original_cost=_summed(breakdowns, "original_cost"),
            discount_percent=first.get("discount_percent"),
            discount_amount=_summed(breakdowns, "discount_amount"),
            margin_percent=first.get("margin_percent"),
            margin_fixed_amount=_summed(breakdowns, "margin_fixed_amount"),
            margin_total_amount=_summed(breakdowns, "margin_total_amount"),
            cache_read_cost=_summed(breakdowns, "cache_read_cost"),
            cache_creation_cost=_summed(breakdowns, "cache_creation_cost"),
            reasoning_cost=_summed(breakdowns, "reasoning_cost"),
            service_tier=first.get("service_tier"),
            data_residency=first.get("data_residency"),
            vertex_location=first.get("vertex_location"),
        )
        return total_cost

    def vertex_ai_live_passthrough_handler(
        self,
        websocket_messages: Sequence[object],
        logging_obj: LiteLLMLoggingObj,
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
            requested_model: Final = kwargs.get("model")
            model: Final = (
                requested_model if isinstance(requested_model, str) else "gemini-2.0-flash-live-preview-04-09"
            )
            custom_llm_provider: Final = kwargs.get("custom_llm_provider", "vertex_ai")
            verbose_proxy_logger.debug(
                "Vertex AI Live API model: %s, custom_llm_provider: %s", model, custom_llm_provider
            )

            usage: Final = self._session_usage(websocket_messages, model)

            if usage is None:
                verbose_proxy_logger.warning("No usage metadata found in Vertex AI Live API WebSocket messages")
                return {
                    "result": None,
                    "kwargs": kwargs,
                }

            response_cost: Final = self._session_cost(websocket_messages, model, logging_obj)

            # Create a mock ModelResponse for standard logging
            litellm_model_response: Final = ModelResponse(
                id=f"vertex-ai-live-{start_time.timestamp()}",
                object="chat.completion",
                created=int(start_time.timestamp()),
                model=model,
                usage=usage,
                choices=[],
            )
            if response_cost is not None:
                litellm_model_response._hidden_params["response_cost"] = response_cost  # pyright: ignore[reportPrivateUsage]  # the logger reads the cost off the response's hidden params; the constructor's hidden_params kwarg is reset by pydantic

            kwargs["model"] = model
            kwargs["custom_llm_provider"] = custom_llm_provider

            # Safely log the model name: only allow known safe formats, redact otherwise.
            import re

            allowed_pattern: Final = re.compile(r"^[A-Za-z0-9._\-:]+$")
            safe_model: Final = model if allowed_pattern.match(model) else "[REDACTED]"
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
