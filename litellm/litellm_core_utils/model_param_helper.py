from functools import lru_cache
from typing import Final

from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.completion_create_params import (
    CompletionCreateParamsNonStreaming as TextCompletionCreateParamsNonStreaming,
)
from openai.types.completion_create_params import (
    CompletionCreateParamsStreaming as TextCompletionCreateParamsStreaming,
)
from openai.types.embedding_create_params import EmbeddingCreateParams
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
    ResponseCreateParamsStreaming,
)

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.rerank import RerankRequest


class ModelParamHelper:
    # Cached at class level — deterministic set built from static OpenAI type annotations
    _relevant_logging_args: frozenset = frozenset()

    @staticmethod
    def get_standard_logging_model_parameters(
        model_parameters: dict,
    ) -> dict:
        """ """
        standard_logging_model_parameters: Final[dict] = {}
        supported_model_parameters: Final = ModelParamHelper._relevant_logging_args

        for key, value in model_parameters.items():
            if key in supported_model_parameters:
                standard_logging_model_parameters[key] = value
        return standard_logging_model_parameters

    @staticmethod
    def get_exclude_params_for_model_parameters() -> set[str]:
        return set(["messages", "prompt", "input", "system"])

    @staticmethod
    def _get_relevant_args_to_use_for_logging() -> set[str]:
        """
        Gets all relevant llm api params besides the ones with prompt content
        """
        all_openai_llm_api_params: Final = ModelParamHelper._get_all_llm_api_params()
        # Exclude parameters that contain prompt content
        combined_kwargs: Final = all_openai_llm_api_params.difference(
            set(ModelParamHelper.get_exclude_params_for_model_parameters())
        )
        return combined_kwargs

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_all_llm_api_params() -> set[str]:
        """
        Gets the supported kwargs for each call type and combines them.

        The result is derived from static type annotations and fixed sets, so it
        is constant for the process lifetime. It is computed once and cached
        because it is rebuilt on every request through both the cache-key path
        (``Cache.get_cache_key``) and the spend-logging path
        (``_get_relevant_args_to_use_for_logging``). Callers treat the result as
        read-only.
        """
        chat_completion_kwargs: Final = ModelParamHelper._get_litellm_supported_chat_completion_kwargs()
        text_completion_kwargs: Final = ModelParamHelper._get_litellm_supported_text_completion_kwargs()
        embedding_kwargs: Final = ModelParamHelper._get_litellm_supported_embedding_kwargs()
        transcription_kwargs: Final = ModelParamHelper._get_litellm_supported_transcription_kwargs()
        rerank_kwargs: Final = ModelParamHelper._get_litellm_supported_rerank_kwargs()
        responses_api_kwargs: Final = ModelParamHelper._get_litellm_supported_responses_api_kwargs()
        anthropic_messages_kwargs: Final = ModelParamHelper._get_litellm_supported_anthropic_messages_kwargs()
        exclude_kwargs: Final = ModelParamHelper._get_exclude_kwargs()

        combined_kwargs = chat_completion_kwargs.union(
            text_completion_kwargs,
            embedding_kwargs,
            transcription_kwargs,
            rerank_kwargs,
            responses_api_kwargs,
            anthropic_messages_kwargs,
        )
        combined_kwargs = combined_kwargs.difference(exclude_kwargs)
        return combined_kwargs

    @staticmethod
    def get_litellm_provider_specific_params_for_chat_params() -> set[str]:
        return set(["thinking"])

    @staticmethod
    def _get_litellm_supported_chat_completion_kwargs() -> set[str]:
        """
        Get the litellm supported chat completion kwargs

        This follows the OpenAI API Spec
        """
        non_streaming_params: set[str] = set(getattr(CompletionCreateParamsNonStreaming, "__annotations__", {}).keys())
        streaming_params: Final[set[str]] = set(getattr(CompletionCreateParamsStreaming, "__annotations__", {}).keys())
        litellm_provider_specific_params: Final[set[str]] = (
            ModelParamHelper.get_litellm_provider_specific_params_for_chat_params()
        )
        all_chat_completion_kwargs: Final[set[str]] = non_streaming_params.union(streaming_params).union(
            litellm_provider_specific_params
        )
        return all_chat_completion_kwargs

    @staticmethod
    def _get_litellm_supported_text_completion_kwargs() -> set[str]:
        """
        Get the litellm supported text completion kwargs

        This follows the OpenAI API Spec
        """
        all_text_completion_kwargs: Final = set(
            getattr(TextCompletionCreateParamsNonStreaming, "__annotations__", {}).keys()
        ).union(set(getattr(TextCompletionCreateParamsStreaming, "__annotations__", {}).keys()))
        return all_text_completion_kwargs

    @staticmethod
    def _get_litellm_supported_rerank_kwargs() -> set[str]:
        """
        Get the litellm supported rerank kwargs
        """
        return set(RerankRequest.model_fields.keys())

    @staticmethod
    def _get_litellm_supported_embedding_kwargs() -> set[str]:
        """
        Get the litellm supported embedding kwargs

        This follows the OpenAI API Spec
        """
        return set(getattr(EmbeddingCreateParams, "__annotations__", {}).keys())

    @staticmethod
    def _get_litellm_supported_transcription_kwargs() -> set[str]:
        """
        Get the litellm supported transcription kwargs

        This follows the OpenAI API Spec
        """
        try:
            from openai.types.audio.transcription_create_params import (
                TranscriptionCreateParamsNonStreaming,
                TranscriptionCreateParamsStreaming,
            )

            non_streaming_kwargs = set(getattr(TranscriptionCreateParamsNonStreaming, "__annotations__", {}).keys())
            streaming_kwargs: Final = set(getattr(TranscriptionCreateParamsStreaming, "__annotations__", {}).keys())

            all_transcription_kwargs: Final = non_streaming_kwargs.union(streaming_kwargs)
            return all_transcription_kwargs
        except Exception as e:
            verbose_logger.debug("Error getting transcription kwargs %s", str(e))
            return set()

    @staticmethod
    def _get_litellm_supported_responses_api_kwargs() -> set[str]:
        """
        Get the litellm supported responses API kwargs

        This follows the OpenAI API Spec
        """
        non_streaming_params: set[str] = set(getattr(ResponseCreateParamsNonStreaming, "__annotations__", {}).keys())
        streaming_params: Final[set[str]] = set(getattr(ResponseCreateParamsStreaming, "__annotations__", {}).keys())
        return non_streaming_params.union(streaming_params)

    @staticmethod
    def _get_litellm_supported_anthropic_messages_kwargs() -> frozenset[str]:
        """
        Get the litellm supported Anthropic /v1/messages kwargs
        """
        return frozenset(AnthropicMessagesRequest.__annotations__.keys())

    @staticmethod
    def _get_exclude_kwargs() -> set[str]:
        """
        Get the kwargs to exclude from the cache key
        """
        return set(["metadata", "litellm_metadata"])


ModelParamHelper._relevant_logging_args = frozenset(ModelParamHelper._get_relevant_args_to_use_for_logging())
