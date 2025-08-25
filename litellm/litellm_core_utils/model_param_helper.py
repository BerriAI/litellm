from typing import Set

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

from litellm._logging import verbose_logger
from litellm.types.rerank import RerankRequest


class ModelParamHelper:
    @staticmethod
    def get_standard_logging_model_parameters(
        model_parameters: dict,
    ) -> dict:
        """ """
        standard_logging_model_parameters: dict = {}
        supported_model_parameters = (
            ModelParamHelper._get_relevant_args_to_use_for_logging()
        )

        for key, value in model_parameters.items():
            if key in supported_model_parameters:
                standard_logging_model_parameters[key] = value
        return standard_logging_model_parameters

    @staticmethod
    def get_exclude_params_for_model_parameters() -> Set[str]:
        return set(["messages", "prompt", "input"])

    @staticmethod
    def _get_relevant_args_to_use_for_logging() -> Set[str]:
        """
        Gets all relevant llm api params besides the ones with prompt content
        """
        all_openai_llm_api_params = ModelParamHelper._get_all_llm_api_params()
        # Exclude parameters that contain prompt content
        combined_kwargs = all_openai_llm_api_params.difference(
            set(ModelParamHelper.get_exclude_params_for_model_parameters())
        )
        return combined_kwargs

    @staticmethod
    def _get_all_llm_api_params() -> Set[str]:
        """
        Gets the supported kwargs for each call type and combines them
        """
        chat_completion_kwargs = (
            ModelParamHelper._get_litellm_supported_chat_completion_kwargs()
        )
        text_completion_kwargs = (
            ModelParamHelper._get_litellm_supported_text_completion_kwargs()
        )
        embedding_kwargs = ModelParamHelper._get_litellm_supported_embedding_kwargs()
        transcription_kwargs = (
            ModelParamHelper._get_litellm_supported_transcription_kwargs()
        )
        rerank_kwargs = ModelParamHelper._get_litellm_supported_rerank_kwargs()
        exclude_kwargs = ModelParamHelper._get_exclude_kwargs()

        combined_kwargs = chat_completion_kwargs.union(
            text_completion_kwargs,
            embedding_kwargs,
            transcription_kwargs,
            rerank_kwargs,
        )
        combined_kwargs = combined_kwargs.difference(exclude_kwargs)
        return combined_kwargs

    @staticmethod
    def get_litellm_provider_specific_params_for_chat_params() -> Set[str]:
        return set(["thinking"])

    @staticmethod
    def _get_litellm_supported_chat_completion_kwargs() -> Set[str]:
        """
        Get the litellm supported chat completion kwargs

        This follows the OpenAI API Spec
        """
        non_streaming_params: Set[str] = set(
            getattr(CompletionCreateParamsNonStreaming, "__annotations__", {}).keys()
        )
        streaming_params: Set[str] = set(
            getattr(CompletionCreateParamsStreaming, "__annotations__", {}).keys()
        )
        litellm_provider_specific_params: Set[str] = (
            ModelParamHelper.get_litellm_provider_specific_params_for_chat_params()
        )
        all_chat_completion_kwargs: Set[str] = non_streaming_params.union(
            streaming_params
        ).union(litellm_provider_specific_params)
        return all_chat_completion_kwargs

    @staticmethod
    def _get_litellm_supported_text_completion_kwargs() -> Set[str]:
        """
        Get the litellm supported text completion kwargs

        This follows the OpenAI API Spec
        """
        all_text_completion_kwargs = set(
            getattr(
                TextCompletionCreateParamsNonStreaming, "__annotations__", {}
            ).keys()
        ).union(
            set(
                getattr(
                    TextCompletionCreateParamsStreaming, "__annotations__", {}
                ).keys()
            )
        )
        return all_text_completion_kwargs

    @staticmethod
    def _get_litellm_supported_rerank_kwargs() -> Set[str]:
        """
        Get the litellm supported rerank kwargs
        """
        return set(RerankRequest.model_fields.keys())

    @staticmethod
    def _get_litellm_supported_embedding_kwargs() -> Set[str]:
        """
        Get the litellm supported embedding kwargs

        This follows the OpenAI API Spec
        """
        return set(getattr(EmbeddingCreateParams, "__annotations__", {}).keys())

    @staticmethod
    def _get_litellm_supported_transcription_kwargs() -> Set[str]:
        """
        Get the litellm supported transcription kwargs

        This follows the OpenAI API Spec
        """
        try:
            from openai.types.audio.transcription_create_params import (
                TranscriptionCreateParamsNonStreaming,
                TranscriptionCreateParamsStreaming,
            )

            non_streaming_kwargs = set(
                getattr(
                    TranscriptionCreateParamsNonStreaming, "__annotations__", {}
                ).keys()
            )
            streaming_kwargs = set(
                getattr(
                    TranscriptionCreateParamsStreaming, "__annotations__", {}
                ).keys()
            )

            all_transcription_kwargs = non_streaming_kwargs.union(streaming_kwargs)
            return all_transcription_kwargs
        except Exception as e:
            verbose_logger.debug("Error getting transcription kwargs %s", str(e))
            return set()

    @staticmethod
    def _get_exclude_kwargs() -> Set[str]:
        """
        Get the kwargs to exclude from the cache key
        """
        return set(["metadata"])
