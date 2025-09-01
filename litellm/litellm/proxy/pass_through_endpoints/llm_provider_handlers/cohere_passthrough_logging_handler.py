from typing import List, Optional, Union

from litellm import stream_chunk_builder
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.llms.cohere.common_utils import (
    ModelResponseIterator as CohereModelResponseIterator,
)
from litellm.types.utils import LlmProviders, ModelResponse, TextCompletionResponse

from .base_passthrough_logging_handler import BasePassthroughLoggingHandler


class CoherePassthroughLoggingHandler(BasePassthroughLoggingHandler):
    @property
    def llm_provider_name(self) -> LlmProviders:
        return LlmProviders.COHERE

    def get_provider_config(self, model: str) -> BaseConfig:
        return CohereV2ChatConfig()

    def _build_complete_streaming_response(
        self,
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        cohere_model_response_iterator = CohereModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )
        litellm_custom_stream_wrapper = CustomStreamWrapper(
            completion_stream=cohere_model_response_iterator,
            model=model,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="cohere",
        )
        all_openai_chunks = []
        for _chunk_str in all_chunks:
            try:
                generic_chunk = (
                    cohere_model_response_iterator.convert_str_chunk_to_generic_chunk(
                        chunk=_chunk_str
                    )
                )
                litellm_chunk = litellm_custom_stream_wrapper.chunk_creator(
                    chunk=generic_chunk
                )
                if litellm_chunk is not None:
                    all_openai_chunks.append(litellm_chunk)
            except (StopIteration, StopAsyncIteration):
                break
        complete_streaming_response = stream_chunk_builder(chunks=all_openai_chunks)
        return complete_streaming_response
