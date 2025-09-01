from typing import Any, List, Optional, cast

from httpx import Response

from litellm import verbose_logger
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _parse_content_for_reasoning,
)
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    LiteLLMLoggingObj,
)
from litellm.types.llms.bedrock import AmazonDeepSeekR1StreamingResponse
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionUsageBlock,
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from .amazon_llama_transformation import AmazonLlamaConfig


class AmazonDeepSeekR1Config(AmazonLlamaConfig):
    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Extract the reasoning content, and return it as a separate field in the response.
        """
        response = super().transform_response(
            model,
            raw_response,
            model_response,
            logging_obj,
            request_data,
            messages,
            optional_params,
            litellm_params,
            encoding,
            api_key,
            json_mode,
        )
        prompt = cast(Optional[str], request_data.get("prompt"))
        message_content = cast(
            Optional[str], cast(Choices, response.choices[0]).message.get("content")
        )
        if prompt and prompt.strip().endswith("<think>") and message_content:
            message_content_with_reasoning_token = "<think>" + message_content
            reasoning, content = _parse_content_for_reasoning(
                message_content_with_reasoning_token
            )
            provider_specific_fields = (
                cast(Choices, response.choices[0]).message.provider_specific_fields
                or {}
            )
            if reasoning:
                provider_specific_fields["reasoning_content"] = reasoning

            message = Message(
                **{
                    **cast(Choices, response.choices[0]).message.model_dump(),
                    "content": content,
                    "provider_specific_fields": provider_specific_fields,
                }
            )
            cast(Choices, response.choices[0]).message = message
        return response


class AmazonDeepseekR1ResponseIterator(BaseModelResponseIterator):
    def __init__(self, streaming_response: Any, sync_stream: bool) -> None:
        super().__init__(streaming_response=streaming_response, sync_stream=sync_stream)
        self.has_finished_thinking = False

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Deepseek r1 starts by thinking, then it generates the response.
        """
        try:
            typed_chunk = AmazonDeepSeekR1StreamingResponse(**chunk)  # type: ignore
            generated_content = typed_chunk["generation"]
            if generated_content == "</think>" and not self.has_finished_thinking:
                verbose_logger.debug(
                    "Deepseek r1: </think> received, setting has_finished_thinking to True"
                )
                generated_content = ""
                self.has_finished_thinking = True

            prompt_token_count = typed_chunk.get("prompt_token_count") or 0
            generation_token_count = typed_chunk.get("generation_token_count") or 0
            usage = ChatCompletionUsageBlock(
                prompt_tokens=prompt_token_count,
                completion_tokens=generation_token_count,
                total_tokens=prompt_token_count + generation_token_count,
            )

            return ModelResponseStream(
                choices=[
                    StreamingChoices(
                        finish_reason=typed_chunk["stop_reason"],
                        delta=Delta(
                            content=(
                                generated_content
                                if self.has_finished_thinking
                                else None
                            ),
                            reasoning_content=(
                                generated_content
                                if not self.has_finished_thinking
                                else None
                            ),
                        ),
                    )
                ],
                usage=usage,
            )

        except Exception as e:
            raise e
