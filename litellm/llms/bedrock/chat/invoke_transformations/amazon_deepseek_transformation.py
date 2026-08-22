from typing import Any, Final, cast

from httpx import Response

from litellm import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Extract the reasoning content, and return it as a separate field in the response.
        """
        response: Final = super().transform_response(
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
        prompt: Final = cast(str | None, request_data.get("prompt"))
        message_content: Final = cast(str | None, cast(Choices, response.choices[0]).message.get("content"))
        if prompt and prompt.strip().endswith("<think>") and message_content:
            message_content_with_reasoning_token: Final = "<think>" + message_content
            reasoning, content = _parse_content_for_reasoning(message_content_with_reasoning_token)
            provider_specific_fields: Final = cast(Choices, response.choices[0]).message.provider_specific_fields or {}
            if reasoning:
                provider_specific_fields["reasoning_content"] = reasoning

            message: Final = Message(
                **{
                    **cast(Choices, response.choices[0]).message.model_dump(),
                    "content": content,
                    "provider_specific_fields": provider_specific_fields,
                }
            )
            cast(Choices, response.choices[0]).message = message
        return response


_END_OF_THINKING: Final = "</think>"


class AmazonDeepseekR1ResponseIterator(BaseModelResponseIterator):
    def __init__(self, streaming_response: Any, sync_stream: bool) -> None:
        super().__init__(streaming_response=streaming_response, sync_stream=sync_stream)
        self.has_finished_thinking = False
        self.held_back = ""

    def _split_on_end_of_thinking(self, generated_content: str, is_last_chunk: bool) -> tuple[str, str]:
        """Split a chunk of the thinking phase into (reasoning, content).

        ``</think>`` is not guaranteed to arrive as a chunk of its own: it can be glued to the
        text on either side, or split across chunks. Matching the whole marker against one chunk
        misses both, leaving every later chunk filed as reasoning and ``content`` empty for the
        entire turn. Text that could still be the start of the marker is held back until the next
        chunk decides it, and released if the stream ends first.
        """
        buffered: Final = self.held_back + generated_content
        reasoning, marker, content = buffered.partition(_END_OF_THINKING)
        if marker:
            verbose_logger.debug("Deepseek r1: </think> received, setting has_finished_thinking to True")
            self.has_finished_thinking = True
            self.held_back = ""
            return reasoning, content
        if is_last_chunk:
            self.held_back = ""
            return buffered, ""
        partial: Final = next(
            (
                length
                for length in range(min(len(buffered), len(_END_OF_THINKING) - 1), 0, -1)
                if buffered.endswith(_END_OF_THINKING[:length])
            ),
            0,
        )
        self.held_back = buffered[len(buffered) - partial :] if partial else ""
        return buffered[: len(buffered) - partial] if partial else buffered, ""

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Deepseek r1 starts by thinking, then it generates the response.
        """
        try:
            typed_chunk: Final = AmazonDeepSeekR1StreamingResponse(**chunk)
            generated_content = typed_chunk["generation"]
            reasoning_delta: str = ""
            if not self.has_finished_thinking:
                reasoning_delta, generated_content = self._split_on_end_of_thinking(
                    generated_content, is_last_chunk=typed_chunk["stop_reason"] is not None
                )

            prompt_token_count: Final = typed_chunk.get("prompt_token_count") or 0
            generation_token_count: Final = typed_chunk.get("generation_token_count") or 0
            usage: Final = ChatCompletionUsageBlock(
                prompt_tokens=prompt_token_count,
                completion_tokens=generation_token_count,
                total_tokens=prompt_token_count + generation_token_count,
            )

            return ModelResponseStream(
                choices=[
                    StreamingChoices(
                        finish_reason=typed_chunk["stop_reason"],
                        delta=Delta(
                            content=generated_content if self.has_finished_thinking else None,
                            reasoning_content=reasoning_delta or None,
                        ),
                    )
                ],
                usage=usage,
            )

        except Exception as e:
            raise e
