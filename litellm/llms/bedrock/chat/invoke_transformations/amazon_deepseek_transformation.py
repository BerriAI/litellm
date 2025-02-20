from typing import Any, List, Optional, cast

from httpx import Response

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _parse_content_for_reasoning,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse

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
