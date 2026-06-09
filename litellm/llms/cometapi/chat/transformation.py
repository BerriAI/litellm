"""
Support for CometAPI's `/v1/chat/completions` endpoint.

Based on OpenAI-compatible API interface implementation
Documentation: [CometAPI Documentation Link]
"""

from typing import Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import CometAPIException, get_cometapi_complete_url


class CometAPIConfig(OpenAIGPTConfig):
    """
    CometAPI configuration class, inherits from OpenAIGPTConfig

    Since CometAPI is OpenAI-compatible API, we inherit from OpenAIGPTConfig
    and only need to override necessary methods to handle CometAPI-specific features
    """

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        extra_body = optional_params.pop("extra_body", {}) or {}
        response = super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
        response.update(extra_body)
        return response

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the CometAPI call.

        Returns:
            str: The complete URL for the API call.
        """
        return get_cometapi_complete_url(api_base, "chat/completions", api_key=api_key)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Return CometAPI-specific error class
        """
        return CometAPIException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Get model response iterator for streaming responses
        """
        return CometAPIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class CometAPIChatCompletionStreamingHandler(BaseModelResponseIterator):
    """
    Handler for CometAPI streaming chat completion responses
    """

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse individual chunks from streaming response
        """
        try:
            # Handle error in chunk
            if "error" in chunk:
                error_chunk = chunk["error"]
                error_message = "CometAPI Error: {}".format(
                    error_chunk.get("message", "Unknown error")
                )
                raise CometAPIException(
                    message=error_message,
                    status_code=error_chunk.get("code", 400),
                    headers={"Content-Type": "application/json"},
                )

            # Process choices
            new_choices = []
            for choice in chunk["choices"]:
                # Handle reasoning content if present
                if "delta" in choice and "reasoning" in choice["delta"]:
                    choice["delta"]["reasoning_content"] = choice["delta"].get(
                        "reasoning"
                    )
                new_choices.append(choice)

            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                usage=chunk.get("usage"),
                model=chunk["model"],
                choices=new_choices,
            )
        except KeyError as e:
            raise CometAPIException(
                message=f"KeyError: {e}, Got unexpected response from CometAPI: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
