"""
Support for CometAPI's `/v1/chat/completions` endpoint.

Based on OpenAI-compatible API interface implementation
Documentation: [CometAPI Documentation Link]
"""

from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import CometAPIException


class CometAPIConfig(OpenAIGPTConfig):
    """
    CometAPI configuration class, inherits from OpenAIGPTConfig
    
    Since CometAPI is OpenAI-compatible API, we inherit from OpenAIGPTConfig
    and only need to override necessary methods to handle CometAPI-specific features
    """
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI format parameters to CometAPI format
        """
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # CometAPI-specific parameters (if any)
        extra_body: dict[str, Any] = {}
        # TODO: Add CometAPI-specific parameter handling here
        # Example:
        # custom_param = non_default_params.pop("custom_param", None)
        # if custom_param is not None:
        #     extra_body["custom_param"] = custom_param
        
        if extra_body:
            mapped_openai_params["extra_body"] = extra_body
        
        return mapped_openai_params

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List["ChatCompletionToolParam"]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List["ChatCompletionToolParam"]]]:
        """
        Remove cache control flags from messages and tools if not supported
        """
        # For CometAPI, use default behavior (remove cache control)
        return super().remove_cache_control_flag_from_messages_and_tools(
            model, messages, tools
        )

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
        extra_body = optional_params.pop("extra_body", {})
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
        # Default base
        if api_base is None:
            api_base = "https://api.cometapi.com/v1"
        endpoint = "chat/completions"

        # Normalize
        api_base = api_base.rstrip("/")

        # If endpoint already present, return as-is
        if endpoint in api_base:
            return api_base

        # Ensure we include /v1 prefix when missing
        if api_base.endswith("/v1"):
            return f"{api_base}/{endpoint}"
        if api_base.endswith("/v1/"):
            return f"{api_base}{endpoint}"
        # If user provided https://api.cometapi.com, add /v1
        if api_base == "https://api.cometapi.com":
            return f"{api_base}/v1/{endpoint}"
        # Generic fallback: if '/v1' not in path, add it
        if "/v1" not in api_base.split("//", 1)[-1]:
            return f"{api_base}/v1/{endpoint}"
        return f"{api_base}/{endpoint}"

    def get_error_class(
        self, 
        error_message: str, 
        status_code: int, 
        headers: Union[dict, httpx.Headers]
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
                    choice["delta"]["reasoning_content"] = choice["delta"].get("reasoning")
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
