"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as OpenRouter is openai-compatible.

Docs: https://openrouter.ai/docs/parameters
"""

from enum import Enum
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union, cast

import httpx
import litellm

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.llms.openrouter import OpenRouterErrorMessage
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import OpenRouterException


class CacheControlSupportedModels(str, Enum):
    """Models that support cache_control in content blocks."""
    CLAUDE = "claude"
    GEMINI = "gemini"


class OpenrouterConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Allow reasoning parameters for models flagged as reasoning-capable.
        """
        supported_params = super().get_supported_openai_params(model=model)
        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider="openrouter"
            ) or litellm.supports_reasoning(model=model):
                supported_params.append("reasoning_effort")
        except Exception:
            pass
        return list(dict.fromkeys(supported_params))

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # OpenRouter-only parameters
        extra_body = {}
        transforms = non_default_params.pop("transforms", None)
        models = non_default_params.pop("models", None)
        route = non_default_params.pop("route", None)
        if transforms is not None:
            extra_body["transforms"] = transforms
        if models is not None:
            extra_body["models"] = models
        if route is not None:
            extra_body["route"] = route
        mapped_openai_params["extra_body"] = (
            extra_body  # openai client supports `extra_body` param
        )
        return mapped_openai_params

    def _supports_cache_control_in_content(self, model: str) -> bool:
        """
        Check if the model supports cache_control in content blocks.
        
        Returns:
            bool: True if model supports cache_control (Claude or Gemini models)
        """
        model_lower = model.lower()
        return any(
            supported_model.value in model_lower
            for supported_model in CacheControlSupportedModels
        )

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List["ChatCompletionToolParam"]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List["ChatCompletionToolParam"]]]:
        if self._supports_cache_control_in_content(model):
            return messages, tools
        else:
            return super().remove_cache_control_flag_from_messages_and_tools(
                model, messages, tools
            )

    def _move_cache_control_to_content(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        Move cache_control from message level to content blocks.
        OpenRouter requires cache_control to be inside content blocks, not at message level.
        
        To avoid exceeding Anthropic's limit of 4 cache breakpoints, cache_control is only
        added to the LAST content block in each message.
        """
        transformed_messages: List[AllMessageValues] = []
        for message in messages:
            message_dict = dict(message)
            cache_control = message_dict.pop("cache_control", None)
            
            if cache_control is not None:
                content = message_dict.get("content")
                
                if isinstance(content, list):
                    # Content is already a list, add cache_control only to the last block
                    if len(content) > 0:
                        content_copy = []
                        for i, block in enumerate(content):
                            block_dict = dict(block)
                            # Only add cache_control to the last content block
                            if i == len(content) - 1:
                                block_dict["cache_control"] = cache_control
                            content_copy.append(block_dict)
                        message_dict["content"] = content_copy
                else:
                    # Content is a string, convert to structured format
                    message_dict["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
                    ]
            
            # Cast back to AllMessageValues after modification
            transformed_messages.append(cast(AllMessageValues, message_dict))
        
        return transformed_messages

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
        if self._supports_cache_control_in_content(model):
            messages = self._move_cache_control_to_content(messages)
        
        extra_body = optional_params.pop("extra_body", {})
        response = super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
        response.update(extra_body)

        # ALWAYS add usage parameter to get cost data from OpenRouter
        # This ensures cost tracking works for all OpenRouter models
        if "usage" not in response:
            response["usage"] = {"include": True}

        return response

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the response from OpenRouter API.

        Extracts cost information from response headers if available.

        Returns:
            ModelResponse: The transformed response with cost information.
        """
        # Call parent transform_response to get the standard ModelResponse
        model_response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        # Extract cost from OpenRouter response body
        # OpenRouter returns cost information in the usage object when usage.include=true
        try:
            response_json = raw_response.json()
            if "usage" in response_json and response_json["usage"]:
                response_cost = response_json["usage"].get("cost")
                if response_cost is not None:
                    # Store cost in hidden params for the cost calculator to use
                    if not hasattr(model_response, "_hidden_params"):
                        model_response._hidden_params = {}
                    if "additional_headers" not in model_response._hidden_params:
                        model_response._hidden_params["additional_headers"] = {}
                    model_response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(response_cost)
        except Exception:
            # If we can't extract cost, continue without it - don't fail the response
            pass

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
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
        return OpenRouterChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OpenRouterChatCompletionStreamingHandler(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            ## HANDLE ERROR IN CHUNK ##
            if "error" in chunk:
                error_chunk = chunk["error"]
                error_message = OpenRouterErrorMessage(
                    message="Message: {}, Metadata: {}, User ID: {}".format(
                        error_chunk["message"],
                        error_chunk.get("metadata", {}),
                        error_chunk.get("user_id", ""),
                    ),
                    code=error_chunk["code"],
                    metadata=error_chunk.get("metadata", {}),
                )
                raise OpenRouterException(
                    message=error_message["message"],
                    status_code=error_message["code"],
                    headers=error_message["metadata"].get("headers", {}),
                )

            new_choices = []
            for choice in chunk["choices"]:
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
            raise OpenRouterException(
                message=f"KeyError: {e}, Got unexpected response from OpenRouter: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
