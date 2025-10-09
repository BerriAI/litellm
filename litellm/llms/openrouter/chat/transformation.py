"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as OpenRouter is openai-compatible.

Docs: https://openrouter.ai/docs/parameters
"""

from enum import Enum
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union

import httpx

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
        
        When cache_control is at message level, it's added to ALL content blocks
        to cache the entire message content.
        """
        transformed_messages = []
        for message in messages:
            message_copy = dict(message)
            cache_control = message_copy.pop("cache_control", None)
            
            if cache_control is not None:
                content = message_copy.get("content")
                
                if isinstance(content, list):
                    # Content is already a list, add cache_control to all blocks
                    if len(content) > 0:
                        content_copy = []
                        for block in content:
                            block_copy = dict(block)
                            block_copy["cache_control"] = cache_control
                            content_copy.append(block_copy)
                        message_copy["content"] = content_copy
                else:
                    # Content is a string, convert to structured format
                    message_copy["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
                    ]
            
            transformed_messages.append(message_copy)
        
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
        return response

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
