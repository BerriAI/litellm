"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as OpenRouter is openai-compatible.

Docs: https://openrouter.ai/docs/parameters
"""

from typing import Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.openrouter import OpenRouterErrorMessage
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import OpenRouterException


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
        stream_options = non_default_params.pop("stream_options", {})
        if transforms is not None:
            extra_body["transforms"] = transforms
        if models is not None:
            extra_body["models"] = models
        if route is not None:
            extra_body["route"] = route
        if stream_options is not None and stream_options.get("include_usage"):
            extra_body["usage"] = {"include": True}

        mapped_openai_params[
            "extra_body"
        ] = extra_body  # openai client supports `extra_body` param
        return mapped_openai_params

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
        print(f"OpenRouter Raw Chunk: {chunk}")
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

            # Get usage information from the chunk
            usage_from_chunk = chunk.get("usage")
            final_usage = None
            if usage_from_chunk:
                # Extract all usage fields from OpenRouter response
                usage_kwargs = {
                    "prompt_tokens": usage_from_chunk.get("prompt_tokens", 0),
                    "completion_tokens": usage_from_chunk.get("completion_tokens", 0),
                    "total_tokens": usage_from_chunk.get("total_tokens", 0),
                    "cost": usage_from_chunk.get("cost"),
                    "is_byok": usage_from_chunk.get("is_byok"),
                }
                
                # Handle prompt_tokens_details if present
                if "prompt_tokens_details" in usage_from_chunk:
                    usage_kwargs["prompt_tokens_details"] = usage_from_chunk["prompt_tokens_details"]
                
                # Handle completion_tokens_details if present
                if "completion_tokens_details" in usage_from_chunk:
                    usage_kwargs["completion_tokens_details"] = usage_from_chunk["completion_tokens_details"]
                
                # Pass any additional fields that might be present in usage_from_chunk
                # This handles fields like cost_details that might be present
                for key, value in usage_from_chunk.items():
                    if key not in usage_kwargs and value is not None:
                        usage_kwargs[key] = value
                
                final_usage = Usage(**usage_kwargs)

            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                usage=final_usage,
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
