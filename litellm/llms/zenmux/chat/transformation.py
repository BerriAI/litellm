

from enum import Enum
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union, cast

import httpx
import litellm

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.llms.zenmux import ZenMuxErrorMessage
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import ZenMuxException




class ZenMuxConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Allow reasoning parameters for models flagged as reasoning-capable.
        """
        supported_params = [
            "max_completion_tokens",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "response_format",
            "stop",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream",
            "stream_options",
            "extra_headers",
            "max_tokens",

            "web_search_options",
            "reasoning_effort",
            "provider",
            "model_routing_config",
        ]
        return list(dict.fromkeys(supported_params))

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # when user passing max_tokens, change it to max_completion_tokens
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_completion_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params



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

        # ALWAYS add usage parameter to get cost data from ZenMux
        # This ensures cost tracking works for all ZenMux models
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
        Transform the response from ZenMux API.
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

        # Extract cost from ZenMux response body
        # ZenMux returns cost information in the usage object when usage.include=true
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
        return ZenMuxException(
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
        return ZenMuxChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class ZenMuxChatCompletionStreamingHandler(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            ## HANDLE ERROR IN CHUNK ##
            if "error" in chunk:
                error_chunk = chunk["error"]
                error_message = ZenMuxErrorMessage(
                    message="Message: {}, Metadata: {}, User ID: {}".format(
                        error_chunk["message"],
                        error_chunk.get("metadata", {}),
                        error_chunk.get("user_id", ""),
                    ),
                    code=error_chunk["code"],
                    metadata=error_chunk.get("metadata", {}),
                )
                raise ZenMuxException(
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
            raise ZenMuxException(
                message=f"KeyError: {e}, Got unexpected response from ZenMux: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
