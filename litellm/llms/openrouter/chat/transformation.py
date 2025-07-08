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
from litellm.types.llms.openai import ChatCompletionThinkingBlock
from litellm.types.llms.openrouter import OpenRouterErrorMessage
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, Usage
from litellm.utils import StreamingChoices

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

            finish_reason = None
            delta = Delta()
            logprobs = None
            # {"id":"gen-1751968953-WEWmZEtpfjrIpGHVkLxh","provider":"Google",
            # "model":"anthropic/claude-sonnet-4","object":"chat.completion.chunk",
            # "created":1751968953,"choices":[{"index":0,
            # "delta":{"role":"assistant","content":"","reasoning":null,"reasoning_details":
            # [{"type":"reasoning.text","signature":"VkQnK3sSOCidzuDxlxmd003R3QYAQ==","provider":"google-vertex"}]}
            # ,"finish_reason":null,"native_finish_reason":null,"logprobs":null}]}

            if "choices" in chunk and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                if "delta" in choice and choice["delta"] is not None:
                    choice["delta"]["reasoning_content"] = choice["delta"].get(
                        "reasoning"
                    )

                    # Process reasoning_details safely
                    print(f"Choice: {choice}")
                    reasoning_details = choice["delta"].get("reasoning_details")
                    if reasoning_details and isinstance(reasoning_details, list) and len(reasoning_details) > 0:
                        first_thinking_block = reasoning_details[0]
                        if (isinstance(first_thinking_block, dict) and 
                            first_thinking_block.get("type") == "reasoning.text" and
                            "signature" in first_thinking_block):
                            # Initialize thinking_blocks if it doesn't exist
                            if "thinking_blocks" not in choice["delta"]:
                                choice["delta"]["thinking_blocks"] = []
                            choice["delta"]["thinking_blocks"].append(
                                ChatCompletionThinkingBlock(
                                    type="thinking",
                                    thinking="",
                                    signature=first_thinking_block["signature"],
                                )
                            )

                    delta = Delta(**choice["delta"])

                if "finish_reason" in choice:
                    finish_reason = choice["finish_reason"]

                if "logprobs" in choice:
                    logprobs = choice["logprobs"]

            new_choices = [
                StreamingChoices(
                    finish_reason=finish_reason,
                    delta=delta,
                    logprobs=logprobs,
                    index=0,
                )
            ]

            model_response = ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                model=chunk["model"],
                choices=new_choices,
            )

            if "usage" in chunk and chunk["usage"] is not None:
                usage_from_chunk = chunk["usage"]
                try:
                    final_usage = Usage(**usage_from_chunk)
                except Exception:
                    final_usage = Usage.parse_obj(usage_from_chunk)
                model_response.usage = final_usage
            return model_response
        except KeyError as e:
            raise OpenRouterException(
                message=f"KeyError: {e}, Got unexpected response from OpenRouter: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
