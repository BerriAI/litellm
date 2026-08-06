"""
Support for OVHCloud AI Endpoints `/v1/chat/completions` endpoint.

Our unified API follows the OpenAI standard.
More information on our website: https://endpoints.ai.cloud.ovh.net
"""

from typing import Final

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.ovhcloud.utils import OVHCloudException
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import ModelResponseStream


class OVHCloudChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "ovhcloud"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1" if api_base is None else api_base.rstrip("/")
        complete_url: Final = f"{api_base}/chat/completions"
        return complete_url

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return OVHCloudException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        return mapped_openai_params

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        extra_body: Final = optional_params.pop("extra_body", {})
        response: Final = super().transform_request(model, messages, optional_params, litellm_params, headers)
        response.update(extra_body)
        return response


class OVHCloudChatCompletionStreamingHandler(BaseModelResponseIterator):
    """
    Handler for OVHCloud AI Endpoints streaming chat completion responses
    """

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse individual chunks from streaming response
        """
        try:
            if "error" in chunk:
                error_chunk: Final = chunk["error"]
                error_message: Final = "OVHCloud Error: {}".format(error_chunk.get("message", "Unknown error"))
                raise OVHCloudException(
                    message=error_message,
                    status_code=error_chunk.get("code", 400),
                    headers={"Content-Type": "application/json"},
                )

            new_choices: Final = []
            for choice in chunk["choices"]:
                if "delta" in choice:
                    delta = choice["delta"]
                    # OVHCloud field migration (deadline: 2026-05-11):
                    # `reasoning_content` is replaced by `reasoning`.
                    # Normalise to `reasoning_content` so downstream consumers
                    # see a consistent key during the transition window.
                    reasoning_new = delta.get("reasoning")
                    reasoning_legacy = delta.get("reasoning_content")
                    if reasoning_new is not None and reasoning_legacy is None:
                        delta["reasoning_content"] = reasoning_new
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
            raise OVHCloudException(
                message=f"KeyError: {e}, Got unexpected response from CometAPI: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
