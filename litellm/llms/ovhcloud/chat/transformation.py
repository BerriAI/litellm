"""
Support for OVHCloud AI Endpoints `/v1/chat/completions` endpoint.

Our unified API follows the OpenAI standard.
More information on our website: https://endpoints.ai.cloud.ovh.net
"""
from typing import Optional, Union, List

import httpx
from litellm import ModelResponseStream, OpenAIGPTConfig, get_model_info, verbose_logger
from litellm.llms.ovhcloud.utils import OVHCloudException
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues

class OVHCloudChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "ovhcloud"

    def get_supported_openai_params(self, model: str) -> list:
        """
        Details about function calling support can be found here:
        https://help.ovhcloud.com/csm/en-gb-public-cloud-ai-endpoints-function-calling?id=kb_article_view&sysparm_article=KB0071907
        """
        supports_function_calling: Optional[bool] = None
        try:
            model_info = get_model_info(model, custom_llm_provider="ovhcloud")
            supports_function_calling = model_info.get(
                "supports_function_calling", False
            )
        except Exception as e:
            verbose_logger.debug(f"Error getting supported OpenAI params: {e}")
            pass

        optional_params = super().get_supported_openai_params(model)
        if supports_function_calling is not True:
            verbose_logger.debug(
                "You can see our models supporting function_calling in our catalog: https://endpoints.ai.cloud.ovh.net/catalog "
            )
            optional_params.remove("tools")
            optional_params.remove("tool_choice")
            optional_params.remove("function_call")
            optional_params.remove("response_format")
        return optional_params
    
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1" if api_base is None else api_base.rstrip("/")
        complete_url = f"{api_base}/chat/completions"
        return complete_url
    
    def get_error_class(
        self, 
        error_message: str, 
        status_code: int, 
        headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
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
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        return mapped_openai_params
    
    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        extra_body = optional_params.pop("extra_body", {})
        response = super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
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
                error_chunk = chunk["error"]
                error_message = "OVHCloud Error: {}".format(
                    error_chunk.get("message", "Unknown error")
                )
                raise OVHCloudException(
                    message=error_message,
                    status_code=error_chunk.get("code", 400),
                    headers={"Content-Type": "application/json"},
                )

            new_choices = []
            for choice in chunk["choices"]:
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
            raise OVHCloudException(
                message=f"KeyError: {e}, Got unexpected response from CometAPI: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e