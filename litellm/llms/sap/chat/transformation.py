"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration Service`/completion`
"""

from typing import List, Optional, Union, Dict

from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class GenAIHubOrchestrationConfig(OpenAIGPTConfig):
    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None  #
    model_version: str = "latest"

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model):
        return [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "prediction",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "extra_headers",
            "parallel_tool_calls",
            "response_format",
            "timeout"
        ]

    def _transform_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: dict,
        litellm_params: Optional[dict] = None,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        model_params = {
            k: v for k, v in optional_params.items() if k in supported_params
        }
        model_version = optional_params.pop("model_version", "latest")
        template = messages

        tools = optional_params.pop("tools", None)
        response_format = model_params.pop("response_format", None)
        stream = model_params.pop("stream", False)
        stream_config = {}
        if stream or "stream_options" in model_params:
            stream_config["enabled"] = True
            stream_options = model_params.pop("stream_options", {})
            stream_config["chunk_size"] = stream_options.get("chunk_size", 100)
            if "delimiters" in stream_options:
                stream_config["delimiters"] = stream_options.get("delimiters")
        else:
            stream_config["enabled"] = False
        config = {
            'config':
                {
                    "modules": {
                        "prompt_templating": {
                            "prompt": {
                                "template": template,
                                "tools": tools if tools else [],
                                "response_format": response_format if response_format else {"type": "text"},
                            },
                            "model": {
                                "name": model,
                                "params": model_params,
                                "version": model_version,
                            },

                        },
                    },
                    "stream": stream_config,
                }
        }


        return config

    def _transform_response(
        self,
        model: str,
        response,
        model_response: ModelResponse,
        # config: OrchestrationConfig,
        logging_obj: Optional[LiteLLMLoggingObject],
        optional_params: dict,
        print_verbose,
        encoding,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        return ModelResponse.model_validate(response.json()["final_result"])
