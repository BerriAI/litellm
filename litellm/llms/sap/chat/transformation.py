"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration Service`v2/completion`
"""
import json
from typing import List, Optional, Union, Dict

from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

from .models import (
    SAPMessage,
    SAPAssistantMessage,
    SAPToolChatMessage,
    ChatCompletionTool,
    ResponseFormatJSONSchema,
    ResponseFormat,
    SAPUserMessage,
)

def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump(by_alias=True)


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
            "timeout",
        ]

    def _transform_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: dict,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        model_params = {
            k: v for k, v in optional_params.items() if k in supported_params
        }
        model_version = optional_params.pop("model_version", "latest")
        template = []
        for message in messages:
            if message["role"] == "user":
                template.append(validate_dict(message, SAPUserMessage))
            elif message["role"] == "assistant":
                template.append(validate_dict(message, SAPAssistantMessage))
            elif message["role"] == "tool":
                template.append(validate_dict(message, SAPToolChatMessage))
            else:
                template.append(validate_dict(message, SAPMessage))

        tools_ = optional_params.pop("tools", [])
        tools = [validate_dict(tool, ChatCompletionTool) for tool in tools_]

        response_format_ = model_params.pop("response_format", {"type": "text"})
        if response_format_["type"] == "json_schema":
            response_format = validate_dict(response_format_, ResponseFormatJSONSchema)
        else:
            response_format = validate_dict(response_format_, ResponseFormat)

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
            "config": {
                "modules": {
                    "prompt_templating": {
                        "prompt": {
                            "template": template,
                            "tools": tools,
                            "response_format": response_format,
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
        if (
            model.startswith('anthropic')
            or model.startswith("amazon")
            or model.startswith("cohere")
            or model.startswith("alephalpha")
            or model == "gpt-4"
        ):
            config["config"]["modules"]["prompt_templating"]["prompt"].pop("response_format")
            config["config"]["modules"]["prompt_templating"]["prompt"]["template"].append(
                {"role":"system",
                 'content':json.dumps(response_format, ensure_ascii=False)})

        return config

    def _transform_response(self, response) -> ModelResponse:
        return ModelResponse.model_validate(response.json()["final_result"])
