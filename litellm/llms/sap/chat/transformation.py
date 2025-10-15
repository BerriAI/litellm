"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration  `/completion`
"""

from typing import List, Optional, Union, Dict, Any
from dataclasses import asdict


from litellm.types.utils import ModelResponse, Usage
from litellm.utils import CustomStreamWrapper
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from .handler import OptionalDependencyError, GenAIHubOrchestrationError

try:
    from gen_ai_hub.orchestration.models.config import OrchestrationConfig
    from gen_ai_hub.orchestration.models.message import (
        Message,
        ToolMessage,
        MessageToolCall
    )
    from gen_ai_hub.orchestration.models.multimodal_items import (
        TextPart,
        ImagePart,
        ImageUrl,
    )
    from gen_ai_hub.orchestration.models.llm import LLM
    from gen_ai_hub.orchestration.models.template import Template
    from gen_ai_hub.orchestration.models.response_format import ResponseFormatJsonSchema
    from gen_ai_hub.orchestration.models.tools import FunctionTool
    from dacite import from_dict

    _gen_ai_hub_import_error = None
except ImportError as err:
    OrchestrationConfig = Any  # type: ignore
    Message = Any  # type: ignore
    LLM = Any  # type: ignore
    Template = Any  # type: ignore
    ResponseFormatJsonSchema = Any  # type: ignore
    FunctionTool = Any  # type: ignore
    TextPart = Any  # type: ignore
    ImagePart = Any  # type: ignore
    ImageUrl = Any  # type: ignore
    from_dict = Any  # type: ignore
    _gen_ai_hub_import_error = err


def modify_content_to_sdk_types(item: str | dict):
    if isinstance(item, dict) and item.get("type", None) == "image_url":
        return ImagePart(image_url=ImageUrl(url=item["image_url"].get("url")))
    elif isinstance(item, dict) and item.get("type") == "text":
        return TextPart(text=item["text"])
    elif isinstance(item, str):
        return TextPart(text=item)
    else:
        raise GenAIHubOrchestrationError(
            status_code=400, message=f"Unsupported content type: {item.get('type')}"
        )


def check_content(content):
    if isinstance(content, list):
        return [modify_content_to_sdk_types(item) for item in content]
    if isinstance(content, dict):
        return modify_content_to_sdk_types(content)
    if content is None:
        return ""
    return content


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
        self._ensure_gen_ai_hub_installed()

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _ensure_gen_ai_hub_installed(self) -> None:
        """Ensure the gen-ai-hub package is available."""
        if _gen_ai_hub_import_error is not None:
            raise OptionalDependencyError(
                "The gen-ai-hub package is required for this functionality. "
                "Please install it with: pip install sap-ai-sdk-gen[all]"
            ) from _gen_ai_hub_import_error

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
        ]

    def _transform_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        optional_params: dict,
        litellm_params: dict | None = None,
    ) -> OrchestrationConfig:
        supported_params = self.get_supported_openai_params(model)
        model_params = {
            k: v for k, v in optional_params.items() if k in supported_params
        }
        messages_ = []
        for message in messages:
            if message.get("role") == "tool":
                content = check_content(message.get("content"))
                tool_call_id = message.get("tool_call_id")
                messages_.append(
                    ToolMessage(tool_call_id=tool_call_id, content=content)
                )

            elif message.get("role") == "assistant" and message.get("tool_calls"):
                content = check_content(message.get("content"))
                tool_calls_list = message.get("tool_calls", []) # type: ignore
                if tool_calls_list:
                    tool_calls = []
                    for tool_call in tool_calls_list:
                        tool_calls.append(
                            from_dict(data_class=MessageToolCall, data=tool_call) # type: ignore
                        )
                    messages_.append(
                        Message(
                            role="assistant", content=content, tool_calls=tool_calls
                        )
                    )
            elif message.get("role") == "assistant":
                messages_.append(
                    Message(
                        role=message["role"],
                        content=check_content(message.get("content")),
                    )
                )
            else:
                message_ = {**message, "content": check_content(message.get("content"))}
                messages_.append(Message(**message_))
        model_version = optional_params.pop("model_version", "latest")
        tools_input = optional_params.pop("tools", None)
        tools = []
        if tools_input is not None:
            for tool in tools_input:
                tools.append(FunctionTool(**tool["function"]))
        response_format = optional_params.pop("response_format", None)
        if isinstance(response_format, dict):
            if (
                response_format.get("type", None) == "json_schema"
                and "json_schema" in response_format
            ):
                schema = response_format["json_schema"]
                if not schema.get("description", None):
                    schema["description"] = ""
                response_format = ResponseFormatJsonSchema(**schema)
            else:
                response_format = response_format["type"]
        config = OrchestrationConfig(
            template=Template(
                messages=messages_, response_format=response_format, tools=tools
            ),
            llm=LLM(name=model, parameters=model_params, version=model_version),
        )

        return config

    def _transform_response(
        self,
        model: str,
        response,
        model_response: ModelResponse,
        config: OrchestrationConfig,
        logging_obj: Optional[LiteLLMLoggingObject],
        optional_params: dict,
        print_verbose,
        encoding,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        llm_response = response.module_results.llm
        obj = model_response.model_dump()  # response.module_results.llm
        obj["choices"] = [asdict(c) for c in llm_response.choices]
        for c in obj["choices"]:
            c["message"]["role"] = (
                c["message"]["role"].value
                if hasattr(c["message"]["role"], "value")
                else c["message"]["role"]
            )
        obj["created"] = llm_response.created
        obj["model"] = llm_response.model
        obj["usage"] = Usage(**asdict(llm_response.usage))
        return ModelResponse.model_validate(obj)
