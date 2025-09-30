"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration  `/completion`
"""

from typing import List, Optional, Union, Dict, Any
from dataclasses import asdict



from litellm.types.utils import ModelResponse, Usage
from litellm.utils import CustomStreamWrapper
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from .handler import OptionalDependencyError
try:
    from gen_ai_hub.orchestration.models.config import OrchestrationConfig
    from gen_ai_hub.orchestration.models.message import Message, ToolMessage
    from gen_ai_hub.orchestration.models.llm import LLM
    from gen_ai_hub.orchestration.models.template import Template
    from gen_ai_hub.orchestration.models.response_format import (
        ResponseFormatJsonSchema,
        ResponseFormatJsonObject,
        ResponseFormatText,
    )
    from gen_ai_hub.orchestration.models.tools import FunctionTool
    _gen_ai_hub_import_error = None
except ImportError as err:
    OrchestrationConfig = Any  # type: ignore
    Message = Any  # type: ignore
    LLM = Any  # type: ignore
    Template = Any  # type: ignore
    ResponseFormatJsonSchema = Any  # type: ignore
    FunctionTool = Any  # type: ignore
    _gen_ai_hub_import_error = err


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
                "Please install it with: pip install gen-ai-hub"
            ) from _gen_ai_hub_import_error

    # def _should_fake_stream(self, optional_params: dict) -> bool:
    #     """
    #     Groq doesn't support 'response_format' while streaming
    #     """
    #     if optional_params.get("response_format") is not None:
    #         return True
    #
    #     return False

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
        litellm_params: dict,
    ) -> OrchestrationConfig:
        supported_params = self.get_supported_openai_params(model)
        model_params = {
            k: v for k, v in optional_params.items() if k in supported_params
        }

        filtered_messages = []
        for msg in messages:
            msg_copy = msg.copy()

            if msg_copy.get("role") == "tool":
                content = msg_copy.get("content", "")
                name = msg_copy.get("name", "function")

                filtered_messages.append({
                    "role": "user",
                    "content": f"Function {name} returned: {content}"
                })
                continue

            msg_copy.pop("tool_call_id", None)
            msg_copy.pop("name", None)

            if "tool_calls" in msg_copy and msg_copy["tool_calls"]:
                if isinstance(msg_copy["tool_calls"], list) and len(msg_copy["tool_calls"]) > 0:
                    if isinstance(msg_copy["tool_calls"][0], dict):
                        msg_copy.pop("tool_calls", None)
                        if not msg_copy.get("content"):
                            msg_copy["content"] = ""

            filtered_messages.append(msg_copy)
        messages_ = []
        for msg in filtered_messages:
            if isinstance(msg, dict):
                messages_.append(Message(**msg))
            else:
                messages_.append(msg)

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


# def remove_keys(dictionary, keys_to_remove):
#     if isinstance(dictionary, dict):
#         return {
#             key: remove_keys(value, keys_to_remove)
#             for key, value in dictionary.items()
#             if key not in keys_to_remove
#         }
#     elif isinstance(dictionary, list):
#         return [remove_keys(item, keys_to_remove) for item in dictionary]
#     else:
#         return dictionary
