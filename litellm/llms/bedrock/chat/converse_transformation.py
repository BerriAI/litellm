"""
Translating between OpenAI's `/chat/completion` format and Amazon's `/converse` format
"""

import copy
import time
import types
from typing import List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.bedrock import *
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import CustomStreamWrapper, add_dummy_tool, has_tool_call_blocks

from ...prompt_templates.factory import _bedrock_converse_messages_pt, _bedrock_tools_pt
from ..common_utils import BedrockError, get_bedrock_tool_name


class AmazonConverseConfig:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    #2 - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
    """

    maxTokens: Optional[int]
    stopSequences: Optional[List[str]]
    temperature: Optional[int]
    topP: Optional[int]

    def __init__(
        self,
        maxTokens: Optional[int] = None,
        stopSequences: Optional[List[str]] = None,
        temperature: Optional[int] = None,
        topP: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "stop",
            "temperature",
            "top_p",
            "extra_headers",
            "response_format",
        ]

        if (
            model.startswith("anthropic")
            or model.startswith("mistral")
            or model.startswith("cohere")
            or model.startswith("meta.llama3-1")
        ):
            supported_params.append("tools")

        if model.startswith("anthropic") or model.startswith("mistral"):
            # only anthropic and mistral support tool choice config. otherwise (E.g. cohere) will fail the call - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            supported_params.append("tool_choice")

        return supported_params

    def map_tool_choice_values(
        self, model: str, tool_choice: Union[str, dict], drop_params: bool
    ) -> Optional[ToolChoiceValuesBlock]:
        if tool_choice == "none":
            if litellm.drop_params is True or drop_params is True:
                return None
            else:
                raise litellm.utils.UnsupportedParamsError(
                    message="Bedrock doesn't support tool_choice={}. To drop it from the call, set `litellm.drop_params = True.".format(
                        tool_choice
                    ),
                    status_code=400,
                )
        elif tool_choice == "required":
            return ToolChoiceValuesBlock(any={})
        elif tool_choice == "auto":
            return ToolChoiceValuesBlock(auto={})
        elif isinstance(tool_choice, dict):
            # only supported for anthropic + mistral models - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            specific_tool = SpecificToolChoiceBlock(
                name=tool_choice.get("function", {}).get("name", "")
            )
            return ToolChoiceValuesBlock(tool=specific_tool)
        else:
            raise litellm.utils.UnsupportedParamsError(
                message="Bedrock doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.".format(
                    tool_choice
                ),
                status_code=400,
            )

    def get_supported_image_types(self) -> List[str]:
        return ["png", "jpeg", "gif", "webp"]

    def map_openai_params(
        self,
        model: str,
        non_default_params: dict,
        optional_params: dict,
        drop_params: bool,
        messages: Optional[List[AllMessageValues]] = None,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "response_format":
                json_schema: Optional[dict] = None
                schema_name: str = ""
                if "response_schema" in value:
                    json_schema = value["response_schema"]
                    schema_name = "json_tool_call"
                elif "json_schema" in value:
                    json_schema = value["json_schema"]["schema"]
                    schema_name = value["json_schema"]["name"]
                """
                Follow similar approach to anthropic - translate to a single tool call. 

                When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
                - You usually want to provide a single tool
                - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
                - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
                """
                if json_schema is not None:
                    _tool_choice = self.map_tool_choice_values(
                        model=model, tool_choice="required", drop_params=drop_params  # type: ignore
                    )

                    _tool = ChatCompletionToolParam(
                        type="function",
                        function=ChatCompletionToolParamFunctionChunk(
                            name=schema_name, parameters=json_schema
                        ),
                    )

                    optional_params["tools"] = [_tool]
                    optional_params["tool_choice"] = _tool_choice
                    optional_params["json_mode"] = True
                else:
                    if litellm.drop_params is True or drop_params is True:
                        pass
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="Bedrock doesn't support response_format={}. To drop it from the call, set `litellm.drop_params = True.".format(
                                value
                            ),
                            status_code=400,
                        )
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["maxTokens"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                if isinstance(value, str):
                    if len(value) == 0:  # converse raises error for empty strings
                        continue
                    value = [value]
                optional_params["stopSequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["topP"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "tool_choice":
                _tool_choice_value = self.map_tool_choice_values(
                    model=model, tool_choice=value, drop_params=drop_params  # type: ignore
                )
                if _tool_choice_value is not None:
                    optional_params["tool_choice"] = _tool_choice_value

        ## VALIDATE REQUEST
        """
        Bedrock doesn't support tool calling without `tools=` param specified.
        """
        if (
            "tools" not in non_default_params
            and messages is not None
            and has_tool_call_blocks(messages)
        ):
            if litellm.modify_params:
                optional_params["tools"] = add_dummy_tool(
                    custom_llm_provider="bedrock_converse"
                )
            else:
                raise litellm.UnsupportedParamsError(
                    message="Bedrock doesn't support tool calling without `tools=` param specified. Pass `tools=` param OR set `litellm.modify_params = True` // `litellm_settings::modify_params: True` to add dummy tool to the request.",
                    model="",
                    llm_provider="bedrock",
                )
        return optional_params

    def _transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
    ) -> RequestObject:
        system_prompt_indices = []
        system_content_blocks: List[SystemContentBlock] = []
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                _system_content_block: Optional[SystemContentBlock] = None
                if isinstance(message["content"], str) and len(message["content"]) > 0:
                    _system_content_block = SystemContentBlock(text=message["content"])
                elif isinstance(message["content"], list):
                    for m in message["content"]:
                        if m.get("type", "") == "text" and len(m["text"]) > 0:
                            _system_content_block = SystemContentBlock(text=m["text"])
                if _system_content_block is not None:
                    system_content_blocks.append(_system_content_block)
                system_prompt_indices.append(idx)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)

        inference_params = copy.deepcopy(optional_params)
        additional_request_keys = []
        additional_request_params = {}
        supported_converse_params = AmazonConverseConfig.__annotations__.keys()
        supported_tool_call_params = ["tools", "tool_choice"]
        supported_guardrail_params = ["guardrailConfig"]
        inference_params.pop("json_mode", None)  # used for handling json_schema
        ## TRANSFORMATION ##

        bedrock_messages: List[MessageBlock] = _bedrock_converse_messages_pt(
            messages=messages,
            model=model,
            llm_provider="bedrock_converse",
            user_continue_message=litellm_params.pop("user_continue_message", None),
        )

        # send all model-specific params in 'additional_request_params'
        for k, v in inference_params.items():
            if (
                k not in supported_converse_params
                and k not in supported_tool_call_params
                and k not in supported_guardrail_params
            ):
                additional_request_params[k] = v
                additional_request_keys.append(k)
        for key in additional_request_keys:
            inference_params.pop(key, None)

        bedrock_tools: List[ToolBlock] = _bedrock_tools_pt(
            inference_params.pop("tools", [])
        )
        bedrock_tool_config: Optional[ToolConfigBlock] = None
        if len(bedrock_tools) > 0:
            tool_choice_values: ToolChoiceValuesBlock = inference_params.pop(
                "tool_choice", None
            )
            bedrock_tool_config = ToolConfigBlock(
                tools=bedrock_tools,
            )
            if tool_choice_values is not None:
                bedrock_tool_config["toolChoice"] = tool_choice_values

        _data: RequestObject = {
            "messages": bedrock_messages,
            "additionalModelRequestFields": additional_request_params,
            "system": system_content_blocks,
            "inferenceConfig": InferenceConfig(**inference_params),
        }

        # Guardrail Config
        guardrail_config: Optional[GuardrailConfigBlock] = None
        request_guardrails_config = inference_params.pop("guardrailConfig", None)
        if request_guardrails_config is not None:
            guardrail_config = GuardrailConfigBlock(**request_guardrails_config)
            _data["guardrailConfig"] = guardrail_config

        # Tool Config
        if bedrock_tool_config is not None:
            _data["toolConfig"] = bedrock_tool_config

        return _data

    def _transform_response(
        self,
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: bool,
        logging_obj: Optional[Logging],
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
    ) -> Union[ModelResponse, CustomStreamWrapper]:

        ## LOGGING
        if logging_obj is not None:
            logging_obj.post_call(
                input=messages,
                api_key=api_key,
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
        print_verbose(f"raw model_response: {response.text}")
        json_mode: Optional[bool] = optional_params.pop("json_mode", None)
        ## RESPONSE OBJECT
        try:
            completion_response = ConverseResponseBlock(**response.json())  # type: ignore
        except Exception as e:
            raise BedrockError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    response.text, str(e)
                ),
                status_code=422,
            )

        """
        Bedrock Response Object has optional message block 

        completion_response["output"].get("message", None)

        A message block looks like this (Example 1): 
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": "Is there anything else you'd like to talk about? Perhaps I can help with some economic questions or provide some information about economic concepts?"
                    }
                ]
            }
        },
        (Example 2):
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_hbTgdi0CSLq_hM4P8csZJA",
                            "name": "top_song",
                            "input": {
                                "sign": "WZPZ"
                            }
                        }
                    }
                ]
            }
        }

        """
        message: Optional[MessageBlock] = completion_response["output"]["message"]
        chat_completion_message: ChatCompletionResponseMessage = {"role": "assistant"}
        content_str = ""
        tools: List[ChatCompletionToolCallChunk] = []
        if message is not None:
            for idx, content in enumerate(message["content"]):
                """
                - Content is either a tool response or text
                """
                if "text" in content:
                    content_str += content["text"]
                if "toolUse" in content:

                    ## check tool name was formatted by litellm
                    _response_tool_name = content["toolUse"]["name"]
                    response_tool_name = get_bedrock_tool_name(
                        response_tool_name=_response_tool_name
                    )
                    _function_chunk = ChatCompletionToolCallFunctionChunk(
                        name=response_tool_name,
                        arguments=json.dumps(content["toolUse"]["input"]),
                    )

                    _tool_response_chunk = ChatCompletionToolCallChunk(
                        id=content["toolUse"]["toolUseId"],
                        type="function",
                        function=_function_chunk,
                        index=idx,
                    )
                    tools.append(_tool_response_chunk)
        chat_completion_message["content"] = content_str

        if json_mode is True and tools is not None and len(tools) == 1:
            # to support 'json_schema' logic on bedrock models
            json_mode_content_str: Optional[str] = tools[0]["function"].get("arguments")
            if json_mode_content_str is not None:
                chat_completion_message["content"] = json_mode_content_str
        else:
            chat_completion_message["tool_calls"] = tools

        ## CALCULATING USAGE - bedrock returns usage in the headers
        input_tokens = completion_response["usage"]["inputTokens"]
        output_tokens = completion_response["usage"]["outputTokens"]
        total_tokens = completion_response["usage"]["totalTokens"]

        model_response.choices = [
            litellm.Choices(
                finish_reason=map_finish_reason(completion_response["stopReason"]),
                index=0,
                message=litellm.Message(**chat_completion_message),
            )
        ]
        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        setattr(model_response, "usage", usage)

        # Add "trace" from Bedrock guardrails - if user has opted in to returning it
        if "trace" in completion_response:
            setattr(model_response, "trace", completion_response["trace"])

        return model_response

    def _supported_cross_region_inference_region(self) -> List[str]:
        """
        Abbreviations of regions AWS Bedrock supports for cross region inference
        """
        return ["us", "eu"]

    def _get_base_model(self, model: str) -> str:
        """
        Get the base model from the given model name.

        Handle model names like - "us.meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
        AND "meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
        """

        potential_region = model.split(".", 1)[0]
        if potential_region in self._supported_cross_region_inference_region():
            return model.split(".", 1)[1]
        return model
