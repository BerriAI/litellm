import os
import types
import traceback
import copy
import asyncio
import httpx
import aiohttp
import asyncio
from typing import Callable, Optional, Union

import litellm
from litellm.utils import ModelResponse, get_secret, Choices, Message, Usage, FunctionCall, ChatCompletionMessageToolCall, Function
from .prompt_templates.factory import prompt_factory, custom_prompt, get_system_prompt
from http import HTTPStatus


class DashScopeError(Exception):
    def __init__(self, request_id, status_code, code, message):
        self.request_id = request_id
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class DashScopeConfig:
    """
    Reference: https://help.aliyun.com/zh/dashscope/developer-reference/api-details

    The class `DashScopeConfig` provides configuration for the DashScope's API interface. Here are the parameters:

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `temperature` (float): Controls the randomness of the output. Note: The default value varies by model, see the Model.temperature attribute of the Model returned the genai.get_model function. Values can range from [0.0,1.0], inclusive. A value closer to 1.0 will produce responses that are more varied and creative, while a value closer to 0.0 will typically result in more straightforward responses from the model.

    - `max_tokens` (int): The maximum number of tokens to include in a candidate. If unset, this will default to output_token_limit specified in the model's specification.

    - `top_p` (float): Optional. The maximum cumulative probability of tokens to consider when sampling.

    - `top_k` (int): Optional. The maximum number of tokens to consider when sampling.

    """
    function_call: Optional[Union[str, dict]] = None,
    functions: Optional[list] = None,
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    def __init__(
        self,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
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


def convert_functions_to_tools(functions, messages):
    tools = []
    for function in functions:
        tool = {
            "function": function,
            "type": "function",
        }
        tools.append(tool)
    for message in messages:
        if message.get("role") == "function":
            message["role"] = "tool"
    return tools


def convert_function_call_to_tool_calls(function_call, messages):
    tool_calls = []
    tool = {
        "function": function_call,
        "type": "function",
    }
    tool_calls.append(tool)
    for message in messages:
        if message.get("role") == "function":
            message["role"] = "tool"
    return tool_calls


def convert_response_to_model_response_object(
    response,
    model_response: ModelResponse,
    support_functions=False
):
    choices_list = []
    for idx, choice in enumerate(response.output["choices"]):
        tool_calls = choice['message'].get("tool_calls")
        function_call = None
        if tool_calls and len(tool_calls) > 0 and support_functions:
            # convert tool_calls to function_call if support_functions is True
            function_call = FunctionCall(
                name=tool_calls[0]['function']['name'],
                arguments=tool_calls[0]['function']['arguments'],
            )
        message = Message(
            content=choice["message"]["content"],
            role=choice["message"]["role"],
            tool_calls=tool_calls,
            function_call=function_call,
        )
        new_choice = Choices(
            finish_reason=choice["finish_reason"], index=idx, message=message
        )
        choices_list.append(new_choice)
    model_response.choices = choices_list
    if "usage" in response:
        usage = Usage(
            prompt_tokens=response["usage"].input_tokens,
            output_tokens=response["usage"].output_tokens,
            total_tokens=response["usage"].total_tokens,
        )
        if not model_response.usage:
            model_response.usage.prompt_tokens += usage.prompt_tokens
            model_response.usage.output_tokens += usage.output_tokens
            model_response.usage.total_tokens += usage.total_tokens
        else:
            model_response.usage = usage


class EnhancedGenerator:
    def __init__(self, generator, incremental_output, support_functions):
        self.generator = generator
        self.incremental_output = incremental_output
        self.support_functions = support_functions

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)


def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    api_key,
    encoding,
    logging_obj,
    acompletion: bool = False,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    try:
        import dashscope  # type: ignore
    except:
        raise Exception(
            "Importing dashscope failed, please run 'pip install -q dashscope"
        )

    dashscope.api_key = api_key
    # Load Config
    inference_params = copy.deepcopy(optional_params)
    stream = inference_params.pop("stream", None)
    tools = None
    tool_calls = None

    functions = inference_params.pop("functions", None)

    support_functions = False

    new_messages = []

    for message in messages:
        if isinstance(message, Message):
            new_messages.append(message.model_dump())
        else:
            new_messages.append(copy.deepcopy(message))

    # If functions are provided, convert them to tools
    if functions:
        support_functions = True
        tools = convert_functions_to_tools(functions, new_messages)
    else:
        tools = inference_params.pop("tools", None)
        # Dashscope doesn't support tool_call_id yet
        inference_params.pop("tool_call_id", None)

    # If function_call is provided in messages, convert it to tool_calls
    for message in new_messages:
        function_call = message.get("function_call")
        if function_call:
            message["function_call"] = None
            support_functions = True
            tool_calls = convert_function_call_to_tool_calls(
                function_call, new_messages)
            message["tool_calls"] = tool_calls

    # If the first message is a system message and the second message is an assistant message, change the first message to a user message
    if len(new_messages) > 1:
        if new_messages[0]["role"] == "system" and new_messages[1]["role"] == "assistant":
            new_messages[0]["role"] = "user"

    # Content length must be greater than 0
    for message in new_messages:
        if message["role"] == "user" and len(message["content"]) == 0:
            message["content"] = " "

    # System message should be the first one
    for idx, message in enumerate(new_messages):
        if idx > 0 and message["role"] == "system":
            message["role"] = "user"

    # If the last message is an assistant message, add one user message with the empty content
    if len(new_messages) > 1 and new_messages[-1]["role"] == "assistant":
        new_messages.append(
            {
                "role": "user",
                "content": " ",
            }
        )
    
    config = DashScopeConfig.get_config()
    for k, v in config.items():
        if k not in inference_params:
            inference_params[k] = v

    # LOGGING
    logging_obj.pre_call(
        input=new_messages,
        api_key="",
        additional_args={
            "complete_input_dict": {
                "inference_params": inference_params,
            }
        },
    )

    # COMPLETION CALL
    try:
        if not stream:
            response = dashscope.Generation.call(
                model=model,
                messages=new_messages,
                tools=tools,
                # set the result to be "message" format.
                result_format='message',
                **inference_params,
            )

            model_response = ModelResponse()
            model_response.model = model
            if response.status_code == HTTPStatus.OK:
                convert_response_to_model_response_object(
                    response, model_response, support_functions)

                # LOGGING
                logging_obj.post_call(
                    input=new_messages,
                    api_key="",
                    original_response=model_response.model_dump(),
                )
            else:
                raise DashScopeError(
                    request_id=response.request_id,
                    status_code=response.status_code,
                    code=response.code,
                    message=response.message,
                )
        else:
            incremental_output = True
            if tools != None or tool_calls != None:
                incremental_output = None
            responses = dashscope.Generation.call(
                model=model,
                messages=new_messages,
                stream=stream,
                tools=tools,
                incremental_output=incremental_output,
                # set the result to be "message" format.
                result_format='message',
                **inference_params,
            )
            return EnhancedGenerator(responses, incremental_output, support_functions)

    except Exception as e:
        raise e

    return model_response


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    # TBD
    pass
