import os, types
import json
from enum import Enum
import requests, copy
import time
from typing import Callable, Optional, List
from litellm.utils import ModelResponse, Usage, map_finish_reason, CustomStreamWrapper
import litellm
from .prompt_templates.factory import prompt_factory, custom_prompt
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from .base import BaseLLM
import httpx


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "

    # constants from https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_constants.py


class AnthropicError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.anthropic.com/v1/messages"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class AnthropicConfig:
    """
    Reference: https://docs.anthropic.com/claude/reference/messages_post

    to pass metadata to anthropic, it's {"user_id": "any-relevant-information"}
    """

    max_tokens: Optional[int] = (
        4096  # anthropic requires a default value (Opus, Sonnet, and Haiku have the same default)
    )
    stop_sequences: Optional[list] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None
    system: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[
            int
        ] = 4096,  # You can pass in a value yourself or use the default value 4096
        stop_sequences: Optional[list] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
        system: Optional[str] = None,
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


# makes headers for API call
def validate_environment(api_key, user_headers):
    if api_key is None:
        raise ValueError(
            "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
        )
    headers = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    if user_headers is not None and isinstance(user_headers, dict):
        headers = {**headers, **user_headers}
    return headers


class AnthropicChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def process_response(
        self,
        model,
        response,
        model_response,
        _is_function_call,
        stream,
        logging_obj,
        api_key,
        data,
        messages,
        print_verbose,
    ):
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise AnthropicError(
                message=response.text, status_code=response.status_code
            )
        if "error" in completion_response:
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
            )
        elif len(completion_response["content"]) == 0:
            raise AnthropicError(
                message="No content in response",
                status_code=response.status_code,
            )
        else:
            text_content = ""
            tool_calls = []
            for content in completion_response["content"]:
                if content["type"] == "text":
                    text_content += content["text"]
                ## TOOL CALLING
                elif content["type"] == "tool_use":
                    tool_calls.append(
                        {
                            "id": content["id"],
                            "type": "function",
                            "function": {
                                "name": content["name"],
                                "arguments": json.dumps(content["input"]),
                            },
                        }
                    )

            _message = litellm.Message(
                tool_calls=tool_calls,
                content=text_content or None,
            )
            model_response.choices[0].message = _message  # type: ignore
            model_response._hidden_params["original_response"] = completion_response[
                "content"
            ]  # allow user to access raw anthropic tool calling response

            model_response.choices[0].finish_reason = map_finish_reason(
                completion_response["stop_reason"]
            )

        print_verbose(f"_is_function_call: {_is_function_call}; stream: {stream}")
        if _is_function_call and stream:
            print_verbose("INSIDE ANTHROPIC STREAMING TOOL CALLING CONDITION BLOCK")
            # return an iterator
            streaming_model_response = ModelResponse(stream=True)
            streaming_model_response.choices[0].finish_reason = model_response.choices[
                0
            ].finish_reason
            # streaming_model_response.choices = [litellm.utils.StreamingChoices()]
            streaming_choice = litellm.utils.StreamingChoices()
            streaming_choice.index = model_response.choices[0].index
            _tool_calls = []
            print_verbose(
                f"type of model_response.choices[0]: {type(model_response.choices[0])}"
            )
            print_verbose(f"type of streaming_choice: {type(streaming_choice)}")
            if isinstance(model_response.choices[0], litellm.Choices):
                if getattr(
                    model_response.choices[0].message, "tool_calls", None
                ) is not None and isinstance(
                    model_response.choices[0].message.tool_calls, list
                ):
                    for tool_call in model_response.choices[0].message.tool_calls:
                        _tool_call = {**tool_call.dict(), "index": 0}
                        _tool_calls.append(_tool_call)
                delta_obj = litellm.utils.Delta(
                    content=getattr(model_response.choices[0].message, "content", None),
                    role=model_response.choices[0].message.role,
                    tool_calls=_tool_calls,
                )
                streaming_choice.delta = delta_obj
                streaming_model_response.choices = [streaming_choice]
                completion_stream = ModelResponseIterator(
                    model_response=streaming_model_response
                )
                print_verbose(
                    "Returns anthropic CustomStreamWrapper with 'cached_response' streaming object"
                )
                return CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="cached_response",
                    logging_obj=logging_obj,
                )

        ## CALCULATING USAGE
        prompt_tokens = completion_response["usage"]["input_tokens"]
        completion_tokens = completion_response["usage"]["output_tokens"]
        total_tokens = prompt_tokens + completion_tokens

        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        model_response.usage = usage
        return model_response

    async def acompletion_stream_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data=None,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
    ):
        self.async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
        response = await self.async_handler.post(
            api_base, headers=headers, data=json.dumps(data)
        )

        if response.status_code != 200:
            raise AnthropicError(
                status_code=response.status_code, message=response.text
            )

        completion_stream = response.aiter_lines()

        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="anthropic",
            logging_obj=logging_obj,
        )
        return streamwrapper

    async def acompletion_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data=None,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
    ):
        self.async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
        response = await self.async_handler.post(
            api_base, headers=headers, data=json.dumps(data)
        )
        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            _is_function_call=_is_function_call,
            stream=stream,
            logging_obj=logging_obj,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params=None,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
    ):
        headers = validate_environment(api_key, headers)
        _is_function_call = False
        messages = copy.deepcopy(messages)
        optional_params = copy.deepcopy(optional_params)
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"],
                initial_prompt_value=model_prompt_details["initial_prompt_value"],
                final_prompt_value=model_prompt_details["final_prompt_value"],
                messages=messages,
            )
        else:
            # Separate system prompt from rest of message
            system_prompt_indices = []
            system_prompt = ""
            for idx, message in enumerate(messages):
                if message["role"] == "system":
                    system_prompt += message["content"]
                    system_prompt_indices.append(idx)
            if len(system_prompt_indices) > 0:
                for idx in reversed(system_prompt_indices):
                    messages.pop(idx)
            if len(system_prompt) > 0:
                optional_params["system"] = system_prompt
            # Format rest of message according to anthropic guidelines
            try:
                messages = prompt_factory(
                    model=model, messages=messages, custom_llm_provider="anthropic"
                )
            except Exception as e:
                raise AnthropicError(status_code=400, message=str(e))

        ## Load Config
        config = litellm.AnthropicConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        ## Handle Tool Calling
        if "tools" in optional_params:
            _is_function_call = True
            headers["anthropic-beta"] = "tools-2024-04-04"

            anthropic_tools = []
            for tool in optional_params["tools"]:
                new_tool = tool["function"]
                new_tool["input_schema"] = new_tool.pop("parameters")  # rename key
                anthropic_tools.append(new_tool)

            optional_params["tools"] = anthropic_tools

        stream = optional_params.pop("stream", None)

        data = {
            "model": model,
            "messages": messages,
            **optional_params,
        }

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )
        print_verbose(f"_is_function_call: {_is_function_call}")
        if acompletion == True:
            if (
                stream and not _is_function_call
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                print_verbose("makes async anthropic streaming POST request")
                data["stream"] = stream
                return self.acompletion_stream_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                )
            else:
                return self.acompletion_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                )
        else:
            ## COMPLETION CALL
            if (
                stream and not _is_function_call
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                print_verbose("makes anthropic streaming POST request")
                data["stream"] = stream
                response = requests.post(
                    api_base,
                    headers=headers,
                    data=json.dumps(data),
                    stream=stream,
                )

                if response.status_code != 200:
                    raise AnthropicError(
                        status_code=response.status_code, message=response.text
                    )

                completion_stream = response.iter_lines()
                streaming_response = CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="anthropic",
                    logging_obj=logging_obj,
                )
                return streaming_response

            else:
                response = requests.post(
                    api_base, headers=headers, data=json.dumps(data)
                )
                if response.status_code != 200:
                    raise AnthropicError(
                        status_code=response.status_code, message=response.text
                    )
        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            _is_function_call=_is_function_call,
            stream=stream,
            logging_obj=logging_obj,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
        )

    def embedding(self):
        # logic for parsing in - calling - parsing out model embedding calls
        pass


class ModelResponseIterator:
    def __init__(self, model_response):
        self.model_response = model_response
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.model_response
