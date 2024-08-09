import copy
import json
import os
import time
import traceback
import types
from enum import Enum
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore
from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_async_httpx_client,
    _get_httpx_client,
)
from litellm.types.llms.anthropic import (
    AnthopicMessagesAssistantMessageParam,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
    AnthropicMessagesUserMessageParam,
    AnthropicResponse,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockToolUse,
    AnthropicResponseUsageBlock,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    ContentJsonBlockDelta,
    ContentTextBlockDelta,
    MessageBlockDelta,
    MessageDelta,
    MessageStartBlock,
    UsageDelta,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    ChatCompletionRequest,
    ChatCompletionResponseMessage,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolChoiceValues,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUsageBlock,
    ChatCompletionUserMessage,
)
from litellm.types.utils import Choices, GenericStreamingChunk
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from .base import BaseLLM
from .prompt_templates.factory import custom_prompt, prompt_factory


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "

    # constants from https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_constants.py


class AnthropicError(Exception):
    def __init__(self, status_code: int, message):
        self.status_code = status_code
        self.message: str = message
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

    def get_supported_openai_params(self):
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "tools",
            "tool_choice",
            "extra_headers",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "tool_choice":
                _tool_choice: Optional[AnthropicMessagesToolChoice] = None
                if value == "auto":
                    _tool_choice = {"type": "auto"}
                elif value == "required":
                    _tool_choice = {"type": "any"}
                elif isinstance(value, dict):
                    _tool_choice = {"type": "tool", "name": value["function"]["name"]}

                if _tool_choice is not None:
                    optional_params["tool_choice"] = _tool_choice
            if param == "stream" and value == True:
                optional_params["stream"] = value
            if param == "stop":
                if isinstance(value, str):
                    if (
                        value == "\n"
                    ) and litellm.drop_params == True:  # anthropic doesn't allow whitespace characters as stop-sequences
                        continue
                    value = [value]
                elif isinstance(value, list):
                    new_v = []
                    for v in value:
                        if (
                            v == "\n"
                        ) and litellm.drop_params == True:  # anthropic doesn't allow whitespace characters as stop-sequences
                            continue
                        new_v.append(v)
                    if len(new_v) > 0:
                        value = new_v
                    else:
                        continue
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
        return optional_params

    ### FOR [BETA] `/v1/messages` endpoint support

    def translatable_anthropic_params(self) -> List:
        """
        Which anthropic params, we need to translate to the openai format.
        """
        return ["messages", "metadata", "system", "tool_choice", "tools"]

    def translate_anthropic_messages_to_openai(
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
    ) -> List:
        new_messages: List[AllMessageValues] = []
        for m in messages:
            user_message: Optional[ChatCompletionUserMessage] = None
            tool_message_list: List[ChatCompletionToolMessage] = []
            new_user_content_list: List[
                Union[ChatCompletionTextObject, ChatCompletionImageObject]
            ] = []
            ## USER MESSAGE ##
            if m["role"] == "user":
                ## translate user message
                if isinstance(m["content"], str):
                    user_message = ChatCompletionUserMessage(
                        role="user", content=m["content"]
                    )
                elif isinstance(m["content"], list):
                    for content in m["content"]:
                        if content["type"] == "text":
                            text_obj = ChatCompletionTextObject(
                                type="text", text=content["text"]
                            )
                            new_user_content_list.append(text_obj)
                        elif content["type"] == "image":
                            image_url = ChatCompletionImageUrlObject(
                                url=f"data:{content['type']};base64,{content['source']}"
                            )
                            image_obj = ChatCompletionImageObject(
                                type="image_url", image_url=image_url
                            )

                            new_user_content_list.append(image_obj)
                        elif content["type"] == "tool_result":
                            if "content" not in content:
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content["tool_use_id"],
                                    content="",
                                )
                                tool_message_list.append(tool_result)
                            elif isinstance(content["content"], str):
                                tool_result = ChatCompletionToolMessage(
                                    role="tool",
                                    tool_call_id=content["tool_use_id"],
                                    content=content["content"],
                                )
                                tool_message_list.append(tool_result)
                            elif isinstance(content["content"], list):
                                for c in content["content"]:
                                    if c["type"] == "text":
                                        tool_result = ChatCompletionToolMessage(
                                            role="tool",
                                            tool_call_id=content["tool_use_id"],
                                            content=c["text"],
                                        )
                                        tool_message_list.append(tool_result)
                                    elif c["type"] == "image":
                                        image_str = (
                                            f"data:{c['type']};base64,{c['source']}"
                                        )
                                        tool_result = ChatCompletionToolMessage(
                                            role="tool",
                                            tool_call_id=content["tool_use_id"],
                                            content=image_str,
                                        )
                                        tool_message_list.append(tool_result)

            if user_message is not None:
                new_messages.append(user_message)

            if len(new_user_content_list) > 0:
                new_messages.append({"role": "user", "content": new_user_content_list})

            if len(tool_message_list) > 0:
                new_messages.extend(tool_message_list)

            ## ASSISTANT MESSAGE ##
            assistant_message_str: Optional[str] = None
            tool_calls: List[ChatCompletionAssistantToolCall] = []
            if m["role"] == "assistant":
                if isinstance(m["content"], str):
                    assistant_message_str = m["content"]
                elif isinstance(m["content"], list):
                    for content in m["content"]:
                        if content["type"] == "text":
                            if assistant_message_str is None:
                                assistant_message_str = content["text"]
                            else:
                                assistant_message_str += content["text"]
                        elif content["type"] == "tool_use":
                            function_chunk = ChatCompletionToolCallFunctionChunk(
                                name=content["name"],
                                arguments=json.dumps(content["input"]),
                            )

                            tool_calls.append(
                                ChatCompletionAssistantToolCall(
                                    id=content["id"],
                                    type="function",
                                    function=function_chunk,
                                )
                            )

            if assistant_message_str is not None or len(tool_calls) > 0:
                assistant_message = ChatCompletionAssistantMessage(
                    role="assistant",
                    content=assistant_message_str,
                )
                if len(tool_calls) > 0:
                    assistant_message["tool_calls"] = tool_calls
                new_messages.append(assistant_message)

        return new_messages

    def translate_anthropic_tool_choice_to_openai(
        self, tool_choice: AnthropicMessagesToolChoice
    ) -> ChatCompletionToolChoiceValues:
        if tool_choice["type"] == "any":
            return "required"
        elif tool_choice["type"] == "auto":
            return "auto"
        elif tool_choice["type"] == "tool":
            tc_function_param = ChatCompletionToolChoiceFunctionParam(
                name=tool_choice.get("name", "")
            )
            return ChatCompletionToolChoiceObjectParam(
                type="function", function=tc_function_param
            )
        else:
            raise ValueError(
                "Incompatible tool choice param submitted - {}".format(tool_choice)
            )

    def translate_anthropic_tools_to_openai(
        self, tools: List[AnthropicMessagesTool]
    ) -> List[ChatCompletionToolParam]:
        new_tools: List[ChatCompletionToolParam] = []
        for tool in tools:
            function_chunk = ChatCompletionToolParamFunctionChunk(
                name=tool["name"],
                parameters=tool["input_schema"],
            )
            if "description" in tool:
                function_chunk["description"] = tool["description"]
            new_tools.append(
                ChatCompletionToolParam(type="function", function=function_chunk)
            )

        return new_tools

    def translate_anthropic_to_openai(
        self, anthropic_message_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """
        This is used by the beta Anthropic Adapter, for translating anthropic `/v1/messages` requests to the openai format.
        """
        new_messages: List[AllMessageValues] = []

        ## CONVERT ANTHROPIC MESSAGES TO OPENAI
        new_messages = self.translate_anthropic_messages_to_openai(
            messages=anthropic_message_request["messages"]
        )
        ## ADD SYSTEM MESSAGE TO MESSAGES
        if "system" in anthropic_message_request:
            new_messages.insert(
                0,
                ChatCompletionSystemMessage(
                    role="system", content=anthropic_message_request["system"]
                ),
            )

        new_kwargs: ChatCompletionRequest = {
            "model": anthropic_message_request["model"],
            "messages": new_messages,
        }
        ## CONVERT METADATA (user_id)
        if "metadata" in anthropic_message_request:
            if "user_id" in anthropic_message_request["metadata"]:
                new_kwargs["user"] = anthropic_message_request["metadata"]["user_id"]

        # Pass litellm proxy specific metadata
        if "litellm_metadata" in anthropic_message_request:
            # metadata will be passed to litellm.acompletion(), it's a litellm_param
            new_kwargs["metadata"] = anthropic_message_request.pop("litellm_metadata")

        ## CONVERT TOOL CHOICE
        if "tool_choice" in anthropic_message_request:
            new_kwargs["tool_choice"] = self.translate_anthropic_tool_choice_to_openai(
                tool_choice=anthropic_message_request["tool_choice"]
            )
        ## CONVERT TOOLS
        if "tools" in anthropic_message_request:
            new_kwargs["tools"] = self.translate_anthropic_tools_to_openai(
                tools=anthropic_message_request["tools"]
            )

        translatable_params = self.translatable_anthropic_params()
        for k, v in anthropic_message_request.items():
            if k not in translatable_params:  # pass remaining params as is
                new_kwargs[k] = v  # type: ignore

        return new_kwargs

    def _translate_openai_content_to_anthropic(
        self, choices: List[Choices]
    ) -> List[
        Union[AnthropicResponseContentBlockText, AnthropicResponseContentBlockToolUse]
    ]:
        new_content: List[
            Union[
                AnthropicResponseContentBlockText, AnthropicResponseContentBlockToolUse
            ]
        ] = []
        for choice in choices:
            if (
                choice.message.tool_calls is not None
                and len(choice.message.tool_calls) > 0
            ):
                for tool_call in choice.message.tool_calls:
                    new_content.append(
                        AnthropicResponseContentBlockToolUse(
                            type="tool_use",
                            id=tool_call.id,
                            name=tool_call.function.name or "",
                            input=json.loads(tool_call.function.arguments),
                        )
                    )
            elif choice.message.content is not None:
                new_content.append(
                    AnthropicResponseContentBlockText(
                        type="text", text=choice.message.content
                    )
                )

        return new_content

    def _translate_openai_finish_reason_to_anthropic(
        self, openai_finish_reason: str
    ) -> AnthropicFinishReason:
        if openai_finish_reason == "stop":
            return "end_turn"
        elif openai_finish_reason == "length":
            return "max_tokens"
        elif openai_finish_reason == "tool_calls":
            return "tool_use"
        return "end_turn"

    def translate_openai_response_to_anthropic(
        self, response: litellm.ModelResponse
    ) -> AnthropicResponse:
        ## translate content block
        anthropic_content = self._translate_openai_content_to_anthropic(choices=response.choices)  # type: ignore
        ## extract finish reason
        anthropic_finish_reason = self._translate_openai_finish_reason_to_anthropic(
            openai_finish_reason=response.choices[0].finish_reason  # type: ignore
        )
        # extract usage
        usage: litellm.Usage = getattr(response, "usage")
        anthropic_usage = AnthropicResponseUsageBlock(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )
        translated_obj = AnthropicResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,
            content=anthropic_content,
            stop_reason=anthropic_finish_reason,
        )

        return translated_obj

    def _translate_streaming_openai_chunk_to_anthropic(
        self, choices: List[OpenAIStreamingChoice]
    ) -> Tuple[
        Literal["text_delta", "input_json_delta"],
        Union[ContentTextBlockDelta, ContentJsonBlockDelta],
    ]:
        text: str = ""
        partial_json: Optional[str] = None
        for choice in choices:
            if choice.delta.content is not None:
                text += choice.delta.content
            elif choice.delta.tool_calls is not None:
                partial_json = ""
                for tool in choice.delta.tool_calls:
                    if (
                        tool.function is not None
                        and tool.function.arguments is not None
                    ):
                        partial_json += tool.function.arguments

        if partial_json is not None:
            return "input_json_delta", ContentJsonBlockDelta(
                type="input_json_delta", partial_json=partial_json
            )
        else:
            return "text_delta", ContentTextBlockDelta(type="text_delta", text=text)

    def translate_streaming_openai_response_to_anthropic(
        self, response: litellm.ModelResponse
    ) -> Union[ContentBlockDelta, MessageBlockDelta]:
        ## base case - final chunk w/ finish reason
        if response.choices[0].finish_reason is not None:
            delta = MessageDelta(
                stop_reason=self._translate_openai_finish_reason_to_anthropic(
                    response.choices[0].finish_reason
                ),
            )
            if getattr(response, "usage", None) is not None:
                litellm_usage_chunk: Optional[litellm.Usage] = response.usage  # type: ignore
            elif (
                hasattr(response, "_hidden_params")
                and "usage" in response._hidden_params
            ):
                litellm_usage_chunk = response._hidden_params["usage"]
            else:
                litellm_usage_chunk = None
            if litellm_usage_chunk is not None:
                usage_delta = UsageDelta(
                    input_tokens=litellm_usage_chunk.prompt_tokens or 0,
                    output_tokens=litellm_usage_chunk.completion_tokens or 0,
                )
            else:
                usage_delta = UsageDelta(input_tokens=0, output_tokens=0)
            return MessageBlockDelta(
                type="message_delta", delta=delta, usage=usage_delta
            )
        (
            type_of_content,
            content_block_delta,
        ) = self._translate_streaming_openai_chunk_to_anthropic(
            choices=response.choices  # type: ignore
        )
        return ContentBlockDelta(
            type="content_block_delta",
            index=response.choices[0].index,
            delta=content_block_delta,
        )


# makes headers for API call
def validate_environment(api_key, user_headers, model):
    if api_key is None:
        raise litellm.AuthenticationError(
            message="Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` in your environment vars",
            llm_provider="anthropic",
            model=model,
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


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
):
    if client is None:
        client = _get_async_httpx_client()  # Create a new client if none provided

    try:
        response = await client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        raise AnthropicError(
            status_code=e.response.status_code, message=await e.response.aread()
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise AnthropicError(status_code=500, message=str(e))

    if response.status_code != 200:
        raise AnthropicError(
            status_code=response.status_code, message=await response.aread()
        )

    completion_stream = ModelResponseIterator(
        streaming_response=response.aiter_lines(), sync_stream=False
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,  # Pass the completion stream for logging
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
):
    if client is None:
        client = HTTPHandler()  # Create a new client if none provided

    try:
        response = client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        raise AnthropicError(
            status_code=e.response.status_code, message=e.response.read()
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise AnthropicError(status_code=500, message=str(e))

    if response.status_code != 200:
        raise AnthropicError(status_code=response.status_code, message=response.read())

    completion_stream = ModelResponseIterator(
        streaming_response=response.iter_lines(), sync_stream=True
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class AnthropicChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def _process_response(
        self,
        model: str,
        response: Union[requests.Response, httpx.Response],
        model_response: ModelResponse,
        stream: bool,
        logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
        json_mode: bool,
    ) -> ModelResponse:
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
        else:
            text_content = ""
            tool_calls: List[ChatCompletionToolCallChunk] = []
            for idx, content in enumerate(completion_response["content"]):
                if content["type"] == "text":
                    text_content += content["text"]
                ## TOOL CALLING
                elif content["type"] == "tool_use":
                    tool_calls.append(
                        ChatCompletionToolCallChunk(
                            id=content["id"],
                            type="function",
                            function=ChatCompletionToolCallFunctionChunk(
                                name=content["name"],
                                arguments=json.dumps(content["input"]),
                            ),
                            index=idx,
                        )
                    )

            _message = litellm.Message(
                tool_calls=tool_calls,
                content=text_content or None,
            )

            ## HANDLE JSON MODE - anthropic returns single function call
            if json_mode and len(tool_calls) == 1:
                json_mode_content_str: Optional[str] = tool_calls[0]["function"].get(
                    "arguments"
                )
                if json_mode_content_str is not None:
                    args = json.loads(json_mode_content_str)
                    values: Optional[dict] = args.get("values")
                    if values is not None:
                        _message = litellm.Message(content=json.dumps(values))
                        completion_response["stop_reason"] = "stop"
            model_response.choices[0].message = _message  # type: ignore
            model_response._hidden_params["original_response"] = completion_response[
                "content"
            ]  # allow user to access raw anthropic tool calling response

            model_response.choices[0].finish_reason = map_finish_reason(
                completion_response["stop_reason"]
            )

        ## CALCULATING USAGE
        prompt_tokens = completion_response["usage"]["input_tokens"]
        completion_tokens = completion_response["usage"]["output_tokens"]
        total_tokens = prompt_tokens + completion_tokens

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        setattr(model_response, "usage", usage)  # type: ignore
        return model_response

    async def acompletion_stream_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data: dict,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
    ):
        data["stream"] = True

        streamwrapper = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                client=None,
                api_base=api_base,
                headers=headers,
                data=json.dumps(data),
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                timeout=timeout,
            ),
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
        timeout: Union[float, httpx.Timeout],
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data: dict,
        optional_params: dict,
        json_mode: bool,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client=None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        async_handler = _get_async_httpx_client()

        try:
            response = await async_handler.post(
                api_base, headers=headers, json=data, timeout=timeout
            )
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=messages,
                api_key=api_key,
                original_response=str(e),
                additional_args={"complete_input_dict": data},
            )
            raise e

        return self._process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
            json_mode=json_mode,
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
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        headers = validate_environment(api_key, headers, model)
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
                    valid_content: bool = False
                    if isinstance(message["content"], str):
                        system_prompt += message["content"]
                        valid_content = True
                    elif isinstance(message["content"], list):
                        for content in message["content"]:
                            system_prompt += content.get("text", "")
                        valid_content = True

                    if valid_content:
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
                raise AnthropicError(
                    status_code=400,
                    message="{}\n{}\nReceived Messages={}".format(
                        str(e), traceback.format_exc(), messages
                    ),
                )

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
            if "anthropic-beta" not in headers:
                # default to v1 of "anthropic-beta"
                headers["anthropic-beta"] = "tools-2024-05-16"

            anthropic_tools = []
            for tool in optional_params["tools"]:
                if "input_schema" in tool:  # assume in anthropic format
                    anthropic_tools.append(tool)
                else:  # assume openai tool call
                    new_tool = tool["function"]
                    new_tool["input_schema"] = new_tool.pop("parameters")  # rename key
                    anthropic_tools.append(new_tool)

            optional_params["tools"] = anthropic_tools

        stream = optional_params.pop("stream", None)
        is_vertex_request: bool = optional_params.pop("is_vertex_request", False)
        json_mode: bool = optional_params.pop("json_mode", False)

        data = {
            "messages": messages,
            **optional_params,
        }

        if is_vertex_request is False:
            data["model"] = model

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
        if acompletion is True:
            if (
                stream is True
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
                    timeout=timeout,
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
                    client=client,
                    json_mode=json_mode,
                    timeout=timeout,
                )
        else:
            ## COMPLETION CALL
            if client is None or not isinstance(client, HTTPHandler):
                client = HTTPHandler(timeout=timeout)  # type: ignore
            else:
                client = client
            if (
                stream is True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                data["stream"] = stream
                return CustomStreamWrapper(
                    completion_stream=None,
                    make_call=partial(
                        make_sync_call,
                        client=None,
                        api_base=api_base,
                        headers=headers,  # type: ignore
                        data=json.dumps(data),
                        model=model,
                        messages=messages,
                        logging_obj=logging_obj,
                        timeout=timeout,
                    ),
                    model=model,
                    custom_llm_provider="anthropic",
                    logging_obj=logging_obj,
                )

            else:
                response = client.post(
                    api_base, headers=headers, data=json.dumps(data), timeout=timeout
                )
                if response.status_code != 200:
                    raise AnthropicError(
                        status_code=response.status_code, message=response.text
                    )

        return self._process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def embedding(self):
        # logic for parsing in - calling - parsing out model embedding calls
        pass


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.content_blocks: List[ContentBlockDelta] = []

    def check_empty_tool_call_args(self) -> bool:
        """
        Check if the tool call block so far has been an empty string
        """
        args = ""
        # if text content block -> skip
        if len(self.content_blocks) == 0:
            return False

        if self.content_blocks[0]["delta"]["type"] == "text_delta":
            return False

        for block in self.content_blocks:
            if block["delta"]["type"] == "input_json_delta":
                args += block["delta"].get("partial_json", "")  # type: ignore

        if len(args) == 0:
            return True
        return False

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            type_chunk = chunk.get("type", "") or ""

            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None

            index = int(chunk.get("index", 0))
            if type_chunk == "content_block_delta":
                """
                Anthropic content chunk
                chunk = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello'}}
                """
                content_block = ContentBlockDelta(**chunk)  # type: ignore
                self.content_blocks.append(content_block)
                if "text" in content_block["delta"]:
                    text = content_block["delta"]["text"]
                elif "partial_json" in content_block["delta"]:
                    tool_use = {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": None,
                            "arguments": content_block["delta"]["partial_json"],
                        },
                        "index": content_block["index"],
                    }
            elif type_chunk == "content_block_start":
                """
                event: content_block_start
                data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}
                """
                content_block_start = ContentBlockStart(**chunk)  # type: ignore
                self.content_blocks = []  # reset content blocks when new block starts
                if content_block_start["content_block"]["type"] == "text":
                    text = content_block_start["content_block"]["text"]
                elif content_block_start["content_block"]["type"] == "tool_use":
                    tool_use = {
                        "id": content_block_start["content_block"]["id"],
                        "type": "function",
                        "function": {
                            "name": content_block_start["content_block"]["name"],
                            "arguments": "",
                        },
                        "index": content_block_start["index"],
                    }
            elif type_chunk == "content_block_stop":
                content_block_stop = ContentBlockStop(**chunk)  # type: ignore
                # check if tool call content block
                is_empty = self.check_empty_tool_call_args()
                if is_empty:
                    tool_use = {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": None,
                            "arguments": "{}",
                        },
                        "index": content_block_stop["index"],
                    }
            elif type_chunk == "message_delta":
                """
                Anthropic
                chunk = {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens', 'stop_sequence': None}, 'usage': {'output_tokens': 10}}
                """
                # TODO - get usage from this chunk, set in response
                message_delta = MessageBlockDelta(**chunk)  # type: ignore
                finish_reason = map_finish_reason(
                    finish_reason=message_delta["delta"].get("stop_reason", "stop")
                    or "stop"
                )
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=message_delta["usage"].get("input_tokens", 0),
                    completion_tokens=message_delta["usage"].get("output_tokens", 0),
                    total_tokens=message_delta["usage"].get("input_tokens", 0)
                    + message_delta["usage"].get("output_tokens", 0),
                )
                is_finished = True
            elif type_chunk == "message_start":
                """
                Anthropic
                chunk = {
                    "type": "message_start",
                    "message": {
                        "id": "msg_vrtx_011PqREFEMzd3REdCoUFAmdG",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-sonnet-20240229",
                        "content": [],
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {
                            "input_tokens": 270,
                            "output_tokens": 1
                        }
                    }
                }
                """
                message_start_block = MessageStartBlock(**chunk)  # type: ignore
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=message_start_block["message"]
                    .get("usage", {})
                    .get("input_tokens", 0),
                    completion_tokens=message_start_block["message"]
                    .get("usage", {})
                    .get("output_tokens", 0),
                    total_tokens=message_start_block["message"]
                    .get("usage", {})
                    .get("input_tokens", 0)
                    + message_start_block["message"]
                    .get("usage", {})
                    .get("output_tokens", 0),
                )
            elif type_chunk == "error":
                """
                {"type":"error","error":{"details":null,"type":"api_error","message":"Internal server error"}      }
                """
                _error_dict = chunk.get("error", {}) or {}
                message = _error_dict.get("message", None) or str(chunk)
                raise AnthropicError(
                    message=message,
                    status_code=500,  # it looks like Anthropic API does not return a status code in the chunk error - default to 500
                )
            returned_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
            )

            return returned_chunk

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):  # Handle binary data
                str_line = chunk.decode("utf-8")  # Convert bytes to string
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]

            if str_line.startswith("data:"):
                data_json = json.loads(str_line[5:])
                return self.chunk_parser(chunk=data_json)
            else:
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):  # Handle binary data
                str_line = chunk.decode("utf-8")  # Convert bytes to string
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]

            if str_line.startswith("data:"):
                data_json = json.loads(str_line[5:])
                return self.chunk_parser(chunk=data_json)
            else:
                return GenericStreamingChunk(
                    text="",
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
