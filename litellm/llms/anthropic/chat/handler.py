"""
Calling + translation logic for anthropic's `/v1/messages` endpoint
"""

import copy
import json
import os
import time
import traceback
import types
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

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
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.anthropic import (
    AnthropicChatCompletionUsageBlock,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageBlockDelta,
    MessageStartBlock,
    UsageDelta,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from ...base import BaseLLM
from ..common_utils import AnthropicError, process_anthropic_headers
from .transformation import AnthropicConfig


# makes headers for API call
def validate_environment(
    api_key, user_headers, model, messages: List[AllMessageValues]
):
    cache_headers = {}
    if api_key is None:
        raise litellm.AuthenticationError(
            message="Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` in your environment vars",
            llm_provider="anthropic",
            model=model,
        )

    if AnthropicConfig().is_cache_control_set(messages=messages):
        cache_headers = AnthropicConfig().get_cache_control_headers()
    headers = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": api_key,
    }

    headers.update(cache_headers)

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
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_aclient

    try:
        response = await client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise AnthropicError(
            status_code=e.response.status_code,
            message=await e.response.aread(),
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise AnthropicError(status_code=500, message=str(e))

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

    return completion_stream, response.headers


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_client  # re-use a module level client

    try:
        response = client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise AnthropicError(
            status_code=e.response.status_code,
            message=e.response.read(),
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise AnthropicError(status_code=500, message=str(e))

    if response.status_code != 200:
        response_headers = getattr(response, "headers", None)
        raise AnthropicError(
            status_code=response.status_code,
            message=response.read(),
            headers=response_headers,
        )

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

    return completion_stream, response.headers


class AnthropicChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def _process_response(
        self,
        model: str,
        response: Union[requests.Response, httpx.Response],
        model_response: ModelResponse,
        stream: bool,
        logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,  # type: ignore
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
        json_mode: bool,
    ) -> ModelResponse:
        _hidden_params: Dict = {}
        _hidden_params["additional_headers"] = process_anthropic_headers(
            dict(response.headers)
        )
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
        except Exception as e:
            response_headers = getattr(response, "headers", None)
            raise AnthropicError(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), response.text
                ),
                status_code=response.status_code,
                headers=response_headers,
            )
        if "error" in completion_response:
            response_headers = getattr(response, "headers", None)
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
                headers=response_headers,
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
        _usage = completion_response["usage"]
        total_tokens = prompt_tokens + completion_tokens

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        if "cache_creation_input_tokens" in _usage:
            usage["cache_creation_input_tokens"] = _usage["cache_creation_input_tokens"]
        if "cache_read_input_tokens" in _usage:
            usage["cache_read_input_tokens"] = _usage["cache_read_input_tokens"]
        setattr(model_response, "usage", usage)  # type: ignore

        model_response._hidden_params = _hidden_params
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
        client: Optional[AsyncHTTPHandler],
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

        completion_stream, headers = await make_call(
            client=client,
            api_base=api_base,
            headers=headers,
            data=json.dumps(data),
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            timeout=timeout,
        )
        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="anthropic",
            logging_obj=logging_obj,
            _response_headers=process_anthropic_headers(headers),
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
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        async_handler = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.ANTHROPIC
        )

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
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise AnthropicError(
                message=error_text,
                status_code=status_code,
                headers=error_headers,
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
        headers = validate_environment(api_key, headers, model, messages=messages)
        _is_function_call = False
        messages = copy.deepcopy(messages)
        optional_params = copy.deepcopy(optional_params)
        stream = optional_params.pop("stream", None)
        json_mode: bool = optional_params.pop("json_mode", False)
        is_vertex_request: bool = optional_params.pop("is_vertex_request", False)

        data = AnthropicConfig()._transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            headers=headers,
            _is_function_call=_is_function_call,
            is_vertex_request=is_vertex_request,
        )

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
                    client=(
                        client
                        if client is not None and isinstance(client, AsyncHTTPHandler)
                        else None
                    ),
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
            if (
                stream is True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                data["stream"] = stream
                completion_stream, headers = make_sync_call(
                    client=client,
                    api_base=api_base,
                    headers=headers,  # type: ignore
                    data=json.dumps(data),
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                    timeout=timeout,
                )
                return CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="anthropic",
                    logging_obj=logging_obj,
                    _response_headers=process_anthropic_headers(headers),
                )

            else:
                if client is None or not isinstance(client, HTTPHandler):
                    client = HTTPHandler(timeout=timeout)  # type: ignore
                else:
                    client = client

                try:
                    response = client.post(
                        api_base,
                        headers=headers,
                        data=json.dumps(data),
                        timeout=timeout,
                    )
                except Exception as e:
                    status_code = getattr(e, "status_code", 500)
                    error_headers = getattr(e, "headers", None)
                    error_text = getattr(e, "text", str(e))
                    error_response = getattr(e, "response", None)
                    if error_headers is None and error_response:
                        error_headers = getattr(error_response, "headers", None)
                    if error_response and hasattr(error_response, "text"):
                        error_text = getattr(error_response, "text", error_text)
                    raise AnthropicError(
                        message=error_text,
                        status_code=status_code,
                        headers=error_headers,
                    )

        return self._process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            api_key=api_key,
            data=data,  # type: ignore
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
        self.tool_index = -1

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

    def _handle_usage(
        self, anthropic_usage_chunk: Union[dict, UsageDelta]
    ) -> AnthropicChatCompletionUsageBlock:

        usage_block = AnthropicChatCompletionUsageBlock(
            prompt_tokens=anthropic_usage_chunk.get("input_tokens", 0),
            completion_tokens=anthropic_usage_chunk.get("output_tokens", 0),
            total_tokens=anthropic_usage_chunk.get("input_tokens", 0)
            + anthropic_usage_chunk.get("output_tokens", 0),
        )

        cache_creation_input_tokens = anthropic_usage_chunk.get(
            "cache_creation_input_tokens"
        )
        if cache_creation_input_tokens is not None and isinstance(
            cache_creation_input_tokens, int
        ):
            usage_block["cache_creation_input_tokens"] = cache_creation_input_tokens

        cache_read_input_tokens = anthropic_usage_chunk.get("cache_read_input_tokens")
        if cache_read_input_tokens is not None and isinstance(
            cache_read_input_tokens, int
        ):
            usage_block["cache_read_input_tokens"] = cache_read_input_tokens

        return usage_block

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
                        "index": self.tool_index,
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
                    self.tool_index += 1
                    tool_use = {
                        "id": content_block_start["content_block"]["id"],
                        "type": "function",
                        "function": {
                            "name": content_block_start["content_block"]["name"],
                            "arguments": "",
                        },
                        "index": self.tool_index,
                    }
            elif type_chunk == "content_block_stop":
                ContentBlockStop(**chunk)  # type: ignore
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
                        "index": self.tool_index,
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
                usage = self._handle_usage(anthropic_usage_chunk=message_delta["usage"])
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
                if "usage" in message_start_block["message"]:
                    usage = self._handle_usage(
                        anthropic_usage_chunk=message_start_block["message"]["usage"]
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
