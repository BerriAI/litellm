"""
Calling + translation logic for anthropic's `/v1/messages` endpoint
"""

import copy
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import httpx  # type: ignore

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm import LlmProviders
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.anthropic import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageBlockDelta,
    MessageStartBlock,
    UsageDelta,
)
from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
)
from litellm.types.utils import (
    Delta,
    GenericStreamingChunk,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from litellm.utils import CustomStreamWrapper, ModelResponse, ProviderConfigManager

from ...base import BaseLLM
from ..common_utils import AnthropicError, process_anthropic_headers
from .transformation import AnthropicConfig


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
    json_mode: bool,
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
        streaming_response=response.aiter_lines(),
        sync_stream=False,
        json_mode=json_mode,
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
    json_mode: bool,
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
        streaming_response=response.iter_lines(), sync_stream=True, json_mode=json_mode
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
        json_mode: bool,
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
            json_mode=json_mode,
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
        litellm_params: dict,
        provider_config: BaseConfig,
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
            if error_response and hasattr(error_response, "text"):
                error_text = getattr(error_response, "text", error_text)
            raise AnthropicError(
                message=error_text,
                status_code=status_code,
                headers=error_headers,
            )

        return provider_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        acompletion=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        optional_params = copy.deepcopy(optional_params)
        stream = optional_params.pop("stream", None)
        json_mode: bool = optional_params.pop("json_mode", False)
        is_vertex_request: bool = optional_params.pop("is_vertex_request", False)
        _is_function_call = False
        messages = copy.deepcopy(messages)
        headers = AnthropicConfig().validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params={**optional_params, "is_vertex_request": is_vertex_request},
            litellm_params=litellm_params,
        )

        config = ProviderConfigManager.get_provider_chat_config(
            model=model,
            provider=LlmProviders(custom_llm_provider),
        )
        if config is None:
            raise ValueError(
                f"Provider config not found for model: {model} and provider: {custom_llm_provider}"
            )

        data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
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
                    json_mode=json_mode,
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
                    provider_config=config,
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
                    json_mode=json_mode,
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

        return config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def embedding(self):
        # logic for parsing in - calling - parsing out model embedding calls
        pass


class ModelResponseIterator:
    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.content_blocks: List[ContentBlockDelta] = []
        self.tool_index = -1
        self.json_mode = json_mode

    def check_empty_tool_call_args(self) -> bool:
        """
        Check if the tool call block so far has been an empty string
        """
        args = ""
        # if text content block -> skip
        if len(self.content_blocks) == 0:
            return False

        if (
            self.content_blocks[0]["delta"]["type"] == "text_delta"
            or self.content_blocks[0]["delta"]["type"] == "thinking_delta"
        ):
            return False

        for block in self.content_blocks:
            if block["delta"]["type"] == "input_json_delta":
                args += block["delta"].get("partial_json", "")  # type: ignore

        if len(args) == 0:
            return True
        return False

    def _handle_usage(self, anthropic_usage_chunk: Union[dict, UsageDelta]) -> Usage:
        return AnthropicConfig().calculate_usage(
            usage_object=cast(dict, anthropic_usage_chunk), reasoning_content=None
        )

    def _content_block_delta_helper(
        self, chunk: dict
    ) -> Tuple[
        str,
        Optional[ChatCompletionToolCallChunk],
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]],
        Dict[str, Any],
    ]:
        """
        Helper function to handle the content block delta
        """
        text = ""
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        provider_specific_fields = {}
        content_block = ContentBlockDelta(**chunk)  # type: ignore
        thinking_blocks: List[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ] = []

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
        elif "citation" in content_block["delta"]:
            provider_specific_fields["citation"] = content_block["delta"]["citation"]
        elif (
            "thinking" in content_block["delta"]
            or "signature" in content_block["delta"]
        ):
            thinking_blocks = [
                ChatCompletionThinkingBlock(
                    type="thinking",
                    thinking=content_block["delta"].get("thinking") or "",
                    signature=content_block["delta"].get("signature"),
                )
            ]
            provider_specific_fields["thinking_blocks"] = thinking_blocks

        return text, tool_use, thinking_blocks, provider_specific_fields

    def _handle_reasoning_content(
        self,
        thinking_blocks: List[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ],
    ) -> Optional[str]:
        """
        Handle the reasoning content
        """
        reasoning_content = None
        for block in thinking_blocks:
            thinking_content = cast(Optional[str], block.get("thinking"))
            if reasoning_content is None:
                reasoning_content = ""
            if thinking_content is not None:
                reasoning_content += thinking_content
        return reasoning_content

    def _handle_redacted_thinking_content(
        self,
        content_block_start: ContentBlockStart,
        provider_specific_fields: Dict[str, Any],
    ) -> Tuple[List[ChatCompletionRedactedThinkingBlock], Dict[str, Any]]:
        """
        Handle the redacted thinking content
        """
        thinking_blocks = [
            ChatCompletionRedactedThinkingBlock(
                type="redacted_thinking",
                data=content_block_start["content_block"]["data"],  # type: ignore
            )
        ]
        provider_specific_fields["thinking_blocks"] = thinking_blocks

        return thinking_blocks, provider_specific_fields

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            type_chunk = chunk.get("type", "") or ""

            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            finish_reason = ""
            usage: Optional[Usage] = None
            provider_specific_fields: Dict[str, Any] = {}
            reasoning_content: Optional[str] = None
            thinking_blocks: Optional[
                List[
                    Union[
                        ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock
                    ]
                ]
            ] = None

            index = int(chunk.get("index", 0))
            if type_chunk == "content_block_delta":
                """
                Anthropic content chunk
                chunk = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello'}}
                """
                (
                    text,
                    tool_use,
                    thinking_blocks,
                    provider_specific_fields,
                ) = self._content_block_delta_helper(chunk=chunk)
                if thinking_blocks:
                    reasoning_content = self._handle_reasoning_content(
                        thinking_blocks=thinking_blocks
                    )
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
                elif (
                    content_block_start["content_block"]["type"] == "redacted_thinking"
                ):
                    (
                        thinking_blocks,
                        provider_specific_fields,
                    ) = self._handle_redacted_thinking_content(  # type: ignore
                        content_block_start=content_block_start,
                        provider_specific_fields=provider_specific_fields,
                    )
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

            text, tool_use = self._handle_json_mode_chunk(text=text, tool_use=tool_use)

            returned_chunk = ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=index,
                        delta=Delta(
                            content=text,
                            tool_calls=[tool_use] if tool_use is not None else None,
                            provider_specific_fields=(
                                provider_specific_fields
                                if provider_specific_fields
                                else None
                            ),
                            thinking_blocks=(
                                thinking_blocks if thinking_blocks else None
                            ),
                            reasoning_content=reasoning_content,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
            )

            return returned_chunk

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    def _handle_json_mode_chunk(
        self, text: str, tool_use: Optional[ChatCompletionToolCallChunk]
    ) -> Tuple[str, Optional[ChatCompletionToolCallChunk]]:
        """
        If JSON mode is enabled, convert the tool call to a message.

        Anthropic returns the JSON schema as part of the tool call
        OpenAI returns the JSON schema as part of the content, this handles placing it in the content

        Args:
            text: str
            tool_use: Optional[ChatCompletionToolCallChunk]
        Returns:
            Tuple[str, Optional[ChatCompletionToolCallChunk]]

            text: The text to use in the content
            tool_use: The ChatCompletionToolCallChunk to use in the chunk response
        """
        if self.json_mode is True and tool_use is not None:
            message = AnthropicConfig._convert_tool_response_to_message(
                tool_calls=[tool_use]
            )
            if message is not None:
                text = message.content or ""
                tool_use = None

        return text, tool_use

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

    def convert_str_chunk_to_generic_chunk(self, chunk: str) -> ModelResponseStream:
        """
        Convert a string chunk to a GenericStreamingChunk

        Note: This is used for Anthropic pass through streaming logging

        We can move __anext__, and __next__ to use this function since it's common logic.
        Did not migrate them to minmize changes made in 1 PR.
        """
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
            return ModelResponseStream()
