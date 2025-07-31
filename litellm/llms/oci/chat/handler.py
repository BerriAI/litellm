"""
Calling + translation logic for anthropic's `/v1/messages` endpoint
"""

import copy
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx  # type: ignore

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.core_helpers import map_finish_reason
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
    LlmProviders,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from ...base import BaseLLM
from ..common_utils import OCIError
from .transformation import OCIChatConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.llms.base_llm.chat.transformation import BaseConfig


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
    signed_json_body: Optional[bytes] = None,
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_aclient

    try:
        response = await client.post(
            api_base,
            headers=headers,
            data=(
                signed_json_body
                if signed_json_body is not None
                else data
            ),
            stream=True,
            timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise OCIError(
            status_code=e.response.status_code,
            message=(await e.response.aread()).decode(),
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise OCIError(status_code=500, message=str(e))

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
    signed_json_body: Optional[bytes] = None,
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_client  # re-use a module level client

    try:
        response = client.post(
            api_base,
            headers=headers,
            data=(
                signed_json_body
                if signed_json_body is not None
                else data
            ),
            stream=True,
            timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise OCIError(
            status_code=e.response.status_code,
            message=e.response.read().decode(),
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise OCIError(status_code=500, message=str(e))

    if response.status_code != 200:
        response_headers = getattr(response, "headers", None)
        raise OCIError(
            status_code=response.status_code,
            message=response.read().decode(),
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


class OCIChatCompletion(BaseLLM):
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
        signed_json_body: Optional[bytes] = None,
    ):
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

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
            signed_json_body=signed_json_body,
        )
        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="anthropic",
            logging_obj=logging_obj,
            # _response_headers=process_anthropic_headers(headers),
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
        provider_config: "BaseConfig",
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> Union[ModelResponse, "CustomStreamWrapper"]:
        async_handler = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.ANTHROPIC
        )

        try:
            response = await async_handler.post(
                api_base,
                headers=headers,
                json=(
                    json.loads(signed_json_body.decode())  # Decode bytes to dict
                    if signed_json_body is not None
                    else data
                ),
                timeout=timeout
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
            raise OCIError(
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
        fake_stream: bool = False,
        acompletion=None,
        logger_fn=None,
        headers={},
        client=None,
        stream: Optional[bool] = False,
    ):
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
        from litellm.utils import ProviderConfigManager

        oci_region = optional_params.get(
            "oci_region", "us-ashburn-1"
        )
        api_base = (
            api_base
            or litellm.api_base
            or f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com/20231130/actions/chat"
        )

        optional_params = copy.deepcopy(optional_params)
        json_mode: bool = optional_params.pop("json_mode", False)
        _is_function_call = False
        messages = copy.deepcopy(messages)
        headers = OCIChatConfig().validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
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

        fake_stream = (
            fake_stream
            or optional_params.pop("fake_stream", False)
            or config.should_fake_stream(
                model=model, custom_llm_provider=custom_llm_provider, stream=stream
            )
        )

        headers, signed_json_body = config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=data,
            api_base=api_base,
            api_key=api_key,
            stream=stream,
            fake_stream=fake_stream,
            model=model,
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
                    signed_json_body=signed_json_body,
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
                    signed_json_body=signed_json_body,
                )
        else:
            ## COMPLETION CALL
            if (
                stream is True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
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
                    signed_json_body=signed_json_body,
                )
                return completion_stream
                # return OCICustomStreamWrapper(
                #     completion_stream=completion_stream,
                #     model=model,
                #     custom_llm_provider="oci",
                #     logging_obj=logging_obj,
                # )

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
                    raise OCIError(
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

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            text: str = ""
            tool_calls = []
            message = chunk.get("message")
            finish_reason = chunk.get("finishReason")
            if finish_reason:
                text = ""
            else:
                if not message:
                    raise ValueError(
                        "Chunk does not contain 'message' key: {}".format(chunk)
                    )
                content = message.get("content")
                if content:
                    text = content[0].get("text", "")
                else:
                    text = ""
                tool_calls_received = message.get("toolCalls", [])
                for tool in tool_calls_received:
                    tool_calls.append({
                        "id": tool.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "arguments": tool.get("arguments", "{}"),
                        },
                    })
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
            returned_chunk = ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=index,
                        delta=Delta(
                            content=text,
                            tool_calls=tool_calls if tool_calls else None,
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
                return next(self)
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
