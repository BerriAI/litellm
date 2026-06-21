"""
Bedrock Invoke streaming dispatch (make_call/make_sync_call) and the event
stream decoders live here. The request/response transforms moved to
`litellm/llms/bedrock/chat/invoke_transformations/base_invoke_transformation.py`.
"""

import types
from typing import (
    AsyncIterator,
    Iterator,
    Optional,
    Tuple,
    cast,
)

import httpx  # type: ignore

import litellm
from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.caching.caching import InMemoryCache
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.bedrock import *
from litellm.types.llms.bedrock_invoke import assert_no_control_params_in_payload
from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
)
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import (
    BedrockError,
    get_bedrock_response_stream_shape,
    get_bedrock_tool_name,
)

bedrock_tool_name_mappings: InMemoryCache = InMemoryCache(
    max_size_in_memory=50, default_ttl=600
)
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

converse_config = AmazonConverseConfig()


class AmazonCohereChatConfig:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
    """

    documents: Optional[List[Document]] = None
    search_queries_only: Optional[bool] = None
    preamble: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    p: Optional[float] = None
    k: Optional[float] = None
    prompt_truncation: Optional[str] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    return_prompt: Optional[bool] = None
    stop_sequences: Optional[List[str]] = None
    raw_prompting: Optional[bool] = None

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        search_queries_only: Optional[bool] = None,
        preamble: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        p: Optional[float] = None,
        k: Optional[float] = None,
        prompt_truncation: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        return_prompt: Optional[bool] = None,
        stop_sequences: Optional[str] = None,
        raw_prompting: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
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

    def get_supported_openai_params(self) -> List[str]:
        return [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stop",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "tools",
            "tool_choice",
        ]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                if isinstance(value, str):
                    value = [value]
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["p"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if "seed":
                optional_params["seed"] = value
        return optional_params


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj: Logging,
    fake_stream: bool = False,
    json_mode: Optional[bool] = False,
    bedrock_invoke_provider: Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL] = None,
    stream_chunk_size: Optional[int] = None,
):
    assert_no_control_params_in_payload(data)
    try:
        if client is None:
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.BEDROCK,
                params=(
                    {"ssl_verify": logging_obj.litellm_params.get("ssl_verify")}
                    if logging_obj
                    and logging_obj.litellm_params
                    and logging_obj.litellm_params.get("ssl_verify")
                    else None
                ),
            )  # Create a new client if none provided

        response = await client.post(
            api_base,
            headers=headers,
            data=data,
            stream=not fake_stream,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise BedrockError(status_code=response.status_code, message=response.text)

        if fake_stream:
            model_response: (
                ModelResponse
            ) = litellm.AmazonConverseConfig()._transform_response(
                model=model,
                response=response,
                model_response=litellm.ModelResponse(),
                stream=True,
                logging_obj=logging_obj,
                optional_params={},
                api_key="",
                data=data,
                messages=messages,
                encoding=litellm.encoding,
            )  # type: ignore
            completion_stream: Any = MockResponseIterator(
                model_response=model_response, json_mode=json_mode
            )
        elif bedrock_invoke_provider == "anthropic":
            decoder: AWSEventStreamDecoder = AmazonAnthropicClaudeStreamDecoder(
                model=model,
                sync_stream=False,
                json_mode=json_mode,
            )
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=stream_chunk_size)
            )
        elif bedrock_invoke_provider == "deepseek_r1":
            decoder = AmazonDeepSeekR1StreamDecoder(
                model=model,
                sync_stream=False,
            )
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=stream_chunk_size)
            )
        else:
            decoder = AWSEventStreamDecoder(model=model, json_mode=json_mode)
            completion_stream = decoder.aiter_bytes(
                response.aiter_bytes(chunk_size=stream_chunk_size)
            )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return completion_stream
    except httpx.HTTPStatusError as err:
        error_code = err.response.status_code
        raise BedrockError(status_code=error_code, message=err.response.text)
    except httpx.TimeoutException:
        raise BedrockError(status_code=408, message="Timeout error occurred.")
    except Exception as e:
        raise BedrockError(status_code=500, message=str(e))


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    signed_json_body: Optional[bytes],
    model: str,
    messages: list,
    logging_obj: Logging,
    fake_stream: bool = False,
    json_mode: Optional[bool] = False,
    bedrock_invoke_provider: Optional[litellm.BEDROCK_INVOKE_PROVIDERS_LITERAL] = None,
    stream_chunk_size: Optional[int] = None,
):
    assert_no_control_params_in_payload(data)
    try:
        if client is None:
            client = _get_httpx_client(
                params=(
                    {"ssl_verify": logging_obj.litellm_params.get("ssl_verify")}
                    if logging_obj
                    and logging_obj.litellm_params
                    and logging_obj.litellm_params.get("ssl_verify")
                    else None
                )
            )

        response = client.post(
            api_base,
            headers=headers,
            data=signed_json_body if signed_json_body is not None else data,
            stream=not fake_stream,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise BedrockError(status_code=response.status_code, message=response.text)

        if fake_stream:
            model_response: (
                ModelResponse
            ) = litellm.AmazonConverseConfig()._transform_response(
                model=model,
                response=response,
                model_response=litellm.ModelResponse(),
                stream=True,
                logging_obj=logging_obj,
                optional_params={},
                api_key="",
                data=data,
                messages=messages,
                encoding=litellm.encoding,
            )  # type: ignore
            completion_stream: Any = MockResponseIterator(
                model_response=model_response, json_mode=json_mode
            )
        elif bedrock_invoke_provider == "anthropic":
            decoder: AWSEventStreamDecoder = AmazonAnthropicClaudeStreamDecoder(
                model=model,
                sync_stream=True,
                json_mode=json_mode,
            )
            completion_stream = decoder.iter_bytes(
                response.iter_bytes(chunk_size=stream_chunk_size)
            )
        elif bedrock_invoke_provider == "deepseek_r1":
            decoder = AmazonDeepSeekR1StreamDecoder(
                model=model,
                sync_stream=True,
            )
            completion_stream = decoder.iter_bytes(
                response.iter_bytes(chunk_size=stream_chunk_size)
            )
        else:
            decoder = AWSEventStreamDecoder(model=model, json_mode=json_mode)
            completion_stream = decoder.iter_bytes(
                response.iter_bytes(chunk_size=stream_chunk_size)
            )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return completion_stream
    except httpx.HTTPStatusError as err:
        error_code = err.response.status_code
        raise BedrockError(status_code=error_code, message=err.response.text)
    except httpx.TimeoutException:
        raise BedrockError(status_code=408, message="Timeout error occurred.")
    except Exception as e:
        raise BedrockError(status_code=500, message=str(e))


class BedrockLLM(BaseAWSLLM):
    """Legacy shell retained only for ``get_bedrock_invoke_provider``, inherited
    from BaseAWSLLM and still referenced by name. Invoke requests now run through
    AmazonInvokeConfig and base_llm_http_handler."""

    def __init__(self) -> None:
        super().__init__()


class AWSEventStreamDecoder:
    def __init__(self, model: str, json_mode: Optional[bool] = False) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.model = model
        self.parser = EventStreamJSONParser()
        self.content_blocks: List[ContentBlockDeltaEvent] = []
        self.tool_calls_index: Optional[int] = None
        self.response_id: Optional[str] = None
        self.json_mode = json_mode
        self._current_tool_name: Optional[str] = None

    def check_empty_tool_call_args(self) -> bool:
        """
        Check if the tool call block so far has been an empty string
        """
        args = ""
        # if text content block -> skip
        if len(self.content_blocks) == 0:
            return False

        if (
            "toolUse" not in self.content_blocks[0]
        ):  # be explicit - only do this if tool use block, as this is to prevent json decoding errors
            return False

        for block in self.content_blocks:
            if "toolUse" in block:
                args += block["toolUse"]["input"]

        if len(args) == 0:
            return True
        return False

    def extract_reasoning_content_str(
        self, reasoning_content_block: BedrockConverseReasoningContentBlockDelta
    ) -> Optional[str]:
        if "text" in reasoning_content_block:
            return reasoning_content_block["text"]
        return None

    def translate_thinking_blocks(
        self, thinking_block: BedrockConverseReasoningContentBlockDelta
    ) -> Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ]:
        """
        Translate the thinking blocks to a string
        """

        thinking_blocks_list: List[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ] = []
        _thinking_block: Optional[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ] = None

        if "text" in thinking_block:
            _thinking_block = ChatCompletionThinkingBlock(type="thinking")
            _thinking_block["thinking"] = thinking_block["text"]
        elif "signature" in thinking_block:
            _thinking_block = ChatCompletionThinkingBlock(type="thinking")
            _thinking_block["signature"] = thinking_block["signature"]
            _thinking_block["thinking"] = ""  # consistent with anthropic response
        elif "redactedContent" in thinking_block:
            _thinking_block = ChatCompletionRedactedThinkingBlock(
                type="redacted_thinking", data=thinking_block["redactedContent"]
            )
        if _thinking_block is not None:
            thinking_blocks_list.append(_thinking_block)
        return thinking_blocks_list

    def _initialize_converse_response_id(self, chunk_data: dict):
        """Initialize response_id from chunk data if not already set."""
        if self.response_id is None:
            if "messageStart" in chunk_data:
                conversation_id = chunk_data["messageStart"].get("conversationId")
                if conversation_id:
                    self.response_id = f"chatcmpl-{conversation_id}"
            else:
                # Fallback to generating a UUID if the first chunk is not messageStart
                self.response_id = f"chatcmpl-{uuid.uuid4()}"

    def _handle_converse_start_event(
        self,
        start_obj: ContentBlockStartEvent,
    ) -> Tuple[
        Optional[ChatCompletionToolCallChunk],
        dict,
        Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ],
    ]:
        """Handle 'start' event in converse chunk parsing."""
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        provider_specific_fields: dict = {}
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ] = None

        self.content_blocks = []  # reset
        if start_obj is not None:
            if "toolUse" in start_obj and start_obj["toolUse"] is not None:
                ## check tool name was formatted by litellm
                _response_tool_name = start_obj["toolUse"]["name"]
                response_tool_name = get_bedrock_tool_name(
                    response_tool_name=_response_tool_name
                )
                self._current_tool_name = response_tool_name

                # When json_mode is True, suppress the internal json_tool_call
                # and convert its content to text in delta events instead
                if (
                    self.json_mode is True
                    and response_tool_name == RESPONSE_FORMAT_TOOL_NAME
                ):
                    return tool_use, provider_specific_fields, thinking_blocks

                self.tool_calls_index = (
                    0 if self.tool_calls_index is None else self.tool_calls_index + 1
                )
                tool_use = {
                    "id": start_obj["toolUse"]["toolUseId"],
                    "type": "function",
                    "function": {
                        "name": response_tool_name,
                        "arguments": "",
                    },
                    "index": self.tool_calls_index,
                }
            elif (
                "reasoningContent" in start_obj
                and start_obj["reasoningContent"] is not None
            ):  # redacted thinking can be in start object
                thinking_blocks = self.translate_thinking_blocks(
                    start_obj["reasoningContent"]
                )
                provider_specific_fields = {
                    "reasoningContent": start_obj["reasoningContent"],
                }
        return tool_use, provider_specific_fields, thinking_blocks

    def _handle_converse_delta_event(
        self,
        delta_obj: ContentBlockDeltaEvent,
        index: int,
    ) -> Tuple[
        str,
        Optional[ChatCompletionToolCallChunk],
        dict,
        Optional[str],
        Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ],
    ]:
        """Handle 'delta' event in converse chunk parsing."""
        text = ""
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        provider_specific_fields: dict = {}
        reasoning_content: Optional[str] = None
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ] = None

        self.content_blocks.append(delta_obj)
        if "text" in delta_obj:
            text = delta_obj["text"]
        elif "toolUse" in delta_obj:
            # When json_mode is True and this is the internal json_tool_call,
            # convert tool input to text content instead of tool call arguments
            if (
                self.json_mode is True
                and self._current_tool_name == RESPONSE_FORMAT_TOOL_NAME
            ):
                text = delta_obj["toolUse"]["input"]
            else:
                tool_use = {
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": None,
                        "arguments": delta_obj["toolUse"]["input"],
                    },
                    "index": (
                        self.tool_calls_index
                        if self.tool_calls_index is not None
                        else index
                    ),
                }
        elif "reasoningContent" in delta_obj:
            provider_specific_fields = {
                "reasoningContent": delta_obj["reasoningContent"],
            }
            reasoning_content = self.extract_reasoning_content_str(
                delta_obj["reasoningContent"]
            )
            thinking_blocks = self.translate_thinking_blocks(
                delta_obj["reasoningContent"]
            )
            if (
                thinking_blocks
                and len(thinking_blocks) > 0
                and reasoning_content is None
            ):
                reasoning_content = (
                    ""  # set to non-empty string to ensure consistency with Anthropic
                )
        elif "citationsContent" in delta_obj:
            # Handle Nova grounding citations in streaming responses
            provider_specific_fields = {
                "citationsContent": delta_obj["citationsContent"],
            }
        return (
            text,
            tool_use,
            provider_specific_fields,
            reasoning_content,
            thinking_blocks,
        )

    def _handle_converse_stop_event(
        self, index: int
    ) -> Optional[ChatCompletionToolCallChunk]:
        """Handle stop/contentBlockIndex event in converse chunk parsing."""
        tool_use: Optional[ChatCompletionToolCallChunk] = None

        # If the ending block was the internal json_tool_call, skip emitting
        # the empty-args tool chunk and reset tracking state
        if (
            self.json_mode is True
            and self._current_tool_name == RESPONSE_FORMAT_TOOL_NAME
        ):
            self._current_tool_name = None
            return tool_use

        self._current_tool_name = None
        is_empty = self.check_empty_tool_call_args()
        if is_empty:
            tool_use = {
                "id": None,
                "type": "function",
                "function": {
                    "name": None,
                    "arguments": "{}",
                },
                "index": (
                    self.tool_calls_index
                    if self.tool_calls_index is not None
                    else index
                ),
            }
        return tool_use

    def converse_chunk_parser(self, chunk_data: dict) -> ModelResponseStream:
        try:
            # Capture the conversationId from the first messageStart event
            # and use it as the consistent ID for all subsequent chunks.
            self._initialize_converse_response_id(chunk_data)

            verbose_logger.debug("\n\nRaw Chunk: {}\n\n".format(chunk_data))
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            finish_reason = ""
            usage: Optional[Usage] = None
            provider_specific_fields: dict = {}
            reasoning_content: Optional[str] = None
            thinking_blocks: Optional[
                List[
                    Union[
                        ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock
                    ]
                ]
            ] = None

            content_block_index = int(chunk_data.get("contentBlockIndex", 0))
            if "start" in chunk_data:
                start_obj = ContentBlockStartEvent(**chunk_data["start"])
                (
                    tool_use,
                    provider_specific_fields,
                    thinking_blocks,
                ) = self._handle_converse_start_event(start_obj)
            elif "delta" in chunk_data:
                delta_obj = ContentBlockDeltaEvent(**chunk_data["delta"])
                (
                    text,
                    tool_use,
                    provider_specific_fields,
                    reasoning_content,
                    thinking_blocks,
                ) = self._handle_converse_delta_event(delta_obj, content_block_index)
            elif (
                "contentBlockIndex" in chunk_data
            ):  # stop block, no 'start' or 'delta' object
                tool_use = self._handle_converse_stop_event(content_block_index)
            elif "stopReason" in chunk_data:
                finish_reason = map_finish_reason(chunk_data.get("stopReason", "stop"))
            elif "usage" in chunk_data:
                usage = converse_config._transform_usage(chunk_data.get("usage", {}))

            model_response_provider_specific_fields = {}
            if "trace" in chunk_data:
                trace = chunk_data.get("trace")
                model_response_provider_specific_fields["trace"] = trace
            response = ModelResponseStream(
                choices=[
                    StreamingChoices(
                        finish_reason=finish_reason,
                        index=0,  # Always 0 - Bedrock never returns multiple choices
                        delta=Delta(
                            content=text,
                            role="assistant",
                            tool_calls=[tool_use] if tool_use else None,
                            provider_specific_fields=(
                                provider_specific_fields
                                if provider_specific_fields
                                else None
                            ),
                            thinking_blocks=thinking_blocks,
                            reasoning_content=reasoning_content,
                        ),
                    )
                ],
                id=self.response_id,
                model=self.model,
                usage=usage,
                provider_specific_fields=model_response_provider_specific_fields,
            )

            return response
        except Exception as e:
            raise Exception("Received streaming error - {}".format(str(e)))

    def _chunk_parser(
        self, chunk_data: dict
    ) -> Union[GChunk, ModelResponseStream, dict]:
        text = ""
        is_finished = False
        finish_reason = ""
        if "outputText" in chunk_data:
            text = chunk_data["outputText"]
        # ai21 mapping
        elif "ai21" in self.model:  # fake ai21 streaming
            text = chunk_data.get("completions")[0].get("data").get("text")  # type: ignore
            is_finished = True
            finish_reason = "stop"
        ######## /bedrock/converse mappings ###############
        elif (
            "contentBlockIndex" in chunk_data
            or "stopReason" in chunk_data
            or "metrics" in chunk_data
            or "trace" in chunk_data
        ):
            return self.converse_chunk_parser(chunk_data=chunk_data)
        ######### /bedrock/invoke nova mappings ###############
        elif "contentBlockDelta" in chunk_data:
            # when using /bedrock/invoke/nova, the chunk_data is nested under "contentBlockDelta"
            _chunk_data = chunk_data.get("contentBlockDelta", {})
            return self.converse_chunk_parser(chunk_data=_chunk_data)
        ######## bedrock.mistral mappings ###############
        elif "outputs" in chunk_data:
            if (
                len(chunk_data["outputs"]) == 1
                and chunk_data["outputs"][0].get("text", None) is not None
            ):
                text = chunk_data["outputs"][0]["text"]
            stop_reason = chunk_data.get("stop_reason", None)
            if stop_reason is not None:
                is_finished = True
                finish_reason = stop_reason
        ######## bedrock.cohere mappings ###############
        # meta mapping
        elif "generation" in chunk_data:
            text = chunk_data["generation"]  # bedrock.meta
        # cohere mapping
        elif "text" in chunk_data:
            text = chunk_data["text"]  # bedrock.cohere
        # cohere mapping for finish reason
        elif "finish_reason" in chunk_data:
            finish_reason = chunk_data["finish_reason"]
            is_finished = True
        elif chunk_data.get("completionReason", None):
            is_finished = True
            finish_reason = chunk_data["completionReason"]
        return GChunk(
            text=text,
            is_finished=is_finished,
            finish_reason=finish_reason,
            usage=None,
            index=0,
            tool_use=None,
        )

    def iter_bytes(
        self, iterator: Iterator[bytes]
    ) -> Iterator[Union[GChunk, ModelResponseStream, dict]]:
        """Given an iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    # sse_event = ServerSentEvent(data=message, event="completion")
                    _data = json.loads(message)
                    yield self._chunk_parser(chunk_data=_data)

    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[Union[GChunk, ModelResponseStream, dict]]:
        """Given an async iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        async for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    _data = json.loads(message)
                    yield self._chunk_parser(chunk_data=_data)

    def _parse_message_from_event(self, event) -> Optional[str]:
        response_stream_shape = get_bedrock_response_stream_shape()
        if response_stream_shape is None:
            raise BedrockError(
                status_code=500,
                message=(
                    "Bedrock event-stream shape could not be loaded from botocore. "
                    "Ensure botocore is correctly installed."
                ),
            )
        response_dict = event.to_response_dict()
        parsed_response = self.parser.parse(response_dict, response_stream_shape)

        if response_dict["status_code"] != 200:
            decoded_body = response_dict["body"].decode()
            if isinstance(decoded_body, dict):
                error_message = decoded_body.get("message")
            elif isinstance(decoded_body, str):
                error_message = decoded_body
            else:
                error_message = ""
            exception_status = response_dict["headers"].get(":exception-type")
            error_message = exception_status + " " + error_message
            raise BedrockError(
                status_code=response_dict["status_code"],
                message=(
                    json.dumps(error_message)
                    if isinstance(error_message, dict)
                    else error_message
                ),
            )
        if "chunk" in parsed_response:
            chunk = parsed_response.get("chunk")
            if not chunk:
                return None
            return chunk.get("bytes").decode()  # type: ignore[no-any-return]
        else:
            chunk = response_dict.get("body")
            if not chunk:
                return None

            return chunk.decode()  # type: ignore[no-any-return]


class AmazonAnthropicClaudeStreamDecoder(AWSEventStreamDecoder):
    def __init__(
        self,
        model: str,
        sync_stream: bool,
        json_mode: Optional[bool] = None,
    ) -> None:
        """
        Child class of AWSEventStreamDecoder that handles the streaming response from the Anthropic family of models

        The only difference between AWSEventStreamDecoder and AmazonAnthropicClaudeStreamDecoder is the `chunk_parser` method
        """
        super().__init__(model=model)
        self.anthropic_model_response_iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def _chunk_parser(self, chunk_data: dict) -> ModelResponseStream:
        return self.anthropic_model_response_iterator.chunk_parser(chunk=chunk_data)


class AmazonDeepSeekR1StreamDecoder(AWSEventStreamDecoder):
    def __init__(
        self,
        model: str,
        sync_stream: bool,
    ) -> None:
        super().__init__(model=model)
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation import (
            AmazonDeepseekR1ResponseIterator,
        )

        self.deepseek_model_response_iterator = AmazonDeepseekR1ResponseIterator(
            streaming_response=None,
            sync_stream=sync_stream,
        )

    def _chunk_parser(
        self, chunk_data: dict
    ) -> Union[GChunk, ModelResponseStream, dict]:
        return self.deepseek_model_response_iterator.chunk_parser(chunk=chunk_data)


class MockResponseIterator:  # for returning ai21 streaming responses
    def __init__(self, model_response, json_mode: Optional[bool] = False):
        self.model_response = model_response
        self.json_mode = json_mode
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def _handle_json_mode_chunk(
        self, text: str, tool_calls: Optional[List[ChatCompletionToolCallChunk]]
    ) -> Tuple[str, Optional[ChatCompletionToolCallChunk]]:
        """
        If JSON mode is enabled, convert the tool call to a message.

        Bedrock returns the JSON schema as part of the tool call
        OpenAI returns the JSON schema as part of the content, this handles placing it in the content

        Args:
            text: str
            tool_use: Optional[ChatCompletionToolCallChunk]
        Returns:
            Tuple[str, Optional[ChatCompletionToolCallChunk]]

            text: The text to use in the content
            tool_use: The ChatCompletionToolCallChunk to use in the chunk response
        """
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        if self.json_mode is True and tool_calls is not None:
            message = litellm.AnthropicConfig()._convert_tool_response_to_message(
                tool_calls=tool_calls
            )
            if message is not None:
                text = message.content or ""
                tool_use = None
        elif tool_calls is not None and len(tool_calls) > 0:
            tool_use = tool_calls[0]
        return text, tool_use

    def _chunk_parser(self, chunk_data: ModelResponse) -> GChunk:
        try:
            chunk_usage: Usage = getattr(chunk_data, "usage")
            text = chunk_data.choices[0].message.content or ""  # type: ignore
            tool_use = None
            _model_response_tool_call = cast(
                Optional[List[ChatCompletionMessageToolCall]],
                cast(Choices, chunk_data.choices[0]).message.tool_calls,
            )
            if self.json_mode is True:
                text, tool_use = self._handle_json_mode_chunk(
                    text=text,
                    tool_calls=chunk_data.choices[0].message.tool_calls,  # type: ignore
                )
            elif _model_response_tool_call is not None:
                tool_use = ChatCompletionToolCallChunk(
                    id=_model_response_tool_call[0].id,
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=_model_response_tool_call[0].function.name,
                        arguments=_model_response_tool_call[0].function.arguments,
                    ),
                    index=0,
                )
            processed_chunk = GChunk(
                text=text,
                tool_use=tool_use,
                is_finished=True,
                finish_reason=map_finish_reason(
                    finish_reason=chunk_data.choices[0].finish_reason or ""
                ),
                usage=ChatCompletionUsageBlock(
                    prompt_tokens=chunk_usage.prompt_tokens,
                    completion_tokens=chunk_usage.completion_tokens,
                    total_tokens=chunk_usage.total_tokens,
                ),
                index=0,
            )
            return processed_chunk
        except Exception as e:
            raise ValueError(f"Failed to decode chunk: {chunk_data}. Error: {e}")

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self._chunk_parser(self.model_response)

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self._chunk_parser(self.model_response)
