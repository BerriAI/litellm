import asyncio
import json
import re
import time
import traceback
import uuid
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_logger
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.types.llms.databricks import DatabricksTool
from litellm.types.llms.openai import (
    ChatCompletionThinkingBlock,
    OpenAIModerationResponse,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    ChatCompletionRedactedThinkingBlock,
    Choices,
    Delta,
    EmbeddingResponse,
    Function,
    HiddenParams,
    ImageResponse,
)
from litellm.types.utils import Logprobs as TextCompletionLogprobs
from litellm.types.utils import (
    Message,
    ModelResponse,
    RerankResponse,
    StreamingChoices,
    TextChoices,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
)

from .get_headers import get_response_headers


def _safe_convert_created_field(created_value) -> int:
    """
    Safely convert a 'created' field value to an integer.
    
    Some providers (like SambaNova) return the 'created' field as a float 
    (Unix timestamp with fractional seconds), but LiteLLM expects an integer.
    
    Args:
        created_value: The value from response_object["created"]
        
    Returns:
        int: Unix timestamp as integer
    """
    if created_value is None:
        return int(time.time())
    elif isinstance(created_value, int):
        return created_value
    elif isinstance(created_value, float):
        return int(created_value)
    else:
        # for strings, etc
        try:
            return int(float(created_value))
        except (ValueError, TypeError):
            # Fallback to current time if conversion fails
            return int(time.time())


def convert_tool_call_to_json_mode(
    tool_calls: List[ChatCompletionMessageToolCall],
    convert_tool_call_to_json_mode: bool,
) -> Tuple[Optional[Message], Optional[str]]:
    if _should_convert_tool_call_to_json_mode(
        tool_calls=tool_calls,
        convert_tool_call_to_json_mode=convert_tool_call_to_json_mode,
    ):
        # to support 'json_schema' logic on older models
        json_mode_content_str: Optional[str] = tool_calls[0]["function"].get(
            "arguments"
        )
        if json_mode_content_str is not None:
            message = litellm.Message(content=json_mode_content_str)
            finish_reason = "stop"
            return message, finish_reason
    return None, None


async def convert_to_streaming_response_async(response_object: Optional[dict] = None):
    """
    Asynchronously converts a response object to a streaming response.

    Args:
        response_object (Optional[dict]): The response object to be converted. Defaults to None.

    Raises:
        Exception: If the response object is None.

    Yields:
        ModelResponse: The converted streaming response object.

    Returns:
        None
    """
    if response_object is None:
        raise Exception("Error in response object format")

    model_response_object = ModelResponse(stream=True)

    if model_response_object is None:
        raise Exception("Error in response creating model response object")

    choice_list = []

    for idx, choice in enumerate(response_object["choices"]):
        if (
            choice["message"].get("tool_calls", None) is not None
            and isinstance(choice["message"]["tool_calls"], list)
            and len(choice["message"]["tool_calls"]) > 0
            and isinstance(choice["message"]["tool_calls"][0], dict)
        ):
            pydantic_tool_calls = []
            for index, t in enumerate(choice["message"]["tool_calls"]):
                if "index" not in t:
                    t["index"] = index
                pydantic_tool_calls.append(ChatCompletionDeltaToolCall(**t))
            choice["message"]["tool_calls"] = pydantic_tool_calls
        delta = Delta(
            content=choice["message"].get("content", None),
            role=choice["message"]["role"],
            function_call=choice["message"].get("function_call", None),
            tool_calls=choice["message"].get("tool_calls", None),
        )
        finish_reason = choice.get("finish_reason", None)

        if finish_reason is None:
            finish_reason = choice.get("finish_details")

        logprobs = choice.get("logprobs", None)

        choice = StreamingChoices(
            finish_reason=finish_reason, index=idx, delta=delta, logprobs=logprobs
        )
        choice_list.append(choice)

    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        setattr(
            model_response_object,
            "usage",
            Usage(
                completion_tokens=response_object["usage"].get("completion_tokens", 0),
                prompt_tokens=response_object["usage"].get("prompt_tokens", 0),
                total_tokens=response_object["usage"].get("total_tokens", 0),
            ),
        )

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "created" in response_object:
        model_response_object.created = _safe_convert_created_field(response_object["created"])

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]

    yield model_response_object
    await asyncio.sleep(0)


def convert_to_streaming_response(response_object: Optional[dict] = None):
    # used for yielding Cache hits when stream == True
    if response_object is None:
        raise Exception("Error in response object format")

    model_response_object = ModelResponse(stream=True)
    choice_list = []
    for idx, choice in enumerate(response_object["choices"]):
        delta = Delta(**choice["message"])
        finish_reason = choice.get("finish_reason", None)
        if finish_reason is None:
            # gpt-4 vision can return 'finish_reason' or 'finish_details'
            finish_reason = choice.get("finish_details")
        logprobs = choice.get("logprobs", None)
        enhancements = choice.get("enhancements", None)
        choice = StreamingChoices(
            finish_reason=finish_reason,
            index=idx,
            delta=delta,
            logprobs=logprobs,
            enhancements=enhancements,
        )

        choice_list.append(choice)
    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        setattr(model_response_object, "usage", Usage())
        model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
        model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
        model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "created" in response_object:
        model_response_object.created = _safe_convert_created_field(response_object["created"])

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]
    yield model_response_object


from collections import defaultdict


def _handle_invalid_parallel_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCall],
):
    """
    Handle hallucinated parallel tool call from openai - https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Code modified from: https://github.com/phdowling/openai_multi_tool_use_parallel_patch/blob/main/openai_multi_tool_use_parallel_patch.py
    """

    if tool_calls is None:
        return
    try:
        replacements: Dict[int, List[ChatCompletionMessageToolCall]] = defaultdict(list)
        for i, tool_call in enumerate(tool_calls):
            current_function = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            if current_function == "multi_tool_use.parallel":
                verbose_logger.debug(
                    "OpenAI did a weird pseudo-multi-tool-use call, fixing call structure.."
                )
                for _fake_i, _fake_tool_use in enumerate(function_args["tool_uses"]):
                    _function_args = _fake_tool_use["parameters"]
                    _current_function = _fake_tool_use["recipient_name"]
                    if _current_function.startswith("functions."):
                        _current_function = _current_function[len("functions.") :]

                    fixed_tc = ChatCompletionMessageToolCall(
                        id=f"{tool_call.id}_{_fake_i}",
                        type="function",
                        function=Function(
                            name=_current_function, arguments=json.dumps(_function_args)
                        ),
                    )
                    replacements[i].append(fixed_tc)

        shift = 0
        for i, replacement in replacements.items():
            tool_calls[:] = (
                tool_calls[: i + shift] + replacement + tool_calls[i + shift + 1 :]
            )
            shift += len(replacement)

        return tool_calls
    except json.JSONDecodeError:
        # if there is a JSONDecodeError, return the original tool_calls
        return tool_calls


def _parse_content_for_reasoning(
    message_text: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the content for reasoning

    Returns:
    - reasoning_content: The content of the reasoning
    - content: The content of the message
    """
    if not message_text:
        return None, message_text

    reasoning_match = re.match(
        r"<(?:think|thinking)>(.*?)</(?:think|thinking)>(.*)", message_text, re.DOTALL
    )

    if reasoning_match:
        return reasoning_match.group(1), reasoning_match.group(2)

    return None, message_text


def _extract_reasoning_content(message: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract reasoning content and main content from a message.

    Args:
        message (dict): The message dictionary that may contain reasoning_content

    Returns:
        tuple[Optional[str], Optional[str]]: A tuple of (reasoning_content, content)
    """
    message_content = message.get("content")
    if "reasoning_content" in message:
        return message["reasoning_content"], message["content"]
    elif "reasoning" in message:
        return message["reasoning"], message["content"]
    elif isinstance(message_content, str):
        return _parse_content_for_reasoning(message_content)
    return None, message_content


class LiteLLMResponseObjectHandler:
    @staticmethod
    def convert_to_image_response(
        response_object: dict,
        model_response_object: Optional[ImageResponse] = None,
        hidden_params: Optional[dict] = None,
    ) -> ImageResponse:
        response_object.update({"hidden_params": hidden_params})

        # Handle gpt-image-1 usage field with None values
        if "usage" in response_object and response_object["usage"] is not None:
            usage = response_object["usage"]
            # Check if usage fields are None and provide defaults
            if usage.get("input_tokens") is None:
                usage["input_tokens"] = 0
            if usage.get("output_tokens") is None:
                usage["output_tokens"] = 0
            if usage.get("total_tokens") is None:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
            if usage.get("input_tokens_details") is None:
                usage["input_tokens_details"] = {
                    "image_tokens": 0,
                    "text_tokens": 0,
                }

        if model_response_object is None:
            model_response_object = ImageResponse(**response_object)
            return model_response_object
        else:
            model_response_dict = model_response_object.model_dump()

            model_response_dict.update(response_object)
            model_response_object = ImageResponse(**model_response_dict)
            return model_response_object

    @staticmethod
    def convert_to_moderation_response(
        response_object: dict,
    ) -> OpenAIModerationResponse:
        return OpenAIModerationResponse(**response_object)

    @staticmethod
    def convert_chat_to_text_completion(
        response: ModelResponse,
        text_completion_response: TextCompletionResponse,
        custom_llm_provider: Optional[str] = None,
    ) -> TextCompletionResponse:
        """
        Converts a chat completion response to a text completion response format.

        Note: This is used for huggingface. For OpenAI / Azure Text the providers files directly return TextCompletionResponse which we then send to user

        Args:
            response (ModelResponse): The chat completion response to convert

        Returns:
            TextCompletionResponse: The converted text completion response

        Example:
            chat_response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi"}])
            text_response = convert_chat_to_text_completion(chat_response)
        """
        transformed_logprobs = LiteLLMResponseObjectHandler._convert_provider_response_logprobs_to_text_completion_logprobs(
            response=response,
            custom_llm_provider=custom_llm_provider,
        )

        text_completion_response["id"] = response.get("id", None)
        text_completion_response["object"] = "text_completion"
        text_completion_response["created"] = response.get("created", None)
        text_completion_response["model"] = response.get("model", None)
        choices_list: List[TextChoices] = []

        # Convert each choice to TextChoices
        for choice in response["choices"]:
            text_choices = TextChoices()
            text_choices["text"] = choice["message"]["content"]
            text_choices["index"] = choice["index"]
            text_choices["logprobs"] = transformed_logprobs
            text_choices["finish_reason"] = choice["finish_reason"]
            choices_list.append(text_choices)

        text_completion_response["choices"] = choices_list
        text_completion_response["usage"] = response.get("usage", None)
        text_completion_response._hidden_params = HiddenParams(
            **response._hidden_params
        )
        return text_completion_response

    @staticmethod
    def _convert_provider_response_logprobs_to_text_completion_logprobs(
        response: ModelResponse,
        custom_llm_provider: Optional[str] = None,
    ) -> Optional[TextCompletionLogprobs]:
        """
        Convert logprobs from provider to OpenAI.Completion() format

        Only supported for HF TGI models
        """
        transformed_logprobs: Optional[TextCompletionLogprobs] = None

        return transformed_logprobs


def _should_convert_tool_call_to_json_mode(
    tool_calls: Optional[
        Union[List[ChatCompletionMessageToolCall], List[DatabricksTool]]
    ] = None,
    convert_tool_call_to_json_mode: Optional[bool] = None,
) -> bool:
    """
    Determine if tool calls should be converted to JSON mode
    """
    if (
        convert_tool_call_to_json_mode
        and tool_calls is not None
        and len(tool_calls) == 1
        and tool_calls[0]["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME
    ):
        return True
    return False


def convert_to_model_response_object(  # noqa: PLR0915
    response_object: Optional[dict] = None,
    model_response_object: Optional[
        Union[
            ModelResponse,
            EmbeddingResponse,
            ImageResponse,
            TranscriptionResponse,
            RerankResponse,
        ]
    ] = None,
    response_type: Literal[
        "completion", "embedding", "image_generation", "audio_transcription", "rerank"
    ] = "completion",
    stream=False,
    start_time=None,
    end_time=None,
    hidden_params: Optional[dict] = None,
    _response_headers: Optional[dict] = None,
    convert_tool_call_to_json_mode: Optional[
        bool
    ] = None,  # used for supporting 'json_schema' on older models
):
    received_args = locals()
    additional_headers = get_response_headers(_response_headers)

    if hidden_params is None:
        hidden_params = {}
    hidden_params["additional_headers"] = additional_headers

    ### CHECK IF ERROR IN RESPONSE ### - openrouter returns these in the dictionary
    if (
        response_object is not None
        and "error" in response_object
        and response_object["error"] is not None
    ):
        error_args = {"status_code": 422, "message": "Error in response object"}
        if isinstance(response_object["error"], dict):
            if "code" in response_object["error"]:
                error_args["status_code"] = response_object["error"]["code"]
            if "message" in response_object["error"]:
                if isinstance(response_object["error"]["message"], dict):
                    message_str = json.dumps(response_object["error"]["message"])
                else:
                    message_str = str(response_object["error"]["message"])
                error_args["message"] = message_str
        raised_exception = Exception()
        setattr(raised_exception, "status_code", error_args["status_code"])
        setattr(raised_exception, "message", error_args["message"])
        raise raised_exception

    try:
        if response_type == "completion" and (
            model_response_object is None
            or isinstance(model_response_object, ModelResponse)
        ):
            if response_object is None or model_response_object is None:
                raise Exception("Error in response object format")
            if stream is True:
                # for returning cached responses, we need to yield a generator
                return convert_to_streaming_response(response_object=response_object)
            choice_list = []

            assert response_object["choices"] is not None and isinstance(
                response_object["choices"], Iterable
            )

            for idx, choice in enumerate(response_object["choices"]):
                ## HANDLE JSON MODE - anthropic returns single function call]
                tool_calls = choice["message"].get("tool_calls", None)
                if tool_calls is not None:
                    _openai_tool_calls = []
                    for _tc in tool_calls:
                        _openai_tc = ChatCompletionMessageToolCall(**_tc)
                        _openai_tool_calls.append(_openai_tc)
                    fixed_tool_calls = _handle_invalid_parallel_tool_calls(
                        _openai_tool_calls
                    )

                    if fixed_tool_calls is not None:
                        tool_calls = fixed_tool_calls

                message: Optional[Message] = None
                finish_reason: Optional[str] = None
                if _should_convert_tool_call_to_json_mode(
                    tool_calls=tool_calls,
                    convert_tool_call_to_json_mode=convert_tool_call_to_json_mode,
                ):
                    # to support 'json_schema' logic on older models
                    json_mode_content_str: Optional[str] = tool_calls[0][
                        "function"
                    ].get("arguments")
                    if json_mode_content_str is not None:
                        message = litellm.Message(content=json_mode_content_str)
                        finish_reason = "stop"
                if message is None:
                    provider_specific_fields = {}
                    message_keys = Message.model_fields.keys()
                    for field in choice["message"].keys():
                        if field not in message_keys:
                            provider_specific_fields[field] = choice["message"][field]

                    # Handle reasoning models that display `reasoning_content` within `content`
                    reasoning_content, content = _extract_reasoning_content(
                        choice["message"]
                    )

                    # Handle thinking models that display `thinking_blocks` within `content`
                    thinking_blocks: Optional[
                        List[
                            Union[
                                ChatCompletionThinkingBlock,
                                ChatCompletionRedactedThinkingBlock,
                            ]
                        ]
                    ] = None
                    if "thinking_blocks" in choice["message"]:
                        thinking_blocks = choice["message"]["thinking_blocks"]
                        provider_specific_fields["thinking_blocks"] = thinking_blocks

                    if reasoning_content:
                        provider_specific_fields[
                            "reasoning_content"
                        ] = reasoning_content

                    message = Message(
                        content=content,
                        role=choice["message"]["role"] or "assistant",
                        function_call=choice["message"].get("function_call", None),
                        tool_calls=tool_calls,
                        audio=choice["message"].get("audio", None),
                        provider_specific_fields=provider_specific_fields,
                        reasoning_content=reasoning_content,
                        thinking_blocks=thinking_blocks,
                        annotations=choice["message"].get("annotations", None),
                    )
                    finish_reason = choice.get("finish_reason", None)
                if finish_reason is None:
                    # gpt-4 vision can return 'finish_reason' or 'finish_details'
                    finish_reason = choice.get("finish_details") or "stop"
                if (
                    finish_reason == "stop"
                    and message.tool_calls
                    and len(message.tool_calls) > 0
                ):
                    finish_reason = "tool_calls"

                ## PROVIDER SPECIFIC FIELDS ##
                provider_specific_fields = {}
                for field in choice.keys():
                    if field not in Choices.model_fields.keys():
                        provider_specific_fields[field] = choice[field]

                logprobs = choice.get("logprobs", None)
                enhancements = choice.get("enhancements", None)
                choice = Choices(
                    finish_reason=finish_reason,
                    index=idx,
                    message=message,
                    logprobs=logprobs,
                    enhancements=enhancements,
                    provider_specific_fields=provider_specific_fields,
                )
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object and response_object["usage"] is not None:
                usage_object = litellm.Usage(**response_object["usage"])
                setattr(model_response_object, "usage", usage_object)
            if "created" in response_object:
                model_response_object.created = _safe_convert_created_field(response_object["created"])

            if "id" in response_object:
                model_response_object.id = response_object["id"] or str(uuid.uuid4())

            if "system_fingerprint" in response_object:
                model_response_object.system_fingerprint = response_object[
                    "system_fingerprint"
                ]

            if "model" in response_object:
                if model_response_object.model is None:
                    model_response_object.model = response_object["model"]
                elif (
                    "/" in model_response_object.model
                    and response_object["model"] is not None
                ):
                    openai_compatible_provider = model_response_object.model.split("/")[
                        0
                    ]
                    model_response_object.model = (
                        openai_compatible_provider + "/" + response_object["model"]
                    )

            if start_time is not None and end_time is not None:
                if isinstance(start_time, type(end_time)):
                    model_response_object._response_ms = (  # type: ignore
                        end_time - start_time
                    ).total_seconds() * 1000

            if hidden_params is not None:
                if model_response_object._hidden_params is None:
                    model_response_object._hidden_params = {}
                model_response_object._hidden_params.update(hidden_params)

            if _response_headers is not None:
                model_response_object._response_headers = _response_headers

            special_keys = list(litellm.ModelResponse.model_fields.keys())
            special_keys.append("usage")
            for k, v in response_object.items():
                if k not in special_keys:
                    setattr(model_response_object, k, v)

            return model_response_object
        elif response_type == "embedding" and (
            model_response_object is None
            or isinstance(model_response_object, EmbeddingResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = EmbeddingResponse()

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            if "object" in response_object:
                model_response_object.object = response_object["object"]

            model_response_object.data = response_object["data"]

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
                model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
                model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

            if start_time is not None and end_time is not None:
                model_response_object._response_ms = (  # type: ignore
                    end_time - start_time
                ).total_seconds() * 1000  # return response latency in ms like openai

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params

            if _response_headers is not None:
                model_response_object._response_headers = _response_headers

            return model_response_object
        elif response_type == "image_generation" and (
            model_response_object is None
            or isinstance(model_response_object, ImageResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            return LiteLLMResponseObjectHandler.convert_to_image_response(
                response_object=response_object,
                model_response_object=model_response_object,
                hidden_params=hidden_params,
            )

        elif response_type == "audio_transcription" and (
            model_response_object is None
            or isinstance(model_response_object, TranscriptionResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = TranscriptionResponse()

            if "text" in response_object:
                model_response_object.text = response_object["text"]

            optional_keys = ["language", "task", "duration", "words", "segments"]
            for key in optional_keys:  # not guaranteed to be in response
                if key in response_object:
                    setattr(model_response_object, key, response_object[key])

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params

            if _response_headers is not None:
                model_response_object._response_headers = _response_headers

            return model_response_object
        elif response_type == "rerank" and (
            model_response_object is None
            or isinstance(model_response_object, RerankResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = RerankResponse(**response_object)
                return model_response_object

            if "id" in response_object:
                model_response_object.id = response_object["id"]

            if "meta" in response_object:
                model_response_object.meta = response_object["meta"]

            if "results" in response_object:
                model_response_object.results = response_object["results"]

            return model_response_object
    except Exception:
        raise Exception(
            f"Invalid response object {traceback.format_exc()}\n\nreceived_args={received_args}"
        )
