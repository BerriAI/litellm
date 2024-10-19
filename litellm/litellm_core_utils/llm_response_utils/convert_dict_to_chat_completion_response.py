import time
import uuid
from datetime import datetime
from typing import Dict, Generator, Iterable, List, Literal, Optional, Union

import litellm
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Message,
    ModelResponse,
    StreamingChoices,
)

from .convert_dict_to_streaming_response import convert_to_streaming_response
from .handle_parallel_tool_calls import _handle_invalid_parallel_tool_calls


def convert_dict_to_chat_completion_response(
    model_response_object: Optional[ModelResponse],
    response_object: Optional[Dict],
    stream: bool,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    hidden_params: Optional[Dict],
    _response_headers: Optional[Dict],
    convert_tool_call_to_json_mode: Optional[bool],
) -> Union[ModelResponse, Generator]:
    if response_object is None or model_response_object is None:
        raise Exception("Error in response object format")

    if stream is True:
        # for returning cached responses, we need to yield a generator
        return convert_to_streaming_response(response_object=response_object)

    assert response_object["choices"] is not None and isinstance(
        response_object["choices"], Iterable
    )
    choice_list: List[Union[Choices, StreamingChoices]] = _process_choices_in_response(
        response_object=response_object,
        convert_tool_call_to_json_mode=convert_tool_call_to_json_mode,
    )
    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        usage_object = litellm.Usage(**response_object["usage"])
        setattr(model_response_object, "usage", usage_object)
    if "created" in response_object:
        model_response_object.created = response_object["created"] or int(time.time())

    if "id" in response_object:
        model_response_object.id = response_object["id"] or str(uuid.uuid4())

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        if model_response_object.model is None:
            model_response_object.model = response_object["model"]
        elif (
            "/" in model_response_object.model and response_object["model"] is not None
        ):
            openai_compatible_provider = model_response_object.model.split("/")[0]
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


def _process_choices_in_response(
    response_object: Dict,
    convert_tool_call_to_json_mode: Optional[bool],
) -> List[Union[Choices, StreamingChoices]]:
    """
    Process the choices in the response object.

    Args:
        response_object (Dict): The API response object to process.
        convert_tool_call_to_json_mode (Optional[bool]): Whether to convert tool calls to JSON mode.

    Returns:
        List[Choices]: The processed choices.
    """

    choice_list: List[Union[Choices, StreamingChoices]] = []
    for idx, choice in enumerate(response_object["choices"]):
        ## HANDLE JSON MODE - anthropic returns single function call]
        tool_calls = choice["message"].get("tool_calls", None)
        if tool_calls is not None:
            _openai_tool_calls = []
            for _tc in tool_calls:
                _openai_tc = ChatCompletionMessageToolCall(**_tc)
                _openai_tool_calls.append(_openai_tc)
            fixed_tool_calls = _handle_invalid_parallel_tool_calls(_openai_tool_calls)

            if fixed_tool_calls is not None:
                tool_calls = fixed_tool_calls

        message: Optional[Message] = None
        finish_reason: Optional[str] = None
        if (
            convert_tool_call_to_json_mode
            and tool_calls is not None
            and len(tool_calls) == 1
        ):
            # to support 'json_schema' logic on older models
            json_mode_content_str: Optional[str] = tool_calls[0]["function"].get(
                "arguments"
            )
            if json_mode_content_str is not None:
                message = litellm.Message(content=json_mode_content_str)
                finish_reason = "stop"
        if message is None:
            message = Message(
                content=choice["message"].get("content", None),
                role=choice["message"]["role"] or "assistant",
                function_call=choice["message"].get("function_call", None),
                tool_calls=tool_calls,
                audio=choice["message"].get("audio", None),
            )
            finish_reason = choice.get("finish_reason", None)
        if finish_reason is None:
            # gpt-4 vision can return 'finish_reason' or 'finish_details'
            finish_reason = choice.get("finish_details") or "stop"
        logprobs = choice.get("logprobs", None)
        enhancements = choice.get("enhancements", None)
        choice = Choices(
            finish_reason=finish_reason,
            index=idx,
            message=message,
            logprobs=logprobs,
            enhancements=enhancements,
        )
        choice_list.append(choice)
    return choice_list
