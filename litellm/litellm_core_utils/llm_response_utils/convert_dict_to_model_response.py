"""
Parent File handles converting dicts to litellm Response objects

Example - converts a dict to a litellm.ModelResponse object

Consists of:
- convert_to_streaming_response_async
- convert_to_streaming_response
- convert_to_model_response_object
"""

import asyncio
import json
import time
import traceback
import uuid
from typing import Dict, Iterable, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    EmbeddingResponse,
    Function,
    ImageResponse,
    Message,
    ModelResponse,
    RerankResponse,
    StreamingChoices,
    TranscriptionResponse,
    Usage,
)

from .llm_response_utils_temp.convert_dict_to_chat_completion_response import (
    convert_dict_to_chat_completion_response,
)
from .llm_response_utils_temp.convert_dict_to_embedding_response import (
    convert_dict_to_embedding_response,
)
from .llm_response_utils_temp.convert_dict_to_image_generation_response import (
    convert_dict_to_image_generation_response,
)
from .llm_response_utils_temp.convert_dict_to_rerank_response import (
    convert_dict_to_rerank_response,
)
from .llm_response_utils_temp.convert_dict_to_transcription_response import (
    convert_dict_to_transcription_response,
)


def convert_to_model_response_object(
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
    _handle_error_in_response_object(response_object=response_object)
    hidden_params = hidden_params or {}
    received_args = locals()
    if _response_headers is not None:
        hidden_params = _set_headers_in_hidden_params(
            hidden_params=hidden_params,
            _response_headers=_response_headers,
        )

    try:
        if response_type == "completion" and (
            model_response_object is None
            or isinstance(model_response_object, ModelResponse)
        ):
            return convert_dict_to_chat_completion_response(
                model_response_object=model_response_object,
                response_object=response_object,
                stream=stream,
                start_time=start_time,
                end_time=end_time,
                hidden_params=hidden_params,
                _response_headers=_response_headers,
                convert_tool_call_to_json_mode=convert_tool_call_to_json_mode,
            )
        elif response_type == "embedding" and (
            model_response_object is None
            or isinstance(model_response_object, EmbeddingResponse)
        ):
            return convert_dict_to_embedding_response(
                model_response_object=model_response_object,
                response_object=response_object,
                start_time=start_time,
                end_time=end_time,
                hidden_params=hidden_params,
                _response_headers=_response_headers,
            )
        elif response_type == "image_generation" and (
            model_response_object is None
            or isinstance(model_response_object, ImageResponse)
        ):
            return convert_dict_to_image_generation_response(
                model_response_object=model_response_object,
                response_object=response_object,
                hidden_params=hidden_params,
            )
        elif response_type == "audio_transcription" and (
            model_response_object is None
            or isinstance(model_response_object, TranscriptionResponse)
        ):
            return convert_dict_to_transcription_response(
                model_response_object=model_response_object,
                response_object=response_object,
                hidden_params=hidden_params,
                _response_headers=_response_headers,
            )
        elif response_type == "rerank" and (
            model_response_object is None
            or isinstance(model_response_object, RerankResponse)
        ):
            return convert_dict_to_rerank_response(
                model_response_object=model_response_object,
                response_object=response_object,
            )
    except Exception:
        raise Exception(
            f"Invalid response object {traceback.format_exc()}\n\nreceived_args={received_args}"
        )


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
        model_response_object.created = response_object["created"]

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
        delta = Delta(
            content=choice["message"].get("content", None),
            role=choice["message"]["role"],
            function_call=choice["message"].get("function_call", None),
            tool_calls=choice["message"].get("tool_calls", None),
        )
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
        model_response_object.created = response_object["created"]

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]
    yield model_response_object


def _handle_error_in_response_object(response_object: Optional[dict] = None):
    """
    Raises an exception if there is an error in the response object.

    (openrouter returns these in the JSON response)
    """
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


def _get_openai_headers(_response_headers: Dict) -> Dict:
    openai_headers = {}
    if "x-ratelimit-limit-requests" in _response_headers:
        openai_headers["x-ratelimit-limit-requests"] = _response_headers[
            "x-ratelimit-limit-requests"
        ]
    if "x-ratelimit-remaining-requests" in _response_headers:
        openai_headers["x-ratelimit-remaining-requests"] = _response_headers[
            "x-ratelimit-remaining-requests"
        ]
    if "x-ratelimit-limit-tokens" in _response_headers:
        openai_headers["x-ratelimit-limit-tokens"] = _response_headers[
            "x-ratelimit-limit-tokens"
        ]
    if "x-ratelimit-remaining-tokens" in _response_headers:
        openai_headers["x-ratelimit-remaining-tokens"] = _response_headers[
            "x-ratelimit-remaining-tokens"
        ]

    return openai_headers


def _set_headers_in_hidden_params(hidden_params: Dict, _response_headers: Dict) -> Dict:
    openai_headers = _get_openai_headers(_response_headers)
    llm_response_headers = {
        "{}-{}".format("llm_provider", k): v for k, v in _response_headers.items()
    }

    if hidden_params is not None:
        hidden_params["additional_headers"] = {
            **llm_response_headers,
            **openai_headers,
        }
    return hidden_params
