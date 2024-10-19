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

from .convert_dict_to_chat_completion_response import (
    convert_dict_to_chat_completion_response,
)
from .convert_dict_to_embedding_response import convert_dict_to_embedding_response
from .convert_dict_to_image_generation_response import (
    convert_dict_to_image_generation_response,
)
from .convert_dict_to_rerank_response import convert_dict_to_rerank_response
from .convert_dict_to_transcription_response import (
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
