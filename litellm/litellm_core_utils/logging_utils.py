from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.utils import ModelResponse, TextCompletionResponse

if TYPE_CHECKING:
    from litellm import ModelResponse as _ModelResponse

    LiteLLMModelResponse = _ModelResponse
else:
    LiteLLMModelResponse = Any


import litellm

"""
Helper utils used for logging callbacks
"""


def convert_litellm_response_object_to_str(
    response_obj: Union[Any, LiteLLMModelResponse]
) -> Optional[str]:
    """
    Get the string of the response object from LiteLLM

    """
    if isinstance(response_obj, litellm.ModelResponse):
        response_str = ""
        for choice in response_obj.choices:
            if isinstance(choice, litellm.Choices):
                if choice.message.content and isinstance(choice.message.content, str):
                    response_str += choice.message.content
        return response_str

    return None


def _assemble_complete_response_from_streaming_chunks(
    result: Union[ModelResponse, TextCompletionResponse],
    start_time: datetime,
    end_time: datetime,
    request_kwargs: dict,
    streaming_chunks: List[Any],
    is_async: bool,
):
    """
    Assemble a complete response from a streaming chunks

    - assemble a complete streaming response if result.choices[0].finish_reason is not None
    - else append the chunk to the streaming_chunks


    Args:
        result: ModelResponse
        start_time: datetime
        end_time: datetime
        request_kwargs: dict
        streaming_chunks: List[Any]
        is_async: bool

    Returns:
        Optional[Union[ModelResponse, TextCompletionResponse]]: Complete streaming response

    """
    complete_streaming_response: Optional[
        Union[ModelResponse, TextCompletionResponse]
    ] = None
    if result.choices[0].finish_reason is not None:  # if it's the last chunk
        streaming_chunks.append(result)
        try:
            complete_streaming_response = litellm.stream_chunk_builder(
                chunks=streaming_chunks,
                messages=request_kwargs.get("messages", None),
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            log_message = (
                "Error occurred building stream chunk in {} success logging: {}".format(
                    "async" if is_async else "sync", str(e)
                )
            )
            verbose_logger.exception(log_message)
            complete_streaming_response = None
    else:
        streaming_chunks.append(result)
    return complete_streaming_response
