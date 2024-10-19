import asyncio
from typing import AsyncGenerator, Generator, Optional

from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    ModelResponse,
    StreamingChoices,
    Usage,
)


async def convert_to_streaming_response_async(
    response_object: Optional[dict] = None,
) -> AsyncGenerator:
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
    model_response_object = _get_streaming_response_object(response_object)
    yield model_response_object
    await asyncio.sleep(0)


def convert_to_streaming_response(response_object: Optional[dict] = None) -> Generator:
    # used for yielding Cache hits when stream == True
    model_response_object = _get_streaming_response_object(response_object)
    yield model_response_object


def _get_streaming_response_object(
    response_object: Optional[dict] = None,
) -> ModelResponse:
    """
    Helper function to get the streaming `litellm.ModelResponse` object

    Args:
        response_object (Optional[dict]): The Dictionary response object to be converted. Defaults to None.

    Raises:
        Exception: If the response object is None.
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

    return model_response_object
