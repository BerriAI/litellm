"""
Interface for Anthropic's messages API

Use this to call LLMs in Anthropic /messages Request/Response format

This is an __init__.py file to allow the following interface

- litellm.messages.acreate
- litellm.messages.create

"""

from collections.abc import AsyncIterator, Coroutine, Iterator, Mapping
from datetime import datetime, timezone
from typing import Final
from uuid import uuid4

from pydantic import TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.response_metadata import update_response_metadata
from litellm.messages import (
    anthropic_messages as _async_anthropic_messages,
)
from litellm.messages import (
    anthropic_messages_handler as _sync_anthropic_messages,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)

_REQUEST_KWARGS: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = TypeAdapter(list[dict[str, object]])
_OPTIONAL_MAPPING: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)


async def acreate(
    max_tokens: int,
    messages: list[dict],
    model: str,
    metadata: dict | None = None,
    stop_sequences: list[str] | None = None,
    stream: bool | None = False,
    system: str | None = None,
    temperature: float | None = None,
    thinking: dict | None = None,
    tool_choice: dict | None = None,
    tools: list[dict] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    container: dict | None = None,
    **kwargs,
) -> AnthropicMessagesResponse | AsyncIterator:
    """
    Async wrapper for Anthropic's messages API

    Args:
        max_tokens (int): Maximum tokens to generate (required)
        messages (List[Dict]): List of message objects with role and content (required)
        model (str): Model name to use (required)
        metadata (Dict, optional): Request metadata
        stop_sequences (List[str], optional): Custom stop sequences
        stream (bool, optional): Whether to stream the response
        system (str, optional): System prompt
        temperature (float, optional): Sampling temperature (0.0 to 1.0)
        thinking (Dict, optional): Extended thinking configuration
        tool_choice (Dict, optional): Tool choice configuration
        tools (List[Dict], optional): List of tool definitions
        top_k (int, optional): Top K sampling parameter
        top_p (float, optional): Nucleus sampling parameter
        container (Dict, optional): Container config with skills for code execution
        **kwargs: Additional arguments

    Returns:
        Dict: Response from the API
    """
    return await _async_anthropic_messages(
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        metadata=metadata,
        stop_sequences=stop_sequences,
        stream=stream,
        system=system,
        temperature=temperature,
        thinking=thinking,
        tool_choice=tool_choice,
        tools=tools,
        top_k=top_k,
        top_p=top_p,
        container=container,
        **kwargs,
    )


def create(
    max_tokens: int,
    messages: list[dict],
    model: str,
    metadata: dict | None = None,
    stop_sequences: list[str] | None = None,
    stream: bool | None = False,
    system: str | None = None,
    temperature: float | None = None,
    thinking: dict | None = None,
    tool_choice: dict | None = None,
    tools: list[dict] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    container: dict | None = None,
    **kwargs,
) -> (
    AnthropicMessagesResponse
    | Iterator[bytes]
    | AsyncIterator[object]
    | Coroutine[object, object, AnthropicMessagesResponse | AsyncIterator[object] | Iterator[bytes]]
):
    """
    Async wrapper for Anthropic's messages API

    Args:
        max_tokens (int): Maximum tokens to generate (required)
        messages (List[Dict]): List of message objects with role and content (required)
        model (str): Model name to use (required)
        metadata (Dict, optional): Request metadata
        stop_sequences (List[str], optional): Custom stop sequences
        stream (bool, optional): Whether to stream the response
        system (str, optional): System prompt
        temperature (float, optional): Sampling temperature (0.0 to 1.0)
        thinking (Dict, optional): Extended thinking configuration
        tool_choice (Dict, optional): Tool choice configuration
        tools (List[Dict], optional): List of tool definitions
        top_k (int, optional): Top K sampling parameter
        top_p (float, optional): Nucleus sampling parameter
        **kwargs: Additional arguments

    Returns:
        Dict: Response from the API
    """
    request_kwargs: Final = _REQUEST_KWARGS.validate_python(kwargs)
    supplied_logging: Final = request_kwargs.get("litellm_logging_obj")
    logging_obj: Final = (
        supplied_logging
        if isinstance(supplied_logging, Logging)
        else Logging(
            model=model,
            messages=_MESSAGES.validate_python(messages),
            stream=stream is True,
            call_type="anthropic_messages",
            start_time=datetime.now(timezone.utc),
            litellm_call_id=str(uuid4()),
            function_id=str(uuid4()),
        )
    )
    response: Final = _sync_anthropic_messages(
        max_tokens=max_tokens,
        messages=_MESSAGES.validate_python(messages),
        model=model,
        metadata=_OPTIONAL_MAPPING.validate_python(metadata),
        stop_sequences=stop_sequences,
        stream=stream,
        system=system,
        temperature=temperature,
        thinking=_OPTIONAL_MAPPING.validate_python(thinking),
        tool_choice=_OPTIONAL_MAPPING.validate_python(tool_choice),
        tools=tools,
        top_k=top_k,
        top_p=top_p,
        container=container,
        **{**request_kwargs, "litellm_logging_obj": logging_obj},
    )
    if not stream:
        update_response_metadata(
            response,
            logging_obj,
            model,
            request_kwargs,
            logging_obj.start_time,
            datetime.now(timezone.utc),
        )
    if stream and isinstance(response, (Iterator, AsyncIterator)):
        from litellm.llms.anthropic.messages.native.streaming_iterator import UsageCostMessagesStream

        return UsageCostMessagesStream(response, logging_obj)
    return response
