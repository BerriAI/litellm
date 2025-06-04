"""
Interface for Anthropic's messages API

Use this to call LLMs in Anthropic /messages Request/Response format

This is an __init__.py file to allow the following interface

- litellm.messages.acreate
- litellm.messages.create

"""

from typing import AsyncIterator, Dict, Iterator, List, Optional, Union

from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages as _async_anthropic_messages,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


async def acreate(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs
) -> Union[AnthropicMessagesResponse, AsyncIterator]:
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
        **kwargs,
    )


async def create(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs
) -> Union[AnthropicMessagesResponse, Iterator]:
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
    raise NotImplementedError("This function is not implemented")
