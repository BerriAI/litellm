"""
Interface for Anthropic's messages API

Use this to call LLMs in Anthropic /messages Request/Response format
"""

from litellm.llms.anthropic.experimental_pass_through.handler import (
    anthropic_messages as _async_anthropic_messages,
)


async def acreate(*args, **kwargs):
    """
    Wrapper around Anthropic's messages API
    """
    return await _async_anthropic_messages(*args, **kwargs)
