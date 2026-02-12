"""
Utility functions for Anthropic /v1/messages endpoint processing.
"""

from typing import List, Optional, Union

from fastapi import Request

from litellm.types.llms.anthropic import AllAnthropicMessageValues


def _is_claude_code_client(request: Request) -> bool:
    """Check if the request originates from a Claude Code client via User-Agent header."""
    user_agent = request.headers.get("user-agent", "") or ""
    return "claude-code" in user_agent.lower()


def _has_cache_control_in_messages(
    messages: Union[List[AllAnthropicMessageValues], List[dict]],
) -> bool:
    """Check if any message content block already has cache_control set."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False


def _inject_cache_control_to_last_message(
    messages: Union[List[AllAnthropicMessageValues], List[dict]],
) -> Union[List[AllAnthropicMessageValues], List[dict]]:
    """
    Inject cache_control: {"type": "ephemeral"} into the last content block
    of the last message. If the content is a string, convert it to list format.

    Returns messages unchanged if content is None or empty.
    """
    if not messages:
        return messages

    last_message = messages[-1]
    if not isinstance(last_message, dict):
        return messages

    content = last_message.get("content")

    if content is None:
        return messages
    elif isinstance(content, str):
        # Convert string content to list format with cache_control
        last_message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and len(content) > 0:
        last_block = content[-1]
        if isinstance(last_block, dict):
            last_block["cache_control"] = {"type": "ephemeral"}

    return messages


def maybe_inject_auto_prompt_caching(
    request: Request,
    data: dict,
    general_settings: Optional[dict] = None,
) -> dict:
    """
    Automatically inject cache_control into messages for Claude Code clients.

    Controlled by general_settings["auto_prompt_caching"] (default: True).
    Only applies when:
    - The User-Agent indicates a Claude Code client
    - No existing cache_control is found in the messages
    - auto_prompt_caching is not disabled in settings
    """
    if general_settings is None:
        general_settings = {}

    auto_prompt_caching = general_settings.get("auto_prompt_caching", True)
    if not auto_prompt_caching:
        return data

    if not _is_claude_code_client(request):
        return data

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return data

    if _has_cache_control_in_messages(messages):
        return data

    data["messages"] = _inject_cache_control_to_last_message(messages)
    return data
