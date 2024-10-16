"""
Common utility functions used for translating messages across providers
"""

import json
from typing import Dict, List

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, ModelResponse, StreamingChoices


def convert_content_list_to_str(message: AllMessageValues) -> str:
    """
    - handles scenario where content is list and not string
    - content list is just text, and no images
    - if image passed in, then just return as is (user-intended)

    Motivation: mistral api + azure ai don't support content as a list
    """
    texts = ""
    message_content = message.get("content")
    if message_content:
        if message_content is not None and isinstance(message_content, list):
            for c in message_content:
                text_content = c.get("text")
                if text_content:
                    texts += text_content
        elif message_content is not None and isinstance(message_content, str):
            texts = message_content

    return texts


def convert_openai_message_to_only_content_messages(
    messages: List[AllMessageValues],
) -> List[Dict[str, str]]:
    """
    Converts OpenAI messages to only content messages

    Used for calling guardrails integrations which expect string content
    """
    converted_messages = []
    user_roles = ["user", "tool", "function"]
    for message in messages:
        if message.get("role") in user_roles:
            converted_messages.append(
                {"role": "user", "content": convert_content_list_to_str(message)}
            )
        elif message.get("role") == "assistant":
            converted_messages.append(
                {"role": "assistant", "content": convert_content_list_to_str(message)}
            )
    return converted_messages


def get_content_from_model_response(response: ModelResponse) -> str:
    """
    Gets content from model response
    """
    content = ""
    for choice in response.choices:
        if isinstance(choice, Choices):
            content += choice.message.content if choice.message.content else ""
            if choice.message.function_call:
                content += choice.message.function_call.model_dump_json()
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    content += tc.model_dump_json()
        elif isinstance(choice, StreamingChoices):
            content += getattr(choice, "delta", {}).get("content", "") or ""
    return content
