"""
Common utility functions used for translating messages across providers
"""

from typing import List

from litellm.types.llms.openai import AllMessageValues


def convert_content_list_to_str(messages: List[AllMessageValues]):
    """
    - handles scenario where content is list and not string
    - content list is just text, and no images
    - if image passed in, then just return as is (user-intended)

    Motivation: mistral api doesn't support content as a list
    """
    new_messages = []
    for m in messages:
        special_keys = ["role", "content", "tool_calls", "function_call"]
        extra_args = {}
        if isinstance(m, dict):
            for k, v in m.items():
                if k not in special_keys:
                    extra_args[k] = v
        texts = ""
        message_content = m.get("content")
        if message_content is not None and isinstance(message_content, list):
            for c in message_content:
                text_content = c.get("text")
                if text_content:
                    texts += text_content
        elif message_content is not None and isinstance(message_content, str):
            texts = message_content

        new_m = {"role": m["role"], "content": texts, **extra_args}

        new_messages.append(new_m)
    return new_messages
