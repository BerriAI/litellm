"""
Common utility functions used for translating messages across providers
"""

from typing import List

from litellm.types.llms.openai import AllMessageValues


def convert_content_list_to_str(message: AllMessageValues) -> AllMessageValues:
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

    if texts:
        message["content"] = texts

    return message
