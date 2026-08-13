"""
Base class for Aliyun guardrails
阿里云护栏基类
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues


class AliyunGuardrailBase:
    """
    Base class for Aliyun guardrails.
    """

    def get_user_prompt(self, messages: list[AllMessageValues]) -> str | None:
        """
        Get the last consecutive block of messages from the user.
        Example:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you!"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]
        get_user_prompt(messages) -> "What is the weather in Tokyo?"
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_content_list_to_str,
        )

        if not messages:
            return None
        # Iterate from the end to find the last consecutive block of user messages
        user_messages = []
        for message in reversed(messages):
            if message.get("role") == "user":
                user_messages.append(message)
            else:
                # Stop when we hit a non-user message
                break
        if not user_messages:
            return None
        # Reverse to get the messages in chronological order
        user_messages.reverse()
        user_prompt = ""
        for message in user_messages:
            text_content = convert_content_list_to_str(message)
            user_prompt += text_content + "\n"
        result = user_prompt.strip()
        return result if result else None

    def get_image_urls(self, messages: list[AllMessageValues]) -> list[str]:
        """
        Extract image URLs from the last consecutive block of user messages.
        Only publicly accessible http(s) URLs are collected (in order,
        de-duplicated). Uses the same message range as ``get_user_prompt``.
        Example:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://a.com/x.png"}},
            ]},
        ]
        get_image_urls(messages) -> ["https://a.com/x.png"]
        """
        if not messages:
            return []
        # Iterate from the end to find the last consecutive block of user messages
        user_messages = []
        for message in reversed(messages):
            if message.get("role") == "user":
                user_messages.append(message)
            else:
                break
        if not user_messages:
            return []
        user_messages.reverse()
        image_urls: list[str] = []
        seen = set()
        for message in user_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                url: str | None = None
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                elif isinstance(image_url, str):
                    url = image_url
                if not isinstance(url, str):
                    continue
                if not (url.startswith("http://") or url.startswith("https://")):
                    continue
                if url not in seen:
                    seen.add(url)
                    image_urls.append(url)
        return image_urls
