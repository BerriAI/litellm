"""
Base class for Aliyun guardrails
阿里云护栏基类
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues


class AliyunGuardrailBase:
    """
    Base class for Aliyun guardrails.
    """

    @staticmethod
    def _iter_user_messages(messages: Sequence[AllMessageValues]) -> Iterator[AllMessageValues]:
        """
        Yield every user message of the request, in order.
        Restricting this to the trailing user block would let a caller hide a
        prohibited turn behind an attacker-supplied assistant message.
        """
        return (message for message in messages if message.get("role") == "user")

    @staticmethod
    def _iter_audited_text_messages(messages: Sequence[AllMessageValues]) -> Iterator[AllMessageValues]:
        return (message for message in messages if message.get("role") in ("user", "tool"))

    @staticmethod
    def _extract_image_url(part: object) -> str | None:
        """
        Return the URL of an ``image_url`` content part.
        Args:
            part: A single content part of a message
        Returns:
            The URL string, or None when the part carries no image URL
        """
        if not isinstance(part, dict) or part.get("type") not in ("image_url", "input_image"):
            return None
        image_url: Final = part.get("image_url")
        if isinstance(image_url, dict):
            url: Final = image_url.get("url")
            return url if isinstance(url, str) else None
        return image_url if isinstance(image_url, str) else None

    def get_user_prompt(self, messages: Sequence[AllMessageValues]) -> str | None:
        """
        Collect the text of every user message in the request.
        Scanning only the trailing user block would let a caller hide a
        prohibited prompt behind an attacker-supplied assistant message, so all
        user turns of the submitted request are audited.
        Example:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you!"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]
        get_user_prompt(messages) -> "Hello, how are you?\nWhat is the weather in Tokyo?"
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_content_list_to_str,
        )

        user_prompt: Final = "\n".join(
            convert_content_list_to_str(message) for message in self._iter_audited_text_messages(messages)
        ).strip()
        return user_prompt or None

    def _iter_public_image_urls(self, messages: Sequence[AllMessageValues]) -> Iterator[str]:
        """Yield publicly reachable image URLs from user and tool messages."""
        for content in (message.get("content") for message in self._iter_audited_text_messages(messages)):
            if not isinstance(content, list):
                continue
            for url in (self._extract_image_url(part) for part in content):
                # Only public http(s) URLs are reachable by the Aliyun API, so
                # data: URIs and other inline payloads are skipped.
                if url is not None and url.startswith(("http://", "https://")):
                    yield url

    def get_image_urls(self, messages: Sequence[AllMessageValues]) -> tuple[str, ...]:
        """
        Extract image URLs from every user and tool message in the request.
        Only publicly accessible http(s) URLs are collected (in order,
        de-duplicated). Uses the same message range as ``get_user_prompt``.
        Example:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://a.com/x.png"}},
            ]},
        ]
        get_image_urls(messages) -> ("https://a.com/x.png",)
        """
        # dict.fromkeys is the order-preserving dedup; it is transient and the
        # result is frozen into a tuple before it leaves this method.
        return tuple(dict.fromkeys(self._iter_public_image_urls(messages)))  # mutable-ok: transient dedup, frozen here
