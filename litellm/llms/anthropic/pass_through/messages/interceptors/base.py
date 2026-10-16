from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


class MessagesInterceptor(ABC):
    """
    Base class for /messages short-circuit interceptors.

    An interceptor can fully replace the normal backend call when it detects
    a pattern it owns (e.g. advisor orchestration, web-search short-circuit).
    ``can_handle`` is checked first; if True, ``handle`` is called and its
    return value is returned directly to the caller.

    See interceptors/README.md for when to add an interceptor vs. a pre-request hook.
    """

    @abstractmethod
    def can_handle(
        self,
        tools: list[dict] | None,
        custom_llm_provider: str | None,
    ) -> bool:
        """Return True if this interceptor should handle the request."""

    @abstractmethod
    async def handle(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool | None,
        max_tokens: int,
        custom_llm_provider: str | None,
        **kwargs,
    ) -> AnthropicMessagesResponse | AsyncIterator:
        """Execute the interception and return the response."""
