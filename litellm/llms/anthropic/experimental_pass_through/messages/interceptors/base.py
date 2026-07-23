from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional, Union

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
        tools: Optional[List[Dict]],
        custom_llm_provider: Optional[str],
    ) -> bool:
        """Return True if this interceptor should handle the request."""

    def resolve_gate_provider(
        self,
        model: str,
        custom_llm_provider: Optional[str],
        tools: Optional[List[Dict]],
    ) -> Optional[str]:
        """Return the provider ``can_handle`` should be gated on (default: unchanged).

        An interceptor may override this when ``custom_llm_provider`` was
        name-inferred from a proxy alias and does not reflect the routed deployment.
        """
        return custom_llm_provider

    @abstractmethod
    async def handle(
        self,
        *,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: Optional[bool],
        max_tokens: int,
        custom_llm_provider: Optional[str],
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        """Execute the interception and return the response."""
