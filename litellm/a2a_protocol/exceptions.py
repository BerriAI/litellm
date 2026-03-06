"""
A2A Protocol Exceptions.

Custom exception types for A2A protocol operations, following LiteLLM's exception pattern.
"""

from typing import Optional

import httpx


class A2AError(Exception):
    """
    Base exception for A2A protocol errors.

    Follows the same pattern as LiteLLM's main exceptions.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        llm_provider: str = "a2a_agent",
        model: Optional[str] = None,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.status_code = status_code
        self.message = f"litellm.A2AError: {message}"
        self.llm_provider = llm_provider
        self.model = model
        self.litellm_debug_info = litellm_debug_info
        self.max_retries = max_retries
        self.num_retries = num_retries
        self.response = response or httpx.Response(
            status_code=self.status_code,
            request=httpx.Request(method="POST", url="https://litellm.ai"),
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        _message = self.message
        if self.num_retries:
            _message += f" LiteLLM Retried: {self.num_retries} times"
        if self.max_retries:
            _message += f", LiteLLM Max Retries: {self.max_retries}"
        return _message

    def __repr__(self) -> str:
        return self.__str__()


class A2AConnectionError(A2AError):
    """
    Raised when connection to an A2A agent fails.

    This typically occurs when:
    - The agent is unreachable
    - The agent card contains a localhost/internal URL
    - Network issues prevent connection
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        model: Optional[str] = None,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_retries: Optional[int] = None,
    ):
        self.url = url
        super().__init__(
            message=message,
            status_code=503,
            llm_provider="a2a_agent",
            model=model,
            response=response,
            litellm_debug_info=litellm_debug_info,
            max_retries=max_retries,
            num_retries=num_retries,
        )


class A2AAgentCardError(A2AError):
    """
    Raised when there's an issue with the agent card.

    This includes:
    - Failed to fetch agent card
    - Invalid agent card format
    - Missing required fields
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        model: Optional[str] = None,
        response: Optional[httpx.Response] = None,
        litellm_debug_info: Optional[str] = None,
    ):
        self.url = url
        super().__init__(
            message=message,
            status_code=404,
            llm_provider="a2a_agent",
            model=model,
            response=response,
            litellm_debug_info=litellm_debug_info,
        )


class A2ALocalhostURLError(A2AConnectionError):
    """
    Raised when an agent card contains a localhost/internal URL.

    Many A2A agents are deployed with agent cards that contain internal URLs
    like "http://0.0.0.0:8001/" or "http://localhost:8000/". This error
    indicates that the URL needs to be corrected and the request should be retried.

    Attributes:
        localhost_url: The localhost/internal URL found in the agent card
        base_url: The public base URL that should be used instead
        original_error: The original connection error that was raised
    """

    def __init__(
        self,
        localhost_url: str,
        base_url: str,
        original_error: Optional[Exception] = None,
        model: Optional[str] = None,
    ):
        self.localhost_url = localhost_url
        self.base_url = base_url
        self.original_error = original_error

        message = (
            f"Agent card contains localhost/internal URL '{localhost_url}'. "
            f"Retrying with base URL '{base_url}'."
        )
        super().__init__(
            message=message,
            url=localhost_url,
            model=model,
        )
