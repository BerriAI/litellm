"""Exceptions raised by the context_management polyfill."""


class AnthropicContextManagementError(Exception):
    """Validation error from the polyfill, surfaced as an Anthropic-format 4xx.

    The `/v1/messages` endpoint catches this in its exception handler and
    emits an Anthropic-shaped error body instead of the default OpenAI shape.
    """

    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
