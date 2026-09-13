"""
ModelScope Common Utilities

Shared constants and error handling for ModelScope API integration.
"""

from typing import Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class ModelScopeError(BaseLLMException):
    """Exception class for ModelScope API errors."""


DEFAULT_POLLING_INTERVAL: Final = 2.0
DEFAULT_MAX_POLLING_TIME: Final = 300

ASYNC_MODE_HEADER: Final = "X-ModelScope-Async-Mode"
TASK_TYPE_HEADER: Final = "X-ModelScope-Task-Type"
IMAGE_GENERATION_TASK_TYPE: Final = "image_generation"

TASK_STATUS_SUCCEED: Final = "SUCCEED"
TASK_STATUS_FAILED: Final = "FAILED"
