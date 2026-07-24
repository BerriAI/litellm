"""
ModelScope Common Utilities

Shared constants and error handling for ModelScope API integration.
"""

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class ModelScopeError(BaseLLMException):
    """Exception class for ModelScope API errors."""

    pass


# Polling configuration for async image generation tasks.
# ModelScope returns a task_id from the submit call and the caller must poll
# GET /v1/tasks/{task_id} until task_status is SUCCEED or FAILED.
DEFAULT_POLLING_INTERVAL = 2.0  # seconds
DEFAULT_MAX_POLLING_TIME = 300  # 5 minutes

ASYNC_MODE_HEADER = "X-ModelScope-Async-Mode"
TASK_TYPE_HEADER = "X-ModelScope-Task-Type"
IMAGE_GENERATION_TASK_TYPE = "image_generation"

TASK_STATUS_SUCCEED = "SUCCEED"
TASK_STATUS_FAILED = "FAILED"
