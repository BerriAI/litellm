"""
MachGen common utilities.

API reference: https://www.machgen.ai/docs/rest_api
"""

from __future__ import annotations

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class MachGenError(BaseLLMException):
    """Exception raised for MachGen API errors."""


DEFAULT_API_BASE = "https://api.machgen.ai"
GENERATE_PATH = "/api/v0/generate"
TASKS_PATH = "/api/v0/tasks"

DEFAULT_POLLING_INTERVAL = 1.5
DEFAULT_MAX_POLLING_TIME = 300.0

STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

TEXT_TO_IMAGE_TASK_TYPE = "T2I"

IMAGE_OUTPUT_KEY = "image"

MACHGEN_IMAGE_CONFIG_PARAMS: frozenset[str] = frozenset(
    {"width", "height", "aspect_ratio", "infer_steps", "guidance_scale"}
)

MACHGEN_TOP_LEVEL_PARAMS: frozenset[str] = frozenset({"seed", "enhance_prompt", "moderate"})
