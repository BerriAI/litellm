"""@private"""

import logging
import typing
from datetime import datetime, timezone

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore

from langfuse.model import ModelUsage, PromptClient

log = logging.getLogger("langfuse")


def _get_timestamp():
    return datetime.now(timezone.utc)


def _create_prompt_context(
    prompt: typing.Optional[PromptClient] = None,
):
    if prompt is not None and not prompt.is_fallback:
        return {"prompt_version": prompt.version, "prompt_name": prompt.name}

    return {"prompt_version": None, "prompt_name": None}


T = typing.TypeVar("T")


def extract_by_priority(
    usage: dict, keys: typing.List[str], target_type: typing.Type[T]
) -> typing.Optional[T]:
    """Extracts the first key that exists in usage and converts its value to target_type"""
    for key in keys:
        if key in usage:
            value = usage[key]
            try:
                if value is None:
                    return None
                return target_type(value)
            except Exception:
                continue
    return None


def _convert_usage_input(usage: typing.Union[pydantic.BaseModel, ModelUsage]):
    """Converts any usage input to a usage object"""
    if isinstance(usage, pydantic.BaseModel):
        usage = usage.dict()

    # sometimes we do not match the pydantic usage object
    # in these cases, we convert to dict manually
    if hasattr(usage, "__dict__"):
        usage = usage.__dict__

    # validate that usage object has input, output, total, usage
    is_langfuse_usage = any(k in usage for k in ("input", "output", "total", "unit"))

    if is_langfuse_usage:
        return usage

    is_openai_usage = any(
        k in usage
        for k in (
            "promptTokens",
            "prompt_tokens",
            "completionTokens",
            "completion_tokens",
            "totalTokens",
            "total_tokens",
            "inputCost",
            "input_cost",
            "outputCost",
            "output_cost",
            "totalCost",
            "total_cost",
        )
    )

    if is_openai_usage:
        # convert to langfuse usage
        usage = {
            "input": extract_by_priority(usage, ["promptTokens", "prompt_tokens"], int),
            "output": extract_by_priority(
                usage,
                ["completionTokens", "completion_tokens"],
                int,
            ),
            "total": extract_by_priority(usage, ["totalTokens", "total_tokens"], int),
            "unit": "TOKENS",
            "inputCost": extract_by_priority(usage, ["inputCost", "input_cost"], float),
            "outputCost": extract_by_priority(
                usage, ["outputCost", "output_cost"], float
            ),
            "totalCost": extract_by_priority(usage, ["totalCost", "total_cost"], float),
        }
        return usage

    if not is_langfuse_usage and not is_openai_usage:
        raise ValueError(
            "Usage object must have either {input, output, total, unit} or {promptTokens, completionTokens, totalTokens}"
        )
