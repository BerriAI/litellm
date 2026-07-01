from typing import cast

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class SiliconFlowException(BaseLLMException):
    pass


def get_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def get_list(value: object) -> list[object]:
    if isinstance(value, list):
        return cast(list[object], value)
    return []


def get_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def get_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def get_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
