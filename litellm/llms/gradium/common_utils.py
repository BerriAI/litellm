from collections.abc import Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.base_llm.chat.transformation import BaseLLMException


class GradiumException(BaseLLMException):
    pass


def gradium_api_url(api_base: str | None, model: str, path: str) -> str:
    if not api_base:
        raise litellm.APIConnectionError(
            message="Gradium API base is required. Set `api_base` on the model, e.g. https://eu.api.gradium.ai.",
            llm_provider="gradium",
            model=model,
        )

    root: Final = api_base.rstrip("/").removesuffix("/api")
    return f"{root}{path}"


def gradium_api_key(api_key: str | None, model: str) -> str:
    if not api_key:
        raise litellm.AuthenticationError(
            message="Gradium API key is required. Set `api_key` on the model.",
            llm_provider="gradium",
            model=model,
        )

    return api_key


def caller_json_config(params: Mapping[str, object]) -> object | None:
    extra_body: Final = as_str_mapping(params.get("extra_body"))
    from_extra_body: Final = extra_body.get("json_config") if extra_body is not None else None
    return from_extra_body if from_extra_body is not None else params.get("json_config")
