from enum import Enum
from typing import Final

from litellm.exceptions import BadRequestError


class OpenAIDecisionsFormat(Enum):
    SYSTEMONE = "systemone"
    DECISIONS = "decisions"


def parse_decisions_model(model: str) -> tuple[OpenAIDecisionsFormat | None, str]:
    prefix, separator, remainder = model.partition("/")
    wire_format: Final = next((item for item in OpenAIDecisionsFormat if item.value == prefix), None)
    if not separator or wire_format is None:
        return None, model
    return wire_format, remainder


def chat_model_name(model: str, custom_llm_provider: str) -> str:
    if custom_llm_provider != "openai":
        return model
    canonical: Final = parse_decisions_model(model)[1]
    if not canonical:
        raise BadRequestError(
            message="A model name is required after the Decisions format prefix",
            model=model,
            llm_provider=custom_llm_provider,
        )
    return canonical


def explicit_decisions_endpoints(model: str, custom_llm_provider: str) -> tuple[str, ...]:
    if custom_llm_provider != "openai":
        return ()
    wire_format, upstream_model = parse_decisions_model(model.removeprefix("openai/"))
    return ("/v1/systemone", "/v1/decisions") if wire_format is not None and upstream_model else ()
