from typing import Final, Literal

from litellm.llms.base_llm.chat.transformation import BaseLLMException

SCRIBE_REALTIME_MODEL: Final = "scribe_v2_realtime"


class ElevenLabsException(BaseLLMException):
    pass


def normalize_model(model: str) -> str:
    return model.strip().removeprefix("elevenlabs/")


def elevenlabs_health_check_mode(model: str) -> Literal["realtime"] | None:
    return "realtime" if normalize_model(model) == SCRIBE_REALTIME_MODEL else None
