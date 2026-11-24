from typing import Final, Literal

from litellm.llms.base_llm.chat.transformation import BaseLLMException

_REALTIME_ENDPOINT: Final = "/v1/realtime"


class ElevenLabsException(BaseLLMException):
    pass


def normalize_model(model: str) -> str:
    return model.strip().removeprefix("elevenlabs/")


def supports_realtime(model: str) -> bool:
    from litellm.utils import ModelInfoLookupMiss, lookup_model_info_for_registration

    model_info: Final = lookup_model_info_for_registration(normalize_model(model), "elevenlabs")
    if isinstance(model_info, ModelInfoLookupMiss):
        return False
    return _REALTIME_ENDPOINT in (model_info.get("supported_endpoints") or ())


def elevenlabs_health_check_mode(model: str) -> Literal["realtime"] | None:
    return "realtime" if supports_realtime(model) else None
