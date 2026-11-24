from collections.abc import Mapping
from typing import Final, Literal

from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException

SCRIBE_REALTIME_MODEL: Final = "scribe_v2_realtime"
_REALTIME_ENDPOINT: Final = "/v1/realtime"
_ENDPOINTS: Final = TypeAdapter(tuple[str, ...])


class ElevenLabsException(BaseLLMException):
    pass


def normalize_model(model: str) -> str:
    return model.strip().removeprefix("elevenlabs/")


def elevenlabs_health_check_mode(
    model: str, model_cost: Mapping[str, Mapping[str, object]]
) -> Literal["realtime"] | None:
    """Realtime when the model's price-map row lists /v1/realtime, so a new realtime model is a JSON change."""
    entry: Final = model_cost.get(f"elevenlabs/{normalize_model(model)}", {})
    try:
        endpoints: Final = _ENDPOINTS.validate_python(entry.get("supported_endpoints") or ())
    except ValidationError:
        return None
    return "realtime" if _REALTIME_ENDPOINT in endpoints else None
