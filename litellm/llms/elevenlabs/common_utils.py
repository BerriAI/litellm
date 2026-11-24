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


def elevenlabs_catalog_key(model: str) -> str:
    return f"elevenlabs/{normalize_model(model)}"


def elevenlabs_health_check_mode(catalog_entry: Mapping[str, object]) -> Literal["realtime"] | None:
    """Realtime when the model's price-map row lists /v1/realtime, so a new realtime model is a JSON change."""
    try:
        endpoints: Final = _ENDPOINTS.validate_python(catalog_entry.get("supported_endpoints") or ())
    except ValidationError:
        return None
    return "realtime" if _REALTIME_ENDPOINT in endpoints else None
