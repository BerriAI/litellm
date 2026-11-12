import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from httpx import Headers
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.litellm_core_utils.audio_utils.utils import BARE_ISO_639_1_TO_BCP47
from litellm.llms.base_llm.chat.transformation import BaseLLMException

ZOOM_AI_TRACKING_ID_PREFIX: Final = "zoom_ai_community_litellm_"

# LiteLLM's shared map covers the common bare codes (pt -> pt-BR, ar -> ar-SA). These are the other languages
# https://developers.zoom.us/docs/ai-services/scribe/speech-configuration/ lists, each with its one documented locale
ZOOM_AI_DEFAULT_LOCALES: Final = MappingProxyType(
    {
        **BARE_ISO_639_1_TO_BCP47,
        "bn": "bn-IN",
        "ca": "ca-ES",
        "cs": "cs-CZ",
        "da": "da-DK",
        "el": "el-GR",
        "et": "et-EE",
        "fa": "fa-IR",
        "fi": "fi-FI",
        "fil": "tl-PH",
        "he": "he-IL",
        "hu": "hu-HU",
        "id": "id-ID",
        "ms": "ms-MY",
        "nl": "nl-NL",
        "pl": "pl-PL",
        "ro": "ro-RO",
        "sv": "sv-SE",
        "ta": "ta-IN",
        "te": "te-IN",
        "th": "th-TH",
        "tl": "tl-PH",
        "tr": "tr-TR",
        "uk": "uk-UA",
        "vi": "vi-VN",
        "yue": "zh-HK",
    }
)


class _ZoomAIErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    message: str


class ZoomAIException(BaseLLMException):
    pass


def zoom_ai_locale(language: str) -> str:
    """A bare code like `en` becomes its documented locale. Anything else is sent as given for Scribe to validate."""
    return ZOOM_AI_DEFAULT_LOCALES.get(language.strip().lower(), language.strip())


def zoom_ai_tracking_id() -> str:
    return f"{ZOOM_AI_TRACKING_ID_PREFIX}{uuid.uuid4()}"


def _zoom_ai_reason(error_body: str) -> str | None:
    try:
        return _ZoomAIErrorBody.model_validate_json(error_body).message
    except ValidationError:
        return None


def zoom_ai_error_message(error_body: str, status_code: int) -> str:
    """Zoom explains a refusal in the `message` of its JSON body. Any other body (an HTML error page, a proxy's
    reply) is left out, so upstream content never reaches the caller."""
    reason: Final = _zoom_ai_reason(error_body)
    if reason is None:
        return f"Zoom AI Services error (HTTP {status_code})"
    return f"Zoom AI Services error (HTTP {status_code}): {reason}"


def zoom_ai_exception(
    error_body: str, status_code: int, headers: Mapping[str, str] | Headers | None
) -> ZoomAIException:
    return ZoomAIException(
        message=zoom_ai_error_message(error_body, status_code),
        status_code=status_code,
        headers=None if headers is None else Headers(headers),
    )
