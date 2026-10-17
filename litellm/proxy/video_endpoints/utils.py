from typing import Any, Final

import orjson
from pydantic import BaseModel, ConfigDict

from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)


def extract_model_from_target_model_names(target_model_names: Any) -> str | None:
    if isinstance(target_model_names, str):
        model_names: Final = [m.strip() for m in target_model_names.split(",") if m.strip()]
        return model_names[0] if model_names else None
    if not isinstance(target_model_names, list):
        return None
    return target_model_names[0] if target_model_names else None


def video_reference_to_id(video_ref: object) -> str:
    if isinstance(video_ref, dict):
        return video_ref.get("id", "")
    if not isinstance(video_ref, str):
        return ""
    try:
        parsed_ref: Final = orjson.loads(video_ref)
    except orjson.JSONDecodeError:
        return video_ref
    return parsed_ref.get("id", "") if isinstance(parsed_ref, dict) else video_ref


def get_custom_provider_from_data(data: dict[str, Any]) -> str | None:
    custom_llm_provider: Final = data.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider

    extra_body = data.get("extra_body")
    if isinstance(extra_body, str):
        try:
            parsed_extra_body: Final = orjson.loads(extra_body)
            if isinstance(parsed_extra_body, dict):
                extra_body = parsed_extra_body
        except Exception:
            extra_body = None

    if isinstance(extra_body, dict):
        extra_body_custom_llm_provider: Final = extra_body.get("custom_llm_provider")
        if isinstance(extra_body_custom_llm_provider, str):
            return extra_body_custom_llm_provider

    return None


def encode_character_id_in_response(response: Any, custom_llm_provider: str, model_id: str | None) -> object:
    if isinstance(response, dict) and response.get("id"):
        response["id"] = encode_character_id_with_provider(
            character_id=response["id"],
            provider=custom_llm_provider,
            model_id=model_id,
        )
        return response

    character_id: Final = getattr(response, "id", None)
    if isinstance(character_id, str) and character_id:
        response.id = encode_character_id_with_provider(
            character_id=character_id,
            provider=custom_llm_provider,
            model_id=model_id,
        )
    return response


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


class VideoRequestRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: object = None
    custom_llm_provider: object = None


def encode_video_id_in_response(
    response: object,
    fallback_provider: object,
    fallback_model_id: object,
) -> object:
    if not isinstance(response, VideoObject) or not response.id:
        return response

    decoded: Final = decode_video_id_with_provider(response.id)
    if decoded.get("model_id"):
        return response

    hidden_params: Final = response.hidden_params
    model_id: Final = _non_empty_str(fallback_model_id) or _non_empty_str(hidden_params.get("model_id"))
    provider: Final = (
        _non_empty_str(decoded.get("custom_llm_provider"))
        or _non_empty_str(hidden_params.get("custom_llm_provider"))
        or _non_empty_str(fallback_provider)
    )
    if not model_id or not provider:
        return response

    restamped_id: Final = encode_video_id_with_provider(
        video_id=decoded.get("video_id") or response.id,
        provider=provider,
        model_id=model_id,
    )
    return response.model_copy(update={"id": restamped_id})
