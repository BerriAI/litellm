from typing import Any, Dict, Optional

import orjson

from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)


def extract_model_from_target_model_names(target_model_names: Any) -> Optional[str]:
    if isinstance(target_model_names, str):
        target_model_names = [m.strip() for m in target_model_names.split(",") if m.strip()]
    elif not isinstance(target_model_names, list):
        return None
    return target_model_names[0] if target_model_names else None


def get_custom_provider_from_data(data: Dict[str, Any]) -> Optional[str]:
    custom_llm_provider = data.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider

    extra_body = data.get("extra_body")
    if isinstance(extra_body, str):
        try:
            parsed_extra_body = orjson.loads(extra_body)
            if isinstance(parsed_extra_body, dict):
                extra_body = parsed_extra_body
        except Exception:
            extra_body = None

    if isinstance(extra_body, dict):
        extra_body_custom_llm_provider = extra_body.get("custom_llm_provider")
        if isinstance(extra_body_custom_llm_provider, str):
            return extra_body_custom_llm_provider

    return None


def encode_character_id_in_response(response: Any, custom_llm_provider: str, model_id: Optional[str]) -> Any:
    if isinstance(response, dict) and response.get("id"):
        response["id"] = encode_character_id_with_provider(
            character_id=response["id"],
            provider=custom_llm_provider,
            model_id=model_id,
        )
        return response

    character_id = getattr(response, "id", None)
    if isinstance(character_id, str) and character_id:
        response.id = encode_character_id_with_provider(
            character_id=character_id,
            provider=custom_llm_provider,
            model_id=model_id,
        )
    return response


def coerce_optional_str(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def encode_video_id_in_response(
    response: object,
    fallback_provider: Optional[str],
    fallback_model_id: Optional[str],
) -> object:
    if not isinstance(response, VideoObject) or not response.id:
        return response

    hidden_params = get_hidden_params_dict(response)
    model_id = coerce_optional_str(hidden_params.get("model_id")) or fallback_model_id

    decoded = decode_video_id_with_provider(response.id)
    provider = (
        decoded.get("custom_llm_provider")
        or coerce_optional_str(hidden_params.get("custom_llm_provider"))
        or fallback_provider
    )
    if not provider:
        return response

    original_video_id = decoded.get("video_id") or response.id
    response.id = encode_video_id_with_provider(
        video_id=original_video_id,
        provider=provider,
        model_id=model_id,
    )
    return response
