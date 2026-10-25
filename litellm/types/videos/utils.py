"""
Utility functions for video ID encoding/decoding with provider information.

Follows the pattern used in responses/utils.py for consistency.
Format: vid_{base64_encoded_string}
"""

import base64
from collections.abc import Sequence
from typing import Final

from litellm._logging import verbose_logger
from litellm.types.utils import SpecialEnums
from litellm.types.videos.main import DecodedVideoId

VIDEO_ID_PREFIX: Final = "video_"
VIDEO_ID_SERVED_MODEL_PREFIX: Final = "served_model:"
VIDEO_ID_DEPLOYMENT_PREFIX: Final = "deployment_id:"
VIDEO_ID_REQUESTED_MODEL_PREFIX: Final = "requested_model:"
VIDEO_ID_CALL_TYPES: Final = frozenset(
    {
        "avideo_status",
        "video_status",
        "avideo_content",
        "video_content",
        "avideo_remix",
        "video_remix",
        "avideo_edit",
        "video_edit",
        "avideo_extension",
        "video_extension",
    }
)
CHARACTER_ID_PREFIX: Final = "character_"
CHARACTER_ID_TEMPLATE: Final = "litellm:custom_llm_provider:{};model_id:{};character_id:{}"


class DecodedCharacterId(dict):
    """Structure representing a decoded character ID."""

    custom_llm_provider: str | None
    model_id: str | None
    character_id: str


def _add_base64_padding(value: str) -> str:
    """
    Add missing base64 padding when IDs are copied without trailing '=' chars.
    """
    missing_padding: Final = len(value) % 4
    if missing_padding:
        value += "=" * (4 - missing_padding)
    return value


def encode_video_id_with_provider(
    video_id: str,
    provider: str,
    model_id: str | None = None,
    served_model: str | None = None,
    deployment_id: str | None = None,
    requested_model: str | None = None,
) -> str:
    """Encode provider and model_id into video_id using base64."""
    if not provider or not video_id:
        return video_id

    # Try to decode the ID first to check if it's already encoded
    # This handles the case where Azure/OpenAI return IDs that start with "video_"
    # but are not yet encoded with provider information
    decoded: Final = decode_video_id_with_provider(video_id)
    if decoded.get("custom_llm_provider") is not None:
        # ID is already encoded, return as-is
        return video_id

    # ID is not encoded (even if it starts with video_), so encode it
    assembled_id = str(SpecialEnums.LITELLM_MANAGED_VIDEO_COMPLETE_STR.value).format(provider, model_id or "", video_id)
    served_part: Final = f";{VIDEO_ID_SERVED_MODEL_PREFIX}{served_model}" if served_model else ""
    deployment_part: Final = f";{VIDEO_ID_DEPLOYMENT_PREFIX}{deployment_id}" if deployment_id else ""
    requested_part: Final = f";{VIDEO_ID_REQUESTED_MODEL_PREFIX}{requested_model}" if requested_model else ""

    base64_encoded_id: Final[str] = base64.b64encode(
        (assembled_id + served_part + deployment_part + requested_part).encode("utf-8")
    ).decode("utf-8")

    return f"{VIDEO_ID_PREFIX}{base64_encoded_id}"


def id_safe_name(name: object) -> str | None:
    return name if isinstance(name, str) and name and ";" not in name else None


def _trailing_part(parts: Sequence[str], prefix: str) -> str | None:
    return next((part.removeprefix(prefix) or None for part in parts[3:] if part.startswith(prefix)), None)


def decode_video_id_with_provider(encoded_video_id: str) -> DecodedVideoId:
    """Decode provider and model_id from encoded video_id."""
    if not encoded_video_id:
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    if not encoded_video_id.startswith(VIDEO_ID_PREFIX):
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    try:
        cleaned_id = encoded_video_id.replace(VIDEO_ID_PREFIX, "")
        cleaned_id = _add_base64_padding(cleaned_id)
        decoded_id: Final = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")
    except Exception as e:
        verbose_logger.debug("Error decoding video_id '%s': %s", encoded_video_id, e)
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    if ";" not in decoded_id:
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    parts: Final = decoded_id.split(";")

    custom_llm_provider = None
    model_id = None
    decoded_video_id = encoded_video_id

    if len(parts) >= 3:
        custom_llm_provider_part: Final = parts[0]
        model_id_part: Final = parts[1]
        video_id_part: Final = parts[2]

        custom_llm_provider = custom_llm_provider_part.replace("litellm:custom_llm_provider:", "")
        model_id = model_id_part.replace("model_id:", "")
        decoded_video_id = video_id_part.replace("video_id:", "")

    served_model: Final = _trailing_part(parts, VIDEO_ID_SERVED_MODEL_PREFIX)
    deployment_id: Final = _trailing_part(parts, VIDEO_ID_DEPLOYMENT_PREFIX)
    requested_model: Final = _trailing_part(parts, VIDEO_ID_REQUESTED_MODEL_PREFIX)
    decoded_parts: Final = DecodedVideoId(
        custom_llm_provider=custom_llm_provider, model_id=model_id, video_id=decoded_video_id
    )
    with_served: Final[DecodedVideoId] = (
        decoded_parts if served_model is None else {**decoded_parts, "served_model": served_model}
    )
    with_deployment: Final[DecodedVideoId] = (
        with_served if deployment_id is None else {**with_served, "deployment_id": deployment_id}
    )
    return with_deployment if requested_model is None else {**with_deployment, "requested_model": requested_model}


def extract_original_video_id(encoded_video_id: str) -> str:
    """Extract original video ID without encoding."""
    decoded: Final = decode_video_id_with_provider(encoded_video_id)
    return decoded.get("video_id", encoded_video_id)


def encode_character_id_with_provider(character_id: str, provider: str, model_id: str | None = None) -> str:
    """Encode provider and model_id into character_id using base64."""
    if not provider or not character_id:
        return character_id

    decoded: Final = decode_character_id_with_provider(character_id)
    if decoded.get("custom_llm_provider") is not None:
        return character_id

    assembled_id: Final = CHARACTER_ID_TEMPLATE.format(provider, model_id or "", character_id)
    base64_encoded_id: Final[str] = base64.b64encode(assembled_id.encode("utf-8")).decode("utf-8")
    return f"{CHARACTER_ID_PREFIX}{base64_encoded_id}"


def decode_character_id_with_provider(encoded_character_id: str) -> DecodedCharacterId:
    """Decode provider and model_id from encoded character_id."""
    if not encoded_character_id:
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )

    if not encoded_character_id.startswith(CHARACTER_ID_PREFIX):
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )

    try:
        cleaned_id = encoded_character_id.replace(CHARACTER_ID_PREFIX, "")
        cleaned_id = _add_base64_padding(cleaned_id)
        decoded_id: Final = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

        if ";" not in decoded_id:
            return DecodedCharacterId(
                custom_llm_provider=None,
                model_id=None,
                character_id=encoded_character_id,
            )

        parts: Final = decoded_id.split(";")

        custom_llm_provider = None
        model_id = None
        decoded_character_id = encoded_character_id

        if len(parts) >= 3:
            custom_llm_provider_part: Final = parts[0]
            model_id_part: Final = parts[1]
            character_id_part: Final = parts[2]

            custom_llm_provider = custom_llm_provider_part.replace("litellm:custom_llm_provider:", "")
            model_id = model_id_part.replace("model_id:", "")
            decoded_character_id = character_id_part.replace("character_id:", "")

        return DecodedCharacterId(
            custom_llm_provider=custom_llm_provider,
            model_id=model_id,
            character_id=decoded_character_id,
        )
    except Exception as e:
        verbose_logger.debug("Error decoding character_id '%s': %s", encoded_character_id, e)
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )


def extract_original_character_id(encoded_character_id: str) -> str:
    """Extract original character ID without encoding."""
    decoded: Final = decode_character_id_with_provider(encoded_character_id)
    return decoded.get("character_id", encoded_character_id)
