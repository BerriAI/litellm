"""
Utility functions for video ID encoding/decoding with provider information.

Follows the pattern used in responses/utils.py for consistency.
Format: vid_{base64_encoded_string}
"""
import base64
from typing import Tuple, Optional
from litellm.types.utils import SpecialEnums
from litellm.types.videos.main import DecodedVideoId    
from litellm._logging import verbose_logger



VIDEO_ID_PREFIX = "video_"


def encode_video_id_with_provider(
    video_id: str, 
    provider: str,
    model_id: Optional[str] = None
) -> str:
    """Encode provider and model_id into video_id using base64."""
    if not provider or not video_id:
        return video_id
    
    if video_id.startswith(VIDEO_ID_PREFIX):
        return video_id
    
    assembled_id = str(
        SpecialEnums.LITELLM_MANAGED_VIDEO_COMPLETE_STR.value
    ).format(provider, model_id or "", video_id)
    
    base64_encoded_id: str = base64.b64encode(assembled_id.encode("utf-8")).decode("utf-8")
    
    return f"{VIDEO_ID_PREFIX}{base64_encoded_id}"


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
        decoded_id = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

        if ";" not in decoded_id:
            return DecodedVideoId(
                custom_llm_provider=None,
                model_id=None,
                video_id=encoded_video_id,
            )

        parts = decoded_id.split(";")

        custom_llm_provider = None
        model_id = None
        decoded_video_id = encoded_video_id

        if len(parts) >= 3:
            custom_llm_provider_part = parts[0]
            model_id_part = parts[1]
            video_id_part = parts[2]

            custom_llm_provider = custom_llm_provider_part.replace(
                "litellm:custom_llm_provider:", ""
            )
            model_id = model_id_part.replace("model_id:", "")
            decoded_video_id = video_id_part.replace("video_id:", "")

        return DecodedVideoId(
            custom_llm_provider=custom_llm_provider,
            model_id=model_id,
            video_id=decoded_video_id,
        )
    except Exception as e:
        verbose_logger.debug(f"Error decoding video_id '{encoded_video_id}': {e}")
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )


def extract_original_video_id(encoded_video_id: str) -> str:
    """Extract original video ID without encoding."""
    decoded = decode_video_id_with_provider(encoded_video_id)
    return decoded.get("video_id", encoded_video_id)
