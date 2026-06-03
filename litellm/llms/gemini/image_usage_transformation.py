from typing import Any

from litellm.types.utils import ImageUsage, ImageUsageInputTokensDetails


def _get_token_count(details: dict) -> int:
    raw_token_count = details.get("tokenCount", details.get("token_count", 0))
    return raw_token_count if isinstance(raw_token_count, int) else 0


def _get_modality_token_details(usage_metadata: dict, *details_keys: str) -> list:
    for details_key in details_keys:
        details = usage_metadata.get(details_key)
        if isinstance(details, list):
            return details
    return []


def _sum_modality_token_details(
    usage_metadata: dict, *details_keys: str
) -> ImageUsageInputTokensDetails:
    tokens_details = ImageUsageInputTokensDetails(
        image_tokens=0,
        text_tokens=0,
    )

    for details in _get_modality_token_details(usage_metadata, *details_keys):
        if isinstance(details, dict):
            modality = str(details.get("modality", "")).upper()
            token_count = _get_token_count(details)
            if modality == "TEXT":
                tokens_details.text_tokens += token_count
            elif modality == "IMAGE":
                tokens_details.image_tokens += token_count

    return tokens_details


def transform_gemini_image_usage(usage_metadata: dict) -> ImageUsage:
    """
    Transform Gemini usageMetadata to ImageUsage format.
    """
    input_tokens_details = _sum_modality_token_details(
        usage_metadata, "promptTokensDetails", "prompt_tokens_details"
    )
    output_tokens = usage_metadata.get("candidatesTokenCount", 0)
    output_tokens_details = _sum_modality_token_details(
        usage_metadata, "candidatesTokensDetails", "candidates_tokens_details"
    )

    if not _get_modality_token_details(
        usage_metadata, "candidatesTokensDetails", "candidates_tokens_details"
    ):
        output_tokens_details.image_tokens = output_tokens
    else:
        known_output_tokens = (
            output_tokens_details.text_tokens + output_tokens_details.image_tokens
        )
        if output_tokens > known_output_tokens:
            output_tokens_details.text_tokens += output_tokens - known_output_tokens

    usage_payload: dict[str, Any] = {
        "input_tokens": usage_metadata.get("promptTokenCount", 0),
        "input_tokens_details": input_tokens_details,
        "output_tokens": output_tokens,
        "total_tokens": usage_metadata.get("totalTokenCount", 0),
        "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
        "prompt_tokens_details": input_tokens_details.model_dump(),
        "completion_tokens": output_tokens,
        "completion_tokens_details": output_tokens_details.model_dump(),
        "output_tokens_details": output_tokens_details.model_dump(),
    }
    return ImageUsage(**usage_payload)
