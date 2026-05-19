from litellm.types.utils import ImageUsage, ImageUsageInputTokensDetails


def transform_gemini_image_usage(usage_metadata: dict) -> ImageUsage:
    """
    Transform Gemini usageMetadata to ImageUsage format.
    """
    input_tokens_details = ImageUsageInputTokensDetails(
        image_tokens=0,
        text_tokens=0,
    )

    for details in usage_metadata.get("promptTokensDetails", []):
        if isinstance(details, dict):
            modality = str(details.get("modality", "")).upper()
            raw_token_count = details.get("tokenCount", details.get("token_count", 0))
            token_count = raw_token_count if isinstance(raw_token_count, int) else 0
            if modality == "TEXT":
                input_tokens_details.text_tokens += token_count
            elif modality == "IMAGE":
                input_tokens_details.image_tokens += token_count

    output_tokens = usage_metadata.get("candidatesTokenCount", 0)
    return ImageUsage(
        input_tokens=usage_metadata.get("promptTokenCount", 0),
        input_tokens_details=input_tokens_details,
        output_tokens=output_tokens,
        total_tokens=usage_metadata.get("totalTokenCount", 0),
        prompt_tokens=usage_metadata.get("promptTokenCount", 0),
        prompt_tokens_details=input_tokens_details.model_dump(),
        completion_tokens=output_tokens,
        completion_tokens_details={
            "text_tokens": 0,
            "image_tokens": output_tokens,
        },
        output_tokens_details={
            "text_tokens": 0,
            "image_tokens": output_tokens,
        },
    )
