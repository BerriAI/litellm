from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.types.utils import ImageUsage, ImageUsageInputTokensDetails

_TOKEN_COUNT_ADAPTER: Final = TypeAdapter(int)
_TOKEN_DETAILS_LIST_ADAPTER: Final[TypeAdapter[list[object]]] = TypeAdapter(list[object])
_TOKEN_DETAILS_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def _get_token_count(details: Mapping[str, object]) -> int:
    raw_token_count: Final = details.get("tokenCount", details.get("token_count", 0))
    return raw_token_count if isinstance(raw_token_count, int) else 0


def _get_usage_token_count(usage_metadata: Mapping[str, object], key: str) -> int:
    return _TOKEN_COUNT_ADAPTER.validate_python(usage_metadata.get(key, 0))


def _get_modality_token_details(
    usage_metadata: Mapping[str, object], *details_keys: str
) -> tuple[Mapping[str, object], ...]:
    details: Final[list[object]] = next(
        (
            _TOKEN_DETAILS_LIST_ADAPTER.validate_python(usage_metadata.get(details_key))
            for details_key in details_keys
            if isinstance(usage_metadata.get(details_key), list)
        ),
        [],
    )
    return tuple(
        _TOKEN_DETAILS_MAPPING_ADAPTER.validate_python(detail) for detail in details if isinstance(detail, Mapping)
    )


def _sum_modality_token_details(
    token_details: tuple[Mapping[str, object], ...],
) -> ImageUsageInputTokensDetails:
    return ImageUsageInputTokensDetails(
        image_tokens=sum(
            _get_token_count(details)
            for details in token_details
            if str(details.get("modality", "")).upper() == "IMAGE"
        ),
        text_tokens=sum(
            _get_token_count(details) for details in token_details if str(details.get("modality", "")).upper() == "TEXT"
        ),
    )


def _output_modality_token_details(
    candidate_tokens: int,
    reasoning_tokens: int,
    candidate_tokens_are_inclusive: bool,
    candidate_token_details: tuple[Mapping[str, object], ...],
    modality_tokens: ImageUsageInputTokensDetails,
) -> ImageUsageInputTokensDetails:
    non_reasoning_candidate_tokens: Final = max(
        candidate_tokens - (reasoning_tokens if candidate_tokens_are_inclusive else 0),
        0,
    )
    if not candidate_token_details:
        return ImageUsageInputTokensDetails(image_tokens=non_reasoning_candidate_tokens, text_tokens=0)

    known_modality_tokens: Final = modality_tokens.text_tokens + modality_tokens.image_tokens
    text_tokens: Final = (
        max(
            modality_tokens.text_tokens - (known_modality_tokens - non_reasoning_candidate_tokens),
            0,
        )
        if known_modality_tokens > non_reasoning_candidate_tokens
        else modality_tokens.text_tokens + non_reasoning_candidate_tokens - known_modality_tokens
    )
    return ImageUsageInputTokensDetails(image_tokens=modality_tokens.image_tokens, text_tokens=text_tokens)


def transform_gemini_image_usage(usage_metadata: Mapping[str, object]) -> ImageUsage:
    """
    Transform Gemini usageMetadata to ImageUsage format.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    validated_counts: Final = ImageUsage.model_validate(
        {
            "input_tokens": usage_metadata.get("promptTokenCount", 0),
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
            "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }
    )
    prompt_tokens: Final = validated_counts.input_tokens
    candidate_tokens: Final = validated_counts.output_tokens
    reasoning_tokens: Final = _get_usage_token_count(usage_metadata, "thoughtsTokenCount")
    candidate_tokens_are_inclusive: Final = VertexGeminiConfig.is_candidate_token_count_inclusive(usage_metadata)
    completion_tokens: Final = candidate_tokens + (0 if candidate_tokens_are_inclusive else reasoning_tokens)
    input_tokens_details: Final = _sum_modality_token_details(
        _get_modality_token_details(usage_metadata, "promptTokensDetails", "prompt_tokens_details")
    )
    candidate_token_details: Final = _get_modality_token_details(
        usage_metadata, "candidatesTokensDetails", "candidates_tokens_details"
    )
    modality_tokens: Final = _sum_modality_token_details(candidate_token_details)
    output_modality_tokens: Final = _output_modality_token_details(
        candidate_tokens,
        reasoning_tokens,
        candidate_tokens_are_inclusive,
        candidate_token_details,
        modality_tokens,
    )

    output_tokens_details: Final[dict[str, int]] = {
        "image_tokens": output_modality_tokens.image_tokens,
        "text_tokens": output_modality_tokens.text_tokens,
        **({"reasoning_tokens": reasoning_tokens} if "thoughtsTokenCount" in usage_metadata else {}),
    }

    usage_payload: Final[dict[str, object]] = {
        "input_tokens": prompt_tokens,
        "input_tokens_details": input_tokens_details,
        "output_tokens": completion_tokens,
        "total_tokens": validated_counts.total_tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_details": input_tokens_details.model_dump(),
        "completion_tokens": completion_tokens,
        "completion_tokens_details": {**output_tokens_details},
        "output_tokens_details": {**output_tokens_details},
    }
    return ImageUsage.model_validate(usage_payload)
