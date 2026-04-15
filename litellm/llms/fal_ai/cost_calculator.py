from typing import Any, Optional

import litellm
from litellm.types.utils import ImageResponse


def _parse_megapixels(size: Optional[str]) -> Optional[float]:
    """Parse size string ('1024x1024' or '1024-x-1024') into megapixels."""
    if not size:
        return None
    normalized = size.replace("-x-", "x").replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        return None
    try:
        w, h = int(parts[0]), int(parts[1])
        return (w * h) / 1_000_000
    except (ValueError, TypeError):
        return None


def _resolve_resolution_tier(size: Optional[str]) -> Optional[str]:
    """Map a size param to a resolution tier ('2K' or '4K').

    Handles three cases:
    1. Already a tier string: '2K', '4K' -> return as-is
    2. Pixel dimensions: '1024x1024' -> compute MP -> classify
    3. Unknown -> None (caller falls back to flat pricing)

    Threshold: < 2.5 MP = '2K', >= 2.5 MP = '4K'
    """
    if not size:
        return None
    if size.upper() in ("2K", "4K"):
        return size.upper()
    mp = _parse_megapixels(size)
    if mp is None:
        return None
    return "4K" if mp >= 2.5 else "2K"


def cost_calculator(
    model: str,
    image_response: Any,
    size: Optional[str] = None,
) -> float:
    """fal.ai image generation cost calculator.

    Dispatches on pricing_basis from model info:
    - PER_MEGAPIXEL: output_cost_per_megapixel * megapixels * num_images
    - PER_IMAGE_RESOLUTION: output_cost_per_image_by_resolution[tier] * num_images
    - PER_CALL / default: output_cost_per_image * num_images
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(
            f"image_response must be of type ImageResponse, got type={type(image_response)}"
        )

    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
    )

    num_images = len(image_response.data) if image_response.data else 0
    effective_size = size or getattr(image_response, "size", None)
    pricing_basis = _model_info.get("pricing_basis", "PER_CALL")

    if pricing_basis == "PER_MEGAPIXEL":
        cost_per_mp: float = _model_info.get("output_cost_per_megapixel") or 0.0
        megapixels = _parse_megapixels(effective_size)
        if cost_per_mp > 0 and megapixels and megapixels > 0:
            return cost_per_mp * megapixels * num_images
        # Fall through to flat if MP can't be computed

    elif pricing_basis == "PER_IMAGE_RESOLUTION":
        cost_by_res = _model_info.get("output_cost_per_image_by_resolution")
        if cost_by_res:
            tier = _resolve_resolution_tier(effective_size)
            if tier and tier in cost_by_res:
                return cost_by_res[tier] * num_images
        # Fall through to flat if tier can't be resolved

    # PER_CALL / fallback
    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    return output_cost_per_image * num_images
