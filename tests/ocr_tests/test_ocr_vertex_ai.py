"""Vertex AI OCR config routing and DeepSeek request shaping (no network)."""

from typing import Final

import pytest


def test_vertex_ai_ocr_routing():
    """
    Test that Vertex AI OCR routing correctly selects the right config based on model name.
    """
    from litellm.llms.vertex_ai.ocr.common_utils import get_vertex_ai_ocr_config
    from litellm.llms.vertex_ai.ocr.deepseek_transformation import (
        VertexAIDeepSeekOCRConfig,
    )
    from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig

    # Test DeepSeek OCR routing
    deepseek_config = get_vertex_ai_ocr_config("vertex_ai/deepseek-ocr-maas")
    assert isinstance(deepseek_config, VertexAIDeepSeekOCRConfig), (
        "DeepSeek model should route to VertexAIDeepSeekOCRConfig"
    )

    # Test Mistral OCR routing (should use default VertexAIOCRConfig)
    mistral_config = get_vertex_ai_ocr_config("vertex_ai/mistral-ocr-2505")
    assert isinstance(mistral_config, VertexAIOCRConfig), "Mistral model should route to VertexAIOCRConfig"

    # Test other DeepSeek variants
    deepseek_variant = get_vertex_ai_ocr_config("vertex_ai/deepseek-ocr-maas")
    assert isinstance(deepseek_variant, VertexAIDeepSeekOCRConfig), (
        "DeepSeek variant should route to VertexAIDeepSeekOCRConfig"
    )


@pytest.mark.parametrize("model", ("deepseek-ocr-maas", "deepseek-ai/deepseek-ocr-maas"))
def test_deepseek_request_uses_single_provider_namespace(model: str) -> None:
    from litellm.llms.vertex_ai.ocr.deepseek_transformation import (
        VertexAIDeepSeekOCRConfig,
    )

    request: Final = VertexAIDeepSeekOCRConfig().transform_ocr_request(
        model=model,
        document={"type": "image_url", "image_url": "data:image/png;base64,AA=="},
        optional_params={},
        headers={},
    )

    assert request.data["model"] == "deepseek-ai/deepseek-ocr-maas"
