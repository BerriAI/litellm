"""
Common utilities for Vertex AI OCR providers.

This module provides routing logic to determine which OCR configuration to use
based on the model name.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig


def get_vertex_ai_ocr_config(model: str) -> Optional["BaseOCRConfig"]:
    """
    Determine which Vertex AI OCR configuration to use based on the model name.
    
    Vertex AI supports multiple OCR services:
    - Vertex AI OCR: vertex_ai/<model>
    
    Args:
        model: The model name (e.g., "vertex_ai/ocr/<model>")
    
    Returns:
        OCR configuration instance for the specified model
        
    Examples:
        >>> get_vertex_ai_ocr_config("vertex_ai/deepseek-ai/deepseek-ocr-maas")
        <VertexAIDeepSeekOCRConfig object>
        
        >>> get_vertex_ai_ocr_config("vertex_ai/ocr/mistral-ocr-maas")
        <VertexAIOCRConfig object>
    """
    from litellm.llms.vertex_ai.ocr.deepseek_transformation import (
        VertexAIDeepSeekOCRConfig,
    )
    from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig
    if "deepseek" in model:
        return VertexAIDeepSeekOCRConfig()
    return VertexAIOCRConfig()

