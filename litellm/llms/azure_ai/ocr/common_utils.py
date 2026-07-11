"""
Common utilities for Azure AI OCR providers.

This module provides routing logic to determine which OCR configuration to use
based on the model name.
"""

from typing import TYPE_CHECKING, Optional

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig


def is_azure_document_intelligence_model(model: str) -> bool:
    """Whether an azure_ai OCR model routes to Azure Document Intelligence.

    Azure AI exposes two OCR services on the same provider; the sub-route in the
    model name (`azure_ai/doc-intelligence/<model>`) selects Document Intelligence
    over Mistral OCR. This is the single source of truth for that routing decision.
    """
    lowered = model.lower()
    return "doc-intelligence" in lowered or "documentintelligence" in lowered


def get_azure_ai_ocr_config(model: str) -> Optional["BaseOCRConfig"]:
    """
    Determine which Azure AI OCR configuration to use based on the model name.

    Azure AI supports multiple OCR services:
    - Azure Document Intelligence: azure_ai/doc-intelligence/<model>
    - Mistral OCR (via Azure AI): azure_ai/<model>

    Args:
        model: The model name (e.g., "azure_ai/doc-intelligence/prebuilt-read",
               "azure_ai/pixtral-12b-2409")

    Returns:
        OCR configuration instance for the specified model

    Examples:
        >>> get_azure_ai_ocr_config("azure_ai/doc-intelligence/prebuilt-read")
        <AzureDocumentIntelligenceOCRConfig object>

        >>> get_azure_ai_ocr_config("azure_ai/pixtral-12b-2409")
        <AzureAIOCRConfig object>
    """
    from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
        AzureDocumentIntelligenceOCRConfig,
    )
    from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig

    # Check for Azure Document Intelligence models
    if is_azure_document_intelligence_model(model):
        verbose_logger.debug(f"Routing {model} to Azure Document Intelligence OCR config")
        return AzureDocumentIntelligenceOCRConfig()

    # Default to Mistral-based OCR for other azure_ai models
    verbose_logger.debug(f"Routing {model} to Azure AI (Mistral) OCR config")
    return AzureAIOCRConfig()
