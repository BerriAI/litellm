"""
Common utilities for Azure AI OCR providers.

This module provides routing logic to determine which OCR configuration to use
based on the model name.
"""

from typing import TYPE_CHECKING, Final, Optional

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig


def is_azure_document_intelligence_model(model: str) -> bool:
    """Whether an azure_ai OCR model routes to Azure Document Intelligence.

    Azure AI exposes two OCR services on the same provider; the sub-route in the
    model name (`azure_ai/doc-intelligence/<model>`) selects Document Intelligence
    over Mistral OCR. This is the single source of truth for that routing decision.
    """
    lowered: Final = model.lower()
    return "doc-intelligence" in lowered or "documentintelligence" in lowered


def is_azure_cohere_parse_model(model: str) -> bool:
    """Whether an Azure AI OCR model routes to Cohere Parse."""
    normalized: Final = model.lower().replace("_", "-").rsplit("/", 1)[-1]
    return normalized == "cohere-parse-v5"


def get_azure_ai_ocr_config(model: str) -> Optional["BaseOCRConfig"]:
    """
    Determine which Azure AI OCR configuration to use based on the model name.

    Azure AI supports multiple OCR services:
    - Azure Document Intelligence: azure_ai/doc-intelligence/<model>
    - Cohere Parse: azure_ai/Cohere-parse-v5
    - Mistral OCR (via Azure AI): other azure_ai OCR models

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
    from litellm.llms.azure_ai.ocr.cohere_parse_transformation import (
        AzureAICohereParseConfig,
    )
    from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
        AzureDocumentIntelligenceOCRConfig,
    )
    from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig

    # Check for Azure Document Intelligence models
    if is_azure_document_intelligence_model(model):
        verbose_logger.debug("Routing %s to Azure Document Intelligence OCR config", model)
        return AzureDocumentIntelligenceOCRConfig()

    if is_azure_cohere_parse_model(model):
        verbose_logger.debug("Routing %s to Azure AI Cohere Parse config", model)
        return AzureAICohereParseConfig()

    # Default to Mistral-based OCR for other azure_ai models
    verbose_logger.debug("Routing %s to Azure AI (Mistral) OCR config", model)
    return AzureAIOCRConfig()
