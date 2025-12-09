"""
Common utilities for Azure AI OCR providers.

This module provides routing logic to determine which OCR configuration to use
based on the model name.
"""

from typing import TYPE_CHECKING, Optional

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig


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
    if "doc-intelligence" in model or "documentintelligence" in model:
        verbose_logger.debug(
            f"Routing {model} to Azure Document Intelligence OCR config"
        )
        return AzureDocumentIntelligenceOCRConfig()
    
    # Default to Mistral-based OCR for other azure_ai models
    verbose_logger.debug(f"Routing {model} to Azure AI (Mistral) OCR config")
    return AzureAIOCRConfig()

