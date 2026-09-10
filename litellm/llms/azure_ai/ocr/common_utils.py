"""
Common utilities for Azure AI OCR providers.

This module provides routing logic to determine which OCR configuration to use
based on the model name.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.ocr.transformation import RustOCRConfig

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
    lowered: Final = model.lower()
    return "cohere" in lowered and "parse" in lowered


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
    from litellm.llms.azure_ai.ocr.cohere_parse_transformation import AzureAICohereParseConfig
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


def azure_rust_ocr_config(kwargs: Mapping[str, object]) -> RustOCRConfig | None:
    if (
        callable(kwargs.get("azure_ad_token_provider"))
        or kwargs.get("azure_username") is not None
        or kwargs.get("azure_password") is not None
    ):
        return None
    return RustOCRConfig(
        config_fields=frozenset(
            {
                "azure_ad_token",
                "tenant_id",
                "client_id",
                "client_secret",
                "azure_scope",
                "azure_authority_host",
                "azure_credential",
                "azure_federated_token_file",
            }
        ),
        extra_params=(("enable_azure_ad_token_refresh", True),)
        if litellm.enable_azure_ad_token_refresh is True
        else (),
    )
