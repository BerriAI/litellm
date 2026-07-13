"""Mistral OCR handler for Unified Guardrails."""

from litellm.llms.mistral.ocr.guardrail_translation.handler import OCRHandler
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.ocr: OCRHandler,
    CallTypes.aocr: OCRHandler,
}

__all__ = ["guardrail_translation_mappings", "OCRHandler"]
