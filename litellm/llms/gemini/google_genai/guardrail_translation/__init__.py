"""Google GenAI generateContent guardrail translation handler."""

from typing import Final

from litellm.llms.gemini.google_genai.guardrail_translation.handler import (
    GoogleGenAIGenerateContentHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings: Final = {  # mutable-ok: discover_guardrail_translation_mappings only accepts isinstance(mappings, dict)
    CallTypes.generate_content: GoogleGenAIGenerateContentHandler,
    CallTypes.agenerate_content: GoogleGenAIGenerateContentHandler,
    CallTypes.generate_content_stream: GoogleGenAIGenerateContentHandler,
    CallTypes.agenerate_content_stream: GoogleGenAIGenerateContentHandler,
}

__all__ = (
    "GoogleGenAIGenerateContentHandler",
    "guardrail_translation_mappings",
)
