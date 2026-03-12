"""
Tone detection: CPU-only regex/keyword checks for inappropriate chatbot tone.

Detects dismissive, condescending, blaming, unprofessional language while
allowing domain-specific safe phrases.
"""

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.tone_detection.base import (
    ToneChecker,
)

__all__ = [
    "ToneChecker",
]
