"""MiniMax voice-cloning request and response transformations."""

from .transformation import (
    MinimaxVoiceCloneConfig,
    MinimaxVoiceCloneError,
    VoiceCloneResponse,
)

__all__ = [ # mutable-ok: public export list is intentionally mutable for package compatibility
    "MinimaxVoiceCloneConfig",
    "MinimaxVoiceCloneError",
    "VoiceCloneResponse",
]
