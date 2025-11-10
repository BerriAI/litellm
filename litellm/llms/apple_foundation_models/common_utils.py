"""
Common utilities for Apple Foundation Models provider.

Handles availability checks and lazy imports for v0.2.0+ SDK.
"""

from typing import Any, Literal

from litellm._logging import verbose_logger


def _get_apple_session_class(session_type: Literal["sync", "async"]) -> Any:
    """
    Internal helper to get Apple Foundation Models Session class with availability checking.

    Args:
        session_type: Either "sync" for Session or "async" for AsyncSession

    Returns:
        Apple Foundation Models Session or AsyncSession class

    Raises:
        ImportError: If apple-foundation-models package is not installed
        RuntimeError: If Apple Intelligence is not available on this system
    """
    # Lazy import - only import when actually needed
    try:
        from applefoundationmodels import (
            AsyncSession,
            Session,
            apple_intelligence_available,
        )
    except ImportError as e:
        raise ImportError(
            "Missing apple-foundation-models package. This is required for the "
            "Apple Foundation Models provider. Install it with: "
            "pip install apple-foundation-models"
        ) from e

    # Check if Apple Intelligence is available (only once)
    verbose_logger.debug("Checking Apple Intelligence availability")
    try:
        available = apple_intelligence_available()
    except Exception as exc:  # pragma: no cover - SDK specific
        raise RuntimeError(
            f"Failed to determine Apple Intelligence availability: {exc}"
        ) from exc

    if not available:
        raise RuntimeError(
            "Apple Intelligence is not available on this system. "
            "Requirements: macOS 26.0+ (Sequoia) with Apple Intelligence enabled."
        )

    return AsyncSession if session_type == "async" else Session


def get_apple_session_class() -> Any:
    """
    Get Apple Foundation Models Session class.

    Returns:
        Apple Foundation Models Session class

    Raises:
        ImportError: If apple-foundation-models package is not installed
        RuntimeError: If Apple Intelligence is not available on this system
    """
    return _get_apple_session_class("sync")


def get_apple_async_session_class() -> Any:
    """
    Get Apple Foundation Models AsyncSession class.

    Returns:
        Apple Foundation Models AsyncSession class

    Raises:
        ImportError: If apple-foundation-models package is not installed
        RuntimeError: If Apple Intelligence is not available on this system
    """
    return _get_apple_session_class("async")
