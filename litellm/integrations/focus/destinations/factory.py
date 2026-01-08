"""Factory helpers for Focus export destinations."""

from __future__ import annotations

from typing import Optional

from .base import FocusDestination


class FocusDestinationFactory:
    """Builds destination instances based on provider/config settings."""

    @staticmethod
    def create(
        *,
        provider: str,
        prefix: str,
        config: Optional[dict] = None,
    ) -> FocusDestination:
        """Return a destination implementation for the requested provider."""
        raise NotImplementedError
