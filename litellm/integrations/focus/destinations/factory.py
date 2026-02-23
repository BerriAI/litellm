"""Factory helpers for Focus export destinations."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .base import FocusDestination
from .s3_destination import FocusS3Destination


class FocusDestinationFactory:
    """Builds destination instances based on provider/config settings."""

    @staticmethod
    def create(
        *,
        provider: str,
        prefix: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> FocusDestination:
        """Return a destination implementation for the requested provider."""
        provider_lower = provider.lower()
        normalized_config = FocusDestinationFactory._resolve_config(
            provider=provider_lower, overrides=config or {}
        )
        if provider_lower == "s3":
            return FocusS3Destination(prefix=prefix, config=normalized_config)
        raise NotImplementedError(
            f"Provider '{provider}' not supported for Focus export"
        )

    @staticmethod
    def _resolve_config(
        *,
        provider: str,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        if provider == "s3":
            resolved = {
                "bucket_name": overrides.get("bucket_name")
                or os.getenv("FOCUS_S3_BUCKET_NAME"),
                "region_name": overrides.get("region_name")
                or os.getenv("FOCUS_S3_REGION_NAME"),
                "endpoint_url": overrides.get("endpoint_url")
                or os.getenv("FOCUS_S3_ENDPOINT_URL"),
                "aws_access_key_id": overrides.get("aws_access_key_id")
                or os.getenv("FOCUS_S3_ACCESS_KEY"),
                "aws_secret_access_key": overrides.get("aws_secret_access_key")
                or os.getenv("FOCUS_S3_SECRET_KEY"),
                "aws_session_token": overrides.get("aws_session_token")
                or os.getenv("FOCUS_S3_SESSION_TOKEN"),
            }
            if not resolved.get("bucket_name"):
                raise ValueError("FOCUS_S3_BUCKET_NAME must be provided for S3 exports")
            return {k: v for k, v in resolved.items() if v is not None}
        raise NotImplementedError(
            f"Provider '{provider}' not supported for Focus export configuration"
        )
