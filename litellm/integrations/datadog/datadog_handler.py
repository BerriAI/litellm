"""Shared helpers for Datadog integrations."""

from __future__ import annotations

import os
import re
from typing import Final

from litellm.types.utils import StandardLoggingPayload


def get_datadog_source() -> str:
    return os.getenv("DD_SOURCE", "litellm")


def get_datadog_service() -> str:
    return os.getenv("DD_SERVICE", "litellm-server")


def get_datadog_hostname() -> str:
    return os.getenv("HOSTNAME", "")


def get_datadog_base_url_from_env() -> str | None:
    """
    Get base URL override from common DD_BASE_URL env var.
    This is useful for testing or custom endpoints.
    """
    return os.getenv("DD_BASE_URL")


def get_datadog_env() -> str:
    return os.getenv("DD_ENV", "unknown")


def get_datadog_pod_name() -> str:
    return os.getenv("POD_NAME", "unknown")


def normalize_datadog_tag_value(value: object) -> str:
    normalized_value: Final = "".join(
        character if character.isalnum() or character in "_-:./" else "_" for character in str(value).lower()
    )
    return re.sub(r"_+", "_", normalized_value).strip("_")


def get_datadog_tags(
    standard_logging_object: StandardLoggingPayload | None = None,
) -> list[str]:
    """Build Datadog tags as a list of individual tag strings.

    Returns a list of "key:value" strings suitable for Datadog LLM Observability
    (which expects tags as an array). For Datadog Logs API (ddtags), join with
    comma: ",".join(get_datadog_tags(...)).
    """

    base_tags: Final = {
        "env": get_datadog_env(),
        "service": get_datadog_service(),
        "version": os.getenv("DD_VERSION", "unknown"),
        "HOSTNAME": get_datadog_hostname(),
        "POD_NAME": get_datadog_pod_name(),
    }

    tags: Final[list[str]] = [f"{k}:{v}" for k, v in base_tags.items()]

    if standard_logging_object:
        request_tags: Final = standard_logging_object.get("request_tags", []) or []
        tags.extend(f"request_tag:{normalize_datadog_tag_value(tag)}" for tag in request_tags)

        # Add Team Tag
        metadata: Final = standard_logging_object.get("metadata", {}) or {}
        team_tag: Final = (
            metadata.get("user_api_key_team_alias")
            or metadata.get("team_alias")
            or metadata.get("user_api_key_team_id")
            or metadata.get("team_id")
        )
        if team_tag:
            tags.append(f"team:{normalize_datadog_tag_value(team_tag)}")

    return tags
