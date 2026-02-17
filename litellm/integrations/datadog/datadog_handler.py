"""Shared helpers for Datadog integrations."""

from __future__ import annotations

import os
from typing import List, Optional

from litellm.types.utils import StandardLoggingPayload


def get_datadog_source() -> str:
    return os.getenv("DD_SOURCE", "litellm")


def get_datadog_service() -> str:
    return os.getenv("DD_SERVICE", "litellm-server")


def get_datadog_hostname() -> str:
    return os.getenv("HOSTNAME", "")


def get_datadog_base_url_from_env() -> Optional[str]:
    """
    Get base URL override from common DD_BASE_URL env var.
    This is useful for testing or custom endpoints.
    """
    return os.getenv("DD_BASE_URL")


def get_datadog_env() -> str:
    return os.getenv("DD_ENV", "unknown")


def get_datadog_pod_name() -> str:
    return os.getenv("POD_NAME", "unknown")


def get_datadog_tags(
    standard_logging_object: Optional[StandardLoggingPayload] = None,
) -> str:
    """Build Datadog tags string used by multiple integrations."""

    base_tags = {
        "env": get_datadog_env(),
        "service": get_datadog_service(),
        "version": os.getenv("DD_VERSION", "unknown"),
        "HOSTNAME": get_datadog_hostname(),
        "POD_NAME": get_datadog_pod_name(),
    }

    tags: List[str] = [f"{k}:{v}" for k, v in base_tags.items()]

    if standard_logging_object:
        request_tags = standard_logging_object.get("request_tags", []) or []
        tags.extend(f"request_tag:{tag}" for tag in request_tags)

    return ",".join(tags)
